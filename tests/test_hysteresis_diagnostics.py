import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from library.hysteresis_diagnostics import HysteresisHistory, plot_history, plot_complex_apertures


class HysteresisDiagnosticTests(unittest.TestCase):
    def test_summaries_are_bounded_do_not_retain_fields_and_count_tails(self):
        history = HysteresisHistory(np.ones((2,8,8)), np.zeros((8,8)), ['a','b'],
                                    np.ones((8,8),bool), max_steps=2, bins=8, max_samples=5)
        fields = np.ones((2,8,8),complex)
        maps = np.ones((2,8,8)); maps[0,0,0]=-3; maps[1,0,0]=3
        field_ref, map_ref = weakref.ref(fields), weakref.ref(maps)
        components = dict(magnetization=maps,state_names=['a','b'],fit_residual_rms=.1,
                          aperture_illumination_factors=np.ones((2,3),complex))
        for step in range(4):
            history(dict(stage='physical_projection', fields=fields,components=components,outer_round=step))
        exported = history.export()
        np.testing.assert_array_equal(exported['step'], [3,4])
        np.testing.assert_array_equal(exported['detector_error'], 0.)
        np.testing.assert_array_equal(exported['histogram'].sum(axis=-1)+exported['overflow'].sum(axis=-1),64)
        np.testing.assert_array_equal(exported['overflow'][-1], [[1,0],[0,1]])
        self.assertLessEqual(len(history.samples[0][0]),5)
        del fields, maps, components
        gc.collect()
        self.assertIsNone(field_ref()); self.assertIsNone(map_ref())
        with tempfile.TemporaryDirectory() as folder, patch('matplotlib.pyplot.show'):
            plot_history(history,Path(folder))
            self.assertEqual(len(list(Path(folder).glob('*.png'))),3)

    def test_single_step_history_plot(self):
        history=HysteresisHistory(np.ones((1,4,4)),np.zeros((4,4)),['a'],np.ones((4,4),bool))
        history(dict(stage='final_physical_projection',fields=np.ones((1,4,4)),
                     components=dict(magnetization=np.ones((1,4,4)),state_names=['a'])))
        with tempfile.TemporaryDirectory() as folder, patch('matplotlib.pyplot.show'):
            plot_history(history,Path(folder))

    def test_driver_records_final_projection_and_history_exports_to_hdf5(self):
        import ast
        import contextlib
        import io
        import json
        import h5py
        from library import phase_retrieval_universal as u
        material=np.zeros((8,8));material[2:6,2:6]=1
        support=material.copy();support[0,0]=1
        signs=np.array([1.,.3,-1.])
        logs=.3+signs[:,None,None]*np.fft.fftshift(material)*(.1+.03j)
        fields=u.object_log_to_fourier_field(logs)
        history=HysteresisHistory(abs(fields)**2,np.zeros((3,8,8)),['p','f','n'],
                                  np.fft.fftshift(material)>0)
        recipe=u.default_universal_phase_retrieval_recipe()
        recipe.update(crop=0,binning=1,modes=[1],material_thickness=material,
                      warmup_Nit=0,outer_iterations=0,final_fourier_constraint=False,
                      saturated_states={'p':1.,'n':-1.})
        with contextlib.redirect_stdout(io.StringIO()):
            u.universal_phase_retrieval_algorithm(abs(fields)**2,np.zeros((3,8,8)),support,
                ['p','f','n'],[781.]*3,[1.]*3,['beam']*3,universal_recipe=recipe,
                start_fields=fields,progress_callback=history)
        h=history.export()
        self.assertEqual(h['stage'].tolist(),['final_physical_projection'])
        np.testing.assert_array_equal(h['histogram'].sum(axis=-1),16)
        np.testing.assert_allclose(h['mean'][0],signs,atol=1e-5)
        # Exercise the notebook's actual tree writer with compact history arrays.
        notebook=json.loads((Path(__file__).resolve().parents[1]/
                            '2601_XPCS/01_physical_hysteresis_263.ipynb').read_text())
        source=next(''.join(c['source']) for c in notebook['cells']
                    if 'def save_tree' in ''.join(c.get('source',[])))
        nodes=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)
               and node.name in ('save_tree','json_value')]
        ns=dict(np=np,json=json)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'export','exec'),ns)
        with tempfile.TemporaryDirectory() as folder:
            with h5py.File(Path(folder)/'history.h5','w') as out:
                ns['save_tree'](out.create_group('history'),h)
            with h5py.File(Path(folder)/'history.h5') as out:
                self.assertEqual(out['history/histogram'].shape,(1,3,80))
                self.assertNotIn('fields',out['history'])

    def test_hsv_windows_are_bounded_and_include_all_apertures(self):
        support=np.zeros((100,100)); support[20:80,20:80]=1
        support[2:5,2:5]=1; support[90:94,90:94]=1
        material=support.copy();material[:10]=0;material[90:]=0
        objects=np.ones((2,100,100),complex); original=objects.copy()
        from matplotlib.axes import Axes
        imshow=Axes.imshow; shapes=[]
        def record(ax, image, *args, **kwargs):
            shapes.append(np.shape(image)); return imshow(ax,image,*args,**kwargs)
        with tempfile.TemporaryDirectory() as folder, patch('matplotlib.pyplot.show'), patch.object(Axes,'imshow',record):
            plot_complex_apertures(objects,support,material,['a','b'],Path(folder),states=(0,5,-1),max_size=16)
            self.assertEqual(len(list(Path(folder).glob('*.png'))),2)
        self.assertEqual(len(shapes),6)
        self.assertTrue(all(shape[-1]==3 and max(shape[:2])<=16 for shape in shapes))
        np.testing.assert_array_equal(objects,original)

    def test_equivalent_modes_and_modal_histories_are_all_visible(self):
        from library import phase_retrieval_universal as u
        from library.hysteresis_diagnostics import plot_equivalent_mode_diagnostics
        support=np.zeros((8,8));support[2:6,2:6]=1;support[0,0]=1
        material=support.copy();material[0,0]=0
        recipe=u.default_universal_phase_retrieval_recipe()
        recipe.update(modes=[1,1],energy_values=[781.],magnetization_mode_groups=[[0],[1]],
                      illumination_mode_groups=[[0],[1]],fit_aperture_illumination=True,
                      align_global_phase=True,physical_projection_object_roi=True)
        logs=np.empty((3,2,8,8),complex)
        for mode,free in enumerate([.3,-.2]):
            logs[:,mode]=.3+mode*.2+np.array([1.,free,-1.])[:,None,None]*np.fft.fftshift(material)*(.1+.03j)
        fields=np.stack([u.object_log_to_fourier_field(logs[:,mode]) for mode in range(2)],axis=1)
        output,c=u._project_physical_modes(fields,['p','f','n'],[781.]*3,[1.]*3,['b']*3,
            recipe=recipe,material_thickness=np.fft.fftshift(material),
            projection_supportmask=np.fft.fftshift(support),saturated_states={'p':1.,'n':-1.},
            physical_iterations=40,return_components=True)
        history=HysteresisHistory(np.sum(abs(fields)**2,axis=1),np.zeros((3,8,8)),
                                  ['p','f','n'],np.fft.fftshift(material)>0,max_steps=2)
        history(dict(stage='final_physical_projection',fields=output,components=c))
        h=history.export()
        self.assertEqual(h['modal_histogram'].shape,(1,2,3,80))
        self.assertEqual(h['modal_factors'].shape[1],2)
        np.testing.assert_allclose(h['modal_mean'][0,:,1],[.3,-.2],atol=1e-5)
        with tempfile.TemporaryDirectory() as folder,patch('matplotlib.pyplot.show'):
            folder=Path(folder)
            plot_equivalent_mode_diagnostics(output,support,material,['p','f','n'],recipe,c,
                                             folder,states=(0,-1),max_size=8)
            for mode in (1,2):
                self.assertTrue((folder/f'final_mode_{mode}_amplitude.png').is_file())
                self.assertTrue((folder/f'final_mode_{mode}_phase.png').is_file())
                self.assertTrue((folder/f'final_mode_{mode}_magnetization.png').is_file())
                self.assertTrue((folder/f'final_mode_{mode}_complex_apertures_state_0.png').is_file())
            plot_history(h,folder)
            for mode in (1,2):
                self.assertTrue((folder/f'step_aperture_factors_mode_{mode}.png').is_file())
                self.assertTrue((folder/f'step_oh_distribution_group_{mode}.png').is_file())


if __name__=='__main__':unittest.main()
