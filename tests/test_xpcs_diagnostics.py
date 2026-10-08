"""Run the notebook's actual plotting and export cells after cropped retrieval."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from library import phase_retrieval_universal as u
from library import phase_retrieval_geometry as geometry
from library import fthcore as fth

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('xpcs_inputs',ROOT/'2601_XPCS/reconstruction_inputs.py')
inputs=importlib.util.module_from_spec(spec);spec.loader.exec_module(inputs)

class PlottingTests(unittest.TestCase):
    def test_setup_loading_and_retrieval_cells(self):
        notebook=json.loads((ROOT/'2601_XPCS/01_physical_hysteresis_263.ipynb').read_text())
        with tempfile.TemporaryDirectory() as folder:
            cache=Path(folder)/'cache.h5'
            support=np.zeros((24,24)); support[9:15,9:15]=1; support[3,3]=1
            material=support.copy(); material[3,3]=0
            labels=['scan263_point0000','scan263_point0020','scan263_point0040']
            with h5py.File(cache,'w') as h:
                h.attrs['config_json']=json.dumps({'binning':1,'dark':{'applied':False}})
                h['state_labels']=np.asarray(labels,dtype=h5py.string_dtype())
                for key,value in dict(scan=[263]*3,point=[0,20,40],
                    holograms=np.ones((3,24,24)),mask_pixel=np.zeros((3,24,24),bool),
                    supportmask=support,material_mask=material,current_A=[1.,0.,-1.],
                    energy_ev=[710.]*3).items(): h[key]=value
            ns={}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(''.join(notebook['cells'][1]['source']),ns)
                ns.update(CACHE=cache,OUTPUT=Path(folder)/'result.h5',FIGURES=Path(folder),
                          WARMUP_ITERATIONS=[2,1],OUTER_ITERATIONS=1,
                          INNER_ITERATIONS=1,PHYSICAL_ITERATIONS=2)
                for i in [2,3]: exec(''.join(notebook['cells'][i]['source']),ns)
            self.assertEqual(ns['state_labels'],labels)
            self.assertEqual(ns['fields'].shape,(3,24,24))
            self.assertTrue(np.isfinite(ns['fields']).all())
            plt.close('all')

    def test_all_diagnostics_and_export(self):
        notebook=json.loads((ROOT/'2601_XPCS/01_physical_hysteresis_263.ipynb').read_text())
        for modes in ([1], [1,2]):
            with self.subTest(modes=modes), tempfile.TemporaryDirectory() as folder:
                support=np.zeros((24,24));support[9:15,9:15]=1
                # No reference holes: optional endpoint diagnostics must not stop plots.
                material=support.copy()
                data=np.stack([np.ones((24,24))*v for v in (1.,1.2,.8)])
                masks=np.zeros_like(data,dtype=bool)
                states=['positive','interior','negative']
                ns=dict(universal=u,np=np,MODES=modes,material=material,
                    saturated={'positive':1.,'negative':-1.},FOCUS_PROP_UM=0.,
                    FOCUS_PHASE_RAD=0.,SETUP={},ENERGY_EV=710.,USE_PARTIAL_COHERENCE=False,
                    OBSERVATION_WORKERS=1,FFT_WORKERS=1,WARMUP_ITERATIONS=[2,1],positive=0,
                    negative=2,INNER_ITERATIONS=1,OUTER_ITERATIONS=1,PHYSICAL_ITERATIONS=2,
                    PROJECTION_RELAXATION=1.,state_labels=states)
                exec(''.join(notebook['cells'][3]['source']).split('universal.print_universal_workflow')[0],ns)
                ns['recipe'].update(crop=3,binning=2)  # Odd 9x9 return grid from 24x24 inputs.
                with contextlib.redirect_stdout(io.StringIO()):
                    fields,warmup,components,bsmasks,errors=u.universal_phase_retrieval_algorithm(
                        data,masks,support,states,[710.]*3,[1.]*3,['beam0']*3,universal_recipe=ns['recipe'])
                ns.update(fields=fields,warmup=warmup,components=components,bsmasks=bsmasks,errors=errors,
                    holograms=data,mask_pixel=masks,support=support,geometry=geometry,fth=fth,plt=plt,
                    inputs=inputs,json=json,h5py=h5py,FIGURES=Path(folder),OUTPUT=Path(folder)/'result.h5',
                    current=np.array([1.,0.,-1.]),indices=[0,20,40],ROI_CENTERS=None,ROI_RADIUS=1,config={})
                with contextlib.redirect_stdout(io.StringIO()):
                    for i in [4,5,6,7]: exec(''.join(notebook['cells'][i]['source']),ns)
                primary=fields if fields.ndim==3 else fields[:,0]
                np.testing.assert_allclose(ns['fth_differences'],np.stack([fth.reconstruct(f-primary[0]) for f in primary]))
                np.testing.assert_array_equal(ns['fth_differences'][0],0)
                self.assertEqual(ns['state_grid'].__kwdefaults__['crop'], ns['roi'])
                self.assertLess(ns['fth_aperture'][ns['fth_roi']].size, ns['fth_aperture'].size)
                np.testing.assert_array_equal(ns['object_differences'][0], 0)
                self.assertTrue(np.isfinite(ns['final_error']).all())
                self.assertLess(ns['model_relative_error'],1e-8)
                self.assertTrue(np.isnan(ns['final_curves']).all())
                for name in ['magnetization_maps.png','fth_field_differences_real.png',
                             'fth_field_differences_amplitude.png','data_diagnostics.png','local_hysteresis.png',
                             'warmup_field_difference_real.png','warmup_field_difference_amplitude.png',
                             'final_field_difference_real.png','final_field_difference_amplitude.png',
                             'field_diagnostics_summary.png','mode_reconstructions_all.png']:
                    self.assertTrue((Path(folder)/name).is_file())
                with h5py.File(ns['OUTPUT']) as h:
                    self.assertEqual(h['holograms'].shape,primary.shape)
                    np.testing.assert_allclose(h['fth_field_differences'][()],ns['fth_differences'])
                self.assertEqual(plt.get_fignums(),[])

if __name__=='__main__': unittest.main()
