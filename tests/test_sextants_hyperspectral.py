"""Raw NeXus/PNG -> physical fit -> portable HDF5; no beamline access needed."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/fth-test-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'library'))
import sextants_hyperspectral as hs
import fth_phase_workflow as wf
from mask_store import save_red_mask_png


class Hyperspectral(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.base=Path(self.temp.name)
        self.raw=self.base/'raw'; self.raw.mkdir()
        self.masks=self.base/'processed/mask_pixels';self.masks.mkdir(parents=True)
        self.shape=(32,32)
        self.support=hs.circle_support(self.shape,[(22,22,4),(16,16,1.5)])
        self.settings=dict(detector_center=[16.,16.],pixel_size_m=11e-6,
             focus={'prop_dist':0.,'phase':0.,'dx':0.,'dy':0.},
             dark_fit_recipe={'rows':[0,4],'columns':[0,4],'percentile':100,'stride':1})
        yy,xx=np.mgrid[:32,:32]
        self.pairs=[]
        material=hs.material_from_support(self.support)
        for ids,energy in [(100,775.),(200,785.),(300,796.)]:
            self.pairs.append(dict(plus=ids,minus=ids+1,energy_eV=energy))
            for image_id,sign in [(ids,1),(ids+1,-1)]:
                log=material*(-.1 + (energy-775)*.0005 + sign*.02*np.sin(xx)*(.5+.2j))
                obj=self.support*np.exp(log)
                field=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(obj)))
                self.write_raw(image_id,np.abs(field)**2*1e6+10,energy)
        self.write_raw(102,np.ones(self.shape)*30,775.)
        self.write_raw(9,np.full(self.shape,10.),775.)
        mask=np.zeros(self.shape,dtype=np.uint8);mask[14:18,14:18]=1
        save_red_mask_png(self.masks/'mask_beamstop_7.png',mask)

    def tearDown(self):
        plt.close('all');self.temp.cleanup()

    def write_raw(self,image_id,image,energy):
        with h5py.File(self.raw/f'scanxx_{image_id:04d}.nxs','w') as f:
            scan=f.create_group(f'scan_{image_id:04d}/scan_data')
            data=scan.create_dataset('data_21',data=np.stack([image,image]))
            data.attrs['interpretation']='image'
            scan.create_dataset('data_03',data=[200.])
            f.create_dataset('scan_data/integration_times',data=[1.,1.])
            f.create_dataset('SEXTANTS/mono/energy',data=energy)

    def load(self,pairs=None):
        return hs.load_energy_scan(pairs or self.pairs,self.raw,self.masks,
            **self.settings, supportmask=self.support, beamstop_mask_ids={'+':7,'-':7},dark_ids={'+':9,'-':9})

    def test_streaming_raw_reader_does_not_materialize_the_stack(self):
        original = h5py.Dataset.__getitem__
        def bounded_read(dataset, key):
            if dataset.name.endswith('/data_21') and key == ():
                raise AssertionError('Whole detector stack must not be read')
            return original(dataset, key)
        with patch.object(h5py.Dataset, '__getitem__', bounded_read):
            dataset = self.load()
        self.assertEqual(dataset['holograms'].shape,(6,32,32))

    def test_raw_loading_and_same_energy_average(self):
        pairs=[dict(self.pairs[0],plus=[100,102])]+self.pairs[1:]
        dataset=self.load(pairs)
        self.assertEqual(dataset['holograms'].shape,(6,32,32))
        self.assertEqual(dataset['records'][0]['ids'],[100,102])
        self.assertEqual(dataset['im_id'],100)
        self.assertTrue(np.isfinite(dataset['holograms']).all())
        self.assertTrue(np.all(dataset['mask_pixel'][:,16,16]))
        np.testing.assert_array_equal(dataset['supportmask'],self.support)
        self.assertFalse((self.masks/'mask_detector.png').exists())
        loader=hs.SextantsNexusLoader(self.raw)
        expected=(loader.load(100).image+loader.load(102).image)/2-10.
        self.assertAlmostEqual(float(dataset['holograms'][0,5,5]),float(expected[5,5]),places=4)

    def test_cross_energy_average_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'do not average energies'):
            self.load([dict(self.pairs[0],plus=[100,200])])

    def test_common_q_resampling_protects_missing_pixels(self):
        image=np.ones((32,32));mask=np.zeros((32,32),dtype=np.uint8);mask[16,16]=1
        output,missing=hs.resample_common_q(image,mask,.5)
        self.assertEqual(missing[16,16],1)
        self.assertEqual(output[16,16],0.)
        self.assertAlmostEqual(output[10,10],.25)
        _,missing=hs.resample_common_q(image,mask,1.2)
        self.assertEqual(missing[0,0],1)

    def short_recipe(self,dataset,partial=False):
        recipe=hs.default_hyperspectral_recipe(dataset,partial_coherence=partial,modes=[1])
        recipe.update(warmup_mode=['ER'],warmup_Nit=[3],warmup_RL_it=[0],warmup_RL_freq=[100],
                      inner_mode=['ER'],inner_Nit=[4],outer_iterations=1,physical_iterations=2,
                      RL_it=1 if partial else 0,RL_freq=1,plot_every=100)
        return recipe

    def test_joint_physical_fit_and_portable_save(self):
        dataset=self.load()
        for partial in (False,True):
            with self.subTest(partial=partial):
                recipe=self.short_recipe(dataset,partial)
                recipe['projection_focus_phase_rad']=.17
                result=hs.reconstruct(dataset,recipe)
                self.assertTrue(np.isfinite(result['fields']).all())
                self.assertEqual(result['focus']['phase'],.17)
                self.assertEqual(result['components']['projection_model'],'physical_factorized')
                self.assertEqual(np.asarray(result['components']['magnetization']).shape,(1,32,32))
                self.assertEqual(np.asarray(result['components']['charge_response']).shape,(3,))
                detector_errors=hs.detector_relative_errors(dataset,result)
                self.assertEqual(detector_errors.shape,(6,))
                self.assertTrue(np.isfinite(detector_errors).all())
                output=self.base/f'result_{partial}.hdf5'
                hs.save_result(result,output)
                loaded=wf.load_data_dict(output)
                self.assertEqual(loaded['records'][0]['ids'],[100])
                self.assertEqual(loaded['recipe']['projection_model'],'physical_factorized')
                self.assertTrue(np.isfinite(loaded['fields']).all())
                with self.assertRaises(FileExistsError):hs.save_result(result,output)

    def test_multimode_physical_reconstruction(self):
        dataset=self.load();recipe=self.short_recipe(dataset)
        recipe['modes']=[1,2]
        result=hs.reconstruct(dataset,recipe)
        self.assertEqual(result['fields'].shape,(6,2,32,32))
        self.assertTrue(np.isfinite(result['fields']).all())
        self.assertEqual(result['components']['physical_model_modes'],[1])

    def test_common_q_focus_is_reversible_and_uses_energy(self):
        fields=np.ones((2,16,16),dtype=complex)
        setup=dict(ccd_dist=.5,px_size=11e-6,reference_energy_eV=775.,dx=.24,dy=-.17)
        energy=[775.,796.]
        focused=hs.engine._projection_focus_transform(fields,energy,3.,.2,setup)
        restored=hs.engine._projection_focus_transform(focused,energy,3.,.2,setup,inverse=True)
        np.testing.assert_allclose(restored,fields,atol=1e-12)
        self.assertFalse(np.allclose(focused[0],focused[1],atol=1e-12,rtol=1e-12))

    def test_physical_projection_reproduces_known_log_objects(self):
        yy,xx=np.mgrid[:8,:8]
        magnetization=np.sin(xx*.6)*np.cos(yy*.3)
        common=.02*np.cos(xx)+.03j*np.sin(yy)
        charge=[-.1+.2j,-.2+.15j,-.15+.1j]
        magnetic=[.01+.02j,.025+.01j,.02+.03j]
        logs=np.stack([common+charge[e]+pol*magnetic[e]*magnetization
                       for e in range(3) for pol in (1,-1)])
        recipe=hs.engine.default_general_phase_retrieval_recipe()
        recipe.update(clip_magnetization=False)
        fitted,components=hs.engine.project_log_objects_physical(
            logs,['sample']*6,[775,775,785,785,796,796],[1,-1]*3,['beam0']*6,
            material_thickness=np.ones((8,8)),iterations=50,recipe=recipe,return_components=True)
        np.testing.assert_allclose(fitted,logs,atol=1e-9)
        self.assertLess(components['fit_residual_rms'],1e-9)

    def test_partial_stage_keeps_beamstop_excluded(self):
        recipe=hs.engine.default_general_phase_retrieval_recipe()
        recipe.update(Nmodes=1,partial_coherence=True,inner_mode=['ER'],inner_Nit=[4],RL_it=1,RL_freq=1)
        schedule=hs.engine._build_update_schedule(recipe,name='inner')
        amplitude=np.ones((8,8));mask=np.zeros((8,8),dtype=np.uint8);mask[3:5,3:5]=1
        captured=[]
        def kernel(**kwargs):
            captured.append(kwargs['bsmask'].copy())
            np.testing.assert_array_equal(kwargs['diffract'],amplitude)
            return kwargs['Phase'].copy(),[.1],[.1],kwargs['gamma']
        hs.engine._run_update_schedule(np.ones((8,8),complex),amplitude,np.ones((8,8)),mask,
                                      schedule,recipe,phase_retrieval_kernel=kernel)
        self.assertEqual(len(captured),1)
        np.testing.assert_array_equal(captured[0],mask)

    def test_standalone_focus_phase_and_ramp_at_zero_propagation(self):
        fields=np.arange(256,dtype=float).reshape(1,16,16).astype(complex)
        setup=dict(ccd_dist=.5,px_size=11e-6,reference_energy_eV=775.,dx=.24,dy=-.17)
        transformed=hs.engine._projection_focus_transform(fields,[775.],0.,.3,setup)
        obj=np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(fields,axes=(-2,-1)),axes=(-2,-1)),axes=(-2,-1))
        yy,xx=np.mgrid[:16,:16]
        expected=obj*np.exp(.3j)*np.exp(2j*np.pi*((xx-8)*.24/16+(yy-8)*(-.17)/16))
        actual=np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(transformed,axes=(-2,-1)),axes=(-2,-1)),axes=(-2,-1))
        np.testing.assert_allclose(actual,expected,atol=1e-9)
        restored=hs.engine._projection_focus_transform(transformed,[775.],0.,.3,setup,inverse=True)
        np.testing.assert_allclose(restored,fields,atol=1e-9)

    def opening_parameters(self):
        notebook=json.loads((ROOT/'05_hyperspectral_physical.ipynb').read_text())
        namespace=dict(BASEFOLDER=self.base,hs=hs,np=np)
        exec(''.join(notebook['cells'][3]['source']),namespace)
        return namespace['RECIPE_PARAMETERS']

    def test_opening_recipe_lists_every_engine_key_explicitly(self):
        notebook=json.loads((ROOT/'05_hyperspectral_physical.ipynb').read_text())
        source=''.join(notebook['cells'][3]['source'])
        assignment=next(node for node in ast.parse(source).body
                        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RECIPE_PARAMETERS' for t in node.targets))
        self.assertIsInstance(assignment.value,ast.Dict)
        keys=[ast.literal_eval(key) for key in assignment.value.keys]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(set(keys),set(hs.engine.default_universal_phase_retrieval_recipe()))
        for cell in notebook['cells'][4:]:
            if cell['cell_type']!='code':continue
            for node in ast.walk(ast.parse(''.join(cell['source']))):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertFalse(isinstance(node.func.value,ast.Name) and node.func.value.id=='recipe' and node.func.attr=='update')

    def test_explicit_recipe_preserves_edits_and_resolves_only_data_bindings(self):
        dataset=self.load();parameters=self.opening_parameters()
        parameters.update(beta_zero=.123,outer_iterations=3,physical_iterations=7,
                          binning=2,fit_material_thickness=False,material_mask=None)
        dataset['focus']['dx']=.4
        resolved=hs.resolve_hyperspectral_recipe(dataset,parameters)
        self.assertEqual(resolved['beta_zero'],.123)
        self.assertEqual(resolved['outer_iterations'],3)
        self.assertEqual(resolved['physical_iterations'],7)
        self.assertFalse(resolved['fit_material_thickness'])
        self.assertIsNone(resolved['material_mask'])
        self.assertEqual(resolved['projection_every'],6)
        np.testing.assert_array_equal(resolved['energy_values'],[775,785,796])
        self.assertAlmostEqual(resolved['projection_focus_setup']['px_size'],22e-6)
        self.assertAlmostEqual(resolved['projection_focus_setup']['dx'],.2)
        self.assertEqual(parameters['projection_every'],hs.AUTO_FROM_DATASET)

    def test_incomplete_or_unknown_recipe_keys_are_rejected(self):
        dataset=self.load();parameters=self.opening_parameters()
        incomplete=dict(parameters);incomplete.pop('RL_it')
        with self.assertRaisesRegex(ValueError,'coverage mismatch'):
            hs.resolve_hyperspectral_recipe(dataset,incomplete)
        with self.assertRaisesRegex(ValueError,'coverage mismatch'):
            hs.resolve_hyperspectral_recipe(dataset,dict(parameters,misspelled_parameter=1))

    def test_auto_cannot_hide_arbitrary_numerical_controls(self):
        dataset=self.load();parameters=self.opening_parameters()
        parameters['beta_zero']=hs.AUTO_FROM_DATASET
        with self.assertRaisesRegex(ValueError,'AUTO is not supported'):
            hs.resolve_hyperspectral_recipe(dataset,parameters)

    def test_new_notebook_runs_end_to_end_with_synthetic_files(self):
        notebook=json.loads((ROOT/'05_hyperspectral_physical.ipynb').read_text())
        settings=dict(BASEFOLDER=self.base,RAW_FOLDER=str(self.raw),DETECTOR_CENTER=(16.,16.),
                      FOCUS={'prop_dist':0.,'phase':0.,'dx':0.,'dy':0.},
                      DARK_FIT_RECIPE=self.settings['dark_fit_recipe'],
                      ENERGIES_EV=[775.,785.,796.],PLUS_IMAGE_IDS=[100,200,300],
                      DARK_IDS={'+':9,'-':9},BEAMSTOP_MASK_IDS={'+':7,'-':7},
                      SUPPORT_COORDINATES=[(22,22,4),(16,16,1.5)],MODES=[1])
        namespace={}
        self.assertFalse((self.base/'processed/Logs').exists())
        with patch.object(wf,'load_data_dict',side_effect=AssertionError('No prior reconstruction may be loaded')):
            for cell in notebook['cells']:
                if cell['cell_type']!='code':continue
                source=''.join(cell['source'])
                for statement in ast.parse(source).body:
                    targets=statement.targets if isinstance(statement,ast.Assign) else []
                    names=[t.id for t in targets if isinstance(t,ast.Name)]
                    if len(names)==1 and names[0] in settings:
                        namespace[names[0]]=settings[names[0]]
                    else:exec(compile(ast.Module(body=[statement],type_ignores=[]),'<hyperspectral notebook>','exec'),namespace)
                if 'print_universal_workflow' in source:
                    namespace['recipe'].update(self.short_recipe(namespace['dataset']))
        self.assertTrue(namespace['RESULT_H5'].is_file())
        self.assertTrue(namespace['PREPARED_H5'].is_file())
        self.assertTrue(namespace['FIGURE_PNG'].is_file())
        self.assertEqual(len(namespace['result']['records']),6)
        self.assertTrue(namespace['recipe']['fit_material_thickness'])
        np.testing.assert_array_equal(namespace['recipe']['material_mask'],self.support.astype(bool))
        self.assertFalse(list(self.base.rglob('data_recon*.hdf5')))

if __name__=='__main__':unittest.main()
