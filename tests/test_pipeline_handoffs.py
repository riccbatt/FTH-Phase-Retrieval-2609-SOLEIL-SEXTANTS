"""Exercise notebook handoffs with small synthetic data, without beamline files."""
import ast
from copy import deepcopy
import hashlib
import h5py
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fth-test-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'library')]
import pipeline_io as io
import fth_phase_workflow as wf
from data_loading import Frame, SextantsNexusLoader, load_average
from beamstop_stitching import stitch_images
from image_preprocessing import fit_dark_frame, load_detector_masks
import fthcore as fth
import CCI_core as cci


def cell(notebook, index):
    notebook = json.loads((ROOT / (notebook + '.ipynb')).read_text())
    source = ''.join(notebook['cells'][index]['source'])
    return '\n'.join((' ' * (len(line)-len(line.lstrip())) + 'pass') if line.lstrip().startswith('%') else line for line in source.splitlines())


class Handoffs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = self.temp.name
        self.shape = (32, 32)
        self.mask = np.zeros(self.shape, dtype=np.uint8)
        self.image = np.arange(1024, dtype=float).reshape(self.shape) + 1
        self.setup = {'ccd_dist': .5, 'energy': 780., 'px_size': 11e-6}

    def tearDown(self):
        plt.close('all')
        self.temp.cleanup()

    def test_ids_and_names(self):
        for value in (1144, [1144, 1146], [[1144, 1146], [1148]], np.array([1144, 1146])):
            self.assertEqual(io.first_id(value), 1144)
            self.assertTrue(io.result_path(self.base, value, 'rb').endswith('data_recon_ImId_1144_rb.hdf5'))
        self.assertNotEqual(io.result_path(self.base, 1144, 'rb'), io.result_path(self.base, 1144, 'rb', series=True))
        with self.assertRaises(ValueError):
            io.first_id([])

    def test_stitch_export_fth_load_and_geometry(self):
        frames = [Frame(str(i), self.image + i, 1., Path(str(i)), {'ccd_dist_m': .5}) for i in range(4)]
        masks = [self.mask.copy() for _ in frames]
        masks[0][3:7, 3:7] = 1
        masks[1][10:14, 10:14] = 1
        # Auxiliary inputs fill the reference holes.
        results = [stitch_images([frames[i], frames[i+2]], [masks[i], masks[i+2]],
                                register=False, fit_intensity=False, use_master_where_valid=True)
                   for i in (0, 1)]
        env = dict(BASEFOLDER=self.base, USER='rb', np=np, os=os, join=os.path.join,
                   first_id=io.first_id, stitched_path=io.stitched_path,
                   group_id_lists=[[1144,1146],[1164],[1304],[1315]],
                   PLUS_REFERENCE_INPUT=0, MINUS_REFERENCE_INPUT=1,
                   POLARIZATIONS=['+','-','+','-'],
                   stitched_image_ids=[[[1144,1146],[1304]], [[1164],[1315]]],
                   stitch_results=results, FIT_ORDER=None, MASK_SHIFTS=[(0,0)]*4,
                   FIT_ROIS=[np.s_[:,:]]*4, mask_detectors=[self.mask]*4,
                   unshifted_mask_beamstops=masks, mask_beamstops=masks, mask_pixels=masks,
                   frames=frames, INPUT_KINDS=['raw']*4, BEAMSTOP_MASK_DILATIONS=[0]*4,
                   DARK_ID_GROUPS=[904]*4, energies_eV=[780.]*4)
        env.update(stitch_images=stitch_images, FIND_IMAGE_SHIFTS=False, MAX_IMAGE_SHIFT=10,
                   REGISTRATION_UPSAMPLE=1, FIT_PERCENTILES=(2,98))
        exec(cell('00b_stitch_beamstops_000', 23), env)
        exec(cell('00b_stitch_beamstops_000', 24), env)
        plus, minus = env['output_files']
        single = dict(env, frames=frames[:2], mask_pixels=masks[:2],
                      group_id_lists=env['group_id_lists'][:2], POLARIZATIONS=['+','-'])
        exec(cell('00b_stitch_beamstops_000', 23), single)
        self.assertEqual(len(single['stitch_results'][0].prepared_frames), 1)

        self.assertEqual(Path(plus).name, 'stitched_ImId_1144_plus_rb.npz')
        self.assertEqual(Path(minus).name, 'stitched_ImId_1144_minus_rb.npz')
        self.assertEqual(io.stitched_ids(self.base, plus), [1144,1146])
        loaded = io.load_stitched(self.base, os.path.relpath(plus, self.base))
        self.assertTrue(np.isfinite(loaded[0]).all())
        self.assertEqual(int(loaded[3][4,4]), 0)
        self.assertEqual(io.stitched_setup_metadata(self.base, plus)['ccd_dist_m'], .5)
        # Execute the actual FTH setup cell: any raw access fails this test.
        class NoRaw:
            def load(self, key):
                raise AssertionError('Stitched input must not reopen raw files')
        env.update(raw_loader=NoRaw(), SextantsNexusLoader=lambda _:NoRaw(), RAW_FOLDER='unavailable',
                   RAW_DATA_KIND='sextants_nexus', PLUS_STITCHED_FILE=plus,
                   hologram_inputs={'+':{'id':[1144,1146]}, '-':{'id':[1164]}},
                   image_ids=io.acquisition_ids, positive_label='+', reference_label='-',
                   stitched_setup_metadata=io.stitched_setup_metadata, result_path=io.result_path,
                   folder_logs=str(Path(self.base)/'processed'/'Logs'),
                   helper=type('Helper',(),{'photon_energy_wavelength':staticmethod(lambda *a,**k:1e-9)}),
                   loading=type('Loading',(),{'mnemonics':{}}), heraldo=False, crop=None)
        exec(cell('01_FTH',8),env)
        self.assertEqual(env['experimental_setup']['ccd_dist'], .5)

    def test_legacy_stitch_mask_and_nonfinite_pixels(self):
        path=Path(self.base)/'old.npz'
        image=self.image.copy();image[0,0]=np.nan
        mask=self.mask.copy();mask[1,1]=1
        np.savez(path,image=image,mask_detector=self.mask,mask_beamstop=self.mask,
                 mask_pixel=mask,energy_eV=780.,reference_exposure=1.,reference_image_id=1144)
        loaded=io.load_stitched(self.base,'old.npz')
        self.assertEqual(io.stitched_ids(self.base,'old.npz'),[1144])
        self.assertTrue(np.isfinite(loaded[0]).all())
        self.assertEqual(loaded[3][0,0],1)
        self.assertEqual(loaded[3][1,1],1)

    def test_series_keeps_reference_and_loads_in_phase_retrieval(self):
        calibration = dict(experimental_setup=self.setup, positive_label='+', reference_label='-',
                           hologram_labels=['+','-'], center=[16.,16.], mask_pixel=self.mask,
                           focus_fth={'roi':[0,32,0,32]}, factor=1.7,offset=.4,
                           holo={'+':{'id':[1144,1146],'image_c':self.image},
                                 '-':{'id':[1164],'image_c':self.image}}, supportmask=np.ones(self.shape))
        focus_values = dict(prop_dist=13.37, phase=-.61, dx=.24, dy=-.17)
        focus_env = dict(data={}, wf=wf, fth=fth, experimental_setup=self.setup,
                         factor=calibration['factor'], np=np,
                         temp_p=(self.image+1145)/1.7,
                         temp_n=self.image+1164+.4, roi=[0,32,0,32], roi_s=np.s_[:,:],
                         slider_prop=SimpleNamespace(value=focus_values['prop_dist']),
                         slider_phase=SimpleNamespace(value=focus_values['phase']),
                         slider_dx=SimpleNamespace(value=focus_values['dx']),
                         slider_dy=SimpleNamespace(value=focus_values['dy']), focus_operation='-')
        exec(cell('01_FTH',34),focus_env)
        calibration.update(focus_env['data'])
        calibration['fth_transmission'] = np.ones(self.shape)
        # Independently evaluate the original focus-widget math.
        import fth_reconstruction as widget_fth
        def widget_image(image):
            return widget_fth.reconstructCDI(widget_fth.propagate(
                image, focus_values['prop_dist']*1e-6, self.setup)*np.exp(1j*focus_values['phase']))
        expected = widget_fth.sub_pixel_centering(
            widget_image(focus_env['temp_p'])-widget_image(focus_env['temp_n']),
            focus_values['dx'],focus_values['dy'])
        np.testing.assert_allclose(calibration['recon'], expected, rtol=1e-12, atol=1e-10)
        reference=io.result_path(self.base,1144,'rb')
        wf.save_data_dict(calibration,reference)
        digest=hashlib.sha256(Path(reference).read_bytes()).digest()
        class Loader:
            def __init__(self, folder): pass
            def load(inner,key):
                return Frame(str(key),self.image+key,1.,Path(str(key)),{})
        recipe={'radius':2,'order':4};smooth={'dilation_pixels':0,'sigma':0}
        env=dict(BASEFOLDER=self.base,USER='rb',RAW_FOLDER='unused',REFERENCE_IMAGE_ID=1144,
                 REFERENCE_H5=reference,SERIES_H5=str(Path(self.base)/'processed/Logs/all.hdf5'),
                 SERIES=[{'+':[1144,1146],'-':[1164],'point':0},{'+':1150,'-':[1170,1172],'point':1}],
                 SextantsNexusLoader=Loader, image_ids=io.acquisition_ids, first_id=io.first_id,
                 result_path=io.result_path, load_average=load_average, deepcopy=deepcopy,
                 wf=wf,cci=cci,fth=fth,np=np,plt=plt,join=os.path.join,calibration=calibration,
                 labels=['+','-'],experimental_setup=self.setup,center=[16,16],mask_pixel=self.mask,
                 beamstop_recipe=recipe,mask_pixel_fth_recipe=smooth,focus_fth=calibration['focus_fth'],
                 project_ewalds_sphere=False,ewald_method='cubic',**focus_values,
                 roi=[0,32,0,32],roi_s=np.s_[:,:],mask_multiplier=np.ones(self.shape),
                 DARK_IDS={'+': None, '-': None}, REUSE_CALIBRATION_SCALING=True, OVERWRITE=True,
                 fit_dark_frame=fit_dark_frame)
        exec(cell('01_FTH_series',10),env)
        self.assertEqual(hashlib.sha256(Path(reference).read_bytes()).digest(),digest)
        self.assertEqual(len(wf.load_data_dict(env['SERIES_H5'])['series']),2)
        for key in (1144,1150):
            loaded=wf.load_data_dict(io.result_path(self.base,key,'rb',series=True))
            self.assertEqual(io.first_id(loaded['holo']['+']['id']),key)
            self.assertIn('experimental_setup',loaded)
            self.assertIn('positive_label',loaded)
            self.assertEqual(loaded['holo']['+']['image_c'].shape,self.shape)
            self.assertEqual(loaded['focus_fth']['roi'],[0,32,0,32])
            if key == 1144:
                from mask_lib import circle_mask
                multiplier = 1 - circle_mask(self.shape, center=[16, 16], radius=100, sigma=5)
                expected_series = wf.fth_reconstruct(
                    loaded['hologram'] * multiplier, self.setup, fth, **focus_values,
                )
                np.testing.assert_allclose(loaded['recon'], expected_series, rtol=1e-12, atol=1e-10)
        # Execute the phase-retrieval loading cell against the saved averaged point.
        source=cell('04_phase_retrieval',5).replace('BASEFOLDER = find_basefolder()',f'BASEFOLDER = {self.base!r}')
        source=source.replace('FTH_SOURCE = "fth"','FTH_SOURCE = "series"')
        env.update(helper=type('Helper',(),{'create_folder':staticmethod(lambda p: (Path(p).mkdir(parents=True,exist_ok=True) or p))}))
        exec(source,env)
        self.assertEqual(env['loaded_im_ids'],[1144,1146])
        self.assertEqual(env['topo_id'],1164)
        # A short CPU retrieval verifies that the persisted arrays satisfy the engine.
        import phase_retrieval_core_unified as pr
        recipe = pr.default_phase_retrieval_recipe()
        recipe.update(algorithm_list=['ER']*6, number_iterations=[2]*6,
                      helicity=['+','+','-','+','+','-'], RL_its=[0]*6,
                      average_img=[1]*6, plot_every=[100]*6,
                      Startimage=[None]*6, Startgamma=[None]*6,
                      output=[True]*6, return_format='dict')
        result = pr.phase_retrieval_algorithm(
            {label: env['data']['holo'][label]['image_c'] for label in ('+','-')},
            mask_pixel=env['data']['mask_pixel'], supportmask=env['data']['supportmask'],
            phase_retrieval_recipe=recipe)
        self.assertEqual(set(result['full_coherence']), {'+','-'})


    def test_raw_single_and_averaged_nexus_with_filename_variants(self):
        for index, prefix in [(7, 'scanxx_'), (8, 'scanx_'), (9, 'scan_')]:
            path = Path(self.base) / f'{prefix}{index:04d}.nxs'
            with h5py.File(path, 'w') as handle:
                scan = handle.create_group(f'scan_{index:04d}/scan_data')
                dataset = scan.create_dataset('data_21', data=np.full((2, 32, 32), index, dtype=np.uint16))
                dataset.attrs['interpretation'] = 'image'
                scan.create_dataset('data_03', data=[200.])
                handle.create_dataset('scan_data/integration_times', data=[1., 1.])
                handle.create_dataset('SEXTANTS/mono/energy', data=780.)
        loader = SextantsNexusLoader(self.base)
        self.assertEqual(loader.load(7).metadata['ccd_dist_m'], .5)
        self.assertEqual(load_average(loader, 7).image.dtype, np.float64)
        np.testing.assert_allclose(load_average(loader, [[7, 8], [9]]).image, 8.)
        self.assertEqual(load_average(loader, [7, 8]).metadata['average_count'], 2)

    def test_detector_mask_filename_and_explicit_disable(self):
        from PIL import Image
        detector = np.zeros((32, 32, 3), dtype=np.uint8)
        beamstop = detector.copy()
        detector[0, 0] = [255, 0, 0]
        beamstop[1, 1] = [255, 0, 0]
        Image.fromarray(detector).save(Path(self.base)/'mask_detector.png')
        Image.fromarray(beamstop).save(Path(self.base)/'mask_beamstop_7.png')
        _, _, combined = load_detector_masks(self.base, 7, self.shape, detector_file="mask_detector.png")
        self.assertEqual(combined[0, 0], 1)
        self.assertEqual(combined[1, 1], 1)
        _, _, disabled = load_detector_masks(self.base, 7, self.shape)
        self.assertEqual(disabled[0, 0], 0)
        self.assertEqual(disabled[1, 1], 1)

    def test_notebook_syntax(self):
        for name in ('01_FTH','01_FTH_series','04_phase_retrieval','00b_stitch_beamstops_000'):
            notebook=json.loads((ROOT/(name+'.ipynb')).read_text())
            for index,item in enumerate(notebook['cells']):
                if item['cell_type']=='code':
                    ast.parse(cell(name,index),filename=f'{name}:cell {index}')

if __name__=='__main__':
    unittest.main()
