import importlib.util
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
import json
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('xpcs_inputs',ROOT/'2601_XPCS/reconstruction_inputs.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)

class XPCSTests(unittest.TestCase):
    def test_notebook_returns_shared_charge_and_secondary_mode(self):
        from library import phase_retrieval_universal as u
        notebook=json.loads((ROOT/'2601_XPCS/01_physical_hysteresis_263.ipynb').read_text())
        source=''.join(notebook['cells'][3]['source']).split('universal.print_universal_workflow')[0]
        support=np.zeros((16,16));support[5:11,5:11]=1;support[2,2]=1
        material=support.copy();material[2,2]=0
        states=['positive','interior','negative']
        for modes in ([1], [1,2]):
            with self.subTest(modes=modes):
                ns=dict(universal=u,np=np,MODES=modes,material=material,
                    MODE_2_COMMON=True,CLIP_MAGNETIZATION=True,
                    saturated={'positive':1.,'negative':-1.},FOCUS_PROP_UM=0.,
                    FOCUS_PHASE_RAD=0.,SETUP={},ENERGY_EV=710.,USE_PARTIAL_COHERENCE=False,
                    OBSERVATION_WORKERS=1,FFT_WORKERS=1,WARMUP_ITERATIONS=[2,1],positive=0,
                    INNER_ITERATIONS=1,OUTER_ITERATIONS=2,PHYSICAL_ITERATIONS=20,
                    PROJECTION_RELAXATION=1.,state_labels=states)
                exec(source,ns)
                ns["recipe"]["crop"] = 0  # Synthetic detector is only 16 by 16 pixels.
                data=np.stack([np.ones((16,16))*v for v in (1.,1.2,.8)])
                with contextlib.redirect_stdout(io.StringIO()):
                    fields,_,c,_,_=u.universal_phase_retrieval_algorithm(
                        data,np.zeros_like(data),support,states,[710.]*3,[1.]*3,['beam0']*3,
                        universal_recipe=ns['recipe'])
                primary=fields if fields.ndim==3 else fields[:,0]
                objects=np.exp(u.fourier_field_to_object_log(primary,unwrap_energy_phase=False))
                charge=np.exp(c['common_log_objects'][0]+c['material_thickness']*c['charge_response'][0])
                magnetic=c['material_thickness'][None]*c['magnetic_response'][0]*c['magnetization']
                active=c['material_thickness']>0
                np.testing.assert_allclose(objects[:,active],(charge[None]*np.exp(magnetic))[:,active],atol=1e-9)
                np.testing.assert_allclose(c['magnetization'][0,active],1.)
                np.testing.assert_allclose(c['magnetization'][2,active],-1.)
                # Removing magnetism must give the same charge inside the object aperture.
                recovered_charge=objects*np.exp(-magnetic)
                np.testing.assert_allclose(recovered_charge[:,active],np.broadcast_to(charge,objects.shape)[:,active],atol=1e-9)
                self.assertTrue(ns['recipe']['physical_projection_object_roi'])
                # The primary physical projector must preserve reference holes and exterior.
                rng=np.random.default_rng(42)
                logs=rng.normal(size=(3,16,16)) + .1j*rng.normal(size=(3,16,16))
                projected=u.project_log_objects_physical(
                    logs,states,[710.]*3,[1.]*3,['beam0']*3,
                    recipe=ns['recipe'],material_thickness=np.fft.fftshift(material),
                    saturated_states=ns['saturated'])
                outside=np.fft.fftshift(material)==0
                np.testing.assert_array_equal(projected[:,outside],logs[:,outside])
                if fields.ndim==4:
                    np.testing.assert_allclose(fields[:,1],np.broadcast_to(fields[0,1],fields[:,1].shape))

    def test_optional_magnetization_clipping(self):
        from library import phase_retrieval_universal as u
        states = ['positive', 'negative', 'above', 'below']
        values = np.array([1., -1., 1.6, -1.6])
        logs = np.broadcast_to(values[:, None, None]*(.2+.1j), (4, 2, 2)).copy()
        for clip in (True, False):
            for roi in (True, False):
                with self.subTest(clip=clip, roi=roi):
                    recipe = u.default_universal_phase_retrieval_recipe()
                    recipe.update(clip_magnetization=clip, physical_projection_object_roi=roi)
                    _, components = u.project_log_objects_physical(
                        logs, states, [710.]*4, [1.]*4, ['beam']*4,
                        recipe=recipe, material_thickness=np.ones((2, 2)),
                        saturated_states={'positive': 1., 'negative': -1.},
                        iterations=200, return_components=True)
                    m = components['magnetization']
                    np.testing.assert_allclose(m[:2], np.broadcast_to(values[:2, None, None], (2, 2, 2)))
                    if clip:
                        self.assertLessEqual(np.max(np.abs(m)), 1.)
                        self.assertEqual(components['magnetization_bounds'], (-1., 1.))
                    else:
                        np.testing.assert_allclose(m, np.broadcast_to(values[:, None, None], m.shape), atol=1e-6)
                        self.assertIsNone(components['magnetization_bounds'])

    def test_subpixel_hologram_alignment(self):
        from scipy.ndimage import gaussian_filter, shift
        rng = np.random.default_rng(25)
        reference = 100 + 20*gaussian_filter(rng.normal(size=(96, 96)), 1.5)
        displacement = np.array([1.35, -2.65])
        moving = 1.7*shift(reference, displacement, order=3, mode='nearest') + 12
        mask = np.zeros(reference.shape, bool); mask[35:44, 40:48] = True
        reference[mask] = 1e9; moving[mask] = -1e9
        correction, scores = x.estimate_hologram_shift(reference, moving, mask, mask,
            max_shift=5., upsample_factor=100, roi=(5, 91, 5, 91))
        np.testing.assert_allclose(correction, -displacement, atol=.08)
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], .99)
        with self.assertRaises(ValueError):
            x.estimate_hologram_shift(reference, moving, mask, mask, max_shift=.5)
        with self.assertRaises(ValueError):
            x.estimate_hologram_shift(np.ones((96, 96)), moving, mask, mask)

    def test_alignment_cache_and_masks(self):
        from scipy.ndimage import gaussian_filter, shift
        rng = np.random.default_rng(12)
        reference = 100 + 30*gaussian_filter(rng.normal(size=(64, 64)), 1.)
        delta = (1.3, -1.7)
        moving = shift(reference, delta, order=3, mode='nearest')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); records = []
            for i, image in enumerate((reference, moving)):
                path = root/f'{i}.nx'; self.make_file(path, image[None], exposure=1.)
                records.append(dict(path=str(path), shape=[64, 64], exposure_s=1.,
                    frames=1, point=i, scan=263, current_A=float(i), energy_ev=710., label=f's{i}'))
            for enabled in (False, True):
                output = root/f'cache_{enabled}.h5'
                with contextlib.redirect_stdout(io.StringIO()):
                    config = x.prepare_cache(records, output, dark_paths=[], center=(32, 32),
                        crop=0, binning=1, support_circles=[(32, 32, 5)],
                        beamstop_radius=2, polygons=[], align_holograms=enabled,
                        alignment_max_shift=4., alignment_upsample_factor=100)
                self.assertEqual(config['alignment']['enabled'], enabled)
                with h5py.File(output) as h:
                    shifts = h['alignment_shifts_raw_pixels'][()]
                    np.testing.assert_array_equal(shifts[0], 0.)
                    if enabled:
                        np.testing.assert_allclose(shifts[1], -np.array(delta), atol=.1)
                        self.assertTrue(h['mask_pixel'][1, -1, :].all())
                        self.assertTrue(h['mask_pixel'][1, :, 0].all())
                        scores = h['alignment_correlation'][1]
                        self.assertGreater(scores[1], scores[0])
                    else:
                        np.testing.assert_array_equal(shifts, 0.)
                        np.testing.assert_allclose(h['holograms'][1], moving, rtol=1e-6)
                        self.assertTrue(np.isnan(h['alignment_correlation'][()]).all())

    def make_file(self,path,frames,exposure=.1):
        with h5py.File(path,'w') as f:
            f.create_dataset(x.DATASET,data=frames)
            f.create_dataset('entry/instrument/detector/exposure',data=[exposure])

    def test_dark_optional_weighted_and_exposure_checked(self):
        dark,info=x.load_dark([], (4,4), .1)
        self.assertIsNone(dark);self.assertFalse(info['applied'])
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.nx',Path(tmp)/'b.nx'
            self.make_file(a,np.ones((1,4,4))*10)
            self.make_file(b,np.ones((3,4,4))*30)
            dark,info=x.load_dark([a,b],(4,4),.1)
            np.testing.assert_allclose(dark,25)
            self.assertEqual(info['frame_count'],4)
            with self.assertRaises(ValueError):x.load_dark([a],(4,4),.2)
            with self.assertRaises(FileNotFoundError):x.load_dark([Path(tmp)/'missing.nx'],(4,4),.1)

    def test_saturation_any_frame_and_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.nx';data=np.ones((3,4,4))*5;data[1,1,1]=100
            self.make_file(p,data)
            mean,invalid,count=x.frame_mean(p,saturation=100)
            self.assertEqual(count,3);self.assertTrue(invalid[1,1]);self.assertEqual(invalid.sum(),1)
            np.testing.assert_allclose(mean,data.mean(axis=0))

    def test_endpoint_scale_and_global_phase_gauge(self):
        material=np.zeros((8,8),bool);material[2:6,2:6]=True
        refs=np.zeros((8,8),bool);refs[0,0]=True
        true=np.array([1,.7,0,-.3,-1])
        fields=np.zeros((len(true),8,8),complex)
        fields[:,refs]=1
        for i,m in enumerate(true):fields[i,material]=np.exp(.4+.2j+(.12+.3j)*m)
        fields*=np.exp(1j*np.array([.2,.8,-.5,.1,.4]))[:,None,None]
        estimate,valid,residual=x.endpoint_magnetization(fields,0,4,material,refs)
        np.testing.assert_allclose(estimate[:,valid],np.broadcast_to(true[:,None],estimate[:,valid].shape),atol=1e-12)
        self.assertLess(np.max(residual[:,valid]),1e-12)

    def test_manifest_preserves_points_and_duplicate_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'2601_xpcs_00263/marana';folder.mkdir(parents=True)
            with h5py.File(root/'2601_xpcs_00263.nxs','w') as f:
                g=f.create_group('scan/instrument/collection');g['point_nb']=[0,1,2];g['m_caena']=[.5,0,.5];g['mono']=[710.5]
            for i in [2,0,1]:self.make_file(folder/f'2601_xpcs_00263_{i:06d}.nx',np.ones((2,4,4)))
            rows=x.manifest(root,[263])
            self.assertEqual([r['point'] for r in rows],[0,1,2]);self.assertEqual([r['current_A'] for r in rows],[.5,0,.5])
            (folder/'2601_xpcs_00263_000001.nx').unlink()
            with self.assertRaises(ValueError):x.manifest(root,[263])

if __name__=='__main__':unittest.main()
