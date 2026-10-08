"""Optional saturation calibration, relative contrast, and notebook exports."""
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

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('xpcs_optional_inputs', ROOT/'2601_XPCS/reconstruction_inputs.py')
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)


class OptionalSaturationTests(unittest.TestCase):
    def test_relative_changes_preserve_scale_and_phase_alignment(self):
        material = np.zeros((8, 8), bool)
        material[2:6, 2:6] = True
        refs = np.zeros((8, 8), bool)
        refs[0, 0] = True
        values = np.array([.3, .1, -.2, .15])  # Neither endpoint is saturated.
        response = .12+.3j
        fields = np.ones((4, 8, 8), complex)
        fields[:, material] = np.exp(.4+.2j+response*values[:, None])
        fields *= np.exp(1j*np.array([.2, -.4, .6, .1]))[:, None, None]
        for reference in (0, 2):
            contrast, valid, residual = inputs.relative_magnetic_contrast(fields, material, refs, reference)
            expected = abs(response)*(values-values[reference])
            np.testing.assert_allclose(contrast[:, valid], np.broadcast_to(expected[:, None], contrast[:, valid].shape), atol=1e-12)
            self.assertLess(np.max(residual[:, valid]), 1e-12)
            self.assertLess(np.max(abs(contrast[:, valid])), 1.)
        with self.assertRaisesRegex(ValueError, 'reference-hole'):
            inputs.relative_magnetic_contrast(fields, material, np.zeros_like(refs))

    def test_notebook_retrieves_plots_and_exports_with_optional_anchors(self):
        notebook = json.loads((ROOT/'2601_XPCS/01_physical_hysteresis_263.ipynb').read_text())
        cases = [({}, None), ({(263, 0): 1.}, None), ({(263, 2): -1.}, None),
                 ({(263, 0): 1., (263, 2): -1.}, None),
                 ({(263, 0): 1., (263, 2): -1.}, [dict(name='magnetic', magnetic=True, charge_response=[.2+.1j], magnetic_response=[.1+.03j]),
                                               dict(name='nonmagnetic', charge_response=[.05+.02j])])]
        for anchors, species in cases:
            with self.subTest(anchors=anchors, species=species), tempfile.TemporaryDirectory() as folder:
                cache = Path(folder)/'cache.h5'
                support = np.zeros((16, 16)); support[5:11, 5:11] = 1; support[2, 2] = 1
                material = support.copy(); material[2, 2] = 0
                with h5py.File(cache, 'w') as h:
                    h.attrs['config_json'] = json.dumps({'binning': 1, 'dark': {'applied': False}})
                    h['state_labels'] = np.asarray(['a', 'b', 'c'], dtype=h5py.string_dtype())
                    for key, value in dict(scan=[263]*3, point=[0, 1, 2],
                        holograms=np.stack([np.full((16, 16), v) for v in (1., 1.2, .8)]),
                        mask_pixel=np.zeros((3, 16, 16), bool), supportmask=support,
                        material_mask=material, current_A=[1., 0., -1.], energy_ev=[710.]*3).items():
                        h[key] = value
                ns = {}
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(''.join(notebook['cells'][1]['source']), ns)
                    ns.update(CACHE=cache, OUTPUT=Path(folder)/'result.h5', FIGURES=Path(folder),
                        POINT_INDICES=[0, 1, 2], SATURATED_POINTS=anchors, MODES=[1], CHEMICAL_SPECIES=species,
                        WARMUP_ITERATIONS=[2, 1], OUTER_ITERATIONS=1, INNER_ITERATIONS=1, PHYSICAL_ITERATIONS=2)
                    exec(''.join(notebook['cells'][2]['source']), ns)
                    # Retain the prepared physical aperture on the synthetic grid.
                    ns['material'] = material
                    source = ''.join(notebook['cells'][3]['source']).replace('crop=200', 'crop=0')
                    exec(source, ns)
                    for i in (6, 9, 10, 11):
                        exec(''.join(notebook['cells'][i]['source']), ns)
                calibrated = len(anchors) == 2
                self.assertEqual(ns['recipe']['clip_magnetization'], calibrated)
                self.assertTrue(np.isfinite(ns['fields']).all())
                with h5py.File(ns['OUTPUT']) as h:
                    self.assertEqual(h.attrs['calibration_status'], 'two_saturated_endpoints' if calibrated else 'relative_contrast')
                    self.assertEqual(h['relative_contrast'].shape, (3, 16, 16))
                    if species is not None:
                        self.assertIn('physical_components/species/magnetic/density_map', h)
                        self.assertIn('physical_components/species/nonmagnetic/charge_response', h)
                        self.assertEqual(h.attrs['hysteresis_diagnostic_species'], 'magnetic')
                        self.assertTrue((Path(folder)/'species_0_density_response.png').is_file())
                        self.assertTrue((Path(folder)/'species_0_magnetization.png').is_file())
                        self.assertTrue((Path(folder)/'species_1_density_response.png').is_file())
                    self.assertTrue(h['final_contrast_valid'][()].any())
                    self.assertTrue(np.isfinite(h['final_contrast'][()][:, h['final_contrast_valid'][()]]).all())
                    if not calibrated:
                        self.assertTrue(np.isnan(h['final_endpoint_m'][()]).all())
                        self.assertFalse(h['final_endpoint_valid'][()].any())
                        self.assertIn('a.u.', h.attrs['contrast_units'])
                        np.testing.assert_allclose(h['relative_contrast'][()][ns['reference_observation']], 0.)
                plt.close('all')


if __name__ == '__main__':
    unittest.main()
