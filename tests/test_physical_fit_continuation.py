import unittest
import numpy as np
from library import phase_retrieval_universal as u


class PhysicalFitContinuationTests(unittest.TestCase):
    def test_measured_initialization_and_previous_fit_preserve_domain_free_signal(self):
        names = ['p', 'a', 'b', 'c', 'n']
        signs = np.array([1., 1., 1., 1., -1.])
        thickness = np.array([[0., 1.], [1., 0.]])
        logs = .3 + signs[:, None, None] * thickness * (.1 + .03j)
        for roi in (False, True):
            recipe = u.default_universal_phase_retrieval_recipe()
            recipe.update(physical_projection_object_roi=roi, clip_magnetization=False)
            def fit(iterations, initial=None):
                return u.project_log_objects_physical(
                    logs, names, [781.]*5, [1.]*5, ['beam']*5,
                    recipe=recipe, material_thickness=thickness,
                    saturated_states={'p': 1., 'n': -1.}, iterations=iterations,
                    initial_components=initial, return_components=True)
            _, previous = fit(100)
            saved = {key: previous[key].copy() for key in
                     ('magnetization', 'common_log_objects', 'charge_response', 'magnetic_response')}
            _, cold = fit(1)
            projected, continued = fit(1, previous)
            self.assertEqual(cold['magnetization_initialization'], 'measured_contrast')
            np.testing.assert_allclose(cold['magnetization'][:, thickness > 0],
                                       np.broadcast_to(signs[:, None], (5, 2)), atol=1e-8)
            self.assertTrue(continued['physical_fit_reused'])
            np.testing.assert_allclose(continued['magnetization'][:, thickness > 0],
                                       np.broadcast_to(signs[:, None], (5, 2)), atol=1e-8)
            np.testing.assert_allclose(projected, logs, atol=1e-8)
            for key, value in saved.items():
                np.testing.assert_array_equal(previous[key], value)
            bad = dict(previous, state_names=list(reversed(names)))
            with self.assertRaisesRegex(ValueError, 'state_names'):
                fit(1, bad)

    def test_nearest_sign_initialization_is_optional_and_not_a_binary_constraint(self):
        from unittest.mock import patch
        thickness = np.array([[0., 1.], [1., 0.]])
        signs = np.array([1., .02, -.03, 0., -1.])
        logs = .3 + signs[:, None, None] * thickness * (.1 + .03j)
        names = ['p', 'a', 'b', 'zero', 'n']
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe.update(initial_binary_magnetization=True, physical_projection_object_roi=True)
        def fit(initial=None):
            return u.project_log_objects_physical(
                logs, names, [781.]*5, [1.]*5, ['beam']*5,
                recipe=recipe, material_thickness=thickness,
                saturated_states={'p': 1., 'n': -1.}, iterations=100,
                initial_components=initial, return_components=True)
        with patch.object(u, '_snap_magnetization', wraps=u._snap_magnetization) as snap:
            _, previous = fit()
            self.assertEqual(snap.call_count, 1)
            guess, values = snap.call_args.args
            np.testing.assert_array_equal(values, [-1., 1.])
            snapped = snap._mock_wraps(guess, values)
            np.testing.assert_array_equal(snapped[:3, :, 0],
                                          np.broadcast_to(np.array([1., 1., -1.])[:, None], (3, 2)))
            self.assertEqual(previous['magnetization_initialization'], 'nearest_sign')
            np.testing.assert_array_equal(previous['initial_magnetization'][:3, thickness > 0],
                                          np.broadcast_to(np.array([1., 1., -1.])[:, None], (3, 2)))
            np.testing.assert_allclose(previous['magnetization'][:, thickness > 0],
                                       np.broadcast_to(signs[:, None], (5, 2)), atol=1e-8)
            snap.reset_mock()
            _, continued = fit(previous)
            snap.assert_not_called()
            self.assertEqual(continued['magnetization_initialization'], 'previous_fit')
            np.testing.assert_array_equal(continued['magnetization'][:, thickness == 0], 0)
        np.testing.assert_array_equal(u._snap_magnetization(np.array([0.]), np.array([-1., 1.])), [-1.])
        recipe['initial_binary_magnetization'] = False
        _, continuous = fit()
        self.assertEqual(continuous['magnetization_initialization'], 'measured_contrast')
        recipe['initial_binary_magnetization'] = 'yes'
        with self.assertRaisesRegex(ValueError, 'initial_binary_magnetization must be bool'):
            fit()

    def test_reuse_through_illumination_and_multimode_projection(self):
        thickness = np.array([[0., 1.], [1., 0.]])
        signs = np.array([1., .7, -1.])
        logs = .3 + signs[:, None, None] * thickness * (.1 + .03j)
        primary = u.object_log_to_fourier_field(logs)
        gains = np.array([1., 1.5, .8])
        fields = np.stack([primary, np.ones_like(primary)*.02], axis=1)
        fields *= np.sqrt(gains)[:, None, None, None]
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe.update(fit_illumination_intensity=True, physical_projection_object_roi=True,
                      constrain_nonphysical_modes_common=True)
        def fit(iterations, initial=None):
            return u._project_physical_modes(
                fields, ['p', 'free', 'n'], [781.]*3, [1.]*3, ['beam']*3,
                recipe=recipe, material_thickness=thickness,
                projection_supportmask=np.ones((2, 2)),
                saturated_states={'p': 1., 'n': -1.}, physical_iterations=iterations,
                initial_components=initial, return_components=True)
        _, previous = fit(100)
        output, continued = fit(1, previous)
        self.assertTrue(continued['physical_fit_reused'])
        np.testing.assert_allclose(output, fields, atol=1e-8)
        np.testing.assert_allclose(continued['illumination_intensity_factors'], gains)

    def test_warmup_diagnostic_plots_initializer_after_fields(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from library import phase_retrieval_geometry as geometry
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/'2601_XPCS/01_physical_hysteresis_263.ipynb').read_text())
        display_thickness = np.zeros((8, 8))
        display_thickness[2:6, 2:6] = 1.
        support = display_thickness.copy()
        support[0, 0] = 1.
        signs = np.array([1., .02, -1.])
        logs = .3 + signs[:, None, None]*np.fft.fftshift(display_thickness)*(.1+.03j)
        fields = u.object_log_to_fourier_field(logs)
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe.update(crop=0, binning=1, material_thickness=display_thickness,
                      zero_magnetization_outside_support=True,
                      projection_constraints_inside_support_only=True,
                      initial_binary_magnetization=True, physical_projection_object_roi=True)
        with tempfile.TemporaryDirectory() as folder:
            ns = dict(np=np, plt=plt, universal=u, geometry=geometry,
                warmup=fields, holograms=abs(fields)**2, mask_pixel=np.zeros((3,8,8)),
                support=support, warmup_recipe=recipe, recipe=recipe,
                warmup_bsmasks=np.zeros((3,8,8)),
                warmup_components={'modal_supportmask_used': support},
                ENERGY_EV=781., FIGURES=Path(folder), current=np.array([1.,0.,-1.]),
                indices=[0,1,2], state_labels=['p','free','n'], saturated={'p':1.,'n':-1.},
                calibrated=True, reference_observation=0, FIT_ILLUMINATION_INTENSITY=True,
                INITIAL_BINARY_MAGNETIZATION=True, BINARY_MAGNETIZATION=False,
                CLIP_MAGNETIZATION=False, USE_EXIT_WAVE_INTENSITY_WEIGHTS=True)
            filenames = []
            from matplotlib.figure import Figure
            savefig = Figure.savefig
            def record(fig, filename, **kwargs):
                filenames.append(Path(filename).name)
                return savefig(fig, filename, **kwargs)
            with patch.object(Figure, 'savefig', record), patch.object(plt, 'show'), contextlib.redirect_stdout(io.StringIO()):
                exec(next(''.join(c['source']) for c in notebook['cells']
                          if 'def plot_warmup_diagnostics' in ''.join(c.get('source', []))), ns)
            self.assertTrue((Path(folder)/'warmup_initial_magnetization.png').is_file())
            self.assertEqual(filenames[4], 'warmup_initial_magnetization.png')
            np.testing.assert_array_equal(ns['warmup'], fields)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
