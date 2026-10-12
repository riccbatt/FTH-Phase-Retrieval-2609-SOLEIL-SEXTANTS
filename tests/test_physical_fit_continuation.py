import unittest
import numpy as np
from library import phase_retrieval_universal as u


class PhysicalFitContinuationTests(unittest.TestCase):
    def test_previous_fit_prevents_one_pass_zero_bias(self):
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
            self.assertLess(cold['magnetization'][1, 0, 1], .5)
            self.assertTrue(continued['physical_fit_reused'])
            np.testing.assert_allclose(continued['magnetization'][:, thickness > 0],
                                       np.broadcast_to(signs[:, None], (5, 2)), atol=1e-8)
            np.testing.assert_allclose(projected, logs, atol=1e-8)
            for key, value in saved.items():
                np.testing.assert_array_equal(previous[key], value)
            bad = dict(previous, state_names=list(reversed(names)))
            with self.assertRaisesRegex(ValueError, 'state_names'):
                fit(1, bad)

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


if __name__ == '__main__':
    unittest.main()
