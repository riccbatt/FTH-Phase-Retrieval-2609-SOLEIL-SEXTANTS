import unittest
import numpy as np
from library import phase_retrieval_universal as u


class IlluminationIntensityTests(unittest.TestCase):
    def setUp(self):
        self.thickness = np.array([[0., 1.], [1., 0.]])
        self.signs = np.array([1., .3, -1.])
        self.gains = np.array([1., 1.8, .7])
        logs = (.1 + .04j) + self.signs[:, None, None] * self.thickness * (.07 + .03j)
        self.fields = u.object_log_to_fourier_field(logs)
        self.recipe = u.default_universal_phase_retrieval_recipe()
        self.recipe.update(fit_illumination_intensity=True, physical_projection_object_roi=True)
        self.kwargs = dict(recipe=self.recipe, material_thickness=self.thickness,
                           projection_supportmask=np.ones((2, 2)),
                           saturated_states={'p': 1., 'n': -1.},
                           relaxation=1., return_components=True)

    def project(self, fields):
        return u._project_physical_modes(fields, ['p', 'free', 'n'], [781.] * 3,
                                         [1.] * 3, ['beam'] * 3, **self.kwargs)

    def test_gain_and_magnetization_recovery(self):
        observed = self.fields * np.sqrt(self.gains)[:, None, None]
        result, components = self.project(observed)
        np.testing.assert_allclose(components['illumination_intensity_factors'], self.gains)
        np.testing.assert_allclose(components['magnetization'][:, self.thickness > 0],
                                   np.broadcast_to(self.signs[:, None], (3, 2)), atol=1e-8)
        np.testing.assert_allclose(result, observed, atol=1e-8)

    def test_common_secondary_modes_are_scaled_by_incident_flux(self):
        self.recipe['constrain_nonphysical_modes_common'] = True
        secondary = np.ones_like(self.fields) * (.02 + .01j)
        modes = np.stack([self.fields, secondary], axis=1)
        observed = modes * np.sqrt(self.gains)[:, None, None, None]
        output, components = self.project(observed)
        np.testing.assert_allclose(output[:, 1] / np.sqrt(self.gains)[:, None, None], secondary)
        np.testing.assert_allclose(components['illumination_intensity_factors'], self.gains)

    def test_disabled_preserves_existing_projection(self):
        self.recipe['fit_illumination_intensity'] = False
        output, components = self.project(self.fields)
        self.assertNotIn('illumination_intensity_factors', components)
        np.testing.assert_allclose(output, self.fields, atol=1e-8)

    def test_universal_driver_preserves_flux_scaled_common_modes(self):
        import contextlib
        import io
        self.recipe.update(modes=[1, 1], crop=0, binning=1,
                           material_thickness=np.fft.ifftshift(self.thickness),
                           projection_constraints_inside_support_only=True,
                           constrain_nonphysical_modes_common=True,
                           warmup_Nit=0, outer_iterations=0,
                           final_projection_relaxation=1., final_fourier_constraint=False,
                           saturated_states={'p': 1., 'n': -1.})
        modes = np.stack([self.fields, np.ones_like(self.fields)*(.02+.01j)], axis=1)
        observed = modes*np.sqrt(self.gains)[:, None, None, None]
        with contextlib.redirect_stdout(io.StringIO()):
            output, _, components, _, _ = u.universal_phase_retrieval_algorithm(
                np.sum(abs(observed)**2, axis=1), np.zeros_like(self.fields.real),
                np.ones((2, 2)), ['p', 'free', 'n'], [781.]*3, [1.]*3, ['beam']*3,
                universal_recipe=self.recipe, start_fields=observed)
        np.testing.assert_allclose(components['illumination_intensity_factors'], self.gains)
        np.testing.assert_allclose(output, observed, atol=1e-8)

    def test_missing_reference_rejected(self):
        self.kwargs['material_thickness'] = np.ones((2, 2))
        with self.assertRaisesRegex(ValueError, 'material-free'):
            self.project(self.fields)

    def test_nonzero_reference_observation(self):
        self.recipe['illumination_intensity_reference_observation'] = 1
        _, components = self.project(self.fields * np.sqrt(self.gains)[:, None, None])
        np.testing.assert_allclose(components['illumination_intensity_factors'], self.gains / self.gains[1])


if __name__ == '__main__':
    unittest.main()
