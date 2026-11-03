import unittest
import pytest
import numpy as np
from library import phase_retrieval_universal as u
from library import phase_retrieval_validation as validation
from library import phase_retrieval_physical as physical


class PhysicalFitContinuationTests(unittest.TestCase):
    def test_normalize_mode_factors_rejects_bad_values_but_accepts_arrays(self):
        self.assertEqual(validation.normalize_mode_factors([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])
        self.assertEqual(validation.normalize_mode_factors(np.array([1.0, 2.0])), [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, 'support factor 1'):
            validation.normalize_mode_factors([2.0, 1.0])
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            validation.normalize_mode_factors([-1.0, 1.0])

    def test_measured_initialization_and_previous_fit_preserve_domain_free_signal(self):
        names = ['p', 'a', 'b', 'c', 'n']
        signs = np.array([1., 1., 1., 1., -1.])
        thickness = np.array([[0., 1.], [1., 0.]])
        logs = .3 + signs[:, None, None] * thickness * (.1 + .03j)
        for roi in (False, True):
            recipe = u.default_universal_phase_retrieval_recipe()
            recipe.update(physical_projection_object_roi=roi, clip_magnetization=False)
            def fit(iterations, initial=None):
                return physical.project_log_objects_physical(
                    logs, names, [781.]*5, [1.]*5, ['beam']*5,
                    recipe=recipe, material_thickness=thickness,
                    saturated_states={'p': 1., 'n': -1.}, iterations=iterations,
                    initial_components=initial)
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
            return physical.project_log_objects_physical(
                logs, names, [781.]*5, [1.]*5, ['beam']*5,
                recipe=recipe, material_thickness=thickness,
                saturated_states={'p': 1., 'n': -1.}, iterations=100,
                initial_components=initial)
        with patch.object(physical, '_snap_magnetization', wraps=physical._snap_magnetization) as snap:
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
        np.testing.assert_array_equal(physical._snap_magnetization(np.array([0.]), np.array([-1., 1.])), [-1.])
        recipe['initial_binary_magnetization'] = False
        _, continuous = fit()
        self.assertEqual(continuous['magnetization_initialization'], 'measured_contrast')

    def test_reuse_through_illumination_and_multimode_projection(self):
        thickness = np.array([[0., 1.], [1., 0.]])
        signs = np.array([1., .7, -1.])
        logs = .3 + signs[:, None, None] * thickness * (.1 + .03j)
        primary = physical.object_log_to_fourier_field(logs)
        gains = np.array([1., 1.5, .8])
        fields = np.stack([primary, np.ones_like(primary)*.02], axis=1)
        fields *= np.sqrt(gains)[:, None, None, None]
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe.update(modes=[1., 1.], fit_illumination_intensity=True, physical_projection_object_roi=True,
                      constrain_nonphysical_modes_common=True)
        def fit(iterations, initial=None):
            return physical._project_physical_modes(
                fields, ['p', 'free', 'n'], [781.]*3, [1.]*3, ['beam']*3,
                recipe=recipe, material_thickness=thickness,
                projection_supportmask=np.ones((2, 2)),
                saturated_states={'p': 1., 'n': -1.}, physical_it=iterations,
                initial_components=initial)
        _, previous = fit(100)
        output, continued = fit(1, previous)
        self.assertTrue(continued['physical_fit_reused'])
        np.testing.assert_allclose(output, fields, atol=1e-8)
        np.testing.assert_allclose(continued['illumination_intensity_factors'], gains)



# Exact synthetic objects expose changes to mode sharing, coordinate frames or gain restoration.

@pytest.mark.parametrize('shared_magnetization', [False, True])
@pytest.mark.parametrize('propagation', [0., 1.37])
def test_grouped_physical_fit_preserves_exact_model(shared_magnetization, propagation):
    thickness = np.zeros((8, 8)); thickness[2:6, 2:6] = 1
    signs = np.array([[1., .4, -1.], [1., .4 if shared_magnetization else -.2, -1.]])
    logs = .3 + thickness[None, None] * signs.T[:, :, None, None] * (.1 + .03j)
    recipe = u.default_universal_phase_retrieval_recipe()
    recipe.update(modes=[1, 1],
                  magnetization_mode_groups=[[0, 1]] if shared_magnetization else [[0], [1]],
                  illumination_mode_groups=[[0, 1]], projection_focus_prop_um=propagation,
                  projection_focus_setup=dict(ccd_dist=.4, px_size=13.5e-6),
                  projection_focus_phase_rad=.12)
    fields = physical._projection_focus_transform(
        physical.object_log_to_fourier_field(logs), [781.]*3, recipe, inverse=True)
    saved = fields.copy()

    def fit(previous=None, iterations=50):
        return physical._project_physical_modes(
            fields, ['p', 'free', 'n'], [781.]*3, [1.]*3, ['beam']*3,
            recipe=recipe, material_thickness=thickness, projection_supportmask=np.ones((8, 8)),
            saturated_states={'p': 1., 'n': -1.}, physical_it=iterations,
            initial_components=previous)

    result, components = fit()
    np.testing.assert_allclose(result, fields, atol=1e-8)
    np.testing.assert_array_equal(fields, saved)
    np.testing.assert_allclose(components['magnetization'][1, thickness > 0], .4, atol=1e-8)
    if not shared_magnetization:
        np.testing.assert_allclose(components['magnetization_by_mode_group_packed']['1'][1], -.2, atol=1e-8)
    continued, _ = fit(components, iterations=1)
    np.testing.assert_allclose(continued, fields, atol=1e-8)


def test_grouped_illumination_restores_mode_and_observation_phases():
    thickness = np.zeros((8, 8)); thickness[2:6, 2:6] = 1
    signs = np.array([1., .4, -1.])
    logs = .3 + thickness[None] * signs[:, None, None] * (.1 + .03j)
    primary = physical.object_log_to_fourier_field(logs)
    fields = np.stack([primary, primary * np.exp(.4j)], axis=1)
    gains = np.array([1., 1.5, .8])
    fields *= np.sqrt(gains)[:, None, None, None] * np.exp(1j*np.array([0., .2, -.1]))[:, None, None, None]
    recipe = u.default_universal_phase_retrieval_recipe()
    recipe.update(modes=[1, 1], magnetization_mode_groups=[[0, 1]], illumination_mode_groups=[[0, 1]],
                  fit_illumination_intensity=True, align_global_phase=True)
    result, components = physical._project_physical_modes(
        fields, ['p', 'free', 'n'], [781.]*3, [1.]*3, ['beam']*3,
        recipe=recipe, material_thickness=thickness, projection_supportmask=np.ones((8, 8)),
        saturated_states={'p': 1., 'n': -1.}, physical_it=50)
    np.testing.assert_allclose(result, fields, atol=1e-8)
    np.testing.assert_allclose(components['illumination_intensity_factors'], gains, atol=1e-8)


@pytest.mark.parametrize('integer_wavelength', [False, True])
def test_focus_round_trip_and_on_axis_phase(integer_wavelength):
    rng = np.random.default_rng(19)
    fields = rng.normal(size=(3, 2, 8, 8)) + 1j*rng.normal(size=(3, 2, 8, 8))
    saved = fields.copy()
    energies = np.array([700., 710., 720.])
    recipe = u.default_universal_phase_retrieval_recipe()
    recipe.update(projection_focus_prop_um=1.37, projection_focus_phase_rad=.17,
                  projection_focus_integer_wavelength=integer_wavelength,
                  projection_focus_setup=dict(ccd_dist=.4, px_size=13.5e-6))
    focused = physical._projection_focus_transform(fields, energies, recipe)
    restored = physical._projection_focus_transform(focused, energies, recipe, inverse=True)
    np.testing.assert_allclose(restored, fields, atol=1e-12)
    np.testing.assert_array_equal(fields, saved)
    wavelength = 1.2398419843320026e-6 / energies
    distance = np.round(1.37e-6 / wavelength)*wavelength if integer_wavelength else 1.37e-6
    on_axis = np.exp(1j*(2*np.pi*distance/wavelength + .17))
    np.testing.assert_allclose(focused[:, :, 4, 4], fields[:, :, 4, 4]*on_axis[:, None], atol=1e-11)


@pytest.mark.parametrize('weighting', [None, 'exit_wave_intensity', 'exit_wave_intensity_per_frame'])
def test_joint_energy_and_beam_fit_preserves_exact_objects(weighting):
    thickness = np.ones((4, 4)); thickness[0, 0] = 0
    charge = np.array([-.1+.04j, .1-.04j])
    magnetic = np.array([.1+.03j, .2+.02j])
    common = np.stack([np.full((4, 4), .3+.02j), np.full((4, 4), .5-.01j)])
    records = [(state, energy, beam) for state in range(2) for energy in range(2) for beam in range(2)]
    signs = [1., -1.]
    logs = np.array([common[b] + thickness*(charge[e] + signs[s]*magnetic[e]) for s, e, b in records])
    recipe = u.default_universal_phase_retrieval_recipe()
    recipe.update(physical_spatial_weighting=weighting)
    result, components = physical.project_log_objects_physical(
        logs, [s for s, e, b in records], [e for s, e, b in records], [1.]*len(records),
        [b for s, e, b in records], recipe=recipe, material_thickness=thickness,
        saturated_states={0: 1., 1: -1.}, iterations=100)
    np.testing.assert_allclose(result, logs, atol=1e-8)
    np.testing.assert_allclose(components['charge_response'], charge, atol=1e-8)
    np.testing.assert_allclose(components['magnetic_response'], magnetic, atol=1e-8)
