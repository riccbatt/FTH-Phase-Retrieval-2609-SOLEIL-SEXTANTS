import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from simulation_hysteresis import reconstruction_inputs as inputs
from library import phase_retrieval_universal as u


class HelicityTests(unittest.TestCase):
    def test_loader_keeps_shared_states_and_opposite_coefficients(self):
        with tempfile.TemporaryDirectory() as directory:
            for i in range(3):
                with h5py.File(Path(directory)/f'{i:02}.h5', 'w') as f:
                    f.attrs['state_name'] = f'state{i}'
                    f.attrs['energy_eV'] = 781.
                    f.attrs['object_mean_mz'] = [1., .25, -1.][i]
                    f.attrs['configuration_json'] = json.dumps({'detector': {
                        'sample_to_detector_distance': .1, 'pixel_size': 2e-5}})
                    f['support_mask'] = np.ones((2, 2), bool)
                    f['masks/beamstop_mask'] = np.zeros((2, 2))
                    for h, offset in [('CR', 0), ('CL', 10)]:
                        f[f'holograms/ideal/{h}'] = np.full((2, 2), i+offset)
            data = inputs.load_dataset(directory, 'ideal', ('CR', 'CL'), [0, 2])
            self.assertEqual(data['state_names'], ['state0', 'state2'])
            self.assertEqual(data['state_labels'], ['state0', 'state0', 'state2', 'state2'])
            np.testing.assert_array_equal(data['polarization_coefficients'], [1, -1, 1, -1])
            np.testing.assert_array_equal(data['holograms'][:, 0, 0], [0, 10, 2, 12])
            np.testing.assert_array_equal(data['truth_mean_mz'], [1, -1])
            self.assertEqual(data['saturated'], {'state0': 1., 'state2': -1.})
            cl = inputs.load_dataset(directory, 'ideal', 'CL', [0, 2])
            np.testing.assert_array_equal(cl['polarization_coefficients'], [-1, -1])

    def test_single_helicity_preserves_magnetization_sign(self):
        truth = np.array([1., .3, -1.])
        for sign in (1., -1.):
            logs = (.4+.1j) + sign*truth[:, None, None]*np.ones((1, 2, 1))*(.1+.04j)
            recipe = u.default_universal_phase_retrieval_recipe()
            recipe.update(physical_spatial_weighting='exit_wave_intensity_per_frame')
            _, c = u.project_log_objects_physical(
                logs, ['p', 'free', 'n'], [781.]*3, [sign]*3, ['beam']*3,
                recipe=recipe, material_thickness=np.ones((2, 1)),
                saturated_states={'p': 1., 'n': -1.}, iterations=100,
                return_components=True)
            np.testing.assert_allclose(c['magnetization'][:, 0, 0], truth, atol=1e-8)

    def test_joint_fit_recovers_one_map_per_state(self):
        states = ['p', 'p', 'free', 'free', 'n', 'n']
        polarizations = np.tile([1., -1.], 3)
        truth = np.array([1., .3, -1.])
        common = np.array([[.2+.1j], [.5-.1j]])
        logs = common[None] + polarizations[:, None, None]*np.repeat(truth, 2)[:, None, None]*(.1+.04j)
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe.update(physical_spatial_weighting='exit_wave_intensity_per_frame',
                      physical_projection_object_roi=True)
        projected, c = u.project_log_objects_physical(
            logs, states, [781.]*6, polarizations, ['beam']*6,
            recipe=recipe, material_thickness=np.ones((2, 1)),
            saturated_states={'p': 1., 'n': -1.}, iterations=100,
            return_components=True)
        self.assertEqual(c['state_names'], ['p', 'free', 'n'])
        np.testing.assert_allclose(c['magnetization'][:, 0, 0], truth, atol=1e-8)
        np.testing.assert_allclose(projected, logs, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
