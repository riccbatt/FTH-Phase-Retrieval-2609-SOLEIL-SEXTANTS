import copy
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from aperiodic_ajajas.workflow_profiles import load_profile, save_profile, save_support, select_numbers
from aperiodic_ajajas.reconstruction_inputs import point_mean, DETECTOR_DATASET


class ProfileTests(unittest.TestCase):
    def test_profile_roundtrip_and_cache_identity(self):
        original = load_profile(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as folder:
            profile = save_profile(folder, 'sample_B', original.data, activate=True)
            self.assertEqual(load_profile(folder).data, original.data)
            config = dict(saturated_scan=2923, loop_scan=2924,
                          profile_name='sample_B', profile=copy.deepcopy(original.data))
            profile.validate_cache(config)
            profile.data['preprocess']['LOOP_FRAME_INDICES'] = [2]
            with self.assertRaises(ValueError):
                profile.validate_cache(config)
            with self.assertRaises(FileExistsError):
                save_profile(folder, 'sample_B', original.data)
            profile.snapshot()
            self.assertEqual(json.loads((profile.output_dir / 'profile.json').read_text()), profile.data)
            mask = np.array([[0, 1, 0], [1, 0, 1]])
            saved = save_support(profile, mask, 'painted')
            np.testing.assert_array_equal(np.asarray(Image.open(saved))[..., 0] == 255, mask)
            with self.assertRaises(FileExistsError):
                save_support(profile, mask, 'painted')

    def test_acquisition_numbers(self):
        self.assertEqual(select_numbers([0, 5, 10], [10, 5]), [2, 1])
        self.assertEqual(select_numbers([5, 10, 15], exclude=[0]), [0, 1, 2])
        for invalid in ([9], [], [5, 5]):
            with self.assertRaises(ValueError):
                select_numbers([0, 5, 10], invalid)

    def test_selected_frames_threshold_before_average(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'frames.h5'
            frames = np.array([[[1., 4.]], [[10., 20.]], [[5., 2.]]])
            with h5py.File(path, 'w') as handle:
                handle[DETECTOR_DATASET] = frames
            actual = point_mean(path, frame_indices=[2, 0], dark=1, photon_level=2)
            np.testing.assert_allclose(actual, [[2., 1.5]])
            for invalid in ([], [0, 0], [3], [0.5]):
                with self.assertRaises(ValueError):
                    point_mean(path, frame_indices=invalid)
            with h5py.File(path, 'w') as handle:
                handle[DETECTOR_DATASET] = frames[0]
            with self.assertRaises(ValueError):
                point_mean(path, frame_indices=[1])
            np.testing.assert_allclose(point_mean(path, frame_indices=[0]), frames[0])


if __name__ == '__main__':
    unittest.main()
