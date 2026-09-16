import unittest

import numpy as np

from library.phase_retrieval_geometry import prepare


class JointGeometryTests(unittest.TestCase):
    def test_binning_preserves_binary_masks_and_all_mode_supports(self):
        images = np.ones((2, 12, 12), dtype=float)
        mask = np.zeros_like(images, dtype=np.uint8)
        mask[1, 2, 2] = 1
        supports = np.zeros((2, 12, 12), dtype=np.uint8)
        supports[0, 4:6, 4:6] = 1
        supports[1, 6:8, 6:8] = 1
        recipe = {"binning": 2, "crop": 1, "roi": [3, 9, 3, 9], "Nmodes": 2}
        binned, used_mask, used_support, _, geometry = prepare(
            images, mask, supports, recipe
        )
        self.assertEqual(binned.shape, (2, 6, 6))
        np.testing.assert_array_equal(binned, 4)
        self.assertEqual(used_mask[1, 1, 1], 1)
        self.assertEqual(used_support.shape, (2, 6, 6))
        self.assertNotEqual(used_support[0].tolist(), used_support[1].tolist())
        fields = np.ones((2, 2, 6, 6), dtype=complex)
        output = geometry.finish(fields, fields, {}, used_mask, {})
        result, _, components, result_mask, errors = output
        self.assertEqual(result.shape, (2, 2, 4, 4))
        np.testing.assert_array_equal(components["supportmask_used"], used_support)
        self.assertEqual(components["supportmask"].shape, (2, 4, 4))
        self.assertEqual(components["mask_pixel"].shape, (2, 4, 4))
        self.assertEqual(result_mask.shape, (2, 4, 4))
        np.testing.assert_array_equal(components["roi"], [0, 4, 0, 4])
        self.assertEqual(errors["geometry"]["binning"], 2)

    def test_crop_keeps_exact_retrieval_support_and_shifts_roi(self):
        image = np.ones((2, 10, 10))
        support = np.zeros((10, 10), dtype=np.uint8)
        support[3:7, 3:7] = 1
        _, mask, used, _, geometry = prepare(
            image, np.zeros((10, 10)), support,
            {"binning": 1, "crop": 2, "roi": [3, 7, 3, 7]},
        )
        _, _, components, _, _ = geometry.finish(image, image, {}, mask, {})
        np.testing.assert_array_equal(components["supportmask_used"], used)
        self.assertEqual(components["supportmask"].shape, (6, 6))
        np.testing.assert_array_equal(components["roi"], [2, 4, 2, 4])


if __name__ == "__main__":
    unittest.main()
