import unittest

import numpy as np

from library.phase_retrieval_geometry import prepare


class JointGeometryTests(unittest.TestCase):
    def test_detector_crop_rescales_object_support_and_roi(self):
        support = np.zeros((16, 16), dtype=np.uint8)
        support[4, 11] = 1
        _, _, used, _, geometry = prepare(
            np.zeros((1, 16, 16)), np.zeros((16, 16)), support,
            {"binning": 1, "crop": 2, "roi": [4, 5, 11, 12]},
        )
        self.assertEqual(used.shape, (12, 12))
        np.testing.assert_array_equal(np.argwhere(used), [[3, 8]])
        np.testing.assert_array_equal(geometry.roi, [3, 4, 8, 9])

    def test_crop_precedes_binning_for_data_mask_start_and_roi(self):
        image = np.arange(100, dtype=float).reshape(10, 10)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, 0] = 1  # Removed by the crop.
        mask[1, 1] = 1  # Excludes the first retained bin.
        support = np.arange(100).reshape(10, 10)
        start = np.arange(100).reshape(10, 10).astype(complex)
        data, used_mask, used_support, used_start, geometry = prepare(
            image[None], mask, support,
            {"binning": 2, "crop": 1, "roi": [4, 6, 4, 6]}, start,
        )
        expected_blocks = image[1:9, 1:9].reshape(4, 2, 4, 2)
        np.testing.assert_array_equal(data[0], expected_blocks.sum(axis=(1, 3)))
        np.testing.assert_array_equal(used_start, expected_blocks.mean(axis=(1, 3)))
        np.testing.assert_array_equal(used_support, (support[3:7, 3:7] != 0))
        self.assertEqual(used_mask[0, 0], 1)
        self.assertEqual(int(used_mask.sum()), 1)
        np.testing.assert_array_equal(geometry.roi, [1, 3, 1, 3])

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
        self.assertEqual(binned.shape, (2, 5, 5))
        np.testing.assert_array_equal(binned, 4)
        self.assertEqual(used_mask[1, 0, 0], 1)
        self.assertEqual(used_support.shape, (2, 5, 5))
        self.assertNotEqual(used_support[0].tolist(), used_support[1].tolist())
        fields = np.ones((2, 2, 5, 5), dtype=complex)
        output = geometry.finish(fields, fields, {}, used_mask, {})
        result, _, components, result_mask, errors = output
        self.assertEqual(result.shape, (2, 2, 5, 5))
        np.testing.assert_array_equal(components["supportmask_used"], used_support)
        self.assertEqual(components["supportmask"].shape, (2, 5, 5))
        self.assertEqual(components["mask_pixel"].shape, (2, 5, 5))
        self.assertEqual(result_mask.shape, (2, 5, 5))
        np.testing.assert_array_equal(components["roi"], [0, 5, 0, 5])
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
