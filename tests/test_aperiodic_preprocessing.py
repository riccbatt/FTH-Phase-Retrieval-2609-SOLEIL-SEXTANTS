"""Regression checks for the aperiodic cache and painted support geometry."""

import tempfile
import json
import unittest
from pathlib import Path

import h5py
import numpy as np

from aperiodic_ajajas.reconstruction_inputs import (
    adapt_retrieval_array_shapes, center_crop_or_pad, crop_array_edges,
    correct_cached_polygon_mask, load_support, load_reference_polygon_mask,
    currents_for_detector_files, fit_reference_intensity, prepare_point,
    point_mean, scan_metadata,
    support_field_coverage,
)


class AperiodicPreprocessingTests(unittest.TestCase):
    def test_filtered_hologram_shape_adapts_support_and_detector_mask(self):
        holograms = np.ones((3, 8, 8))
        detector_mask = np.zeros((3, 12, 12), dtype=np.uint8)
        detector_mask[:, 3:9, 3:9] = 1
        support = np.zeros((12, 12), dtype=np.uint8)
        support[3:9, 3:9] = 1

        images, mask, used_support, changes = adapt_retrieval_array_shapes(
            holograms, detector_mask, support,
        )

        self.assertEqual(images.shape, (3, 8, 8))
        self.assertEqual(mask.shape, (3, 8, 8))
        self.assertEqual(used_support.shape, (8, 8))
        np.testing.assert_array_equal(used_support, support[2:10, 2:10])
        self.assertEqual(len(changes), 2)

    def test_object_filter_crop_uses_last_two_axes(self):
        stack = np.arange(2 * 10 * 12).reshape(2, 10, 12)
        np.testing.assert_array_equal(
            crop_array_edges(stack, 2), stack[:, 2:-2, 2:-2],
        )
        padded = center_crop_or_pad(stack[:, 2:-2, 2:-2], (10, 12))
        np.testing.assert_array_equal(padded[:, 2:-2, 2:-2], stack[:, 2:-2, 2:-2])

    def test_photon_level_is_applied_after_dark_subtraction_per_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "point.nx"
            with h5py.File(path, "w") as h:
                h.create_dataset(
                    "/entry/instrument/detector/data",
                    data=np.array([[[1, 10]], [[5, 6]]], dtype=np.float32),
                )
            result = point_mean(path, dark=np.array([[2, 2]]), photon_level=0)
            np.testing.assert_allclose(result, [[1.5, 6]])

    def test_painted_reference_support_survives_default_geometry(self):
        support_png = Path(__file__).parents[1] / "aperiodic_ajajas/supportmask_00_2.png"
        support = load_support(support_png, 4, (2048, 2048))
        self.assertEqual(support_field_coverage(support, 400, 1), 1.0)
        self.assertLess(support_field_coverage(support, 200, 4), 0.02)

    def test_crop_precedes_binning_and_moves_masks_together(self):
        image = np.arange(144, dtype=np.float32).reshape(12, 12)
        image[2, 2] = 1000
        support = np.ones((12, 12), dtype=np.uint8)
        extra_mask = np.zeros((12, 12), dtype=np.uint8)
        extra_mask[9, 9] = 1
        prepared, excluded, used_support, _ = prepare_point(
            image, np.zeros_like(image), (6, 6), support,
            crop=2, binning=2, beamstop_radius=0,
            saturation=500, extra_mask=extra_mask,
        )
        self.assertEqual(prepared.shape, (4, 4))
        self.assertEqual(used_support.shape, (4, 4))
        np.testing.assert_allclose(prepared[0, 0], image[2:4, 2:4].sum())
        self.assertEqual(excluded[0, 0], 1)  # Saturated raw pixel.
        self.assertEqual(excluded[-1, -1], 1)  # Additional detector mask.

    def test_current_metadata_uses_collection_channel(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scan.nxs"
            with h5py.File(path, "w") as h:
                h.create_dataset("/scan/instrument/collection/mono", data=[783.0])
                h.create_dataset("/scan/instrument/collection/m_caena", data=[1., 0.])
            energy, current = scan_metadata(path)
            np.testing.assert_array_equal(energy, [783.])
            np.testing.assert_array_equal(current, [1., 0.])

    def test_sparse_detector_files_use_their_scan_point_numbers(self):
        files = [f"scan_02924_{index:06d}.nx" for index in (0, 10, 20, 100)]
        indices, currents = currents_for_detector_files(
            np.linspace(1, -1, 101), files,
        )
        np.testing.assert_array_equal(indices, [0, 10, 20, 100])
        np.testing.assert_allclose(currents, [1, 0.8, 0.6, -1])

    def test_reference_intensity_fit_ignores_masked_pixels(self):
        reference = np.arange(1, 17, dtype=float).reshape(4, 4)
        image = 1.25 * reference + 3
        image[0, 0] = 10000
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 1
        factor, offset = fit_reference_intensity(image, reference, mask)
        self.assertAlmostEqual(factor, 1.25)
        self.assertAlmostEqual(offset, 3)

    def test_reference_polygons_are_centered_and_cropped_with_image(self):
        with tempfile.TemporaryDirectory() as folder:
            polygon_path = Path(folder) / "polygons.json"
            polygon_path.write_text(json.dumps({"wire": [[[4, 4], [4, 5],
                                                            [5, 5], [5, 4]]]}))
            polygon_mask = load_reference_polygon_mask(
                polygon_path, (12, 12), dilation=0,
            )
            image = np.ones((12, 12), dtype=np.float32)
            _, excluded, _, _ = prepare_point(
                image, np.zeros_like(image), (5, 5),
                np.ones_like(image), crop=2, binning=1,
                beamstop_radius=0, saturation=100,
                centered_mask=polygon_mask,
            )
            self.assertEqual(excluded[2, 2], 1)

    def test_cached_polygon_adjustment_preserves_other_masked_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            polygon_path = Path(folder) / "polygons.json"
            polygon_path.write_text(json.dumps({"beamstop": [
                [[2, 2], [2, 5], [5, 5], [5, 2]],
            ]}))
            old = load_reference_polygon_mask(polygon_path, (12, 12), dilation=0)
            old[10, 10] = True  # Independent bad detector pixel.
            config = {"reference_polygons": str(polygon_path),
                      "crop_raw_pixels": 0, "binning": 1,
                      "reference_mask_dilation_pixels": 0,
                      "reference_polygon_coordinate_order": "xy"}
            adjusted = correct_cached_polygon_mask(
                old, config, shift_pixels=(2, 1), erosion=0, dilation=0,
            )
            self.assertEqual(adjusted[10, 10], 1)
            self.assertEqual(adjusted[2, 2], 0)
            self.assertEqual(adjusted[7, 6], 1)

    def test_copy2_polygon_vertices_are_xy_and_legacy_cache_is_corrected(self):
        with tempfile.TemporaryDirectory() as folder:
            polygon_path = Path(folder) / "polygons.json"
            polygon_path.write_text(json.dumps({"wire": [
                [[2, 5], [5, 5], [5, 7], [2, 7]],
            ]}))
            legacy = load_reference_polygon_mask(
                polygon_path, (12, 12), dilation=0, coordinate_order="yx",
            )
            config = {"reference_polygons": str(polygon_path),
                      "crop_raw_pixels": 0, "binning": 1,
                      "reference_mask_dilation_pixels": 0}
            fixed = correct_cached_polygon_mask(
                legacy, config, shift_pixels=(0, 0), erosion=0, dilation=0,
            )
            self.assertEqual(fixed[6, 3], 1)
            self.assertEqual(fixed[3, 6], 0)


if __name__ == "__main__":
    unittest.main()
