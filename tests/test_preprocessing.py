import tempfile
import unittest
import json
from pathlib import Path

import h5py
import numpy as np

from library.beamstop_stitching import stitch_exposures
from library.data_loading import Frame, NexusLoader, SextantsNexusLoader
from library.mask_store import MaskStore, load_red_mask_png
from library.scan_workflow import load_scan_channel, save_diode_scans


class PreprocessingTests(unittest.TestCase):
    def test_diode_scan_discovers_channels_and_saves_hdf5_and_png(self):
        with tempfile.TemporaryDirectory() as folder:
            raw_path = Path(folder) / "scanx_0007.nxs"
            with h5py.File(raw_path, "w") as handle:
                energy = handle.create_dataset("scan_0007/scan_data/data_19", data=[700, 701])
                energy.attrs["long_name"] = "monochromator/energy"
                diode = handle.create_dataset("scan_0007/scan_data/data_14", data=[2, 4])
                diode.attrs["long_name"] = "detectors/diode"
            x = load_scan_channel(raw_path, "energy")
            y = load_scan_channel(raw_path, "diode")
            h5_path, png_path = save_diode_scans(
                Path(folder) / "processed", [7], [x], [y],
                x_channel="energy", y_channel="diode", user="test",
            )
            self.assertTrue(h5_path.is_file())
            self.assertTrue(png_path.is_file())
            with h5py.File(h5_path, "r") as handle:
                np.testing.assert_array_equal(handle["scan_7/x"], [700, 701])
                np.testing.assert_array_equal(handle["scan_7/diode"], [2, 4])
                self.assertEqual(handle.attrs["x_channel"], "energy")

    def test_diode_field_trace_is_flattened_in_acquisition_order(self):
        with tempfile.TemporaryDirectory() as folder:
            field = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
            diode = np.arange(6).reshape(2, 3)
            h5_path, _ = save_diode_scans(
                folder, [12], [field], [diode],
                x_channel="mtesla", y_channel="diode",
            )
            with h5py.File(h5_path, "r") as handle:
                np.testing.assert_array_equal(
                    handle["scan_12/x"], [-1.0, 0.0, 1.0, 1.0, 0.0, -1.0]
                )
                np.testing.assert_array_equal(
                    handle["scan_12/diode"], np.arange(6)
                )

    def test_final_result_cells_save_png_and_hdf5_together(self):
        root = Path(__file__).parents[1]
        for notebook in ("01_FTH.ipynb", "04_phase_retrieval.ipynb"):
            with (root / notebook).open(encoding="utf-8") as handle:
                cells = json.load(handle)["cells"]
            final_save_cells = [
                "".join(cell.get("source", []))
                for cell in cells
                if "wf.save_data_dict(" in "".join(cell.get("source", []))
            ]
            self.assertEqual(len(final_save_cells), 1, notebook)
            self.assertIn("fig.savefig(png_name", final_save_cells[0], notebook)

    def test_napari_preview_does_not_overwrite_mask_png(self):
        root = Path(__file__).parents[1]
        with (root / "00a_define_mask_pixel_napari.ipynb").open(
            encoding="utf-8"
        ) as handle:
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in json.load(handle)["cells"]
            )
        self.assertIn("saved_path.stem + '_preview.png'", source)
        self.assertNotIn("preview_path = saved_path.with_suffix('.png')", source)

    def test_support_notebook_offers_all_three_mask_methods(self):
        root = Path(__file__).parents[1]
        with (root / "03_define_supportmask.ipynb").open(encoding="utf-8") as handle:
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in json.load(handle)["cells"]
            )
        self.assertIn("Paint support mask with PNG", source)
        self.assertIn("Napari support mask", source)
        self.assertIn("Support coordinates / circle widget", source)

    def test_long_exposure_is_filled_and_scaled(self):
        frames = [
            Frame("short", np.full((2, 2), 10.0), 1.0, Path("short")),
            Frame("long", np.full((2, 2), 100.0), 10.0, Path("long")),
        ]
        masks = [np.zeros((2, 2), bool), np.array([[1, 0], [0, 0]], bool)]
        result = stitch_exposures(frames, masks)
        np.testing.assert_allclose(result.image, np.full((2, 2), 100.0))
        self.assertEqual(result.ordered_ids, ("long", "short"))
        self.assertEqual(result.source_count[0, 0], 1)

    def test_equal_exposures_are_averaged_where_both_are_valid(self):
        frames = [
            Frame("a", np.full((4, 4), 10.0), 2.0, Path("a")),
            Frame("b", np.full((4, 4), 14.0), 2.0, Path("b")),
        ]
        result = stitch_exposures(frames, [np.zeros((4, 4)), np.zeros((4, 4))])
        np.testing.assert_allclose(result.image, 12.0)
        np.testing.assert_array_equal(result.source_count, 2)

    def test_stitched_mask_is_intersection_of_input_masks(self):
        frames = [
            Frame("a", np.ones((2, 3)), 2.0, Path("a")),
            Frame("b", np.ones((2, 3)), 1.0, Path("b")),
        ]
        masks = [
            np.array([[1, 1, 0], [0, 0, 0]], dtype=bool),
            np.array([[1, 0, 1], [0, 0, 0]], dtype=bool),
        ]
        result = stitch_exposures(frames, masks)
        expected = np.logical_and(masks[0], masks[1])
        np.testing.assert_array_equal(result.missing_mask, expected)
        self.assertTrue(np.isnan(result.image[0, 0]))
        self.assertTrue(np.isfinite(result.image[0, 1]))

    def test_linear_fit_maps_short_exposure_to_long_reference(self):
        base = np.arange(100.0).reshape(10, 10) + 5
        frames = [
            Frame("long", base * 10, 10.0, Path("long")),
            Frame("short", (base - 3) / 2, 1.0, Path("short")),
        ]
        long_mask = np.zeros((10, 10), bool)
        long_mask[:2] = True
        result = stitch_exposures(
            frames, [long_mask, np.zeros((10, 10))], fit_degree=1,
            fit_percentiles=(0, 100),
        )
        np.testing.assert_allclose(result.image, base * 10, rtol=1e-10, atol=1e-10)

    def test_mask_store_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MaskStore(folder)
            store.save(42, np.array([[0, 2]], dtype=float))
            np.testing.assert_array_equal(store.load(42, (1, 2)), [[0, 1]])
            self.assertEqual(store.path_for(42).name, "mask_pixel_42.png")

    def test_red_png_ignores_non_red_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            from PIL import Image

            path = Path(folder) / "painted.png"
            rgb = np.array(
                [[[255, 0, 0], [255, 255, 255], [180, 0, 0]]], dtype=np.uint8
            )
            Image.fromarray(rgb).save(path)
            np.testing.assert_array_equal(
                load_red_mask_png(path, (1, 3)), [[1, 0, 0]]
            )

    def test_nexus_loader_accepts_top_level_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "00007.nxs"
            with h5py.File(path, "w") as handle:
                handle["entry/scan_data/data_22"] = np.ones((2, 3))
                handle["entry/scan_data/integration_times"] = 4.0
            loader = NexusLoader(folder, filename="{id:05d}.nxs")
            frame = loader.load(7)
            self.assertEqual(frame.image.shape, (2, 3))
            self.assertEqual(frame.exposure, 4.0)

    def test_sextants_loader_discovers_numbered_detector_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scanx_0024.nxs"
            with h5py.File(path, "w") as handle:
                entry = handle.create_group("scan_0024")
                scan_data = entry.create_group("scan_data")
                scan_data["data_22"] = 17.0  # Numbered keys are not stable.
                detector = scan_data.create_dataset("data_21", data=np.ones((1, 8, 9)))
                detector.attrs["interpretation"] = "image"
                detector.attrs["long_name"] = "dhyana95/image"
                scan_data["integration_times"] = [20.0]
                entry["SEXTANTS/mono/energy"] = [702.8]
                entry["SEXTANTS/hu80.2_energy/polarisation"] = [3]
            frame = SextantsNexusLoader(folder).load(24)
            self.assertEqual(frame.image.shape, (8, 9))
            self.assertEqual(frame.exposure, 20.0)
            self.assertEqual(frame.metadata["energy_eV"], 702.8)
            self.assertEqual(frame.metadata["polarization_code"], 3)
            self.assertTrue(frame.metadata["image_path"].endswith("data_21"))


if __name__ == "__main__":
    unittest.main()
