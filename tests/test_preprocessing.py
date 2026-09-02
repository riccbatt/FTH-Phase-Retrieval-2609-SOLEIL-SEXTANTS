import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from library.beamstop_stitching import stitch_exposures
from library.data_loading import Frame, NexusLoader, SextantsNexusLoader
from library.mask_store import MaskStore


class PreprocessingTests(unittest.TestCase):
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
