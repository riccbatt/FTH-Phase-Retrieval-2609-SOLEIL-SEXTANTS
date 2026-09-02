import tempfile
import unittest
import json
from pathlib import Path

import h5py
import numpy as np

from library.data_loading import SextantsNexusLoader


class SextantsLoadingTests(unittest.TestCase):
    def test_main_notebooks_keep_sextants_loader_and_metadata_geometry(self):
        root = Path(__file__).parents[1]
        geometry_notebooks = (
            "FTH_CDI.ipynb",
            "FTH_CDI_Basic.ipynb",
            "FTH_CDI_Stitching.ipynb",
            "FTH_CDI_Topo.ipynb",
            "SAXS_Basic.ipynb",
        )
        for name in geometry_notebooks:
            with (root / name).open(encoding="utf-8") as handle:
                source = "".join(
                    line
                    for cell in json.load(handle)["cells"]
                    for line in cell.get("source", [])
                )
            self.assertIn("SextantsNexusLoader", source, name)
            self.assertIn("COMET_20260902_Cocoons_Laser_raw", source, name)
            self.assertIn("sextants_loader.experimental_setup(", source, name)

    def test_napari_mask_notebook_supports_per_image_dark_subtraction(self):
        root = Path(__file__).parents[1]
        with (root / "00a_define_mask_pixel_napari.ipynb").open(encoding="utf-8") as handle:
            source = "".join(
                line
                for cell in json.load(handle)["cells"]
                for line in cell.get("source", [])
            )
        self.assertIn("DARK_IDS =", source)
        self.assertIn("image_rate = np.asarray(frame.image, dtype=float) / image_exposure", source)
        self.assertIn("dark_frame = loader.load(dark_id)", source)
        self.assertIn("dark_rate = np.asarray(dark_frame.image, dtype=float) / dark_exposure", source)
        self.assertIn("corrected = image_rate - dark_rate", source)
        self.assertIn("corrected_images.append(corrected)", source)
        self.assertIn("name='exposure-normalized, dark-corrected images'", source)
        self.assertIn("np.nanpercentile(image_stack, (0.1, 99.9))", source)
        self.assertIn("image_stack = np.stack(corrected_images)", source)
        self.assertNotIn("name='raw images'", source)

    def test_notebook_image_and_scan_ids_use_uppercase_configuration_variables(self):
        root = Path(__file__).parents[1]
        expected = {
            "00a_define_mask_pixel_napari.ipynb": ("IMAGE_IDS =", "DARK_IDS ="),
            "FTH_CDI.ipynb": ("IMAGE_ID =", "REFERENCE_ID ="),
            "FTH_CDI_Basic.ipynb": ("IMAGE_ID =", "REFERENCE_ID =", "IMAGE_DARK_ID ="),
            "FTH_CDI_Stitching.ipynb": ("IMAGE_IDS =", "REFERENCE_IDS =", "IMAGE_DARK_IDS ="),
            "FTH_CDI_Topo.ipynb": ("IMAGE_ID =", "DARK_ID ="),
            "SAXS_Basic.ipynb": ("IMAGE_IDS =", "DARK_IDS ="),
            "Diode_Scans.ipynb": ("SCAN_IDS =",),
            "Reconstruction_Scheme.ipynb": ("IMAGE_IDS =",),
        }
        for name, variables in expected.items():
            source = (root / name).read_text(encoding="utf-8")
            for variable in variables:
                self.assertIn(variable, source, name)

    def test_image_channels_and_geometry_are_read_from_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "scanx_0095.nxs"
            with h5py.File(source, "w") as handle:
                scan = handle.create_group("scan_0095/scan_data")
                image = scan.create_dataset("data_22", data=np.ones((2, 8, 9)))
                image.attrs["interpretation"] = "image"
                image.attrs["long_name"] = "camera/image"
                scan.create_dataset("integration_times", data=[0.1, 0.1])
                distance = scan.create_dataset("data_03", data=250.0)
                distance.attrs["long_name"] = "ccd-ts/position"
                handle.create_dataset("scan_0095/SEXTANTS/mono/energy", data=780.0)
                field = scan.create_dataset("data_14", data=[1.0, 2.0])
                field.attrs["long_name"] = "field/mtesla"

            loader = SextantsNexusLoader(folder)
            frame = loader.load(95)
            self.assertEqual(frame.image.shape, (8, 9))
            self.assertAlmostEqual(frame.metadata["ccd_dist_m"], 0.45)
            np.testing.assert_array_equal(loader.load_channel(95, "mtesla"), [1, 2])
            setup = loader.experimental_setup(95)
            self.assertAlmostEqual(setup["ccd_dist"], 0.45)
            self.assertEqual(setup["ccd_dist_source"], "scan_0095/scan_data/data_03")
            self.assertEqual(setup["energy"], 780.0)
            self.assertEqual(setup["px_size"], 11e-6)


if __name__ == "__main__":
    unittest.main()
