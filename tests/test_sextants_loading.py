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
