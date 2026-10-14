"""Execute the actual four notebooks against small synthetic CMOS scans."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import shift, gaussian_filter
import matplotlib.pyplot as plt

from permanent_magnets import workflow

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "permanent_magnets"


class PermanentMagnetsWorkflowTests(unittest.TestCase):
    def test_inline_filtering_and_frame_weighting(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for index, stacks in [(0, [np.full((1, 1, 4), 10.)]),
                                  (1, [np.array([[[11., 20., np.nan, 65535.]]]),
                                       np.array([[[13., 16., 19., 18.]], [[19., 18., 19., 18.]]])])]:
                directory = temp / f"test_{index:05d}" / "marana"
                directory.mkdir(parents=True)
                for j, frames in enumerate(stacks):
                    with h5py.File(directory/f"point_{j}.nx", "w") as h:
                        h["entry/instrument/detector/data"] = frames
            notebook = json.loads((FOLDER/"01_average_raw_cmos.ipynb").read_text())
            namespace = {"__name__": "__main__"}
            for cell in notebook["cells"][:5]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                exec(compile(source, "inline-averaging", "exec"), namespace)
                if "OUTPUT = PROCESSED" in source:
                    namespace.update(RAW_ROOT=temp, PREFIX="test", SCAN_INDICES=[1], DARK_INDICES=[0],
                                     ENERGIES_EV=[781.], POLARIZATIONS=[1.], OUTPUT=temp/"average.h5")
            with h5py.File(temp/"average.h5") as h:
                np.testing.assert_allclose(h["averages"][0], [[3., 6., 6., 16/3]])
                np.testing.assert_array_equal(h["frame_counts"], [3])
                self.assertNotIn("invalid", h)
                self.assertNotIn("dark_invalid", h)
                self.assertTrue(np.isfinite(h["averages"][()]).all())

    def test_four_notebook_handoffs_and_physical_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            shape = (80, 80)
            rng = np.random.default_rng(42)
            pattern = 20 + 30*gaussian_filter(rng.random(shape), .7)
            dark = np.full(shape, 100.)
            for index, image in [(0, dark), (1, dark+pattern), (2, dark+shift(pattern, (1.3, -.7), order=1, mode="nearest"))]:
                directory = temp / f"test_{index:05d}" / "marana"
                directory.mkdir(parents=True)
                # Different file frame counts verify aggregation across a whole index.
                for j, n in enumerate([1, 2]):
                    with h5py.File(directory/f"point_{j:05d}.nx", "w") as h:
                        h["entry/instrument/detector/data"] = np.repeat(image[None], n, axis=0)
            rows, cols = np.indices(shape)
            support = (rows-40)**2+(cols-40)**2 <= 5**2
            for row, col in [(28, 30), (28, 50), (52, 40)]:
                support |= (rows-row)**2+(cols-col)**2 <= 2**2
            exclusion = np.zeros(shape, bool); exclusion[0, :] = True
            for name, mask in [("support.png", support), ("mask.png", exclusion)]:
                rgb = np.zeros((*shape, 3), dtype=np.uint8); rgb[mask, 0] = 255
                Image.fromarray(rgb).save(temp/name)
            configs = [
                dict(RAW_ROOT=temp, PREFIX="test", SCAN_INDICES=[1, 2], DARK_INDICES=[0], ENERGIES_EV=[781., 781.], POLARIZATIONS=[1., 1.]),
                dict(MASK_PNG=temp/"mask.png", MAX_SHIFT=3., DETECTOR_CENTER=None, ALIGNMENT_ROI=(5, 75, 5, 75)),
                dict(SUPPORT_PNG=temp/"support.png", MODES=[1, 2], WARMUP_ITERATIONS=[2, 1]),
                dict(SATURATED_SCAN_INDICES=[1, 2], SATURATION_SIGNS=[1., -1.], OUTER_ITERATIONS=1, INNER_ITERATIONS=1, PHYSICAL_ITERATIONS=2),
            ]
            notebooks = sorted(FOLDER.glob("0*.ipynb"))
            self.assertEqual(len(notebooks), 4)
            with patch("matplotlib.pyplot.show", side_effect=lambda: plt.close("all")):
                for notebook, config in zip(notebooks, configs):
                    namespace = {"__name__": "__main__"}
                    for cell in json.loads(notebook.read_text())["cells"]:
                        if cell["cell_type"] != "code":
                            continue
                        source = "".join(cell["source"])
                        exec(compile(source, str(notebook), "exec"), namespace)
                        if "OUTPUT = PROCESSED" in source:
                            namespace.update(config)
                            namespace["PROCESSED"] = temp
                            namespace["OUTPUT"] = temp / namespace["OUTPUT"].name
                            if "INPUT" in namespace:
                                namespace["INPUT"] = temp / namespace["INPUT"].name
            with h5py.File(temp/"01_averages.h5") as h:
                np.testing.assert_array_equal(h["frame_counts"], [3, 3])
                np.testing.assert_allclose(h["averages"][0], pattern, atol=1e-5)
            with h5py.File(temp/"02_holograms.h5") as h:
                np.testing.assert_allclose(h["alignment_shifts"][1], [-1.3, .7], atol=.3)
                self.assertGreaterEqual(h["alignment_correlation"][1, 1], h["alignment_correlation"][1, 0])
                self.assertTrue(h["mask_pixel"][1].any())
                np.testing.assert_array_equal(h["diagnostic_indices"], [0, 1])
                self.assertTrue(np.isfinite(h["diagnostic_fth_difference"][()]).all())
                from library.fthcore import reconstruct
                expected = reconstruct(np.where(h["diagnostic_difference_mask"][()], 0, h["holograms"][0] - h["holograms"][1]))
                np.testing.assert_allclose(h["diagnostic_fth_difference"][()], expected)
            with h5py.File(temp/"03_warmup.h5") as warm, h5py.File(temp/"04_physical.h5") as physical:
                np.testing.assert_array_equal(physical["warmup_fields"], warm["warmup_fields"])
                self.assertEqual(physical["fields"].shape, warm["warmup_fields"].shape)
                self.assertEqual(physical["magnetization"].shape, (2, *shape))
                self.assertIn("fit_residual_rms", physical["components"])
                self.assertTrue(np.all(np.isfinite(physical["fields"][()])))
            # Rerun only notebook 4 in projection-only mode, still from disk.
            namespace["OUTER_ITERATIONS"] = 0
            namespace["SATURATED_SCAN_INDICES"] = []
            namespace["SATURATION_SIGNS"] = []
            cells = json.loads(notebooks[-1].read_text())["cells"]
            stage = next(c for c in cells if c["cell_type"] == "code" and "received_start" in "".join(c["source"]))
            exec(compile("".join(stage["source"]), "projection-only", "exec"), namespace)
            self.assertIn("magnetization", namespace["components"])
            self.assertFalse(namespace["calibrated"])


if __name__ == "__main__":
    unittest.main()
