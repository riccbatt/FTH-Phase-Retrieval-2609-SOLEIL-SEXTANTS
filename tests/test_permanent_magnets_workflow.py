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
            for index, stacks in [(0, [np.full((1, 1, 6), 10.)]),
                                  (1, [np.array([[[11., 20., np.nan, 65535., np.nan, 65535.]]]),
                                       np.array([[[13., 16., 19., 18., np.nan, 65535.]], [[19., 18., 19., 18., np.nan, 65535.]]])])]:
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
                np.testing.assert_allclose(h["averages"][0], [[3., 6., 9., 8., np.nan, np.nan]], equal_nan=True)
                np.testing.assert_array_equal(h["frame_counts"], [3])
                self.assertNotIn("invalid", h)
                self.assertNotIn("dark_invalid", h)
                np.testing.assert_array_equal(h["valid_counts"][0], [[3, 3, 2, 2, 0, 0]])
                self.assertTrue(np.isnan(h["averages"][0, 0, 4:]).all())
            namespace["INVALID_PIXEL_POLICY"] = "any_frame"
            exec(compile("".join(notebook["cells"][4]["source"]), "strict-averaging", "exec"), namespace)
            with h5py.File(temp/"average.h5") as h:
                np.testing.assert_allclose(h["averages"][0, 0, :2], [3., 6.])
                self.assertTrue(np.isnan(h["averages"][0, 0, 2:]).all())

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
                        frames = np.repeat(image[None], n, axis=0)
                        if index == 1:
                            frames[:, 3, 3] = np.nan  # No usable exposure at this pixel.
                        h["entry/instrument/detector/data"] = frames
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
                dict(SUPPORT_PNG=temp/"support.png", MODES=[1, 2], WARMUP_ITERATIONS=[2, 1], DATA_MODE="single_polarization", XMCD_PAIRS=[]),
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
                expected = pattern.copy(); expected[3, 3] = np.nan
                np.testing.assert_allclose(h["averages"][0], expected, atol=1e-5, equal_nan=True)
            with h5py.File(temp/"02_holograms.h5") as h:
                np.testing.assert_allclose(h["alignment_shifts"][1], [-1.3, .7], atol=.3)
                self.assertGreaterEqual(h["alignment_correlation"][1, 1], h["alignment_correlation"][1, 0])
                self.assertTrue(h["mask_pixel"][1].any())
                self.assertTrue(h["mask_pixel"][0, 3, 3])
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
            # Reuse the same scans as an opposite-polarization pair, one state.
            with h5py.File(temp/"02_holograms.h5", "a") as h:
                h["polarization"][...] = [1., -1.]
            xmcd_configs = [dict(configs[2], DATA_MODE="xmcd", XMCD_PAIRS=[[0, 1]]),
                            dict(configs[3], SATURATED_SCAN_INDICES=[], SATURATION_SIGNS=[], OUTER_ITERATIONS=1)]
            with patch("matplotlib.pyplot.show", side_effect=lambda: plt.close("all")):
                for notebook, config in zip(notebooks[2:], xmcd_configs):
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
                            namespace["INPUT"] = temp / namespace["INPUT"].name
                    if notebook.name.startswith("03"):
                        widget_cell = json.loads(notebook.read_text())["cells"][-1]
                        widget_source = "".join(widget_cell["source"])
                        widget_source = widget_source.replace("ENABLE_FOCUS_SLIDERS = False", "ENABLE_FOCUS_SLIDERS = True")
                        widget_source = widget_source.replace("DETECTOR_PIXEL_SIZE_M = None", "DETECTOR_PIXEL_SIZE_M = 13.5e-6")
                        widget_source = widget_source.replace("DETECTOR_DISTANCE_M = None", "DETECTOR_DISTANCE_M = .2")
                        with patch("IPython.display.display"):
                            exec(compile(widget_source, "focus-slider-preview", "exec"), namespace)
                            namespace["propagation_slider"].value = 1.
                            namespace["phase_slider"].value = .2
                            namespace["save_focus"](None)
                        with h5py.File(temp/"03_warmup.h5", "a") as h:
                            h.attrs["focus_prop_um"] = 1.
                            h.attrs["focus_phase_rad"] = .2
                            h.attrs["focus_pixel_size_m"] = 13.5e-6
                            h.attrs["focus_detector_distance_m"] = .2
            with h5py.File(temp/"04_physical.h5") as h:
                self.assertEqual(h["magnetization"].shape, (1, *shape))
                np.testing.assert_array_equal(h["image_indices"], [0, 1])
                np.testing.assert_allclose(h["mean_magnetization_by_image"], np.repeat(h["mean_magnetization"][()], 2))
                self.assertEqual(h["magnetization_state_labels"].asstr()[()].tolist(), ["state_0"])
                self.assertEqual(h.attrs["focus_prop_um"], 1.)
                self.assertEqual(h["state_labels"].asstr()[()].tolist(), ["state_0", "state_0"])

    def test_unsaturated_helicity_pairs_recover_relative_magnetization(self):
        from library import phase_retrieval_universal as u
        # Every physical state is unsaturated; each has measured + and - helicity.
        true_means = np.array([-.6, .2, .7])
        thickness = np.zeros((6, 6)); thickness[2:4, 2:4] = 1.
        states = np.repeat(["a", "b", "c"], 2).tolist()
        polarizations = np.tile([1., -1.], 3)
        magnetic_response = .08 + .03j
        logs = (.1+.02j) + thickness*(.2+.01j) + (
            np.repeat(true_means, 2)*polarizations)[:, None, None]*thickness*magnetic_response
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe.update(clip_magnetization=False, initial_binary_magnetization=False,
                      physical_projection_object_roi=True)
        projected, components = u.project_log_objects_physical(
            logs, states, [781.]*6, polarizations, ["beam0"]*6, recipe=recipe,
            material_thickness=thickness, saturated_states={}, iterations=100,
            return_components=True)
        np.testing.assert_allclose(projected, logs, atol=1e-8)
        fitted_means = components["magnetization"][:, thickness>0].mean(axis=1)
        scale = np.dot(fitted_means, true_means)/np.dot(true_means, true_means)
        self.assertGreater(abs(scale), 1e-6)
        np.testing.assert_allclose(fitted_means, scale*true_means, atol=1e-8)
        self.assertFalse(components["magnetic_scale_anchored"])
        self.assertEqual(components["saturated_states"], {})
        self.assertTrue(np.isfinite(components["magnetization"]).all())

    def test_difference_selection_and_focus(self):
        pairs = workflow.difference_pairs("xmcd", [[0, 1], [2, 3]], 0, [1, -1, 1, -1], [781]*4)
        np.testing.assert_array_equal(pairs, [[0, 1], [2, 3]])
        reference_pairs = workflow.difference_pairs("single_polarization", [], 1, [1]*3, [781]*3)
        np.testing.assert_array_equal(reference_pairs, [[0, 1], [1, 1], [2, 1]])
        objects = np.array([1+2j, 3+4j, 5+6j, 7+8j])[:, None, None, None]
        np.testing.assert_allclose(workflow.object_differences(objects, pairs), -2-2j)
        fields = np.ones((2, 8, 8), dtype=complex)
        baseline = workflow.focused_objects(fields, [781]*2)
        rotated = workflow.focused_objects(fields, [781]*2, phase_rad=.3)
        np.testing.assert_allclose(rotated, baseline*np.exp(.3j))
        with self.assertRaisesRegex(ValueError, "positive, negative"):
            workflow.difference_pairs("xmcd", [[0, 1]], 0, [1, 1], [781]*2)


if __name__ == "__main__":
    unittest.main()
