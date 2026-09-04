import tempfile
import unittest
import json
from pathlib import Path

import h5py
import numpy as np

from library.beamstop_stitching import shift_mask, stitch_exposures, stitch_images
from library.data_loading import (
    Frame,
    NexusLoader,
    SextantsNexusLoader,
    load_average,
    load_processing,
)
from library.mask_store import MaskStore, load_red_mask_png
from library.image_preprocessing import (
    fit_dark_frame,
    fit_horizontal_band,
    load_detector_masks,
)
from library.nexus_inspection import inspect_nexus, scalar_metadata, search_inventory
from library import fth_phase_workflow
from library import fthcore
from library import phase_retrieval_core_unified as unified_pr
from library.scan_workflow import load_scan_channel, save_diode_scans


class PreprocessingTests(unittest.TestCase):
    def test_detector_and_beamstop_masks_form_clipped_union(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            detector = np.zeros((3, 4, 3), dtype=np.uint8)
            beamstop = np.zeros((3, 4, 3), dtype=np.uint8)
            detector[0, 0] = (255, 0, 0)
            detector[1, 1] = (255, 0, 0)
            beamstop[1, 1] = (255, 0, 0)
            beamstop[2, 3] = (255, 0, 0)
            Image.fromarray(detector).save(folder / "mask_detector.png")
            Image.fromarray(beamstop).save(folder / "mask_beamstop_7.png")
            fixed, moving, combined = load_detector_masks(folder, 7, (3, 4))
            self.assertEqual(int(fixed.sum()), 2)
            self.assertEqual(int(moving.sum()), 2)
            self.assertEqual(int(combined.sum()), 3)
            np.testing.assert_array_equal(combined, np.clip(fixed + moving, 0, 1))

    def test_dark_fit_ignores_masked_pixels(self):
        dark = np.arange(16, dtype=float).reshape(4, 4)
        frame = 2.5 * dark + 7.0
        frame[0, 0] = 10000
        mask = np.zeros_like(dark, dtype=np.uint8)
        mask[0, 0] = 1
        corrected, scale, offset, _, used = fit_dark_frame(
            frame, dark, slice(None), slice(None), mask=mask
        )
        self.assertEqual(used.sum(), 15)
        self.assertAlmostEqual(scale, 2.5)
        self.assertAlmostEqual(offset, 7.0)
        np.testing.assert_allclose(corrected[mask == 0], 0, atol=1e-10)

    def test_horizontal_band_fit_supports_arbitrary_polynomial_order(self):
        row_count, column_count = 120, 40
        rows = np.arange(row_count, dtype=float)
        centered = rows - row_count / 2
        polynomial = 20 + 0.03 * centered + 0.002 * centered**2
        band = -8 * (
            (rows >= 50) & (rows <= 70)
        ).astype(float)
        pattern = np.repeat((polynomial + band)[:, None], column_count, axis=1)
        result = fit_horizontal_band(
            pattern,
            edge_columns=10,
            polynomial_order=2,
            band_center=60,
            band_width=20,
            band_edge=1,
            band_amplitude=8,
        )
        self.assertEqual(result.polynomial_coefficients.shape, (3,))
        self.assertEqual(result.band_image.shape, pattern.shape)
        self.assertLess(result.band_profile.min(), -6)
        self.assertAlmostEqual(result.band_center, 60, delta=2)

    def test_mode_values_expand_a_shared_support_spatially(self):
        support = np.zeros((9, 9), dtype=np.uint8)
        support[3:6, 3:6] = 1
        modal = unified_pr._mode_supports(support, [1, 2], support.shape)
        self.assertEqual(modal.shape, (2, 9, 9))
        np.testing.assert_array_equal(modal[0], support)
        self.assertGreater(modal[1].sum(), modal[0].sum())
        self.assertEqual(modal[1][4, 4], 1)

    def test_startimage_intercept_correction_is_optional(self):
        field = np.array([[5 + 2j, 9 + 2j]])
        measured = np.array([1.0, 3.0])
        amplitude_fit = np.array([5.0, 9.0])  # slope=2, intercept=3
        slope_only = unified_pr._normalize_startimage_amplitude(
            field, measured, amplitude_fit, subtract_intercept=False
        )
        legacy = unified_pr._normalize_startimage_amplitude(
            field, measured, amplitude_fit, subtract_intercept=True
        )
        np.testing.assert_allclose(slope_only, field / 2)
        np.testing.assert_allclose(legacy, (field - 3) / 2)

    def test_load_average_accepts_one_id_or_a_list(self):
        class FakeLoader:
            def load(self, image_id):
                return Frame(
                    str(image_id), np.full((2, 3), float(image_id)),
                    float(image_id), Path(f"{image_id}.nxs"), {"value": image_id},
                )

        single = load_average(FakeLoader(), 4)
        multiple = load_average(FakeLoader(), [2, 4, 6])
        np.testing.assert_array_equal(single.image, 4)
        np.testing.assert_array_equal(multiple.image, 4)
        self.assertEqual(multiple.metadata["image_ids"], ["2", "4", "6"])
        self.assertEqual(multiple.metadata["average_count"], 3)

    def test_rescale_roi_preserves_relative_image_coverage(self):
        scaled = fth_phase_workflow.rescale_roi(
            [200, 600, 100, 500], (1000, 800), (600, 400)
        )
        np.testing.assert_array_equal(scaled, [120, 360, 50, 250])

    def test_fth_pixel_mask_is_dilated_and_gaussian_smoothed(self):
        binary = np.zeros((31, 31), dtype=np.uint8)
        binary[15, 15] = 1
        original = binary.copy()
        smooth = fth_phase_workflow.smooth_binary_mask(binary, 3, 3)
        self.assertEqual(smooth.shape, binary.shape)
        self.assertTrue(np.issubdtype(smooth.dtype, np.floating))
        self.assertGreater(smooth[15, 18], 0)
        self.assertTrue(np.all((smooth >= 0) & (smooth <= 1)))
        np.testing.assert_array_equal(binary, original)

    def test_fth_notebook_uses_explicit_detector_and_beamstop_masks(self):
        root = Path(__file__).parents[1]
        with (root / "01_FTH.ipynb").open(encoding="utf-8") as handle:
            source = "".join(
                line
                for cell in json.load(handle)["cells"]
                for line in cell.get("source", [])
            )
        self.assertIn('MASK_DETECTOR_FILE = MASK_FOLDER / "mask_detector.png"', source)
        self.assertIn('PLUS_BEAMSTOP_MASK_FILE', source)
        self.assertIn('load_detector_masks(', source)
        self.assertIn('fit_dark_frame(', source)
        self.assertIn('mask=mask_detector', source)
        self.assertIn('state["mask_detector_c"] + state["mask_beamstop_c"]', source)
        self.assertIn("MASK_PIXEL_DILATION =", source)
        self.assertIn("MASK_PIXEL_BLUR_SIGMA =", source)
        self.assertIn("BUTTERWORTH_RADIUS =", source)
        self.assertIn("BUTTERWORTH_ORDER =", source)
        self.assertIn("mask_multiplier = (1 - mask_pixel_fth) * (1 - mask_beamstop_smooth)", source)

    def test_support_preview_uses_float_smoothed_pixel_mask(self):
        root = Path(__file__).parents[1]
        with (root / "03_define_supportmask.ipynb").open(encoding="utf-8") as handle:
            source = "".join(
                line
                for cell in json.load(handle)["cells"]
                for line in cell.get("source", [])
            )
        self.assertIn("mask_pixel.astype(float)", source)
        self.assertIn("mask_pixel_fth = wf.smooth_binary_mask(", source)
        self.assertIn(
            "holo = holo_unmasked * (1 - mask_beamstop_smooth) * (1 - mask_pixel_fth)",
            source,
        )
        self.assertIn("recon = wf.fth_reconstruct(", source)
        self.assertIn("supportmask = (supportmask > 0).astype(np.uint8)", source)

    def test_unified_phase_retrieval_accepts_modes_crop_and_scalar_settings(self):
        shape = (8, 8)
        result = unified_pr.phase_retrieval_algorithm(
            {"+": np.ones(shape)},
            np.zeros(shape, dtype=np.uint8),
            np.ones(shape, dtype=np.uint8),
            {
                "algorithm_list": ["ER"],
                "number_iterations": [1],
                "helicity": ["+"],
                "beta_zero": 0.5,
                "beta_mode": "const",
                "alpha_zero": 0.0,
                "alpha_mode": "const",
                "RL_its": 0,
                "RL_freqs": 1e9,
                "TV_freqs": 1e9,
                "plot_every": 1,
                "average_img": 1,
                "Fourier_last": True,
                "Startimage": [None],
                "Startgamma": [None],
                "output": True,
                "modes": [1],
                "crop": 1,
                "return_format": "dict",
            },
        )
        self.assertEqual(result["full_coherence"]["+"].shape, (6, 6))
        self.assertEqual(result["bsmasks"]["+"].shape, (6, 6))
        self.assertEqual(result["gamma"]["+"].shape, (6, 6))
        self.assertEqual(result["recipe"]["modes"], [1])

    def test_unified_phase_retrieval_accepts_per_hologram_offsets(self):
        shape = (8, 8)
        result = unified_pr.phase_retrieval_algorithm(
            {"+": np.full(shape, 5.0), "-": np.full(shape, 2.0)},
            np.zeros(shape, dtype=np.uint8),
            np.ones(shape, dtype=np.uint8),
            {
                "algorithm_list": ["ER", "ER"],
                "number_iterations": [1, 1],
                "helicity": ["+", "-"],
                "beta_zero": 0.5,
                "beta_mode": "const",
                "alpha_zero": 0.0,
                "alpha_mode": "const",
                "RL_its": 0,
                "RL_freqs": 1e9,
                "TV_freqs": 1e9,
                "plot_every": 1,
                "average_img": 1,
                "Fourier_last": True,
                "Startimage": [None, "+"],
                "Startgamma": [None, None],
                "hologram_offset": {"+": 3.0, "-": 0.5},
                "output": [True, True],
                "return_format": "dict",
            },
        )
        self.assertEqual(result["hologram_offsets"], {"+": 3.0, "-": 0.5})

    def test_data_dict_round_trip_supports_path_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "data.hdf5"
            png = Path(folder) / "supportmask.png"
            fth_phase_workflow.save_data_dict(
                {"supportmask_png": png}, output, overwrite=True
            )
            loaded = fth_phase_workflow.load_data_dict(output)
            self.assertEqual(loaded["supportmask_png"], str(png))

    def test_saved_fth_uses_widget_micrometres_and_subpixel_shift(self):
        class FakeFth:
            propagated_distance = None
            shift = None

            @classmethod
            def propagate(cls, image, distance, setup):
                cls.propagated_distance = distance
                return image

            @staticmethod
            def global_phase_shift(image, phase):
                return image

            @staticmethod
            def reconstruct(image):
                return image

            @classmethod
            def sub_pixel_centering(cls, image, dx, dy):
                cls.shift = (dx, dy)
                return image

        fth_phase_workflow.fth_reconstruct(
            np.ones((2, 2)), {}, FakeFth,
            prop_dist=-10.27, phase=-0.73, dx=0, dy=0.03,
        )
        self.assertAlmostEqual(FakeFth.propagated_distance, -10.27e-6)
        self.assertEqual(FakeFth.shift, (0, 0.03))

        root = Path(__file__).parents[1]
        with (root / "01_FTH_series.ipynb").open(encoding="utf-8") as handle:
            series_source = "".join(
                line
                for cell in json.load(handle)["cells"]
                for line in cell.get("source", [])
            )
        self.assertIn('prop_dist_unit = focus_fth.get("prop_dist_unit", "um")', series_source)
        self.assertIn("REFERENCE_IMAGE_ID = 95", series_source)
        self.assertIn("data_recon_ImId_{REFERENCE_IMAGE_ID:04d}", series_source)
        self.assertIn("CENTER_OVERRIDE = None", series_source)
        self.assertIn("ROI_OVERRIDE = None", series_source)
        self.assertIn("PLUS_IMAGE_IDS =", series_source)
        self.assertIn("MINUS_IMAGE_IDS =", series_source)
        self.assertIn("SERIES_POINTS = None", series_source)
        self.assertIn("FTH_series_{USER}.gif", series_source)
        self.assertIn("save_all=True", series_source)
        self.assertIn("load_average(loader, requested_ids)", series_source)
        self.assertIn("load_average(loader, dark_id)", series_source)
        self.assertIn(
            "prop_dist=prop_dist, phase=phase, dx=dx, dy=dy", series_source
        )

    def test_saved_fth_matches_focus_widget_transform(self):
        rng = np.random.default_rng(4)
        hologram = rng.normal(size=(16, 16))
        setup = {"ccd_dist": 0.5, "energy": 778.831, "px_size": 11e-6}
        prop_um, phase, dx, dy = -10.27, -0.73, 0.0, 0.03
        expected = fthcore.reconstructCDI(
            fthcore.propagate(hologram, prop_um * 1e-6, setup)
            * np.exp(1j * phase)
        )
        expected = fthcore.sub_pixel_centering(expected, dx, dy)
        actual = fth_phase_workflow.fth_reconstruct(
            hologram, setup, fthcore,
            prop_dist=prop_um, phase=phase, dx=dx, dy=dy,
        )
        np.testing.assert_allclose(actual, expected)

    def test_fth_setup_is_derived_after_positive_image_definition(self):
        root = Path(__file__).parents[1]
        with (root / "01_FTH.ipynb").open(encoding="utf-8") as handle:
            cells = json.load(handle)["cells"]
        sources = ["".join(cell.get("source", [])) for cell in cells]
        folder_cell = next(i for i, source in enumerate(sources) if "RAW_FOLDER =" in source)
        image_cell = next(i for i, source in enumerate(sources) if "hologram_inputs =" in source)
        setup_cell = next(i for i, source in enumerate(sources) if "setup_frame =" in source)
        self.assertLess(folder_cell, image_cell)
        self.assertLess(image_cell, setup_cell)
        self.assertIn('im_ids = image_ids(hologram_inputs[positive_label]["id"])', sources[setup_cell])
        self.assertIn("setup_frame = raw_loader.load(im_id)", sources[setup_cell])
        self.assertIn('"px_size": 11.0e-6', sources[setup_cell])

    def test_nexus_inspection_lists_attributes_without_loading_large_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scanx_0012.nxs"
            with h5py.File(path, "w") as handle:
                distance = handle.create_dataset("scan_0012/scan_data/data_03", data=[200.0])
                distance.attrs["long_name"] = "ccd-ts/position"
                distance.attrs["units"] = "mm"
                handle.create_dataset("scan_0012/scan_data/image", data=np.ones((1, 8, 9)))
            rows = inspect_nexus(path)
            distance_rows = search_inventory(rows, "ccd-ts")
            self.assertEqual(len(distance_rows), 1)
            self.assertEqual(distance_rows[0]["value"], [200.0])
            image_row = next(row for row in rows if row["path"].endswith("/image"))
            self.assertEqual(image_row["value"], None)
            self.assertEqual(scalar_metadata(path)["/scan_0012/scan_data/data_03"], [200.0])

    def test_phase_retrieval_uses_configurable_image_id(self):
        root = Path(__file__).parents[1]
        with (root / "04_phase_retrieval.ipynb").open(encoding="utf-8") as handle:
            cells = json.load(handle)["cells"]
        source = "".join(line for cell in cells for line in cell.get("source", []))
        self.assertRegex(source, r"(?m)^im_id = \d+")
        self.assertRegex(source, r"(?m)^topo_id = \d+")
        self.assertIn("data_recon_ImId_{im_id:04d}", source)
        self.assertNotIn("data_recon_ImId_0095", source)
        self.assertIn('"hologram_intensity_cutoff_vmin": offset_vmin', source)
        self.assertIn('"hologram_offset": 0.0', source)

    def test_phase_retrieval_plots_inputs_and_bsmasks_before_run(self):
        root = Path(__file__).parents[1]
        with (root / "04_phase_retrieval.ipynb").open(encoding="utf-8") as handle:
            sources = [
                "".join(cell.get("source", []))
                for cell in json.load(handle)["cells"]
            ]
        diagnostic = next(
            index for index, source in enumerate(sources)
            if "phase-retrieval input intensity" in source
        )
        retrieval = next(
            index for index, source in enumerate(sources)
            if "phase_retrieval_result = pr.phase_retrieval_algorithm(" in source
        )
        self.assertLess(diagnostic, retrieval)
        self.assertIn("_bsmask[_input < 0] = 1", sources[diagnostic])
        self.assertIn("bsmask used (white = excluded)", sources[diagnostic])

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

    def test_retired_mask_notebooks_are_archived(self):
        root = Path(__file__).parents[1]
        self.assertFalse((root / "00a_define_mask_pixel_napari.ipynb").exists())
        self.assertFalse((root / "00_define_mask_pixel_paint.ipynb").exists())
        self.assertTrue(
            (root / "legacy/mask_creation_notebooks/00a_define_mask_pixel_napari.ipynb").exists()
        )

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

    def test_same_pattern_stitch_fits_factor_offset_and_averages_valid_pixels(self):
        reference = np.arange(100.0).reshape(10, 10) + 10
        moving = (reference - 7.0) / 2.5
        masks = [np.zeros((10, 10), bool), np.zeros((10, 10), bool)]
        masks[0][:, :2] = True
        masks[1][:, -2:] = True
        frames = [
            Frame("a", reference, 1.0, Path("a")),
            Frame("b", moving, 1.0, Path("b")),
        ]

        result = stitch_images(
            frames, masks, register=False, fit_intensity=True,
            fit_percentiles=(0, 100),
        )

        np.testing.assert_allclose(result.image, reference, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.prepared_frames[1].coefficients, (2.5, 7.0))
        np.testing.assert_array_equal(result.source_count[:, :2], 1)
        np.testing.assert_array_equal(result.source_count[:, 2:-2], 2)
        np.testing.assert_array_equal(result.source_count[:, -2:], 1)

    def test_shift_mask_uses_row_column_shift(self):
        mask = np.zeros((5, 6), dtype=bool)
        mask[1, 2] = True
        shifted = shift_mask(mask, (2, -1))
        expected = np.zeros_like(mask)
        expected[3, 1] = True
        np.testing.assert_array_equal(shifted, expected)

    def test_same_pattern_stitch_recovers_image_translation(self):
        from scipy.ndimage import shift as translate

        rows, columns = np.mgrid[:96, :96]
        reference = np.exp(-((rows - 35) ** 2 + (columns - 57) ** 2) / 72)
        reference += 0.7 * np.exp(
            -((rows - 68) ** 2 + (columns - 25) ** 2) / 162
        )
        moving = translate(reference, (2, -3), order=1, mode="constant", cval=0)
        frames = [
            Frame("reference", reference, 1.0, Path("reference")),
            Frame("moving", moving, 1.0, Path("moving")),
        ]
        masks = [np.zeros_like(reference), np.zeros_like(reference)]

        result = stitch_images(
            frames, masks, register=True, max_shift=5, fit_intensity=False
        )

        np.testing.assert_allclose(result.prepared_frames[1].shift, (-2, 3))
        overlap = result.prepared_frames[1].valid
        np.testing.assert_allclose(
            result.prepared_frames[1].image[overlap], reference[overlap], atol=1e-12
        )

    def test_stitch_estimation_roi_limits_fit_but_not_final_coverage(self):
        reference = np.arange(100.0).reshape(10, 10) + 10
        moving = (reference - 7.0) / 2.5
        moving[:2] = 10000  # Deliberately corrupt pixels outside the fit ROI.
        frames = [
            Frame("reference", reference, 1.0, Path("reference")),
            Frame("moving", moving, 1.0, Path("moving")),
        ]
        masks = [np.zeros_like(reference), np.zeros_like(reference)]

        result = stitch_images(
            frames,
            masks,
            register=False,
            fit_intensity=True,
            fit_percentiles=(0, 100),
            estimation_roi=np.s_[2:8, 2:8],
        )

        np.testing.assert_allclose(result.prepared_frames[1].coefficients, (2.5, 7.0))
        np.testing.assert_array_equal(result.source_count, 2)

    def test_master_pixels_are_kept_and_auxiliary_images_only_fill_its_mask(self):
        master = np.arange(100.0).reshape(10, 10) + 10
        auxiliary = (master - 5) / 2
        master_mask = np.zeros_like(master, dtype=bool)
        master_mask[:, :3] = True
        frames = [
            Frame("master", master, 1.0, Path("master")),
            Frame("auxiliary", auxiliary, 1.0, Path("auxiliary")),
        ]

        result = stitch_images(
            frames,
            [master_mask, np.zeros_like(master_mask)],
            register=False,
            fit_intensity=True,
            fit_percentiles=(0, 100),
            use_master_where_valid=True,
        )

        np.testing.assert_allclose(result.image, master)
        np.testing.assert_array_equal(result.source_count, 1)

    def test_same_pattern_stitch_supports_nonlinear_intensity_fit(self):
        moving = np.linspace(1, 20, 400).reshape(20, 20)
        master = 0.25 * moving**2 + 1.5 * moving + 8
        frames = [
            Frame("master", master, 1.0, Path("master")),
            Frame("auxiliary", moving, 1.0, Path("auxiliary")),
        ]
        masks = [np.zeros_like(master), np.zeros_like(master)]

        result = stitch_images(
            frames,
            masks,
            register=False,
            fit_intensity=True,
            fit_degree=2,
            fit_percentiles=(0, 100),
        )

        np.testing.assert_allclose(
            result.prepared_frames[1].coefficients, (0.25, 1.5, 8.0), atol=1e-10
        )
        np.testing.assert_allclose(result.image, master, atol=1e-10)

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
                distance = scan_data.create_dataset("data_03", data=[200.0])
                distance.attrs["long_name"] = "i14-m-cx2/ex-fth/ccd-ts/position"
                distance.attrs["units"] = "mm"
                entry["SEXTANTS/mono/energy"] = [702.8]
                entry["SEXTANTS/hu80.2_energy/polarisation"] = [3]
            frame = SextantsNexusLoader(folder).load(24)
            self.assertEqual(frame.image.shape, (8, 9))
            self.assertEqual(frame.exposure, 20.0)
            self.assertEqual(frame.metadata["energy_eV"], 702.8)
            self.assertEqual(frame.metadata["polarization_code"], 3)
            self.assertEqual(frame.metadata["ccd_dist_m"], 0.5)
            self.assertTrue(frame.metadata["ccd_dist_path"].endswith("data_03"))
            self.assertTrue(frame.metadata["image_path"].endswith("data_21"))

    def test_sextants_loader_can_preserve_the_frame_axis(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scanx_0024.nxs"
            with h5py.File(path, "w") as handle:
                scan_data = handle.create_group("scan_0024/scan_data")
                detector = scan_data.create_dataset(
                    "data_21", data=np.arange(24).reshape(2, 3, 4)
                )
                detector.attrs["interpretation"] = "image"
                scan_data["integration_times"] = [1.0, 2.0]
            loader = SextantsNexusLoader(folder)
            stack = loader.load_stack(24)
            averaged = loader.load(24)
            self.assertEqual(stack.image.shape, (2, 3, 4))
            np.testing.assert_array_equal(
                stack.metadata["frame_exposures"], [1.0, 2.0]
            )
            np.testing.assert_allclose(averaged.image, stack.image.mean(axis=0))

    def test_sextants_loader_accepts_constant_per_frame_ccd_distance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scanx_0413.nxs"
            with h5py.File(path, "w") as handle:
                scan_data = handle.create_group("scan_0413/scan_data")
                detector = scan_data.create_dataset(
                    "data_24", data=np.ones((3, 4, 5), dtype=np.uint32)
                )
                detector.attrs["interpretation"] = "image"
                scan_data["integration_times"] = [0.05, 0.05, 0.05]
                distance = scan_data.create_dataset("data_03", data=[200.0] * 3)
                distance.attrs["long_name"] = "ccd-ts/position"

            frame = SextantsNexusLoader(folder).load_stack(413)
            self.assertEqual(frame.image.shape, (3, 4, 5))
            self.assertEqual(frame.metadata["ccd_dist_m"], 0.5)

    def test_load_processing_returns_average_and_joined_3d_frames(self):
        class StackLoader:
            def load_stack(self, image_id):
                values = np.full((int(image_id), 2, 3), int(image_id), dtype=np.uint32)
                return Frame(str(image_id), values, 1.0, Path(f"{image_id}.nxs"))

        average, frames = load_processing(StackLoader(), [1, 2])
        self.assertEqual(frames.shape, (3, 2, 3))
        self.assertTrue(np.issubdtype(frames.dtype, np.floating))
        self.assertTrue(np.issubdtype(average.dtype, np.floating))
        np.testing.assert_array_equal(frames[:, 0, 0], [1.0, 2.0, 2.0])
        np.testing.assert_allclose(average, np.full((2, 3), 5.0 / 3.0))


if __name__ == "__main__":
    unittest.main()
