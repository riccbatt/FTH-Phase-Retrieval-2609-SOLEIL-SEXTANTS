"""Support fitting must precede Fourier-field initialization."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import center_of_mass, label

from library import phase_retrieval_core_unified as unified
from library.phase_retrieval_geometry import (
    recenter_modal_supports,
    recenter_source_support,
)


class SupportRecenteringTests(unittest.TestCase):
    def test_harmonic_scales_aperture_sizes_and_separations_together(self):
        support = np.zeros((64, 64), dtype=np.uint8)
        support[5:8, 10:13] = 1
        support[14:17, 22:25] = 1
        modes, shifts = recenter_modal_supports(support, [1, 2], center="image")
        np.testing.assert_array_equal(modes[0], support)
        self.assertEqual(shifts[0], (0, 0))
        self.assertTrue(any(value != 0 for value in shifts[1]))
        centers = []
        areas = []
        for mask in modes:
            components, count = label(mask)
            self.assertEqual(count, 2)
            centers.append(np.array(center_of_mass(mask, components, [1, 2])))
            areas.append([np.count_nonzero(components == i) for i in [1, 2]])
            self.assertTrue(set(np.unique(mask)) <= {0, 1})
        np.testing.assert_allclose(centers[1][1] - centers[1][0],
                                   2 * (centers[0][1] - centers[0][0]))
        np.testing.assert_array_equal(areas[1], 4 * np.asarray(areas[0]))
        # A common translation preserves both aperture positions after scaling.
        image_center = (np.array(support.shape) - 1) / 2
        np.testing.assert_allclose(centers[1], image_center
                                   + 2 * (centers[0] - image_center) + shifts[1])
        self.assertFalse(modes[1, 0].any() or modes[1, -1].any()
                         or modes[1, :, 0].any() or modes[1, :, -1].any())

    def test_harmonic_does_not_shift_when_whole_support_already_fits(self):
        support = np.zeros((64, 64), dtype=np.uint8)
        support[25:28, 25:28] = 1
        support[34:37, 34:37] = 1
        _, shifts = recenter_modal_supports(support, [1, 2], center="image")
        self.assertEqual(shifts, [(0, 0), (0, 0)])

    def test_harmonic_rejects_span_that_translation_cannot_fit(self):
        support = np.zeros((32, 32), dtype=np.uint8)
        support[4:7, 4:7] = 1
        support[25:28, 25:28] = 1
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            recenter_modal_supports(support, [1, 2], center="image")

    def test_source_support_moves_into_cropped_object_grid(self):
        support = np.zeros((64, 64), dtype=np.uint8)
        support[2:5, 25:30] = 1
        shifted, translation = recenter_source_support(support, crop=16, binning=1)
        self.assertGreater(translation[0], 0)
        self.assertEqual(int(shifted.sum()), int(support.sum()))

    def test_second_mode_is_fully_retained_and_initial_field_uses_it(self):
        support = np.zeros((32, 32), dtype=np.uint8)
        support[2:5, 13:16] = 1
        modes, translations = recenter_modal_supports(support, [1, 2])
        self.assertEqual(translations[0], (0, 0))
        self.assertGreater(translations[1][0], 0)
        self.assertEqual(int(modes[1].sum()), 4 * int(support.sum()))

        rng = np.random.default_rng(3)
        hologram = 1 + rng.uniform(size=(32, 32))
        recipe = unified.default_phase_retrieval_recipe()
        recipe.update(
            algorithm_list=["ER"], number_iterations=[1], helicity=["field"],
            beta_zero=[0.5], beta_mode=["const"], alpha_zero=[0.0],
            alpha_mode=["const"], RL_its=[0], RL_freqs=[1e9],
            TV_freqs=[1e9], plot_every=[1e9], average_img=[1],
            Fourier_last=[True], output=[True], Startimage=[None],
            Startgamma=[None], modes=[1, 2], recenter_modal_supports=True,
            return_format="dict",
        )
        captured = []

        def fake_kernel(**kwargs):
            captured.append(np.asarray(kwargs["Phase"]).copy())
            return kwargs["Phase"], np.zeros(1), np.zeros(1), None

        with patch.object(unified, "PhaseRtrv_core", side_effect=fake_kernel):
            result = unified.phase_retrieval_algorithm(
                {"field": hologram}, np.zeros((32, 32), dtype=np.uint8),
                support, recipe,
            )
        np.testing.assert_array_equal(result["supportmask"], modes)
        self.assertEqual(result["mode_support_shifts"], translations)
        for mode_index in range(2):
            expected = np.fft.ifftshift(np.fft.ifft2(
                np.fft.fftshift(modes[mode_index]),
            ))
            nonzero = np.abs(expected) > 1e-12
            ratio = captured[0][mode_index][nonzero] / expected[nonzero]
            np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6, atol=1e-6)

    def test_gaussian_real_space_mode_is_supported(self):
        rng = np.random.default_rng(7)
        hologram = 1 + rng.uniform(size=(28, 28))
        recipe = unified.default_phase_retrieval_recipe()
        recipe.update(
            algorithm_list=["ER"], number_iterations=[1], helicity=["field"],
            beta_zero=[0.5], beta_mode=["const"], alpha_zero=[0.0],
            alpha_mode=["const"], RL_its=[0], RL_freqs=[1e9],
            TV_freqs=[1e9], plot_every=[1e9], average_img=[1],
            Fourier_last=[True], output=[True], Startimage=[None],
            Startgamma=[None], modes=[0, 1], recenter_modal_supports=False,
            gaussian_mode_sigma_x=[5.0], gaussian_mode_sigma_y=[4.0],
            gaussian_mode_angle=[0.25 * np.pi], gaussian_mode_amplitude=[1.5],
            gaussian_mode_center=[None], gaussian_mode_coherent=[False],
            return_format="dict",
        )
        support = np.zeros((28, 28), dtype=np.uint8)
        support[10:18, 10:18] = 1
        with patch.object(unified, "PhaseRtrv_core", return_value=(np.ones((2, 28, 28), dtype=complex), np.zeros(1), np.zeros(1), None)):
            result = unified.phase_retrieval_algorithm(
                {"field": hologram}, np.zeros((28, 28), dtype=np.uint8),
                support, recipe,
            )
        self.assertEqual(result["supportmask"].shape, (2, 28, 28))
        self.assertGreater(np.count_nonzero(result["supportmask"][0]), 0)
        self.assertGreater(np.count_nonzero(result["supportmask"][1]), 0)


if __name__ == "__main__":
    unittest.main()
