"""Support fitting must precede Fourier-field initialization."""

import unittest
from unittest.mock import patch

import numpy as np

from library import phase_retrieval_core_unified as unified
from library.phase_retrieval_geometry import (
    recenter_modal_supports,
    recenter_source_support,
)


class SupportRecenteringTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
