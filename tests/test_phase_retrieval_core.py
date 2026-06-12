import unittest

import numpy as np

from library import phase_retrieval_core as core


class PhaseRetrievalCoreTests(unittest.TestCase):
    def test_short_full_coherence_run_does_not_average_with_zero_slots(self):
        diffract = np.ones((4, 4), dtype=float)
        support = np.ones_like(diffract)
        phase = np.ones_like(diffract, dtype=complex)

        result, errors, _, gamma = core.PhaseRtrv_core(
            diffract=diffract,
            mask=support,
            mode="ER",
            Nit=1,
            Phase=phase,
            average_img=10,
            Fourier_last=True,
        )

        np.testing.assert_allclose(np.abs(result), diffract)
        self.assertEqual(len(errors), 1)
        self.assertIsNone(gamma)

    def test_cross_helicity_scaling_uses_intensity_slope(self):
        source = np.arange(1, 17, dtype=float).reshape(4, 4)
        target = 4 * source + 3
        phase = np.ones((4, 4), dtype=complex)
        masks = {
            "pos": np.zeros((4, 4), dtype=int),
            "neg": np.zeros((4, 4), dtype=int),
        }

        scaled = core._scale_phase_between_helicities(
            phase,
            source_helicity="pos",
            target_helicity="neg",
            intensity_data={"pos": source, "neg": target},
            bsmasks=masks,
        )

        np.testing.assert_allclose(scaled, 2 * phase)

    def test_zero_intensity_pixels_remain_constrained(self):
        pos = np.ones((4, 4), dtype=float)
        neg = np.ones((4, 4), dtype=float)
        pos[0, 0] = 0
        neg[1, 1] = 0
        mask = np.zeros((4, 4), dtype=int)
        support = np.ones((4, 4), dtype=float)
        recipe = {
            "algorithm_list": ["ER"],
            "number_iterations": [1],
            "helicity": ["pos"],
            "beta_zero": [0.5],
            "beta_mode": ["const"],
            "alpha_zero": [0.0],
            "alpha_mode": ["const"],
            "RL_its": [0],
            "RL_freqs": [2],
            "TV_freqs": [2],
            "plot_every": [1],
            "average_img": [1],
            "Fourier_last": [True],
            "hologram_intensity_cutoff_vmin": -1,
            "Startimage": [np.ones((4, 4), dtype=complex)],
            "Startgamma": [None],
        }

        result = core.phase_retrieval_algorithm(
            pos, neg, mask, support, phase_retrieval_recipe=recipe
        )

        bsmask_p, bsmask_n = result[4], result[5]
        self.assertEqual(bsmask_p[0, 0], 0)
        self.assertEqual(bsmask_n[1, 1], 0)

    def test_unknown_recipe_keys_are_rejected(self):
        data = np.ones((4, 4), dtype=float)

        with self.assertRaisesRegex(ValueError, "Unknown phase-retrieval recipe"):
            core.phase_retrieval_algorithm(
                data,
                data,
                np.zeros_like(data),
                np.ones_like(data),
                phase_retrieval_recipe={"algorithm_list_full_coherence": ["ER"]},
            )


if __name__ == "__main__":
    unittest.main()
