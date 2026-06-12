import unittest

import numpy as np

from library import phase_retrieval_core as single
from library import phase_retrieval_core_multimode as multi


class PhaseRetrievalCoreMultimodeTests(unittest.TestCase):
    def test_single_mode_matches_core_full_coherence(self):
        rng = np.random.default_rng(11)
        shape = (8, 8)
        diffract = rng.uniform(0.2, 2.0, shape)
        support = np.zeros(shape)
        support[2:6, 2:6] = 1
        bsmask = np.zeros(shape, dtype=int)
        bsmask[0, 0] = 1
        phase = diffract * np.exp(
            1j * rng.uniform(-np.pi, np.pi, shape)
        )
        kwargs = {
            "diffract": diffract,
            "mask": support,
            "mode": "HAPRE",
            "Nit": 20,
            "beta_zero": 0.5,
            "beta_mode": "arctan",
            "Phase": phase,
            "plot_every": 7,
            "bsmask": bsmask,
            "average_img": 5,
            "Fourier_last": True,
        }

        expected, expected_errors, _, _ = single.PhaseRtrv_core(**kwargs)
        result, errors, _, _ = multi.PhaseRtrv_core(**kwargs, Nmodes=1)

        np.testing.assert_allclose(result, expected, atol=3e-7, rtol=1e-7)
        np.testing.assert_allclose(errors, expected_errors, atol=1e-10)

    def test_single_mode_matches_core_partial_coherence(self):
        rng = np.random.default_rng(21)
        shape = (6, 6)
        diffract = rng.uniform(0.5, 1.5, shape)
        phase = diffract * np.exp(
            1j * rng.uniform(-np.pi, np.pi, shape)
        )
        gamma = np.ones(shape, dtype=float)
        gamma /= np.sum(gamma)
        kwargs = {
            "diffract": diffract,
            "mask": np.ones(shape),
            "mode": "ER",
            "Nit": 5,
            "Phase": phase,
            "average_img": 2,
            "plot_every": 2,
            "Fourier_last": True,
            "gamma": gamma,
            "RL_freq": 2,
            "RL_it": 1,
        }

        expected, expected_errors, _, expected_gamma = (
            single.PhaseRtrv_core(**kwargs)
        )
        result, errors, _, result_gamma = multi.PhaseRtrv_core(
            **kwargs, Nmodes=1
        )

        np.testing.assert_allclose(result, expected, atol=4e-7, rtol=1e-7)
        np.testing.assert_allclose(errors, expected_errors, atol=1e-10)
        np.testing.assert_allclose(
            result_gamma, expected_gamma, atol=3e-10, rtol=1e-7
        )

    def test_single_mode_public_wrapper_matches_standard_wrapper(self):
        rng = np.random.default_rng(22)
        shape = (6, 6)
        pos = rng.uniform(0.5, 2.0, shape)
        neg = rng.uniform(0.5, 2.0, shape)
        mask = np.zeros(shape, dtype=int)
        support = np.zeros(shape)
        support[1:5, 1:5] = 1
        start = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(support))
        )
        recipe = {
            "algorithm_list": ["ER", "ER"],
            "number_iterations": [3, 3],
            "helicity": ["pos", "neg"],
            "beta_zero": [0.5, 0.5],
            "beta_mode": ["const", "const"],
            "alpha_zero": [0.0, 0.0],
            "alpha_mode": ["const", "const"],
            "RL_its": [0, 0],
            "RL_freqs": [4, 4],
            "TV_freqs": [10, 10],
            "plot_every": [2, 2],
            "average_img": [2, 2],
            "Fourier_last": [True, True],
            "hologram_intensity_cutoff_vmin": -1,
            "Startimage": [start, "pos"],
            "Startgamma": [None, None],
        }

        expected = single.phase_retrieval_algorithm(
            pos, neg, mask, support, recipe
        )
        result = multi.phase_retrieval_algorithm(
            pos, neg, mask, support, {**recipe, "Nmodes": 1}
        )

        for index in [0, 1, 4, 5, 6, 7]:
            np.testing.assert_allclose(
                result[index], expected[index], atol=3e-7, rtol=1e-7
            )
        for result_step, expected_step in zip(
            result[8]["steps"], expected[8]["steps"]
        ):
            np.testing.assert_allclose(
                result_step["error"],
                expected_step["error"],
                atol=1e-7,
                rtol=1e-7,
            )

    def test_multimode_final_constraint_uses_summed_intensity(self):
        rng = np.random.default_rng(4)
        shape = (8, 8)
        nmodes = 2
        diffract = rng.uniform(0.5, 1.5, shape)
        phase = np.stack(
            [
                diffract
                / np.sqrt(nmodes)
                * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
                for _ in range(nmodes)
            ]
        )

        result, errors, _, gamma = multi.PhaseRtrv_core(
            diffract=diffract,
            mask=np.ones((nmodes, *shape)),
            mode="ER",
            Nit=5,
            Phase=phase,
            average_img=3,
            plot_every=2,
            Fourier_last=True,
            Nmodes=nmodes,
        )

        modal_amplitude = np.sqrt(np.sum(np.abs(result) ** 2, axis=0))
        np.testing.assert_allclose(modal_amplitude, diffract, atol=1e-12)
        self.assertEqual(result.shape, (nmodes, *shape))
        self.assertTrue(np.isfinite(errors).all())
        self.assertIsNone(gamma)

    def test_random_multimode_start_is_not_mode_degenerate(self):
        result, _, _, _ = multi.PhaseRtrv_core(
            diffract=np.ones((8, 8)),
            mask=np.ones((8, 8)),
            mode="ER",
            Nit=1,
            Phase=None,
            seed=True,
            average_img=1,
            plot_every=1,
            Fourier_last=False,
            Nmodes=2,
        )

        self.assertGreater(np.max(np.abs(result[0] - result[1])), 1e-6)

    def test_short_multimode_run_does_not_average_empty_slots(self):
        diffract = np.ones((4, 4))
        phase = np.stack(
            [
                np.ones((4, 4), dtype=complex) / np.sqrt(2),
                1j * np.ones((4, 4), dtype=complex) / np.sqrt(2),
            ]
        )

        result, _, _, _ = multi.PhaseRtrv_core(
            diffract=diffract,
            mask=np.ones((2, 4, 4)),
            mode="ER",
            Nit=1,
            Phase=phase,
            average_img=10,
            Fourier_last=True,
            Nmodes=2,
        )

        modal_amplitude = np.sqrt(np.sum(np.abs(result) ** 2, axis=0))
        np.testing.assert_allclose(modal_amplitude, diffract)


if __name__ == "__main__":
    unittest.main()
