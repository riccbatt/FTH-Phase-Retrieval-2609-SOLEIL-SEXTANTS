import unittest

import numpy as np

from library import phase_retrieval_core_multienergy as multi_energy
from library import phase_retrieval_core_multienergy_multimode as combined


class PhaseRetrievalCoreMultienergyMultimodeTests(unittest.TestCase):
    def test_single_mode_matches_multi_energy_driver(self):
        rng = np.random.default_rng(31)
        holograms = rng.uniform(0.5, 2.0, (3, 6, 6))
        start_fields = np.sqrt(holograms) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, holograms.shape)
        )
        recipe = {
            "mode": "ER",
            "outer_iterations": 1,
            "inner_iterations": 3,
            "warmup_iterations": 0,
            "shuffle_energies": False,
            "projection_model": "none",
            "average_img": 2,
            "plot_every": 2,
        }
        mask = np.zeros((6, 6), dtype=int)
        support = np.ones((6, 6))

        expected, _, expected_masks, _ = (
            multi_energy.multi_energy_phase_retrieval_algorithm(
                holograms,
                mask,
                support,
                multi_energy_recipe=recipe,
                start_fields=start_fields,
            )
        )
        result, components, result_masks, _ = (
            combined.multi_energy_phase_retrieval_algorithm(
                holograms,
                mask,
                support,
                multi_energy_recipe={**recipe, "Nmodes": 1},
                start_fields=start_fields,
            )
        )

        np.testing.assert_allclose(result, expected, atol=3e-7, rtol=1e-7)
        np.testing.assert_array_equal(result_masks, expected_masks)
        self.assertEqual(result.shape, holograms.shape)
        self.assertEqual(components["Nmodes"], 1)

    def test_multimode_output_obeys_summed_fourier_intensity(self):
        rng = np.random.default_rng(32)
        holograms = rng.uniform(0.5, 2.0, (4, 6, 6))
        nmodes = 3

        result, components, _, _ = (
            combined.multi_energy_phase_retrieval_algorithm(
                holograms,
                np.zeros((6, 6), dtype=int),
                np.ones((6, 6)),
                multi_energy_recipe={
                    "mode": "ER",
                    "outer_iterations": 1,
                    "inner_iterations": 2,
                    "warmup_iterations": 0,
                    "shuffle_energies": False,
                    "projection_model": "none",
                    "Nmodes": nmodes,
                    "average_img": 1,
                    "plot_every": 2,
                },
            )
        )

        reconstructed_amplitude = np.sqrt(
            np.sum(np.abs(result) ** 2, axis=1)
        )
        np.testing.assert_allclose(
            reconstructed_amplitude,
            np.sqrt(holograms),
            atol=1e-12,
        )
        self.assertEqual(result.shape, (4, nmodes, 6, 6))
        self.assertEqual(components["Nmodes"], nmodes)

    def test_spectral_projection_is_applied_to_every_mode(self):
        rng = np.random.default_rng(33)
        n_energy = 5
        nmodes = 2
        fields = (
            rng.normal(size=(n_energy, nmodes, 5, 5))
            + 1j * rng.normal(size=(n_energy, nmodes, 5, 5))
        )

        projected, components = (
            combined.project_fourier_fields_multi_energy_multimode(
                fields,
                projection_model="svd",
                rank=1,
                return_components=True,
            )
        )

        self.assertEqual(projected.shape, fields.shape)
        self.assertEqual(len(components["mode_components"]), nmodes)
        for mode_components in components["mode_components"]:
            self.assertEqual(mode_components["projection_model"], "svd")

    def test_default_multimode_initialization_is_not_degenerate(self):
        fields = combined._initialize_modal_fields(
            supportmask=np.ones((6, 6)),
            amplitudes=np.ones((2, 6, 6)),
            intensities=np.ones((2, 6, 6)),
            mask_stack=np.zeros((2, 6, 6), dtype=int),
            nmodes=2,
            random_seed=0,
        )

        self.assertGreater(
            np.max(np.abs(fields[:, 0] - fields[:, 1])),
            1e-6,
        )
        np.testing.assert_allclose(
            np.sqrt(np.sum(np.abs(fields) ** 2, axis=1)),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
