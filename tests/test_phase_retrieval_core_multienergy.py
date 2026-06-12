import unittest

import numpy as np

from library import phase_retrieval_core_multienergy as multi


class PhaseRetrievalCoreMultienergyTests(unittest.TestCase):
    def test_fourier_svd_dispatch_matches_explicit_log_object_pipeline(self):
        rng = np.random.default_rng(17)
        fields = (
            rng.normal(size=(5, 4, 6))
            + 1j * rng.normal(size=(5, 4, 6))
        )
        settings = {
            "rank": 2,
            "static_mode": "mean",
            "weights": np.linspace(1.0, 2.0, fields.shape[0]),
            "relaxation": 0.6,
        }

        log_objects = multi.fourier_field_to_object_log(fields)
        expected_logs, expected_components = (
            multi.project_log_object_low_rank(
                log_objects,
                return_components=True,
                **settings,
            )
        )
        expected = multi.object_log_to_fourier_field(expected_logs)

        projected, components = multi.project_fourier_fields_multi_energy(
            fields,
            projection_model="svd",
            return_components=True,
            **settings,
        )

        np.testing.assert_allclose(projected, expected, atol=1e-12)
        np.testing.assert_allclose(
            components["static_log_object"],
            expected_components["static_log_object"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            components["energy_dependent_log_object"],
            expected_components["energy_dependent_log_object"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            components["singular_values"],
            expected_components["singular_values"],
            atol=1e-12,
        )

    def test_fourier_rank1_dispatch_matches_explicit_log_object_pipeline(self):
        rng = np.random.default_rng(18)
        fields = (
            rng.normal(size=(6, 5, 4))
            + 1j * rng.normal(size=(6, 5, 4))
        )
        settings = {
            "static_mode": "first",
            "weights": np.linspace(0.5, 1.5, fields.shape[0]),
            "relaxation": 0.7,
            "spectral_constraint": "free",
        }

        log_objects = multi.fourier_field_to_object_log(fields)
        expected_logs = multi.project_log_object_rank1_spectral(
            log_objects,
            **settings,
        )
        expected = multi.object_log_to_fourier_field(expected_logs)

        projected = multi.project_fourier_fields_multi_energy(
            fields,
            projection_model="rank1_spectral",
            **settings,
        )

        np.testing.assert_allclose(projected, expected, atol=1e-12)

    def test_explicit_free_model_recovers_exact_rank1_log_object(self):
        rng = np.random.default_rng(0)
        n_energy = 7
        shape = (5, 6)
        static = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        spatial = rng.normal(size=shape)
        spectrum = np.linspace(-1, 1, n_energy) + 1j * np.sin(
            np.linspace(0, np.pi, n_energy)
        )
        log_object = (
            static[None]
            + spectrum[:, None, None] * spatial[None]
        )

        projected, components = multi.project_log_object_rank1_spectral(
            log_object,
            spectral_constraint="free",
            return_components=True,
        )

        np.testing.assert_allclose(projected, log_object, atol=1e-12)
        self.assertEqual(components["spectral_spatial_map"].shape, shape)
        self.assertEqual(
            components["spectral_coefficients"].shape,
            (n_energy,),
        )

    def test_svd_projection_enforces_requested_residual_rank(self):
        rng = np.random.default_rng(1)
        log_object = (
            rng.normal(size=(6, 4, 5))
            + 1j * rng.normal(size=(6, 4, 5))
        )

        projected, components = multi.project_log_object_low_rank(
            log_object,
            rank=2,
            return_components=True,
        )
        residual = (
            projected - components["static_log_object"][None]
        ).reshape(6, -1).T

        self.assertLessEqual(np.linalg.matrix_rank(residual, tol=1e-10), 2)

    def test_known_beta_kk_keeps_delta_shape(self):
        energies = np.linspace(770, 790, 11)
        beta = np.exp(-((energies - 780) / 3) ** 2)
        delta = np.linspace(-0.5, 0.5, energies.size)

        constrained, _ = multi.constrain_complex_spectrum(
            beta.astype(complex),
            spectral_constraint="known_beta_kk",
            energy_values=energies,
            known_beta_spectrum=beta,
            known_delta_spectrum=delta,
            fit_known_beta_scale=False,
        )

        np.testing.assert_allclose(np.real(constrained), beta)
        np.testing.assert_allclose(
            np.imag(constrained) - np.mean(np.imag(constrained)),
            delta - np.mean(delta),
            atol=1e-15,
        )
        self.assertGreater(np.std(np.imag(constrained)), 0)

    def test_rank1_gauge_aligns_absorption_with_known_beta(self):
        rng = np.random.default_rng(13)
        n_energy = 9
        shape = (5, 6)
        known_absorption = np.linspace(-1, 1, n_energy) ** 3
        dispersion = np.sin(np.linspace(-1, 1, n_energy) * np.pi)
        spectrum = known_absorption + 1j * dispersion
        static = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        spatial = rng.uniform(0.2, 1.0, size=shape)
        log_object = (
            static[None]
            + spectrum[:, None, None] * spatial[None]
        )

        _, components = multi.project_log_object_rank1_spectral(
            log_object,
            spectral_constraint="known_beta",
            known_beta_spectrum=known_absorption,
            return_components=True,
        )

        fitted_absorption = np.real(components["spectral_coefficients"])
        correlation = np.corrcoef(
            fitted_absorption,
            known_absorption,
        )[0, 1]
        self.assertGreater(correlation, 0.999)

    def test_no_projection_matches_independent_energy_updates(self):
        rng = np.random.default_rng(3)
        n_energy = 3
        shape = (6, 6)
        holograms = rng.uniform(0.5, 2.0, (n_energy, *shape))
        support = np.ones(shape)
        masks = np.zeros((n_energy, *shape), dtype=int)
        start_fields = np.sqrt(holograms) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, (n_energy, *shape))
        )
        recipe = {
            "inner_mode": "ER",
            "outer_iterations": 1,
            "inner_Nit": 3,
            "warmup_Nit": 0,
            "shuffle_energies": False,
            "projection_model": "none",
            "plot_every": 2,
            "average_img": 2,
        }

        retrieved, components, _, _ = (
            multi.multi_energy_phase_retrieval_algorithm(
                holograms,
                masks,
                support,
                multi_energy_recipe=recipe,
                start_fields=start_fields,
            )
        )

        expected = np.empty_like(retrieved)
        for j in range(n_energy):
            expected[j], _, _, _ = multi.PhaseRtrv_core(
                diffract=np.sqrt(holograms[j]),
                mask=support,
                mode="ER",
                Nit=3,
                Phase=start_fields[j],
                plot_every=2,
                bsmask=masks[j],
                average_img=2,
                Fourier_last=True,
                RL_freq=4,
                RL_it=0,
                TV_freq=1e9,
            )

        np.testing.assert_allclose(retrieved, expected)
        self.assertEqual(components["projection_model"], "none")

    def test_zero_intensity_remains_constrained(self):
        holograms = np.ones((2, 4, 4))
        holograms[0, 0, 0] = 0

        _, _, masks = multi._prepare_energy_amplitudes(
            holograms,
            np.zeros((4, 4), dtype=int),
        )

        self.assertEqual(masks[0, 0, 0], 0)

    def test_joint_projection_returns_measured_fourier_amplitudes(self):
        rng = np.random.default_rng(8)
        holograms = rng.uniform(0.5, 2.0, (5, 6, 6))
        start_fields = np.sqrt(holograms) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, holograms.shape)
        )

        retrieved, components, _, _ = (
            multi.multi_energy_phase_retrieval_algorithm(
                holograms,
                np.zeros((6, 6), dtype=int),
                np.ones((6, 6)),
                multi_energy_recipe={
                    "inner_mode": "ER",
                    "outer_iterations": 1,
                    "inner_Nit": 2,
                    "warmup_Nit": 0,
                    "shuffle_energies": False,
                    "projection_model": "svd",
                    "rank": 1,
                    "plot_every": 2,
                    "average_img": 1,
                },
                start_fields=start_fields,
            )
        )

        np.testing.assert_allclose(
            np.abs(retrieved),
            np.sqrt(holograms),
            atol=1e-12,
        )
        self.assertTrue(components["final_fourier_constraint_applied"])


if __name__ == "__main__":
    unittest.main()
