import unittest
from unittest import mock

import numpy as np

from library import phase_retrieval_core_general as general


class PhaseRetrievalCoreGeneralTests(unittest.TestCase):
    def test_projection_separates_beam_energy_state_and_polarization(self):
        shape = (3, 4)
        common = {
            "beam_1": np.full(shape, 0.20 + 0.10j),
            "beam_2": np.full(shape, -0.10 + 0.05j),
        }
        response = {
            ("s1", "e1"): np.full(shape, 0.03 + 0.04j),
            ("s2", "e1"): np.full(shape, -0.02 + 0.01j),
            ("s1", "e2"): np.full(shape, 0.06 - 0.03j),
        }
        states = ["s1", "s1", "s1", "s2", "s1"]
        energies = ["e1", "e1", "e1", "e1", "e2"]
        polarizations = [1, -1, 1, 1, 1]
        beams = ["beam_1", "beam_1", "beam_2", "beam_1", "beam_1"]
        log_objects = np.stack([
            common[beam] + polarization * response[(state, energy)]
            for state, energy, polarization, beam in zip(
                states,
                energies,
                polarizations,
                beams,
            )
        ])

        projected, components = general.project_log_objects_general(
            log_objects,
            states,
            energies,
            polarizations,
            beams,
            return_components=True,
        )

        np.testing.assert_allclose(projected, log_objects, atol=1e-12)
        for beam, expected in common.items():
            np.testing.assert_allclose(
                components["common_log_objects_by_beam"][beam],
                expected,
                atol=1e-12,
            )
        for state_energy, expected in response.items():
            np.testing.assert_allclose(
                components[
                    "response_log_objects_by_state_energy"
                ][state_energy],
                expected,
                atol=1e-12,
            )
        self.assertTrue(components["identifiable"])

    def test_beam_change_preserves_material_response(self):
        shape = (2, 3)
        response = np.full(shape, 0.04 + 0.02j)
        common_1 = np.full(shape, 0.1 + 0.0j)
        common_2 = np.full(shape, -0.2 + 0.1j)
        log_objects = np.stack([
            common_1 + response,
            common_1 - response,
            common_2 + response,
        ])

        _, components = general.project_log_objects_general(
            log_objects,
            state_labels=["state", "state", "state"],
            energy_labels=["energy", "energy", "energy"],
            polarization_coefficients=[1, -1, 1],
            beam_labels=["beam_1", "beam_1", "beam_2"],
            return_components=True,
        )

        np.testing.assert_allclose(
            components["response_log_objects"][0],
            response,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            components["common_log_objects_by_beam"]["beam_2"],
            common_2,
            atol=1e-12,
        )

    def test_rank_deficient_geometry_is_rejected(self):
        log_objects = np.zeros((2, 2, 2), dtype=complex)
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            general.project_log_objects_general(
                log_objects,
                state_labels=["s1", "s2"],
                energy_labels=["e1", "e1"],
                polarization_coefficients=[1, 1],
                beam_labels=["b1", "b2"],
            )

    def test_state_energy_beam_projection_can_be_limited_to_support(self):
        support = np.zeros((3, 4), dtype=bool)
        support[1:, 1:3] = True
        common = np.full((3, 4), 0.1 + 0.02j)
        response = np.full((3, 4), 0.04 - 0.03j)
        log_objects = np.stack([
            common + response,
            common - response,
            common + response,
        ])
        log_objects[:, ~support] += np.asarray([0.3j, 0.5, -0.2j])[:, None]

        projected, components = general.project_log_objects_general(
            log_objects,
            state_labels=["state", "state", "state"],
            energy_labels=["energy", "energy", "energy"],
            polarization_coefficients=[1, -1, 1],
            beam_labels=["beam_1", "beam_1", "beam_2"],
            projection_supportmask=support,
            return_components=True,
        )

        np.testing.assert_allclose(projected[:, ~support], log_objects[:, ~support])
        self.assertTrue(components["projection_supportmask_applied"])

    def test_response_to_refractive_index(self):
        response = np.full((2, 2, 3), 2j)
        refractive_index = general.response_to_refractive_index(
            response,
            wave_numbers=[2.0, 4.0],
            thickness=0.5,
            response_energy_indices=[0, 1],
        )
        np.testing.assert_allclose(refractive_index[0], 2.0)
        np.testing.assert_allclose(refractive_index[1], 1.0)

    def test_physical_model_recovers_charge_magnetic_and_magnetization(self):
        shape = (3, 4)
        common = {
            "b1": np.full(shape, 0.2 + 0.1j),
            "b2": np.full(shape, -0.1 + 0.04j),
        }
        charge = {"e1": -0.02 + 0.01j, "e2": 0.02 - 0.01j}
        magnetic = {"e1": -0.03 + 0.06j, "e2": -0.05 + 0.08j}
        mz = {
            "sat": np.ones(shape),
            "domains": np.linspace(-0.8, 0.8, np.prod(shape)).reshape(shape),
        }
        observations = [
            (state, energy, polarization, beam)
            for beam in ("b1", "b2")
            for energy in ("e1", "e2")
            for state in ("sat", "domains")
            for polarization in (1, -1)
        ]
        log_objects = np.stack([
            common[beam]
            + charge[energy]
            + polarization * magnetic[energy] * mz[state]
            for state, energy, polarization, beam in observations
        ])

        projected, components = general.project_log_objects_physical(
            log_objects,
            state_labels=[item[0] for item in observations],
            energy_labels=[item[1] for item in observations],
            polarization_coefficients=[item[2] for item in observations],
            beam_labels=[item[3] for item in observations],
            saturated_states={"sat": 1},
            iterations=30,
            return_components=True,
        )

        np.testing.assert_allclose(projected, log_objects, atol=1e-8)
        np.testing.assert_allclose(
            components["charge_response"],
            [charge["e1"], charge["e2"]],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            components["magnetic_response"],
            [magnetic["e1"], magnetic["e2"]],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            components["magnetization_by_state"]["domains"],
            mz["domains"],
            atol=1e-8,
        )
        self.assertLessEqual(np.max(np.abs(components["magnetization"])), 1.0)

    def test_physical_model_can_zero_magnetization_outside_support(self):
        support = np.zeros((3, 4), dtype=bool)
        support[1:, 1:3] = True
        common = np.full((3, 4), 0.1 + 0.02j)
        magnetic = 0.04 - 0.03j
        mz_domains = np.zeros((3, 4), dtype=float)
        mz_domains[support] = np.linspace(-0.8, 0.8, support.sum())
        mz_sat = support.astype(float)
        observations = [
            ("sat", 1, mz_sat),
            ("sat", -1, mz_sat),
            ("domains", 1, mz_domains),
            ("domains", -1, mz_domains),
        ]
        log_objects = np.stack([
            common + polarization * magnetic * mz
            for _, polarization, mz in observations
        ])

        projected, components = general.project_log_objects_physical(
            log_objects,
            state_labels=[item[0] for item in observations],
            energy_labels=["energy"] * len(observations),
            polarization_coefficients=[item[1] for item in observations],
            beam_labels=["beam"] * len(observations),
            saturated_states={"sat": 1},
            iterations=20,
            magnetization_supportmask=support,
            return_components=True,
        )

        np.testing.assert_allclose(projected, log_objects, atol=1e-10)
        self.assertTrue(components["magnetization_supportmask_applied"])
        self.assertTrue(
            np.all(
                components["magnetization_by_state"]["domains"][~support] == 0
            )
        )
        self.assertTrue(
            np.all(components["magnetization_by_state"]["sat"][~support] == 0)
        )
        np.testing.assert_allclose(
            components["magnetization_by_state"]["domains"][support],
            mz_domains[support],
            atol=1e-10,
        )

    def test_physical_constraints_can_be_limited_to_support(self):
        support = np.zeros((3, 4), dtype=bool)
        support[1:, 1:3] = True
        common = np.full((3, 4), 0.1 + 0.02j)
        magnetic = 0.04 - 0.03j
        mz = np.zeros((3, 4), dtype=float)
        mz[support] = np.linspace(-0.8, 0.8, support.sum())
        observations = [
            ("domains", 1, mz),
            ("domains", -1, mz),
        ]
        log_objects = np.stack([
            common + polarization * magnetic * mz
            for _, polarization, mz in observations
        ])
        log_objects[:, ~support] += np.asarray([0.7 - 0.2j, -0.4 + 0.5j])[:, None]

        projected, components = general.project_log_objects_physical(
            log_objects,
            state_labels=[item[0] for item in observations],
            energy_labels=["energy"] * len(observations),
            polarization_coefficients=[item[1] for item in observations],
            beam_labels=["beam"] * len(observations),
            iterations=20,
            projection_supportmask=support,
            return_components=True,
        )

        np.testing.assert_allclose(projected[:, ~support], log_objects[:, ~support])
        self.assertTrue(components["physical_projection_supportmask_applied"])

    def test_driver_shifts_supportmask_for_magnetization_projection(self):
        support = np.zeros((4, 6), dtype=bool)
        support[:2, :3] = True
        object_support = np.fft.fftshift(support)
        common = np.full((4, 6), 0.1 + 0.02j)
        magnetic = 0.04 - 0.03j
        mz_domains = np.zeros((4, 6), dtype=float)
        mz_domains[object_support] = np.linspace(
            -0.8,
            0.8,
            object_support.sum(),
        )
        mz_sat = object_support.astype(float)
        observations = [
            ("sat", 1, mz_sat),
            ("sat", -1, mz_sat),
            ("domains", 1, mz_domains),
            ("domains", -1, mz_domains),
        ]
        log_objects = np.stack([
            common + polarization * magnetic * mz
            for _, polarization, mz in observations
        ])
        fields = general.core.object_log_to_fourier_field(log_objects)

        def identity_core(*args, **kwargs):
            return kwargs["Phase"], [], [], None

        with mock.patch.object(general, "PhaseRtrv_core", side_effect=identity_core):
            _, components, _, _ = general.general_phase_retrieval_algorithm(
                np.abs(fields) ** 2,
                np.zeros_like(support),
                support,
                state_labels=[item[0] for item in observations],
                energy_labels=["energy"] * len(observations),
                polarization_coefficients=[item[1] for item in observations],
                beam_labels=["beam"] * len(observations),
                saturated_states={"sat": 1},
                general_recipe={
                    "warmup_Nit": 0,
                    "inner_Nit": [1],
                    "outer_iterations": 1,
                    "projection_every": 1,
                    "projection_start": 1,
                    "zero_magnetization_outside_support": True,
                    "final_fourier_constraint": False,
                    "physical_iterations": 20,
                },
                start_fields=fields,
            )

        recovered = components["magnetization_by_state"]["domains"]
        np.testing.assert_allclose(
            recovered[object_support],
            mz_domains[object_support],
            atol=1e-10,
        )
        self.assertTrue(np.all(recovered[~object_support] == 0))

    def test_physical_model_applies_response_bounds(self):
        shape = (2, 2)
        common = np.zeros(shape, dtype=complex)
        magnetic = -0.5 + 0.8j
        log_objects = np.stack([
            common + magnetic,
            common - magnetic,
        ])
        recipe = general.default_general_phase_retrieval_recipe()
        recipe.update({
            "magnetic_response_real_range": (-0.04, -0.02),
            "magnetic_response_imag_range": (0.05, 0.07),
        })

        _, components = general.project_log_objects_physical(
            log_objects,
            state_labels=["sat", "sat"],
            energy_labels=["e", "e"],
            polarization_coefficients=[1, -1],
            beam_labels=["b", "b"],
            saturated_states={"sat": 1},
            recipe=recipe,
            return_components=True,
        )

        response = components["magnetic_response"][0]
        self.assertGreaterEqual(response.real, -0.04)
        self.assertLessEqual(response.real, -0.02)
        self.assertGreaterEqual(response.imag, 0.05)
        self.assertLessEqual(response.imag, 0.07)

    def test_physical_model_applies_known_magnetic_spectrum(self):
        shape = (2, 3)
        known_beta = np.array([-0.04, -0.03, -0.02])
        dispersion = np.array([0.02, 0.04, 0.01])
        magnetic = known_beta + 1j * dispersion
        observations = [
            (energy, polarization)
            for energy in range(3)
            for polarization in (1, -1)
        ]
        log_objects = np.stack([
            polarization * magnetic[energy] * np.ones(shape)
            for energy, polarization in observations
        ])
        recipe = general.default_general_phase_retrieval_recipe()
        recipe.update({
            "magnetic_spectral_constraint": "known_beta",
            "known_magnetic_beta_spectrum": known_beta,
            "fit_known_spectrum_scale": False,
        })

        _, components = general.project_log_objects_physical(
            log_objects,
            state_labels=["sat"] * len(observations),
            energy_labels=[item[0] for item in observations],
            polarization_coefficients=[item[1] for item in observations],
            beam_labels=["beam"] * len(observations),
            saturated_states={"sat": 1},
            recipe=recipe,
            return_components=True,
        )

        np.testing.assert_allclose(
            np.real(components["magnetic_response"]),
            known_beta,
            atol=1e-12,
        )

    def test_driver_runs_and_returns_general_components(self):
        shape = (4, 4)
        start_fields = np.ones((2, *shape), dtype=complex)
        holograms = np.abs(start_fields) ** 2

        def fake_core(*, Phase, **kwargs):
            return Phase, np.array([]), np.array([]), None

        with mock.patch.object(
            general,
            "PhaseRtrv_core",
            side_effect=fake_core,
        ):
            fields, components, _, _ = (
                general.general_phase_retrieval_algorithm(
                    holograms,
                    np.zeros(shape, dtype=int),
                    np.ones(shape),
                    state_labels=["state", "state"],
                    energy_labels=["energy", "energy"],
                    polarization_coefficients=[1, -1],
                    beam_labels=["beam", "beam"],
                    general_recipe={
                        "outer_iterations": 1,
                        "warmup_Nit": 0,
                        "shuffle_observations": False,
                        "final_fourier_constraint": False,
                    },
                    start_fields=start_fields,
                )
            )

        self.assertEqual(fields.shape, start_fields.shape)
        self.assertEqual(
            components["projection_model"],
            "physical_factorized",
        )
        self.assertIn("common_exit_waves", components)
        self.assertFalse(components["magnetic_scale_anchored"])


if __name__ == "__main__":
    unittest.main()
