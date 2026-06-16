import unittest
from unittest import mock

import numpy as np

from library import phase_retrieval_core_dichroic as dichroic


class PhaseRetrievalCoreDichroicTests(unittest.TestCase):
    def test_paired_state_recovers_charge_and_magnetic_components(self):
        rng = np.random.default_rng(41)
        charge = (
            rng.normal(scale=0.1, size=(5, 6))
            + 1j * rng.normal(scale=0.1, size=(5, 6))
        )
        magnetic = (
            rng.normal(scale=0.05, size=(5, 6))
            + 1j * rng.normal(scale=0.05, size=(5, 6))
        )
        log_objects = np.stack([charge + magnetic, charge - magnetic])

        projected, components = dichroic.project_log_objects_dichroic(
            log_objects,
            state_labels=["state", "state"],
            polarization_signs=[1, -1],
            return_components=True,
        )

        np.testing.assert_allclose(projected, log_objects, atol=1e-12)
        np.testing.assert_allclose(
            components["charge_log_object"],
            charge,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            components["magnetic_log_objects_by_state"]["state"],
            magnetic,
            atol=1e-12,
        )
        self.assertTrue(components["identifiable"])

    def test_one_opposite_partner_anchors_multiple_positive_states(self):
        rng = np.random.default_rng(42)
        charge = rng.normal(scale=0.1, size=(4, 5)).astype(complex)
        magnetic = {
            state: (
                rng.normal(scale=0.05, size=(4, 5))
                + 1j * rng.normal(scale=0.05, size=(4, 5))
            )
            for state in ("a", "b", "c")
        }
        labels = ["a", "a", "b", "c"]
        signs = [1, -1, 1, 1]
        log_objects = np.stack(
            [
                charge + magnetic["a"],
                charge - magnetic["a"],
                charge + magnetic["b"],
                charge + magnetic["c"],
            ]
        )

        _, components = dichroic.project_log_objects_dichroic(
            log_objects,
            labels,
            signs,
            return_components=True,
        )

        np.testing.assert_allclose(
            components["charge_log_object"],
            charge,
            atol=1e-12,
        )
        for state in ("a", "b", "c"):
            np.testing.assert_allclose(
                components["magnetic_log_objects_by_state"][state],
                magnetic[state],
                atol=1e-12,
            )
        self.assertEqual(components["design_rank"], 4)

    def test_same_polarization_only_is_rejected_as_nonidentifiable(self):
        log_objects = np.zeros((2, 4, 4), dtype=complex)

        with self.assertRaisesRegex(ValueError, "rank deficient"):
            dichroic.project_log_objects_dichroic(
                log_objects,
                state_labels=["domains", "uniform"],
                polarization_signs=[1, 1],
            )

    def test_saturated_reference_recovers_response_and_real_magnetization(self):
        rng = np.random.default_rng(44)
        charge = (
            rng.normal(scale=0.1, size=(4, 5))
            + 1j * rng.normal(scale=0.1, size=(4, 5))
        )
        magnetization = rng.uniform(-1, 1, size=(4, 5))
        response = (
            rng.uniform(-0.05, -0.02, size=(4, 5))
            + 1j * rng.uniform(0.04, 0.08, size=(4, 5))
        )
        log_objects = np.stack(
            [
                charge + response,
                charge - response,
                charge + response * magnetization,
                charge - response * magnetization,
            ]
        )

        projected, components = (
            dichroic.project_log_objects_saturated_reference(
                log_objects,
                state_labels=["saturated", "saturated", "domains", "domains"],
                polarization_signs=[1, -1, 1, -1],
                saturated_states=["saturated"],
                return_components=True,
            )
        )

        np.testing.assert_allclose(projected, log_objects, atol=1e-12)
        np.testing.assert_allclose(
            components["charge_log_object"],
            charge,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            components["magnetic_response"],
            response,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            components["magnetization_by_state"]["domains"],
            magnetization,
            atol=1e-12,
        )
        self.assertFalse(components["physical_response_bounds_applied"])

    def test_saturated_reference_can_zero_magnetization_outside_support(self):
        support = np.zeros((3, 4), dtype=bool)
        support[1:, 1:3] = True
        charge = np.full((3, 4), 0.1 + 0.02j)
        response = 0.04 - 0.03j
        mz_domains = np.zeros((3, 4), dtype=float)
        mz_domains[support] = np.linspace(-0.8, 0.8, support.sum())
        mz_sat = support.astype(float)
        log_objects = np.stack([
            charge + response * mz_sat,
            charge - response * mz_sat,
            charge + response * mz_domains,
            charge - response * mz_domains,
        ])

        projected, components = (
            dichroic.project_log_objects_saturated_reference(
                log_objects,
                state_labels=["sat", "sat", "domains", "domains"],
                polarization_signs=[1, -1, 1, -1],
                saturated_states={"sat": 1},
                magnetization_supportmask=support,
                return_components=True,
            )
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

    def test_saturated_reference_constraints_can_be_limited_to_support(self):
        support = np.zeros((3, 4), dtype=bool)
        support[1:, 1:3] = True
        charge = np.full((3, 4), 0.1 + 0.02j)
        response = 0.04 - 0.03j
        mz_domains = np.zeros((3, 4), dtype=float)
        mz_domains[support] = np.linspace(-0.8, 0.8, support.sum())
        mz_sat = support.astype(float)
        log_objects = np.stack([
            charge + response * mz_sat,
            charge - response * mz_sat,
            charge + response * mz_domains,
            charge - response * mz_domains,
        ])
        log_objects[:, ~support] += np.asarray([
            0.7 - 0.2j,
            -0.4 + 0.5j,
            0.3 + 0.1j,
            -0.2 - 0.6j,
        ])[:, None]

        projected, components = (
            dichroic.project_log_objects_saturated_reference(
                log_objects,
                state_labels=["sat", "sat", "domains", "domains"],
                polarization_signs=[1, -1, 1, -1],
                saturated_states={"sat": 1},
                projection_supportmask=support,
                return_components=True,
            )
        )

        np.testing.assert_allclose(projected[:, ~support], log_objects[:, ~support])
        self.assertTrue(components["physical_projection_supportmask_applied"])

    def test_driver_shifts_supportmask_for_magnetization_projection(self):
        support = np.zeros((4, 6), dtype=bool)
        support[:2, :3] = True
        object_support = np.fft.fftshift(support)
        charge = np.full((4, 6), 0.1 + 0.02j)
        response = 0.04 - 0.03j
        mz_domains = np.zeros((4, 6), dtype=float)
        mz_domains[object_support] = np.linspace(
            -0.8,
            0.8,
            object_support.sum(),
        )
        mz_sat = object_support.astype(float)
        log_objects = np.stack([
            charge + response * mz_sat,
            charge - response * mz_sat,
            charge + response * mz_domains,
            charge - response * mz_domains,
        ])
        fields = dichroic.core.object_log_to_fourier_field(log_objects)

        def identity_core(*args, **kwargs):
            return kwargs["Phase"], [], [], None

        with mock.patch.object(
            dichroic,
            "PhaseRtrv_core",
            side_effect=identity_core,
        ):
            _, components, _, _ = dichroic.dichroic_phase_retrieval_algorithm(
                np.abs(fields) ** 2,
                np.zeros_like(support),
                support,
                state_labels=["sat", "sat", "domains", "domains"],
                polarization_signs=[1, -1, 1, -1],
                saturated_states={"sat": 1},
                dichroic_recipe={
                    "warmup_Nit": 0,
                    "inner_Nit": [1],
                    "outer_iterations": 1,
                    "projection_every": 1,
                    "projection_start": 1,
                    "zero_magnetization_outside_support": True,
                    "final_fourier_constraint": False,
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

    def test_optional_beta_range_constrains_attenuation_only(self):
        charge = np.full((3, 4), 0.02 + 0.01j)
        response = np.full((3, 4), -0.5 + 0.8j)
        log_objects = np.stack([charge + response, charge - response])
        kt_beta_range = (0.01, 0.02)

        projected, components = (
            dichroic.project_log_objects_saturated_reference(
                log_objects,
                state_labels=["saturated", "saturated"],
                polarization_signs=[1, -1],
                saturated_states=["saturated"],
                kt_beta_m_range=kt_beta_range,
                return_components=True,
            )
        )

        expected_response = np.full(
            (3, 4),
            -kt_beta_range[1] + 0.8j,
        )
        np.testing.assert_allclose(
            components["magnetic_response_unconstrained"],
            response,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            components["magnetic_response"],
            expected_response,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            projected,
            np.stack(
                [
                    charge + expected_response,
                    charge - expected_response,
                ]
            ),
            atol=1e-12,
        )
        self.assertTrue(components["physical_response_bounds_applied"])
        self.assertNotIn(
            "magnetic_phase_shift",
            components["response_bounds"],
        )

    def test_optional_delta_range_constrains_phase_only(self):
        response = np.full((2, 2), -0.5 + 0.8j)
        constrained, info = dichroic._constrain_magnetic_response(
            response,
            kt_delta_m_range=(0.1, 0.2),
        )

        np.testing.assert_allclose(constrained, -0.5 + 0.2j)
        self.assertEqual(
            info["response_bounds"]["magnetic_phase_shift"],
            (0.1, 0.2),
        )
        self.assertNotIn(
            "magnetic_log_attenuation",
            info["response_bounds"],
        )

    def test_product_ranges_are_validated_directly(self):
        response = np.ones((2, 2), dtype=complex)

        with self.assertRaisesRegex(ValueError, "kt_beta_m_range"):
            dichroic._constrain_magnetic_response(
                response,
                kt_beta_m_range=(0.02, 0.01),
            )

        with self.assertRaisesRegex(ValueError, "kt_delta_m_range"):
            dichroic._constrain_magnetic_response(
                response,
                kt_delta_m_range=(0.01, np.inf),
            )

    def test_shared_charge_bounds_response_and_reduced_magnetization(self):
        charge = np.full((3, 4), 0.02 + 0.01j)
        response = np.full((3, 4), -0.03 + 0.06j)
        mz_a = np.full((3, 4), 0.8)
        mz_b = np.full((3, 4), -0.4)
        log_objects = np.stack([
            charge + response * mz_a,
            charge - response * mz_a,
            charge + response * mz_b,
        ])

        projected, components = dichroic.project_log_objects_dichroic(
            log_objects,
            state_labels=["a", "a", "b"],
            polarization_signs=[1, -1, 1],
            kt_delta_m_range=(0.05, 0.07),
            kt_beta_m_range=(0.02, 0.04),
            return_components=True,
        )

        np.testing.assert_allclose(projected, log_objects, atol=1e-10)
        self.assertTrue(components["physical_response_bounds_applied"])
        self.assertLessEqual(np.max(np.abs(components["magnetization"])), 1.0)
        self.assertTrue(
            np.all(
                (components["magnetic_phase_shift"] >= 0.05)
                & (components["magnetic_phase_shift"] <= 0.07)
            )
        )
        self.assertTrue(
            np.all(
                (components["magnetic_log_attenuation"] >= 0.02)
                & (components["magnetic_log_attenuation"] <= 0.04)
            )
        )

    def test_saturated_reference_accepts_negative_saturation_flag(self):
        rng = np.random.default_rng(45)
        charge = (
            rng.normal(scale=0.1, size=(4, 5))
            + 1j * rng.normal(scale=0.1, size=(4, 5))
        )
        response = -0.03 + 0.06j
        log_objects = np.stack(
            [
                charge - response,
                charge + response,
            ]
        )

        projected, components = (
            dichroic.project_log_objects_saturated_reference(
                log_objects,
                state_labels=["saturated_down", "saturated_down"],
                polarization_signs=[1, -1],
                saturated_states={"saturated_down": -1},
                return_components=True,
            )
        )

        np.testing.assert_allclose(projected, log_objects, atol=1e-12)
        np.testing.assert_allclose(
            components["charge_log_object"],
            charge,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            components["magnetic_response"],
            response,
            atol=1e-12,
        )

    def test_same_polarization_saturated_pair_remains_nonidentifiable(self):
        log_objects = np.zeros((2, 4, 4), dtype=complex)

        with self.assertRaisesRegex(ValueError, "rank deficient"):
            dichroic.project_log_objects_saturated_reference(
                log_objects,
                state_labels=["domains", "saturated"],
                polarization_signs=[1, 1],
                saturated_states=["saturated"],
            )

    def test_fourier_projection_preserves_exact_physical_model(self):
        rng = np.random.default_rng(43)
        charge = (
            rng.normal(scale=0.02, size=(5, 5))
            + 1j * rng.normal(scale=0.02, size=(5, 5))
        )
        magnetic = (
            rng.normal(scale=0.01, size=(5, 5))
            + 1j * rng.normal(scale=0.01, size=(5, 5))
        )
        log_objects = np.stack([charge + magnetic, charge - magnetic])
        fields = np.empty_like(log_objects)
        for index in range(2):
            fields[index] = np.fft.ifftshift(
                np.fft.ifft2(np.exp(log_objects[index]))
            )

        projected = dichroic.project_fourier_fields_dichroic(
            fields,
            state_labels=["state", "state"],
            polarization_signs=[1, -1],
        )

        np.testing.assert_allclose(projected, fields, atol=1e-12)

    def test_driver_saturation_flag_selects_blind_response_projection(self):
        charge = np.full((4, 4), 0.02 + 0.01j)
        response = np.full((4, 4), -0.03 + 0.06j)
        log_objects = np.stack([charge + response, charge - response])
        start_fields = np.stack(
            [
                np.fft.ifftshift(np.fft.ifft2(np.exp(log_object)))
                for log_object in log_objects
            ]
        )
        holograms = np.abs(start_fields) ** 2

        def fake_core(*, Phase, **kwargs):
            return Phase, np.array([]), np.array([]), None

        with mock.patch.object(
            dichroic,
            "PhaseRtrv_core",
            side_effect=fake_core,
        ):
            _, components, _, _ = dichroic.dichroic_phase_retrieval_algorithm(
                holograms,
                np.zeros((4, 4), dtype=int),
                np.ones((4, 4)),
                state_labels=["saturated", "saturated"],
                polarization_signs=[1, -1],
                saturated_states=["saturated"],
                dichroic_recipe={
                    "outer_iterations": 1,
                    "warmup_Nit": 0,
                    "shuffle_observations": False,
                    "final_fourier_constraint": False,
                },
                start_fields=start_fields,
            )

        self.assertEqual(components["projection_model"], "saturated_reference")
        np.testing.assert_allclose(
            components["magnetic_response"],
            response,
            atol=1e-12,
        )

    def test_driver_runs_stage_schedule_for_every_observation(self):
        holograms = np.ones((2, 4, 4))
        start_fields = np.zeros_like(holograms, dtype=complex)
        calls = []

        def fake_core(*, mode, Nit, Phase, beta_mode, TV_freq, **kwargs):
            calls.append(
                (
                    mode,
                    Nit,
                    beta_mode,
                    TV_freq,
                    float(np.real(Phase[0, 0])),
                )
            )
            return Phase + 1, np.array([Nit]), np.array([Nit]), None

        recipe = {
            "inner_mode": ["HAPRE", "ER"],
            "inner_Nit": [7, 5],
            "beta_mode": ["arctan", "const"],
            "TV_freq": [4, 1e9],
            "outer_iterations": 1,
            "warmup_Nit": 0,
            "shuffle_observations": False,
            "projection_model": "none",
            "final_fourier_constraint": False,
        }

        with mock.patch.object(
            dichroic,
            "PhaseRtrv_core",
            side_effect=fake_core,
        ):
            result, _, _, errors = dichroic.dichroic_phase_retrieval_algorithm(
                holograms,
                np.zeros((4, 4), dtype=int),
                np.ones((4, 4)),
                state_labels=["state", "state"],
                polarization_signs=[1, -1],
                dichroic_recipe=recipe,
                start_fields=start_fields,
            )

        self.assertEqual(
            calls,
            2
            * [
                ("HAPRE", 7, "arctan", 4, 0.0),
                ("ER", 5, "const", 1e9, 1.0),
            ],
        )
        np.testing.assert_allclose(result, 2)
        self.assertEqual(len(errors["observation_steps"]), 4)

    def test_projection_every_counts_completed_observation_updates(self):
        holograms = np.ones((2, 4, 4))

        def fake_core(*, Phase, **kwargs):
            return Phase, np.array([]), np.array([]), None

        def fake_projection(fields, *args, **kwargs):
            return fields, {
                "projection_model": "shared_charge",
                "identifiable": True,
                "fit_residual_rms": 0.0,
            }

        for cadence, projection_start, expected_observations, expected_updates in (
            (1, 0, [0, 1, 0, 1], [1, 2, 3, 4]),
            (2, 0, [1, 1], [2, 4]),
            (None, 0, [1, 1], [2, 4]),
            (1, 2, [1, 0, 1], [2, 3, 4]),
            (None, None, [1, 1], [2, 4]),
        ):
            with self.subTest(
                projection_every=cadence,
                projection_start=projection_start,
            ), mock.patch.object(
                dichroic,
                "PhaseRtrv_core",
                side_effect=fake_core,
            ), mock.patch.object(
                dichroic,
                "_project_fields",
                side_effect=fake_projection,
            ):
                _, _, _, errors = dichroic.dichroic_phase_retrieval_algorithm(
                    holograms,
                    np.zeros((4, 4), dtype=int),
                    np.ones((4, 4)),
                    state_labels=["state", "state"],
                    polarization_signs=[1, -1],
                    dichroic_recipe={
                        "inner_mode": "ER",
                        "inner_Nit": 1,
                        "outer_iterations": 2,
                        "warmup_Nit": 0,
                        "shuffle_observations": False,
                        "projection_every": cadence,
                        "projection_start": projection_start,
                        "final_fourier_constraint": False,
                    },
                    start_fields=np.ones_like(holograms, dtype=complex),
                )

            steps = errors["projection_steps"]
            self.assertEqual(
                [step["observation"] for step in steps],
                expected_observations,
            )
            self.assertEqual(
                [step["completed_update"] for step in steps],
                expected_updates,
            )


if __name__ == "__main__":
    unittest.main()
