import unittest
from unittest import mock

import numpy as np

from library import phase_retrieval_core_multienergy as multi


class PhaseRetrievalCoreMultienergyScheduleTests(unittest.TestCase):
    def test_schedule_validation_accepts_lists_and_scalar_shorthand(self):
        self.assertEqual(
            multi._normalize_update_schedule(
                ["HAPRE", "ER"],
                [700, 50],
                name="inner",
            ),
            [("HAPRE", 700), ("ER", 50)],
        )
        self.assertEqual(
            multi._normalize_update_schedule("ER", 10, name="inner"),
            [("ER", 10)],
        )
        self.assertEqual(
            multi._normalize_update_schedule(
                "HAPRE",
                0,
                name="warmup",
                allow_disabled=True,
            ),
            [],
        )

    def test_schedule_validation_rejects_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            multi._normalize_update_schedule(
                ["HAPRE", "ER"],
                [10],
                name="inner",
            )

    def test_stage_specific_controls_are_forwarded_to_phase_retrieval_core(self):
        holograms = np.ones((2, 4, 4))
        captured = []

        def fake_core(
            *,
            mode,
            Nit,
            beta_zero,
            beta_mode,
            alpha_zero,
            alpha_mode,
            TV_freq,
            Phase,
            **kwargs,
        ):
            captured.append(
                {
                    "mode": mode,
                    "Nit": Nit,
                    "beta_zero": beta_zero,
                    "beta_mode": beta_mode,
                    "alpha_zero": alpha_zero,
                    "alpha_mode": alpha_mode,
                    "TV_freq": TV_freq,
                }
            )
            return Phase, np.array([]), np.array([]), None

        recipe = {
            "inner_mode": ["HAPRE", "ER"],
            "inner_Nit": [7, 5],
            "beta_zero": [0.4, 0.9],
            "beta_mode": ["arctan", "const"],
            "alpha_zero": [0.1, 0.0],
            "alpha_mode": ["smoothstep", "const"],
            "TV_freq": [4, 1e9],
            "outer_iterations": 1,
            "warmup_Nit": 0,
            "shuffle_energies": False,
            "projection_model": "none",
            "final_fourier_constraint": False,
        }

        with mock.patch.object(multi, "PhaseRtrv_core", side_effect=fake_core):
            multi.multi_energy_phase_retrieval_algorithm(
                holograms,
                np.zeros((4, 4), dtype=int),
                np.ones((4, 4)),
                multi_energy_recipe=recipe,
                start_fields=np.ones_like(holograms, dtype=complex),
            )

        self.assertEqual(
            captured,
            2 * [
                {
                    "mode": "HAPRE",
                    "Nit": 7,
                    "beta_zero": 0.4,
                    "beta_mode": "arctan",
                    "alpha_zero": 0.1,
                    "alpha_mode": "smoothstep",
                    "TV_freq": 4,
                },
                {
                    "mode": "ER",
                    "Nit": 5,
                    "beta_zero": 0.9,
                    "beta_mode": "const",
                    "alpha_zero": 0.0,
                    "alpha_mode": "const",
                    "TV_freq": 1e9,
                },
            ],
        )

    def test_warmup_and_joint_schedules_run_in_order_and_handoff_fields(self):
        n_energy = 2
        holograms = np.ones((n_energy, 4, 4))
        support = np.ones((4, 4))
        start_fields = np.zeros_like(holograms, dtype=complex)
        calls = []

        def fake_core(*, mode, Nit, Phase, **kwargs):
            input_level = float(np.real(Phase[0, 0]))
            calls.append((mode, Nit, input_level))
            output = np.asarray(Phase, dtype=complex) + 1
            return output, np.array([Nit]), np.array([Nit]), None

        recipe = {
            "inner_mode": ["HAPRE", "ER"],
            "inner_Nit": [7, 5],
            "outer_iterations": 2,
            "warmup_mode": ["ER", "HAPRE"],
            "warmup_Nit": [2, 3],
            "shuffle_energies": False,
            "projection_model": "none",
            "final_fourier_constraint": False,
        }

        with mock.patch.object(multi, "PhaseRtrv_core", side_effect=fake_core):
            retrieved, _, _, errors = multi.multi_energy_phase_retrieval_algorithm(
                holograms,
                np.zeros((4, 4), dtype=int),
                support,
                multi_energy_recipe=recipe,
                start_fields=start_fields,
            )

        expected_per_energy = [
            ("ER", 2),
            ("HAPRE", 3),
            ("HAPRE", 7),
            ("ER", 5),
            ("HAPRE", 7),
            ("ER", 5),
        ]
        expected_calls = [
            ("ER", 2, 0.0),
            ("HAPRE", 3, 1.0),
            ("ER", 2, 0.0),
            ("HAPRE", 3, 1.0),
            ("HAPRE", 7, 2.0),
            ("ER", 5, 3.0),
            ("HAPRE", 7, 2.0),
            ("ER", 5, 3.0),
            ("HAPRE", 7, 4.0),
            ("ER", 5, 5.0),
            ("HAPRE", 7, 4.0),
            ("ER", 5, 5.0),
        ]
        self.assertEqual(calls, expected_calls)
        np.testing.assert_allclose(retrieved, 6)

        recorded = errors["energy_steps"]
        for energy in range(n_energy):
            actual = [
                (step["mode"], step["Nit"])
                for step in recorded
                if step["energy"] == energy
            ]
            self.assertEqual(actual, expected_per_energy)

    def test_scalar_and_one_stage_list_schedules_match(self):
        rng = np.random.default_rng(4)
        holograms = rng.uniform(0.5, 2.0, (3, 6, 6))
        support = np.ones((6, 6))
        masks = np.zeros((3, 6, 6), dtype=int)
        start_fields = np.sqrt(holograms) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, holograms.shape)
        )
        shared = {
            "outer_iterations": 2,
            "shuffle_energies": False,
            "projection_model": "none",
            "plot_every": 2,
            "average_img": 1,
        }

        expected, _, _, _ = multi.multi_energy_phase_retrieval_algorithm(
            holograms,
            masks,
            support,
            multi_energy_recipe={
                **shared,
                "inner_mode": "ER",
                "inner_Nit": 3,
                "warmup_Nit": 0,
            },
            start_fields=start_fields,
        )
        retrieved, _, _, _ = multi.multi_energy_phase_retrieval_algorithm(
            holograms,
            masks,
            support,
            multi_energy_recipe={
                **shared,
                "inner_mode": ["ER"],
                "inner_Nit": [3],
                "warmup_Nit": 0,
            },
            start_fields=start_fields,
        )

        np.testing.assert_allclose(retrieved, expected, atol=1e-12)

    def test_projection_every_counts_completed_energy_updates(self):
        holograms = np.ones((3, 4, 4))

        def fake_core(*, Phase, **kwargs):
            return Phase, np.array([]), np.array([]), None

        def fake_projection(fields, **kwargs):
            return fields, {"projection_model": kwargs["projection_model"]}

        for cadence, projection_start, expected_energies, expected_updates in (
            (1, 0, [0, 1, 2, 0, 1, 2], [1, 2, 3, 4, 5, 6]),
            (3, 0, [2, 2], [3, 6]),
            (None, 0, [2, 2], [3, 6]),
            (1, 3, [2, 0, 1, 2], [3, 4, 5, 6]),
            (2, None, [1, 0, 2], [2, 4, 6]),
            (None, None, [2, 2], [3, 6]),
        ):
            with self.subTest(
                projection_every=cadence,
                projection_start=projection_start,
            ), mock.patch.object(
                multi,
                "PhaseRtrv_core",
                side_effect=fake_core,
            ), mock.patch.object(
                multi,
                "project_fourier_fields_multi_energy",
                side_effect=fake_projection,
            ):
                _, _, _, errors = multi.multi_energy_phase_retrieval_algorithm(
                    holograms,
                    np.zeros((4, 4), dtype=int),
                    np.ones((4, 4)),
                    multi_energy_recipe={
                        "inner_mode": "ER",
                        "inner_Nit": 1,
                        "outer_iterations": 2,
                        "warmup_Nit": 0,
                        "shuffle_energies": False,
                        "projection_every": cadence,
                        "projection_start": projection_start,
                        "final_fourier_constraint": False,
                    },
                    start_fields=np.ones_like(holograms, dtype=complex),
                )

            steps = errors["projection_steps"]
            self.assertEqual(
                [step["energy"] for step in steps],
                expected_energies,
            )
            self.assertEqual(
                [step["completed_update"] for step in steps],
                expected_updates,
            )


if __name__ == "__main__":
    unittest.main()
