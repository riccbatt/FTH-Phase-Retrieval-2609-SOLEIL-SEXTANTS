import unittest
from unittest import mock

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

    def test_gradient_descent_recipe_stage_uses_beta_as_learning_rate(self):
        data = np.ones((4, 4), dtype=float)
        support = np.ones((4, 4), dtype=float)
        start = np.ones((4, 4), dtype=complex)
        refined_field = 2 * start
        recipe = {
            "algorithm_list": ["gradient_descent"],
            "number_iterations": [3],
            "helicity": ["pos"],
            "beta_zero": [0.2],
            "beta_mode": ["const"],
            "alpha_zero": [0.7],
            "alpha_mode": ["const"],
            "RL_its": [0],
            "RL_freqs": [10],
            "TV_freqs": [10],
            "plot_every": [1],
            "average_img": [1],
            "Fourier_last": [False],
            "hologram_intensity_cutoff_vmin": -1,
            "Startimage": [start],
            "Startgamma": [None],
        }

        fake_result = core.gradient.GradientRefinementResult(
            fields=refined_field,
            loss=np.array([3.0, 2.0, 1.0]),
            diffraction_loss=np.array([0.3, 0.2, 0.1]),
            support_loss=np.array([0.03, 0.02, 0.01]),
        )

        with mock.patch.object(
            core.gradient,
            "refine_field_gradient",
            return_value=fake_result,
        ) as refine:
            result = core.phase_retrieval_algorithm(
                data,
                data,
                np.zeros_like(data, dtype=int),
                support,
                phase_retrieval_recipe=recipe,
            )

        np.testing.assert_allclose(result[0], refined_field)
        call = refine.call_args.kwargs
        self.assertEqual(call["n_steps"], 3)
        np.testing.assert_allclose(call["learning_rate"], [0.2, 0.2, 0.2])
        np.testing.assert_allclose(call["support_weight"], [0.7, 0.7, 0.7])
        self.assertIs(call["supportmask"], support)
        np.testing.assert_allclose(result[-1]["steps"][0]["error"], [0.3, 0.2, 0.1])
        np.testing.assert_allclose(
            result[-1]["steps"][0]["support_error"],
            [0.03, 0.02, 0.01],
        )
        np.testing.assert_allclose(
            result[-1]["steps"][0]["field_after"],
            refined_field,
        )

    def test_recipe_diagnostics_keep_full_partial_and_gradient_outputs(self):
        data = np.ones((4, 4), dtype=float)
        support = np.ones((4, 4), dtype=float)
        start = np.ones((4, 4), dtype=complex)
        recipe = {
            "algorithm_list": ["ER", "ER", "gradient_descent"],
            "number_iterations": [1, 1, 1],
            "helicity": ["pos", "pos", "pos"],
            "beta_zero": [0.5, 0.5, 0.2],
            "beta_mode": ["const", "const", "const"],
            "alpha_zero": [0.0, 0.0, 0.0],
            "alpha_mode": ["const", "const", "const"],
            "RL_its": [0, 1, 0],
            "RL_freqs": [10, 1, 10],
            "TV_freqs": [10, 10, 10],
            "plot_every": [1, 1, 1],
            "average_img": [1, 1, 1],
            "Fourier_last": [False, False, False],
            "hologram_intensity_cutoff_vmin": -1,
            "Startimage": [start, "pos", "pos"],
            "Startgamma": [None, "pos", None],
        }
        core_outputs = [
            2 * start,
            4 * start,
        ]

        def fake_core(**kwargs):
            return core_outputs.pop(0), np.array([0.1]), np.array([0.01]), None

        fake_gradient = core.gradient.GradientRefinementResult(
            fields=7 * start,
            loss=np.array([0.7]),
            diffraction_loss=np.array([0.07]),
            support_loss=np.array([0.007]),
        )

        with (
            mock.patch.object(core, "PhaseRtrv_core", side_effect=fake_core),
            mock.patch.object(
                core.gradient,
                "refine_field_gradient",
                return_value=fake_gradient,
            ),
        ):
            result = core.phase_retrieval_algorithm(
                data,
                data,
                np.zeros_like(data, dtype=int),
                support,
                phase_retrieval_recipe=recipe,
            )

        errors = result[-1]
        np.testing.assert_allclose(
            errors["latest"]["full_coherence"]["pos"],
            2 * start,
        )
        np.testing.assert_allclose(
            errors["latest"]["partial_coherence"]["pos"],
            4 * start,
        )
        np.testing.assert_allclose(
            errors["latest"]["gradient_descent"]["pos"],
            7 * start,
        )
        self.assertEqual([step["mode"] for step in errors["steps"]], [
            "ER",
            "ER",
            "gradient_descent",
        ])
        for step in errors["steps"]:
            self.assertIn("field_after", step)

    def test_default_outputs_select_last_step_of_each_helicity(self):
        data = np.ones((4, 4), dtype=float)
        support = np.ones((4, 4), dtype=float)
        start = np.ones((4, 4), dtype=complex)
        recipe = {
            "algorithm_list": ["ER", "ER", "ER"],
            "number_iterations": [1, 1, 1],
            "helicity": ["pos", "neg", "pos"],
            "beta_zero": [0.5, 0.5, 0.5],
            "beta_mode": ["const", "const", "const"],
            "alpha_zero": [0.0, 0.0, 0.0],
            "alpha_mode": ["const", "const", "const"],
            "RL_its": [0, 0, 0],
            "RL_freqs": [10, 10, 10],
            "TV_freqs": [10, 10, 10],
            "plot_every": [1, 1, 1],
            "average_img": [1, 1, 1],
            "Fourier_last": [False, False, False],
            "hologram_intensity_cutoff_vmin": -1,
            "Startimage": [start, start, "pos"],
            "Startgamma": [None, None, None],
        }
        core_outputs = [2 * start, 3 * start, 4 * start]

        def fake_core(**kwargs):
            return core_outputs.pop(0), np.array([0.1]), np.array([0.01]), None

        with mock.patch.object(core, "PhaseRtrv_core", side_effect=fake_core):
            result = core.phase_retrieval_algorithm(
                data,
                data,
                np.zeros_like(data, dtype=int),
                support,
                phase_retrieval_recipe=recipe,
            )

        outputs = result[-1]["outputs"]
        self.assertEqual([item["step"] for item in outputs], [1, 2])
        self.assertEqual([item["helicity"] for item in outputs], ["neg", "pos"])

    def test_explicit_output_flags_select_requested_steps(self):
        data = np.ones((4, 4), dtype=float)
        support = np.ones((4, 4), dtype=float)
        start = np.ones((4, 4), dtype=complex)
        recipe = {
            "algorithm_list": ["ER", "ER", "ER"],
            "number_iterations": [1, 1, 1],
            "helicity": ["pos", "neg", "pos"],
            "beta_zero": [0.5, 0.5, 0.5],
            "beta_mode": ["const", "const", "const"],
            "alpha_zero": [0.0, 0.0, 0.0],
            "alpha_mode": ["const", "const", "const"],
            "RL_its": [0, 0, 0],
            "RL_freqs": [10, 10, 10],
            "TV_freqs": [10, 10, 10],
            "plot_every": [1, 1, 1],
            "average_img": [1, 1, 1],
            "Fourier_last": [False, False, False],
            "output": [True, False, True],
            "hologram_intensity_cutoff_vmin": -1,
            "Startimage": [start, start, "pos"],
            "Startgamma": [None, None, None],
        }
        core_outputs = [2 * start, 3 * start, 4 * start]

        def fake_core(**kwargs):
            return core_outputs.pop(0), np.array([0.1]), np.array([0.01]), None

        with mock.patch.object(core, "PhaseRtrv_core", side_effect=fake_core):
            result = core.phase_retrieval_algorithm(
                data,
                data,
                np.zeros_like(data, dtype=int),
                support,
                phase_retrieval_recipe=recipe,
            )

        outputs = result[-1]["outputs"]
        self.assertEqual([item["step"] for item in outputs], [0, 2])
        self.assertEqual(
            [item["step"] for item in result[-1]["outputs_by_helicity"]["pos"]],
            [0, 2],
        )

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
