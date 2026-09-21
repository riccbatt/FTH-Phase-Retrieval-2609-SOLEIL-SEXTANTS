import unittest
import contextlib
import io
from unittest.mock import patch

import numpy as np

from library import phase_retrieval_core as single
from library import phase_retrieval_core_multimode as multimode
from library import phase_retrieval_core_multienergy as multienergy
from library import phase_retrieval_core_multienergy_000 as multienergy_000
from library import phase_retrieval_core_general as general
from library import phase_retrieval_core_dichroic as dichroic
from library import phase_retrieval_core_multienergy_multimode as energy_modes
from library import phase_retrieval_core_unified as unified
from library import phase_retrieval_universal as universal
from library import fthcore


class PhaseRetrievalHandoffTests(unittest.TestCase):
    def test_projection_focus_transform_is_reversible_and_zero_is_noop(self):
        rng = np.random.default_rng(9)
        fields = rng.normal(size=(2, 10, 10)) + 1j * rng.normal(size=(2, 10, 10))
        setup = {"ccd_dist": 0.125, "px_size": 11e-6}
        focused = universal._projection_focus_transform(
            fields, [780.0, 790.0], 2.05, -0.76, setup,
        )
        restored = universal._projection_focus_transform(
            focused, [780.0, 790.0], 2.05, -0.76, setup, inverse=True,
        )

        np.testing.assert_allclose(restored, fields, rtol=1e-12, atol=1e-12)
        focus_setup = {**setup, "energy": 780.0}
        expected = fthcore.propagate(
            fields[0], 2.05e-6, focus_setup,
        ) * np.exp(-0.76j)
        np.testing.assert_allclose(focused[0], expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(
            universal._projection_focus_transform(
                fields, [780.0, 790.0], 0.0, 2.0, None,
            ),
            fields,
        )

    def test_nonphysical_mode_can_be_common_across_states(self):
        rng = np.random.default_rng(22)
        fields = (rng.normal(size=(3, 2, 8, 8))
                  + 1j * rng.normal(size=(3, 2, 8, 8)))
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            constrain_nonphysical_modes_common=True,
            nonphysical_modes_common_relaxation=1.0,
        )

        projected, groups = universal._project_nonphysical_modes_common(
            fields,
            energy_labels=[780.0, 780.0, 790.0],
            beam_labels=["beam", "beam", "beam"],
            recipe=recipe,
        )

        np.testing.assert_array_equal(projected[:, 0], fields[:, 0])
        np.testing.assert_allclose(projected[0, 1], projected[1, 1])
        self.assertFalse(np.allclose(projected[0, 1], projected[2, 1]))
        self.assertEqual(groups[0]["observations"], [0, 1])

    def test_reference_and_other_warmup_recipes_are_independent(self):
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            warmup_mode=["HAPRE", "ER"], warmup_Nit=[700, 50],
            warmup_RL_it=[0, 0], warmup_RL_freq=[1e9, 1e9],
            warmup_reference_mode=["HAPRE", "ER"],
            warmup_reference_Nit=[700, 50],
            warmup_other_mode=["ER"], warmup_other_Nit=[50],
            warmup_other_RL_it=[0], warmup_other_RL_freq=[1e9],
        )

        reference = universal._build_role_warmup_schedule(recipe, "reference")
        other = universal._build_role_warmup_schedule(recipe, "other")

        self.assertEqual(
            [(stage["mode"], stage["Nit"]) for stage in reference],
            [("HAPRE", 700), ("ER", 50)],
        )
        self.assertEqual(
            [(stage["mode"], stage["Nit"]) for stage in other],
            [("ER", 50)],
        )

    def test_workflow_tree_reports_recipes_and_startimage_handoff(self):
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            warmup_reference_observation=0,
            warmup_start_from_first=True,
            warmup_reference_mode=["HAPRE", "ER"],
            warmup_reference_Nit=[700, 50],
            warmup_other_mode=["ER"], warmup_other_Nit=[50],
            inner_mode=["ER"], inner_Nit=[20], outer_iterations=3,
            freeze_saturated_fields=True,
            saturated_states={"saturated": 1},
        )

        tree = universal.format_universal_workflow(
            recipe, state_labels=["saturated", "field_1", "field_2"],
        )

        self.assertIn("Reference hologram: 'saturated'", tree)
        self.assertIn("HAPRE × 700", tree)
        self.assertIn("scaled final field from reference 'saturated'", tree)
        self.assertIn("Other holograms: 'field_1', 'field_2'", tree)
        self.assertIn("Joint reconstruction × 3 outer loops", tree)
        self.assertIn("Saturated observations remain fixed", tree)

    def test_universal_multimode_support_fft_start_matches_unified_start(self):
        support = np.zeros((2, 8, 8), dtype=np.uint8)
        support[0, 3:5, 3:5] = 1
        support[1, 2:6, 2:6] = 1
        intensities = np.arange(64, dtype=float).reshape(1, 8, 8) + 1
        amplitudes = np.sqrt(intensities)
        mask = np.zeros((1, 8, 8), dtype=np.uint8)
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(mode_initialization="support_fft", Nmodes=2)

        actual = universal._initialize_physical_modal_fields(
            support, amplitudes, intensities, mask, recipe,
        )
        expected = np.stack([
            np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(mode)))
            for mode in support
        ])
        measured = amplitudes[0].ravel()
        current = np.sqrt(np.sum(np.abs(expected) ** 2, axis=0)).ravel()
        scale = np.dot(measured, current) / np.dot(measured, measured)

        np.testing.assert_allclose(actual[0], expected / scale)

    def test_two_mode_universal_warmup_matches_unified_full_coherence(self):
        rng = np.random.default_rng(14)
        saturated = rng.uniform(0.5, 3.0, (12, 12))
        loop = 0.9 * saturated + rng.uniform(0.05, 0.2, (12, 12))
        support = np.zeros((12, 12), dtype=np.uint8)
        support[4:8, 4:8] = 1
        mask = np.zeros((12, 12), dtype=np.uint8)

        pair_recipe = unified.default_phase_retrieval_recipe()
        pair_recipe.update(
            algorithm_list=["HAPRE", "ER", "ER"],
            number_iterations=[2, 1, 1],
            helicity=["saturated", "saturated", "loop"],
            beta_zero=[0.5] * 3,
            beta_mode=["arctan", "const", "const"],
            alpha_zero=[0.0] * 3, alpha_mode=["const"] * 3,
            RL_its=[0] * 3, RL_freqs=[1e9] * 3,
            TV_freqs=[1e9] * 3, plot_every=[1e9] * 3,
            average_img=[1] * 3, Fourier_last=[True] * 3,
            Startimage=[None, "saturated", "saturated"],
            Startgamma=[None] * 3, output=[False, True, True],
            return_format="dict", modes=[1, 2],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            pair = unified.phase_retrieval_algorithm(
                {"saturated": saturated, "loop": loop}, mask, support,
                pair_recipe,
            )

        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            modes=[1, 2], mode_initialization="support_fft",
            constrain_nonphysical_modes_common=True,
            nonphysical_modes_common_relaxation=1.0,
            warmup_start_from_first=True, warmup_reference_observation=0,
            warmup_seed_scale_fit="linear", startimage_scale_fit="through_origin",
            warmup_mode=["HAPRE", "ER"], warmup_Nit=[2, 1],
            warmup_reference_mode=["HAPRE", "ER"],
            warmup_reference_Nit=[2, 1],
            warmup_other_mode=["ER"], warmup_other_Nit=[1],
            warmup_beta_mode=["arctan", "const"],
            warmup_reference_beta_mode=["arctan", "const"],
            warmup_other_beta_mode=["const"],
            inner_mode=["ER"], inner_Nit=[1], outer_iterations=1,
            physical_iterations=1, projection_every=2,
            saturated_states={"saturated": 1}, shuffle_observations=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            final_fields, warmup, _, _, _ = universal.universal_phase_retrieval_algorithm(
                np.stack([saturated, loop]), mask, support,
                ["saturated", "loop"], [783, 783], [1, 1], ["beam", "beam"],
                universal_recipe=recipe,
                phase_retrieval_kernel=unified.PhaseRtrv_core,
            )

        np.testing.assert_allclose(
            warmup[0], pair["full_coherence"]["saturated"], rtol=1e-7, atol=1e-7,
        )
        np.testing.assert_allclose(
            warmup[1], pair["full_coherence"]["loop"], rtol=1e-7, atol=1e-7,
        )
        np.testing.assert_allclose(final_fields[0, 1], final_fields[1, 1])

    def test_partial_coherence_can_use_fixed_kernel(self):
        shape = (8, 8)
        gamma = np.zeros(shape)
        gamma[4, 4] = 1
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(partial_coherence=True, final_fourier_constraint=False,
                      Nmodes=1, average_img=1, plot_every=100)
        stage = dict(mode="ER", Nit=4, beta_zero=0.5, beta_mode="const",
                     alpha_zero=0, alpha_mode="const", TV_freq=100,
                     RL_it=1, RL_freq=4)
        field, results, updated_gamma = universal._run_update_schedule(
            np.ones(shape, dtype=complex), np.ones(shape),
            np.ones(shape), np.zeros(shape), [stage], recipe,
            gamma=gamma, return_gamma=True,
        )
        self.assertEqual(results[0]["coherence"], "partial")
        self.assertEqual(field.shape, shape)
        np.testing.assert_allclose(updated_gamma, gamma, atol=1e-7)

    def test_saturated_first_universal_warmup_matches_unified_pair(self):
        rng = np.random.default_rng(31)
        saturated = rng.uniform(0.5, 3.0, (16, 16))
        loop = 0.8 * saturated + rng.uniform(0.1, 0.4, (16, 16))
        support = np.zeros((16, 16), dtype=np.uint8)
        support[5:11, 5:11] = 1
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[8, 8] = 1

        pair_recipe = unified.default_phase_retrieval_recipe()
        pair_recipe.update(
            algorithm_list=["HAPRE", "ER", "HAPRE", "HAPRE", "ER", "ER"],
            number_iterations=[4] * 6,
            helicity=["saturated", "saturated", "loop", "saturated",
                      "saturated", "loop"],
            beta_zero=[0.5] * 6,
            beta_mode=["arctan", "const", "const", "arctan", "const", "const"],
            alpha_zero=[0] * 6, alpha_mode=["const"] * 6,
            RL_its=[0, 0, 0, 1, 1, 1],
            RL_freqs=[1e9, 1e9, 1e9, 1, 1, 1],
            TV_freqs=[1e9] * 6, plot_every=[1e9] * 6,
            average_img=[1] * 6, Fourier_last=[True] * 6,
            Startimage=[None] + ["saturated"] * 5,
            Startgamma=[None] * 3 + [None, "saturated", "saturated"],
            output=[False] * 6, return_format="dict", modes=[1],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            pair = unified.phase_retrieval_algorithm(
                {"saturated": saturated, "loop": loop}, mask, support,
                pair_recipe,
            )

        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            modes=[1], partial_coherence=True,
            coherence_kernel_scope="shared", final_fourier_constraint=False,
            warmup_mode=["HAPRE", "ER", "HAPRE", "ER"],
            warmup_Nit=[4] * 4, warmup_beta_mode=["arctan", "const", "arctan", "const"],
            warmup_RL_it=[0, 0, 1, 1], warmup_RL_freq=[1e9, 1e9, 1, 1],
            warmup_start_from_first=True, warmup_reference_observation=0,
            warmup_seed_scale_fit="linear", warmup_seeded_stage_indices=[3],
            startimage_scale_fit="through_origin", average_img=1,
            inner_mode=["ER"], inner_Nit=[1], outer_iterations=1,
            physical_iterations=1, saturated_states={"saturated": 1},
            projection_every=2, shuffle_observations=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            _, warmup, _, _, _ = universal.universal_phase_retrieval_algorithm(
                np.stack([saturated, loop]), mask, support,
                ["saturated", "loop"], [783, 783], [1, 1], ["beam", "beam"],
                universal_recipe=recipe,
                phase_retrieval_kernel=unified.PhaseRtrv_core,
            )
        for index, label in enumerate(("saturated", "loop")):
            if label == "saturated":
                np.testing.assert_allclose(
                    warmup[index], pair["partial_coherence"][label],
                    rtol=1e-6, atol=1e-6,
                )
            else:
                # 01 fills the masked pixels for the loop's PC step using
                # its earlier coherent loop reconstruction. 02 starts that
                # step directly from saturated, so its fill is slightly different.
                relative_error = (np.linalg.norm(warmup[index] - pair["partial_coherence"][label])
                                  / np.linalg.norm(pair["partial_coherence"][label]))
                self.assertLess(relative_error, 0.02)

    def test_unified_returns_every_effective_modal_support(self):
        shape = (16, 16)
        support = np.zeros(shape, dtype=np.uint8)
        support[6:10, 6:10] = 1
        recipe = unified.default_phase_retrieval_recipe()
        recipe.update(
            algorithm_list=["ER"], number_iterations=[1], helicity=["pos"],
            RL_its=[0], RL_freqs=[1e9], Startimage=[None],
            Startgamma=[None], average_img=[1], plot_every=[1],
            output=[False], return_format="dict", modes=[1, 2], crop=2,
        )
        for key in ("beta_zero", "beta_mode", "alpha_zero", "alpha_mode",
                    "TV_freqs", "Fourier_last"):
            recipe[key] = recipe[key][:1]

        passed_supports = []
        def fake_core(**kwargs):
            passed_supports.append(kwargs["mask"].copy())
            return kwargs["Phase"], np.zeros(1), np.zeros(1), None

        with patch.object(unified, "PhaseRtrv_core", side_effect=fake_core):
            result = unified.phase_retrieval_algorithm(
                {"pos": np.ones(shape)}, np.zeros(shape), support, recipe
            )
        used = unified._mode_supports(
            unified._support_on_output_grid(support, (12, 12)), [1, 2], (12, 12)
        )
        np.testing.assert_array_equal(result["supportmask_used"], used)
        np.testing.assert_array_equal(passed_supports[0], used)
        expected = used
        np.testing.assert_array_equal(result["supportmask"], expected)
        self.assertEqual(result["supportmask"].shape, (2, 12, 12))
        self.assertGreater(result["supportmask"][1].sum(), result["supportmask"][0].sum())
        explicit = [support, np.roll(support, 3, axis=0)]
        with patch.object(unified, "PhaseRtrv_core", side_effect=fake_core):
            listed = unified.phase_retrieval_algorithm(
                {"pos": np.ones(shape)}, np.zeros(shape), explicit, recipe
            )
        np.testing.assert_array_equal(listed["supportmask_used"], unified._support_on_output_grid(np.asarray(explicit), (12, 12)))
        np.testing.assert_array_equal(passed_supports[1], unified._support_on_output_grid(np.asarray(explicit), (12, 12)))
        np.testing.assert_array_equal(
            listed["supportmask"],
            unified._support_on_output_grid(np.asarray(explicit), (12, 12)),
        )

    def test_unified_binning_masks_and_roi(self):
        shape = (12, 12)
        image = np.arange(144, dtype=float).reshape(shape) + 1
        support = np.zeros(shape, dtype=np.uint8)
        support[3:9, 3:9] = 1
        detector_mask = np.zeros(shape, dtype=np.uint8)
        detector_mask[2, 2] = 1
        recipe = unified.default_phase_retrieval_recipe()
        recipe.update(
            algorithm_list=["ER"], number_iterations=[1], helicity=["pos"],
            RL_its=[0], RL_freqs=[1e9], Startimage=[None],
            Startgamma=[None], average_img=[1], plot_every=[1],
            output=[False], return_format="dict", binning=2, crop=1,
            roi=[3, 9, 3, 9],
        )
        for key in ("beta_zero", "beta_mode", "alpha_zero", "alpha_mode",
                    "TV_freqs", "Fourier_last"):
            recipe[key] = recipe[key][:1]
        captured = []

        def fake_core(**kwargs):
            captured.append(kwargs)
            return kwargs["Phase"], np.zeros(1), np.zeros(1), None

        with patch.object(unified, "PhaseRtrv_core", side_effect=fake_core):
            result = unified.phase_retrieval_algorithm(
                {"pos": image}, detector_mask, support, recipe
            )
        self.assertEqual(captured[0]["diffract"].shape, (5, 5))
        self.assertEqual(result["supportmask"].shape, (5, 5))
        self.assertEqual(result["mask_pixel"].shape, (5, 5))
        self.assertEqual(set(np.unique(result["mask_pixel"])), {0, 1})
        self.assertTrue(set(np.unique(result["supportmask"])) <= {0, 1})
        self.assertEqual(result["mask_pixel"][0, 0], 1)
        # [3:9] covers the centered 5x5 support used after input crop and binning.
        np.testing.assert_array_equal(result["roi"], [0, 5, 0, 5])
        shifted_recipe = dict(recipe, roi=[3, 6, 6, 9])
        with patch.object(unified, "PhaseRtrv_core", side_effect=fake_core):
            shifted = unified.phase_retrieval_algorithm(
                {"pos": image}, detector_mask, support, shifted_recipe
            )
        np.testing.assert_array_equal(shifted["roi"], [0, 2, 2, 5])

    def test_default_startimage_uses_kernel_frame_on_odd_grid(self):
        shape = (9, 11)
        support = np.zeros(shape)
        support[2:5, 3:7] = 1
        support[6, 8] = 1
        recipe = unified.default_phase_retrieval_recipe()
        recipe.update(
            algorithm_list=["ER"], number_iterations=[1], helicity=["pos"],
            RL_its=[0], RL_freqs=[1e9], Startimage=[None],
            Startgamma=[None], average_img=[1], plot_every=[1],
            output=[False], return_format="dict",
        )
        for key in ("beta_zero", "beta_mode", "alpha_zero", "alpha_mode",
                    "TV_freqs", "Fourier_last"):
            recipe[key] = recipe[key][:1]
        captured = []

        def fake_core(**kwargs):
            captured.append(kwargs["Phase"].copy())
            return kwargs["Phase"], np.zeros(1), np.zeros(1), None

        with patch.object(unified, "PhaseRtrv_core", side_effect=fake_core):
            unified.phase_retrieval_algorithm(
                {"pos": np.ones(shape)}, np.zeros(shape), support, recipe
            )
        expected = np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(support))
        )
        nonzero = np.abs(expected) > 1e-10
        np.testing.assert_allclose(
            captured[0][nonzero] / np.abs(captured[0][nonzero]),
            expected[nonzero] / np.abs(expected[nonzero]), atol=1e-12,
        )

    def test_centered_fourier_amplitude_stays_pixel_aligned(self):
        for shape in ((8, 10), (9, 11)):
            with self.subTest(shape=shape):
                rng = np.random.default_rng(17)
                amplitude = rng.uniform(0.5, 3.0, shape)
                phase = np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
                field, _, _, _ = unified.PhaseRtrv_core_single(
                    diffract=amplitude,
                    mask=np.ones(shape),
                    mode="ER",
                    Nit=2,
                    Phase=phase,
                    average_img=1,
                    plot_every=10,
                    Fourier_last=True,
                )
                np.testing.assert_allclose(np.abs(field), amplitude, atol=1e-6)

    def test_schedule_engines_reject_silent_rl_fallback(self):
        stage = {"mode": "ER", "Nit": 2, "RL_it": 1, "RL_freq": 1}
        shape = (8, 8)
        field = np.ones(shape, complex)
        amplitude = np.ones(shape)
        support = np.ones(shape)
        bsmask = np.zeros(shape)
        for module in (general, dichroic, universal):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(ValueError, "no coherence kernel"):
                    module._run_update_schedule(
                        field, amplitude, support, bsmask,
                        [stage], {},
                    )
        with self.assertRaisesRegex(ValueError, "no coherence kernel"):
            energy_modes._run_energy_update_schedule(
                np.ones((2, *shape), complex), amplitude,
                np.ones((2, *shape)), bsmask, [stage], {}, 2, shape,
            )
        with self.assertRaisesRegex(ValueError, "no coherence kernel"):
            multienergy._run_energy_update_schedule(
                field, amplitude, support, bsmask, [stage], {},
            )
        with self.assertRaisesRegex(ValueError, "no coherence kernel"):
            universal._run_energy_update_schedule(
                field, amplitude, support, bsmask, [stage], {},
            )

    def test_new_fc_cycle_refreshes_cached_fill(self):
        shape = (8, 8)
        mask_pixel = np.zeros(shape, dtype=np.uint8)
        mask_pixel[4, 4] = 1
        recipe = unified.default_phase_retrieval_recipe()
        recipe.update(
            algorithm_list=["ER"] * 5,
            number_iterations=[1, 2, 2, 1, 2],
            helicity=["pos"] * 5,
            RL_its=[0, 1, 1, 0, 1],
            RL_freqs=[1e9, 1, 1, 1e9, 1],
            Startimage=[None, "pos", "pos", "pos", "pos"],
            Startgamma=[None, None, "pos", None, "pos"],
            average_img=[1] * 5,
            plot_every=[1] * 5,
            output=[False] * 5,
            return_format="dict",
        )
        for key in (
            "beta_zero", "beta_mode", "alpha_zero", "alpha_mode",
            "TV_freqs", "Fourier_last",
        ):
            recipe[key] = recipe[key][:5]
        calls = []

        def fake_core(**kwargs):
            calls.append(kwargs)
            field = np.full(shape, [2, 7, 8, 5, 9][len(calls) - 1], complex)
            return field, np.zeros(1), np.zeros(1), np.ones(shape)

        with patch.object(unified, "PhaseRtrv_core", side_effect=fake_core):
            unified.phase_retrieval_algorithm(
                {"pos": np.full(shape, 4.0)}, mask_pixel,
                np.ones(shape, dtype=np.uint8), recipe,
            )
        self.assertEqual([call["diffract"][4, 4] for call in calls[1:3]], [2, 2])
        self.assertEqual(calls[4]["diffract"][4, 4], 5)

    def test_multimode_rl_updates_at_first_interval(self):
        for module in (multimode, unified):
            with self.subTest(module=module.__name__):
                shape = (8, 8)
                phases = np.ones((2, *shape), dtype=np.complex128)
                kernels = np.zeros((2, *shape), dtype=float)
                kernels[:, 4, 4] = 1
                with patch.object(module, "RL", side_effect=lambda **kw: kw["gamma_cp"]) as rl:
                    core = (
                        module.PhaseRtrv_core
                        if module is multimode
                        else module.PhaseRtrv_core_multimode
                    )
                    core(
                        diffract=np.ones(shape),
                        mask=np.ones((2, *shape)),
                        mode="ER",
                        Nit=21,
                        Phase=phases,
                        gamma=kernels,
                        RL_freq=20,
                        RL_it=1,
                        Nmodes=2,
                        average_img=1,
                        plot_every=100,
                    )
                self.assertEqual(rl.call_count, 2)

    def test_rl_stages_keep_fc_beamstop_fill_for_each_helicity(self):
        modules = (
            single, multimode, multienergy, multienergy_000, unified, universal
        )
        shape = (8, 8)
        pos = np.full(shape, 4.0)
        neg = np.full(shape, 9.0)
        pos[0, 0] = 0.0
        mask_pixel = np.zeros(shape, dtype=np.uint8)
        mask_pixel[4, 4] = 1
        support = np.ones(shape, dtype=np.uint8)

        for module in modules:
            with self.subTest(module=module.__name__):
                recipe = module.default_phase_retrieval_recipe()
                recipe.update(
                    algorithm_list=["ER"] * 6,
                    number_iterations=[1, 1, 1, 2, 2, 2],
                    helicity=["pos", "pos", "neg", "pos", "pos", "neg"],
                    RL_its=[0, 0, 0, 1, 1, 1],
                    RL_freqs=[1e9, 1e9, 1e9, 1, 1, 1],
                    Startimage=[None, "pos", "pos", "pos", "pos", "pos"],
                    Startgamma=[None, None, None, None, "pos", "pos"],
                    average_img=[1] * 6,
                    plot_every=[1] * 6,
                )
                if "output" in recipe:
                    recipe["output"] = [False] * 6
                if module is unified:
                    recipe["return_format"] = "dict"
                    recipe["normalize_startimage_between_holograms"] = False

                calls = []

                def fake_core(**kwargs):
                    calls.append(kwargs)
                    amplitude = [2, 3, 4, 7, 8, 9][len(calls) - 1]
                    field = np.full(shape, amplitude, dtype=np.complex128)
                    gamma = np.full(shape, amplitude, dtype=float)
                    errors = np.zeros(1)
                    return field, errors, errors, gamma

                with patch.object(module, "PhaseRtrv_core", side_effect=fake_core):
                    module.phase_retrieval_algorithm(
                        pos, neg, mask_pixel, support, recipe
                    )

                self.assertEqual(len(calls), 6)
                np.testing.assert_allclose(calls[0]["diffract"][1, 1], 2)
                np.testing.assert_allclose(calls[2]["diffract"][1, 1], 3)
                np.testing.assert_allclose(calls[3]["diffract"][4, 4], 3)
                np.testing.assert_allclose(calls[4]["diffract"][4, 4], 3)
                np.testing.assert_allclose(calls[5]["diffract"][4, 4], 4)
                self.assertEqual(calls[0]["bsmask"][0, 0], 1)
                self.assertTrue(np.all(calls[3]["bsmask"] == 0))
                np.testing.assert_allclose(calls[4]["gamma"], 7)
                np.testing.assert_allclose(calls[5]["gamma"], 8)


if __name__ == "__main__":
    unittest.main()
