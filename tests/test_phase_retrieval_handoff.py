import unittest
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


class PhaseRetrievalHandoffTests(unittest.TestCase):
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
