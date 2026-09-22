"""Synchronous observation rounds and shared partial-coherence updates."""
import contextlib
import io
import threading
import unittest

import numpy as np

from library import phase_retrieval_universal as universal
from library import phase_retrieval_core_unified as unified


class ParallelRetrievalTests(unittest.TestCase):
    def recipe(self, strategy, workers, modes):
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            modes=modes, partial_coherence=True, coherence_kernel_scope="shared",
            shared_coherence_update=strategy, observation_workers=workers, fft_workers=1,
            coherence_round_iterations=2, preserve_warmup_masked_intensity=True,
            warmup_mode=["ER", "ER"], warmup_Nit=[4, 4],
            warmup_RL_it=[0, 1], warmup_RL_freq=[100, 1],
            warmup_start_from_first=True, warmup_seed_scale_fit="sum",
            inner_mode=["ER"], inner_Nit=[4], RL_it=1, RL_freq=1,
            outer_iterations=2, shuffle_observations=False,
            projection_model="none", final_projection_relaxation=0,
            final_fourier_constraint=False, average_img=1,
        )
        return recipe

    def run_recipe(self, recipe, kernel):
        shape = (8, 8)
        holograms = np.stack([np.full(shape, value ** 2, dtype=float) for value in (2, 3, 4)])
        masks = np.zeros_like(holograms)
        masks[:, 1, 2] = 1
        with contextlib.redirect_stdout(io.StringIO()):
            return universal.universal_phase_retrieval_algorithm(
                holograms, masks, np.ones(shape), ["sat", "a", "b"],
                [783] * 3, [1] * 3, ["beam"] * 3,
                universal_recipe=recipe, phase_retrieval_kernel=kernel,
            )

    def test_common_mode_is_invariant_to_global_phase_and_phase_wrapping(self):
        rows, columns = np.indices((32, 32))
        support = (abs(rows - 16) < 8) & (abs(columns - 16) < 8)
        obj = support * np.exp(1j * (3.05 + .03 * (columns - 16)))
        field = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(obj)))
        fields = np.zeros((3, 2, 32, 32), dtype=complex)
        fields[:, 0] = field
        fields[:, 1] = field * np.exp(1j * np.array([0, .2, -2.9]))[:, None, None]
        recipe = universal.default_universal_phase_retrieval_recipe()
        projected, _ = universal._project_nonphysical_modes_common(
            fields, [783]*3, ["beam"]*3, recipe,
        )
        np.testing.assert_array_equal(projected[:, 0], fields[:, 0])
        for observation in range(3):
            np.testing.assert_allclose(projected[observation, 1], field, atol=1e-15)
        recipe["nonphysical_modes_common_relaxation"] = 0
        bypass, _ = universal._project_nonphysical_modes_common(fields, [783]*3, ["beam"]*3, recipe)
        np.testing.assert_array_equal(bypass, fields)
        zeros, _ = universal._project_nonphysical_modes_common(np.zeros_like(fields), [783]*3, ["beam"]*3,
                                                              universal.default_universal_phase_retrieval_recipe())
        np.testing.assert_array_equal(zeros, 0)

    def test_coherent_updates_ignore_saved_masked_targets(self):
        rng = np.random.default_rng(18)
        field = rng.normal(size=(8,8)) + 1j*rng.normal(size=(8,8))
        mask = np.zeros((8,8)); mask[0,0] = 1
        support = np.zeros((8,8)); support[2:6,2:6] = 1
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(modes=[1], Nmodes=1, partial_coherence=False, inner_mode=["ER"], inner_Nit=[4])
        schedule = universal._build_update_schedule(recipe, name="inner")
        outputs = [universal._run_update_schedule(
            field, np.ones((8,8)), support, mask, schedule, recipe,
            phase_retrieval_kernel=unified.PhaseRtrv_core,
            fixed_masked_intensity=np.full((8,8), fill),
        )[0] for fill in (1, 1e9)]
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertGreater(abs(outputs[0][0,0]-field[0,0]), 1e-6)

    def test_workers_really_overlap_and_preserve_result_order(self):
        barrier = threading.Barrier(2)
        def task(index):
            barrier.wait(timeout=5)
            return index * 10
        self.assertEqual(list(universal._observation_map(task, [0, 1], 2)), [(0, 0), (1, 10)])

    def test_average_uses_one_snapshot_and_preserves_masked_targets(self):
        for modes in ([1], [1, 2]):
            with self.subTest(modes=modes):
                calls = []
                lock = threading.Lock()
                def kernel(**kw):
                    index = int(round(kw["diffract"][0, 0])) - 2
                    gamma = kw["gamma"]
                    with lock:
                        calls.append((index, None if gamma is None else gamma.copy(),
                                      kw["diffract"].copy()))
                    if gamma is not None:
                        gamma = np.zeros_like(gamma, dtype=float)
                        gamma[..., 4, index + 2] = 1
                    field = np.full_like(kw["Phase"], index + 2 if gamma is None else 20)
                    return field, np.array([0.]), np.array([0.]), gamma
                recipe = self.recipe("average", 2, modes)
                recipe["observation_weights"] = [1, 2, 3]
                result = self.run_recipe(recipe, kernel)
                partial = [(i, g, d) for i, g, d in calls if g is not None]
                self.assertEqual(len(partial), 9)
                expected = np.zeros_like(partial[0][1])
                expected[..., 4, 2:5] = np.array([1, 2, 3]) / 6
                for index, (observation, gamma, amplitude) in enumerate(partial):
                    np.testing.assert_allclose(gamma, partial[0][1] if index < 3 else expected)
                    self.assertAlmostEqual(amplitude[1, 2] ** 2,
                                           (observation + 2) ** 2 * len(modes))
                np.testing.assert_allclose(result[2]["coherence_kernel"], expected)

    def test_calibrated_warmup_seeds_loop_gamma_and_records_concurrency(self):
        import time
        calls = []
        lock = threading.Lock()
        def kernel(**kw):
            index = int(round(kw["diffract"][0, 0])) - 2
            gamma = kw["gamma"]
            with lock:
                calls.append((index, None if gamma is None else gamma.copy()))
            time.sleep(.02)  # Represent a numerical operation that releases the GIL.
            if gamma is not None:
                gamma = np.zeros_like(gamma)
                gamma[4, index + 2] = 1
            return kw["Phase"].copy(), np.array([0.]), np.array([0.]), gamma
        recipe = self.recipe("average", 2, [1])
        recipe["warmup_calibrate_shared_gamma"] = True
        results = self.run_recipe(recipe, kernel)
        partial = [(index, gamma) for index, gamma in calls if gamma is not None]
        reference_gamma = np.zeros((8, 8))
        reference_gamma[4, 2] = 1
        self.assertEqual(partial[0][0], 0)
        for _, gamma in partial[1:3]:
            np.testing.assert_array_equal(gamma, reference_gamma)
        average_gamma = np.zeros((8, 8))
        average_gamma[4, 2:5] = 1 / 3
        for _, gamma in partial[3:]:
            np.testing.assert_allclose(gamma, average_gamma)
        self.assertEqual(results[4]["execution_summary"]["max_concurrent_updates"], 2)
        self.assertEqual([p["pass"] for p in results[4]["warmup_passes"]], ["coherent", "partial"])

    def test_pooled_fit_recovers_weighted_delta_blur_and_handles_dark_mode(self):
        shape = (8, 8)
        fields = np.zeros((2, 2, *shape), dtype=complex)
        fields[:, 0, 4, 4] = 1
        targets = np.zeros((2, *shape))
        targets[0, 4, 3] = 1
        targets[1, 4, 5] = 1
        gamma = np.full((2, *shape), 1 / 64)
        fitted = universal._pooled_coherence_update(fields, targets, gamma, 2, [1, 3])
        expected = np.zeros(shape)
        expected[4, 3] = .25
        expected[4, 5] = .75
        np.testing.assert_allclose(fitted[0], expected, atol=1e-14)
        np.testing.assert_allclose(fitted[1], gamma[1], atol=1e-14)
        np.testing.assert_allclose(fitted.sum(axis=(-2, -1)), 1)

    def test_real_kernel_serial_and_parallel_rounds_agree_with_frozen_reference(self):
        for strategy in ("average", "pooled"):
            for modes in ([1], [1, 2]):
                with self.subTest(strategy=strategy, modes=modes):
                    results = []
                    for workers in (1, 2):
                        recipe = self.recipe(strategy, workers, modes)
                        recipe.update(freeze_saturated_fields=True, saturated_states={"sat": 1})
                        results.append(self.run_recipe(recipe, unified.PhaseRtrv_core))
                    for serial, parallel in zip(results[0][:2], results[1][:2]):
                        np.testing.assert_allclose(serial, parallel, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(results[0][2]["coherence_kernel"],
                                               results[1][2]["coherence_kernel"], atol=1e-12)
                    np.testing.assert_array_equal(results[1][0][0], results[1][1][0])
                    gamma = results[1][2]["coherence_kernel"]
                    self.assertTrue(np.all(gamma >= 0))
                    np.testing.assert_allclose(gamma.sum(axis=(-2, -1)), 1)

    def test_parallel_coherent_rounds_match_serial_physical_projections(self):
        results = []
        for workers in (1, 2):
            recipe = self.recipe("average", workers, [1, 2])
            recipe.update(
                partial_coherence=False, warmup_mode=["ER"], warmup_Nit=[4],
                warmup_RL_it=[0], warmup_RL_freq=[100], RL_it=0,
                projection_model="physical_factorized", projection_every=3,
                projection_relaxation=.1, physical_iterations=2, projection_diagnostic_observation=1,
                saturated_states={"sat": 1}, freeze_saturated_fields=True,
            )
            results.append(self.run_recipe(recipe, unified.PhaseRtrv_core))
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-12, atol=1e-12)
        self.assertEqual(len(results[1][4]["projection_steps"]), 2)
        trace = results[1][4]["detector_constraint_diagnostics"]
        physical = [r for r in trace if r["stage"] == "physical_primary"]
        common = [r for r in trace if r["stage"] == "common_secondary"]
        self.assertEqual(len(physical), 2)
        for before, after in zip(physical, common):
            self.assertEqual(before["primary_outer_masked_mean"], after["primary_outer_masked_mean"])
        np.testing.assert_array_equal(results[1][0][0], results[1][1][0])

    def test_mid_sweep_physical_projection_is_rejected(self):
        recipe = self.recipe("average", 2, [1])
        recipe.update(projection_model="physical_factorized", projection_every=1)
        with self.assertRaisesRegex(ValueError, "complete observation sweeps"):
            self.run_recipe(recipe, unified.PhaseRtrv_core)
