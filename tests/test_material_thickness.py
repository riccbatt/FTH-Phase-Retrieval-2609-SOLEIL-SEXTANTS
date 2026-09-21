"""Checks for known vacuum holes in universal material projections."""

import unittest
import contextlib
import io

import numpy as np

from library import phase_retrieval_universal as universal
from library import phase_retrieval_core_multienergy as multienergy
from library import phase_retrieval_core_general as standalone_general
from library import phase_retrieval_geometry as geometry


class MaterialThicknessTests(unittest.TestCase):
    def setUp(self):
        self.thickness = np.ones((8, 8), dtype=float)
        self.thickness[:2, :2] = 0
        self.beam = np.full((8, 8), 0.2 + 0.1j)

    def test_largest_aperture_binary_thickness(self):
        support = np.zeros((8, 8), dtype=np.uint8)
        support[1:5, 1:5] = 1
        support[6, 6] = 1
        thickness = universal.largest_support_component(support)
        np.testing.assert_array_equal(thickness, (support == 1) & (np.indices(support.shape)[0] < 5))
        self.assertEqual(thickness.dtype, np.uint8)
        with self.assertRaisesRegex(ValueError, "no nonzero aperture"):
            universal.largest_support_component(np.zeros_like(support))

    def test_physical_holes_have_no_charge_or_magnetic_response(self):
        logs = np.stack([
            self.beam + self.thickness * (0.12 + 0.03j + sign * (0.04 + 0.02j))
            for sign in (1, -1)
        ])
        projected, components = universal.project_log_objects_physical(
            logs, ["positive", "negative"], [780, 780], [1, 1], ["beam", "beam"],
            saturated_states={"positive": 1, "negative": -1},
            material_thickness=self.thickness, iterations=3,
            return_components=True,
        )
        np.testing.assert_allclose(
            projected[:, :2, :2], np.broadcast_to(self.beam[:2, :2], (2, 2, 2)),
        )
        np.testing.assert_array_equal(components["magnetization"][:, :2, :2], 0)
        np.testing.assert_array_equal(components["material_thickness"], self.thickness)
        np.testing.assert_array_equal(
            components["magnetization"][:, components["material_thickness"] == 0], 0,
        )

    def test_fitted_thickness_can_recover_after_reaching_zero(self):
        rng = np.random.default_rng(12)
        logs = (rng.normal(size=(3, 4, 4))
                + 1j * rng.normal(size=(3, 4, 4)))
        args = (logs, ["sat", "a", "b"], [780] * 3, [1] * 3, ["beam"] * 3)

        def fit(iterations):
            _, components = universal.project_log_objects_physical(
                *args, saturated_states={"sat": 1},
                material_thickness=np.ones((4, 4)),
                fit_material_thickness=True, iterations=iterations,
                return_components=True,
            )
            return components

        first = fit(1)
        second = fit(2)
        self.assertEqual(first["material_thickness"][1, 2], 0)
        self.assertEqual(first["magnetization"][0, 1, 2], 0)
        self.assertGreater(second["material_thickness"][1, 2], 0)
        self.assertEqual(second["magnetization"][0, 1, 2], 1)

    def test_standalone_physical_projector_uses_fixed_thickness(self):
        logs = np.stack([
            self.beam + self.thickness * (0.1 + sign * 0.02)
            for sign in (1, -1)
        ])
        projected, components = standalone_general.project_log_objects_physical(
            logs, ["positive", "negative"], [780, 780], [1, 1],
            ["beam", "beam"], saturated_states={"positive": 1, "negative": -1},
            material_thickness=self.thickness, iterations=3,
            return_components=True,
        )
        np.testing.assert_allclose(projected[:, :2, :2],
                                   np.broadcast_to(self.beam[:2, :2], (2, 2, 2)))
        np.testing.assert_array_equal(components["magnetization"][:, :2, :2], 0)

    def test_two_mode_physical_projection_changes_only_first_mode(self):
        rng = np.random.default_rng(6)
        fields = (rng.normal(size=(4, 2, 8, 8))
                  + 1j * rng.normal(size=(4, 2, 8, 8)))
        recipe = universal.default_universal_phase_retrieval_recipe()
        projected, components = universal._project_physical_modes(
            fields, ["state"] * 4, [1, 1, 2, 2], [1, -1, 1, -1],
            ["beam"] * 4, recipe=recipe, physical_iterations=2,
            material_thickness=self.thickness, return_components=True,
        )
        np.testing.assert_array_equal(projected[:, 1], fields[:, 1])
        self.assertEqual(components["physical_model_modes"], [1])
        np.testing.assert_array_equal(
            components["magnetization"][:, self.thickness == 0], 0,
        )

    def test_two_mode_driver_matches_summed_intensity(self):
        rng = np.random.default_rng(9)
        holograms = np.abs(rng.normal(size=(4, 8, 8))) + 1
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            Nmodes=2, material_mask=(self.thickness > 0).astype(np.uint8),
            warmup_Nit=0, inner_mode=["ER"], inner_Nit=[1],
            outer_iterations=1, physical_iterations=1,
            projection_every=4, shuffle_observations=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            fields, _, components, _, _ = universal.universal_phase_retrieval_algorithm(
                holograms, np.zeros((8, 8), dtype=np.uint8),
                np.ones((8, 8), dtype=np.uint8),
                ["state"] * 4, [1, 1, 2, 2], [1, -1, 1, -1],
                ["beam"] * 4, universal_recipe=recipe,
            )
        self.assertEqual(fields.shape, (4, 2, 8, 8))
        np.testing.assert_allclose(np.sum(np.abs(fields) ** 2, axis=1),
                                   holograms, rtol=1e-10, atol=1e-10)
        self.assertEqual(components["physical_model_modes"], [1])

    def test_two_mode_physical_driver_fits_partial_coherence_kernels(self):
        rng = np.random.default_rng(19)
        holograms = np.abs(rng.normal(size=(3, 8, 8))) + 1
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            Nmodes=2, partial_coherence=True, final_fourier_constraint=False,
            saturated_states={"saturated": 1}, warmup_mode=["ER"],
            warmup_Nit=[4], warmup_RL_it=[1], warmup_RL_freq=[1],
            inner_mode=["ER"], inner_Nit=[4], RL_it=1, RL_freq=1,
            average_img=1, outer_iterations=1, physical_iterations=1,
            projection_every=3, shuffle_observations=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            fields, _, components, _, _ = universal.universal_phase_retrieval_algorithm(
                holograms, np.zeros((8, 8), dtype=np.uint8),
                np.ones((8, 8), dtype=np.uint8),
                ["saturated", "field_a", "field_b"], [780] * 3,
                [1] * 3, ["beam"] * 3, universal_recipe=recipe,
            )
        self.assertEqual(fields.shape, (3, 2, 8, 8))
        self.assertEqual(components["coherence_kernels"].shape, fields.shape)
        self.assertTrue(np.isfinite(fields).all())
        self.assertTrue(np.isfinite(components["coherence_kernels"]).all())
        self.assertEqual(components["physical_model_modes"], [1])
        self.assertFalse(components["final_fourier_constraint_applied"])

    def test_shared_coherence_kernel_across_field_states(self):
        rng = np.random.default_rng(23)
        holograms = np.abs(rng.normal(size=(3, 8, 8))) + 1
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            Nmodes=2, partial_coherence=True, coherence_kernel_scope="shared",
            final_fourier_constraint=False, saturated_states={"saturated": 1},
            freeze_saturated_fields=True,
            warmup_mode=["ER"], warmup_Nit=[4],
            warmup_RL_it=[1], warmup_RL_freq=[1],
            inner_mode=["ER"], inner_Nit=[4], RL_it=1, RL_freq=1,
            average_img=1, outer_iterations=1, physical_iterations=1,
            projection_every=3, shuffle_observations=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            fields, warmup, components, _, _ = universal.universal_phase_retrieval_algorithm(
                holograms, np.zeros((8, 8), dtype=np.uint8),
                np.ones((8, 8), dtype=np.uint8),
                ["saturated", "field_a", "field_b"], [780] * 3,
                [1] * 3, ["beam"] * 3, universal_recipe=recipe,
            )
        self.assertEqual(fields.shape, (3, 2, 8, 8))
        self.assertEqual(components["coherence_kernel"].shape, (2, 8, 8))
        self.assertTrue(np.isfinite(components["coherence_kernel"]).all())
        self.assertNotIn("coherence_kernels", components)
        self.assertEqual(components["physical_model_modes"], [1])
        self.assertEqual(components["frozen_saturated_observations"], [0])
        np.testing.assert_array_equal(fields[0], warmup[0])

    def test_zero_final_projection_skips_model_decomposition(self):
        rng = np.random.default_rng(44)
        start = (1 + rng.normal(size=(2, 8, 8))
                 + 1j * rng.normal(size=(2, 8, 8)))
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            warmup_Nit=0, inner_mode=["ER"], inner_Nit=[1],
            outer_iterations=1, physical_iterations=1,
            projection_every=3, final_projection_relaxation=0.0,
            final_fourier_constraint=False, shuffle_observations=False,
            saturated_states={"sat": 1},
            material_mask=(np.indices((8, 8))[0] >= 2).astype(np.uint8),
        )

        def unchanged_kernel(**kwargs):
            return kwargs["Phase"], np.zeros(1), np.zeros(1), None

        with contextlib.redirect_stdout(io.StringIO()):
            fields, _, components, _, _ = universal.universal_phase_retrieval_algorithm(
                np.ones((2, 8, 8)), np.zeros((8, 8), dtype=np.uint8),
                np.ones((8, 8), dtype=np.uint8),
                ["sat", "field"], [780, 780], [1, 1], ["beam", "beam"],
                universal_recipe=recipe, start_fields=start,
                phase_retrieval_kernel=unchanged_kernel,
            )
        np.testing.assert_array_equal(fields, start)
        self.assertEqual(components["final_projection_relaxation"], 0.0)
        self.assertTrue(components["final_projection_skipped"])
        self.assertNotIn("magnetization", components)

    def test_same_energy_same_helicity_keeps_distinct_field_states(self):
        row, col = np.indices((8, 8))
        true_magnetization = np.stack([
            np.ones((8, 8)),
            np.where(col < 4, -0.8, 0.3),
            np.where(row < 4, 0.5, -0.4),
        ])
        common = np.full((8, 8), 0.2 + 0.1j)
        logs = np.stack([
            common + 0.1 + 0.02j + (0.04 + 0.03j) * state
            for state in true_magnetization
        ])
        projected, components = universal.project_log_objects_physical(
            logs, ["saturated", "field_a", "field_b"], [783] * 3,
            [1] * 3, ["beam"] * 3,
            saturated_states={"saturated": 1}, iterations=20,
            return_components=True,
        )
        self.assertEqual(components["state_names"],
                         ["saturated", "field_a", "field_b"])
        self.assertGreater(np.max(np.abs(
            components["magnetization"][1]
            - components["magnetization"][2]
        )), 0.5)
        np.testing.assert_allclose(projected, logs, atol=1e-4)

    def test_legacy_mode_factors_expand_only_second_support(self):
        rng = np.random.default_rng(10)
        support = np.zeros((16, 16), dtype=np.uint8)
        support[6:10, 6:10] = 1
        support[2, 2] = 1
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            modes=[1, 2], material_mask=universal.largest_support_component(support),
            warmup_Nit=0, inner_mode=["ER"], inner_Nit=[1],
            outer_iterations=1, physical_iterations=1,
            projection_every=4, shuffle_observations=False,
        )
        holograms = np.abs(rng.normal(size=(4, 16, 16))) + 1
        with contextlib.redirect_stdout(io.StringIO()):
            fields, _, components, _, _ = universal.universal_phase_retrieval_algorithm(
                holograms, np.zeros((16, 16), dtype=np.uint8), support,
                ["state"] * 4, [1, 1, 2, 2], [1, -1, 1, -1],
                ["beam"] * 4, universal_recipe=recipe,
            )
        modal_support = components["modal_supportmask_used"]
        self.assertEqual(fields.shape, (4, 2, 16, 16))
        np.testing.assert_array_equal(modal_support[0], support)
        self.assertGreater(modal_support[1].sum(), modal_support[0].sum())
        self.assertEqual(components["physical_model_modes"], [1])

    def test_component_centered_modes_keep_off_center_apertures(self):
        support = np.zeros((32, 32), dtype=np.uint8)
        support[2:6, 3:7] = 1
        support[23:25, 26:28] = 1
        modal_support = universal._component_centered_modal_supports(
            support, [1, 2],
        )
        np.testing.assert_array_equal(modal_support[0], support)
        self.assertGreater(modal_support[1].sum(), support.sum())
        self.assertTrue(np.all(modal_support[1][support != 0]))
        self.assertTrue(np.any(modal_support[1][2:6, 3:7]))
        self.assertTrue(np.any(modal_support[1][23:25, 26:28]))

    def test_warmup_can_seed_field_states_from_saturated_result(self):
        starts = []

        def recording_kernel(**kwargs):
            starts.append(kwargs["Phase"].copy())
            return kwargs["Phase"] + 1, np.array([0.]), np.array([0.]), None

        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            modes=[1], warmup_start_from_first=True,
            warmup_mode=["ER"], warmup_Nit=[1],
            inner_mode=["ER"], inner_Nit=[1], outer_iterations=1,
            physical_iterations=1, projection_every=2,
            shuffle_observations=False, saturated_states={"sat": 1},
        )
        support = np.zeros((8, 8), dtype=np.uint8)
        support[3:5, 3:5] = 1
        holograms = np.stack([np.ones((8, 8)), 4 * np.ones((8, 8))])
        with contextlib.redirect_stdout(io.StringIO()):
            universal.universal_phase_retrieval_algorithm(
                holograms, np.zeros((8, 8), dtype=np.uint8), support,
                ["sat", "field"], [783, 783], [1, 1], ["beam", "beam"],
                universal_recipe=recipe,
                phase_retrieval_kernel=recording_kernel,
            )
        np.testing.assert_allclose(starts[1], 2 * (starts[0] + 1))

    def test_warmup_can_start_from_loop_observation_with_linear_scaling(self):
        starts = []

        def recording_kernel(**kwargs):
            starts.append(kwargs["Phase"].copy())
            return kwargs["Phase"] + 1, np.array([0.]), np.array([0.]), None

        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update(
            modes=[1], warmup_start_from_first=True,
            warmup_reference_observation=1, warmup_seed_scale_fit="linear",
            warmup_seeded_stage_indices=[1],
            startimage_scale_fit="through_origin", warmup_mode=["ER", "ER"],
            warmup_Nit=[1, 1], inner_mode=["ER"], inner_Nit=[1],
            outer_iterations=1, physical_iterations=1,
            projection_every=2, shuffle_observations=False,
            saturated_states={"sat": 1},
        )
        support = np.ones((8, 8), dtype=np.uint8)
        loop = np.arange(64, dtype=float).reshape(8, 8) + 1
        holograms = np.stack([4 * loop + 2, loop])
        with contextlib.redirect_stdout(io.StringIO()):
            universal.universal_phase_retrieval_algorithm(
                holograms, np.zeros((8, 8), dtype=np.uint8), support,
                ["sat", "field"], [783, 783], [1, 1], ["beam", "beam"],
                universal_recipe=recipe,
                phase_retrieval_kernel=recording_kernel,
            )
        np.testing.assert_allclose(starts[1], starts[0] + 1)
        np.testing.assert_allclose(starts[2], 2 * (starts[0] + 2))

    def test_multi_energy_holes_are_energy_independent(self):
        logs = np.stack([
            self.beam + self.thickness * response
            for response in (0.1 + 0.02j, 0.2 + 0.04j, 0.3 + 0.06j)
        ])
        for projector in (
            universal.project_log_object_rank1_spectral,
            universal.project_log_object_low_rank,
        ):
            projected, components = projector(
                logs, material_thickness=self.thickness, return_components=True,
            )
            np.testing.assert_allclose(projected, logs, atol=1e-12)
            np.testing.assert_array_equal(
                components["energy_dependent_log_object"][:, :2, :2], 0,
            )

    def test_fitted_thickness_respects_known_holes(self):
        true_thickness = np.tile(np.linspace(0.5, 1.5, 8), (8, 1))
        true_thickness[:2, :2] = 0
        known_material = (true_thickness > 0).astype(float)
        expected = true_thickness / np.mean(true_thickness[true_thickness > 0])
        logs = np.stack([
            self.beam + true_thickness * response
            for response in (0.1 + 0.02j, 0.2 + 0.04j, 0.3 + 0.06j)
        ])
        projected, components = universal.project_log_object_rank1_spectral(
            logs, material_thickness=known_material,
            fit_material_thickness=True, return_components=True,
        )
        np.testing.assert_allclose(projected, logs, atol=1e-12)
        np.testing.assert_allclose(components["material_thickness"], expected, atol=1e-12)
        self.assertTrue(components["material_thickness_fitted"])

        magnetic_logs = np.stack([
            self.beam + true_thickness * (0.1 + 0.03j + sign * 0.02)
            for sign in (1, -1)
        ])
        projected, components = universal.project_log_objects_physical(
            magnetic_logs, ["positive", "negative"], [780, 780], [1, 1],
            ["beam", "beam"], saturated_states={"positive": 1, "negative": -1},
            iterations=5, material_thickness=known_material,
            fit_material_thickness=True, return_components=True,
        )
        np.testing.assert_allclose(projected, magnetic_logs, atol=1e-12)
        np.testing.assert_array_equal(components["material_thickness"][:2, :2], 0)

    def test_known_holes_override_projection_support_and_relaxation(self):
        logs = np.stack([
            self.beam + self.thickness * response
            for response in (0.1, 0.2, 0.3)
        ])
        logs[0, :2, :2] += 0.1  # An inconsistent hole measurement.
        support = np.ones((8, 8), dtype=bool)
        support[:2, :2] = False
        projected = universal.project_log_object_rank1_spectral(
            logs, material_thickness=self.thickness,
            projection_supportmask=support, relaxation=0.25,
        )
        np.testing.assert_allclose(
            projected[:, :2, :2],
            np.broadcast_to(np.mean(logs[:, :2, :2], axis=0), (3, 2, 2)),
        )

    def test_standalone_multienergy_library_accepts_material_map(self):
        logs = np.stack([
            self.beam + self.thickness * response
            for response in (0.1, 0.2, 0.3)
        ])
        projected, components = multienergy.project_log_object_rank1_spectral(
            logs, material_thickness=self.thickness, return_components=True,
        )
        np.testing.assert_allclose(projected, logs, atol=1e-12)
        np.testing.assert_array_equal(components["material_thickness"], self.thickness)

    def test_binary_material_mask_maps_after_crop_and_bin(self):
        source_mask = np.ones((16, 16), dtype=np.uint8)
        source_mask[6:10, 6:10] = 0
        _, _, _, _, input_geometry = geometry.prepare(
            np.ones((2, 16, 16)), np.zeros((16, 16)),
            np.ones((16, 16)), {"crop": 2, "binning": 2, "roi": None},
        )
        mapped = universal._recipe_material_thickness(
            {"material_mask": source_mask, "material_thickness": None}, input_geometry,
        )
        self.assertEqual(mapped.shape, (6, 6))
        self.assertEqual(mapped[3, 3], 0)
        self.assertEqual(mapped[0, 0], 1)

    def test_negative_thickness_is_rejected(self):
        invalid = self.thickness.copy()
        invalid[3, 3] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            universal.project_log_object_rank1_spectral(
                np.ones((2, 8, 8)), material_thickness=invalid,
            )

    def test_universal_drivers_map_mask_through_crop_then_bin(self):
        rng = np.random.default_rng(3)
        holograms = np.abs(rng.normal(size=(3, 16, 16))) + 1
        detector_mask = np.zeros((16, 16), dtype=np.uint8)
        support = np.ones((16, 16), dtype=np.uint8)
        material = np.ones((16, 16), dtype=np.uint8)
        material[6:10, 6:10] = 0
        for model, states, energies in (
            ("rank1_spectral", ["s"] * 3, [1., 2., 3.]),
            ("physical_factorized", ["a", "b", "c"], [1.] * 3),
        ):
            recipe = universal.default_universal_phase_retrieval_recipe()
            recipe.update(
                projection_model=model, material_mask=material,
                inner_mode=["ER"], inner_Nit=[1], warmup_Nit=0,
                outer_iterations=1, physical_iterations=1,
                crop=2, binning=2, final_fourier_constraint=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                fields, _, components, _, _ = (
                    universal.universal_phase_retrieval_algorithm(
                        holograms, detector_mask, support, states, energies,
                        [1] * 3, ["beam"] * 3, universal_recipe=recipe,
                    )
                )
            self.assertEqual(fields.shape, (3, 6, 6))
            self.assertEqual(components["material_thickness"].shape, (6, 6))
            self.assertGreater(np.count_nonzero(components["material_thickness"] == 0), 0)

    def test_physical_thickness_can_be_zero_outside_support(self):
        holograms = np.abs(np.random.default_rng(4).normal(size=(2, 16, 16))) + 1
        support = np.zeros((16, 16), dtype=np.uint8)
        support[4:12, 4:12] = 1
        material = np.ones((16, 16), dtype=np.uint8)
        material[7:9, 7:9] = 0  # A known hole inside the support.
        _, _, used_support, _, _ = geometry.prepare(
            holograms, np.zeros((16, 16)), support,
            {"crop": 2, "binning": 2, "roi": None},
        )
        outside = np.fft.fftshift(used_support) == 0
        for fit in (False, True):
            recipe = universal.default_universal_phase_retrieval_recipe()
            recipe.update(
                material_mask=material,
                fit_material_thickness=fit,
                zero_thickness_outside_support=True,
                zero_magnetization_outside_support=False,
                projection_constraints_inside_support_only=True,
                inner_mode=["ER"], inner_Nit=[1], warmup_Nit=0,
                outer_iterations=1, physical_iterations=1,
                crop=2, binning=2, final_fourier_constraint=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, components, _, _ = universal.universal_phase_retrieval_algorithm(
                    holograms, np.zeros((16, 16)), support,
                    ["positive", "negative"], [780, 780], [1, 1],
                    ["beam", "beam"], universal_recipe=recipe,
                )
            thickness = components["material_thickness"]
            np.testing.assert_array_equal(thickness[outside], 0)
            self.assertTrue(np.any(thickness[~outside] > 0))
            self.assertGreater(np.count_nonzero(thickness == 0), np.count_nonzero(outside))

    def test_direct_projection_requires_support_for_support_thickness_option(self):
        fields = np.ones((2, 8, 8), dtype=np.complex128)
        recipe = {"zero_thickness_outside_support": True, "physical_iterations": 1}
        with self.assertRaisesRegex(ValueError, "supportmask is required"):
            universal.project_fourier_fields_universal(
                fields, ["a", "b"], [780, 780], [1, 1], ["beam", "beam"],
                universal_recipe=recipe,
            )


if __name__ == "__main__":
    unittest.main()
