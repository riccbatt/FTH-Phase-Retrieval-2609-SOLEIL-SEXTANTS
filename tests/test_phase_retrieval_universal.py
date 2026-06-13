import ast
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from library import phase_retrieval_core_multienergy as multienergy
from library import phase_retrieval_universal as universal


class PhaseRetrievalUniversalTests(unittest.TestCase):
    def test_physical_projection_recovers_mixed_dataset(self):
        shape = (3, 4)
        common = {
            "beam_a": np.full(shape, 0.2 + 0.1j),
            "beam_b": np.full(shape, -0.1 + 0.04j),
        }
        charge = {"e1": -0.02 + 0.01j, "e2": 0.02 - 0.01j}
        magnetic = {"e1": 0.03 - 0.06j, "e2": 0.05 - 0.08j}
        mz = {
            "sat": np.ones(shape),
            "domains": np.linspace(-0.8, 0.8, np.prod(shape)).reshape(shape),
        }
        observations = [
            (state, energy, polarization, beam)
            for beam in common
            for energy in charge
            for state in mz
            for polarization in (1, -1)
        ]
        logs = np.stack([
            common[beam]
            + charge[energy]
            + polarization * magnetic[energy] * mz[state]
            for state, energy, polarization, beam in observations
        ])
        fields = multienergy.object_log_to_fourier_field(logs)

        projected, components = universal.project_fourier_fields_universal(
            fields,
            state_labels=[item[0] for item in observations],
            energy_labels=[item[1] for item in observations],
            polarization_coefficients=[item[2] for item in observations],
            illumination_labels=[item[3] for item in observations],
            saturated_states={"sat": 1},
            universal_recipe={"physical_iterations": 30},
            return_components=True,
        )

        np.testing.assert_allclose(projected, fields, atol=1e-8)
        np.testing.assert_allclose(
            components["magnetization_by_state"]["domains"],
            mz["domains"],
            atol=1e-8,
        )

    def test_kt_bounds_are_applied_with_correct_sign_convention(self):
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update({
            "charge_kt_beta_range": (0.01, 0.02),
            "charge_kt_delta_range": (0.03, 0.04),
        })

        translated = universal._apply_kt_product_bounds(recipe)

        self.assertEqual(
            translated["charge_response_real_range"],
            (0.01, 0.02),
        )
        self.assertEqual(
            translated["charge_response_imag_range"],
            (-0.04, -0.03),
        )

    def test_known_refractive_index_spectrum_is_converted_to_q(self):
        recipe = universal.default_universal_phase_retrieval_recipe()
        recipe.update({
            "energy_values": np.array([100.0, 200.0]),
            "thickness": 10e-9,
            "known_charge_beta_spectrum": np.array([1e-3, 2e-3]),
            "known_charge_delta_spectrum": np.array([3e-3, 4e-3]),
        })

        translated = universal._prepare_physical_recipe(recipe, 2)
        kt = (
            recipe["energy_values"]
            * universal._EV_TO_WAVENUMBER_PER_METRE
            * recipe["thickness"]
        )

        np.testing.assert_allclose(
            translated["known_charge_beta_spectrum"],
            kt * recipe["known_charge_beta_spectrum"],
        )
        np.testing.assert_allclose(
            translated["known_charge_delta_spectrum"],
            -kt * recipe["known_charge_delta_spectrum"],
        )

    def test_svd_projection_matches_multi_energy_projection(self):
        rng = np.random.default_rng(4)
        fields = (
            rng.normal(size=(4, 5, 6))
            + 1j * rng.normal(size=(4, 5, 6))
        )
        settings = {
            "projection_model": "svd",
            "rank": 2,
            "projection_static_mode": "mean",
            "projection_relaxation": 0.7,
        }

        actual = universal.project_fourier_fields_universal(
            fields,
            state_labels=["state"] * 4,
            energy_labels=[0, 1, 2, 3],
            polarization_coefficients=[1] * 4,
            illumination_labels=["beam"] * 4,
            universal_recipe=settings,
        )
        expected = multienergy.project_fourier_fields_multi_energy(
            fields,
            projection_model="svd",
            rank=2,
            static_mode="mean",
            relaxation=0.7,
        )

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_svd_driver_matches_multi_energy_driver(self):
        rng = np.random.default_rng(5)
        holograms = rng.uniform(0.5, 2.0, size=(3, 5, 5))
        start_fields = np.sqrt(holograms) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, size=holograms.shape)
        )
        universal_recipe = {
            "projection_model": "svd",
            "rank": 1,
            "inner_mode": "ER",
            "inner_Nit": 1,
            "outer_iterations": 1,
            "warmup_Nit": 0,
            "shuffle_observations": False,
            "final_fourier_constraint": False,
        }
        multi_recipe = {
            "projection_model": "svd",
            "rank": 1,
            "inner_mode": "ER",
            "inner_Nit": 1,
            "outer_iterations": 1,
            "warmup_Nit": 0,
            "shuffle_energies": False,
            "final_fourier_constraint": False,
        }

        actual, _, _, _ = universal.universal_phase_retrieval_algorithm(
            holograms,
            np.zeros((5, 5), dtype=int),
            np.ones((5, 5)),
            state_labels=["state"] * 3,
            energy_labels=[0, 1, 2],
            polarization_coefficients=[1] * 3,
            illumination_labels=["beam"] * 3,
            universal_recipe=universal_recipe,
            start_fields=start_fields,
        )
        expected, _, _, _ = (
            multienergy.multi_energy_phase_retrieval_algorithm(
                holograms,
                np.zeros((5, 5), dtype=int),
                np.ones((5, 5)),
                multi_energy_recipe=multi_recipe,
                start_fields=start_fields,
            )
        )

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_mixed_metadata_rejects_multi_energy_only_projection(self):
        fields = np.ones((2, 4, 4), dtype=complex)
        with self.assertRaisesRegex(ValueError, "pure energy scan"):
            universal.project_fourier_fields_universal(
                fields,
                state_labels=["s1", "s2"],
                energy_labels=["e1", "e2"],
                polarization_coefficients=[1, 1],
                illumination_labels=["beam", "beam"],
                universal_recipe={"projection_model": "svd"},
            )

    def test_universal_module_imports_no_phase_retrieval_library(self):
        source = Path(universal.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or "")

        forbidden = [
            name
            for name in imported_modules
            if "phase_retrieval" in name
        ]
        self.assertEqual(forbidden, [])

    def test_no_projection_runs_with_local_phase_retrieval_kernel(self):
        shape = (4, 4)
        holograms = np.ones((2, *shape))
        start_fields = np.ones((2, *shape), dtype=complex)

        with mock.patch.object(
            universal,
            "PhaseRtrv_core",
            side_effect=lambda *, Phase, **kwargs: (
                Phase,
                np.array([]),
                np.array([]),
                None,
            ),
        ):
            fields, components, _, errors = (
                universal.universal_phase_retrieval_algorithm(
                    holograms,
                    np.zeros(shape, dtype=int),
                    np.ones(shape),
                    state_labels=["state", "state"],
                    energy_labels=["energy", "energy"],
                    polarization_coefficients=[1, -1],
                    illumination_labels=["beam", "beam"],
                    universal_recipe={
                        "projection_model": "none",
                        "outer_iterations": 1,
                        "warmup_Nit": 0,
                        "shuffle_observations": False,
                        "final_fourier_constraint": False,
                    },
                    start_fields=start_fields,
                )
            )

        np.testing.assert_allclose(fields, start_fields)
        self.assertEqual(components["universal_projection_model"], "none")
        self.assertIn("observation_metadata", errors)


if __name__ == "__main__":
    unittest.main()
