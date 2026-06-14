import unittest
from unittest import mock

import numpy as np

from library import phase_retrieval_core_jit as jit_core
from library import phase_retrieval_universal as universal
from library import phase_retrieval_universal_jit as universal_jit


class PhaseRetrievalJitTests(unittest.TestCase):
    def test_status_is_available_without_cuda(self):
        status = jit_core.cuda_jit_status()
        self.assertIn("available", status)
        self.assertEqual(status["backend"], "cupy_nvrtc")

    def test_core_falls_back_when_cuda_is_unavailable(self):
        expected = (
            np.ones((2, 2), dtype=complex),
            [1.0],
            [],
            None,
        )
        with (
            mock.patch.object(jit_core, "CUDA_JIT_AVAILABLE", False),
            mock.patch.object(
                universal,
                "PhaseRtrv_core",
                return_value=expected,
            ) as fallback,
        ):
            actual = jit_core.PhaseRtrv_core_jit(
                np.ones((2, 2)),
                np.ones((2, 2)),
            )

        fallback.assert_called_once()
        np.testing.assert_allclose(actual[0], expected[0])

    def test_core_can_require_cuda_without_fallback(self):
        with mock.patch.object(jit_core, "CUDA_JIT_AVAILABLE", False):
            with self.assertRaisesRegex(RuntimeError, "CUDA JIT"):
                jit_core.PhaseRtrv_core_jit(
                    np.ones((2, 2)),
                    np.ones((2, 2)),
                    fallback=False,
                )

    def test_universal_front_end_injects_jit_kernel(self):
        sentinel = object()
        with mock.patch.object(
            universal,
            "universal_phase_retrieval_algorithm",
            return_value=sentinel,
        ) as driver:
            result = (
                universal_jit.universal_phase_retrieval_algorithm_jit(
                    "holograms",
                    "mask",
                    "support",
                    fallback=False,
                )
            )

        self.assertIs(result, sentinel)
        kernel = driver.call_args.kwargs["phase_retrieval_kernel"]
        self.assertIs(kernel.func, jit_core.PhaseRtrv_core_jit)
        self.assertFalse(kernel.keywords["fallback"])

    def test_existing_driver_accepts_an_explicit_kernel(self):
        shape = (4, 4)
        holograms = np.ones((2, *shape))
        start_fields = np.ones((2, *shape), dtype=complex)

        def identity_kernel(*, Phase, **kwargs):
            return Phase, np.array([]), np.array([]), None

        fields, _, _, _ = universal.universal_phase_retrieval_algorithm(
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
            phase_retrieval_kernel=identity_kernel,
        )

        np.testing.assert_allclose(fields, start_fields)


if __name__ == "__main__":
    unittest.main()
