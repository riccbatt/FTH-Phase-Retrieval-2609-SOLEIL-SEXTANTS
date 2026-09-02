"""
CUDA-JIT front end for :mod:`library.phase_retrieval_universal`.

The public functions mirror the universal, general, and pure-energy drivers
while injecting ``PhaseRtrv_core_jit`` explicitly. All object models and
recipes therefore remain compatible with the definitive universal library.
"""

from functools import partial

from . import phase_retrieval_core_jit as jit_core
from . import phase_retrieval_universal as universal


cuda_jit_status = jit_core.cuda_jit_status
warm_up_jit = jit_core.warm_up_jit
PhaseRtrv_core_jit = jit_core.PhaseRtrv_core_jit
default_universal_phase_retrieval_recipe = (
    universal.default_universal_phase_retrieval_recipe
)


def _configured_kernel(fallback):
    """Return the JIT kernel with the selected unsupported-feature behavior."""
    return partial(jit_core.PhaseRtrv_core_jit, fallback=fallback)


def universal_phase_retrieval_algorithm_jit(*args, fallback=True, **kwargs):
    """
    Run the universal reconstruction with the CUDA-JIT phase-retrieval core.

    Set ``fallback=False`` to require every requested stage to use the JIT
    implementation and raise when CUDA or a requested feature is unsupported.
    """
    if "phase_retrieval_kernel" in kwargs:
        raise ValueError(
            "phase_retrieval_kernel is selected by the JIT front end."
        )
    return universal.universal_phase_retrieval_algorithm(
        *args,
        phase_retrieval_kernel=_configured_kernel(fallback),
        **kwargs,
    )


def multi_energy_phase_retrieval_algorithm_jit(
    *args,
    fallback=True,
    **kwargs,
):
    """Run the self-contained pure-energy driver with CUDA-JIT updates."""
    if "phase_retrieval_kernel" in kwargs:
        raise ValueError(
            "phase_retrieval_kernel is selected by the JIT front end."
        )
    return universal.multi_energy_phase_retrieval_algorithm(
        *args,
        phase_retrieval_kernel=_configured_kernel(fallback),
        **kwargs,
    )


def general_phase_retrieval_algorithm_jit(*args, fallback=True, **kwargs):
    """Run the metadata-aware physical driver with CUDA-JIT updates."""
    if "phase_retrieval_kernel" in kwargs:
        raise ValueError(
            "phase_retrieval_kernel is selected by the JIT front end."
        )
    return universal.general_phase_retrieval_algorithm(
        *args,
        phase_retrieval_kernel=_configured_kernel(fallback),
        **kwargs,
    )

