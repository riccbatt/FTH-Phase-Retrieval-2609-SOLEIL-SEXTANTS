"""Public library surface for the physical-projection workflow.

The package intentionally exposes only the generic, paper-active modules and
keeps the API explicit instead of silently carrying legacy names.
"""

__all__ = [
    "kramers_kronig",
    "magnitude_constraints",
    "mask_store",
    "phase_retrieval_geometry",
    "phase_retrieval_physical",
    "phase_retrieval_universal",
    "real_space_constraints",
    "scan_io",
]
