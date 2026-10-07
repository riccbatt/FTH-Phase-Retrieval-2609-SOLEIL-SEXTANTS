"""Compatibility API for universal's consolidated multimode spectral driver.

Implementation, initialization, projections and kernels live in universal.
"""
try:
    from . import phase_retrieval_universal as _universal
    from .phase_retrieval_universal import *
except ImportError:
    import phase_retrieval_universal as _universal
    from phase_retrieval_universal import *

default_multi_energy_phase_retrieval_recipe = (
    _universal.default_multi_energy_multimode_phase_retrieval_recipe
)
multi_energy_phase_retrieval_algorithm = (
    _universal.multi_energy_multimode_phase_retrieval_algorithm
)
_run_energy_update_schedule = _universal._run_modal_energy_update_schedule

def __getattr__(name):
    return getattr(_universal, name)
