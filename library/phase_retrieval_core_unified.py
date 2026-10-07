"""Compatibility import for the consolidated universal phase retrieval library.

All implementations live in phase_retrieval_universal. Alias the module itself
so legacy imports, private helpers and kernel patches retain their behavior.
New code should import phase_retrieval_universal directly.
"""
import sys

try:
    from . import phase_retrieval_universal as _universal
except ImportError:
    import phase_retrieval_universal as _universal

sys.modules[__name__] = _universal
