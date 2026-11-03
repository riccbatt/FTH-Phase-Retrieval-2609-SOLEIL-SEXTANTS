"""Real-space support and projection constraints for the phase-retrieval solver."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform


def _expand_support_about_center(support, factor):
    """Enlarge a 2-D support spatially about the array center."""
    support = np.asarray(support)
    if support.ndim != 2:
        raise ValueError("support must be 2D")
    if isinstance(factor, bool) or not isinstance(factor, (int, float, np.number)):
        raise TypeError("mode support expansion factors must be numeric")
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("mode support expansion factors must be finite and > 0")
    if factor == 1:
        return support.copy()

    inverse_scale = np.eye(2) / factor
    center = (np.asarray(support.shape, dtype=float) - 1) / 2
    offset = center - inverse_scale @ center
    expanded = affine_transform(
        support.astype(float),
        inverse_scale,
        offset=offset,
        output_shape=support.shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    if np.issubdtype(support.dtype, np.bool_) or np.array_equal(
        support, support.astype(bool)
    ):
        return (expanded > 0.5).astype(support.dtype)
    return expanded.astype(support.dtype, copy=False)


def _mode_supports(supportmask, mode_factors, shape_2d):
    """Build modal supports from support-size expansion factors."""
    supportmask = np.asarray(supportmask)
    nmodes = len(mode_factors)

    if supportmask.ndim == 2:
        return np.stack(
            [_expand_support_about_center(supportmask, factor) for factor in mode_factors],
            axis=0,
        )

    if supportmask.ndim == 3:
        if supportmask.shape[1:] != shape_2d:
            raise ValueError(
                f"supportmask has incompatible spatial shape {supportmask.shape[1:]}; "
                f"expected {shape_2d}."
            )
        if supportmask.shape[0] == nmodes:
            return supportmask.copy()
        if supportmask.shape[0] == 1:
            return np.repeat(supportmask, nmodes, axis=0)

    raise ValueError(
        "supportmask must be 2D, (1, nx, ny), or (Nmodes, nx, ny)."
    )


def _project_support_space(inv, prev, mask, proj_fn, beta, step_idx, Nit, alpha):
    """Support-space projection: enforce support constraints after the Fourier step."""
    if alpha > 0:
        try:
            from . import phase_retrieval_universal as universal
        except Exception:
            universal = None
        if universal is not None and hasattr(universal, "TV"):
            inv = inv + alpha * universal.TV(inv, 1)
    return proj_fn(inv, prev, mask, beta, step_idx, Nit)
