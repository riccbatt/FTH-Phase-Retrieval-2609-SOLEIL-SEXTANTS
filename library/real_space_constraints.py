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

from .phase_retrieval_backend import xp

def _proj_ER(inv, prev, mask, beta):
    """Error-reduction projection: keep only values inside the support."""
    return inv * mask


def _proj_SF(inv, prev, mask, beta):
    """Solvent-flipping projection."""
    return inv * (2 * mask - 1)


def _proj_hapre(inv, prev, mask, beta):
    """HAPRE projection step."""
    return inv + beta * (prev - 2 * inv) * (1 - mask)


def _proj_RAAR(inv, prev, mask, beta):
    """Relaxed averaged alternating reflections projection."""
    return inv + beta * (prev - 2 * inv) * (1 - mask) * (2 * inv - prev < 0)


def _proj_HIOs(inv, prev, mask, beta):
    """Simplified hybrid-input-output projection."""
    return inv + (1 - mask) * (prev - (beta + 1) * inv)


def _proj_HIO(inv, prev, mask, beta):
    """Hybrid-input-output projection with positivity condition."""
    return (
        inv
        + (1 - mask) * (prev - (beta + 1) * inv)
        + mask * (prev - (beta + 1) * inv) * (xp.real(inv) < 0)
    )


def _proj_CHIO(inv, prev, mask, beta):
    """Constrained hybrid-input-output projection using a fixed alpha2 parameter."""
    alpha2 = 0.4
    return (
        (prev - beta * inv)
        + mask * (xp.real(inv - alpha2 * prev) >= 0) * (-prev + (beta + 1) * inv)
        + (xp.real(-inv + alpha2 * prev) >= 0)
        * (xp.real(inv) >= 0)
        * ((beta - (1 - alpha2) / alpha2) * inv)
    )


def _proj_HPR(inv, prev, mask, beta):
    """Hybrid projection-reflection update."""
    return (
        inv
        + (1 - mask) * (prev - (beta + 1) * inv)
        + mask * (prev - (beta + 1) * inv) * (xp.real(prev - (beta - 3) * inv) > 0)
    )


PROJECTIONS = {
    "ER": _proj_ER,
    "SF": _proj_SF,
    "HAPRE": _proj_hapre,
    "RAAR": _proj_RAAR,
    "HIOs": _proj_HIOs,
    "HIO": _proj_HIO,
    "CHIO": _proj_CHIO,
    "HPR": _proj_HPR,
}



def total_variation_gradient(u, epsilon=1e-4):
    """TV gradient on spatial axes only; observation and mode axes stay independent."""
    gradients = [xp.diff(u, axis=axis, prepend=xp.take(u, [0], axis=axis))
                 for axis in (-2, -1)]
    magnitude = epsilon + xp.sqrt(sum(xp.abs(g)**2 for g in gradients))
    result = xp.zeros_like(u)
    for axis, gradient in zip((-2, -1), gradients):
        normalized = gradient / magnitude
        # Zero flux at the final edge matches the finite-difference boundary.
        tail = xp.zeros_like(xp.take(normalized, [-1], axis=axis))
        result -= xp.diff(normalized, axis=axis, append=tail)
    return result


def largest_support_component(supportmask):
    """Binary thickness template for the largest connected support aperture."""
    from scipy.ndimage import label

    support = np.asarray(supportmask)
    if support.ndim != 2:
        raise ValueError("supportmask must be a 2D array.")
    labels, count = label(support != 0, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        raise ValueError("supportmask has no nonzero aperture.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return (labels == np.argmax(sizes)).astype(np.uint8)


def _component_centered_modal_supports(supportmask, factors):
    """Expand each disconnected aperture in place for off-center samples."""
    from scipy.ndimage import affine_transform, center_of_mass, label

    support = np.asarray(supportmask) != 0
    labels, count = label(support, structure=np.ones((3, 3), dtype=int))
    centers = center_of_mass(support, labels, range(1, count + 1))
    result = []
    for factor in factors:
        if float(factor) == 1.0:
            result.append(support.astype(np.uint8))
            continue
        expanded = np.zeros(support.shape, dtype=bool)
        matrix = np.eye(2) / float(factor)
        for component, center in enumerate(centers, start=1):
            center = np.asarray(center, dtype=float)
            offset = center - matrix @ center
            expanded |= affine_transform(
                (labels == component).astype(np.uint8), matrix,
                offset=offset, output_shape=support.shape, order=0,
                mode="constant", cval=0, prefilter=False,
            ) != 0
        result.append(expanded.astype(np.uint8))
    return np.stack(result)
