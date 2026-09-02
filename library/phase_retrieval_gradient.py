"""
Gradient-based refinement steps for phase retrieval fields.

This module is intentionally separate from the projection-based phase-retrieval
libraries.  The functions operate on the same returned Fourier-field convention
used by :mod:`phase_retrieval_core` and :mod:`phase_retrieval_universal`: fields
are shifted with the zero frequency at the array center.

The first implementation is a conservative NumPy optimizer.  It refines one
field or a stack of independent fields against the measured diffraction
amplitude or intensity, with an optional real-space support leakage penalty.
It is meant as a polishing step between or after normal ER/HAPRE/HIO style
iterations, not as a replacement for those algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


LossMode = Literal["amplitude", "intensity"]


@dataclass
class GradientRefinementResult:
    """Container returned by gradient-refinement helpers."""

    fields: np.ndarray
    loss: np.ndarray
    diffraction_loss: np.ndarray
    support_loss: np.ndarray


def fourier_field_to_object(field: ArrayLike) -> np.ndarray:
    """
    Convert a centered Fourier field to the real-space support convention.

    This mirrors the low-level phase-retrieval kernels, where the support-space
    image is computed as ``fft2(fftshift(field))``.
    """

    field = np.asarray(field)
    return np.fft.fft2(np.fft.fftshift(field, axes=(-2, -1)), axes=(-2, -1))


def fourier_field_to_display_object(field: ArrayLike) -> np.ndarray:
    """
    Convert a centered Fourier field to a centered real-space display object.

    This is ``fftshift(fourier_field_to_object(field))`` on the last two axes,
    which is the frame expected by normal ``imshow`` inspection with a centered
    support mask.
    """

    obj = fourier_field_to_object(field)
    return np.fft.fftshift(obj, axes=(-2, -1))


def object_to_fourier_field(obj: ArrayLike) -> np.ndarray:
    """Convert a support-space object back to a centered Fourier field."""

    obj = np.asarray(obj)
    return np.fft.ifftshift(np.fft.ifft2(obj, axes=(-2, -1)), axes=(-2, -1))


def display_object_to_fourier_field(obj: ArrayLike) -> np.ndarray:
    """Convert a centered display object back to a centered Fourier field."""

    obj = np.asarray(obj)
    support_obj = np.fft.ifftshift(obj, axes=(-2, -1))
    return object_to_fourier_field(support_obj)


def diffraction_loss(
    field: ArrayLike,
    measurement: ArrayLike,
    mask_pixel: ArrayLike | None = None,
    *,
    loss_mode: LossMode = "amplitude",
    eps: float = 1e-12,
) -> float:
    """
    Return the mean diffraction mismatch on unmasked pixels.

    Parameters
    ----------
    field
        Centered Fourier-domain field.
    measurement
        Measured diffraction amplitude for ``loss_mode="amplitude"`` or
        intensity for ``loss_mode="intensity"``.
    mask_pixel
        Beamstop / invalid-pixel mask. Nonzero pixels are excluded.
    loss_mode
        ``"amplitude"`` minimizes ``|field|`` against ``measurement``.
        ``"intensity"`` minimizes ``|field|**2`` against ``measurement``.
    eps
        Small denominator guard.
    """

    field = np.asarray(field)
    measurement = np.asarray(measurement, dtype=float)
    weights = _valid_pixel_weights(measurement.shape, mask_pixel)
    if loss_mode == "amplitude":
        residual = np.abs(field) - measurement
    elif loss_mode == "intensity":
        residual = np.abs(field) ** 2 - measurement
    else:
        raise ValueError("loss_mode must be 'amplitude' or 'intensity'")
    denom = np.sum(weights) + eps
    return float(0.5 * np.sum(weights * residual**2) / denom)


def support_loss(field: ArrayLike, supportmask: ArrayLike, eps: float = 1e-12) -> float:
    """
    Return the mean object power outside ``supportmask``.

    ``supportmask`` follows the same user-facing convention as
    ``PhaseRtrv_core``: it is supplied in the centered display frame and shifted
    internally before being applied to ``fourier_field_to_object(field)``.
    """

    support = _supportmask_to_object_frame(supportmask, np.asarray(field).shape)
    outside = ~support
    obj = fourier_field_to_object(field)
    denom = np.sum(outside) + eps
    return float(0.5 * np.sum(outside * np.abs(obj) ** 2) / denom)


def refine_field_gradient(
    field: ArrayLike,
    measurement: ArrayLike,
    supportmask: ArrayLike | None = None,
    mask_pixel: ArrayLike | None = None,
    *,
    n_steps: int = 25,
    learning_rate: float = 0.05,
    loss_mode: LossMode = "amplitude",
    support_weight: float = 0.0,
    support_projection: bool = False,
    fourier_projection: bool = False,
    clip_update: float | None = 0.25,
    eps: float = 1e-12,
) -> GradientRefinementResult:
    """
    Refine a single Fourier field with gradient-descent steps.

    The update combines a diffraction-data gradient with an optional support
    leakage gradient.  ``support_projection=True`` additionally zeros the object
    outside the support after every step, behaving like a soft gradient polish
    plus an ER-style support projection.
    """

    current = np.asarray(field, dtype=np.complex128).copy()
    measurement = np.asarray(measurement, dtype=float)
    if current.shape != measurement.shape:
        raise ValueError("field and measurement must have the same shape")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    learning_rate_schedule = _normalize_step_schedule(
        learning_rate,
        int(n_steps),
        name="learning_rate",
        minimum=0.0,
        include_minimum=False,
    )
    support_weight_schedule = _normalize_step_schedule(
        support_weight,
        int(n_steps),
        name="support_weight",
        minimum=0.0,
        include_minimum=True,
    )

    support = (
        None
        if supportmask is None
        else _supportmask_to_object_frame(supportmask, current.shape)
    )

    weights = _valid_pixel_weights(current.shape, mask_pixel)
    if np.sum(weights) == 0:
        raise ValueError("mask_pixel excludes every diffraction pixel")

    total_history = []
    diffraction_history = []
    support_history = []

    for step in range(int(n_steps)):
        d_loss, grad = _diffraction_gradient(
            current,
            measurement,
            weights,
            loss_mode=loss_mode,
            eps=eps,
        )
        s_loss = 0.0
        support_weight_step = support_weight_schedule[step]
        if support is not None and support_weight_step > 0:
            s_loss, s_grad = _support_gradient(current, support, eps=eps)
            grad = grad + support_weight_step * s_grad

        update = learning_rate_schedule[step] * grad
        if clip_update is not None:
            update = _clip_complex_update(update, current, float(clip_update), eps)
        current = current - update

        if support_projection and support is not None:
            obj = fourier_field_to_object(current)
            obj = np.where(support, obj, 0.0)
            current = object_to_fourier_field(obj)

        if fourier_projection:
            current = _project_fourier_amplitude(
                current,
                measurement,
                weights,
                loss_mode=loss_mode,
                eps=eps,
            )

        total_history.append(d_loss + support_weight_step * s_loss)
        diffraction_history.append(d_loss)
        support_history.append(s_loss)

    return GradientRefinementResult(
        fields=current,
        loss=np.asarray(total_history, dtype=float),
        diffraction_loss=np.asarray(diffraction_history, dtype=float),
        support_loss=np.asarray(support_history, dtype=float),
    )


def refine_stack_gradient(
    fields: ArrayLike,
    measurements: ArrayLike,
    supportmask: ArrayLike | None = None,
    mask_pixel: ArrayLike | None = None,
    **kwargs,
) -> GradientRefinementResult:
    """
    Refine a stack of independent Fourier fields.

    ``fields`` and ``measurements`` must have shape ``(n, ny, nx)``.  A 2D
    ``mask_pixel`` is broadcast to every field; a 3D mask is applied per field.
    """

    fields = np.asarray(fields)
    measurements = np.asarray(measurements, dtype=float)
    if fields.ndim != 3 or measurements.ndim != 3:
        raise ValueError("fields and measurements must be 3D stacks")
    if fields.shape != measurements.shape:
        raise ValueError("fields and measurements must have matching shapes")

    refined = np.empty_like(fields, dtype=np.complex128)
    losses = []
    diffraction_losses = []
    support_losses = []
    for index in range(fields.shape[0]):
        mask_i = _select_stack_mask(mask_pixel, index)
        result = refine_field_gradient(
            fields[index],
            measurements[index],
            supportmask=supportmask,
            mask_pixel=mask_i,
            **kwargs,
        )
        refined[index] = result.fields
        losses.append(result.loss)
        diffraction_losses.append(result.diffraction_loss)
        support_losses.append(result.support_loss)

    return GradientRefinementResult(
        fields=refined,
        loss=np.stack(losses),
        diffraction_loss=np.stack(diffraction_losses),
        support_loss=np.stack(support_losses),
    )


def _valid_pixel_weights(shape, mask_pixel):
    if mask_pixel is None:
        return np.ones(shape, dtype=float)
    mask = np.asarray(mask_pixel)
    if mask.shape != tuple(shape):
        raise ValueError("mask_pixel must have the same shape as the field")
    return (mask == 0).astype(float)


def _select_stack_mask(mask_pixel, index):
    if mask_pixel is None:
        return None
    mask = np.asarray(mask_pixel)
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        return mask[index]
    raise ValueError("mask_pixel must be 2D, 3D, or None")


def _normalize_step_schedule(values, n_steps, *, name, minimum, include_minimum):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_steps, float(arr), dtype=float)
    elif arr.shape != (n_steps,):
        raise ValueError(f"{name} must be a scalar or have length n_steps")

    if include_minimum:
        valid = arr >= minimum
        comparison = f">= {minimum}"
    else:
        valid = arr > minimum
        comparison = f"> {minimum}"

    if not np.all(np.isfinite(arr)) or not np.all(valid):
        raise ValueError(f"{name} values must be finite and {comparison}")
    return arr


def _supportmask_to_object_frame(supportmask, shape):
    support = np.asarray(supportmask) != 0
    if support.shape != tuple(shape):
        raise ValueError("supportmask must have the same shape as field")
    return np.fft.fftshift(support, axes=(-2, -1))


def _diffraction_gradient(field, measurement, weights, *, loss_mode, eps):
    denom = np.sum(weights) + eps
    amplitude = np.abs(field)
    if loss_mode == "amplitude":
        residual = amplitude - measurement
        loss = 0.5 * np.sum(weights * residual**2) / denom
        grad = weights * residual * field / (amplitude + eps) / denom
    elif loss_mode == "intensity":
        intensity = amplitude**2
        residual = intensity - measurement
        loss = 0.5 * np.sum(weights * residual**2) / denom
        grad = 2.0 * weights * residual * field / denom
    else:
        raise ValueError("loss_mode must be 'amplitude' or 'intensity'")
    return float(loss), grad


def _support_gradient(field, support, eps):
    outside = ~support
    denom = np.sum(outside) + eps
    if np.sum(outside) == 0:
        return 0.0, np.zeros_like(field)
    obj = fourier_field_to_object(field)
    residual_obj = outside * obj
    loss = 0.5 * np.sum(np.abs(residual_obj) ** 2) / denom
    grad = object_to_fourier_field(residual_obj) / denom
    return float(loss), grad


def _clip_complex_update(update, field, fraction, eps):
    if fraction <= 0:
        raise ValueError("clip_update must be positive or None")
    max_update = fraction * (np.median(np.abs(field)) + eps)
    update_abs = np.abs(update)
    scale = np.minimum(1.0, max_update / (update_abs + eps))
    return update * scale


def _project_fourier_amplitude(field, measurement, weights, *, loss_mode, eps):
    if loss_mode == "amplitude":
        target_amplitude = measurement
    elif loss_mode == "intensity":
        target_amplitude = np.sqrt(np.maximum(measurement, 0.0))
    else:
        raise ValueError("loss_mode must be 'amplitude' or 'intensity'")
    projected = target_amplitude * np.exp(1j * np.angle(field))
    return np.where(weights > 0, projected, field)
