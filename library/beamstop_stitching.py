"""Registration, intensity calibration, and beamstop-aware image stitching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

from .data_loading import Frame


@dataclass(frozen=True)
class PreparedFrame:
    image_id: str
    image: np.ndarray
    valid: np.ndarray
    exposure: float
    shift: tuple[float, float]
    coefficients: tuple[float, ...]
    original_index: int


@dataclass(frozen=True)
class StitchResult:
    image: np.ndarray
    missing_mask: np.ndarray
    source_count: np.ndarray
    source_exposure: np.ndarray
    reference_exposure: float
    ordered_ids: tuple[str, ...]
    prepared_frames: tuple[PreparedFrame, ...]


def _validate(frames: Sequence[Frame], masks: Sequence[np.ndarray]) -> tuple[int, int]:
    if not frames or len(frames) != len(masks):
        raise ValueError("frames and masks must have the same non-zero length")
    shape = np.asarray(frames[0].image).shape
    if len(shape) != 2:
        raise ValueError("Detector images must be 2-D")
    for frame, mask in zip(frames, masks):
        if np.asarray(frame.image).shape != shape or np.asarray(mask).shape != shape:
            raise ValueError(f"All images and masks must share shape {shape}")
        if not np.isfinite(frame.exposure) or frame.exposure <= 0:
            raise ValueError("All exposure times must be finite and positive")
    return shape


def _register(reference, moving, reference_valid, moving_valid, upsample_factor, max_shift):
    overlap = reference_valid & moving_valid
    if overlap.sum() < 16:
        raise ValueError("At least 16 common valid pixels are required for registration")
    ref_work = np.where(overlap, reference, np.nanmedian(reference[overlap]))
    mov_work = np.where(overlap, moving, np.nanmedian(moving[overlap]))
    measured, _, _ = phase_cross_correlation(
        ref_work, mov_work, upsample_factor=upsample_factor, normalization=None
    )
    measured = np.asarray(measured, dtype=float)
    if max_shift is not None:
        bounds = np.broadcast_to(np.asarray(max_shift, dtype=float), (2,))
        measured = np.clip(measured, -bounds, bounds)
    aligned = ndi_shift(moving, measured, order=1, mode="constant", cval=np.nan)
    aligned_valid = ndi_shift(
        moving_valid.astype(float), measured, order=0, mode="constant", cval=0
    ) > 0.5
    return aligned, aligned_valid, tuple(float(v) for v in measured)


def _fit_intensity(reference, moving, overlap, degree, fit_percentiles):
    x, y = moving[overlap], reference[overlap]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < max(16, degree + 2):
        raise ValueError("Not enough common valid pixels for intensity fitting")
    low, high = np.percentile(x, fit_percentiles)
    selected = (x >= low) & (x <= high)
    if selected.sum() < degree + 2:
        raise ValueError("Intensity-fit bounds leave too few pixels")
    coefficients = np.polyfit(x[selected], y[selected], degree)
    return np.polyval(coefficients, moving), tuple(float(v) for v in coefficients)


def stitch_exposures(
    frames: Sequence[Frame],
    masks: Sequence[np.ndarray],
    *,
    register: bool = False,
    max_shift: float | tuple[float, float] | None = 10.0,
    upsample_factor: int = 10,
    fit_degree: int | None = None,
    fit_percentiles: tuple[float, float] = (2.0, 98.0),
    exposure_rtol: float = 1e-6,
) -> StitchResult:
    """Combine exposure groups longest-first.

    Counts are divided by exposure before processing. Equal-exposure frames are
    averaged in every valid overlap. Shorter groups only fill pixels unavailable
    in longer groups. ``fit_degree=1`` selects a linear calibration; higher values
    fit nonlinearities. Registration and fit use only mutually valid pixels.
    """
    shape = _validate(frames, masks)
    order = sorted(range(len(frames)), key=lambda i: frames[i].exposure, reverse=True)
    reference_exposure = float(frames[order[0]].exposure)
    result = np.full(shape, np.nan, dtype=float)
    source_count = np.zeros(shape, dtype=np.int32)
    source_exposure = np.full(shape, np.nan, dtype=float)
    prepared = []

    groups: list[list[int]] = []
    for index in order:
        if not groups or not np.isclose(
            frames[index].exposure, frames[groups[-1][0]].exposure, rtol=exposure_rtol
        ):
            groups.append([])
        groups[-1].append(index)

    for group in groups:
        group_sum = np.zeros(shape, dtype=float)
        group_count = np.zeros(shape, dtype=np.int32)
        calibration = result / reference_exposure
        calibration_valid = np.isfinite(calibration)

        for position, index in enumerate(group):
            frame = frames[index]
            image = np.asarray(frame.image, dtype=float) / float(frame.exposure)
            valid = (~np.asarray(masks[index], dtype=bool)) & np.isfinite(image)
            shift = (0.0, 0.0)
            coefficients = (1.0, 0.0)
            if not calibration_valid.any() and position == 0:
                calibration, calibration_valid = image.copy(), valid.copy()
            else:
                if register:
                    image, valid, shift = _register(
                        calibration, image, calibration_valid, valid,
                        upsample_factor, max_shift,
                    )
                if fit_degree is not None:
                    image, coefficients = _fit_intensity(
                        calibration, image, calibration_valid & valid,
                        fit_degree, fit_percentiles,
                    )
            group_sum[valid] += image[valid]
            group_count[valid] += 1
            if np.isnan(result).all():
                calibration = np.divide(
                    group_sum, group_count, out=np.full(shape, np.nan), where=group_count > 0
                )
                calibration_valid = group_count > 0
            prepared.append(PreparedFrame(
                frame.image_id, image, valid, float(frame.exposure), shift,
                coefficients, index,
            ))

        group_average = np.divide(
            group_sum, group_count, out=np.full(shape, np.nan), where=group_count > 0
        )
        fill = np.isnan(result) & (group_count > 0)
        result[fill] = group_average[fill] * reference_exposure
        source_count[fill] = group_count[fill]
        source_exposure[fill] = frames[group[0]].exposure

    # A stitched pixel remains masked only when it was unavailable in every
    # aligned input: the logical intersection (AND) of all input masks.
    # ``PreparedFrame.valid`` also excludes non-finite and registration-edge
    # pixels, which cannot contribute usable detector data.
    missing_mask = np.logical_and.reduce([~item.valid for item in prepared])
    result[missing_mask] = np.nan

    return StitchResult(
        result, missing_mask, source_count, source_exposure,
        reference_exposure, tuple(frames[i].image_id for i in order), tuple(prepared)
    )
