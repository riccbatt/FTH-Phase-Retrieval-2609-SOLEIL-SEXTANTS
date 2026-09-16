"""Registration, intensity calibration, and beamstop-aware image stitching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

try:
    from .data_loading import Frame
except ImportError:  # Notebook imports when ``library`` itself is on sys.path.
    from data_loading import Frame


@dataclass(frozen=True)
class PreparedFrame:
    image_id: str
    image: np.ndarray
    valid: np.ndarray
    exposure: float
    shift: tuple[float, float]
    coefficients: tuple[float, ...]
    original_index: int
    fit_input: np.ndarray | None = None


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


def _register(
    reference,
    moving,
    reference_valid,
    moving_valid,
    upsample_factor,
    max_shift,
    estimation_region=None,
):
    overlap = reference_valid & moving_valid
    if estimation_region is not None:
        overlap &= estimation_region
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


def shift_mask(mask: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    """Move a detector mask by ``(row_shift, column_shift)`` pixels."""
    shifted = ndi_shift(
        np.asarray(mask, dtype=float), shift, order=0, mode="constant", cval=0
    )
    return shifted > 0.5


def stitch_images(
    frames: Sequence[Frame],
    masks: Sequence[np.ndarray],
    *,
    register: bool = True,
    max_shift: float | tuple[float, float] | None = 10.0,
    upsample_factor: int = 10,
    fit_intensity: bool = True,
    fit_degree: int = 1,
    fit_percentiles: tuple[float, float] = (2.0, 98.0),
    estimation_roi: tuple[slice, slice] | None = None,
    estimation_rois: Sequence[tuple[slice, slice] | None] | None = None,
    use_master_where_valid: bool = False,
) -> StitchResult:
    """Align and average images of one scattering pattern.

    The first image is the reference. Registration and intensity calibration
    use only pixels that are valid in both images. ``estimation_rois`` may
    provide a separate ROI for every input; it takes precedence over the
    shared ``estimation_roi``. Masked pixels are excluded in either case.
    ``fit_degree=1`` fits a factor and offset; larger values fit a polynomial. If
    ``use_master_where_valid`` is true, auxiliary images contribute only where
    the first image is masked or otherwise invalid.
    """
    shape = _validate(frames, masks)
    if fit_intensity and (not isinstance(fit_degree, int) or fit_degree < 1):
        raise ValueError("fit_degree must be an integer of at least 1")
    identity_coefficients = (
        tuple([0.0] * (fit_degree - 1) + [1.0, 0.0])
        if fit_intensity
        else (1.0, 0.0)
    )
    reference = np.asarray(frames[0].image, dtype=float)
    reference_valid = (~np.asarray(masks[0], dtype=bool)) & np.isfinite(reference)
    if estimation_rois is not None and len(estimation_rois) != len(frames):
        raise ValueError("estimation_rois must have one entry per frame")

    def estimation_region_for(index):
        roi = estimation_rois[index] if estimation_rois is not None else estimation_roi
        region = np.ones(shape, dtype=bool)
        if roi is not None:
            region[:] = False
            region[roi] = True
            if region.sum() < 16:
                raise ValueError("Each estimation ROI must contain at least 16 pixels")
        return region
    prepared = [
        PreparedFrame(
            frames[0].image_id,
            reference.copy(),
            reference_valid.copy(),
            float(frames[0].exposure),
            (0.0, 0.0),
            identity_coefficients,
            0,
            reference.copy(),
        )
    ]

    for index in range(1, len(frames)):
        estimation_region = estimation_region_for(index)
        image = np.asarray(frames[index].image, dtype=float)
        valid = (~np.asarray(masks[index], dtype=bool)) & np.isfinite(image)
        measured_shift = (0.0, 0.0)
        coefficients = identity_coefficients

        if register:
            image, valid, measured_shift = _register(
                reference,
                image,
                reference_valid,
                valid,
                upsample_factor,
                max_shift,
                estimation_region,
            )
            valid &= np.isfinite(image)

        fit_input = image.copy()
        overlap = reference_valid & valid & estimation_region
        if fit_intensity:
            image, coefficients = _fit_intensity(
                reference, image, overlap, fit_degree, fit_percentiles
            )

        prepared.append(
            PreparedFrame(
                frames[index].image_id,
                image,
                valid,
                float(frames[index].exposure),
                measured_shift,
                coefficients,
                index,
                fit_input,
            )
        )

    image_sum = np.zeros(shape, dtype=float)
    source_count = np.zeros(shape, dtype=np.int32)
    if use_master_where_valid:
        master = prepared[0]
        master_usable = master.valid & np.isfinite(master.image)
        image_sum[master_usable] = master.image[master_usable]
        source_count[master_usable] = 1
        fill_region = ~master_usable
        for item in prepared[1:]:
            usable = fill_region & item.valid & np.isfinite(item.image)
            image_sum[usable] += item.image[usable]
            source_count[usable] += 1
    else:
        for item in prepared:
            usable = item.valid & np.isfinite(item.image)
            image_sum[usable] += item.image[usable]
            source_count[usable] += 1

    stitched = np.divide(
        image_sum,
        source_count,
        out=np.full(shape, np.nan, dtype=float),
        where=source_count > 0,
    )
    missing_mask = source_count == 0
    source_exposure = np.full(shape, np.nan, dtype=float)
    source_exposure[~missing_mask] = 1.0
    return StitchResult(
        stitched,
        missing_mask,
        source_count,
        source_exposure,
        1.0,
        tuple(frame.image_id for frame in frames),
        tuple(prepared),
    )


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
