"""Readable building blocks for the 3-D detector preprocessing notebook."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

try:
    from .mask_store import load_red_mask_png
except ImportError:  # Notebook imports when ``library`` itself is on sys.path.
    from mask_store import load_red_mask_png


@dataclass
class HorizontalBandFit:
    """Components of the fitted vertical detector profile."""

    rows: np.ndarray
    measured_profile: np.ndarray
    fitted_profile: np.ndarray
    polynomial_profile: np.ndarray
    band_profile: np.ndarray
    band_image: np.ndarray
    polynomial_image: np.ndarray
    polynomial_coefficients: np.ndarray
    band_amplitude: float
    band_center: float
    band_width: float
    band_edge: float


def load_detector_masks(
    folder: str | Path,
    image_id: int,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load detector and beamstop PNGs and return their binary union.

    A value of one always means unusable. The expected filenames are
    ``mask_detector.png`` and ``mask_beamstop_<image_id>.png``.
    """
    folder = Path(folder)
    detector_mask = load_red_mask_png(
        folder / "mask_detector.png", expected_shape
    )
    beamstop_mask = load_red_mask_png(
        folder / f"mask_beamstop_{image_id}.png", expected_shape
    )
    pixel_mask = np.clip(detector_mask + beamstop_mask, 0, 1).astype(np.uint8)
    return detector_mask, beamstop_mask, pixel_mask


def fit_dark_frame(
    frame: np.ndarray,
    dark: np.ndarray,
    rows: slice,
    columns: slice,
    percentile: float = 100,
    stride: int = 1,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Fit ``frame = scale * dark + offset`` within a detector region."""
    frame = np.asarray(frame, dtype=float)
    dark = np.asarray(dark, dtype=float)
    image_values = frame[rows, columns][::stride, ::stride].ravel()
    dark_values = dark[rows, columns][::stride, ::stride].ravel()
    used = np.isfinite(image_values) & np.isfinite(dark_values)
    if mask is not None:
        mask_values = np.asarray(mask)[rows, columns][::stride, ::stride].ravel()
        used &= mask_values == 0
    if not np.any(used):
        raise ValueError("The dark-fit region contains no usable pixels")
    intensity_limit = np.percentile(image_values[used], percentile)
    used &= image_values <= intensity_limit
    if used.sum() < 2 or np.ptp(dark_values[used]) == 0:
        scale, offset = 1.0, 0.0
    else:
        scale, offset = np.polyfit(dark_values[used], image_values[used], 1)
    corrected = frame - (scale * dark + offset)
    return corrected, float(scale), float(offset), dark_values, used


def fit_horizontal_band(
    pattern: np.ndarray,
    edge_columns: int = 20,
    skipped_edge_columns: int = 0,
    polynomial_order: int = 2,
    band_center: float | None = None,
    band_width: float | None = None,
    band_edge: float | None = None,
    band_amplitude: float | None = None,
    mask: np.ndarray | None = None,
) -> HorizontalBandFit:
    """Fit a polynomial profile plus a negative smoothed horizontal band.

    The vertical profile is the mean of strips at the left and right detector
    edges. Masked pixels are ignored. Subtract ``result.band_image`` from every
    frame to remove only the fitted negative band; the polynomial is retained.
    """
    pattern = np.asarray(pattern, dtype=float)
    if pattern.ndim != 2:
        raise ValueError("pattern must be a 2-D image")
    if polynomial_order < 0:
        raise ValueError("polynomial_order must be at least zero")
    if edge_columns < 1:
        raise ValueError("edge_columns must be positive")

    row_count, column_count = pattern.shape
    rows = np.arange(row_count, dtype=float)
    centered_rows = rows - row_count / 2
    valid_pixels = np.isfinite(pattern)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != pattern.shape:
            raise ValueError("mask and pattern must have the same shape")
        valid_pixels &= mask == 0

    left = slice(skipped_edge_columns, skipped_edge_columns + edge_columns)
    right_stop = column_count - skipped_edge_columns
    right = slice(right_stop - edge_columns, right_stop)

    def strip_profile(columns: slice) -> np.ndarray:
        values = np.where(valid_pixels[:, columns], pattern[:, columns], np.nan)
        return np.nanmean(values, axis=1)

    measured_profile = np.nanmean(
        np.stack((strip_profile(left), strip_profile(right))), axis=0
    )
    valid_rows = np.isfinite(measured_profile)
    if valid_rows.sum() <= polynomial_order + 4:
        raise ValueError("Not enough valid detector rows for the band fit")

    if band_center is None:
        band_center = row_count / 2
    if band_width is None:
        band_width = row_count * 0.08
    if band_edge is None:
        band_edge = max(2.0, band_width / 10)

    initial_polynomial = np.polyfit(
        centered_rows[valid_rows], measured_profile[valid_rows], polynomial_order
    )[::-1]
    polynomial_guess = np.polynomial.polynomial.polyval(
        centered_rows, initial_polynomial
    )
    if band_amplitude is None:
        distance = np.abs(rows - band_center)
        inside = valid_rows & (distance < band_width / 2)
        outside = valid_rows & (distance > band_width) & (distance < 2 * band_width)
        if inside.any() and outside.any():
            band_amplitude = np.nanmedian(
                measured_profile[outside] - polynomial_guess[outside]
            ) - np.nanmedian(measured_profile[inside] - polynomial_guess[inside])
        else:
            band_amplitude = np.nanstd(measured_profile[valid_rows])
        band_amplitude = max(float(band_amplitude), np.finfo(float).eps)

    def smooth_box(y, amplitude, center, width, edge_sigma):
        lower_edge = center - width / 2
        upper_edge = center + width / 2
        return amplitude / 2 * (
            erf((y - lower_edge) / (np.sqrt(2) * edge_sigma))
            - erf((y - upper_edge) / (np.sqrt(2) * edge_sigma))
        )

    def model(y, *parameters):
        coefficients = parameters[: polynomial_order + 1]
        amplitude, center, width, edge_sigma = parameters[polynomial_order + 1 :]
        polynomial = np.polynomial.polynomial.polyval(
            y - row_count / 2, coefficients
        )
        return polynomial - smooth_box(y, amplitude, center, width, edge_sigma)

    initial = [
        *initial_polynomial,
        band_amplitude,
        band_center,
        band_width,
        band_edge,
    ]
    lower = [-np.inf] * (polynomial_order + 1) + [0, 0, 1, 0.2]
    upper = [np.inf] * (polynomial_order + 1) + [
        np.inf,
        row_count,
        row_count,
        row_count / 2,
    ]
    fitted, _ = curve_fit(
        model,
        rows[valid_rows],
        measured_profile[valid_rows],
        p0=initial,
        bounds=(lower, upper),
        maxfev=50_000,
    )
    coefficients = fitted[: polynomial_order + 1]
    amplitude, center, width, edge_sigma = fitted[polynomial_order + 1 :]
    polynomial_profile = np.polynomial.polynomial.polyval(
        centered_rows, coefficients
    )
    band_profile = -smooth_box(rows, amplitude, center, width, edge_sigma)
    fitted_profile = polynomial_profile + band_profile
    band_image = np.repeat(band_profile[:, None], column_count, axis=1)
    polynomial_image = np.repeat(polynomial_profile[:, None], column_count, axis=1)
    return HorizontalBandFit(
        rows=rows,
        measured_profile=measured_profile,
        fitted_profile=fitted_profile,
        polynomial_profile=polynomial_profile,
        band_profile=band_profile,
        band_image=band_image,
        polynomial_image=polynomial_image,
        polynomial_coefficients=np.asarray(coefficients),
        band_amplitude=float(amplitude),
        band_center=float(center),
        band_width=float(width),
        band_edge=float(edge_sigma),
    )
