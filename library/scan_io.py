"""Generic scan-data helpers for raw detector streams and averaged hologram caches.

This module intentionally avoids detector-specific assumptions such as XPCS-only
coordinate conventions or beamstop geometry; callers pass the mask arrays they
need instead of baking scan geometry into the library itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import shift

DATASET = "entry/instrument/detector/data"

__all__ = [
    "manifest",
    "frame_mean",
    "load_dark",
    "estimate_hologram_shift",
    "prepare_cache",
]


def manifest(raw_root, scans):
    """Build a generic acquisition manifest for a detector stream.

    The manifest is intentionally generic: it records raw-file metadata and the
    detector shape, exposure, and point label without hard-coding XPCS or beamline
    conventions into the library.
    """
    records = []
    for scan in scans:
        stem = f"{scan:05d}" if not isinstance(scan, str) else scan
        master = Path(raw_root) / f"{stem}.nxs"
        if not master.exists():
            raise FileNotFoundError(f"Missing master file for scan {scan}: {master}")
        with h5py.File(master) as f:
            group = f["scan/instrument/collection"]
            points = np.asarray(group["point_nb"][()], int)
            current = np.asarray(group["m_caena"][()], float)
            energy = float(np.asarray(group["mono"][()]).ravel()[0])
        if len(set(points)) != len(points) or len(points) != len(current):
            raise ValueError("Nonunique points or mismatched current metadata")
        paths = list((Path(raw_root) / stem / "marana").glob("*.nx"))
        lookup = {}
        for path in paths:
            number = int(re.search(r"_(\d+)\.nx$", path.name).group(1))
            if number in lookup:
                raise ValueError(f"Duplicate detector point {number} in {path.parent}")
            lookup[number] = path
        if set(lookup) != set(points):
            raise ValueError("Detector files and point metadata differ")
        for point, amp in zip(points, current):
            with h5py.File(lookup[point]) as f:
                data = f[DATASET]
                exposure = float(np.asarray(f["entry/instrument/detector/exposure"][()]).ravel()[0])
                shape, frames = data.shape[-2:], data.shape[0]
            if exposure <= 0 or frames < 1:
                raise ValueError("Invalid exposure/frame count")
            records.append(
                dict(
                    scan=int(scan),
                    point=int(point),
                    current_A=float(amp),
                    energy_ev=energy,
                    path=str(lookup[point]),
                    exposure_s=exposure,
                    frames=frames,
                    shape=list(shape),
                    label=f"scan{scan}_point{point:04d}",
                )
            )
    if not records:
        raise ValueError("No observations")
    if len({r["label"] for r in records}) != len(records):
        raise ValueError("Duplicate scan selection")
    return records


def frame_mean(path, saturation=65535):
    """Average a detector file and flag saturated pixels across all frames."""
    with h5py.File(path) as f:
        ds = f[DATASET]
        total = np.zeros(ds.shape[-2:], float)
        invalid = np.zeros(total.shape, bool)
        for i in range(ds.shape[0]):
            frame = np.asarray(ds[i], float)
            invalid |= ~np.isfinite(frame) | (frame >= saturation)
            total += np.nan_to_num(frame)
        return total / ds.shape[0], invalid, ds.shape[0]


def load_dark(paths, shape, exposure_s):
    """Load a matched dark correction for a detector exposure group."""
    if not paths:
        return None, {"applied": False, "files": [], "frame_count": 0}
    total = np.zeros(shape, float)
    count = 0
    for path in paths:
        with h5py.File(path) as f:
            exp = float(np.asarray(f["entry/instrument/detector/exposure"][()]).ravel()[0])
        if not np.isclose(exp, exposure_s, rtol=0.01, atol=1e-6):
            raise ValueError(f"Dark exposure mismatch: {path}; supply matched darks")
        mean, _, frames = frame_mean(path)
        if mean.shape != tuple(shape):
            raise ValueError("Dark detector shape mismatch")
        total += mean * frames
        count += frames
    return total / count, {"applied": True, "files": [str(p) for p in paths], "frame_count": count}


def estimate_hologram_shift(reference, moving, reference_mask, moving_mask, *, max_shift=10.0, upsample_factor=20, roi=None):
    """Return corrective (row, column) raw-pixel shift and Pearson scores."""
    from scipy.ndimage import minimum_filter
    from scipy.optimize import minimize
    from skimage.registration import phase_cross_correlation

    if not np.isfinite(max_shift) or max_shift <= 0:
        raise ValueError("alignment max_shift must be positive and finite")
    if isinstance(upsample_factor, bool) or not isinstance(upsample_factor, (int, np.integer)) or upsample_factor < 1:
        raise ValueError("alignment upsample_factor must be a positive integer")

    reference, moving = np.asarray(reference, float), np.asarray(moving, float)
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError("Alignment requires matching 2D holograms")

    valid_ref = ~np.asarray(reference_mask, bool) & np.isfinite(reference)
    valid_mov = ~np.asarray(moving_mask, bool) & np.isfinite(moving)
    if roi is not None:
        if len(roi) != 4 or any(not isinstance(v, (int, np.integer)) for v in roi):
            raise ValueError("alignment roi must be (row_start, row_stop, col_start, col_stop)")
        r0, r1, c0, c1 = roi
        if not (0 <= r0 < r1 <= reference.shape[0] and 0 <= c0 < c1 <= reference.shape[1]):
            raise ValueError("alignment roi is outside the detector")
        region = np.s_[r0:r1, c0:c1]
        reference, moving, valid_ref, valid_mov = [a[region] for a in (reference, moving, valid_ref, valid_mov)]

    margin = int(np.ceil(max_shift)) + 1
    safe = valid_ref & minimum_filter(valid_mov.astype(np.uint8), size=2 * margin + 1, mode="constant", cval=0).astype(bool)
    rows, cols = np.nonzero(safe)
    if len(rows) < 64:
        raise ValueError("Too few valid pixels for alignment; enlarge ROI or reduce max_shift")

    target = reference[safe]
    target = target - target.mean()
    norm = np.linalg.norm(target)
    if norm <= 1e-12:
        raise ValueError("Reference has no contrast for alignment")

    from scipy.ndimage import map_coordinates

    moving = np.where(valid_mov, moving, 0.0)

    def score(delta):
        values = map_coordinates(moving, [rows - delta[0], cols - delta[1]], order=1, prefilter=False)
        values -= values.mean()
        denominator = norm * np.linalg.norm(values)
        return float(np.dot(target, values) / denominator) if denominator > 1e-12 else -1.0

    if np.std(moving[valid_mov]) <= 1e-12:
        raise ValueError("Moving hologram has no contrast for alignment")

    seed = phase_cross_correlation(
        np.where(valid_ref, reference, 0.0),
        moving,
        reference_mask=valid_ref,
        moving_mask=valid_mov,
        overlap_ratio=0.5,
    )[0]
    if np.any(np.abs(seed) > max_shift):
        raise ValueError(f"Estimated alignment shift {seed} exceeds max_shift={max_shift}")

    result = minimize(
        lambda delta: -score(delta),
        seed,
        method="Powell",
        bounds=[(max(-max_shift, v - 1.5), min(max_shift, v + 1.5)) for v in seed],
        options={"xtol": 1 / upsample_factor, "ftol": 1e-9},
    )
    if not result.success or np.any(np.abs(result.x) >= max_shift - 1 / upsample_factor):
        raise ValueError("Alignment failed or reached maximum shift; inspect ROI and max_shift")

    before, after = score((0.0, 0.0)), score(result.x)
    if after < before:
        return np.zeros(2), np.array([before, before])
    return result.x, np.array([before, after])


def prepare_cache(
    records,
    output,
    *,
    dark_paths,
    supportmask,
    mask_pixel,
    crop=0,
    binning=1,
    saturation=65535,
    align_holograms=False,
    alignment_reference=0,
    alignment_max_shift=10.0,
    alignment_upsample_factor=20,
    alignment_roi=None,
):
    """Prepare a hologram cache from a generic detector stream.

    The mask arrays are explicit inputs rather than geometry-derived objects; that
    keeps the routine reusable across beamlines and acquisition modes.
    """
    shape = tuple(records[0]["shape"])
    if any(tuple(r["shape"]) != shape for r in records):
        raise ValueError("Inconsistent detector shapes")
    if dark_paths and not np.allclose([r["exposure_s"] for r in records], records[0]["exposure_s"], rtol=0.01):
        raise ValueError("Use separately matched darks for different exposure groups")

    dark, dark_info = load_dark(dark_paths, shape, records[0]["exposure_s"])
    support = np.asarray(supportmask, dtype=bool)
    if support.shape != shape:
        raise ValueError("Support mask must match the detector grid")
    mask = np.asarray(mask_pixel, dtype=bool)
    if mask.shape != shape:
        raise ValueError("Detector mask must match the detector grid")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.h5")

    config = dict(
        crop=crop,
        binning=binning,
        saturation=saturation,
        dark=dark_info,
        normalization="mean frame / exposure seconds; bins summed",
        supportmask_shape=list(support.shape),
        mask_pixel_shape=list(mask.shape),
        alignment=dict(
            enabled=align_holograms,
            reference_index=alignment_reference,
            max_shift_raw_pixels=alignment_max_shift,
            upsample_factor=alignment_upsample_factor,
            roi=alignment_roi,
            shift_convention="corrective row, column; raw detector pixels",
            method="masked FFT cross-correlation with subpixel normalized-correlation refinement",
        ),
    )

    with h5py.File(partial, "w") as h:
        h.attrs["config_json"] = json.dumps(config)
        h.attrs["manifest_json"] = json.dumps(records)
        h.create_dataset("point", data=[r["point"] for r in records])
        h.create_dataset("scan", data=[r["scan"] for r in records])
        h.create_dataset("current_A", data=[r["current_A"] for r in records])
        h.create_dataset("energy_ev", data=[r["energy_ev"] for r in records])
        h.create_dataset("frame_count", data=[r["frames"] for r in records])
        h.create_dataset("exposure_s", data=[r["exposure_s"] for r in records])
        h.create_dataset("state_labels", data=np.asarray([r["label"] for r in records], dtype=h5py.string_dtype()))
        h.create_dataset("raw_detector_mask", data=mask, compression="gzip")
        h.create_dataset("source_supportmask", data=support, compression="gzip")
        if dark is not None:
            h.create_dataset("dark_mean", data=dark, compression="gzip")

        shifts = h.create_dataset("alignment_shifts_raw_pixels", data=np.zeros((len(records), 2)))
        scores = h.create_dataset("alignment_correlation", data=np.full((len(records), 2), np.nan))
        scores.attrs["columns"] = "before, after; Pearson correlation on fixed valid ROI"

        for i, record in enumerate(records):
            raw, invalid, _ = frame_mean(record["path"], saturation)
            corrected = raw if dark is None else raw - dark
            delta = np.zeros(2)
            if align_holograms:
                if i == alignment_reference:
                    scores[i] = [1.0, 1.0]
                else:
                    ref_raw, ref_invalid, _ = frame_mean(records[alignment_reference]["path"], saturation)
                    reference = (ref_raw if dark is None else ref_raw - dark) / records[alignment_reference]["exposure_s"]
                    reference_mask = mask | ref_invalid
                    try:
                        delta, quality = estimate_hologram_shift(
                            reference,
                            corrected / record["exposure_s"],
                            reference_mask,
                            mask | invalid,
                            max_shift=alignment_max_shift,
                            upsample_factor=alignment_upsample_factor,
                            roi=alignment_roi,
                        )
                    except ValueError as exc:
                        raise ValueError(f"Alignment failed for {record['label']}: {exc}") from exc
                    scores[i] = quality
            shifts[i] = delta

            # Keep the API generic: the caller decides how to crop, shift, and bin the data.
            total_shift = np.asarray((0.0, 0.0)) + delta
            centered = shift(corrected / record["exposure_s"], total_shift, order=1, mode="constant", cval=0, prefilter=False)
            if i == 0:
                h.create_dataset("holograms", shape=(len(records),) + centered.shape, dtype="f4", chunks=(1, *centered.shape), compression="gzip")
                h.create_dataset("mask_pixel", shape=(len(records),) + centered.shape, dtype="bool", chunks=(1, *centered.shape), compression="gzip")
                h.create_dataset("supportmask", data=support, compression="gzip")
                h.create_dataset("first_raw_mean", data=raw, compression="gzip")
            h["holograms"][i] = centered
            h["mask_pixel"][i] = mask
            print(f"Prepared {i + 1}/{len(records)}: {record['label']}", flush=True)

    partial.replace(output)
    return config
