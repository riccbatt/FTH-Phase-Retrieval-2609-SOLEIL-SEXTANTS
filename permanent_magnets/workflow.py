"""Small I/O and plotting helpers for the four permanent-magnet notebooks."""
from pathlib import Path
import json

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def scan_paths(raw_root, prefix, index):
    """One scan index selects every Marana detector file in that scan."""
    paths = sorted((Path(raw_root) / f"{prefix}_{int(index):05d}" / "marana").glob("*.nx"))
    if not paths:
        raise FileNotFoundError(f"No detector files for scan {index} in {raw_root}")
    return paths


def red_mask(path, shape):
    """Red pixels are True; require the supplied PNG to match its working grid."""
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=float)
    mask = ((rgba[..., 0] > 100) & (rgba[..., 0] > 1.5*rgba[..., 1])
            & (rgba[..., 0] > 1.5*rgba[..., 2]) & (rgba[..., 3] > 0))
    if mask.shape != tuple(shape):
        raise ValueError(f"{path}: {mask.shape} does not match {tuple(shape)}; do not resize masks silently")
    if not mask.any():
        raise ValueError(f"No red pixels found in {path}")
    return mask


def estimate_hologram_shift(reference, moving, reference_mask, moving_mask, *,
                             max_shift=10., upsample_factor=20, roi=None):
    """Return corrective (row, column) raw-pixel shift and Pearson scores.

    Masked FFT cross-correlation gives an integer seed. Refine normalized
    cross-correlation on a fixed valid interior to subpixel precision. Intensities
    are linearly interpolated; no periodic wrapping or masked values enter the fit.
    """
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import minimum_filter
    from scipy.optimize import minimize
    if not np.isfinite(max_shift) or max_shift <= 0:
        raise ValueError('alignment max_shift must be positive and finite')
    if isinstance(upsample_factor, bool) or not isinstance(upsample_factor, (int, np.integer)) or upsample_factor < 1:
        raise ValueError('alignment upsample_factor must be a positive integer')
    reference, moving = np.asarray(reference, float), np.asarray(moving, float)
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError('Alignment requires matching 2D holograms')
    valid_ref = ~np.asarray(reference_mask, bool) & np.isfinite(reference)
    valid_mov = ~np.asarray(moving_mask, bool) & np.isfinite(moving)
    if roi is not None:
        if len(roi) != 4 or any(not isinstance(v, (int, np.integer)) for v in roi):
            raise ValueError('alignment roi must be (row_start, row_stop, col_start, col_stop)')
        r0, r1, c0, c1 = roi
        if not (0 <= r0 < r1 <= reference.shape[0] and 0 <= c0 < c1 <= reference.shape[1]):
            raise ValueError('alignment roi is outside the detector')
        region = np.s_[r0:r1, c0:c1]
        reference, moving, valid_ref, valid_mov = [a[region] for a in (reference, moving, valid_ref, valid_mov)]
    margin = int(np.ceil(max_shift)) + 1
    safe = valid_ref & minimum_filter(valid_mov.astype(np.uint8), size=2*margin+1, mode='constant', cval=0).astype(bool)
    rows, cols = np.nonzero(safe)
    if len(rows) < 64:
        raise ValueError('Too few valid pixels for alignment; enlarge ROI or reduce max_shift')
    target = reference[safe]; target = target-target.mean()
    norm = np.linalg.norm(target)
    if norm <= 1e-12:
        raise ValueError('Reference has no contrast for alignment')
    from scipy.ndimage import map_coordinates
    moving = np.where(valid_mov, moving, 0.)
    def score(delta):
        values = map_coordinates(moving, [rows-delta[0], cols-delta[1]], order=1, prefilter=False)
        values -= values.mean()
        denominator = norm*np.linalg.norm(values)
        return float(np.dot(target, values)/denominator) if denominator > 1e-12 else -1.
    if np.std(moving[valid_mov]) <= 1e-12:
        raise ValueError('Moving hologram has no contrast for alignment')
    seed = phase_cross_correlation(np.where(valid_ref, reference, 0.), moving,
        reference_mask=valid_ref, moving_mask=valid_mov, overlap_ratio=.5)[0]
    if np.any(np.abs(seed) > max_shift):
        raise ValueError(f'Estimated alignment shift {seed} exceeds max_shift={max_shift}')
    result = minimize(lambda delta: -score(delta), seed, method='Powell',
        bounds=[(max(-max_shift, v-1.5), min(max_shift, v+1.5)) for v in seed],
        options={'xtol': 1/upsample_factor, 'ftol': 1e-9})
    if not result.success or np.any(np.abs(result.x) >= max_shift-1/upsample_factor):
        raise ValueError('Alignment failed or reached maximum shift; inspect ROI and max_shift')
    before, after = score((0., 0.)), score(result.x)
    if after < before:
        return np.zeros(2), np.array([before, before])
    return result.x, np.array([before, after])


def plot_stack(images, labels, title, cmap="viridis", vmin=None, vmax=None):
    images = np.asarray(images)
    columns = min(4, len(images))
    rows = (len(images)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(4*columns, 3.5*rows), squeeze=False)
    for ax, image, label in zip(axes.flat, images, labels):
        artist = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(str(label)); ax.set_axis_off()
        fig.colorbar(artist, ax=ax, shrink=.75)
    for ax in list(axes.flat)[len(images):]:
        ax.set_visible(False)
    fig.suptitle(title); fig.tight_layout(); plt.show()
    return fig


def field_diagnostics(fields, holograms, masks, labels):
    """Plot every reconstructed mode and report measured-pixel intensity errors."""
    from library.phase_retrieval_universal import fourier_field_to_display_object
    modes = fields[:, None] if fields.ndim == 3 else fields
    objects = np.stack([fourier_field_to_display_object(modes[:, i]) for i in range(modes.shape[1])], axis=1)
    for mode in range(modes.shape[1]):
        plot_stack(abs(objects[:, mode]), labels, f"Object amplitude, mode {mode+1}")
        plot_stack(np.angle(objects[:, mode]), labels, f"Object phase, mode {mode+1}", "twilight", -np.pi, np.pi)
    prediction = np.sum(abs(modes)**2, axis=1)
    valid = ~np.asarray(masks, bool) & np.isfinite(holograms)
    numerator = np.sum(np.where(valid, (prediction-holograms)**2, 0), axis=(1, 2))
    denominator = np.sum(np.where(valid, holograms**2, 0), axis=(1, 2))
    residual = np.sqrt(numerator/np.maximum(denominator, 1e-30))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(residual, "o-"); ax.set(xlabel="Observation", ylabel="Relative intensity RMS", title="Measured-pixel agreement")
    plt.show()
    plot_stack(np.where(valid, prediction-holograms, np.nan), labels, "Intensity residual (masked pixels hidden)", "RdBu_r")
    return objects, residual


def save_tree(group, values):
    """Keep library diagnostic/component outputs in readable HDF5 groups."""
    for key, value in values.items():
        key = str(key)
        if isinstance(value, dict):
            save_tree(group.create_group(key), value)
        elif value is None:
            group.attrs[key] = "None"
        else:
            try:
                array = np.asarray(value)
                if array.dtype.kind in "US":
                    group.create_dataset(key, data=array.astype(object), dtype=h5py.string_dtype())
                elif array.dtype.kind != "O":
                    group.create_dataset(key, data=array, compression="gzip" if array.ndim else None)
                else:
                    group.attrs[key] = json.dumps(value, default=lambda x: np.asarray(x).tolist())
            except (TypeError, ValueError):
                group.attrs[key] = json.dumps(value, default=str)
