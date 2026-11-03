"""Low-level helpers for binary masks encoded as bright-red pixels in PNG files.

This is the supported source-of-truth format for manual mask editing: the user
sees a viridis reference image and paints a bright-red mask on top, then the
binary mask is recovered by thresholding the red channel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RED_MIN = 200
OTHER_MAX = 80

__all__ = [
    "save_mask_reference_png",
    "mask_png_paths",
    "load_red_mask_png",
    "save_red_mask_png",
    "save_binary_mask_png",
    "write_mask_pngs",
]


def _normalize_reference(values: np.ndarray, log_scale: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Reference image must be 2-D, got shape {values.shape}")
    if log_scale:
        values = np.log1p(np.maximum(values - np.nanmin(values), 0.0))
    values = np.nan_to_num(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo, hi = np.percentile(finite, (1, 99)) if finite.size else (0.0, 1.0)
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def save_mask_reference_png(image, path, log_scale: bool = False, colormap: str = "viridis") -> Path:
    """Save a 2-D image as a viridis-colormap RGB PNG for manual mask painting."""
    values = _normalize_reference(np.asarray(image, dtype=float), log_scale=log_scale)
    cmap = plt.get_cmap(colormap)
    rgb = (cmap(values)[..., :3] * 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


def mask_png_paths(basefolder, kind: str, image_id=None):
    """Return the reference and painted PNG paths for a binary mask kind."""
    allowed = {"mask_pixel", "supportmask"}
    if kind not in allowed:
        raise ValueError(f"Unknown mask kind {kind!r}; use one of {sorted(allowed)}")
    folder = Path(basefolder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = kind if image_id is None else f"{kind}_{image_id}"
    reference = folder / f"{stem}_reference.png"
    painted = folder / f"{stem}.png"
    legacy = folder / f"{stem}_painted.png"
    if legacy.exists() and not painted.exists():
        painted = legacy
    return reference, painted


def load_red_mask_png(path: str | Path, expected_shape=None) -> np.ndarray:
    """Load a binary mask encoded as bright red pixels in an RGB PNG."""
    path = Path(path)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    mask = (
        (rgb[..., 0] >= RED_MIN)
        & (rgb[..., 1] <= OTHER_MAX)
        & (rgb[..., 2] <= OTHER_MAX)
    ).astype(np.uint8)
    if expected_shape is not None and mask.shape[:2] != tuple(expected_shape):
        raise ValueError(
            f"Mask {path} has shape {mask.shape[:2]}, expected {tuple(expected_shape)}. "
            "Do not resize the reference PNG while editing it."
        )
    return mask


def save_red_mask_png(path: str | Path, mask: np.ndarray, reference: str | Path | None = None) -> Path:
    """Save a binary mask in bright red, optionally overlayed on a viridis reference image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.asarray(mask, dtype=bool)
    if reference is not None:
        reference = Path(reference)
        if reference.is_file():
            base = np.asarray(Image.open(reference).convert("RGB"))
        else:
            base = np.zeros((*binary.shape, 3), dtype=np.uint8)
    else:
        base = np.zeros((*binary.shape, 3), dtype=np.uint8)
    rgb = base.copy()
    rgb[binary] = (255, 0, 0)
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


def save_binary_mask_png(basefolder, kind, image_id, mask, reference=None):
    reference_png, painted_png = mask_png_paths(basefolder, kind, image_id)
    if reference is not None:
        save_mask_reference_png(reference, reference_png, log_scale=True, colormap="viridis")
    return save_red_mask_png(painted_png, mask, reference=reference_png if reference is not None else None)


def write_mask_pngs(basefolder, kind, image, mask, image_id=None):
    reference_png, painted_png = mask_png_paths(basefolder, kind, image_id)
    save_mask_reference_png(image, reference_png, log_scale=True, colormap="viridis")
    save_red_mask_png(painted_png, mask, reference=reference_png)
    return reference_png, painted_png
