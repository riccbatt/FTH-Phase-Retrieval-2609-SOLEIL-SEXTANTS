"""PNG export/import helpers for masks edited in Microsoft Paint."""

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .mask_store import load_red_mask_png, save_red_mask_png
except ImportError:  # Notebooks add library/ directly to sys.path.
    from mask_store import load_red_mask_png, save_red_mask_png


def mask_png_paths(basefolder, kind: str, image_id: int | str):
    """Return ``(reference, painted_mask)`` paths for a supported mask kind."""
    folders = {"mask_pixel": "mask_pixels", "supportmask": "supportmask"}
    if kind not in folders:
        raise ValueError(f"Unknown mask kind {kind!r}; use one of {tuple(folders)}")
    folder = Path(basefolder) / "processed" / folders[kind]
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{kind}_{image_id}"
    return folder / f"{stem}_reference.png", folder / f"{stem}.png"


def save_mask_reference_png(image, path, log_scale=False):
    """Save an 8-bit grayscale reference without changing its dimensions."""
    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Reference image must be 2-D, got shape {values.shape}")
    if log_scale:
        values = np.log1p(np.maximum(values - np.nanmin(values), 0))
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, (1, 99)) if finite.size else (0.0, 1.0)
    scaled = np.clip((values - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    output = (np.nan_to_num(scaled) * 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="L").save(path)
    return path


def load_bright_red_mask_png(path, expected_shape=None):
    return load_red_mask_png(path, expected_shape)


def save_binary_mask_png(basefolder, kind, image_id, mask):
    _, path = mask_png_paths(basefolder, kind, image_id)
    return save_red_mask_png(path, mask)
