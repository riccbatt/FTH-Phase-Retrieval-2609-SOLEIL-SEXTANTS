"""Shared helpers for the FTH and phase-retrieval notebooks.

This module deliberately depends only on NumPy and h5py at import time.  The
experiment-specific reconstruction modules are passed into the functions that
need them, which keeps the notebook workflow usable on machines without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, zoom


def normalize_image(image: Any) -> np.ndarray:
    """Scale finite values to [0, 1], preserving the input shape."""
    array = np.asarray(image)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=float)
    low = np.min(array[finite])
    span = np.max(array[finite]) - low
    if span == 0:
        return np.zeros(array.shape, dtype=float)
    result = (array - low) / span
    return np.where(finite, result, 0)


def center_image(image: Any, center: Sequence[float], cci: Any) -> np.ndarray:
    """Move ``center=(row, column)`` to the array centre."""
    array = np.asarray(image)
    if array.ndim != 2 or len(center) != 2:
        raise ValueError("center_image expects a 2-D image and a two-value center")
    shift = np.asarray(array.shape, dtype=float) / 2 - np.asarray(center, dtype=float)
    return np.asarray(cci.shift_image(array, shift))


def define_centered_holograms(
    data: dict[str, Any],
    cci: Any,
    *,
    PhR: Any | None = None,
    project_ewalds_sphere: bool = False,
    ewald_method: str = "cubic",
) -> dict[str, Any]:
    """Center every ``data['holo'][label]['image']`` and optionally project it."""
    center = data["center"]
    setup = data["experimental_setup"]
    for state in data["holo"].values():
        image = center_image(state["image"], center, cci)
        if project_ewalds_sphere:
            if PhR is None or not hasattr(PhR, "inv_gnomonic"):
                raise RuntimeError("Ewald projection requested but inv_gnomonic is unavailable")
            image = PhR.inv_gnomonic(
                image,
                center=np.asarray(image.shape) / 2,
                experimental_setup=setup,
                method=ewald_method,
            )
        state["image_c"] = np.asarray(image)
    data["project_ewalds_sphere"] = bool(project_ewalds_sphere)
    data["ewald_method"] = ewald_method
    return data


def butterworth_disk_mask(shape: Sequence[int], radius: float, order: float) -> np.ndarray:
    """Return a smooth mask that is one at the centre and decays outwards."""
    rows, columns = (int(value) for value in shape)
    if rows <= 0 or columns <= 0 or radius <= 0 or order <= 0:
        raise ValueError("shape, radius, and order must be positive")
    yy, xx = np.ogrid[:rows, :columns]
    distance = np.hypot(yy - rows / 2, xx - columns / 2)
    return 1.0 / (1.0 + (distance / float(radius)) ** (2 * float(order)))


def smooth_binary_mask(
    mask: Any, dilation_pixels: int = 3, sigma: float | None = None
) -> np.ndarray:
    """Dilate a binary bad-pixel mask, then soften its edge for FTH only."""
    binary = np.asarray(mask) != 0
    if binary.ndim != 2:
        raise ValueError("smooth_binary_mask expects a 2-D mask")
    if isinstance(dilation_pixels, bool) or not isinstance(
        dilation_pixels, (int, np.integer)
    ):
        raise ValueError("dilation_pixels must be a non-negative integer")
    dilation_pixels = int(dilation_pixels)
    if dilation_pixels < 0:
        raise ValueError("dilation_pixels must be a non-negative integer")
    if sigma is None:
        sigma = float(dilation_pixels)
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    dilated = binary_dilation(binary, iterations=dilation_pixels) if dilation_pixels else binary
    if not sigma:
        return dilated.astype(float)
    return np.clip(gaussian_filter(dilated.astype(float), sigma=sigma), 0.0, 1.0)


def fth_reconstruct(
    hologram: Any,
    experimental_setup: Mapping[str, Any],
    fth: Any,
    *,
    prop_dist: float = 0,
    phase: float = 0,
    dx: float = 0,
    dy: float = 0,
) -> np.ndarray:
    """Reconstruct using propagation in micrometres, matching ``focusCDI``."""
    propagated = fth.propagate(
        np.asarray(hologram), float(prop_dist) * 1e-6, experimental_setup
    )
    if hasattr(fth, "global_phase_shift"):
        propagated = fth.global_phase_shift(propagated, phase)
    else:
        propagated = propagated * np.exp(1j * phase)
    # Match reconstruct_rb.focusCDI exactly: that widget uses the forward FFT
    # reconstruction, not fth.reconstruct's inverse FFT.
    reconstruct = getattr(fth, "reconstructCDI", fth.reconstruct)
    reconstruction = np.asarray(reconstruct(propagated))
    if dx or dy:
        reconstruction = np.asarray(fth.sub_pixel_centering(reconstruction, dx, dy))
    return reconstruction


def as_modes(array: Any) -> np.ndarray:
    """Represent one 2-D field or a stack of fields as ``(mode, y, x)``."""
    result = np.asarray(array)
    if result.ndim == 2:
        return result[np.newaxis, ...]
    if result.ndim == 3:
        return result
    raise ValueError(f"Expected a 2-D field or 3-D mode stack, got {result.shape}")


def spatial_shape(array: Any) -> tuple[int, int]:
    result = np.asarray(array)
    if result.ndim < 2:
        raise ValueError("An array must have at least two spatial dimensions")
    return tuple(int(value) for value in result.shape[-2:])


def reconstruct_cdi_modes(
    holograms: Any,
    mask: Any,
    fth: Any,
    experimental_setup: Mapping[str, Any],
    *,
    prop_dist: float = 0,
    phase: float = 0,
    dx: float = 0,
    dy: float = 0,
) -> np.ndarray:
    modes = as_modes(holograms)
    mask_array = np.asarray(mask)
    if mask_array.shape != modes.shape[-2:]:
        raise ValueError(f"Mask shape {mask_array.shape} != field shape {modes.shape[-2:]}")
    output = []
    for mode in modes:
        propagated = fth.propagate(
            mode * mask_array, float(prop_dist) * 1e-6, experimental_setup
        )
        if hasattr(fth, "global_phase_shift"):
            propagated = fth.global_phase_shift(propagated, phase)
        else:
            propagated = propagated * np.exp(1j * phase)
        reconstruct = getattr(fth, "reconstructCDI", fth.reconstruct)
        reconstruction = np.asarray(reconstruct(propagated))
        if dx or dy:
            reconstruction = np.asarray(fth.sub_pixel_centering(reconstruction, dx, dy))
        output.append(reconstruction)
    result = np.stack(output)
    return result[0] if np.asarray(holograms).ndim == 2 else result


def slices_to_roi(region: tuple[slice, slice]) -> list[int]:
    if len(region) != 2 or not all(isinstance(value, slice) for value in region):
        raise TypeError("ROI must be a pair of slices")
    if any(value.step not in (None, 1) for value in region):
        raise ValueError("Strided ROIs are not supported")
    return [region[0].start or 0, region[0].stop, region[1].start or 0, region[1].stop]


def roi_to_slices(roi: Sequence[int]) -> tuple[slice, slice]:
    if len(roi) != 4:
        raise ValueError("ROI must contain [row_start, row_stop, col_start, col_stop]")
    r0, r1, c0, c1 = (int(value) for value in roi)
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"Invalid ROI: {list(roi)}")
    return slice(r0, r1), slice(c0, c1)


def spatial_roi(array: Any, region: tuple[slice, slice]) -> np.ndarray:
    return np.asarray(array)[(...,) + tuple(region)]


def apply_spatial_mask(array: Any, mask: Any) -> np.ndarray:
    result = np.asarray(array)
    mask_array = np.asarray(mask)
    if result.shape[-2:] != mask_array.shape:
        raise ValueError(f"Mask shape {mask_array.shape} != spatial shape {result.shape[-2:]}")
    return result * mask_array


def resize_binary_to_shape(mask: Any, shape: Sequence[int]) -> np.ndarray:
    source = np.asarray(mask, dtype=float)
    target = tuple(int(value) for value in shape)
    if source.ndim != 2 or len(target) != 2:
        raise ValueError("resize_binary_to_shape expects two-dimensional shapes")
    if source.shape == target:
        return (source > 0.5).astype(np.uint8)
    resized = zoom(source, (target[0] / source.shape[0], target[1] / source.shape[1]), order=0)
    # scipy rounding can occasionally produce a one-pixel discrepancy.
    fitted = np.zeros(target, dtype=np.uint8)
    rows, columns = min(target[0], resized.shape[0]), min(target[1], resized.shape[1])
    fitted[:rows, :columns] = resized[:rows, :columns] > 0.5
    return fitted


def finite_percentile_limits(array: Any, percentiles: Sequence[float] = (1, 99)) -> tuple[float, float]:
    values = np.asarray(array)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("Cannot calculate limits for an array without finite values")
    low, high = np.percentile(finite, percentiles)
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def get_hologram_labels(data: Mapping[str, Any]) -> list[str]:
    labels = data.get("hologram_labels")
    return list(labels) if labels is not None else list(data["holo"].keys())


def _write_value(parent: h5py.Group, name: str, value: Any) -> None:
    if isinstance(value, Mapping):
        group = parent.create_group(name)
        group.attrs["_python_type"] = "dict"
        for key, item in value.items():
            _write_value(group, str(key), item)
    elif isinstance(value, (list, tuple)):
        group = parent.create_group(name)
        group.attrs["_python_type"] = "tuple" if isinstance(value, tuple) else "list"
        for index, item in enumerate(value):
            _write_value(group, str(index), item)
    elif value is None:
        group = parent.create_group(name)
        group.attrs["_python_type"] = "none"
    elif isinstance(value, (str, bytes, Path)):
        parent.create_dataset(name, data=str(value), dtype=h5py.string_dtype("utf-8"))
    else:
        parent.create_dataset(name, data=value)


def _read_value(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value.item() if isinstance(value, np.generic) else value
    kind = node.attrs.get("_python_type", "dict")
    if isinstance(kind, bytes):
        kind = kind.decode("utf-8")
    if kind == "none":
        return None
    if kind in ("list", "tuple"):
        values = [_read_value(node[key]) for key in sorted(node, key=lambda key: int(key))]
        return tuple(values) if kind == "tuple" else values
    return {key: _read_value(node[key]) for key in node}


def save_data_dict(data: Mapping[str, Any], filename: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w" if overwrite else "x") as handle:
        for key, value in data.items():
            _write_value(handle, str(key), value)
    return path


def load_data_dict(filename: str | Path) -> dict[str, Any]:
    with h5py.File(filename, "r") as handle:
        return {key: _read_value(handle[key]) for key in handle}


def load_processing(folder: str | Path, image_id: int | str, crop: int | None = None):
    """Load a legacy NumPy detector file by image id.

    Modern SEXTANTS data should use ``SextantsNexusLoader``.  This fallback
    accepts ``.npy`` and single-array ``.npz`` files used by older notebooks.
    """
    folder = Path(folder)
    candidates = sorted(folder.glob(f"*{int(image_id)}*.npy")) + sorted(folder.glob(f"*{int(image_id)}*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No NumPy detector file containing image id {image_id} in {folder}")
    path = candidates[0]
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            key = "image" if "image" in loaded.files else loaded.files[0]
            image = np.asarray(loaded[key])
        finally:
            loaded.close()
    else:
        image = np.asarray(loaded)
    if crop:
        image = image[crop:-crop, crop:-crop]
    return image, str(path)


def load_scan_value(folder: str | Path, image_id: int | str, mnemonic: str) -> float:
    """Read a scalar HDF5/Nexus value for a legacy image id."""
    folder = Path(folder)
    candidates = sorted(folder.glob(f"*{int(image_id)}*.h5")) + sorted(folder.glob(f"*{int(image_id)}*.nxs"))
    if not candidates:
        raise FileNotFoundError(f"No HDF5/Nexus file containing image id {image_id} in {folder}")
    with h5py.File(candidates[0], "r") as handle:
        if mnemonic in handle:
            value = handle[mnemonic][()]
        else:
            entry = next(iter(handle))
            value = handle[f"{entry}/{mnemonic.lstrip('/')}"][()]
    values = np.asarray(value).squeeze()
    if values.size != 1:
        raise ValueError(f"Expected scalar at {mnemonic}, got shape {values.shape}")
    return float(values)


__all__ = [name for name in globals() if not name.startswith("_")]
