"""Streaming PETRA Marana inputs for the aperiodic phase retrieval notebooks."""

from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import shift, zoom
from scipy.ndimage import binary_dilation


DETECTOR_DATASET = "/entry/instrument/detector/data"
CURRENT_DATASET = "/scan/instrument/collection/m_caena"
ENERGY_DATASET = "/scan/instrument/collection/mono"


def crop_array_edges(array, pixels):
    """Crop the same number of pixels from every spatial edge."""
    result = np.asarray(array)
    if isinstance(pixels, bool) or not isinstance(pixels, (int, np.integer)):
        raise TypeError("pixels must be a non-negative integer")
    pixels = int(pixels)
    if pixels < 0 or 2 * pixels >= min(result.shape[-2:]):
        raise ValueError("pixels must leave a nonempty spatial array")
    if pixels == 0:
        return result.copy()
    return result[..., pixels:-pixels, pixels:-pixels].copy()


def center_crop_or_pad(array, target_shape, fill_value=0):
    """Center-crop or zero-pad the last two axes without changing pixel scale."""
    source = np.asarray(array)
    target_shape = tuple(int(value) for value in target_shape)
    if len(target_shape) != 2 or min(target_shape) < 1:
        raise ValueError("target_shape must contain two positive integers")
    result = np.full((*source.shape[:-2], *target_shape), fill_value,
                     dtype=source.dtype)
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(source.shape[-2:], target_shape):
        copied = min(source_size, target_size)
        source_start = (source_size - copied) // 2
        target_start = (target_size - copied) // 2
        source_slices.append(slice(source_start, source_start + copied))
        target_slices.append(slice(target_start, target_start + copied))
    result[(..., *target_slices)] = source[(..., *source_slices)]
    return result


def rescale_detector_mask(mask, target_shape):
    """Resize a 2-D or stacked detector mask with nearest-neighbour sampling."""
    source = np.asarray(mask)
    target_shape = tuple(int(value) for value in target_shape)
    if source.ndim not in (2, 3):
        raise ValueError("mask must be 2-D or an observation stack")
    if source.shape[-2:] == target_shape:
        return (source != 0).astype(np.uint8)
    factors = (*([1.0] * (source.ndim - 2)),
               target_shape[0] / source.shape[-2],
               target_shape[1] / source.shape[-1])
    resized = zoom((source != 0).astype(np.uint8), factors, order=0,
                   mode="nearest", prefilter=False)
    if resized.shape[-2:] != target_shape:
        resized = center_crop_or_pad(resized, target_shape, fill_value=1)
    return (resized != 0).astype(np.uint8)


def adapt_retrieval_array_shapes(holograms, mask_pixel, supportmask):
    """Match detector masks and object supports to a hologram spatial shape.

    Detector masks are resized because a cropped object produces a Fourier
    grid with changed detector sampling. Object supports retain their pixel
    scale and are center-cropped or padded.
    """
    images = np.asarray(holograms)
    if images.ndim != 3:
        raise ValueError("holograms must have shape (observations, rows, columns)")
    target_shape = images.shape[-2:]
    mask = np.asarray(mask_pixel)
    if mask.ndim == 3 and mask.shape[0] != images.shape[0]:
        raise ValueError("stacked mask_pixel must match the observation count")
    if mask.ndim not in (2, 3):
        raise ValueError("mask_pixel must be 2-D or an observation stack")
    support = np.asarray(supportmask)
    if support.ndim not in (2, 3):
        raise ValueError("supportmask must be 2-D or a modal stack")
    changes = []
    if mask.shape[-2:] != target_shape:
        changes.append(f"mask_pixel {mask.shape[-2:]} -> {target_shape} (nearest resize)")
        mask = rescale_detector_mask(mask, target_shape)
    else:
        mask = (mask != 0).astype(np.uint8)
    if support.shape[-2:] != target_shape:
        changes.append(f"supportmask {support.shape[-2:]} -> {target_shape} (center crop/pad)")
        support = center_crop_or_pad(support, target_shape, fill_value=0)
    support = (support != 0).astype(np.uint8)
    return images, mask, support, changes


def scan_files(raw_root, prefix, scan_id):
    root = Path(raw_root)
    stem = f"{prefix}_{int(scan_id):05d}"
    master = root / f"{stem}.nxs"
    detector = sorted((root / stem / "marana").glob("*.nx"))
    if not master.is_file() or not detector:
        raise FileNotFoundError(
            f"Scan {scan_id} needs {master} and detector files in "
            f"{root / stem / 'marana'}. Add the raw data or correct RAW_ROOT/PREFIX."
        )
    return master, detector


def scan_metadata(master):
    with h5py.File(master) as handle:
        energy = np.asarray(handle[ENERGY_DATASET][()]).ravel().astype(float)
        current = (np.asarray(handle[CURRENT_DATASET][()]).ravel().astype(float)
                   if CURRENT_DATASET in handle else None)
    if not len(energy) or not np.all(np.isfinite(energy)):
        raise ValueError(f"Invalid photon energy in {master}")
    return energy, current


def detector_file_point_indices(files):
    """Read acquisition point numbers from Marana file suffixes."""
    indices = []
    for path in files:
        suffix = Path(path).stem.rsplit("_", 1)[-1]
        if not suffix.isdecimal():
            raise ValueError(f"Marana filename has no point index: {path}")
        indices.append(int(suffix))
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate Marana point indices")
    return np.asarray(indices, dtype=int)


def currents_for_detector_files(current, files):
    """Associate each downloaded Marana file with its scan-current reading."""
    if current is None:
        raise ValueError("Scan has no magnet-current readings")
    values = np.asarray(current, dtype=float).ravel()
    indices = detector_file_point_indices(files)
    if len(indices) == 0 or np.any(indices >= len(values)):
        raise ValueError(
            f"Marana point indices {indices.tolist()} do not fit "
            f"{len(values)} current readings"
        )
    return indices, values[indices]


def fit_reference_intensity(image, reference, mask):
    """Fit image = factor * reference + offset on valid detector pixels.

    Copy2's ``worker`` uses this correlation fit, then divides the image by
    ``factor`` before phase retrieval. Its fitted offset is diagnostic and is
    not subtracted from the phase-retrieval hologram.
    """
    image = np.asarray(image)
    reference = np.asarray(reference)
    mask = np.asarray(mask)
    if image.shape != reference.shape or mask.shape != image.shape:
        raise ValueError("image, reference, and mask must have the same shape")
    valid = ((mask == 0) & np.isfinite(image) & np.isfinite(reference)
             & (image != 0) & (reference != 0))
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two unmasked nonzero pixels are needed")
    x = reference[valid].astype(np.float64)
    y = image[valid].astype(np.float64)
    xmean = np.mean(x)
    ymean = np.mean(y)
    denominator = np.sum((x - xmean) ** 2)
    if denominator <= 0:
        raise ValueError("Reference intensity is constant on valid pixels")
    factor = float(np.sum((x - xmean) * (y - ymean)) / denominator)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"Invalid reference intensity factor: {factor}")
    offset = float(ymean - factor * xmean)
    return factor, offset


def point_mean(path, frame_indices=None, chunk_size=16, dark=None,
               photon_level=None):
    """Average frames; optionally dark-correct and threshold each frame first.

    ``photon_level``: corrected frame pixels at or below this value
    become zero before the frame average. With ``dark`` supplied, the returned
    image is already dark-subtracted.
    """
    with h5py.File(path) as handle:
        data = handle[DETECTOR_DATASET]
        if frame_indices is not None:
            selected = np.asarray(frame_indices)
            count = 1 if data.ndim == 2 else data.shape[0]
            if (selected.ndim != 1 or not selected.size
                    or not np.issubdtype(selected.dtype, np.integer)
                    or np.any(selected < 0) or np.any(selected >= count)
                    or len(np.unique(selected)) != selected.size):
                raise ValueError(f"Invalid frame selection for {path}")
            frame_indices = np.sort(selected)
        if data.ndim == 2:
            frames = np.asarray(data, dtype=np.float64)[None]
            if dark is not None:
                frames -= dark
            if photon_level is not None:
                frames[frames <= photon_level] = 0
            return frames[0].astype(np.float32)
        if data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D detector data in {path}: {data.shape}")
        indices = (np.arange(data.shape[0]) if frame_indices is None
                   else np.asarray(frame_indices, dtype=int))
        if not len(indices) or np.any(indices < 0) or np.any(indices >= data.shape[0]):
            raise ValueError(f"Invalid frame selection for {path}")
        total = np.zeros(data.shape[-2:], dtype=np.float64)
        for offset in range(0, len(indices), chunk_size):
            frames = np.asarray(data[indices[offset:offset + chunk_size]],
                                dtype=np.float64)
            if dark is not None:
                frames -= dark
            if photon_level is not None:
                frames[frames <= photon_level] = 0
            total += frames.sum(axis=0)
        return (total / len(indices)).astype(np.float32)


def centered(image, detector_center):
    delta = np.asarray(image.shape) // 2 - np.asarray(detector_center)
    return shift(image, delta, order=1, mode="constant", cval=0, prefilter=False)


def load_support(path, radius,shape):
    """Scale the working notebook's painted support onto the raw object grid."""
    # Copy2 selects pixels whose red channel is exactly one in plt.imread.
    # The PNG is a colored overlay, so a grayscale threshold includes the
    # scattering image and would make the support much too broad.
    rgb = np.asarray(Image.open(path).convert("RGB"))
    support = rgb[:, :, 0] == 255
    yy, xx = np.ogrid[:support.shape[0], :support.shape[1]]
    cy, cx = np.asarray(support.shape) // 2
    support |= (yy - cy)**2 + (xx - cx)**2 <= radius**2
    if support.shape != tuple(shape):
        # Copy2 mask_lib.scale_mask uses scipy zoom with its cubic default,
        # followed by a 0.5 threshold.
        support = zoom(support.astype(float),
                       np.asarray(shape) / np.asarray(support.shape),
                       order=3) > 0.5
        if support.shape != tuple(shape):
            raise ValueError(f"Scaled support shape {support.shape} != {shape}")
    return support.astype(np.uint8)


def load_detector_mask(path, shape):
    if path is None:
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(Image.open(path).convert("L")) != 0
    if mask.shape != tuple(shape):
        raise ValueError(f"Detector mask {path} has shape {mask.shape}, expected {shape}")
    return mask


def load_reference_polygon_mask(path, shape, dilation=2,
                                shift_pixels=(0, 0), erosion=0,
                                coordinate_order="xy"):
    """Rasterize the centered detector exclusions from the Copy2 notebook."""
    import json
    from skimage.draw import polygon

    with open(path, encoding="utf-8") as handle:
        groups = json.load(handle)
    mask = np.zeros(shape, dtype=bool)
    for polygons in groups.values():
        for corners in polygons:
            coords = np.asarray(corners, dtype=float)
            # Copy2's mask_lib passes vertices to matplotlib.path.Path as (x, y).
            if coordinate_order == "xy":
                rr, cc = polygon(coords[:, 1], coords[:, 0], shape=shape)
            elif coordinate_order == "yx":
                rr, cc = polygon(coords[:, 0], coords[:, 1], shape=shape)
            else:
                raise ValueError("coordinate_order must be 'xy' or 'yx'")
            mask[rr, cc] = True
    if any(float(value) != 0 for value in shift_pixels):
        mask = shift(mask.astype(float), shift_pixels, order=1,
                     mode="constant", cval=0, prefilter=False) >= 0.5
    if erosion:
        from skimage.morphology import disk, erosion as binary_erosion_disk
        mask = binary_erosion_disk(mask, disk(int(erosion)))
    return binary_dilation(mask, iterations=dilation) if dilation else mask


def _polygon_mask_on_cached_grid(path, raw_shape, target_shape, crop, binning,
                                 shift_pixels, erosion, dilation,
                                 coordinate_order):
    """Rasterize one polygon mask, then crop and bin it like the detector."""
    polygon = load_reference_polygon_mask(
        path, raw_shape, dilation=dilation, shift_pixels=shift_pixels,
        erosion=erosion, coordinate_order=coordinate_order,
    )
    if crop:
        polygon = polygon[crop:-crop, crop:-crop]
    if binning > 1:
        ny, nx = target_shape
        trim_y = (polygon.shape[0] - ny * binning) // 2
        trim_x = (polygon.shape[1] - nx * binning) // 2
        polygon = polygon[trim_y:trim_y + ny * binning,
                          trim_x:trim_x + nx * binning]
        polygon = polygon.reshape(ny, binning, nx, binning).any(axis=(1, 3))
    return polygon


def correct_cached_polygon_mask(mask, config, shift_pixels=(2.5, -2),
                                erosion=2, dilation=2):
    """Replace only the painted polygon part of an existing cached pixel mask."""
    path = config.get("reference_polygons")
    if not path:
        return np.asarray(mask)
    old_shift = tuple(config.get("reference_mask_shift_pixels", (0, 0)))
    old_erosion = int(config.get("reference_mask_erosion_pixels", 0))
    old_dilation = int(config.get("reference_mask_dilation_pixels", 2))
    old_order = config.get("reference_polygon_coordinate_order", "yx")
    if (old_shift == tuple(shift_pixels) and old_erosion == erosion
            and old_dilation == dilation and old_order == "xy"):
        return np.asarray(mask)
    result = np.asarray(mask)
    crop = int(config["crop_raw_pixels"])
    binning = int(config["binning"])
    raw_shape = tuple(int(n * binning + 2 * crop) for n in result.shape[-2:])

    old = _polygon_mask_on_cached_grid(
        path, raw_shape, result.shape[-2:], crop, binning,
        old_shift, old_erosion, old_dilation, old_order,
    )
    new = _polygon_mask_on_cached_grid(
        path, raw_shape, result.shape[-2:], crop, binning,
        shift_pixels, erosion, dilation, "xy",
    )
    return ((result != 0) & ~old | new).astype(np.uint8)


def support_field_coverage(support, crop, binning):
    """Fraction of the painted support retained by crop followed by binning.

    Detector binning narrows the object field of view. This catches settings
    that silently remove reference apertures from a painted object support.
    """
    from library.phase_retrieval_geometry import _resample_spatial

    shape = np.asarray(support.shape, dtype=int)
    crop = int(crop)
    binning = int(binning)
    cropped = shape - 2 * crop
    if np.any(cropped <= 0) or binning < 1:
        raise ValueError("Invalid crop or binning for support coverage")
    if crop:
        full = _resample_spatial(
            support, tuple(cropped), order=0,
            coordinate_scale=shape / cropped,
        ) != 0
    else:
        full = np.asarray(support) != 0
    final_shape = cropped // binning
    origin = (cropped - final_shape) // 2
    retained = full[tuple(slice(o, o + n) for o, n in zip(origin, final_shape))]
    return float(np.count_nonzero(retained) / max(np.count_nonzero(full), 1))


def prepare_point(image, dark, detector_center, support, crop, binning,
                  beamstop_radius, saturation, extra_mask=None,
                  centered_mask=None):
    """Center, dark correct, crop, then bin one intensity and exclusion mask."""
    from library.phase_retrieval_geometry import prepare

    raw = np.asarray(image, dtype=np.float32)
    # Preprocessing can prepare detector data without an object-support input.
    if support is None:
        support = np.ones(raw.shape, dtype=np.uint8)
    if raw.shape != support.shape:
        raise ValueError(f"Image {raw.shape} and support {support.shape} differ")
    corrected = centered(raw - dark, detector_center)
    excluded = centered((raw >= saturation).astype(np.float32), detector_center) > 0
    if extra_mask is not None:
        excluded |= centered(np.asarray(extra_mask, dtype=np.float32), detector_center) > 0
    if centered_mask is not None:
        if np.shape(centered_mask) != raw.shape:
            raise ValueError("centered_mask must match the detector shape")
        excluded |= np.asarray(centered_mask) != 0
    yy, xx = np.ogrid[:raw.shape[0], :raw.shape[1]]
    cy, cx = np.asarray(raw.shape) // 2
    excluded |= (yy - cy)**2 + (xx - cx)**2 <= beamstop_radius**2
    excluded |= ~np.isfinite(corrected)
    images, mask, used_support, _, geometry = prepare(
        np.nan_to_num(corrected)[None], excluded.astype(np.uint8), support,
        {"crop": int(crop), "binning": int(binning), "roi": None},
    )
    return images[0].astype(np.float32), mask, used_support, geometry
