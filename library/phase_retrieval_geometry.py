"""Shared detector and object-grid geometry for joint phase retrieval."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform, center_of_mass, label, shift as image_shift


GEOMETRY_DEFAULTS = {"binning": 1, "crop": 0, "roi": None}


def recenter_source_support(support, crop, binning, margin=2):
    """Translate an object support so the detector crop/bin grid retains it.

    The returned integer source-grid shift can also be applied to any known
    object-space thickness or material map. Detector masks must stay fixed.
    """
    support = np.asarray(support)
    if support.ndim != 2 or not np.any(support):
        raise ValueError("support must be a nonempty 2D array")
    shape = np.asarray(support.shape, dtype=int)
    crop = _positive_integer(crop, "crop", 0)
    binning = _positive_integer(binning, "binning", 1)
    cropped = shape - 2 * crop
    if np.any(cropped <= 0):
        raise ValueError("crop removes the complete support grid")
    used = cropped // binning
    if np.any(used < 1):
        raise ValueError("binning is larger than the cropped grid")
    if crop:
        source_center = (shape - 1) / 2
        used_center = (used - 1) / 2
        scale = shape / cropped
        visible_low = source_center + (margin - used_center) * scale
        visible_high = source_center + (used - 1 - margin - used_center) * scale
    else:
        origin = (shape - used) // 2
        visible_low = origin + margin
        visible_high = origin + used - 1 - margin
    points = np.argwhere(support != 0)
    lower = np.ceil(visible_low - points.min(axis=0)).astype(int)
    upper = np.floor(visible_high - points.max(axis=0)).astype(int)
    if np.any(lower > upper):
        raise ValueError(
            f"Support span cannot fit retrieval grid {tuple(used)} "
            f"with crop={crop}, binning={binning}."
        )
    translation = np.minimum(np.maximum(np.zeros(2, dtype=int), lower), upper)
    if np.any(translation):
        shifted = image_shift(support.astype(float), translation, order=0,
                              mode="constant", cval=0, prefilter=False)
        if np.count_nonzero(shifted) != np.count_nonzero(support):
            raise ValueError("Recentering would cut support at the source-grid edge")
        support = shifted.astype(support.dtype)
    return support.copy(), tuple(int(v) for v in translation)


def recenter_modal_supports(support, factors, center="image", margin=2):
    """Expand each mode without clipping and move only modes that need it.

    Each modal support gets its own translation. This leaves the factor-1
    physical support at its calibrated position while fitting factor-2 inside
    the available object field of view.
    """
    support = np.asarray(support)
    if support.ndim != 2 or not np.any(support):
        raise ValueError("support must be a nonempty 2D array")
    if center not in {"image", "components"}:
        raise ValueError("center must be 'image' or 'components'")
    shape = np.asarray(support.shape, dtype=int)
    image_center = (shape - 1) / 2
    if center == "components":
        labels, count = label(support != 0, structure=np.ones((3, 3), dtype=int))
        component_centers = center_of_mass(support != 0, labels, range(1, count + 1))
    else:
        labels, count, component_centers = None, 1, [image_center]

    masks, shifts = [], []
    for factor in factors:
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("mode factors must be finite and positive")
        if factor == 1:
            masks.append((support != 0).astype(np.uint8))
            shifts.append((0, 0))
            continue
        bounds = []
        for component in range(1, count + 1):
            points = np.argwhere((labels == component) if labels is not None
                                 else (support != 0))
            center_point = (np.asarray(component_centers[component - 1])
                            if labels is not None else image_center)
            bounds.append((center_point + factor * (points.min(axis=0) - center_point),
                           center_point + factor * (points.max(axis=0) - center_point)))
        minimum = np.min([pair[0] for pair in bounds], axis=0)
        maximum = np.max([pair[1] for pair in bounds], axis=0)
        lower = np.ceil(margin - minimum).astype(int)
        upper = np.floor(shape - 1 - margin - maximum).astype(int)
        if np.any(lower > upper):
            raise ValueError(
                f"Mode factor {factor:g} support cannot fit retrieval grid "
                f"{tuple(shape)}; reduce binning/crop or the mode factor."
            )
        translation = np.minimum(np.maximum(np.zeros(2, dtype=int), lower), upper)
        expanded = np.zeros(tuple(shape), dtype=bool)
        for component in range(1, count + 1):
            center_point = (np.asarray(component_centers[component - 1])
                            if labels is not None else image_center)
            component_mask = ((labels == component) if labels is not None
                              else (support != 0))
            inverse = np.eye(2) / factor
            # Equivalent to center + (output - center - translation)/factor.
            offset = center_point - inverse @ (center_point + translation)
            expanded |= affine_transform(
                component_mask.astype(np.uint8), inverse, offset=offset,
                output_shape=tuple(shape), order=0, mode="constant", cval=0,
                prefilter=False,
            ) != 0
        masks.append(expanded.astype(np.uint8))
        shifts.append(tuple(int(v) for v in translation))
    return np.stack(masks), shifts


def _positive_integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"recipe[{name!r}] must be an integer >= {minimum}.")
    return int(value)


def _resample_spatial(array, target_shape, order, coordinate_scale=None):
    """Map object pixels about image center onto the retrieval grid."""
    source = np.asarray(array)
    old_shape = np.asarray(source.shape[-2:])
    target_shape = tuple(int(n) for n in target_shape)
    if tuple(old_shape) == target_shape:
        return source.copy()
    scale = (old_shape / np.asarray(target_shape) if coordinate_scale is None
             else np.asarray(coordinate_scale, dtype=float))
    offset = (old_shape - 1) / 2 - scale * (np.asarray(target_shape) - 1) / 2
    flat = source.reshape((-1, *old_shape))

    def sample(plane):
        if np.iscomplexobj(plane):
            return sample(plane.real) + 1j * sample(plane.imag)
        return affine_transform(
            plane.astype(float), np.diag(scale), offset=offset,
            output_shape=target_shape, order=order, mode="constant",
            cval=0, prefilter=order > 1,
        )

    sampled = np.stack([sample(plane) for plane in flat])
    return sampled.reshape((*source.shape[:-2], *target_shape))


_SPATIAL_COMPONENT_KEYS = {
    "static_log_object", "energy_dependent_log_object", "spectral_spatial_map",
    "common_log_objects", "common_exit_waves", "response_log_objects",
    "magnetization", "common_log_objects_by_beam",
    "response_log_objects_by_state_energy", "magnetization_by_state",
    "mode_components",
}


def _map_components(value, old_shape, target_shape, *, force=False):
    """
    Output helper for nested component structures. Resample only recognized spatial maps;
    spectra and observation labels must not be interpreted as images merely because they are
    arrays.
    """
    if isinstance(value, dict):
        return {
            key: _map_components(
                item, old_shape, target_shape,
                force=force or key in _SPATIAL_COMPONENT_KEYS - {"mode_components"},
            ) if force or key in _SPATIAL_COMPONENT_KEYS else item
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_map_components(item, old_shape, target_shape, force=force) for item in value]
    array = np.asarray(value)
    if array.ndim < 2 or array.shape[-2:] != old_shape:
        return value
    return _resample_spatial(array, target_shape, order=1)


@dataclass
class Geometry:
    source_shape: tuple[int, int]
    used_shape: tuple[int, int]
    final_shape: tuple[int, int]
    binning: int
    crop: int
    roi: np.ndarray | None
    mask_used: np.ndarray
    support_used: np.ndarray

    def finish(self, fields, warmup, components, bsmasks, errors):
        """
        Output bookkeeping: attach the exact masks/support/geometry used by retrieval. Returned
        fields remain on the retrieval grid; this method does not invent source-resolution
        information.
        """
        components["supportmask_used"] = self.support_used.copy()
        components["supportmask"] = self.support_used.copy()
        components["mask_pixel"] = self.mask_used.copy()
        components["roi"] = None if self.roi is None else self.roi.copy()
        errors["geometry"] = {
            "source_shape": self.source_shape,
            "retrieval_shape": self.used_shape,
            "output_shape": self.final_shape,
            "binning": self.binning,
            "crop": self.crop,
            "roi": None if self.roi is None else self.roi.copy(),
        }
        return fields, warmup, components, bsmasks, errors


def prepare(holograms, mask_pixel, supportmask, recipe, start_fields=None):
    """
    Crop the detector grid first, then bin intensities and masks.

    Context:
    Input geometry boundary shared by retrieval and refinement. Crop detector edges, sum
    intensity bins, propagate invalid pixels and map the object support. Return prepared
    images, mask, support, optional starts and a Geometry record.
    """
    binning = _positive_integer(recipe["binning"], "binning", 1)
    crop = _positive_integer(recipe["crop"], "crop", 0)
    images = np.asarray(holograms)
    if images.ndim != 3:
        raise ValueError("holograms must have shape (observations, rows, columns).")
    source_shape = tuple(images.shape[-2:])
    mask = np.asarray(mask_pixel)
    support = np.asarray(supportmask)
    if mask.shape not in {source_shape, images.shape}:
        raise ValueError("mask_pixel must be 2D or match the hologram stack.")
    if support.ndim not in (2, 3):
        raise ValueError("supportmask must be 2D or modal 3D.")
    start = None if start_fields is None else np.asarray(start_fields)
    if start is not None and start.shape[-2:] != source_shape:
        raise ValueError("start_fields spatial shape must match the holograms.")
    if 2 * crop >= min(source_shape):
        raise ValueError("recipe['crop'] removes the complete detector image.")
    cropped_shape = tuple(n - 2 * crop for n in source_shape)
    used_shape = tuple(n // binning for n in cropped_shape)
    if min(used_shape) < 1:
        raise ValueError("recipe['binning'] is larger than the cropped hologram.")
    final_shape = used_shape
    if support.shape[-2:] not in {source_shape, used_shape}:
        raise ValueError(
            "supportmask must match the source hologram grid or the cropped/binned retrieval grid."
        )
    trimmed = tuple(n * binning for n in used_shape)
    trim_origin = tuple((n - t) // 2 for n, t in zip(cropped_shape, trimmed))
    detector_origin = tuple(crop + o for o in trim_origin)
    detector_slice = tuple(slice(o, o + t) for o, t in zip(detector_origin, trimmed))
    # Detector cropping changes the object pixel scale. Binning alone does not:
    # N*q_step remains constant after binning, but shrinks after a detector crop.
    object_scale = np.asarray(cropped_shape, dtype=float) / np.asarray(source_shape)
    source_center = (np.asarray(source_shape) - 1) / 2
    output_center = (np.asarray(used_shape) - 1) / 2

    roi = recipe["roi"]
    if roi is not None:
        roi = np.asarray(roi)
        if roi.shape != (4,) or not np.issubdtype(roi.dtype, np.integer):
            raise ValueError("recipe['roi'] must be [row_start, row_stop, column_start, column_stop].")
        roi = roi.astype(int)
        if not (0 <= roi[0] < roi[1] <= source_shape[0] and
                0 <= roi[2] < roi[3] <= source_shape[1]):
            raise ValueError("recipe['roi'] must lie within the input support grid.")
        for axis, bounds in enumerate(((0, 1), (2, 3))):
            mapped = (roi[list(bounds)] - source_center[axis]) * object_scale[axis] + output_center[axis]
            roi[list(bounds)] = np.rint(mapped).astype(int)
            roi[list(bounds)] = np.clip(roi[list(bounds)], 0, used_shape[axis])
        if roi[1] <= roi[0] or roi[3] <= roi[2]:
            raise ValueError("recipe['roi'] falls outside the returned object grid.")

    def blockify(array):
        clipped = array[(...,) + detector_slice]
        return clipped.reshape((*clipped.shape[:-2], used_shape[0], binning,
                                used_shape[1], binning))

    images = blockify(images).sum(axis=(-3, -1))
    mask = blockify(mask != 0).any(axis=(-3, -1)).astype(np.uint8)
    if start is not None:
        start = blockify(start).mean(axis=(-3, -1))
    if support.shape[-2:] == used_shape:
        support = support.copy()
    elif crop:
        support = _resample_spatial(
            support, used_shape, order=0,
            coordinate_scale=np.asarray(source_shape) / np.asarray(cropped_shape),
        )
    else:
        support_origin = tuple((n - t) // 2 for n, t in zip(source_shape, used_shape))
        support_slice = tuple(slice(o, o + t) for o, t in zip(support_origin, used_shape))
        support = support[(...,) + support_slice]
    support = (support != 0).astype(np.uint8)
    nmodes = recipe.get("Nmodes", 1)
    if isinstance(nmodes, (int, np.integer)) and not isinstance(nmodes, bool) and nmodes > 1:
        if support.ndim == 2:
            support = np.repeat(support[None], nmodes, axis=0)
        elif support.shape[0] == 1:
            support = np.repeat(support, nmodes, axis=0)
    geometry = Geometry(source_shape, used_shape, final_shape, binning, crop,
                        roi, mask.copy(), support.copy())
    return images, mask, support, start, geometry


def support_bounding_roi(support, padding_fraction=0.15):
    """
    Centered display ROI enclosing the largest aperture on the current grid.

    Call after crop/binning/support recentering. Fractional padding scales with
    the aperture, so no manually maintained source-to-retrieval pixel offsets.

    Context:
    Display helper only: bound the largest connected aperture on the prepared object grid.
    The physical fit uses its positive-thickness pixel selection, which need not be
    rectangular or identical to this display crop.
    """
    from scipy.ndimage import label
    support = np.asarray(support) != 0
    if support.ndim != 2 or padding_fraction < 0:
        raise ValueError("Expected 2D support and nonnegative padding fraction.")
    labels, count = label(support, structure=np.ones((3, 3)))
    if not count:
        raise ValueError("Support has no aperture.")
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    points = np.argwhere(labels == np.argmax(sizes))
    low, high = points.min(axis=0), points.max(axis=0) + 1
    pad = np.ceil((high - low) * padding_fraction).astype(int)
    low = np.maximum(low - pad, 0)
    high = np.minimum(high + pad, support.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))
