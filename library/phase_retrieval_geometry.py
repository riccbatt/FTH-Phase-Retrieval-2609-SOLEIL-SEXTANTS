"""Shared detector and object-grid geometry for joint phase retrieval."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform


GEOMETRY_DEFAULTS = {"binning": 1, "crop": 0, "roi": None}


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
    """Crop the detector grid first, then bin intensities and masks."""
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
    if support.ndim not in (2, 3) or support.shape[-2:] != source_shape:
        raise ValueError("supportmask must be 2D or modal 3D and match the holograms.")
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
    if crop:
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
