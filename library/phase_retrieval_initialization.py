"""Prepare detector geometry, metadata, fields, and the explicit iteration schedule."""
import numpy as np
from . import phase_retrieval_geometry as geometry
from . import phase_retrieval_validation as validation
from .phase_retrieval_recipe import default_universal_phase_retrieval_recipe, compile_stages
from .real_space_constraints import _mode_supports, _component_centered_modal_supports
from . import magnitude_constraints as magnitude


def _recipe_material_thickness(recipe, input_geometry):
    """
    Map optional object-grid material inputs through crop/bin geometry.

    """
    mask = recipe.get("material_mask")
    thickness = recipe.get("material_thickness")
    if mask is None and thickness is None and not recipe.get("fit_material_thickness", False):
        return None
    source_shape = input_geometry.source_shape
    used_shape = input_geometry.used_shape
    crop = input_geometry.crop
    scale = (np.asarray(source_shape, dtype=float)
             / (np.asarray(source_shape, dtype=float) - 2 * crop))

    def map_object(value, name, order):
        array = np.asarray(value)
        if array.shape == used_shape:
            return array.copy()
        if array.shape != source_shape:
            raise ValueError(
                f"{name} must have source shape {source_shape} or retrieval shape {used_shape}."
            )
        if source_shape == used_shape:
            return array.copy()
        if crop:
            return geometry._resample_spatial(
                array, used_shape, order=order, coordinate_scale=scale,
            )
        # Match geometry.prepare: binning alone keeps object pixel scale.
        origin = tuple((n - u) // 2 for n, u in zip(source_shape, used_shape))
        return array[tuple(slice(o, o + u) for o, u in zip(origin, used_shape))]

    result = (np.ones(used_shape, dtype=float) if thickness is None else
              np.asarray(map_object(thickness, "material_thickness", 1), float))
    if mask is not None:
        material = map_object(mask, "material_mask", 0)
        if not np.all(np.isfinite(material)):
            raise ValueError("material_mask must be finite.")
        result = result * (material != 0)
    if not np.any(result > 0):
        raise ValueError("material_mask/material_thickness contains no material pixels.")
    return result


def initialize_phase_retrieval(holograms, mask_pixel, supportmask, recipe=None, start_fields=None):
    """Prepare geometry, masks, modal fields and schedule; apply no retrieval constraints."""
    recipe = validation.recipe_inputs(default_universal_phase_retrieval_recipe(), recipe)
    mode_factors = recipe["modes"]
    # Geometry is shared; modal supports are expanded after crop/bin preparation.
    preparation_recipe = dict(recipe, Nmodes=1)
    holograms, mask_pixel, supportmask, start_fields, input_geometry = geometry.prepare(
        holograms, mask_pixel, supportmask, preparation_recipe, start_fields
    )
    n_observations, nx, ny = holograms.shape
    material_thickness = _recipe_material_thickness(recipe, input_geometry)
    if material_thickness is not None:
        material_thickness = np.fft.fftshift(material_thickness)
    metadata = validation.normalize_metadata(
        recipe["state"], recipe["energy"],
        recipe["polarization"], recipe["illumination"],
        n_observations,
    )
    recipe["saturated_states"] = validation.normalize_saturated_states(recipe["saturated_states"], metadata["state_names"])
    recipe["observation_weights"] = validation.observation_weights(recipe["observation_weights"], n_observations)
    if recipe["Nmodes"] > 1:
        if recipe["recenter_modal_supports"]:
            modal_supportmask, _ = geometry.recenter_modal_supports(
                supportmask, mode_factors, center=recipe["mode_support_center"],
            )
        elif recipe["mode_support_center"] == "components":
            modal_supportmask = _component_centered_modal_supports(
                supportmask, mode_factors,
            )
        else:
            modal_supportmask = _mode_supports(
                supportmask, mode_factors, (nx, ny),
            )
    else:
        modal_supportmask = supportmask
    thickness_supportmask = None
    if recipe["zero_thickness_outside_support"]:
        if material_thickness is None:
            material_thickness = np.ones((nx, ny), dtype=float)
        thickness_supportmask = np.fft.fftshift(supportmask) != 0
        material_thickness = material_thickness * thickness_supportmask
        if not np.any(material_thickness > 0):
            raise ValueError("The support contains no material pixels for thickness fitting.")
    # Support constraints operate in the shifted object frame.
    # The log-object projection uses the same frame via fft2(fftshift(field)).
    magnetization_supportmask = (
        np.fft.fftshift(supportmask)
        if recipe["zero_magnetization_outside_support"]
        else None
    )
    projection_supportmask = np.fft.fftshift(supportmask) if recipe["projection_constraints_inside_support_only"] else None

    # Convert measured intensities to amplitudes and invalid-pixel masks.
    amplitudes, bsmasks = _prepare_energy_amplitudes(
        holograms,
        mask_pixel,
        hologram_intensity_cutoff_vmin=recipe[
            "hologram_intensity_cutoff_vmin"
        ],
    )

    shape = (n_observations, recipe['Nmodes'], nx, ny)
    if start_fields is not None:
        fields = np.array(start_fields, dtype=complex, copy=True).reshape(shape)
    elif recipe['mode_initialization'] == 'support_fft':
        base = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(modal_supportmask, axes=(-2, -1))), axes=(-2, -1))
        fields = np.broadcast_to(base.reshape(1, recipe['Nmodes'], nx, ny), shape).copy()
        # Match initial total flux; the first Fourier step imposes pixel amplitudes.
        valid = mask_pixel == 0
        power = np.sum(abs(fields)**2, axis=1)
        scale = np.sqrt(np.sum(amplitudes**2 * valid, axis=(-2, -1))
                        / np.maximum(np.sum(power * valid, axis=(-2, -1)), 1e-30))
        fields *= scale[:, None, None, None]
    else:
        phase = np.random.default_rng(recipe['random_seed']).uniform(-np.pi, np.pi, shape)
        fields = amplitudes[:, None] * np.exp(1j * phase) / np.sqrt(recipe['Nmodes'])

    # Resolve timing once; the solver consumes one control entry per complete sweep.
    schedule = compile_stages(recipe)
    kernels = None
    if any(control["RL"] for control in schedule):
        count = 1 if recipe["RL_scope"] == "shared" else n_observations
        kernels = magnitude.RL_kernel(shape=(count, recipe["Nmodes"], nx, ny))
    # Initial feedback and missing-pixel targets are prepared without any projection.
    previous = np.fft.fft2(np.fft.fftshift(fields, axes=(-2, -1)))
    RL_targets = np.where(bsmasks != 0, np.sum(abs(fields)**2, axis=1), amplitudes**2)
    object_support = np.fft.fftshift(np.asarray(modal_supportmask).reshape(recipe['Nmodes'], nx, ny), axes=(-2, -1))
    return dict(
        previous=previous, object_support=object_support, RL_targets=RL_targets,
        recipe=recipe, schedule=schedule, fields=fields, initial=fields.copy(),
        amplitudes=amplitudes, bsmasks=bsmasks,
        support=supportmask,
        metadata=metadata, geometry=input_geometry, material_thickness=material_thickness,
        magnetization_support=magnetization_supportmask, projection_support=projection_supportmask,
        thickness_support=thickness_supportmask,
        RL_kernels=kernels,
    )


def _prepare_energy_amplitudes(holograms, mask_pixel, hologram_intensity_cutoff_vmin=-1):
    """Subtract the optional intensity baseline and exclude unusable detector pixels."""
    intensity = np.asarray(holograms, dtype=float).copy()
    invalid = np.asarray(mask_pixel) != 0
    for image in intensity:
        if hologram_intensity_cutoff_vmin >= 0:
            valid = image[(image != 0) & np.isfinite(image)]
            if valid.size:
                image -= np.percentile(valid, hologram_intensity_cutoff_vmin)
    invalid = invalid | ~np.isfinite(intensity) | (intensity <= 0)
    intensity = np.where(np.isfinite(intensity), np.maximum(intensity, 0), 0)
    return np.sqrt(intensity), invalid
