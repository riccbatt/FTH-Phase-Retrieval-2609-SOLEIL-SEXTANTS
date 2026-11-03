"""Log-object physical factorization, illumination corrections and spectral constraints."""

from __future__ import annotations

import numpy as np
from .phase_retrieval_recipe import default_universal_phase_retrieval_recipe
from . import kramers_kronig as kk
from . import phase_retrieval_validation as validation


def apply_physical_constraint(fields, reconstruction, components, relaxation, iterations):
    """Project all observations together using the prepared physical supports."""
    recipe, metadata = reconstruction['recipe'], reconstruction['metadata']
    return _project_physical_modes(
        fields, metadata['states'], metadata['energies'], metadata['polarizations'], metadata['beams'],
        recipe=recipe, weights=recipe['observation_weights'], relaxation=relaxation,
        saturated_states=recipe['saturated_states'], physical_it=iterations,
        log_floor=recipe['log_floor'], material_thickness=reconstruction['material_thickness'],
        fit_material_thickness=recipe['fit_material_thickness'],
        magnetization_supportmask=reconstruction['magnetization_support'],
        projection_supportmask=reconstruction['projection_support'],
        thickness_supportmask=reconstruction['thickness_support'],
        initial_components=components if 'magnetization' in components else None,
    )


def _project_physical_modes(fields, state, energy, polarization, illumination, **kwargs):
    """Fit primary/grouped physical modes, then share the remaining modes if requested."""
    recipe = kwargs.get('recipe')
    if recipe is None:
        recipe = default_universal_phase_retrieval_recipe()
        kwargs['recipe'] = recipe
    fields = np.asarray(fields)
    single = fields.ndim == 3
    fields = fields[:, None] if single else fields
    groups = _physical_mode_groups(recipe, fields.shape[1])
    restore_all = None
    if groups is not None:
        group_kwargs = dict(kwargs)
        corrections = {}
        normalized = np.asarray(fields)
        if any(recipe.get(key, False) for key in ('fit_illumination_intensity', 'align_global_phase', 'fit_aperture_illumination')):
            normalized = normalized.copy()
            mode_phase_offsets = {}
            if recipe.get('align_global_phase'):
                reference = recipe.get('illumination_intensity_reference_observation', 0)
                reference_mask = (np.asarray(kwargs['material_thickness']) == 0) & (np.asarray(kwargs['projection_supportmask']) != 0)

                reference_objects = np.fft.fft2(np.fft.fftshift(
                    _projection_focus_transform(fields[reference:reference+1],
                                                [energy[reference]], recipe), axes=(-2, -1)))[0]
                for illumination_group in groups[1]:
                    anchor = reference_objects[illumination_group[0], reference_mask]
                    for mode in illumination_group:
                        overlap = np.vdot(anchor, reference_objects[mode, reference_mask])
                        if abs(overlap) == 0:
                            raise ValueError('Shared illumination modes require reference phase overlap.')
                        mode_phase_offsets[mode] = np.angle(overlap)
            for mode in sorted(m for group in groups[0] for m in group):
                normalized[:, mode], restore, details = _reference_illumination_correction(
                    fields[:, mode], energy, illumination, recipe,
                    kwargs.get('material_thickness'), kwargs.get('projection_supportmask'),
                    phase_gauge_offset=mode_phase_offsets.get(mode, 0.0),
                )
                corrections[mode] = (restore, details)
            for illumination_group in groups[1]:
                if len(illumination_group) < 2:
                    continue
                details_list = [corrections[mode][1] for mode in illumination_group]
                if recipe.get('fit_aperture_illumination'):
                    relative = np.stack([
                        d['aperture_illumination_factors'] * np.exp(-1j * d['global_phase_offsets_rad'])[:, None]
                        for d in details_list
                    ])
                    shared = np.exp(np.mean(np.log(abs(relative)), axis=0))
                    phase_mean = np.mean(relative / abs(relative), axis=0)
                    if np.any(abs(phase_mean) < 1e-12):
                        raise ValueError('Shared illumination modes have incompatible reference phases.')
                    shared = shared * np.exp(1j * np.angle(phase_mean))
                    overrides = [shared * np.exp(1j * d['global_phase_offsets_rad'])[:, None] for d in details_list]
                else:
                    shared = np.exp(np.mean(np.log([d['illumination_intensity_factors'] for d in details_list]), axis=0))
                    overrides = [shared] * len(illumination_group)
                for mode, override in zip(illumination_group, overrides):
                    normalized[:, mode], restore, details = _reference_illumination_correction(
                        fields[:, mode], energy, illumination, recipe,
                        kwargs.get('material_thickness'), kwargs.get('projection_supportmask'),
                        factor_override=override, phase_gauge_offset=mode_phase_offsets.get(mode, 0.0),
                    )
                    corrections[mode] = (restore, details)
            group_kwargs['recipe'] = dict(recipe, fit_illumination_intensity=False,
                                         align_global_phase=False, fit_aperture_illumination=False)
        output, components = _project_grouped_physical_modes(
            normalized, state, energy, polarization, illumination,
            groups, **group_kwargs,
        )
        modal_corrections = {}
        for mode, (restore, details) in corrections.items():
            output[:, mode] = restore(output[:, mode], reference_relaxation=kwargs.get('relaxation', 1.0), weights=kwargs.get('weights'))
            modal_corrections[str(mode)] = details
        if corrections:
            components.update(corrections[0][1])
            components['illumination_corrections_by_mode'] = modal_corrections
        physical_modes = sorted(mode for group in groups[0] for mode in group)
    else:
        output = fields
        if any(recipe[key] for key in ('fit_illumination_intensity', 'align_global_phase', 'fit_aperture_illumination')):
            output, restore_all, details = _reference_illumination_correction(
                output, energy, illumination, recipe,
                kwargs.get('material_thickness'), kwargs.get('projection_supportmask'))
        output = output.copy()
        output[:, 0], components = project_fourier_fields_physical(
            output[:, 0], state, energy, polarization, illumination, **kwargs)
        physical_modes = [0]

    # Apply common-mode averaging once, in the illumination-normalized frame.
    residual_modes = sorted(set(range(fields.shape[1])) - set(physical_modes))
    common_groups = []
    if recipe['constrain_nonphysical_modes_common'] and residual_modes:
        output, common_groups = _project_nonphysical_modes_common(
            output, energy, illumination, recipe, weights=kwargs.get('weights'), mode_indices=residual_modes)
    if restore_all is not None:
        output = restore_all(output, reference_relaxation=kwargs.get('relaxation', 1.), weights=kwargs.get('weights'))
        components.update(details)
    components.update(Nmodes=fields.shape[1], physical_model_modes=[mode+1 for mode in physical_modes],
                      nonphysical_modes_common=bool(common_groups), nonphysical_mode_common_groups=common_groups)
    return (output[:, 0] if single else output), components


def project_fourier_fields_physical(fields, state, energy, polarization,
                                    illumination, recipe=None, physical_it=20, log_floor=1e-12,
                                    **fit_options):
    """Focus, fit complex log-objects, and return to the retrieval plane."""
    recipe = default_universal_phase_retrieval_recipe() if recipe is None else recipe
    focused = _projection_focus_transform(fields, energy, recipe)
    logs, components = project_log_objects_physical(
        fourier_field_to_object_log(focused, log_floor), state, energy,
        polarization, illumination, recipe=recipe, iterations=physical_it,
        **fit_options)
    projected = _projection_focus_transform(object_log_to_fourier_field(logs), energy, recipe, inverse=True)
    components['physical_components_frame'] = (
        'focused_object_plane' if recipe['projection_focus_prop_um'] else 'retrieval_object_plane')
    return projected, components


def project_log_objects_physical(
    log_objects,
    state,
    energy,
    polarization,
    illumination,
    weights=None,
    relaxation=1.0,
    saturated_states=None,
    iterations=20,
    recipe=None,
    magnetization_supportmask=None,
    projection_supportmask=None,
    material_thickness=None,
    fit_material_thickness=False,
    thickness_supportmask=None,

    initial_components=None,
):
    """Fit the physical-factorized log-object model for a state/energy/beam dataset."""

    log_objects = np.asarray(log_objects, dtype=complex)
    recipe = default_universal_phase_retrieval_recipe() if recipe is None else recipe
    clip_magnetization = recipe.get("clip_magnetization", True)
    clip_range = np.asarray(recipe.get("magnetization_clip_range", (-1.0, 1.0)), dtype=float)
    binary_magnetization = recipe.get("binary_magnetization", False)
    initial_binary = recipe.get("initial_binary_magnetization", False)
    binary_values = None
    if binary_magnetization:
        binary_values = np.unique(np.asarray(recipe['binary_magnetization_values'], float))
        binary_values = binary_values[np.argsort(abs(binary_values), kind='stable')]
    trend = recipe.get("magnetization_trend", "none")
    median_bounds = recipe.get("bound_magnetization_median", False)
    median_scope = recipe.get("magnetization_median_scope", "whole")
    trend_cutoff = recipe.get("magnetization_trend_intensity_cutoff", 0.01)
    trend_correction = recipe.get("magnetization_trend_correction", "offset")
    aperture = None
    if recipe['physical_projection_object_roi']:
        if material_thickness is None:
            raise ValueError('Object-ROI fitting requires a material thickness map.')
        aperture = np.asarray(material_thickness) > 0
        if projection_supportmask is not None:
            aperture &= np.asarray(projection_supportmask) != 0
        if not np.any(aperture):
            raise ValueError('Physical projection aperture is empty.')
        full_logs = log_objects
        log_objects = log_objects[:, aperture, None]
        if initial_components is not None:
            initial_components = dict(initial_components)
            for key in ('magnetization', 'common_log_objects'):
                initial_components[key] = np.asarray(initial_components[key])[..., aperture, None]
        magnetization_supportmask = None if magnetization_supportmask is None else np.asarray(magnetization_supportmask)[aperture, None]
        thickness_supportmask = None if thickness_supportmask is None else np.asarray(thickness_supportmask)[aperture, None]
        material_thickness = np.asarray(material_thickness)[aperture, None]
        projection_supportmask = np.ones(log_objects.shape[-2:], bool)

    if recipe.get("physical_phase_reference", False):
        log_objects = log_objects.copy()
        phase_groups = {}
        for index, key in enumerate(zip(energy, illumination)):
            phase_groups.setdefault(key, []).append(index)
        for indices in phase_groups.values():
            reference_phase = log_objects[indices[0]].imag.copy()
            log_objects.imag[indices] = reference_phase + np.angle(
                np.exp(1j * (log_objects.imag[indices] - reference_phase))
            )
    n_observations, nx, ny = log_objects.shape
    thickness = np.ones((nx, ny)) if material_thickness is None else np.array(material_thickness, dtype=float, copy=True)
    if recipe["zero_thickness_outside_support"]:
        if thickness_supportmask is None:
            raise ValueError("thickness_supportmask is required when zero_thickness_outside_support=True.")
        thickness_supportmask = np.asarray(thickness_supportmask) != 0
        if thickness_supportmask.shape != (nx, ny):
            raise ValueError("thickness_supportmask must have shape (nx, ny).")
        thickness *= thickness_supportmask
    allowed_material = thickness > 0
    metadata = validation.normalize_metadata(
        state,
        energy,
        polarization,
        illumination,
        n_observations,
    )
    weights = validation.observation_weights(weights, n_observations)
    saturated = validation.normalize_saturated_states(saturated_states, metadata["state_names"])
    n_states = len(metadata["state_names"])
    n_energies = len(metadata["energy_names"])
    n_beams = len(metadata["beam_names"])
    spatial_weighting = recipe.get("physical_spatial_weighting")
    spatial_weights = np.ones((nx, ny), dtype=float)
    if spatial_weighting == "exit_wave_intensity":
        anchors = [a for a, state in enumerate(metadata["states"]) if state in saturated]
        selected = anchors or list(range(n_observations))
        log_intensity = 2 * log_objects[selected].real
        peak = np.max(log_intensity[:, allowed_material])
        spatial_weights = np.mean(np.exp(np.minimum(log_intensity - peak, 0.0)), axis=0)
        spatial_weights[~allowed_material] = 0.0
        scale = np.mean(spatial_weights[allowed_material])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Exit-wave intensity weights require nonzero finite material intensity.")
        spatial_weights /= scale
    if spatial_weighting == "exit_wave_intensity_per_frame":
        log_intensity = 2 * log_objects.real
        peak = np.max(log_intensity[:, allowed_material], axis=1)
        spatial_weights = np.exp(np.minimum(log_intensity - peak[:, None, None], 0.0))
        spatial_weights[:, ~allowed_material] = 0.0
        scale = np.mean(spatial_weights[:, allowed_material], axis=1)
        if np.any(~np.isfinite(scale) | (scale <= 0)):
            raise ValueError("Each frame requires nonzero finite material intensity.")
        spatial_weights /= scale[:, None, None]
    observation_spatial_weights = np.broadcast_to(spatial_weights, log_objects.shape)
    pixel_weights = weights[:, None, None] * np.where(
        allowed_material[None], np.maximum(observation_spatial_weights, 1e-30), 1.0
    )
    if spatial_weighting != "exit_wave_intensity_per_frame":
        pixel_weights = np.broadcast_to(weights[:, None, None], log_objects.shape)
    if magnetization_supportmask is not None:
        magnetization_supportmask = np.asarray(magnetization_supportmask) != 0
        if magnetization_supportmask.shape != (nx, ny):
            raise ValueError("magnetization_supportmask must have shape (nx, ny).")
    if projection_supportmask is not None:
        projection_supportmask = np.asarray(projection_supportmask) != 0
        if projection_supportmask.shape != (nx, ny):
            raise ValueError("projection_supportmask must have shape (nx, ny).")

    common = np.zeros((n_beams, nx, ny), dtype=np.complex128)
    beam_weight = np.zeros((n_beams, nx, ny), dtype=float)
    for observation in range(n_observations):
        beam = metadata["beam_indices"][observation]
        common[beam] += pixel_weights[observation] * log_objects[observation]
        beam_weight[beam] += pixel_weights[observation]
    common /= beam_weight

    charge = np.zeros(n_energies, dtype=np.complex128)
    magnetic = np.ones(n_energies, dtype=np.complex128)
    magnetization = np.zeros((n_states, nx, ny), dtype=float)
    for state, sign in saturated.items():
        state_index = metadata["state_names"].index(state)
        magnetization[state_index] = sign
    magnetization *= allowed_material if fit_material_thickness else thickness > 0
    if magnetization_supportmask is not None:
        magnetization *= magnetization_supportmask

    for energy in range(n_energies):
        selection = metadata["energy_indices"] == energy
        residuals = []
        for observation in np.flatnonzero(selection):
            polarization = metadata["polarizations"][observation]
            residuals.append((log_objects[observation] - common[metadata["beam_indices"][observation]]) / polarization)
        if residuals:
            residuals = np.asarray(residuals)
            moment = np.sum(residuals ** 2)
            if np.isfinite(moment) and abs(moment) > 1e-30:
                direction = np.exp(0.5j * np.angle(moment))
                amplitude = np.max(np.abs(np.real(np.conj(direction) * residuals)))
                magnetic[energy] = direction * max(amplitude, 1e-12)
        anchor_rhs = 0j
        anchor_norm = 0.0
        anchor_thickness = thickness.copy()
        if magnetization_supportmask is not None:
            anchor_thickness *= magnetization_supportmask
        for beam in range(n_beams):
            anchors = [
                a for a in np.flatnonzero(selection & (metadata["beam_indices"] == beam))
                if metadata["states"][a] in saturated
            ]
            if len(anchors) < 2:
                continue
            anchor_weights = weights[anchors, None, None] * np.maximum(observation_spatial_weights[anchors], 1e-30)
            coefficients = np.asarray([
                metadata["polarizations"][a] * saturated[metadata["states"][a]]
                for a in anchors
            ])
            centered = coefficients[:, None, None] - (
                np.sum(coefficients[:, None, None] * anchor_weights, axis=0)
                / np.sum(anchor_weights, axis=0)
            )
            for a, weight, coefficient in zip(anchors, anchor_weights, centered):
                basis = coefficient * anchor_thickness
                anchor_rhs += np.vdot(basis, weight * log_objects[a])
                anchor_norm += float(np.vdot(basis, weight * basis).real)
        if anchor_norm > 1e-30:
            magnetic[energy] = anchor_rhs / anchor_norm

    if initial_components is None:
        numerator = np.zeros_like(magnetization)
        denominator = np.zeros_like(magnetization)
        for energy in range(n_energies):
            for beam in range(n_beams):
                observations = np.flatnonzero((metadata["energy_indices"] == energy) & (metadata["beam_indices"] == beam))
                if not len(observations):
                    continue
                anchors = [a for a in observations if metadata["states"][a] in saturated]
                baseline = common[beam]
                if anchors:
                    anchor_weights = pixel_weights[anchors]
                    anchor_signs = np.asarray([
                        metadata["polarizations"][a] * saturated[metadata["states"][a]]
                        for a in anchors
                    ])
                    baseline = np.sum(
                        anchor_weights * (log_objects[anchors] - anchor_signs[:, None, None] * magnetic[energy] * thickness),
                        axis=0,
                    ) / np.sum(anchor_weights, axis=0)
                for observation in observations:
                    state = metadata["state_indices"][observation]
                    coefficient = metadata["polarizations"][observation] * magnetic[energy] * thickness
                    weight = pixel_weights[observation]
                    numerator[state] += weight * np.real(np.conj(coefficient) * (log_objects[observation] - baseline))
                    denominator[state] += weight * abs(coefficient) ** 2
        for state, name in enumerate(metadata["state_names"]):
            if name not in saturated:
                magnetization[state] = np.divide(
                    numerator[state], denominator[state],
                    out=np.zeros((nx, ny)),
                    where=denominator[state] > 1e-30,
                )
        if initial_binary:
            magnetization = _snap_magnetization(magnetization, np.array([-1.0, 1.0]))
        if binary_magnetization:
            magnetization = _snap_magnetization(magnetization, binary_values)
        elif clip_magnetization:
            magnetization = np.clip(magnetization, *clip_range)
        magnetization *= allowed_material
        if magnetization_supportmask is not None:
            magnetization *= magnetization_supportmask

    reused_fit = initial_components is not None
    if reused_fit:
        for key in ("state_names", "energy_names", "beam_names"):
            if list(initial_components[key]) != list(metadata[key]):
                raise ValueError(f"initial_components {key} must match the current fit.")

        def initial_array(key, shape, dtype):
            value = np.asarray(initial_components[key], dtype=dtype)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"initial_components {key} must be finite with shape {shape}.")
            return value.copy()

        common = initial_array("common_log_objects", common.shape, complex)
        charge = initial_array("charge_response", charge.shape, complex)
        magnetic = initial_array("magnetic_response", magnetic.shape, complex)
        magnetization = initial_array("magnetization", magnetization.shape, float)
        if binary_magnetization:
            magnetization = _snap_magnetization(magnetization, binary_values)
        elif clip_magnetization:
            magnetization = np.clip(magnetization, *clip_range)
        for state, sign in saturated.items():
            magnetization[metadata["state_names"].index(state)].fill(sign)
        magnetization *= allowed_material
        if magnetization_supportmask is not None:
            magnetization *= magnetization_supportmask

    trend_preferences = {}
    trend_spatial_weights = np.ones((nx, ny), dtype=float)
    if trend != "none" or median_bounds:
        log_intensity = 2 * log_objects.real
        peak = np.max(log_intensity[:, allowed_material])
        trend_spatial_weights = np.mean(np.exp(np.minimum(log_intensity - peak, 0.0)), axis=0)
        trend_spatial_weights[~allowed_material] = 0.0
        trend_spatial_weights[trend_spatial_weights < trend_cutoff * np.max(trend_spatial_weights)] = 0.0
        if binary_values is not None:
            trend_spatial_weights = (trend_spatial_weights > 0).astype(float)
    if magnetization_supportmask is not None:
        trend_spatial_weights[~magnetization_supportmask] = 0.0
    trend_state_weights = np.bincount(metadata["state_indices"], weights=weights, minlength=n_states)

    def constrain_trend():
        active = thickness > 0
        if magnetization_supportmask is not None:
            active &= magnetization_supportmask
        _constrain_magnetization_trend(
            magnetization,
            active,
            metadata["state_names"],
            saturated,
            trend,
            clip_magnetization,
            recipe.get("_magnetization_trend_groups"),
            binary_values,
            trend_preferences,
            trend_spatial_weights,
            trend_state_weights,
            trend_correction,
            median_bounds,
            clip_range=(-1.0, 1.0) if binary_magnetization else clip_range,
            median_scope=median_scope,
        )

    constrain_trend()
    initial_magnetization = magnetization.copy()

    for _ in range(int(iterations)):
        common.fill(0.0)
        material = _material_log_objects(thickness, charge, magnetic, magnetization, metadata)
        np.add.at(common, metadata['beam_indices'], pixel_weights * (log_objects - material))
        common /= beam_weight

        magnetization_fit_thickness = thickness
        if fit_material_thickness:
            positive = thickness[allowed_material & (thickness > 0)]
            probe_thickness = 1e-3 * np.mean(positive) if positive.size else 1e-3
            magnetization_fit_thickness = np.where(
                allowed_material & (thickness == 0), probe_thickness, thickness,
            )

        trend_preferences.clear()
        for state, state_name in enumerate(metadata["state_names"]):
            if state_name in saturated:
                magnetization[state].fill(saturated[state_name])
                magnetization[state] *= allowed_material if fit_material_thickness else thickness > 0
                if magnetization_supportmask is not None:
                    magnetization[state] *= magnetization_supportmask
                continue
            numerator = np.zeros((nx, ny), dtype=float)
            denominator = np.zeros((nx, ny), dtype=float)
            for observation in np.flatnonzero(metadata["state_indices"] == state):
                energy = metadata["energy_indices"][observation]
                coefficient = metadata["polarizations"][observation] * magnetic[energy] * magnetization_fit_thickness
                residual = log_objects[observation] - common[metadata["beam_indices"][observation]] - thickness * charge[energy]
                numerator += pixel_weights[observation] * np.real(np.conj(coefficient) * residual)
                denominator += pixel_weights[observation] * abs(coefficient) ** 2
            magnetization[state] = np.divide(
                numerator,
                denominator,
                out=(magnetization[state].copy() if fit_material_thickness else np.zeros_like(numerator)),
                where=denominator > 1e-30,
            )
            if binary_magnetization:
                if trend != "none":
                    active = thickness > 0
                    if magnetization_supportmask is not None:
                        active &= magnetization_supportmask
                    active &= trend_spatial_weights > 0
                    trend_preferences[state] = magnetization[state][active].copy()
                magnetization[state] = _snap_magnetization(magnetization[state], binary_values)
            elif clip_magnetization:
                magnetization[state] = np.clip(magnetization[state], *clip_range)
            if magnetization_supportmask is not None:
                magnetization[state] *= magnetization_supportmask

        constrain_trend()

        for energy in range(n_energies):
            observations = np.flatnonzero(metadata["energy_indices"] == energy)
            gram = np.zeros((2, 2), dtype=np.complex128)
            rhs = np.zeros(2, dtype=np.complex128)
            for observation in observations:
                state = metadata["state_indices"][observation]
                magnetic_basis = metadata["polarizations"][observation] * magnetization[state] * thickness
                residual = log_objects[observation] - common[metadata["beam_indices"][observation]]
                weight = weights[observation] * observation_spatial_weights[observation]
                gram[0, 0] += np.vdot(thickness, weight * thickness)
                gram[0, 1] += np.vdot(thickness, weight * magnetic_basis)
                gram[1, 1] += np.vdot(magnetic_basis, weight * magnetic_basis)
                rhs[0] += np.vdot(thickness, weight * residual)
                rhs[1] += np.vdot(magnetic_basis, weight * residual)
            gram[1, 0] = np.conj(gram[0, 1])
            coefficients = np.linalg.pinv(gram) @ rhs
            charge[energy], magnetic[energy] = coefficients

        if fit_material_thickness:
            numerator = np.zeros((nx, ny), dtype=float)
            denominator = np.zeros((nx, ny), dtype=float)
            for beam in range(n_beams):
                observations = np.flatnonzero(metadata["beam_indices"] == beam)
                total_weight = np.sum(pixel_weights[observations], axis=0)
                responses = np.stack([
                    charge[metadata["energy_indices"][a]]
                    + metadata["polarizations"][a]
                    * magnetic[metadata["energy_indices"][a]]
                    * magnetization[metadata["state_indices"][a]]
                    for a in observations
                ])
                mean_response = np.sum(responses * pixel_weights[observations], axis=0) / total_weight
                mean_log = np.sum(log_objects[observations] * pixel_weights[observations], axis=0) / total_weight
                for index, observation in enumerate(observations):
                    direction = responses[index] - mean_response
                    residual = log_objects[observation] - mean_log
                    numerator += pixel_weights[observation] * np.real(np.conj(direction) * residual)
                    denominator += pixel_weights[observation] * np.abs(direction) ** 2
            if np.any(denominator[allowed_material] > 1e-30):
                updated = np.maximum(0, np.divide(numerator, denominator, out=thickness.copy(), where=denominator > 1e-30))
                updated[~allowed_material] = 0
                scale = np.mean(updated[allowed_material])
                if scale > 1e-30:
                    thickness = updated / scale
                    charge *= scale
                    magnetic *= scale

        charge_offset = np.mean(charge)
        charge -= charge_offset
        common += charge_offset * thickness

        responses = [charge, magnetic]
        for i, kind in enumerate(('charge', 'magnetic')):
            responses[i] = constrain_complex_spectrum(
                responses[i], spectral_constraint=recipe[kind+'_spectral_constraint'],
                energy_values=recipe['energy_values'], absorption_part=recipe[kind+'_absorption_part'],
                kk_sign=recipe['kk_sign'], kk_subtract_baseline=recipe['kk_subtract_baseline'],
                kk_normalize_input=recipe['kk_normalize_input'],
                real_range=recipe[kind+'_response_real_range'], imag_range=recipe[kind+'_response_imag_range'])
        charge, magnetic = responses

    magnetization *= thickness > 0
    trend_preferences.clear()
    constrain_trend()

    if fit_material_thickness:
        common.fill(0.0)
        material = _material_log_objects(thickness, charge, magnetic, magnetization, metadata)
        np.add.at(common, metadata['beam_indices'], pixel_weights * (log_objects - material))
        common /= beam_weight

    fitted = common[metadata['beam_indices']] + _material_log_objects(
        thickness, charge, magnetic, magnetization, metadata)
    projected = fitted if relaxation == 1 else (1 - relaxation) * log_objects + relaxation * fitted
    if material_thickness is not None or thickness_supportmask is not None:
        projected[:, thickness == 0] = fitted[:, thickness == 0]
    if projection_supportmask is not None:
        if material_thickness is not None or thickness_supportmask is not None:
            projection_supportmask = projection_supportmask | (thickness == 0)
        projected = np.where(projection_supportmask[None], projected, log_objects)

    residual = log_objects - fitted
    components = {
        "projection_model": "physical_factorized",
        "physical_fit_reused": reused_fit,
        "initial_magnetization": initial_magnetization,
        "magnetization_initialization": (
            "previous_fit" if reused_fit else "nearest_sign" if initial_binary else "measured_contrast"
        ),
        "physical_spatial_weighting": spatial_weighting,
        "physical_spatial_weights": spatial_weights,
        "magnetization_trend": trend,
        "magnetization_trend_intensity_cutoff": float(trend_cutoff),
        "magnetization_trend_correction": trend_correction,
        "bound_magnetization_median": median_bounds,
        "magnetization_median_scope": median_scope,
        "magnetization_trend_spatial_weights": trend_spatial_weights,
        "magnetization_trend_means": np.array([
            np.average(item[(thickness > 0) & (trend_spatial_weights > 0)], weights=trend_spatial_weights[(thickness > 0) & (trend_spatial_weights > 0)])
            if np.any((thickness > 0) & (trend_spatial_weights > 0)) else np.nan
            for item in magnetization
        ]),
        "weighted_fit_residual_rms": float(
            np.sqrt(
                np.sum(weights[:, None, None] * spatial_weights * abs(residual) ** 2)
                / np.sum(weights[:, None, None] * observation_spatial_weights)
            )
        ),
        "common_log_objects": common,
        "common_log_objects_by_beam": {name: common[index] for index, name in enumerate(metadata["beam_names"])},
        "common_exit_waves": np.exp(common),
        "charge_response": charge,
        "magnetic_response": magnetic,
        "magnetization": magnetization,
        "material_thickness": thickness.copy(),
        "material_thickness_fitted": bool(fit_material_thickness),
        "thickness_supportmask_applied": thickness_supportmask is not None,
        "magnetization_by_state": {name: magnetization[index] for index, name in enumerate(metadata["state_names"])},
        "magnetization_supportmask_applied": magnetization_supportmask is not None,
        "physical_projection_supportmask_applied": projection_supportmask is not None,
        "saturated_states": saturated,
        "state_names": metadata["state_names"],
        "energy_names": metadata["energy_names"],
        "beam_names": metadata["beam_names"],
        "charge_gauge": "zero_mean_before_optional_constraints",
        "magnetic_scale_anchored": bool(saturated),
        "charge_absolute_gauge_anchored": True,
        "identifiable": bool(saturated) and not fit_material_thickness,
        "fit_residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        "magnetization_bounds": (
            (float(binary_values.min()), float(binary_values.max())) if binary_magnetization else tuple(clip_range) if clip_magnetization else None
        ),
        "binary_magnetization": binary_magnetization,
        "binary_magnetization_values": binary_values,
    }
    if aperture is not None:
        full_projected = full_logs.copy()
        full_projected[:, aperture] = projected[..., 0]
        projected = full_projected
        components = _expand_spatial_components(components, aperture)
        components.update(physical_phase_reference=bool(recipe['physical_phase_reference']),
                          physical_projection_roi_mask=aperture.copy(),
                          physical_projection_fit_pixels=int(aperture.sum()),
                          physical_projection_full_pixels=int(aperture.size))
    return projected, components


def _project_grouped_physical_modes(fields, state, energy, polarizations, illumination,
                                    groups, **kwargs):
    """Joint physical fit with independent sharing of magnetic maps and beam fields."""
    recipe = kwargs['recipe']
    magnetic_groups, illumination_groups = groups
    selected = sorted(mode for group in magnetic_groups for mode in group)
    magnetic_id = {mode: i for i, group in enumerate(magnetic_groups) for mode in group}
    illumination_id = {mode: i for i, group in enumerate(illumination_groups) for mode in group}
    thickness = kwargs.get('material_thickness')
    if thickness is None:
        raise ValueError('Physical mode groups require a material thickness/aperture map.')
    thickness = np.asarray(thickness)
    aperture = thickness > 0
    support = kwargs.get('projection_supportmask')
    if support is not None:
        aperture &= np.asarray(support) != 0
    if not np.any(aperture):
        raise ValueError('Physical mode aperture is empty.')
    n_observations = len(fields)
    metadata = validation.normalize_metadata(state, energy, polarizations, illumination, n_observations)
    records = [(observation, mode) for observation in range(n_observations) for mode in selected]

    objects = np.fft.fft2(np.fft.fftshift(
        _projection_focus_transform(fields, energy, recipe), axes=(-2, -1)))
    values = objects[:, selected][:, :, aperture].reshape(len(records), -1, 1)
    logs = np.log(np.maximum(abs(values), kwargs.get('log_floor', 1e-12))) + 1j*np.angle(values)
    state_keys = [f's{metadata["state_indices"][obs]}_m{magnetic_id[mode]}' for obs, mode in records]
    beam_keys = [f'b{metadata["beam_indices"][obs]}_i{illumination_id[mode]}' for obs, mode in records]
    anchors = validation.normalize_saturated_states(kwargs.get('saturated_states'), metadata['state_names'])
    joint_anchors = {
        f's{i}_m{group}': anchors[state]
        for i, state in enumerate(metadata['state_names'])
        if state in anchors
        for group in range(len(magnetic_groups))
    }
    weights = validation.observation_weights(kwargs.get('weights'), n_observations)
    initial = kwargs.get('initial_components')
    previous = None if initial is None else initial.get('modal_physical_fit')
    if previous is not None and (
        initial['magnetization_mode_groups'] != magnetic_groups
        or initial['illumination_mode_groups'] != illumination_groups
    ):
        raise ValueError('Physical mode sharing changed; start a new physical run.')


    local_recipe = dict(recipe, physical_projection_object_roi=False)
    joint_names = list(dict.fromkeys(state_keys))
    local_recipe['_magnetization_trend_groups'] = [
        [joint_names.index(f's{i}_m{group}') for i in range(len(metadata['state_names']))]
        for group in range(len(magnetic_groups))
    ]

    projected, fit = project_log_objects_physical(
        logs,
        state_keys,
        [energy[obs] for obs, _ in records],
        [polarizations[obs] for obs, _ in records],
        beam_keys,
        weights=np.asarray([weights[obs] for obs, _ in records]),
        relaxation=kwargs.get('relaxation', 1.0),
        saturated_states=joint_anchors,
        iterations=kwargs.get('physical_it', 20),
        recipe=local_recipe,
        magnetization_supportmask=(None if kwargs.get('magnetization_supportmask') is None else np.asarray(kwargs.get('magnetization_supportmask'))[aperture, None]),
        projection_supportmask=np.ones((int(aperture.sum()), 1), bool),
        material_thickness=(None if thickness is None else np.asarray(thickness)[aperture, None]),
        fit_material_thickness=kwargs.get('fit_material_thickness', False),
        thickness_supportmask=(None if kwargs.get('thickness_supportmask') is None else np.asarray(kwargs.get('thickness_supportmask'))[aperture, None]),
        initial_components=previous,
    )
    for index, (observation, mode) in enumerate(records):
        objects[observation, mode, aperture] = np.exp(projected[index, :, 0])
    output = np.asarray(fields).copy()
    output[:, selected] = _projection_focus_transform(
        np.fft.ifftshift(np.fft.ifft2(objects[:, selected]), axes=(-2, -1)), energy, recipe, inverse=True)

    primary_states = [fit['state_names'].index(f's{i}_m{magnetic_id[0]}') for i in range(len(metadata['state_names']))]
    primary_beams = [fit['beam_names'].index(f'b{i}_i{illumination_id[0]}') for i in range(len(metadata['beam_names']))]
    components = dict(fit)
    components.update(
        state_names=metadata['state_names'],
        beam_names=metadata['beam_names'],
        magnetization=_expand_spatial_components(fit['magnetization'][primary_states], aperture),
        initial_magnetization=_expand_spatial_components(fit['initial_magnetization'][primary_states], aperture),
        common_log_objects=_expand_spatial_components(fit['common_log_objects'][primary_beams], aperture),
        common_exit_waves=_expand_spatial_components(fit['common_exit_waves'][primary_beams], aperture),
        material_thickness=_expand_spatial_components(fit['material_thickness'], aperture),
        saturated_states=anchors,
        modal_physical_fit=fit,
        modal_physical_fit_aperture=aperture,
        magnetization_mode_groups=magnetic_groups,
        illumination_mode_groups=illumination_groups,
        physical_model_modes=[mode + 1 for mode in selected],
        Nmodes=fields.shape[1],
        physical_components_frame='focused_object_plane' if recipe.get('projection_focus_prop_um', 0.0) else 'retrieval_object_plane',
    )
    components['magnetization_by_state'] = dict(zip(metadata['state_names'], components['magnetization']))
    components['common_log_objects_by_beam'] = dict(zip(metadata['beam_names'], components['common_log_objects']))
    components['common_exit_waves_by_beam'] = dict(zip(metadata['beam_names'], components['common_exit_waves']))
    components['magnetization_by_mode_group_packed'] = {
        str(group): fit['magnetization'][[fit['state_names'].index(f's{i}_m{group}') for i in range(len(metadata['state_names']))]]
        for group in range(len(magnetic_groups))
    }
    return output, components


def _reference_illumination_correction(fields, energy, illumination, recipe, thickness, support, factor_override=None, phase_gauge_offset=0.):
    """Estimate scalar/aperture complex gains from material-free support only."""
    from scipy.ndimage import label

    aperture_fit = recipe.get("fit_aperture_illumination", False)
    phase_fit = recipe.get("align_global_phase", False)
    object_mode = recipe.get("object_illumination_mode", "pooled")
    factor_type = recipe.get("aperture_factor_type", "complex")
    if object_mode not in {"none", "pooled", "distance_weighted", "nearest", "linear_plane"}:
        raise ValueError("object_illumination_mode must be none, pooled, distance_weighted, nearest, or linear_plane.")
    if factor_type not in {"complex", "amplitude"}:
        raise ValueError("aperture_factor_type must be complex or amplitude.")
    intensity_fit = recipe.get("fit_illumination_intensity", False) or aperture_fit
    if thickness is None or support is None:
        raise ValueError("Reference illumination correction requires material thickness and support.")
    reference_mask = (np.asarray(thickness) == 0) & (np.asarray(support) != 0)
    if not np.any(reference_mask):
        raise ValueError("Reference illumination correction requires material-free reference pixels inside the support.")
    reference = recipe.get("illumination_intensity_reference_observation", 0)
    if (isinstance(reference, bool) or not isinstance(reference, (int, np.integer))
            or not 0 <= reference < len(fields)):
        raise ValueError("illumination_intensity_reference_observation must index an observation.")
    if len(set(energy)) != 1 or len(set(illumination)) != 1:
        raise ValueError("Reference illumination correction currently requires one energy and one beam.")

    array = np.asarray(fields)
    modes = array[:, None] if array.ndim == 3 else array

    objects = np.fft.fft2(np.fft.fftshift(_projection_focus_transform(modes, energy, recipe), axes=(-2, -1)))
    primary = objects[:, 0]
    powers = np.sum(abs(primary[:, reference_mask]) ** 2, axis=1)
    if np.any(~np.isfinite(powers)) or np.any(powers <= 0):
        raise ValueError("Reference apertures require finite positive reconstructed power in every observation.")
    overlap = np.sum(primary[reference, reference_mask].conj()[None] * primary[:, reference_mask], axis=1)
    if (phase_fit or (aperture_fit and factor_type == "complex")) and (
        np.any(~np.isfinite(overlap))
        or np.any(abs(overlap) <= 1e-12 * np.sqrt(powers * powers[reference]))
    ):
        raise ValueError("Reference overlap is too small for global phase alignment.")
    phases = np.angle(overlap) + phase_gauge_offset if phase_fit else np.zeros(len(array))
    gains = powers / powers[reference] if intensity_fit else np.ones(len(array))
    pooled = np.sqrt(gains) * np.exp(1j * phases)
    correction = np.broadcast_to(pooled[:, None, None], primary.shape).copy()
    details = dict(
        illumination_intensity_factors=gains,
        illumination_intensity_reference_observation=int(reference),
        illumination_intensity_reference_pixels=int(reference_mask.sum()),
        illumination_intensity_fit="primary reference-aperture power ratio",
        global_phase_offsets_rad=phases,
        pooled_reference_phase_offsets_rad=np.angle(overlap),
        global_phase_aligned=bool(phase_fit),
    )

    if aperture_fit:
        centered_labels, count = label(np.fft.fftshift(np.asarray(support) != 0))
        labels = np.fft.ifftshift(centered_labels)
        centroids = np.asarray([
            np.mean(np.argwhere(centered_labels == k), axis=0)
            for k in range(1, count + 1)
        ])
        object_ids = [k for k in range(1, count + 1) if np.any(np.asarray(thickness)[labels == k] > 0)]
        reference_ids = [k for k in range(1, count + 1) if k not in object_ids]
        if not reference_ids:
            raise ValueError("Aperture correction requires separate material-free reference apertures.")
        gauge = np.exp(1j * phases)
        factors = np.broadcast_to(gauge[:, None], (len(array), count)).copy()
        kinds = ["reference" if k in reference_ids else "object_" + object_mode for k in range(1, count + 1)]

        for aperture in reference_ids:
            mask = labels == aperture
            reference_values = primary[reference, mask]
            norm = np.sum(abs(reference_values) ** 2)
            if norm <= 0 or not np.isfinite(norm):
                raise ValueError("Each reference aperture requires finite positive reference power.")
            factor = np.sum(reference_values.conj()[None] * primary[:, mask], axis=1) / norm
            if np.any(~np.isfinite(factor)) or np.any(abs(factor) < 1e-12):
                raise ValueError("Reference aperture overlap is too small for a complex factor.")
            factors[:, aperture - 1] = factor * np.exp(1j * phase_gauge_offset) if factor_type == "complex" else abs(factor) * gauge

        object_weights = np.zeros((len(object_ids), len(reference_ids)))
        reference_factors = factors[:, np.asarray(reference_ids) - 1]
        for j, aperture in enumerate(object_ids):
            if object_mode == "none":
                factor = gauge
            elif object_mode == "pooled":
                reference_power = np.asarray([
                    np.sum(abs(primary[reference, labels == k]) ** 2) for k in reference_ids
                ])
                object_weights[j] = reference_power / reference_power.sum()
                pooled_phase = np.angle(overlap) + phase_gauge_offset if factor_type == "complex" else phases
                factor = np.sqrt(gains) * np.exp(1j * pooled_phase)
            else:
                distance = np.linalg.norm(centroids[np.asarray(reference_ids) - 1] - centroids[aperture - 1], axis=1)
                if object_mode == "linear_plane":
                    offsets = centroids[np.asarray(reference_ids) - 1] - centroids[aperture - 1]
                    scale = max(float(np.max(abs(offsets))), 1.0)
                    design = np.column_stack((np.ones(len(reference_ids)), offsets / scale))
                    if np.linalg.matrix_rank(design) < 3:
                        raise ValueError("linear_plane OH factors require at least three non-collinear reference apertures.")
                    spatial_weights = np.linalg.pinv(design)[0]
                elif object_mode == "nearest":
                    spatial_weights = (distance == np.min(distance)).astype(float)
                else:
                    spatial_weights = 1.0 / np.maximum(distance, 1.0) ** 2
                spatial_weights /= spatial_weights.sum()
                object_weights[j] = spatial_weights
                branch_phase = np.angle(overlap) + phase_gauge_offset if factor_type == "complex" else phases
                relative_phase = np.angle(reference_factors * np.exp(-1j * branch_phase[:, None]))
                factor = np.exp(
                    np.sum(spatial_weights[None] * (np.log(abs(reference_factors)) + 1j * relative_phase), axis=1)
                    + 1j * branch_phase
                )
            factors[:, aperture - 1] = factor

        correction = np.broadcast_to(gauge[:, None, None], primary.shape).copy()
        for aperture in range(1, count + 1):
            correction[:, labels == aperture] = factors[:, aperture - 1, None]
        oh_factor = factors[:, object_ids[0] - 1] if object_ids else gauge
        details.update(
            illumination_intensity_factors=abs(oh_factor) ** 2,
            pooled_reference_intensity_factors=gains,
            aperture_illumination_factors=factors,
            aperture_illumination_labels=labels,
            aperture_illumination_kinds=kinds,
            aperture_centroids_yx=centroids,
            object_aperture_ids=np.asarray(object_ids),
            reference_aperture_ids=np.asarray(reference_ids),
            object_reference_weights=object_weights,
            object_illumination_constraint=object_mode,
            aperture_factor_type=factor_type,
        )

    if factor_override is not None:
        if aperture_fit:
            factors = np.asarray(factor_override, complex)
            if factors.shape != details['aperture_illumination_factors'].shape:
                raise ValueError('Shared aperture factors have incompatible shape.')
            for aperture in range(1, count + 1):
                correction[:, labels == aperture] = factors[:, aperture - 1, None]
            details['aperture_illumination_factors'] = factors.copy()
            details['illumination_intensity_factors'] = abs(factors[:, object_ids[0] - 1]) ** 2
        else:
            shared_gain = np.asarray(factor_override, float)
            correction = np.broadcast_to(
                np.sqrt(shared_gain)[:, None, None] * np.exp(1j * phases)[:, None, None],
                primary.shape,
            ).copy()
            details['illumination_intensity_factors'] = shared_gain.copy()

    def to_fields(value):
        result = _projection_focus_transform(np.fft.ifftshift(np.fft.ifft2(value), axes=(-2, -1)), energy, recipe, inverse=True)
        return result[:, 0] if array.ndim == 3 else result

    normalized = to_fields(objects / correction[:, None])

    def restore(value, reference_relaxation=0.0, weights=None):
        value = value[:, None] if array.ndim == 3 else value
        focused_objects = np.fft.fft2(np.fft.fftshift(_projection_focus_transform(value, energy, recipe), axes=(-2, -1)))
        if aperture_fit and reference_relaxation > 0:
            observation_weights = validation.observation_weights(weights, len(array))
            residual_norm = 0.0
            residual_pixels = 0
            for aperture in reference_ids:
                mask = labels == aperture
                values = focused_objects[:, 0, mask]
                precision = observation_weights * abs(factors[:, aperture - 1]) ** 2
                common_reference = np.sum(precision[:, None] * values, axis=0) / precision.sum()
                residual_norm += float(np.sum(abs(values - common_reference) ** 2))
                residual_pixels += values.size
                focused_objects[:, 0, mask] = ((1 - reference_relaxation) * values + reference_relaxation * common_reference)
            details["reference_aperture_common_profile"] = True
            details["reference_aperture_profile_residual_rms"] = float(np.sqrt(residual_norm / max(residual_pixels, 1)))
        return to_fields(focused_objects * correction[:, None])

    return normalized, restore, details


def _project_nonphysical_modes_common(fields, energy, illumination, recipe, mode_indices, weights=None):
    """Phase-align and average residual modes within each energy/beam group."""
    output = np.asarray(fields).copy()
    relaxation = recipe['nonphysical_modes_common_relaxation']
    if relaxation == 0:
        return output, []
    weights = np.ones(len(fields)) if weights is None else np.asarray(weights)
    groups = {}
    for observation, key in enumerate(zip(energy, illumination)):
        groups.setdefault(key, []).append(observation)
    records = []
    for mode in mode_indices:
        for (energy, beam), indices in groups.items():
            values = fields[indices, mode].copy()
            anchor = values[0]
            if not np.any(anchor):
                anchor = values[np.argmax(np.sum(abs(values)**2, axis=(-2, -1)))]
            overlap = np.sum(anchor.conj() * values, axis=(-2, -1))
            values *= np.exp(-1j*np.angle(overlap))[:, None, None]
            common = np.average(values, axis=0, weights=weights[indices])
            output[indices, mode] = (1-relaxation)*values + relaxation*common
            records.append(dict(mode_index=mode, energy=energy, beam=beam, observations=indices))
    return output, records


def constrain_complex_spectrum(response, spectral_constraint='free', energy_values=None,
                               absorption_part='real', kk_sign=1., kk_subtract_baseline=True,
                               kk_normalize_input=False, real_range=None, imag_range=None):
    """Constrain one complex response by KK and optional component bounds."""
    response = np.array(response, dtype=complex, copy=True)
    if spectral_constraint != 'free':
        if spectral_constraint != 'kk':
            raise ValueError("spectral_constraint must be 'free' or 'kk'.")
        absorption, dispersion = (response.real, response.imag) if absorption_part == 'real' else (response.imag, response.real)
        constrained = -kk_sign * kk.kramers_kronig_real_from_imaginary(
            energy_values, absorption, subtract_baseline=kk_subtract_baseline,
            normalize_input=kk_normalize_input, output_mean=dispersion.mean())
        if absorption_part == 'real':
            response.imag = constrained
        else:
            response.real = constrained
    if real_range is not None:
        response.real = np.clip(response.real, *real_range)
    if imag_range is not None:
        response.imag = np.clip(response.imag, *imag_range)
    return response


def _projection_focus_transform(fields, energy, recipe, inverse=False):
    """Reversible detector-plane propagation, batched over observations and modes."""
    array = np.asarray(fields)
    distance = recipe['projection_focus_prop_um'] * 1e-6
    if distance == 0:
        return array
    setup = recipe['projection_focus_setup']
    energy = np.full(len(array), setup['energy']) if 'energy' in setup else np.asarray(energy, float)
    wavelength = 1.2398419843320026e-6 / energy
    if recipe['projection_focus_integer_wavelength']:
        distance = np.round(distance / wavelength) * wavelength
    rows, columns = array.shape[-2:]
    yy, xx = np.mgrid[:rows, :columns]
    angular = 1 - (setup['px_size'] / setup['ccd_dist'])**2 * ((yy-rows/2)**2 + (xx-columns/2)**2)
    if np.any(angular < 0):
        raise ValueError('Projection-focus geometry produces evanescent detector pixels.')
    phase = 2*np.pi * distance / wavelength
    phase = np.broadcast_to(phase, (len(array),))[:, None, None] * np.sqrt(angular)
    phase += recipe['projection_focus_phase_rad']
    if array.ndim == 4:
        phase = phase[:, None]
    return array * np.exp((-1j if inverse else 1j) * phase)


def fourier_field_to_object_log(fields, log_floor=1e-12):
    """Centered detector fields to log-amplitude and wrapped object phase."""
    obj = np.fft.fft2(np.fft.fftshift(fields, axes=(-2, -1)))
    return np.log(np.maximum(abs(obj), log_floor)) + 1j*np.angle(obj)


def object_log_to_fourier_field(logs):
    return np.fft.ifftshift(np.fft.ifft2(np.exp(logs)), axes=(-2, -1))


def _physical_mode_groups(recipe, n_modes):
    """Resolve independent magnetic-map and illumination sharing partitions."""
    magnetic = recipe['magnetization_mode_groups']
    illumination = recipe['illumination_mode_groups']
    if magnetic is None and illumination is None:
        return None
    selected = sorted(mode for group in (magnetic if magnetic is not None else illumination) for mode in group)
    magnetic = [[mode] for mode in selected] if magnetic is None else [list(group) for group in magnetic]
    illumination = [[mode] for mode in selected] if illumination is None else [list(group) for group in illumination]
    for groups in (magnetic, illumination):
        flattened = [mode for group in groups for mode in group]
        if (any(not group for group in groups) or sorted(flattened) != selected
                or len(set(flattened)) != len(flattened) or not selected
                or selected[0] != 0 or selected[-1] >= n_modes):
            raise ValueError('Mode partitions must cover the same distinct valid modes, including mode 0.')
    return magnetic, illumination


def _material_log_objects(thickness, charge, magnetic, magnetization, metadata):
    """Shared forward model: t * (q_charge(E) + p*q_magnetic(E)*mz(state))."""
    energies, states = metadata['energy_indices'], metadata['state_indices']
    return thickness * (charge[energies, None, None]
                        + metadata['polarizations'][:, None, None] * magnetic[energies, None, None]
                        * magnetization[states])


def _expand_spatial_components(value, aperture):
    """Restore packed pixel-by-one maps to their full spatial grid."""
    if isinstance(value, dict):
        return {key: _expand_spatial_components(item, aperture) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[-2:] == (int(aperture.sum()), 1):
        full = np.zeros(value.shape[:-2] + aperture.shape, dtype=value.dtype)
        full[..., aperture] = value[..., 0]
        return full
    return value


def _snap_magnetization(magnetization, values):
    """Nearest allowed value, preferring smaller magnitude then negative ties."""
    nearest = np.full_like(magnetization, values[0])
    distance = np.abs(magnetization - values[0])
    for value in values[1:]:
        candidate_distance = np.abs(magnetization - value)
        closer = candidate_distance < distance
        nearest = np.where(closer, value, nearest)
        distance = np.minimum(distance, candidate_distance)
    return nearest


def _constrain_magnetization_trend(
    maps,
    active,
    names,
    saturated,
    trend,
    clip=True,
    groups=None,
    discrete_values=None,
    preferences=None,
    spatial_weights=None,
    state_weights=None,
    correction="offset",
    median_bounds=False,
    clip_range=(-1.0, 1.0),
    median_scope="whole",
):
    """Project state means onto a monotone sequence, preserving fixed anchors."""
    if (trend == "none" and not median_bounds) or not np.any(active):
        return
    if correction not in ("offset", "factor", "affine_domains"):
        raise ValueError("magnetization_trend_correction must be 'offset', 'factor', or 'affine_domains'.")
    spatial_weights = (
        np.ones(active.shape)
        if spatial_weights is None
        else np.asarray(spatial_weights, float)
    )
    if (
        spatial_weights.shape != active.shape
        or np.any(~np.isfinite(spatial_weights))
        or np.any(spatial_weights < 0)
    ):
        raise ValueError("Trend spatial weights must be finite nonnegative and match the maps")
    active = active & (spatial_weights > 0)
    if not np.any(active):
        return
    pixel_weights = spatial_weights[active]

    state_weights = (
        np.ones(len(names))
        if state_weights is None
        else np.asarray(state_weights, float)
    )
    if (
        state_weights.shape != (len(names),)
        or np.any(~np.isfinite(state_weights))
        or np.any(state_weights <= 0)
    ):
        raise ValueError("Trend state weights must be finite and positive")
    if discrete_values is not None and (
        correction != "offset" or not np.allclose(pixel_weights, pixel_weights[0])
    ):
        raise ValueError("Discrete trend correction requires offset mode and uniform retained-pixel weights")
    if median_bounds and discrete_values is not None and np.any(abs(discrete_values) > 1):
        raise ValueError("Median bounds require discrete magnetization values within [-1, 1]")
    clip_low, clip_high = clip_range

    def median_valid(values):
        selections = [np.ones(values.shape, bool)] if median_scope == "whole" else [values < 0, values > 0]
        return all(
            abs(_weighted_magnetization_median(values[mask], pixel_weights[mask])) <= 1. + 1e-12
            for mask in selections
            if np.any(mask)
        )

    if median_scope not in ("whole", "domains"):
        raise ValueError("magnetization_median_scope must be 'whole' or 'domains'.")
    if median_bounds and median_scope == "domains" and discrete_values is None:
        for index, name in enumerate(names):
            if name in saturated:
                continue
            values = maps[index][active].copy()
            for mask in (values < 0, values > 0):
                if np.any(mask):
                    median = _weighted_magnetization_median(values[mask], pixel_weights[mask])
                    if abs(median) > 1.:
                        values[mask] /= abs(median)
            maps[index][active] = values
    direction = -1. if trend == "decreasing" else 1.
    directed_low, directed_high = sorted((direction * clip_low, direction * clip_high))
    levels = None if discrete_values is None else np.sort(direction * discrete_values)
    if levels is not None and len(levels) > 2 and not np.allclose(
        np.diff(levels), levels[1] - levels[0], rtol=1e-12, atol=1e-14
    ):
        raise ValueError("magnetization_trend with discrete magnetization requires equally spaced values.")
    sequences = [list(range(len(names)))] if groups is None else groups
    for indices in sequences:
        means = direction * np.array([np.average(maps[index][active], weights=pixel_weights) for index in indices])
        anchors = {i: direction * saturated[names[index]] for i, index in enumerate(indices) if names[index] in saturated}
        positions = sorted(anchors) if trend != "none" else []
        if any(anchors[a] > anchors[b] for a, b in zip(positions, positions[1:])):
            raise ValueError("magnetization_trend contradicts saturated_states in state order.")
        targets = means.copy()
        medians = direction * np.array([
            _weighted_magnetization_median(maps[index][active], pixel_weights)
            for index in indices
        ])
        lower = means - 1. - medians if median_bounds and levels is None else np.full(len(indices), -np.inf)
        upper = means + 1. - medians if median_bounds and levels is None else np.full(len(indices), np.inf)
        if median_bounds and median_scope == "domains" and levels is None:
            lower = np.full(len(indices), -1. if trend != "none" else -np.inf)
            upper = np.full(len(indices), 1. if trend != "none" else np.inf)
        if clip:
            lower = np.maximum(lower, directed_low)
            upper = np.minimum(upper, directed_high)
        boundaries = [-1] + positions + [len(indices)]
        for left, right in zip(boundaries, boundaries[1:]):
            if levels is not None and right > left + 1 and (
                anchors.get(left, -np.inf) > levels[-1]
                or anchors.get(right, np.inf) < levels[0]
            ):
                raise ValueError("magnetization_trend anchors are incompatible with binary_magnetization_values.")
            if right <= left + 1:
                continue
            targets[left + 1:right] = np.clip(targets[left + 1:right], lower[left + 1:right], upper[left + 1:right])
        domain_centers = {}
        if correction == "affine_domains":
            for index in indices:
                centers = _two_domain_distribution(maps[index][active], pixel_weights)
                if centers is not None:
                    domain_centers[index] = centers
        reference_gap = (
            np.median([centers[1] - centers[0] for centers in domain_centers.values()])
            if domain_centers else None
        )
        for i, index in enumerate(indices):
            if i in anchors:
                continue
            values = maps[index][active]
            target = direction * targets[i]
            if levels is not None:
                if len(levels) == 1:
                    continue
                spacing = levels[1] - levels[0]
                ranks = np.rint((direction * values - levels[0]) / spacing).astype(int)
                target_count = int(np.rint((targets[i] - levels[0]) * len(values) / spacing))
                target_count = np.clip(target_count, 0, (len(levels) - 1) * len(values))
                delta = int(target_count - ranks.sum())
                preferred = direction * (preferences.get(index, values) if preferences is not None else values)
                while delta:
                    step = 1 if delta > 0 else -1
                    candidates = np.flatnonzero(ranks < len(levels) - 1 if step > 0 else ranks > 0)
                    current = levels[ranks[candidates]]
                    updated = levels[ranks[candidates] + step]
                    cost = (updated - preferred[candidates]) ** 2 - (current - preferred[candidates]) ** 2
                    count = min(abs(delta), len(candidates))
                    chosen = candidates[np.argsort(cost, kind="stable")[:count]]
                    ranks[chosen] += step
                    delta -= step * count
                maps[index][active] = direction * levels[ranks]
                continue
            current_mean = np.average(values, weights=pixel_weights)
            if abs(target - current_mean) < 1e-14:
                continue
            if correction == "affine_domains" and index in domain_centers:
                centers = domain_centers[index]
                factor = reference_gap / (centers[1] - centers[0]) if reference_gap is not None else 1.0
                offset = target - factor * current_mean
                adjusted = factor * values + offset
                adjusted_centers = factor * centers + offset
                if (
                    (not median_bounds or median_valid(adjusted))
                    and adjusted_centers[0] < 0 < adjusted_centers[1]
                    and (not clip or np.all((adjusted >= clip_low) & (adjusted <= clip_high)))
                ):
                    maps[index][active] = adjusted
                    continue
            if correction == "factor" and abs(current_mean) > 1e-14:
                factor = target / current_mean
                scaled = values * factor
                if (
                    factor >= 0
                    and (not median_bounds or median_valid(scaled))
                    and (not clip or np.all((scaled >= clip_low) & (scaled <= clip_high)))
                ):
                    maps[index][active] = scaled
                    continue
            if clip:
                lo, hi = clip_low - values.max(), clip_high - values.min()
                for _ in range(60):
                    shift = (lo + hi) / 2
                    if np.average(np.clip(values + shift, clip_low, clip_high), weights=pixel_weights) < target:
                        lo = shift
                    else:
                        hi = shift
                maps[index][active] = np.clip(values + (lo + hi) / 2, clip_low, clip_high)
            else:
                maps[index][active] = values + target - current_mean
            if median_bounds and median_scope == "domains" and not median_valid(maps[index][active]):
                adjusted = maps[index][active].copy()
                if abs(target) > 1. + 1e-12:
                    raise ValueError("Domain median bounds require a feasible trend target within [-1, 1].")
                lo, hi = 0.0, 1.0
                for _ in range(60):
                    factor = (lo + hi) / 2
                    candidate = target + factor * (adjusted - target)
                    if median_valid(candidate):
                        lo = factor
                    else:
                        hi = factor
                maps[index][active] = target + lo * (adjusted - target)

    if median_bounds and median_scope == "domains":
        if any(not median_valid(maps[index][active]) for index in range(len(names))):
            raise ValueError("Domain median bounds and trend/clipping cannot be satisfied by the selected correction; revise the constraints.")


def _weighted_magnetization_median(values, weights):
    """Interpolate the weighted CDF at its midpoint; equal weights give np.median."""
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    centers = (np.cumsum(weights) - .5 * weights) / np.sum(weights)
    return float(np.interp(.5, centers, values))


def _two_domain_distribution(values, weights):
    """Conservative two-population estimate, not an absolute calibration."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if len(values) < 32 or not np.all(np.isfinite(values)):
        return None
    centers = np.quantile(values, [.1, .9])
    if centers[0] == centers[1]:
        return None
    for _ in range(40):
        membership = values > centers.mean()
        if not np.any(membership) or np.all(membership):
            return None
        updated = np.array([
            np.average(values[~membership], weights=weights[~membership]),
            np.average(values[membership], weights=weights[membership]),
        ])
        if np.allclose(updated, centers, rtol=0, atol=1e-12):
            centers = updated
            break
        centers = updated
    membership = values > centers.mean()
    masses = np.array([weights[~membership].sum(), weights[membership].sum()])
    if masses.min() < .1 * masses.sum() or not centers[0] < 0 < centers[1]:
        return None
    deviations = [
        np.sqrt(np.average((values[selection] - center) ** 2, weights=weights[selection]))
        for selection, center in zip([~membership, membership], centers)
    ]
    gap = centers[1] - centers[0]
    if gap <= 4 * max(deviations):
        return None
    valley = abs(values - centers.mean()) < .1 * gap
    if weights[valley].sum() > .05 * weights.sum():
        return None
    return centers
