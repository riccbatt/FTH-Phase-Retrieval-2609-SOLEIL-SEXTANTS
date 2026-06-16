"""
General joint retrieval across states, energies, polarizations, and beams.

Observation ``a`` is modeled in log-object space as

    L_a(r) = C_beam(a)(r) + p_a R_state(a),energy(a)(r),

where ``C_b`` changes with beam condition, ``p_a`` is the known polarization
coefficient, and ``R_je = i * wave_number_e * thickness * n_je`` is shared
across beam conditions. The weighted linear fit reports whether the supplied
measurement geometry uniquely separates beam and material components.
"""

import time

import numpy as np
from scipy import stats

try:
    from . import phase_retrieval_core_multienergy as core
except ImportError:
    import phase_retrieval_core_multienergy as core


PhaseRtrv_core = core.PhaseRtrv_core


def default_general_phase_retrieval_recipe():
    """Return defaults for joint state/energy/polarization/beam retrieval."""
    return {
        # Phase-retrieval stages applied independently to every observation.
        "inner_mode": ["HAPRE"],
        "inner_Nit": [1],
        "outer_iterations": 300,
        "warmup_mode": ["HAPRE"],
        "warmup_Nit": [20],
        "shuffle_observations": True,
        "random_seed": None,
        "beta_zero": 0.5,
        "beta_mode": "arctan",
        "alpha_zero": 0.0,
        "alpha_mode": "const",
        "TV_freq": 1e9,
        "warmup_beta_zero": None,
        "warmup_beta_mode": None,
        "warmup_alpha_zero": None,
        "warmup_alpha_mode": None,
        "warmup_TV_freq": None,
        "plot_every": 1e9,
        "average_img": 1,
        "Fourier_last": True,
        "final_fourier_constraint": True,
        "hologram_intensity_cutoff_vmin": -1,
        # General log-object projection settings.
        "projection_model": "physical_factorized",
        "projection_every": 1,
        "projection_start": 0,
        "projection_relaxation": 1.0,
        "observation_weights": None,
        "rank_deficient": "error",
        # Physical factorization L = C_m + q_c(E) + p*q_m(E)*mz_s.
        "physical_iterations": 20,
        "saturated_states": None,
        "zero_magnetization_outside_support": False,
        "charge_spectral_constraint": "free",
        "magnetic_spectral_constraint": "free",
        "energy_values": None,
        "known_charge_beta_spectrum": None,
        "known_charge_delta_spectrum": None,
        "known_magnetic_beta_spectrum": None,
        "known_magnetic_delta_spectrum": None,
        "charge_absorption_part": "real",
        "magnetic_absorption_part": "real",
        "charge_response_real_range": None,
        "charge_response_imag_range": None,
        "magnetic_response_real_range": None,
        "magnetic_response_imag_range": None,
        "kk_sign": 1.0,
        "kk_subtract_baseline": True,
        "kk_normalize_input": False,
        "known_spectrum_normalization": "none",
        "fit_known_spectrum_scale": True,
        "fit_known_spectrum_offset": True,
        "log_floor": 1e-12,
    }


def _ordered_unique(values):
    """Return unique hashable values in first-occurrence order."""
    unique = []
    lookup = {}
    indices = np.empty(len(values), dtype=int)
    for index, value in enumerate(values):
        try:
            value_index = lookup[value]
        except KeyError:
            lookup[value] = len(unique)
            unique.append(value)
            value_index = lookup[value]
        except TypeError as exc:
            raise ValueError("Metadata labels must be hashable.") from exc
        indices[index] = value_index
    return unique, indices


def _normalize_metadata(
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    n_observations,
):
    """Validate and encode all observation metadata used by the model."""
    states = np.asarray(state_labels)
    energies = np.asarray(energy_labels)
    polarizations = np.asarray(polarization_coefficients, dtype=float)
    beams = np.asarray(beam_labels)
    expected_shape = (n_observations,)
    if states.shape != expected_shape:
        raise ValueError("state_labels must have shape (n_observations,).")
    if energies.shape != expected_shape:
        raise ValueError("energy_labels must have shape (n_observations,).")
    if polarizations.shape != expected_shape:
        raise ValueError(
            "polarization_coefficients must have shape (n_observations,)."
        )
    if beams.shape != expected_shape:
        raise ValueError("beam_labels must have shape (n_observations,).")
    if np.any(~np.isfinite(polarizations)) or np.any(polarizations == 0):
        raise ValueError(
            "polarization_coefficients must be finite and nonzero."
        )

    state_names, state_indices = _ordered_unique(states.tolist())
    energy_names, energy_indices = _ordered_unique(energies.tolist())
    beam_names, beam_indices = _ordered_unique(beams.tolist())
    response_names, response_indices = _ordered_unique(
        list(zip(states.tolist(), energies.tolist()))
    )
    return {
        "states": states,
        "energies": energies,
        "polarizations": polarizations,
        "beams": beams,
        "state_names": state_names,
        "state_indices": state_indices,
        "energy_names": energy_names,
        "energy_indices": energy_indices,
        "beam_names": beam_names,
        "beam_indices": beam_indices,
        "response_names": response_names,
        "response_indices": response_indices,
    }


def _observation_weights(weights, n_observations):
    """Return validated positive weights, defaulting to equal weighting."""
    if weights is None:
        return np.ones(n_observations, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_observations,):
        raise ValueError(
            "observation_weights must have shape (n_observations,)."
        )
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("observation_weights must be finite and positive.")
    return weights


def _normalize_saturated_states(saturated_states, state_names):
    """Return validated ``{state: +1/-1}`` saturated-state metadata."""
    if saturated_states is None:
        return {}
    if isinstance(saturated_states, dict):
        saturated = dict(saturated_states)
    elif isinstance(saturated_states, (list, tuple, set, np.ndarray)):
        saturated = {state: 1.0 for state in list(saturated_states)}
    else:
        raise ValueError(
            "saturated_states must be a sequence or a state-to-sign mapping."
        )
    unknown = set(saturated) - set(state_names)
    if unknown:
        raise ValueError(
            "saturated_states contains unknown states: "
            f"{sorted(unknown, key=str)}"
        )
    for state, value in saturated.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or abs(float(value)) != 1.0
        ):
            raise ValueError(
                f"Saturation value for state {state!r} must be +1 or -1."
            )
        saturated[state] = float(value)
    return saturated


def _normalize_range(value, name):
    """Validate an optional finite ``(minimum, maximum)`` interval."""
    if value is None:
        return None
    values = np.asarray(value, dtype=float)
    if values.shape != (2,) or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be a finite (minimum, maximum) pair.")
    minimum, maximum = map(float, values)
    if minimum > maximum:
        raise ValueError(f"{name} minimum must not exceed its maximum.")
    return minimum, maximum


def _constrain_response_values(response, real_range=None, imag_range=None):
    """Clip real and imaginary response components to optional intervals."""
    constrained = np.asarray(response, dtype=np.complex128).copy()
    real_range = _normalize_range(real_range, "response_real_range")
    imag_range = _normalize_range(imag_range, "response_imag_range")
    if real_range is not None:
        constrained.real = np.clip(constrained.real, *real_range)
    if imag_range is not None:
        constrained.imag = np.clip(constrained.imag, *imag_range)
    return constrained


def _apply_spectral_constraint(response, kind, recipe):
    """Apply the selected charge or magnetic energy-spectrum constraint."""
    prefix = f"{kind}_"
    return core.constrain_complex_spectrum(
        response,
        spectral_constraint=recipe[f"{prefix}spectral_constraint"],
        energy_values=recipe["energy_values"],
        known_beta_spectrum=recipe[f"known_{kind}_beta_spectrum"],
        known_delta_spectrum=recipe[f"known_{kind}_delta_spectrum"],
        absorption_part=recipe[f"{prefix}absorption_part"],
        kk_sign=recipe["kk_sign"],
        kk_subtract_baseline=recipe["kk_subtract_baseline"],
        kk_normalize_input=recipe["kk_normalize_input"],
        known_beta_normalization=recipe["known_spectrum_normalization"],
        fit_known_beta_scale=recipe["fit_known_spectrum_scale"],
        fit_known_beta_offset=recipe["fit_known_spectrum_offset"],
    )


def project_log_objects_general(
    log_objects,
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    weights=None,
    relaxation=1.0,
    rank_deficient="error",
    return_components=False,
):
    """
    Fit ``L_a = C_beam(a) + p_a R_state(a),energy(a)``.

    The design is identifiable only when the observation rows separate every
    requested beam component and state/energy response. By default a
    rank-deficient geometry is rejected instead of choosing an arbitrary gauge.
    """
    log_objects = core._as_energy_stack(log_objects, name="log_objects")
    if not (0 <= relaxation <= 1):
        raise ValueError("relaxation must be between 0 and 1.")
    if rank_deficient not in {"error", "minimum_norm"}:
        raise ValueError(
            "rank_deficient must be 'error' or 'minimum_norm'."
        )

    n_observations, nx, ny = log_objects.shape
    metadata = _normalize_metadata(
        state_labels,
        energy_labels,
        polarization_coefficients,
        beam_labels,
        n_observations,
    )
    weights = _observation_weights(weights, n_observations)

    # Build one explicit linear model for beam fields and material responses.
    n_beams = len(metadata["beam_names"])
    n_responses = len(metadata["response_names"])
    design = np.zeros(
        (n_observations, n_beams + n_responses),
        dtype=float,
    )
    rows = np.arange(n_observations)
    design[rows, metadata["beam_indices"]] = 1.0
    design[
        rows,
        n_beams + metadata["response_indices"],
    ] = metadata["polarizations"]

    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    design_rank = np.linalg.matrix_rank(weighted_design)
    n_components = design.shape[1]
    if design_rank < n_components and rank_deficient == "error":
        raise ValueError(
            "The general state/energy/polarization/beam design is rank "
            "deficient. Add observations that connect beam conditions to "
            "shared state-energy responses, or add opposite-polarization "
            "measurements. Use rank_deficient='minimum_norm' only if an "
            "arbitrary pseudoinverse solution is acceptable."
        )

    data = log_objects.reshape(n_observations, -1)
    coefficients = np.linalg.pinv(weighted_design) @ (
        data * sqrt_weights[:, None]
    )
    fitted_data = design @ coefficients
    fitted = fitted_data.reshape(n_observations, nx, ny)
    projected = (
        fitted
        if relaxation == 1
        else (1 - relaxation) * log_objects + relaxation * fitted
    )

    if not return_components:
        return projected

    common_log_objects = coefficients[:n_beams].reshape(n_beams, nx, ny)
    response_log_objects = coefficients[n_beams:].reshape(
        n_responses,
        nx,
        ny,
    )
    response_energy_lookup = {
        label: index
        for index, label in enumerate(metadata["energy_names"])
    }
    response_energy_indices = np.asarray(
        [
            response_energy_lookup[energy]
            for _, energy in metadata["response_names"]
        ],
        dtype=int,
    )
    residual = data - fitted_data
    components = {
        "projection_model": "state_energy_beam",
        "common_log_objects": common_log_objects,
        "common_log_objects_by_beam": {
            label: common_log_objects[index]
            for index, label in enumerate(metadata["beam_names"])
        },
        "common_exit_waves": np.exp(common_log_objects),
        "response_log_objects": response_log_objects,
        "response_log_objects_by_state_energy": {
            pair: response_log_objects[index]
            for index, pair in enumerate(metadata["response_names"])
        },
        "state_names": metadata["state_names"],
        "energy_names": metadata["energy_names"],
        "beam_names": metadata["beam_names"],
        "response_names": metadata["response_names"],
        "response_energy_indices": response_energy_indices,
        "state_indices": metadata["state_indices"],
        "energy_indices": metadata["energy_indices"],
        "beam_indices": metadata["beam_indices"],
        "response_indices": metadata["response_indices"],
        "polarization_coefficients": metadata["polarizations"],
        "design_matrix": design,
        "design_rank": design_rank,
        "design_condition_number": float(np.linalg.cond(weighted_design)),
        "identifiable": design_rank == n_components,
        "fit_residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
    }
    return projected, components


def project_log_objects_physical(
    log_objects,
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    weights=None,
    relaxation=1.0,
    saturated_states=None,
    iterations=20,
    recipe=None,
    magnetization_supportmask=None,
    return_components=False,
):
    """
    Fit ``L = C_beam + q_charge(E) + p*q_magnetic(E)*mz_state``.

    ``mz_state`` is real and clipped to ``[-1, 1]``. Saturated states, when
    supplied, are fixed to +1 or -1. Charge and magnetic response spectra can
    be constrained through the same free/KK/known-beta options used by the
    multi-energy library, followed by optional rectangular value bounds.
    """
    log_objects = core._as_energy_stack(log_objects, name="log_objects")
    if not (0 <= relaxation <= 1):
        raise ValueError("relaxation must be between 0 and 1.")
    if isinstance(iterations, bool) or not isinstance(
        iterations,
        (int, np.integer),
    ) or iterations <= 0:
        raise ValueError("physical_iterations must be a positive integer.")
    recipe = default_general_phase_retrieval_recipe() if recipe is None else recipe

    n_observations, nx, ny = log_objects.shape
    metadata = _normalize_metadata(
        state_labels,
        energy_labels,
        polarization_coefficients,
        beam_labels,
        n_observations,
    )
    weights = _observation_weights(weights, n_observations)
    saturated = _normalize_saturated_states(
        saturated_states,
        metadata["state_names"],
    )
    n_states = len(metadata["state_names"])
    n_energies = len(metadata["energy_names"])
    n_beams = len(metadata["beam_names"])
    if magnetization_supportmask is not None:
        magnetization_supportmask = np.asarray(magnetization_supportmask) != 0
        if magnetization_supportmask.shape != (nx, ny):
            raise ValueError(
                "magnetization_supportmask must have shape (nx, ny)."
            )

    # Initialize beam fields from weighted observation means.
    common = np.zeros((n_beams, nx, ny), dtype=np.complex128)
    beam_weight = np.zeros(n_beams, dtype=float)
    for observation in range(n_observations):
        beam = metadata["beam_indices"][observation]
        common[beam] += weights[observation] * log_objects[observation]
        beam_weight[beam] += weights[observation]
    common /= beam_weight[:, None, None]

    charge = np.zeros(n_energies, dtype=np.complex128)
    magnetic = np.ones(n_energies, dtype=np.complex128)
    magnetization = np.zeros((n_states, nx, ny), dtype=float)
    for state, sign in saturated.items():
        state_index = metadata["state_names"].index(state)
        magnetization[state_index] = sign
    if magnetization_supportmask is not None:
        magnetization *= magnetization_supportmask

    # Estimate an initial complex magnetic direction independently per energy.
    for energy in range(n_energies):
        selection = metadata["energy_indices"] == energy
        residuals = []
        for observation in np.flatnonzero(selection):
            beam = metadata["beam_indices"][observation]
            polarization = metadata["polarizations"][observation]
            residuals.append(
                (log_objects[observation] - common[beam]) / polarization
            )
        if residuals:
            residuals = np.asarray(residuals)
            moment = np.sum(residuals**2)
            if np.isfinite(moment) and abs(moment) > 1e-30:
                direction = np.exp(0.5j * np.angle(moment))
                amplitude = np.max(
                    np.abs(
                        np.real(np.conj(direction) * residuals)
                    )
                )
                magnetic[energy] = direction * max(amplitude, 1e-12)

    charge_spectral_info = {}
    magnetic_spectral_info = {}
    for _ in range(int(iterations)):
        # Fit one common complex field for every beam condition.
        common.fill(0.0)
        beam_weight.fill(0.0)
        for observation in range(n_observations):
            beam = metadata["beam_indices"][observation]
            energy = metadata["energy_indices"][observation]
            state = metadata["state_indices"][observation]
            polarization = metadata["polarizations"][observation]
            modeled_material = (
                charge[energy]
                + polarization
                * magnetic[energy]
                * magnetization[state]
            )
            common[beam] += weights[observation] * (
                log_objects[observation] - modeled_material
            )
            beam_weight[beam] += weights[observation]
        common /= beam_weight[:, None, None]

        # Fit real reduced-magnetization maps while enforcing |mz| <= 1.
        for state, state_name in enumerate(metadata["state_names"]):
            if state_name in saturated:
                magnetization[state].fill(saturated[state_name])
                if magnetization_supportmask is not None:
                    magnetization[state] *= magnetization_supportmask
                continue
            numerator = np.zeros((nx, ny), dtype=float)
            denominator = 0.0
            for observation in np.flatnonzero(
                metadata["state_indices"] == state
            ):
                beam = metadata["beam_indices"][observation]
                energy = metadata["energy_indices"][observation]
                coefficient = (
                    metadata["polarizations"][observation]
                    * magnetic[energy]
                )
                residual = (
                    log_objects[observation]
                    - common[beam]
                    - charge[energy]
                )
                numerator += weights[observation] * np.real(
                    np.conj(coefficient) * residual
                )
                denominator += weights[observation] * abs(coefficient) ** 2
            if denominator > 1e-30:
                magnetization[state] = np.clip(
                    numerator / denominator,
                    -1.0,
                    1.0,
                )
                if magnetization_supportmask is not None:
                    magnetization[state] *= magnetization_supportmask

        # Fit scalar charge and magnetic responses independently at each energy.
        for energy in range(n_energies):
            observations = np.flatnonzero(
                metadata["energy_indices"] == energy
            )
            rows = []
            values = []
            row_weights = []
            for observation in observations:
                beam = metadata["beam_indices"][observation]
                state = metadata["state_indices"][observation]
                coefficient = (
                    metadata["polarizations"][observation]
                    * magnetization[state].ravel()
                )
                rows.append(
                    np.column_stack(
                        [
                            np.ones(coefficient.size),
                            coefficient,
                        ]
                    )
                )
                values.append(
                    (
                        log_objects[observation] - common[beam]
                    ).ravel()
                )
                row_weights.append(
                    np.full(coefficient.size, weights[observation])
                )
            design = np.concatenate(rows, axis=0)
            target = np.concatenate(values)
            sqrt_weight = np.sqrt(np.concatenate(row_weights))
            coefficients = np.linalg.pinv(
                design * sqrt_weight[:, None]
            ) @ (target * sqrt_weight)
            charge[energy], magnetic[energy] = coefficients

        # Fix the charge/common-field offset gauge before applying priors.
        charge_offset = np.mean(charge)
        charge -= charge_offset
        common += charge_offset

        charge, charge_spectral_info = _apply_spectral_constraint(
            charge,
            "charge",
            recipe,
        )
        magnetic, magnetic_spectral_info = _apply_spectral_constraint(
            magnetic,
            "magnetic",
            recipe,
        )
        charge = _constrain_response_values(
            charge,
            recipe["charge_response_real_range"],
            recipe["charge_response_imag_range"],
        )
        magnetic = _constrain_response_values(
            magnetic,
            recipe["magnetic_response_real_range"],
            recipe["magnetic_response_imag_range"],
        )

    fitted = np.stack([
        common[metadata["beam_indices"][observation]]
        + charge[metadata["energy_indices"][observation]]
        + metadata["polarizations"][observation]
        * magnetic[metadata["energy_indices"][observation]]
        * magnetization[metadata["state_indices"][observation]]
        for observation in range(n_observations)
    ])
    projected = (
        fitted
        if relaxation == 1
        else (1 - relaxation) * log_objects + relaxation * fitted
    )
    if not return_components:
        return projected

    residual = log_objects - fitted
    charge_mode = str(recipe["charge_spectral_constraint"]).lower()
    magnetic_mode = str(recipe["magnetic_spectral_constraint"]).lower()
    known_modes = {"known_beta", "known-beta", "known_beta_kk", "known-beta-kk"}
    charge_absolute_anchored = (
        charge_mode in known_modes
        and not recipe["fit_known_spectrum_scale"]
        and not recipe["fit_known_spectrum_offset"]
    )
    magnetic_scale_anchored = (
        bool(saturated)
        or (
            magnetic_mode in known_modes
            and not recipe["fit_known_spectrum_scale"]
        )
    )
    components = {
        "projection_model": "physical_factorized",
        "common_log_objects": common,
        "common_log_objects_by_beam": {
            name: common[index]
            for index, name in enumerate(metadata["beam_names"])
        },
        "common_exit_waves": np.exp(common),
        "charge_response": charge,
        "magnetic_response": magnetic,
        "magnetization": magnetization,
        "magnetization_by_state": {
            name: magnetization[index]
            for index, name in enumerate(metadata["state_names"])
        },
        "magnetization_supportmask_applied": (
            magnetization_supportmask is not None
        ),
        "saturated_states": saturated,
        "state_names": metadata["state_names"],
        "energy_names": metadata["energy_names"],
        "beam_names": metadata["beam_names"],
        "charge_gauge": "zero_mean_before_optional_constraints",
        "magnetic_scale_anchored": magnetic_scale_anchored,
        "charge_absolute_gauge_anchored": charge_absolute_anchored,
        "identifiable": (
            magnetic_scale_anchored and charge_absolute_anchored
        ),
        "fit_residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        "charge_spectral_info": charge_spectral_info,
        "magnetic_spectral_info": magnetic_spectral_info,
        "magnetization_bounds": (-1.0, 1.0),
    }
    return projected, components


def project_fourier_fields_general(
    fields,
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    weights=None,
    relaxation=1.0,
    rank_deficient="error",
    projection_model="physical_factorized",
    saturated_states=None,
    physical_iterations=20,
    recipe=None,
    log_floor=1e-12,
    magnetization_supportmask=None,
    return_components=False,
):
    """Apply the selected general model to Fourier-domain fields."""
    log_objects = core.fourier_field_to_object_log(
        fields,
        log_floor=log_floor,
        unwrap_energy_phase=False,
    )
    if projection_model == "physical_factorized":
        projected = project_log_objects_physical(
            log_objects,
            state_labels,
            energy_labels,
            polarization_coefficients,
            beam_labels,
            weights=weights,
            relaxation=relaxation,
            saturated_states=saturated_states,
            iterations=physical_iterations,
            recipe=recipe,
            magnetization_supportmask=magnetization_supportmask,
            return_components=return_components,
        )
    else:
        projected = project_log_objects_general(
            log_objects,
            state_labels,
            energy_labels,
            polarization_coefficients,
            beam_labels,
            weights=weights,
            relaxation=relaxation,
            rank_deficient=rank_deficient,
            return_components=return_components,
        )
    if return_components:
        projected_log_objects, components = projected
        return (
            core.object_log_to_fourier_field(projected_log_objects),
            components,
        )
    return core.object_log_to_fourier_field(projected)


def response_to_refractive_index(
    response_log_objects,
    wave_numbers,
    thickness,
    response_energy_indices,
):
    """
    Convert ``R_je = i*k_e*t*n_je`` to ``n_je`` when ``k_e`` and ``t`` are known.

    This conversion is deliberately separate from phase retrieval because the
    diffraction data determine the product ``i*k_e*t*n_je``, not its factors.
    """
    response = np.asarray(response_log_objects, dtype=np.complex128)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    response_energy_indices = np.asarray(
        response_energy_indices,
        dtype=int,
    )
    if response.ndim != 3:
        raise ValueError(
            "response_log_objects must have shape (n_responses, nx, ny)."
        )
    if response_energy_indices.shape != (response.shape[0],):
        raise ValueError(
            "response_energy_indices must have shape (n_responses,)."
        )
    if wave_numbers.ndim != 1 or np.any(~np.isfinite(wave_numbers)):
        raise ValueError("wave_numbers must be a finite one-dimensional array.")
    if (
        isinstance(thickness, bool)
        or not np.isscalar(thickness)
        or not np.isfinite(thickness)
        or thickness <= 0
    ):
        raise ValueError("thickness must be a finite positive scalar.")
    if (
        np.any(response_energy_indices < 0)
        or np.any(response_energy_indices >= wave_numbers.size)
    ):
        raise ValueError("response_energy_indices contains an invalid index.")
    selected_wave_numbers = wave_numbers[response_energy_indices]
    if np.any(selected_wave_numbers == 0):
        raise ValueError("wave_numbers must be nonzero.")
    return response / (
        1j * selected_wave_numbers[:, None, None] * float(thickness)
    )


def _verify_recipe(recipe, n_observations, n_energies=None):
    """Validate the general retrieval recipe and observation weights."""
    core._build_update_schedule(recipe, name="inner")
    core._build_update_schedule(recipe, name="warmup", allow_disabled=True)
    for key in (
        "outer_iterations",
        "projection_every",
        "average_img",
    ):
        value = recipe[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{key} must be an integer.")
    if recipe["outer_iterations"] <= 0:
        raise ValueError("outer_iterations must be > 0.")
    if recipe["projection_every"] <= 0:
        raise ValueError("projection_every must be > 0.")
    projection_start = recipe["projection_start"]
    if projection_start is not None and (
        isinstance(projection_start, bool)
        or not isinstance(projection_start, (int, np.integer))
        or projection_start < 0
    ):
        raise ValueError("projection_start must be None or a non-negative integer.")
    if recipe["average_img"] <= 0:
        raise ValueError("average_img must be > 0.")
    if recipe["plot_every"] <= 0:
        raise ValueError("plot_every must be > 0.")
    if not isinstance(recipe["Fourier_last"], bool):
        raise ValueError("Fourier_last must be bool.")
    if not isinstance(recipe["final_fourier_constraint"], bool):
        raise ValueError("final_fourier_constraint must be bool.")
    if not (0 <= recipe["projection_relaxation"] <= 1):
        raise ValueError("projection_relaxation must be between 0 and 1.")
    if recipe["projection_model"] not in {
        "physical_factorized",
        "state_energy_beam",
        "none",
    }:
        raise ValueError(
            "projection_model must be 'physical_factorized', "
            "'state_energy_beam', or 'none'."
        )
    if recipe["rank_deficient"] not in {"error", "minimum_norm"}:
        raise ValueError(
            "rank_deficient must be 'error' or 'minimum_norm'."
        )
    physical_iterations = recipe["physical_iterations"]
    if (
        isinstance(physical_iterations, bool)
        or not isinstance(physical_iterations, (int, np.integer))
        or physical_iterations <= 0
    ):
        raise ValueError("physical_iterations must be a positive integer.")
    if not isinstance(recipe["zero_magnetization_outside_support"], bool):
        raise ValueError("zero_magnetization_outside_support must be bool.")
    for key in (
        "charge_response_real_range",
        "charge_response_imag_range",
        "magnetic_response_real_range",
        "magnetic_response_imag_range",
    ):
        _normalize_range(recipe[key], key)
    allowed_spectral = {
        "none",
        "free",
        "unconstrained",
        "kk",
        "kramers-kronig",
        "kramers_kronig",
        "known_beta",
        "known-beta",
        "known_beta_kk",
        "known-beta-kk",
    }
    for kind in ("charge", "magnetic"):
        mode = str(recipe[f"{kind}_spectral_constraint"]).lower()
        if mode not in allowed_spectral:
            raise ValueError(
                f"{kind}_spectral_constraint has an unsupported value."
            )
        if n_energies is not None and mode not in {
            "none",
            "free",
            "unconstrained",
        }:
            _apply_spectral_constraint(
                np.zeros(n_energies, dtype=np.complex128),
                kind,
                recipe,
            )
    _observation_weights(recipe["observation_weights"], n_observations)


def _run_update_schedule(
    field,
    amplitude,
    supportmask,
    bsmask,
    schedule,
    recipe,
):
    """Run the configured phase-retrieval stages for one observation."""
    stage_results = []
    for stage_index, stage in enumerate(schedule):
        iterations = stage["Nit"]
        field, error, support_error, _ = PhaseRtrv_core(
            diffract=amplitude,
            mask=supportmask,
            mode=stage["mode"],
            Nit=iterations,
            beta_zero=stage["beta_zero"],
            beta_mode=stage["beta_mode"],
            alpha_zero=stage["alpha_zero"],
            alpha_mode=stage["alpha_mode"],
            Phase=field,
            seed=False,
            plot_every=recipe["plot_every"],
            bsmask=bsmask,
            real_object=False,
            average_img=min(max(1, recipe["average_img"]), iterations),
            Fourier_last=recipe["Fourier_last"],
            gamma=None,
            RL_freq=iterations + 1,
            RL_it=0,
            TV_freq=stage["TV_freq"],
        )
        stage_results.append({
            "schedule_stage": stage_index,
            **stage,
            "error": np.asarray(error),
            "support_error": np.asarray(support_error),
        })
    return field, stage_results


def _apply_measured_amplitudes(fields, amplitudes, bsmasks):
    """Reapply measured Fourier amplitudes outside invalid-pixel regions."""
    constrained = np.asarray(fields).copy()
    observed = bsmasks == 0
    constrained[observed] = (
        amplitudes[observed] * np.exp(1j * np.angle(constrained[observed]))
    )
    return constrained


def _initialize_fields(
    supportmask,
    amplitudes,
    intensities,
    mask_stack,
):
    """Create and approximately normalize one initial field per observation."""
    n_observations = amplitudes.shape[0]
    start = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(supportmask)))
    fields = np.repeat(start[None], n_observations, axis=0).astype(
        np.complex128
    )
    for observation in range(n_observations):
        valid = (
            (mask_stack[observation] == 0)
            & (intensities[observation] > 0)
        )
        if not np.any(valid):
            continue
        measured = amplitudes[observation][valid].ravel()
        current = np.abs(fields[observation][valid]).ravel()
        if (
            measured.size >= 2
            and np.ptp(measured) > 0
            and np.ptp(current) > 0
        ):
            fit = stats.linregress(measured, current)
            if abs(fit.slope) > 1e-12:
                fields[observation] /= fit.slope
    return fields


def general_phase_retrieval_algorithm(
    holograms,
    mask_pixel,
    supportmask,
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    saturated_states=None,
    general_recipe=None,
    start_fields=None,
):
    """
    Jointly retrieve arbitrary states, energies, polarizations, and beams.

    The algorithm alternates independent Fourier/support updates with the
    projection ``L_a = C_beam(a) + p_a R_state(a),energy(a)``.

    ``energy_labels`` may represent energy or any illumination category
    expected to change the material response. ``beam_labels`` represent
    conditions that change only the common illumination field. Polarization
    coefficients are normally +1 and -1 but may be any finite nonzero values.
    ``saturated_states`` optionally maps state labels to known +1/-1 reduced
    magnetization; a sequence is interpreted as +1 saturation.

    Returns
    -------
    fields : ndarray, shape (n_observations, nx, ny)
        Final Fourier-domain fields.
    components : dict
        Common beam fields, state-energy responses, and design diagnostics.
    bsmasks : ndarray, shape (n_observations, nx, ny)
        Invalid-pixel masks used by the reconstruction.
    errors : dict
        Per-stage errors and projection diagnostics.
    """
    recipe = default_general_phase_retrieval_recipe()
    if general_recipe is not None:
        if not isinstance(general_recipe, dict):
            raise TypeError("general_recipe must be a dictionary.")
        unknown = set(general_recipe) - set(recipe)
        if unknown:
            raise ValueError(
                f"Unknown general recipe key(s): {sorted(unknown)}"
            )
        recipe.update(general_recipe)
    if saturated_states is not None:
        recipe["saturated_states"] = saturated_states

    holograms = core._as_energy_stack(holograms, name="holograms")
    n_observations, nx, ny = holograms.shape
    metadata = _normalize_metadata(
        state_labels,
        energy_labels,
        polarization_coefficients,
        beam_labels,
        n_observations,
    )
    _verify_recipe(
        recipe,
        n_observations,
        n_energies=len(metadata["energy_names"]),
    )

    supportmask = np.asarray(supportmask)
    if supportmask.shape != (nx, ny):
        raise ValueError("supportmask must have shape (nx, ny).")
    # PhaseRtrv_core applies support constraints in the shifted object frame.
    # The log-object projection uses the same frame via fft2(fftshift(field)).
    magnetization_supportmask = (
        np.fft.fftshift(supportmask)
        if recipe["zero_magnetization_outside_support"]
        else None
    )

    # Convert measured intensities to amplitudes and invalid-pixel masks.
    amplitudes, intensities, bsmasks = core._prepare_energy_amplitudes(
        holograms,
        mask_pixel,
        hologram_intensity_cutoff_vmin=recipe[
            "hologram_intensity_cutoff_vmin"
        ],
    )
    mask_stack = core._as_energy_mask(
        mask_pixel,
        nE=n_observations,
        image_shape=(nx, ny),
    )

    # Initialize each observation independently unless fields are supplied.
    if start_fields is None:
        fields = _initialize_fields(
            supportmask,
            amplitudes,
            intensities,
            mask_stack,
        )
    else:
        fields = core._as_energy_stack(
            start_fields,
            name="start_fields",
        ).astype(np.complex128, copy=True)
        if fields.shape != holograms.shape:
            raise ValueError(
                "start_fields must have the same shape as holograms."
            )

    errors = {
        "observation_steps": [],
        "projection_steps": [],
        "settings": recipe.copy(),
    }
    start_time = time.time()

    # Optional independent warmup before coupling observations.
    warmup_schedule = core._build_update_schedule(
        recipe,
        name="warmup",
        allow_disabled=True,
    )
    if warmup_schedule:
        for observation in range(n_observations):
            fields[observation], results = _run_update_schedule(
                fields[observation],
                amplitudes[observation],
                supportmask,
                bsmasks[observation],
                warmup_schedule,
                recipe,
            )
            for result in results:
                errors["observation_steps"].append({
                    "outer": -1,
                    "observation": observation,
                    "state": metadata["states"][observation],
                    "energy": metadata["energies"][observation],
                    "polarization": metadata["polarizations"][observation],
                    "beam": metadata["beams"][observation],
                    "stage": "warmup",
                    **result,
                })

    inner_schedule = core._build_update_schedule(recipe, name="inner")
    rng = np.random.default_rng(recipe["random_seed"])
    components = {"projection_model": recipe["projection_model"]}

    # Alternate independent data updates with the joint physical fit.
    for outer in range(recipe["outer_iterations"]):
        order = (
            rng.permutation(n_observations)
            if recipe["shuffle_observations"]
            else np.arange(n_observations)
        )
        for observation in order:
            fields[observation], results = _run_update_schedule(
                fields[observation],
                amplitudes[observation],
                supportmask,
                bsmasks[observation],
                inner_schedule,
                recipe,
            )
            for result in results:
                errors["observation_steps"].append({
                    "outer": outer,
                    "observation": int(observation),
                    "state": metadata["states"][observation],
                    "energy": metadata["energies"][observation],
                    "polarization": metadata["polarizations"][observation],
                    "beam": metadata["beams"][observation],
                    "stage": "joint",
                    **result,
                })

        projection_start = (
            recipe["projection_every"]
            if recipe["projection_start"] is None
            else recipe["projection_start"]
        )
        projection_due = (
            recipe["projection_model"] != "none"
            and outer >= projection_start
            and ((outer - projection_start) % recipe["projection_every"] == 0)
        )
        if projection_due:
            fields, components = project_fourier_fields_general(
                fields,
                metadata["states"],
                metadata["energies"],
                metadata["polarizations"],
                metadata["beams"],
                weights=recipe["observation_weights"],
                relaxation=recipe["projection_relaxation"],
                rank_deficient=recipe["rank_deficient"],
                projection_model=recipe["projection_model"],
                saturated_states=recipe["saturated_states"],
                physical_iterations=recipe["physical_iterations"],
                recipe=recipe,
                log_floor=recipe["log_floor"],
                magnetization_supportmask=magnetization_supportmask,
                return_components=True,
            )
            errors["projection_steps"].append({
                "outer": outer,
                "fit_residual_rms": components["fit_residual_rms"],
                "projection_model": components["projection_model"],
                "design_rank": components.get("design_rank"),
                "identifiable": components.get("identifiable"),
            })

    # Return components from a final full-strength model decomposition.
    if recipe["projection_model"] != "none":
        fields, components = project_fourier_fields_general(
            fields,
            metadata["states"],
            metadata["energies"],
            metadata["polarizations"],
            metadata["beams"],
            weights=recipe["observation_weights"],
            relaxation=1.0,
            rank_deficient=recipe["rank_deficient"],
            projection_model=recipe["projection_model"],
            saturated_states=recipe["saturated_states"],
            physical_iterations=recipe["physical_iterations"],
            recipe=recipe,
            log_floor=recipe["log_floor"],
            magnetization_supportmask=magnetization_supportmask,
            return_components=True,
        )

    # Optionally finish exactly on the measured Fourier amplitudes.
    if recipe["final_fourier_constraint"]:
        fields = _apply_measured_amplitudes(fields, amplitudes, bsmasks)
        components["final_fourier_constraint_applied"] = True
    else:
        components["final_fourier_constraint_applied"] = False

    errors["runtime_seconds"] = float(np.round(time.time() - start_time, 3))
    return fields, components, bsmasks, errors
