"""
Joint phase retrieval for dichroic measurements of multiple magnetic states.

The real-space exit waves are coupled through the complex log-object model

    L_j(r) = C(r) + p_j M_s(r),

where ``C`` is the common charge log-exit-wave, ``M_s`` is the magnetic
log-exit-wave for state ``s``, and ``p_j`` is the polarization coefficient of
observation ``j`` (normally +1 or -1).

This model supports multiple magnetic states with either polarization. A single
opposite-polarization partner can anchor the common charge component for a
larger collection of single-polarization states, provided the resulting design
matrix has full column rank.

Two projection strengths are available:

``shared_charge``
    Fits an unconstrained complex magnetic log-exit-wave for each state.
``saturated_reference``
    Estimates the complex magnetic response from states flagged as saturated,
    then constrains every state to a real magnetization map.

The saturated-reference model is the physically stronger stabilizing
constraint and does not require prior values of ``delta_m`` or ``beta_m``.
The shared-charge model is useful for exploratory joint reconstruction, but a
single opposite-polarization pair is exactly representable and is therefore
not regularized by that model alone.

The retrieved complex response is proportional to
``k * t * (delta_m + i beta_m)``. Holograms alone do not separate the
refractive-index terms from an unknown thickness ``t``.

Optional ranges for the directly observable products ``k*t*delta_m`` and
``k*t*beta_m`` may constrain this response. They are never required: with both
ranges set to ``None``, the reconstruction remains exclusively data driven.
"""

import time

import numpy as np
from scipy import stats

try:
    from . import phase_retrieval_core_multienergy as core
except ImportError:
    import phase_retrieval_core_multienergy as core


PhaseRtrv_core = core.PhaseRtrv_core


def default_dichroic_phase_retrieval_recipe():
    """Return defaults for joint multi-state dichroic phase retrieval."""
    return {
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
        # 'shared_charge' fits complex M_s maps. 'saturated_reference'
        # estimates their common complex response from saturated states and
        # constrains the remaining factors to real mz maps.
        "projection_model": "shared_charge",
        "projection_every": 1,
        "projection_start": 0,
        "projection_relaxation": 1.0,
        "observation_weights": None,
        "rank_deficient": "error",
        "saturated_states": None,
        "clip_magnetization": False,
        "kt_delta_m_range": None,
        "kt_beta_m_range": None,
        "log_floor": 1e-12,
    }


def _normalize_state_metadata(state_labels, polarization_signs, n_observations):
    """Validate observation metadata and return encoded state information."""
    labels = np.asarray(state_labels)
    if labels.ndim != 1 or labels.shape[0] != n_observations:
        raise ValueError("state_labels must have shape (n_observations,).")

    signs = np.asarray(polarization_signs, dtype=float)
    if signs.shape != (n_observations,):
        raise ValueError(
            "polarization_signs must have shape (n_observations,)."
        )
    if np.any(~np.isfinite(signs)) or np.any(signs == 0):
        raise ValueError("polarization_signs must be finite and nonzero.")

    state_names = []
    state_indices = np.empty(n_observations, dtype=int)
    state_lookup = {}
    for observation, label in enumerate(labels.tolist()):
        try:
            index = state_lookup[label]
        except KeyError:
            state_lookup[label] = len(state_names)
            state_names.append(label)
            index = state_lookup[label]
        except TypeError as exc:
            raise ValueError("state_labels entries must be hashable.") from exc
        state_indices[observation] = index

    return labels, signs, state_names, state_indices


def dichroic_design_matrix(state_labels, polarization_signs):
    """
    Build the design matrix for ``L_j = C + p_j M_state(j)``.

    Returns the matrix and the state names in first-occurrence order.
    """
    labels = np.asarray(state_labels)
    if labels.ndim != 1:
        raise ValueError("state_labels must be one-dimensional.")
    _, signs, state_names, state_indices = _normalize_state_metadata(
        labels,
        polarization_signs,
        labels.size,
    )
    design = np.zeros((labels.size, 1 + len(state_names)), dtype=float)
    design[:, 0] = 1.0
    design[np.arange(labels.size), 1 + state_indices] = signs
    return design, state_names


def _observation_weights(weights, n_observations):
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


def project_log_objects_dichroic(
    log_objects,
    state_labels,
    polarization_signs,
    weights=None,
    relaxation=1.0,
    rank_deficient="error",
    return_components=False,
):
    """
    Project log-exit-waves onto a common charge plus state-magnetic model.

    ``rank_deficient='error'`` rejects data that cannot uniquely separate the
    charge and magnetic terms. ``'minimum_norm'`` permits a Moore-Penrose
    solution, but its charge/magnetic gauge is not physically unique.
    """
    log_objects = core._as_energy_stack(log_objects, name="log_objects")
    if not (0 <= relaxation <= 1):
        raise ValueError("relaxation must be between 0 and 1.")
    if rank_deficient not in {"error", "minimum_norm"}:
        raise ValueError(
            "rank_deficient must be 'error' or 'minimum_norm'."
        )

    n_observations, nx, ny = log_objects.shape
    design, state_names = dichroic_design_matrix(
        state_labels,
        polarization_signs,
    )
    weights = _observation_weights(weights, n_observations)
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    design_rank = np.linalg.matrix_rank(weighted_design)
    n_components = design.shape[1]

    if design_rank < n_components and rank_deficient == "error":
        raise ValueError(
            "The dichroic observation design is rank deficient: charge and "
            "state magnetic components are not uniquely identifiable. Add an "
            "opposite-polarization observation for at least one state, or use "
            "rank_deficient='minimum_norm' only if an arbitrary gauge is "
            "acceptable."
        )

    data = log_objects.reshape(n_observations, -1)
    weighted_data = data * sqrt_weights[:, None]
    coefficients = np.linalg.pinv(weighted_design) @ weighted_data
    fitted = design @ coefficients
    projected = fitted.reshape(n_observations, nx, ny)
    if relaxation < 1:
        projected = (
            (1 - relaxation) * log_objects
            + relaxation * projected
        )

    if not return_components:
        return projected

    magnetic = coefficients[1:].reshape(len(state_names), nx, ny)
    residual = data - fitted
    components = {
        "projection_model": "shared_charge",
        "charge_log_object": coefficients[0].reshape(nx, ny),
        "magnetic_log_objects": magnetic,
        "magnetic_log_objects_by_state": {
            state: magnetic[index]
            for index, state in enumerate(state_names)
        },
        "state_names": state_names,
        "design_matrix": design,
        "design_rank": design_rank,
        "design_condition_number": float(
            np.linalg.cond(weighted_design)
        ),
        "identifiable": design_rank == n_components,
        "fit_residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
    }
    return projected, components


def _normalize_saturated_states(saturated_states, state_names):
    """
    Return ``{state: mz}`` for saturated reference states.

    A sequence of labels marks each listed state as ``mz=+1``. A dictionary
    allows explicit ``+1`` or ``-1`` saturation signs.
    """
    if saturated_states is None:
        return {}
    if isinstance(saturated_states, dict):
        saturated = dict(saturated_states)
    elif isinstance(saturated_states, (list, tuple, set, np.ndarray)):
        saturated = {state: 1.0 for state in list(saturated_states)}
    else:
        raise ValueError(
            "saturated_states must be a sequence of state labels or a "
            "dictionary mapping labels to +1/-1."
        )

    state_set = set(state_names)
    unknown = set(saturated) - state_set
    if unknown:
        raise ValueError(
            "saturated_states contains labels absent from state_labels: "
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


def _normalize_physical_range(value, name, strictly_positive=False):
    """Validate an optional finite ``(minimum, maximum)`` interval."""
    if value is None:
        return None
    values = np.asarray(value, dtype=float)
    if values.shape != (2,) or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be a finite (minimum, maximum) pair.")
    minimum, maximum = map(float, values)
    if minimum > maximum:
        raise ValueError(f"{name} minimum must not exceed its maximum.")
    if strictly_positive and minimum <= 0:
        raise ValueError(f"{name} values must be strictly positive.")
    return minimum, maximum


def _constrain_magnetic_response(
    response,
    kt_delta_m_range=None,
    kt_beta_m_range=None,
):
    """
    Project the inferred response onto optional observable-product intervals.

    The response convention is
    ``imag(q)=k*t*delta_m`` and ``-real(q)=k*t*beta_m``. The two dimensionless
    product bounds are applied independently.
    """
    delta_range = _normalize_physical_range(
        kt_delta_m_range,
        "kt_delta_m_range",
    )
    beta_range = _normalize_physical_range(
        kt_beta_m_range,
        "kt_beta_m_range",
    )
    uses_prior = delta_range is not None or beta_range is not None

    if not uses_prior:
        return np.asarray(response, dtype=np.complex128).copy(), {
            "physical_response_bounds_applied": False,
            "kt_delta_m_range": delta_range,
            "kt_beta_m_range": beta_range,
            "response_bounds": {},
        }
    constrained = np.asarray(response, dtype=np.complex128).copy()
    response_bounds = {}

    if delta_range is not None:
        constrained.imag = np.clip(constrained.imag, *delta_range)
        response_bounds["magnetic_phase_shift"] = delta_range

    if beta_range is not None:
        attenuation = np.clip(-constrained.real, *beta_range)
        constrained.real = -attenuation
        response_bounds["magnetic_log_attenuation"] = beta_range

    return constrained, {
        "physical_response_bounds_applied": True,
        "kt_delta_m_range": delta_range,
        "kt_beta_m_range": beta_range,
        "response_bounds": response_bounds,
    }


def project_log_objects_saturated_reference(
    log_objects,
    state_labels,
    polarization_signs,
    saturated_states,
    weights=None,
    relaxation=1.0,
    rank_deficient="error",
    clip_magnetization=False,
    kt_delta_m_range=None,
    kt_beta_m_range=None,
    return_components=False,
):
    """
    Infer the magnetic response from saturated states and retrieve real ``mz``.

    The first step fits ``L_j = C + p_j M_state(j)``. Saturated states anchor
    the factorization through ``M_sat = q * mz_sat``, where ``mz_sat`` is +1 or
    -1. The estimated complex response ``q(r)`` is then used to project every
    state magnetic term onto a real magnetization map.

    The state/polarization observation design must already separate charge and
    state magnetic terms. A same-polarization domain/reference pair alone is
    rank deficient even when the reference is marked saturated.

    The complex response is inferred from the reconstructed saturated state.
    Optional ``kt_delta_m_range`` and ``kt_beta_m_range`` bounds may constrain
    the directly observable phase-shift and attenuation products. With both
    ranges omitted, no physical prior is applied.
    """
    original_log_objects = core._as_energy_stack(
        log_objects,
        name="log_objects",
    )
    _, shared = project_log_objects_dichroic(
        original_log_objects,
        state_labels,
        polarization_signs,
        weights=weights,
        relaxation=1.0,
        rank_deficient=rank_deficient,
        return_components=True,
    )
    state_names = shared["state_names"]
    saturated = _normalize_saturated_states(saturated_states, state_names)
    if not saturated:
        raise ValueError(
            "projection_model='saturated_reference' requires at least one "
            "state flagged as saturated."
        )

    magnetic_by_state = shared["magnetic_log_objects_by_state"]
    response_candidates = [
        magnetic_by_state[state] / mz
        for state, mz in saturated.items()
    ]
    response_unconstrained = np.mean(response_candidates, axis=0)
    response, physical_info = _constrain_magnetic_response(
        response_unconstrained,
        kt_delta_m_range=kt_delta_m_range,
        kt_beta_m_range=kt_beta_m_range,
    )
    response_power = np.abs(response) ** 2
    valid_response = response_power > 1e-30

    magnetization_by_state = {}
    projected_magnetic_by_state = {}
    for state in state_names:
        if state in saturated:
            magnetization = np.full(
                response.shape,
                saturated[state],
                dtype=float,
            )
        else:
            magnetization = np.zeros(response.shape, dtype=float)
            magnetic = magnetic_by_state[state]
            magnetization[valid_response] = np.real(
                np.conj(response[valid_response])
                * magnetic[valid_response]
            ) / response_power[valid_response]
            if clip_magnetization:
                magnetization = np.clip(magnetization, -1.0, 1.0)
        magnetization_by_state[state] = magnetization
        projected_magnetic_by_state[state] = response * magnetization

    labels = np.asarray(state_labels)
    signs = np.asarray(polarization_signs, dtype=float)
    charge = shared["charge_log_object"]
    projected = np.stack(
        [
            charge + signs[index] * projected_magnetic_by_state[state]
            for index, state in enumerate(labels.tolist())
        ]
    )
    if relaxation < 1:
        projected = (
            (1 - relaxation) * original_log_objects
            + relaxation * projected
        )
    if not return_components:
        return projected

    magnetization = np.stack(
        [magnetization_by_state[state] for state in state_names]
    )
    residual = original_log_objects - projected
    components = {
        "projection_model": "saturated_reference",
        "charge_log_object": charge,
        "magnetic_response": response,
        "magnetic_response_unconstrained": response_unconstrained,
        "magnetic_log_attenuation": -np.real(response),
        "magnetic_phase_shift": np.imag(response),
        "magnetization": magnetization,
        "magnetization_by_state": magnetization_by_state,
        "magnetic_log_objects": response[None] * magnetization,
        "state_names": state_names,
        "saturated_states": saturated,
        "design_matrix": shared["design_matrix"],
        "design_rank": shared["design_rank"],
        "design_condition_number": shared["design_condition_number"],
        "identifiable": shared["identifiable"],
        "fit_residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
    }
    components.update(physical_info)
    return projected, components


def project_fourier_fields_dichroic(
    fields,
    state_labels,
    polarization_signs,
    weights=None,
    relaxation=1.0,
    rank_deficient="error",
    log_floor=1e-12,
    return_components=False,
):
    """Apply the dichroic projection to Fourier-domain reconstruction fields."""
    log_objects = core.fourier_field_to_object_log(
        fields,
        log_floor=log_floor,
        unwrap_energy_phase=False,
    )
    projected = project_log_objects_dichroic(
        log_objects,
        state_labels,
        polarization_signs,
        weights=weights,
        relaxation=relaxation,
        rank_deficient=rank_deficient,
        return_components=return_components,
    )
    if return_components:
        projected_log, components = projected
        return core.object_log_to_fourier_field(projected_log), components
    return core.object_log_to_fourier_field(projected)


def project_fourier_fields_saturated_reference(
    fields,
    state_labels,
    polarization_signs,
    saturated_states,
    weights=None,
    relaxation=1.0,
    rank_deficient="error",
    clip_magnetization=False,
    kt_delta_m_range=None,
    kt_beta_m_range=None,
    log_floor=1e-12,
    return_components=False,
):
    """Infer the magnetic response from saturated reference reconstruction(s)."""
    log_objects = core.fourier_field_to_object_log(
        fields,
        log_floor=log_floor,
        unwrap_energy_phase=False,
    )
    projected = project_log_objects_saturated_reference(
        log_objects,
        state_labels,
        polarization_signs,
        saturated_states,
        weights=weights,
        relaxation=relaxation,
        rank_deficient=rank_deficient,
        clip_magnetization=clip_magnetization,
        kt_delta_m_range=kt_delta_m_range,
        kt_beta_m_range=kt_beta_m_range,
        return_components=return_components,
    )
    if return_components:
        projected_log, components = projected
        return core.object_log_to_fourier_field(projected_log), components
    return core.object_log_to_fourier_field(projected)


def _project_fields(fields, labels, signs, recipe, relaxation):
    model = recipe["projection_model"]
    if model == "shared_charge":
        return project_fourier_fields_dichroic(
            fields,
            labels,
            signs,
            weights=recipe["observation_weights"],
            relaxation=relaxation,
            rank_deficient=recipe["rank_deficient"],
            log_floor=recipe["log_floor"],
            return_components=True,
        )
    if model == "saturated_reference":
        return project_fourier_fields_saturated_reference(
            fields,
            labels,
            signs,
            recipe["saturated_states"],
            weights=recipe["observation_weights"],
            relaxation=relaxation,
            rank_deficient=recipe["rank_deficient"],
            clip_magnetization=recipe["clip_magnetization"],
            kt_delta_m_range=recipe["kt_delta_m_range"],
            kt_beta_m_range=recipe["kt_beta_m_range"],
            log_floor=recipe["log_floor"],
            return_components=True,
        )
    return fields, {"projection_model": "none"}


def _verify_recipe(recipe, n_observations):
    core._build_update_schedule(recipe, name="inner")
    core._build_update_schedule(
        recipe,
        name="warmup",
        allow_disabled=True,
    )

    for key in (
        "outer_iterations",
        "projection_every",
        "projection_start",
        "average_img",
    ):
        value = recipe[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{key} must be an integer.")

    if recipe["outer_iterations"] <= 0:
        raise ValueError("outer_iterations must be > 0.")
    if recipe["projection_every"] <= 0:
        raise ValueError("projection_every must be > 0.")
    if recipe["projection_start"] < 0:
        raise ValueError("projection_start must be >= 0.")
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
        "shared_charge",
        "saturated_reference",
        "none",
    }:
        raise ValueError(
            "projection_model must be 'shared_charge', "
            "'saturated_reference', or 'none'."
        )
    if (
        recipe["projection_model"] == "saturated_reference"
        and recipe["saturated_states"] is None
    ):
        raise ValueError(
            "saturated_states is required for "
            "projection_model='saturated_reference'."
        )
    if not isinstance(recipe["clip_magnetization"], bool):
        raise ValueError("clip_magnetization must be bool.")
    _constrain_magnetic_response(
        np.zeros((1, 1), dtype=np.complex128),
        kt_delta_m_range=recipe["kt_delta_m_range"],
        kt_beta_m_range=recipe["kt_beta_m_range"],
    )
    if recipe["rank_deficient"] not in {"error", "minimum_norm"}:
        raise ValueError(
            "rank_deficient must be 'error' or 'minimum_norm'."
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
    stage_results = []
    for stage_index, stage in enumerate(schedule):
        Nit = stage["Nit"]
        field, err_d, err_s, _ = PhaseRtrv_core(
            diffract=amplitude,
            mask=supportmask,
            mode=stage["mode"],
            Nit=Nit,
            beta_zero=stage["beta_zero"],
            beta_mode=stage["beta_mode"],
            alpha_zero=stage["alpha_zero"],
            alpha_mode=stage["alpha_mode"],
            Phase=field,
            seed=False,
            plot_every=recipe["plot_every"],
            bsmask=bsmask,
            real_object=False,
            average_img=min(max(1, recipe["average_img"]), Nit),
            Fourier_last=recipe["Fourier_last"],
            gamma=None,
            RL_freq=Nit + 1,
            RL_it=0,
            TV_freq=stage["TV_freq"],
        )
        stage_results.append(
            {
                "schedule_stage": stage_index,
                **stage,
                "error": np.asarray(err_d),
                "support_error": np.asarray(err_s),
            }
        )
    return field, stage_results


def _apply_measured_amplitudes(fields, amplitudes, bsmasks):
    constrained = np.asarray(fields).copy()
    observed = bsmasks == 0
    constrained[observed] = (
        amplitudes[observed]
        * np.exp(1j * np.angle(constrained[observed]))
    )
    return constrained


def dichroic_phase_retrieval_algorithm(
    holograms,
    mask_pixel,
    supportmask,
    state_labels,
    polarization_signs,
    saturated_states=None,
    dichroic_recipe=None,
    start_fields=None,
):
    """
    Jointly reconstruct multiple magnetic states and polarization observations.

    Parameters
    ----------
    holograms : array, shape (n_observations, nx, ny)
        Measured diffraction intensities.
    state_labels : sequence, shape (n_observations,)
        State identifier for every hologram. Labels may be strings or numbers.
    polarization_signs : array, shape (n_observations,)
        Magnetic coupling coefficient, normally +1 for one helicity and -1 for
        the opposite helicity. The sign convention is user-defined but must be
        used consistently.
    saturated_states : sequence, dict, or None
        Optional saturated-state metadata. A sequence marks each listed state
        as ``mz=+1``. A dictionary may assign ``+1`` or ``-1`` explicitly.
        Supplying this argument automatically selects the saturated-reference
        projection unless the recipe explicitly chooses another model.
    """
    recipe = default_dichroic_phase_retrieval_recipe()
    projection_model_explicit = False
    if dichroic_recipe is not None:
        if not isinstance(dichroic_recipe, dict):
            raise TypeError("dichroic_recipe must be a dictionary.")
        unknown = set(dichroic_recipe) - set(recipe)
        if unknown:
            raise ValueError(f"Unknown dichroic recipe key(s): {sorted(unknown)}")
        projection_model_explicit = "projection_model" in dichroic_recipe
        recipe.update(dichroic_recipe)
    if saturated_states is not None:
        recipe["saturated_states"] = saturated_states
        if not projection_model_explicit:
            recipe["projection_model"] = "saturated_reference"

    holograms = core._as_energy_stack(holograms, name="holograms")
    n_observations, nx, ny = holograms.shape
    labels, signs, state_names, _ = _normalize_state_metadata(
        state_labels,
        polarization_signs,
        n_observations,
    )
    _verify_recipe(recipe, n_observations)

    supportmask = np.asarray(supportmask)
    if supportmask.shape != (nx, ny):
        raise ValueError("supportmask must have shape (nx, ny).")

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

    if start_fields is None:
        start = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(supportmask)))
        fields = np.repeat(
            start[None],
            n_observations,
            axis=0,
        ).astype(np.complex128)
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
    else:
        fields = core._as_energy_stack(
            start_fields,
            name="start_fields",
        ).astype(np.complex128, copy=True)
        if fields.shape != holograms.shape:
            raise ValueError("start_fields must have the same shape as holograms.")

    errors = {
        "observation_steps": [],
        "projection_steps": [],
        "settings": recipe.copy(),
    }
    rng = np.random.default_rng(recipe["random_seed"])
    start_time = time.time()

    warmup_schedule = core._build_update_schedule(
        recipe,
        name="warmup",
        allow_disabled=True,
    )
    for observation in range(n_observations):
        if not warmup_schedule:
            break
        fields[observation], results = _run_update_schedule(
            fields[observation],
            amplitudes[observation],
            supportmask,
            bsmasks[observation],
            warmup_schedule,
            recipe,
        )
        for result in results:
            errors["observation_steps"].append(
                {
                    "outer": -1,
                    "observation": observation,
                    "state": labels[observation],
                    "polarization_sign": signs[observation],
                    "stage": "warmup",
                    **result,
                }
            )

    inner_schedule = core._build_update_schedule(recipe, name="inner")
    components = {
        "projection_model": recipe["projection_model"],
        "state_names": state_names,
    }

    for outer in range(recipe["outer_iterations"]):
        if recipe["shuffle_observations"]:
            observation_order = rng.permutation(n_observations)
        else:
            observation_order = np.arange(n_observations)

        for observation in observation_order:
            fields[observation], results = _run_update_schedule(
                fields[observation],
                amplitudes[observation],
                supportmask,
                bsmasks[observation],
                inner_schedule,
                recipe,
            )
            for result in results:
                errors["observation_steps"].append(
                    {
                        "outer": outer,
                        "observation": int(observation),
                        "state": labels[observation],
                        "polarization_sign": signs[observation],
                        "stage": "joint",
                        **result,
                    }
                )

        do_projection = (
            recipe["projection_model"] != "none"
            and outer >= recipe["projection_start"]
            and (
                (outer - recipe["projection_start"])
                % recipe["projection_every"]
                == 0
            )
        )
        if do_projection:
            fields, components = _project_fields(
                fields,
                labels,
                signs,
                recipe,
                relaxation=recipe["projection_relaxation"],
            )
            errors["projection_steps"].append(
                {
                    "outer": outer,
                    "identifiable": components["identifiable"],
                    "fit_residual_rms": components["fit_residual_rms"],
                }
            )

    if recipe["projection_model"] != "none":
        fields, components = _project_fields(
            fields,
            labels,
            signs,
            recipe,
            relaxation=1.0,
        )

    if recipe["final_fourier_constraint"]:
        fields = _apply_measured_amplitudes(fields, amplitudes, bsmasks)
        components["final_fourier_constraint_applied"] = True
    else:
        components["final_fourier_constraint_applied"] = False

    components["state_labels"] = labels.copy()
    components["polarization_signs"] = signs.copy()
    errors["runtime_seconds"] = float(np.round(time.time() - start_time, 3))
    return fields, components, bsmasks, errors
