"""
Joint multi-energy and incoherent-multimode phase retrieval.

This module composes:

- ``phase_retrieval_core_multienergy`` for the cross-energy object models and
  spectral constraints.
- ``phase_retrieval_core_multimode`` for the modal Fourier constraint
  ``I = sum_m |Psi_m|**2`` and modal phase-retrieval updates.

Cross-energy object projections are applied independently to each mode. This
assumes that a given mode index describes the corresponding physical mode at
all energies; the driver preserves mode order but cannot resolve arbitrary
permutations or unitary mixing between degenerate modes.

For ``Nmodes > 1``, reconstructed fields have shape
``(nE, Nmodes, nx, ny)``. For ``Nmodes == 1``, the mode axis is squeezed and
the API matches ``phase_retrieval_core_multienergy``.
"""

import time

import numpy as np
from scipy import stats

try:
    from . import phase_retrieval_core_multienergy as multi_energy
    from . import phase_retrieval_core_multimode as multimode
    from .phase_retrieval_core_multienergy import *
except ImportError:
    import phase_retrieval_core_multienergy as multi_energy
    import phase_retrieval_core_multimode as multimode
    from phase_retrieval_core_multienergy import *


# The ordinary two-helicity API in this combined module is the multimode one.
default_phase_retrieval_recipe = multimode.default_phase_retrieval_recipe
phase_retrieval_algorithm = multimode.phase_retrieval_algorithm
PhaseRtrv_core = multimode.PhaseRtrv_core
plot_phase_retrieval_errors = multimode.plot_phase_retrieval_errors


def _validate_nmodes(nmodes):
    if isinstance(nmodes, bool) or not isinstance(nmodes, (int, np.integer)):
        raise ValueError("Nmodes must be a positive integer.")
    nmodes = int(nmodes)
    if nmodes <= 0:
        raise ValueError("Nmodes must be a positive integer.")
    return nmodes


def _as_energy_mode_stack(fields, name="fields"):
    """
    Return fields as ``(nE, Nmodes, nx, ny)`` and report whether modes squeezed.
    """
    fields = np.asarray(fields)
    if fields.ndim == 3:
        return fields[:, None, :, :], True
    if fields.ndim == 4:
        return fields, False
    raise ValueError(
        f"{name} must have shape (nE, nx, ny) or (nE, Nmodes, nx, ny)."
    )


def _maybe_squeeze_energy_modes(fields, nmodes):
    if nmodes == 1 and np.asarray(fields).ndim == 4:
        return fields[:, 0]
    return fields


def _modal_amplitude(fields):
    fields, _ = _as_energy_mode_stack(fields)
    return np.sqrt(np.sum(np.abs(fields) ** 2, axis=1))


def project_fourier_fields_multi_energy_multimode(
    phase_stack,
    projection_model="svd",
    rank=1,
    static_mode="mean",
    weights=None,
    relaxation=1.0,
    log_floor=1e-12,
    spectral_constraint="free",
    energy_values=None,
    known_beta_spectrum=None,
    known_delta_spectrum=None,
    absorption_part="real",
    kk_sign=1.0,
    kk_subtract_baseline=True,
    kk_normalize_input=False,
    known_beta_normalization="none",
    fit_known_beta_scale=True,
    fit_known_beta_offset=True,
    return_components=False,
):
    """
    Apply the selected cross-energy projection independently to each mode.

    Mode indices must represent corresponding incoherent modes across energy.
    The reconstruction driver preserves this ordering by propagating each
    energy's modal fields between joint iterations.
    """
    modal_fields, squeezed = _as_energy_mode_stack(
        phase_stack, name="phase_stack"
    )
    n_energy, nmodes, nx, ny = modal_fields.shape
    projected = np.empty_like(modal_fields, dtype=np.complex128)
    mode_components = []

    for mode_index in range(nmodes):
        result = multi_energy.project_fourier_fields_multi_energy(
            modal_fields[:, mode_index],
            projection_model=projection_model,
            rank=rank,
            static_mode=static_mode,
            weights=weights,
            relaxation=relaxation,
            log_floor=log_floor,
            spectral_constraint=spectral_constraint,
            energy_values=energy_values,
            known_beta_spectrum=known_beta_spectrum,
            known_delta_spectrum=known_delta_spectrum,
            absorption_part=absorption_part,
            kk_sign=kk_sign,
            kk_subtract_baseline=kk_subtract_baseline,
            kk_normalize_input=kk_normalize_input,
            known_beta_normalization=known_beta_normalization,
            fit_known_beta_scale=fit_known_beta_scale,
            fit_known_beta_offset=fit_known_beta_offset,
            return_components=return_components,
        )
        if return_components:
            projected[:, mode_index], components = result
            mode_components.append(components)
        else:
            projected[:, mode_index] = result

    output = projected[:, 0] if squeezed else projected
    if not return_components:
        return output

    components = {
        "projection_model": str(projection_model).lower(),
        "Nmodes": nmodes,
        "mode_components": mode_components,
    }
    if nmodes == 1 and mode_components:
        components.update(mode_components[0])
        components["Nmodes"] = 1
        components["mode_components"] = mode_components
    return output, components


def default_multi_energy_phase_retrieval_recipe():
    """Return defaults for joint multi-energy, multimode reconstruction."""
    recipe = multi_energy.default_multi_energy_phase_retrieval_recipe()
    recipe.update(
        {
            "Nmodes": 1,
            "mode_initialization_seed": 0,
        }
    )
    return recipe


def _verify_multi_energy_multimode_recipe(recipe, n_energy):
    multi_energy._verify_multi_energy_recipe(recipe, n_energy)
    _validate_nmodes(recipe["Nmodes"])
    seed = recipe["mode_initialization_seed"]
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError("mode_initialization_seed must be an integer or None.")


def _initialize_modal_fields(
    supportmask,
    amplitudes,
    intensities,
    mask_stack,
    nmodes,
    random_seed,
):
    n_energy, nx, ny = amplitudes.shape
    support_modes = multimode._as_modes(
        supportmask,
        nmodes,
        (nx, ny),
        "supportmask",
        dtype=np.asarray(supportmask).dtype,
    )
    base_modes = np.empty((nmodes, nx, ny), dtype=np.complex128)
    for mode_index in range(nmodes):
        base_modes[mode_index] = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(support_modes[mode_index]))
        )

    if nmodes > 1:
        rng = np.random.default_rng(random_seed)
        modal_phase = np.exp(
            1j
            * rng.uniform(
                -np.pi,
                np.pi,
                (nmodes, nx, ny),
            )
        )
        return (
            amplitudes[:, None, :, :]
            * modal_phase[None, :, :, :]
            / np.sqrt(nmodes)
        )

    fields = np.repeat(base_modes[None, :, :, :], n_energy, axis=0)

    for energy_index in range(n_energy):
        valid = (
            (mask_stack[energy_index] == 0)
            & (intensities[energy_index] > 0)
        )
        if not np.any(valid):
            continue
        measured = amplitudes[energy_index][valid].ravel()
        current = np.sqrt(
            np.sum(np.abs(fields[energy_index]) ** 2, axis=0)
        )[valid].ravel()
        if (
            measured.size >= 2
            and np.ptp(measured) > 0
            and np.ptp(current) > 0
        ):
            fit = stats.linregress(measured, current)
            if abs(fit.slope) > 1e-12:
                fields[energy_index] /= fit.slope

    return fields


def _apply_measured_modal_amplitude(fields, amplitudes, bsmasks):
    """Apply each measured amplitude to the summed modal intensity."""
    constrained = fields.copy()
    total_amplitude = np.sqrt(np.sum(np.abs(constrained) ** 2, axis=1))
    safe_amplitude = np.where(total_amplitude > 1e-30, total_amplitude, 1e-30)
    factor = amplitudes / safe_amplitude
    for energy_index in range(fields.shape[0]):
        observed = bsmasks[energy_index] == 0
        for mode_index in range(fields.shape[1]):
            constrained[energy_index, mode_index, observed] *= factor[
                energy_index, observed
            ]
    return constrained


def _run_energy_update_schedule(
    field,
    amplitude,
    supportmask,
    bsmask,
    schedule,
    recipe,
    nmodes,
    image_shape,
):
    """Run all scheduled multimode updates sequentially for one energy."""
    stage_results = []
    for stage_index, stage in enumerate(schedule):
        mode = stage["mode"]
        Nit = stage["Nit"]
        result, err_d, err_s, _ = multimode.PhaseRtrv_core(
            diffract=amplitude,
            mask=supportmask,
            mode=mode,
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
            Nmodes=nmodes,
        )
        field = multimode._as_modes(
            result,
            nmodes,
            image_shape,
            "retrieved fields",
            dtype=np.complex128,
        )
        stage_results.append(
            {
                "schedule_stage": stage_index,
                "mode": mode,
                "Nit": Nit,
                "beta_zero": stage["beta_zero"],
                "beta_mode": stage["beta_mode"],
                "alpha_zero": stage["alpha_zero"],
                "alpha_mode": stage["alpha_mode"],
                "TV_freq": stage["TV_freq"],
                "Nmodes": nmodes,
                "error": np.asarray(err_d),
                "support_error": np.asarray(err_s),
            }
        )
    return field, stage_results


def multi_energy_phase_retrieval_algorithm(
    holograms,
    mask_pixel,
    supportmask,
    multi_energy_recipe=None,
    start_fields=None,
):
    """
    Jointly reconstruct multiple energies and incoherent modes.

    Parameters are the same as in
    ``phase_retrieval_core_multienergy.multi_energy_phase_retrieval_algorithm``
    with two additions in the recipe:

    ``Nmodes``
        Number of incoherent modes. ``Nmodes=1`` returns ``(nE, nx, ny)``.
        Larger values return ``(nE, Nmodes, nx, ny)``.
    ``mode_initialization_seed``
        Seed used to break modal degeneracy in the default initialization.

    ``inner_mode`` / ``inner_Nit``
        Parallel scalar-or-list settings defining the update stages performed
        at every energy during each outer iteration. Stage-specific beta,
        alpha, and TV controls use the same convention as the single-mode
        multi-energy driver.

    Returns
    -------
    retrieved : ndarray
        Shape ``(nE, nx, ny)`` for one mode or
        ``(nE, Nmodes, nx, ny)`` for multiple modes.
    components : dict
        Per-mode cross-energy projection results under ``mode_components``.
    bsmasks : ndarray
        Energy-dependent invalid-pixel masks with shape ``(nE, nx, ny)``.
    errors : dict
        Update and projection diagnostics.
    """
    recipe = default_multi_energy_phase_retrieval_recipe()
    if multi_energy_recipe is not None:
        if not isinstance(multi_energy_recipe, dict):
            raise TypeError("multi_energy_recipe must be a dictionary.")
        unknown_keys = set(multi_energy_recipe) - set(recipe)
        if unknown_keys:
            raise ValueError(
                f"Unknown multi-energy multimode recipe key(s): "
                f"{sorted(unknown_keys)}"
            )
        recipe.update(multi_energy_recipe)

    holograms = multi_energy._as_energy_stack(holograms)
    n_energy, nx, ny = holograms.shape
    nmodes = _validate_nmodes(recipe["Nmodes"])
    _verify_multi_energy_multimode_recipe(recipe, n_energy)

    supportmask = np.asarray(supportmask)
    if supportmask.ndim == 2:
        if supportmask.shape != (nx, ny):
            raise ValueError("supportmask must have shape (nx, ny).")
    elif supportmask.ndim == 3:
        if supportmask.shape[1:] != (nx, ny):
            raise ValueError("modal supportmask must have shape (Nmodes, nx, ny).")
        if supportmask.shape[0] not in {1, nmodes}:
            raise ValueError("supportmask first axis must be 1 or Nmodes.")
    else:
        raise ValueError("supportmask must be 2D or 3D.")

    amplitudes, intensities, bsmasks = multi_energy._prepare_energy_amplitudes(
        holograms,
        mask_pixel,
        hologram_intensity_cutoff_vmin=recipe[
            "hologram_intensity_cutoff_vmin"
        ],
    )
    mask_stack = multi_energy._as_energy_mask(
        mask_pixel, nE=n_energy, image_shape=(nx, ny)
    )

    if start_fields is None:
        fields = _initialize_modal_fields(
            supportmask,
            amplitudes,
            intensities,
            mask_stack,
            nmodes,
            recipe["mode_initialization_seed"],
        )
    else:
        fields, _ = _as_energy_mode_stack(start_fields, name="start_fields")
        fields = fields.astype(np.complex128, copy=True)
        if fields.shape != (n_energy, nmodes, nx, ny):
            raise ValueError(
                "start_fields must have shape (nE, nx, ny) for Nmodes=1 or "
                "(nE, Nmodes, nx, ny)."
            )

    rng = np.random.default_rng(recipe["random_seed"])
    errors = {
        "energy_steps": [],
        "projection_steps": [],
        "settings": recipe.copy(),
    }
    start_time = time.time()

    warmup_schedule = multi_energy._build_update_schedule(
        recipe,
        name="warmup",
        allow_disabled=True,
    )
    if warmup_schedule:
        for energy_index in range(n_energy):
            fields[energy_index], stage_results = _run_energy_update_schedule(
                fields[energy_index],
                amplitudes[energy_index],
                supportmask,
                bsmasks[energy_index],
                warmup_schedule,
                recipe,
                nmodes,
                (nx, ny),
            )
            for stage_result in stage_results:
                errors["energy_steps"].append({
                    "outer": -1,
                    "energy": energy_index,
                    "stage": "warmup",
                    **stage_result,
                })

    outer_iterations = int(recipe["outer_iterations"])
    inner_schedule = multi_energy._build_update_schedule(
        recipe,
        name="inner",
    )
    projection_every = int(recipe["projection_every"])
    projection_start = int(recipe["projection_start"])
    components = {
        "projection_model": recipe["projection_model"],
        "Nmodes": nmodes,
    }

    for outer in range(outer_iterations):
        if recipe["shuffle_energies"]:
            energy_order = rng.permutation(n_energy)
        else:
            energy_order = np.arange(n_energy)

        for energy_index in energy_order:
            fields[energy_index], stage_results = _run_energy_update_schedule(
                fields[energy_index],
                amplitudes[energy_index],
                supportmask,
                bsmasks[energy_index],
                inner_schedule,
                recipe,
                nmodes,
                (nx, ny),
            )
            for stage_result in stage_results:
                errors["energy_steps"].append({
                    "outer": outer,
                    "energy": int(energy_index),
                    "stage": "joint",
                    **stage_result,
                })

        do_projection = (
            outer >= projection_start
            and ((outer - projection_start) % projection_every == 0)
        )
        if do_projection:
            fields, components = (
                project_fourier_fields_multi_energy_multimode(
                    fields,
                    projection_model=recipe["projection_model"],
                    rank=recipe["rank"],
                    static_mode=recipe["projection_static_mode"],
                    weights=recipe["energy_weights"],
                    relaxation=recipe["projection_relaxation"],
                    log_floor=recipe["log_floor"],
                    spectral_constraint=recipe["spectral_constraint"],
                    energy_values=recipe["energy_values"],
                    known_beta_spectrum=recipe["known_beta_spectrum"],
                    known_delta_spectrum=recipe["known_delta_spectrum"],
                    absorption_part=recipe["absorption_part"],
                    kk_sign=recipe["kk_sign"],
                    kk_subtract_baseline=recipe["kk_subtract_baseline"],
                    kk_normalize_input=recipe["kk_normalize_input"],
                    known_beta_normalization=recipe[
                        "known_beta_normalization"
                    ],
                    fit_known_beta_scale=recipe["fit_known_beta_scale"],
                    fit_known_beta_offset=recipe["fit_known_beta_offset"],
                    return_components=True,
                )
            )
            errors["projection_steps"].append(
                {
                    "outer": outer,
                    "projection_model": components.get("projection_model"),
                    "Nmodes": nmodes,
                    "rank": recipe["rank"],
                    "spectral_constraint": recipe["spectral_constraint"],
                    "relaxation": recipe["projection_relaxation"],
                }
            )

    fields, components = project_fourier_fields_multi_energy_multimode(
        fields,
        projection_model=recipe["projection_model"],
        rank=recipe["rank"],
        static_mode=recipe["projection_static_mode"],
        weights=recipe["energy_weights"],
        relaxation=1.0,
        log_floor=recipe["log_floor"],
        spectral_constraint=recipe["spectral_constraint"],
        energy_values=recipe["energy_values"],
        known_beta_spectrum=recipe["known_beta_spectrum"],
        known_delta_spectrum=recipe["known_delta_spectrum"],
        absorption_part=recipe["absorption_part"],
        kk_sign=recipe["kk_sign"],
        kk_subtract_baseline=recipe["kk_subtract_baseline"],
        kk_normalize_input=recipe["kk_normalize_input"],
        known_beta_normalization=recipe["known_beta_normalization"],
        fit_known_beta_scale=recipe["fit_known_beta_scale"],
        fit_known_beta_offset=recipe["fit_known_beta_offset"],
        return_components=True,
    )
    fields, _ = _as_energy_mode_stack(fields)

    if recipe["final_fourier_constraint"]:
        fields = _apply_measured_modal_amplitude(fields, amplitudes, bsmasks)
        components["final_fourier_constraint_applied"] = True
    else:
        components["final_fourier_constraint_applied"] = False

    components["Nmodes"] = nmodes
    errors["runtime_seconds"] = float(np.round(time.time() - start_time, 3))
    return (
        _maybe_squeeze_energy_modes(fields, nmodes),
        components,
        bsmasks,
        errors,
    )
