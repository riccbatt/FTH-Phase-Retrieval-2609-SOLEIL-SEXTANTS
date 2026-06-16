"""
Parallelized joint multi-energy and incoherent-multimode phase retrieval.

This module keeps the public API of
``phase_retrieval_core_multienergy_multimode`` but executes the independent
warmup updates and the independent energy-update batches between cross-energy
projections in parallel. The cadence defined by ``projection_every`` is
preserved exactly: updates are only batched up to the next projection boundary,
and the projection itself still runs serially.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from . import phase_retrieval_core_multienergy as multi_energy
    from . import phase_retrieval_core_multienergy_multimode as serial
except ImportError:
    import phase_retrieval_core_multienergy as multi_energy
    import phase_retrieval_core_multienergy_multimode as serial


default_phase_retrieval_recipe = serial.default_phase_retrieval_recipe
phase_retrieval_algorithm = serial.phase_retrieval_algorithm
PhaseRtrv_core = serial.PhaseRtrv_core
plot_phase_retrieval_errors = serial.plot_phase_retrieval_errors
default_multi_energy_phase_retrieval_recipe = (
    serial.default_multi_energy_phase_retrieval_recipe
)
project_fourier_fields_multi_energy_multimode = (
    serial.project_fourier_fields_multi_energy_multimode
)


def _worker_count(batch_size):
    """Return a conservative thread count for one parallel batch."""
    return max(1, min(int(batch_size), os.cpu_count() or 1))


def _next_projection_boundary_after(
    completed_updates,
    projection_start,
    projection_every,
):
    """Return the next projection boundary strictly after completed_updates."""
    if projection_every <= 0:
        raise ValueError("projection_every must be > 0.")

    if completed_updates < projection_start:
        return projection_start

    offset = completed_updates - projection_start
    return projection_start + ((offset // projection_every) + 1) * projection_every


def _run_energy_update_task(task_args):
    """Execute one warmup or inner energy update."""
    (
        energy_index,
        field,
        amplitude,
        supportmask,
        bsmask,
        schedule,
        recipe,
        nmodes,
        image_shape,
    ) = task_args

    updated_field, stage_results = serial._run_energy_update_schedule(
        field,
        amplitude,
        supportmask,
        bsmask,
        schedule,
        recipe,
        nmodes,
        image_shape,
    )
    return energy_index, updated_field, stage_results


def _parallel_run_energy_updates(
    fields,
    energy_indices,
    amplitudes,
    supportmask,
    bsmasks,
    schedule,
    recipe,
    nmodes,
    image_shape,
):
    """Run one independent energy-update batch in parallel."""
    energy_indices = list(map(int, energy_indices))
    if not energy_indices:
        return []

    task_args = [
        (
            energy_index,
            fields[energy_index].copy(),
            amplitudes[energy_index],
            supportmask,
            bsmasks[energy_index],
            schedule,
            recipe,
            nmodes,
            image_shape,
        )
        for energy_index in energy_indices
    ]

    worker_count = _worker_count(len(task_args))
    if worker_count == 1:
        return [_run_energy_update_task(task_args_item) for task_args_item in task_args]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_run_energy_update_task, task_args))


def multi_energy_phase_retrieval_algorithm(
    holograms,
    mask_pixel,
    supportmask,
    multi_energy_recipe=None,
    start_fields=None,
    parallel_workers=None,
):
    """
    Jointly reconstruct multiple energies and incoherent modes in parallel.

    The public API matches the serial multimode driver, with one extra optional
    ``parallel_workers`` argument to cap the thread count used for warmup and
    between-projection batches.
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
    nmodes = serial._validate_nmodes(recipe["Nmodes"])
    serial._verify_multi_energy_multimode_recipe(recipe, n_energy)

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

    projection_supportmask = (
        np.fft.fftshift(supportmask, axes=(-2, -1))
        if (
            recipe["projection_constraints_inside_support_only"]
            or recipe["physical_constraints_inside_support_only"]
        )
        else None
    )

    amplitudes, intensities, bsmasks = multi_energy._prepare_energy_amplitudes(
        holograms,
        mask_pixel,
        hologram_intensity_cutoff_vmin=recipe[
            "hologram_intensity_cutoff_vmin"
        ],
    )
    mask_stack = multi_energy._as_energy_mask(
        mask_pixel,
        nE=n_energy,
        image_shape=(nx, ny),
    )

    if start_fields is None:
        fields = serial._initialize_modal_fields(
            supportmask,
            amplitudes,
            intensities,
            mask_stack,
            nmodes,
            recipe["mode_initialization_seed"],
        )
    else:
        fields, _ = serial._as_energy_mode_stack(start_fields, name="start_fields")
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
        warmup_indices = np.arange(n_energy)
        warmup_results = _parallel_run_energy_updates(
            fields,
            warmup_indices,
            amplitudes,
            supportmask,
            bsmasks,
            warmup_schedule,
            recipe,
            nmodes,
            (nx, ny),
        )
        for energy_index, updated_field, stage_results in warmup_results:
            fields[energy_index] = updated_field
            for stage_result in stage_results:
                errors["energy_steps"].append(
                    {
                        "outer": -1,
                        "energy": int(energy_index),
                        "stage": "warmup",
                        **stage_result,
                    }
                )
            print(f"Warmup energy {int(energy_index)}: completed")
        fieldswarmup = fields[:].copy()

    outer_iterations = int(recipe["outer_iterations"])
    inner_schedule = multi_energy._build_update_schedule(
        recipe,
        name="inner",
    )
    projection_every, projection_start = multi_energy._resolve_projection_cadence(
        recipe,
        default_every=n_energy,
    )
    completed_updates = 0
    components = {
        "projection_model": recipe["projection_model"],
        "Nmodes": nmodes,
    }

    for outer in range(outer_iterations):
        if recipe["shuffle_energies"]:
            energy_order = rng.permutation(n_energy)
        else:
            energy_order = np.arange(n_energy)

        cursor = 0
        while cursor < n_energy:
            next_boundary = _next_projection_boundary_after(
                completed_updates,
                projection_start,
                projection_every,
            )
            batch_size = min(n_energy - cursor, next_boundary - completed_updates)
            if batch_size <= 0:
                batch_size = 1

            batch_indices = energy_order[cursor : cursor + batch_size]
            batch_results = _parallel_run_energy_updates(
                fields,
                batch_indices,
                amplitudes,
                supportmask,
                bsmasks,
                inner_schedule,
                recipe,
                nmodes,
                (nx, ny),
            )

            for energy_index, updated_field, stage_results in batch_results:
                fields[energy_index] = updated_field
                for stage_result in stage_results:
                    errors["energy_steps"].append(
                        {
                            "outer": outer,
                            "energy": int(energy_index),
                            "stage": "joint",
                            **stage_result,
                        }
                    )

                completed_updates += 1
                if not multi_energy._projection_is_due(
                    completed_updates,
                    projection_start,
                    projection_every,
                ):
                    print(
                        f"Outer {outer}: completed update {completed_updates} "
                        f"for energy {int(energy_index)} (no projection due)"
                    )
                    continue

                fields, components = project_fourier_fields_multi_energy_multimode(
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
                    projection_supportmask=projection_supportmask,
                    return_components=True,
                )
                errors["projection_steps"].append(
                    {
                        "outer": outer,
                        "energy": int(energy_index),
                        "completed_update": completed_updates,
                        "projection_model": components.get("projection_model"),
                        "Nmodes": nmodes,
                        "rank": recipe["rank"],
                        "spectral_constraint": recipe["spectral_constraint"],
                        "relaxation": recipe["projection_relaxation"],
                    }
                )
                print(
                    f"Outer {outer}: projection applied after update "
                    f"{completed_updates} for energy {int(energy_index)} - "
                    f"model={components.get('projection_model')}, Nmodes={nmodes}, "
                    f"rank={recipe['rank']}, "
                    f"spectral_constraint={recipe['spectral_constraint']}"
                )

            cursor += batch_size

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
        projection_supportmask=projection_supportmask,
        return_components=True,
    )
    fields, _ = serial._as_energy_mode_stack(fields)

    if recipe["final_fourier_constraint"]:
        fields = serial._apply_measured_modal_amplitude(fields, amplitudes, bsmasks)
        components["final_fourier_constraint_applied"] = True
    else:
        components["final_fourier_constraint_applied"] = False

    components["Nmodes"] = nmodes
    errors["runtime_seconds"] = float(np.round(time.time() - start_time, 3))
    print("--- %s seconds ---" % np.round((time.time() - start_time), 2))
    print("Multi-energy multimode Phase Retrieval Done!")
    return (
        serial._maybe_squeeze_energy_modes(fields, nmodes),
        fieldswarmup if warmup_schedule else None,
        components,
        bsmasks,
        errors,
    )
