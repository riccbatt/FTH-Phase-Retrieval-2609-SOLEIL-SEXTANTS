"""
Parallelized joint multi-energy and incoherent-multimode phase retrieval with
CUDA JIT acceleration for supported single-mode stages.

This module keeps the public API of
``phase_retrieval_core_multienergy_multimode`` but executes independent energy
updates in parallel batches and uses the CUDA JIT kernel from
``phase_retrieval_core_jit`` when the local stage is compatible. The cadence
defined by ``projection_every`` is preserved exactly: updates are batched only
up to the next projection boundary, and the projection itself still runs
serially.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from . import phase_retrieval_core_jit as jit_core
    from . import phase_retrieval_core_multienergy as multi_energy
    from . import phase_retrieval_core_multienergy_multimode as serial
except ImportError:
    import phase_retrieval_core_jit as jit_core
    import phase_retrieval_core_multienergy as multi_energy
    import phase_retrieval_core_multienergy_multimode as serial


cuda_jit_status = jit_core.cuda_jit_status
warm_up_jit = jit_core.warm_up_jit
PhaseRtrv_core_jit = jit_core.PhaseRtrv_core_jit

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


def _worker_count(batch_size, parallel_workers=None):
    """Return a conservative thread count for one parallel batch."""
    limit = os.cpu_count() or 1
    if parallel_workers is not None:
        if isinstance(parallel_workers, bool) or not isinstance(
            parallel_workers,
            (int, np.integer),
        ):
            raise ValueError("parallel_workers must be a positive integer or None.")
        if parallel_workers <= 0:
            raise ValueError("parallel_workers must be a positive integer or None.")
        limit = min(limit, int(parallel_workers))
    return max(1, min(int(batch_size), limit))


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


def _stage_supports_jit(stage, nmodes):
    """Return whether one stage can run through the CUDA-JIT kernel."""
    if nmodes != 1:
        return False
    if float(stage["alpha_zero"]) != 0.0:
        return False
    mode = stage["mode"]
    return mode in {"ER", "SF", "HAPRE", "RAAR", "HIOs", "HIO", "CHIO", "HPR"}


def _run_energy_update_stage_serial(
    field,
    amplitude,
    supportmask,
    bsmask,
    stage,
    recipe,
    nmodes,
    image_shape,
):
    """Run one energy-update stage using the serial multimode implementation."""
    schedule = [stage]
    field, stage_results = serial._run_energy_update_schedule(
        field,
        amplitude,
        supportmask,
        bsmask,
        schedule,
        recipe,
        nmodes,
        image_shape,
    )
    return field, stage_results


def _run_energy_update_stage_jit(
    field,
    amplitude,
    supportmask,
    bsmask,
    stage,
    recipe,
):
    """Run one single-mode stage through the CUDA-JIT kernel."""
    result, error_diffract, error_support, _ = PhaseRtrv_core_jit(
        diffract=amplitude,
        mask=supportmask,
        mode=stage["mode"],
        Nit=stage["Nit"],
        beta_zero=stage["beta_zero"],
        beta_mode=stage["beta_mode"],
        alpha_zero=stage["alpha_zero"],
        alpha_mode=stage["alpha_mode"],
        Phase=field,
        seed=False,
        plot_every=recipe["plot_every"],
        bsmask=bsmask,
        real_object=False,
        average_img=min(max(1, recipe["average_img"]), stage["Nit"]),
        Fourier_last=recipe["Fourier_last"],
        gamma=None,
        RL_freq=recipe["RL_freq"],
        RL_it=recipe["RL_it"],
        TV_freq=stage["TV_freq"],
        fallback=True,
    )
    stage_results = [
        {
            "schedule_stage": 0,
            "mode": stage["mode"],
            "Nit": stage["Nit"],
            "beta_zero": stage["beta_zero"],
            "beta_mode": stage["beta_mode"],
            "alpha_zero": stage["alpha_zero"],
            "alpha_mode": stage["alpha_mode"],
            "TV_freq": stage["TV_freq"],
            "error": np.asarray(error_diffract),
            "support_error": np.asarray(error_support),
        }
    ]
    return result, stage_results


def _run_energy_update_schedule_parallel_jit(
    field,
    amplitude,
    supportmask,
    bsmask,
    schedule,
    recipe,
    nmodes,
    image_shape,
):
    """Run all scheduled stages for one energy, using JIT when possible."""
    stage_results = []
    for stage_index, stage in enumerate(schedule):
        if _stage_supports_jit(stage, nmodes):
            try:
                field, result = _run_energy_update_stage_jit(
                    field,
                    amplitude,
                    supportmask,
                    bsmask,
                    stage,
                    recipe,
                )
            except Exception:
                field, result = _run_energy_update_stage_serial(
                    field,
                    amplitude,
                    supportmask,
                    bsmask,
                    stage,
                    recipe,
                    nmodes,
                    image_shape,
                )
        else:
            field, result = _run_energy_update_stage_serial(
                field,
                amplitude,
                supportmask,
                bsmask,
                stage,
                recipe,
                nmodes,
                image_shape,
            )

        stage_result = dict(result[0])
        stage_result["schedule_stage"] = stage_index
        stage_results.append(stage_result)
    return field, stage_results


def _parallel_run_energy_updates_jit(
    fields,
    energy_indices,
    amplitudes,
    supportmask,
    bsmasks,
    schedule,
    recipe,
    nmodes,
    image_shape,
    parallel_workers=None,
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

    worker_count = _worker_count(len(task_args), parallel_workers=parallel_workers)

    def _task(task_args_item):
        (
            energy_index,
            field,
            amplitude,
            supportmask_local,
            bsmask_local,
            schedule_local,
            recipe_local,
            nmodes_local,
            image_shape_local,
        ) = task_args_item
        updated_field, stage_results = _run_energy_update_schedule_parallel_jit(
            field,
            amplitude,
            supportmask_local,
            bsmask_local,
            schedule_local,
            recipe_local,
            nmodes_local,
            image_shape_local,
        )
        return energy_index, updated_field, stage_results

    if worker_count == 1:
        return [_task(task_args_item) for task_args_item in task_args]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_task, task_args))


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
        warmup_results = _parallel_run_energy_updates_jit(
            fields,
            np.arange(n_energy),
            amplitudes,
            supportmask,
            bsmasks,
            warmup_schedule,
            recipe,
            nmodes,
            (nx, ny),
            parallel_workers=parallel_workers,
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
            batch_results = _parallel_run_energy_updates_jit(
                fields,
                batch_indices,
                amplitudes,
                supportmask,
                bsmasks,
                inner_schedule,
                recipe,
                nmodes,
                (nx, ny),
                parallel_workers=parallel_workers,
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
        components,
        bsmasks,
        errors,
    )
