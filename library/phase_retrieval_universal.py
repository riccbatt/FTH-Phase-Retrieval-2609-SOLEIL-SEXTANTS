"""One loop: Fourier magnitude, support/TV, optional physical projection and RL."""
from contextlib import nullcontext
import numpy as np
from scipy.fft import set_workers
from .phase_retrieval_backend import xp, fft2, ifft2, GPU, to_numpy
from .phase_retrieval_recipe import default_universal_phase_retrieval_recipe
from .phase_retrieval_initialization import initialize_phase_retrieval
from . import magnitude_constraints as magnitude
from .real_space_constraints import PROJECTIONS, total_variation_gradient
from .phase_retrieval_physical import apply_physical_constraint

__all__ = ['default_universal_phase_retrieval_recipe', 'initialize_phase_retrieval',
           'universal_phase_retrieval_algorithm']


def universal_phase_retrieval_algorithm(reconstruction):
    """Execute the prepared recipe and return fields, initialization, components, masks, errors."""
    recipe = reconstruction['recipe']
    fields = xp.asarray(reconstruction['fields']).copy()  # observation, mode, row, column
    previous = xp.asarray(reconstruction['previous']).copy()
    support = xp.asarray(reconstruction['object_support'])
    amplitudes = xp.asarray(reconstruction['amplitudes'])
    invalid = xp.asarray(reconstruction['bsmasks'] != 0)
    kernels = reconstruction['RL_kernels']
    targets = reconstruction['RL_targets'].copy()
    components, errors = {}, dict(detector=[], projection_steps=[])
    RL_fitted = False

    with set_workers(recipe['fft_workers']) if not GPU else nullcontext():
        for iteration, control in enumerate(reconstruction['schedule'], start=1):
            if control['RL_start']:
                targets = np.where(reconstruction['bsmasks'], to_numpy(xp.sum(abs(fields)**2, axis=1)),
                                   reconstruction['amplitudes']**2)
            # Use the blur model throughout an RL stage, including between fitting steps.
            active_kernels = kernels if control['RL'] else None
            fields = magnitude.apply_fourier_constraint(fields, amplitudes, invalid, active_kernels)

            # Object-space constraints act on spatial axes, independently for each mode/observation.
            obj = fft2(xp.fft.fftshift(fields, axes=(-2, -1)))
            if control['tv_step']:
                obj -= control['tv_step'] * total_variation_gradient(obj)
            obj = PROJECTIONS[control['mode']](obj, previous, support, control['beta'])
            fields = xp.fft.ifftshift(ifft2(obj), axes=(-2, -1))

            if control['physical_it']:
                fields, components = apply_physical_constraint(
                    to_numpy(fields), reconstruction, components, control['projection_relaxation'], control['physical_it'])
                fields = xp.asarray(fields)
                components['physical_fit_iteration'] = iteration
                errors['projection_steps'].append(dict(iteration=iteration, fit_residual_rms=components['fit_residual_rms']))

            if control['RL_it']:
                kernels = magnitude.refine_RL(fields, targets, kernels, recipe,
                                              recipe['observation_weights'], control['RL_it'], RL_fitted)
                RL_fitted = True
            # Feedback includes the physical update and persists across recipe stages.
            previous = fft2(xp.fft.fftshift(fields, axes=(-2, -1)))
            active_kernels = kernels if control['RL'] else None
            errors['detector'].append(magnitude.detector_error(fields, amplitudes, invalid, active_kernels))

    fields, initial = to_numpy(fields), reconstruction['initial']
    if kernels is not None:
        components['RL_kernels'] = kernels
    if recipe['Nmodes'] == 1:
        fields, initial = fields[:, 0], initial[:, 0]
    return fields, initial, components, reconstruction['bsmasks'], errors
