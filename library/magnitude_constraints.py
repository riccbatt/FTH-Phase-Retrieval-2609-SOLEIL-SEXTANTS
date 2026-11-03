"""Fourier-magnitude constraints and Richardson–Lucy (RL) blur fitting."""
import numpy as np
from .phase_retrieval_backend import xp, fft2, ifft2, to_numpy


def RL_kernel(gamma=None, *, shape=None):
    """Initialize a near-delta blur or normalize a fitted blur per spatial plane."""
    if gamma is None:
        gamma = np.full(shape, 2e-6, dtype=float)
        gamma[..., shape[-2] // 2, shape[-1] // 2] = 0.7
    kernel = np.maximum(np.asarray(gamma).real, 0.0).copy()
    mass = kernel.sum(axis=(-2, -1), keepdims=True)
    if np.any(~np.isfinite(kernel)) or np.any(mass <= 0):
        raise ValueError("RL kernels must have finite positive mass.")
    return kernel / mass


def fit_RL_kernel(fields, targets, gamma, iterations, weights=None):
    """
    RL fit of one kernel per mode to the summed, weighted state likelihood.

    Fields and targets use centered detector coordinates. All states contribute,
    including every observation. Unlike averaging separate fits,
    every RL step uses the joint residual and the sum of modal predictions.

    """
    from scipy.fft import fft2 as cpu_fft2, ifft2 as cpu_ifft2

    fields = np.asarray(fields)
    if fields.ndim == 3:
        fields = fields[:, None]
    squeeze = np.ndim(gamma) == 2
    kernel = RL_kernel(gamma)
    if squeeze:
        kernel = kernel[None]
    kernel = np.fft.ifftshift(kernel, axes=(-2, -1))
    weights = np.ones(len(fields)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(fields),) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Pooled RL needs nonnegative weights with positive total.")
    # Stream states instead of caching an extra complex FFT stack for the loop.
    for _ in range(iterations):
        correction = np.zeros_like(kernel)
        kernel_fft = cpu_fft2(kernel)
        for index, field in enumerate(fields):
            if weights[index] == 0:
                continue
            intensity_fft = cpu_fft2(np.fft.ifftshift(np.abs(field) ** 2, axes=(-2, -1)))
            prediction = cpu_ifft2(intensity_fft * kernel_fft).real.sum(axis=0)
            target = np.fft.ifftshift(np.maximum(targets[index], 0))
            ratio_fft = cpu_fft2(target / np.maximum(prediction, 1e-30))
            correction += weights[index] * np.maximum(
                cpu_ifft2(np.conj(intensity_fft) * ratio_fft).real, 0,
            )
        updated = kernel * correction
        mass = updated.sum(axis=(-2, -1), keepdims=True)
        # A mode with no illumination cannot inform its kernel.
        kernel = np.divide(updated, mass, out=kernel.copy(), where=mass > 0)
    kernel = np.fft.fftshift(kernel, axes=(-2, -1))
    return kernel[0] if squeeze else kernel



def modeled_intensity(fields, kernels=None):
    """Sum incoherent modal intensities, optionally blurred by centered kernels."""
    power = xp.abs(fields)**2
    if kernels is not None:
        kernel = xp.asarray(kernels)
        power = xp.fft.fftshift(
            ifft2(fft2(xp.fft.ifftshift(power, axes=(-2, -1)))
                  * fft2(xp.fft.ifftshift(kernel, axes=(-2, -1)))).real,
            axes=(-2, -1),
        )
    return xp.maximum(power.sum(axis=1), 0)


def apply_fourier_constraint(fields, amplitudes, invalid, kernels=None):
    """Match the detector intensity model while leaving invalid pixels unchanged."""
    amplitude = xp.sqrt(modeled_intensity(fields, kernels))
    factor = xp.where(invalid, 1, amplitudes / xp.maximum(amplitude, 1e-30))
    return fields * factor[:, None]


def refine_RL(fields, targets, kernels, recipe, weights, iterations, fitted=False):
    """Fit per-observation or shared RL kernels to the current exit waves."""
    fields = to_numpy(fields)
    if recipe['RL_scope'] == 'per_observation':
        return np.stack([fit_RL_kernel(field[None], target[None], kernel, iterations)
                         for field, target, kernel in zip(fields, targets, kernels)])
    if recipe['RL_freeze_after_reference']:
        if fitted:
            return kernels
        index = recipe['illumination_intensity_reference_observation']
        return fit_RL_kernel(fields[index:index+1], targets[index:index+1], kernels[0], iterations)[None]
    strategy = recipe['RL_shared_update']
    if strategy == 'pooled':
        kernel = fit_RL_kernel(fields, targets, kernels[0], iterations, weights)
    elif strategy == 'average':
        fitted_kernels = [fit_RL_kernel(field[None], target[None], kernels[0], iterations)
                          for field, target in zip(fields, targets)]
        kernel = RL_kernel(np.average(fitted_kernels, axis=0, weights=weights))
    else:
        kernel = kernels[0]
        for field, target in zip(fields, targets):
            kernel = fit_RL_kernel(field[None], target[None], kernel, iterations)
    return kernel[None]


def detector_error(fields, amplitudes, invalid, kernels=None):
    """Relative intensity residual on measured detector pixels, one value per observation."""
    target = xp.where(invalid, 0, amplitudes**2)
    residual = xp.where(invalid, 0, modeled_intensity(fields, kernels) - target)
    return to_numpy(xp.linalg.norm(residual, axis=(-2, -1))
                    / xp.maximum(xp.linalg.norm(target, axis=(-2, -1)), 1e-30))
