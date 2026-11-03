"""Fourier-magnitude constraint routines for the phase-retrieval kernel."""

from __future__ import annotations

import numpy as np


def _universal_module():
    """Return the active universal solver module without importing it eagerly."""
    from . import phase_retrieval_universal as universal

    return universal


def _modal_convolved_intensities(current_guess, current_gamma, nmodes):
    """Return one coherent or partially coherent intensity plane per mode."""
    universal = _universal_module()
    xp = universal.xp
    fft2 = universal.fft2
    ifft2 = universal.ifft2

    convolved = xp.zeros_like(current_guess, dtype=xp.complex128)
    for mode_index in range(nmodes):
        intensity = xp.abs(current_guess[mode_index]) ** 2
        if current_gamma is None:
            convolved[mode_index] = intensity
        else:
            convolved[mode_index] = ifft2(
                fft2(intensity) * fft2(current_gamma[mode_index])
            )
    return convolved


def _apply_modal_fourier_constraint(
    current_guess,
    current_convolved,
    measured_amplitude,
    observed,
    invalid,
):
    """Rescale each modal field so the modeled detector sum matches the measured amplitude."""
    universal = _universal_module()
    xp = universal.xp

    total_intensity = xp.sum(current_convolved, axis=0)
    modal_amplitude = xp.sqrt(total_intensity)
    modal_amplitude = xp.where(
        xp.abs(modal_amplitude) > 1e-30,
        modal_amplitude,
        1e-30,
    )
    factor = measured_amplitude / modal_amplitude
    return current_guess * (
        observed[None, :, :] * factor[None, :, :]
        + invalid[None, :, :]
    )
