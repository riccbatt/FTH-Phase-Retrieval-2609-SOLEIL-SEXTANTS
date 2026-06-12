"""
Utilities for preparing X-ray optical constants and applying Kramers-Kronig.

This module contains no phase-retrieval code and performs no network access.
Henke/CXRO reference data must be supplied by the user as arrays or local text
files.

The refractive-index convention used here is

    n(E) = 1 - delta(E) + 1j * beta(E),

where beta >= 0 describes attenuation. With this convention,

    mu(E) = 4*pi*beta(E) / wavelength(E).
"""

import numpy as np


PLANCK_CONSTANT_EV_S = 4.135667696e-15
SPEED_OF_LIGHT_M_S = 299792458.0
CLASSICAL_ELECTRON_RADIUS_M = 2.8179403262e-15
HC_EV_M = PLANCK_CONSTANT_EV_S * SPEED_OF_LIGHT_M_S


def _validate_energy_axis(energy_ev, name="energy_ev", minimum_size=2):
    energy_ev = np.asarray(energy_ev, dtype=float)
    if energy_ev.ndim != 1 or energy_ev.size < minimum_size:
        raise ValueError(
            f"{name} must be a one-dimensional array with at least "
            f"{minimum_size} values."
        )
    if np.any(~np.isfinite(energy_ev)) or np.any(energy_ev <= 0):
        raise ValueError(f"{name} must contain finite, strictly positive values.")
    if np.any(np.diff(energy_ev) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return energy_ev


def _validate_spectrum(values, energy_ev, name):
    values = np.asarray(values, dtype=float)
    if values.shape != energy_ev.shape:
        raise ValueError(f"{name} must have the same shape as the energy axis.")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    return values


def absorption_to_beta(
    energy_ev,
    absorption,
    input_kind="absorption_coefficient",
    thickness_m=None,
):
    """
    Convert measured absorption data to the imaginary refractive index beta.

    Parameters
    ----------
    energy_ev : array, shape (nE,)
        Photon energy in electronvolts.
    absorption : array, shape (nE,)
        Data interpreted according to ``input_kind``.
    input_kind : {"absorption_coefficient", "optical_depth", "transmission"}
        ``absorption_coefficient`` expects mu in m^-1.
        ``optical_depth`` expects -ln(I/I0) = mu*t.
        ``transmission`` expects I/I0 and converts it to optical depth.
    thickness_m : float or None
        Sample thickness in metres. Required for optical depth or transmission.

    Returns
    -------
    beta : array, shape (nE,)
        Positive imaginary part in n = 1 - delta + 1j*beta.

    Notes
    -----
    An XAS spectrum in arbitrary units cannot be converted to absolute beta
    without an independent scale calibration.
    """
    energy_ev = _validate_energy_axis(energy_ev, minimum_size=1)
    absorption = _validate_spectrum(absorption, energy_ev, "absorption")
    kind = str(input_kind).lower()

    if kind in {"absorption_coefficient", "mu", "linear_attenuation"}:
        mu_m_inv = absorption
    elif kind in {"optical_depth", "absorbance_natural", "mu_t"}:
        if thickness_m is None or not np.isfinite(thickness_m) or thickness_m <= 0:
            raise ValueError("A finite positive thickness_m is required.")
        mu_m_inv = absorption / thickness_m
    elif kind in {"transmission", "i_over_i0"}:
        if thickness_m is None or not np.isfinite(thickness_m) or thickness_m <= 0:
            raise ValueError("A finite positive thickness_m is required.")
        if np.any(absorption <= 0) or np.any(absorption > 1):
            raise ValueError("transmission values must satisfy 0 < I/I0 <= 1.")
        mu_m_inv = -np.log(absorption) / thickness_m
    else:
        raise ValueError(
            "input_kind must be 'absorption_coefficient', 'optical_depth', "
            "or 'transmission'."
        )

    if np.any(mu_m_inv < 0):
        raise ValueError("The derived absorption coefficient must be non-negative.")
    wavelength_m = HC_EV_M / energy_ev
    return mu_m_inv * wavelength_m / (4.0 * np.pi)


def scattering_factor_f2_to_beta(energy_ev, f2, number_density_m3):
    """
    Convert atomic/molecular f2 values to beta for a known number density.

    For a mixture, supply the stoichiometrically summed f2 and the number
    density of formula units.
    """
    energy_ev = _validate_energy_axis(energy_ev, minimum_size=1)
    f2 = _validate_spectrum(f2, energy_ev, "f2")
    if not np.isfinite(number_density_m3) or number_density_m3 <= 0:
        raise ValueError("number_density_m3 must be finite and positive.")
    wavelength_m = HC_EV_M / energy_ev
    return (
        CLASSICAL_ELECTRON_RADIUS_M
        * wavelength_m**2
        * number_density_m3
        * f2
        / (2.0 * np.pi)
    )


def load_henke_optical_constants(
    path,
    energy_column=0,
    beta_column=2,
    delimiter=None,
    skiprows=0,
):
    """
    Load energy and beta columns from a local Henke/CXRO text export.

    Column numbering is zero based. The common ``energy, delta, beta`` layout
    is the default; override the column indices for a different export format.
    """
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    if data.ndim == 1:
        data = data[None, :]
    required_column = max(energy_column, beta_column)
    if data.ndim != 2 or data.shape[1] <= required_column:
        raise ValueError("The text file does not contain the requested columns.")
    energy_ev = _validate_energy_axis(data[:, energy_column])
    beta = _validate_spectrum(data[:, beta_column], energy_ev, "beta")
    if np.any(beta < 0):
        raise ValueError("Henke beta values must be non-negative.")
    return energy_ev, beta


def load_henke_refractive_index(
    path,
    energy_column=0,
    delta_column=1,
    beta_column=2,
    delimiter=None,
    skiprows=0,
):
    """
    Load energy, delta, and beta from a local Henke/CXRO text export.

    The default column order is ``energy_eV, delta, beta``. Both optical
    constants must follow this module's convention
    ``n = 1 - delta + 1j*beta``.
    """
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    if data.ndim == 1:
        data = data[None, :]
    required_column = max(energy_column, delta_column, beta_column)
    if data.ndim != 2 or data.shape[1] <= required_column:
        raise ValueError("The text file does not contain the requested columns.")

    energy_ev = _validate_energy_axis(data[:, energy_column])
    delta = _validate_spectrum(data[:, delta_column], energy_ev, "delta")
    beta = _validate_spectrum(data[:, beta_column], energy_ev, "beta")
    if np.any(beta < 0):
        raise ValueError("Henke beta values must be non-negative.")
    return energy_ev, delta, beta


def extend_beta_with_reference(
    measured_energy_ev,
    measured_beta,
    reference_energy_ev,
    reference_beta,
    output_energy_ev=None,
    transition_width_ev=0.0,
):
    """
    Extend measured beta with a broad-range reference spectrum.

    The measured values replace the reference inside the measured interval.
    A linear blend can be requested near both boundaries. No automatic scaling
    is applied: measured and reference beta must use the same absolute units.
    """
    measured_energy_ev = _validate_energy_axis(measured_energy_ev)
    measured_beta = _validate_spectrum(
        measured_beta, measured_energy_ev, "measured_beta"
    )
    reference_energy_ev = _validate_energy_axis(reference_energy_ev)
    reference_beta = _validate_spectrum(
        reference_beta, reference_energy_ev, "reference_beta"
    )
    if np.any(measured_beta < 0) or np.any(reference_beta < 0):
        raise ValueError("beta values must be non-negative.")
    if transition_width_ev < 0 or not np.isfinite(transition_width_ev):
        raise ValueError("transition_width_ev must be finite and non-negative.")
    if (
        measured_energy_ev[0] < reference_energy_ev[0]
        or measured_energy_ev[-1] > reference_energy_ev[-1]
    ):
        raise ValueError("The reference spectrum must span the measured spectrum.")

    if output_energy_ev is None:
        output_energy_ev = np.unique(
            np.concatenate([reference_energy_ev, measured_energy_ev])
        )
    else:
        output_energy_ev = _validate_energy_axis(output_energy_ev)
        if (
            output_energy_ev[0] < reference_energy_ev[0]
            or output_energy_ev[-1] > reference_energy_ev[-1]
        ):
            raise ValueError("output_energy_ev must lie inside the reference range.")

    extended = np.interp(output_energy_ev, reference_energy_ev, reference_beta)
    inside = (
        (output_energy_ev >= measured_energy_ev[0])
        & (output_energy_ev <= measured_energy_ev[-1])
    )
    measured_interp = np.interp(
        output_energy_ev[inside], measured_energy_ev, measured_beta
    )

    if transition_width_ev == 0:
        extended[inside] = measured_interp
        return output_energy_ev, extended

    distance_from_edge = np.minimum(
        output_energy_ev[inside] - measured_energy_ev[0],
        measured_energy_ev[-1] - output_energy_ev[inside],
    )
    measured_weight = np.clip(distance_from_edge / transition_width_ev, 0.0, 1.0)
    extended[inside] = (
        measured_weight * measured_interp
        + (1.0 - measured_weight) * extended[inside]
    )
    return output_energy_ev, extended


def refractive_index_from_beta_with_reference(
    measured_energy_ev,
    measured_beta,
    reference_energy_ev,
    reference_delta,
    reference_beta,
    transition_width_ev=0.0,
    return_extended=False,
):
    """
    Merge measured beta into a reference and update delta by differential KK.

    This implements the near-edge strategy described by Cross et al. and Watts:

    1. Keep the broad-range reference ``delta`` and ``beta`` as the baseline.
    2. Form the localized correction ``measured_beta - reference_beta``.
    3. Kramers-Kronig transform only that correction.
    4. Add the resulting delta correction to the reference delta.

    Transforming only the local correction avoids treating a finite Henke table
    as though beta were exactly zero outside its tabulated range. The correction
    is linearly blended to zero at the measurement boundaries when
    ``transition_width_ev > 0``.

    Returns
    -------
    beta, delta : arrays, shape (nE,)
        Corrected optical constants at the measured energies.
    extended_energy, extended_beta, beta_correction : arrays, optional
        Also returned when ``return_extended=True``.
    """
    measured_energy_ev = _validate_energy_axis(measured_energy_ev)
    measured_beta = _validate_spectrum(
        measured_beta, measured_energy_ev, "measured_beta"
    )
    reference_energy_ev = _validate_energy_axis(reference_energy_ev)
    reference_delta = _validate_spectrum(
        reference_delta, reference_energy_ev, "reference_delta"
    )
    reference_beta = _validate_spectrum(
        reference_beta, reference_energy_ev, "reference_beta"
    )
    if np.any(measured_beta < 0) or np.any(reference_beta < 0):
        raise ValueError("beta values must be non-negative.")

    extended_energy, extended_beta = extend_beta_with_reference(
        measured_energy_ev,
        measured_beta,
        reference_energy_ev,
        reference_beta,
        transition_width_ev=transition_width_ev,
    )
    reference_beta_extended = np.interp(
        extended_energy, reference_energy_ev, reference_beta
    )
    beta_correction = extended_beta - reference_beta_extended

    delta_correction = beta_to_delta(
        extended_energy,
        beta_correction,
        evaluation_energy_ev=measured_energy_ev,
    )
    delta = (
        np.interp(measured_energy_ev, reference_energy_ev, reference_delta)
        + delta_correction
    )
    beta = np.interp(measured_energy_ev, extended_energy, extended_beta)

    if return_extended:
        return beta, delta, extended_energy, extended_beta, beta_correction
    return beta, delta


def kramers_kronig_real_from_imaginary(
    integration_energy_ev,
    imaginary_part,
    evaluation_energy_ev=None,
    subtract_baseline=False,
    normalize_input=False,
    output_mean=None,
):
    """
    Calculate the real response from its imaginary part.

    This uses the exact piecewise-linear specialization of the piecewise
    Laurent-polynomial method of B. Watts, Opt. Express 22, 23628 (2014):

        real(E) = (2/pi) P integral x*imaginary(x)/(x^2-E^2) dx.

    ``integration_energy_ev`` may be a broad Henke-extended grid while
    ``evaluation_energy_ev`` contains only the experimental energies.
    """
    energy = _validate_energy_axis(integration_energy_ev)
    values = _validate_spectrum(imaginary_part, energy, "imaginary_part").copy()

    if evaluation_energy_ev is None:
        evaluation_energy = energy
    else:
        evaluation_energy = np.asarray(evaluation_energy_ev, dtype=float)
        if evaluation_energy.ndim != 1 or evaluation_energy.size == 0:
            raise ValueError("evaluation_energy_ev must be a non-empty 1D array.")
        if np.any(~np.isfinite(evaluation_energy)) or np.any(evaluation_energy <= 0):
            raise ValueError(
                "evaluation_energy_ev must contain finite positive values."
            )
        if (
            np.any(evaluation_energy < energy[0])
            or np.any(evaluation_energy > energy[-1])
        ):
            raise ValueError(
                "evaluation_energy_ev must lie inside the integration range."
            )

    if subtract_baseline:
        baseline = values[0] + (values[-1] - values[0]) * (
            energy - energy[0]
        ) / (energy[-1] - energy[0])
        values -= baseline
    if normalize_input:
        scale = np.max(np.abs(values))
        if np.isfinite(scale) and scale > 0:
            values /= scale

    slopes = np.diff(values) / np.diff(energy)
    intercepts = values[:-1] - slopes * energy[:-1]
    interval_widths = np.diff(energy)
    result = np.empty_like(evaluation_energy)

    endpoint_scale = max(1.0, float(np.max(np.abs(values))))
    endpoint_tolerance = 100 * np.finfo(float).eps * endpoint_scale

    for i, target in enumerate(evaluation_energy):
        if (
            np.isclose(target, energy[0], rtol=0, atol=endpoint_tolerance)
            and abs(values[0]) > endpoint_tolerance
        ) or (
            np.isclose(target, energy[-1], rtol=0, atol=endpoint_tolerance)
            and abs(values[-1]) > endpoint_tolerance
        ):
            raise ValueError(
                "Evaluation at a finite integration endpoint requires the "
                "imaginary part to be zero there."
            )

        singular_coeff = np.empty_like(energy)
        left_coeff = 0.5 * (slopes * target + intercepts)
        singular_coeff[0] = -left_coeff[0]
        singular_coeff[1:-1] = left_coeff[:-1] - left_coeff[1:]
        singular_coeff[-1] = left_coeff[-1]

        distance = np.abs(energy - target)
        nonzero_distance = distance > 0
        singular_log_sum = np.sum(
            singular_coeff[nonzero_distance]
            * np.log(distance[nonzero_distance])
        )

        regular_coeff = 0.5 * (intercepts - slopes * target)
        regular_log_sum = np.sum(
            regular_coeff
            * np.log((energy[1:] + target) / (energy[:-1] + target))
        )
        integral = (
            np.sum(slopes * interval_widths)
            + singular_log_sum
            + regular_log_sum
        )
        result[i] = (2.0 / np.pi) * integral

    if output_mean is not None:
        if not np.isfinite(output_mean):
            raise ValueError("output_mean must be finite or None.")
        result = result - np.mean(result) + output_mean
    return result


def beta_to_delta(
    integration_energy_ev,
    beta,
    evaluation_energy_ev=None,
    subtract_baseline=False,
    normalize_input=False,
    output_mean=None,
):
    """
    Calculate delta from beta for n = 1 - delta + 1j*beta.

    For an absolute calculation, provide a broad, physically extended beta
    spectrum and leave ``subtract_baseline=False`` and ``output_mean=None``.
    """
    real_response = kramers_kronig_real_from_imaginary(
        integration_energy_ev,
        beta,
        evaluation_energy_ev=evaluation_energy_ev,
        subtract_baseline=subtract_baseline,
        normalize_input=normalize_input,
        output_mean=output_mean,
    )
    return -real_response


# Backward-compatible descriptive alias.
discrete_kramers_kronig_dispersion = kramers_kronig_real_from_imaginary
