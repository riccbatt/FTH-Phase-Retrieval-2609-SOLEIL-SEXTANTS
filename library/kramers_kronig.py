"""Piecewise-linear Kramers–Kronig integral for fitted complex responses."""
import numpy as np

__all__ = ['kramers_kronig_real_from_imaginary']


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

    ``integration_energy_ev`` may be a broader integration grid while
    ``evaluation_energy_ev`` contains only the experimental energies.
    """
    energy = np.asarray(integration_energy_ev, float)
    values = np.array(imaginary_part, dtype=float, copy=True)
    if energy.ndim != 1 or len(energy) < 2 or np.any(~np.isfinite(energy) | (energy <= 0)) or np.any(np.diff(energy) <= 0):
        raise ValueError('KK energies must be finite, positive and strictly increasing, with at least two samples.')
    if values.shape != energy.shape or np.any(~np.isfinite(values)):
        raise ValueError('KK response must be finite and match the energy axis.')

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
