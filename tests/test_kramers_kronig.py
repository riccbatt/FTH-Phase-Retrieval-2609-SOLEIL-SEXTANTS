import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy import integrate

from library import kramers_kronig as kk


class KramersKronigTests(unittest.TestCase):
    def test_absorption_coefficient_to_beta(self):
        energy = np.array([500.0, 1000.0])
        mu = np.array([2.0e6, 3.0e6])

        beta = kk.absorption_to_beta(energy, mu)

        expected = mu * kk.HC_EV_M / energy / (4.0 * np.pi)
        np.testing.assert_allclose(beta, expected)

    def test_transmission_and_optical_depth_give_same_beta(self):
        energy = np.array([700.0, 710.0, 720.0])
        thickness = 80e-9
        optical_depth = np.array([0.2, 0.5, 0.3])
        transmission = np.exp(-optical_depth)

        from_depth = kk.absorption_to_beta(
            energy,
            optical_depth,
            input_kind="optical_depth",
            thickness_m=thickness,
        )
        from_transmission = kk.absorption_to_beta(
            energy,
            transmission,
            input_kind="transmission",
            thickness_m=thickness,
        )

        np.testing.assert_allclose(from_depth, from_transmission)

    def test_load_henke_optical_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "henke.txt"
            np.savetxt(
                path,
                np.array(
                    [
                        [100.0, 1.0e-3, 2.0e-4],
                        [200.0, 8.0e-4, 1.0e-4],
                    ]
                ),
            )

            energy, beta = kk.load_henke_optical_constants(path)

        np.testing.assert_allclose(energy, [100.0, 200.0])
        np.testing.assert_allclose(beta, [2.0e-4, 1.0e-4])

    def test_load_henke_refractive_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "henke.txt"
            np.savetxt(
                path,
                np.array(
                    [
                        [100.0, 1.0e-3, 2.0e-4],
                        [200.0, 8.0e-4, 1.0e-4],
                    ]
                ),
            )

            energy, delta, beta = kk.load_henke_refractive_index(path)

        np.testing.assert_allclose(energy, [100.0, 200.0])
        np.testing.assert_allclose(delta, [1.0e-3, 8.0e-4])
        np.testing.assert_allclose(beta, [2.0e-4, 1.0e-4])

    def test_extend_beta_uses_measurement_inside_reference_range(self):
        reference_energy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        reference_beta = np.ones(5)
        measured_energy = np.array([2.0, 3.0, 4.0])
        measured_beta = np.array([2.0, 3.0, 2.0])

        energy, beta = kk.extend_beta_with_reference(
            measured_energy,
            measured_beta,
            reference_energy,
            reference_beta,
        )

        np.testing.assert_allclose(energy, reference_energy)
        np.testing.assert_allclose(beta, [1.0, 2.0, 3.0, 2.0, 1.0])

    def test_piecewise_linear_kk_matches_principal_value_quadrature(self):
        energies = np.array([1.0, 1.7, 2.8, 4.6, 7.0])
        imaginary = np.array([0.0, 1.2, -0.3, 0.8, 0.0])

        calculated = kk.kramers_kronig_real_from_imaginary(
            energies,
            imaginary,
            output_mean=0.0,
        )

        expected = []
        for target in energies:
            if target in {energies[0], energies[-1]}:
                def integrand(x):
                    if x == target:
                        return 0.0
                    return (
                        x
                        * np.interp(x, energies, imaginary)
                        / (x**2 - target**2)
                    )

                value = integrate.quad(
                    integrand,
                    energies[0],
                    energies[-1],
                    points=energies,
                    limit=200,
                )[0]
            else:
                def numerator(x):
                    return x * np.interp(x, energies, imaginary) / (x + target)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", integrate.IntegrationWarning)
                    value = integrate.quad(
                        numerator,
                        energies[0],
                        energies[-1],
                        weight="cauchy",
                        wvar=target,
                        limit=200,
                    )[0]
            expected.append((2.0 / np.pi) * value)

        expected = np.asarray(expected)
        expected -= np.mean(expected)
        np.testing.assert_allclose(calculated, expected, atol=2e-8)

    def test_beta_to_delta_has_opposite_kk_sign(self):
        energy = np.array([1.0, 2.0, 3.0, 4.0])
        beta = np.array([0.0, 1.0, 0.5, 0.0])

        real_response = kk.kramers_kronig_real_from_imaginary(energy, beta)
        delta = kk.beta_to_delta(energy, beta)

        np.testing.assert_allclose(delta, -real_response)

    def test_reference_method_returns_reference_when_beta_is_unchanged(self):
        reference_energy = np.array([100.0, 500.0, 700.0, 710.0, 720.0, 1000.0])
        reference_beta = np.array([1e-5, 2e-5, 3e-4, 5e-4, 2e-4, 1e-5])
        reference_delta = np.array([2e-3, 1e-3, 8e-4, 7e-4, 6e-4, 4e-4])
        measured_energy = reference_energy[2:5]
        measured_beta = reference_beta[2:5]

        beta, delta = kk.refractive_index_from_beta_with_reference(
            measured_energy,
            measured_beta,
            reference_energy,
            reference_delta,
            reference_beta,
        )

        np.testing.assert_allclose(beta, measured_beta)
        np.testing.assert_allclose(delta, reference_delta[2:5])

    def test_reference_method_adds_only_differential_kk_correction(self):
        reference_energy = np.array([100.0, 500.0, 700.0, 710.0, 720.0, 1000.0])
        reference_beta = np.array([1e-5, 2e-5, 3e-4, 5e-4, 2e-4, 1e-5])
        reference_delta = np.array([2e-3, 1e-3, 8e-4, 7e-4, 6e-4, 4e-4])
        measured_energy = reference_energy[2:5]
        measured_beta = reference_beta[2:5] + np.array([0.0, 2e-4, 0.0])

        beta, delta, extended_energy, _, correction = (
            kk.refractive_index_from_beta_with_reference(
                measured_energy,
                measured_beta,
                reference_energy,
                reference_delta,
                reference_beta,
                return_extended=True,
            )
        )
        expected_delta = reference_delta[2:5] + kk.beta_to_delta(
            extended_energy,
            correction,
            evaluation_energy_ev=measured_energy,
        )

        np.testing.assert_allclose(beta, measured_beta)
        np.testing.assert_allclose(delta, expected_delta)

    def test_extended_grid_can_be_evaluated_only_at_experimental_energies(self):
        integration_energy = np.array([10.0, 100.0, 700.0, 710.0, 720.0, 1.0e4])
        beta = np.array([1e-5, 2e-5, 3e-4, 8e-4, 2e-4, 1e-8])
        experimental_energy = np.array([700.0, 710.0, 720.0])

        delta = kk.beta_to_delta(
            integration_energy,
            beta,
            evaluation_energy_ev=experimental_energy,
        )

        self.assertEqual(delta.shape, experimental_energy.shape)
        self.assertTrue(np.all(np.isfinite(delta)))
