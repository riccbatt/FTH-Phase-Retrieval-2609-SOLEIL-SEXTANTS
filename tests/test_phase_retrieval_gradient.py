import unittest

import numpy as np

from library import phase_retrieval_gradient as grad


class PhaseRetrievalGradientTests(unittest.TestCase):
    def test_display_object_conversion_keeps_centered_support_centered(self):
        support = np.zeros((6, 8), dtype=float)
        support[2:4, 3:5] = 1.0

        field = grad.display_object_to_fourier_field(support)

        np.testing.assert_allclose(
            grad.fourier_field_to_object(field),
            np.fft.ifftshift(support),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            grad.fourier_field_to_display_object(field),
            support,
            atol=1e-12,
        )

    def test_support_loss_uses_core_centered_supportmask_convention(self):
        support = np.zeros((6, 8), dtype=float)
        support[2:4, 3:5] = 1.0
        field = grad.display_object_to_fourier_field(support)

        self.assertAlmostEqual(grad.support_loss(field, support), 0.0)

    def test_amplitude_gradient_step_reduces_diffraction_loss(self):
        rng = np.random.default_rng(5)
        shape = (16, 16)
        target_amplitude = rng.uniform(0.5, 2.0, shape)
        phase = np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
        field = 0.4 * target_amplitude * phase

        before = grad.diffraction_loss(field, target_amplitude)
        result = grad.refine_field_gradient(
            field,
            target_amplitude,
            n_steps=20,
            learning_rate=20.0,
            clip_update=None,
        )
        after = grad.diffraction_loss(result.fields, target_amplitude)

        self.assertLess(after, before)
        self.assertEqual(result.loss.shape, (20,))

    def test_gradient_refinement_accepts_step_schedules(self):
        rng = np.random.default_rng(8)
        shape = (6, 6)
        target_amplitude = rng.uniform(0.5, 2.0, shape)
        field = rng.normal(size=shape) + 1j * rng.normal(size=shape)

        result = grad.refine_field_gradient(
            field,
            target_amplitude,
            n_steps=3,
            learning_rate=np.array([0.1, 0.05, 0.02]),
            support_weight=np.array([0.0, 0.5, 1.0]),
            supportmask=np.ones(shape),
        )

        self.assertEqual(result.loss.shape, (3,))

    def test_fourier_projection_matches_amplitude_on_valid_pixels(self):
        rng = np.random.default_rng(6)
        shape = (10, 10)
        target_amplitude = rng.uniform(0.5, 2.0, shape)
        field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        mask = np.zeros(shape, dtype=int)
        mask[0, 0] = 1

        result = grad.refine_field_gradient(
            field,
            target_amplitude,
            mask_pixel=mask,
            n_steps=1,
            learning_rate=0.1,
            fourier_projection=True,
        )

        np.testing.assert_allclose(
            np.abs(result.fields[mask == 0]),
            target_amplitude[mask == 0],
            atol=1e-12,
        )

    def test_stack_refinement_returns_stack_history(self):
        rng = np.random.default_rng(7)
        fields = rng.normal(size=(3, 8, 8)) + 1j * rng.normal(size=(3, 8, 8))
        measurements = np.abs(fields) * 1.2
        support = np.ones((8, 8), dtype=bool)

        result = grad.refine_stack_gradient(
            fields,
            measurements,
            supportmask=support,
            n_steps=3,
            learning_rate=1.0,
        )

        self.assertEqual(result.fields.shape, fields.shape)
        self.assertEqual(result.loss.shape, (3, 3))
        self.assertEqual(result.diffraction_loss.shape, (3, 3))
        self.assertEqual(result.support_loss.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
