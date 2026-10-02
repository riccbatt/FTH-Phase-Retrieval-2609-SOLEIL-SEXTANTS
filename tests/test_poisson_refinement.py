import unittest
import numpy as np
from library import poisson_refinement as p
from library import phase_retrieval_geometry as g
from library import phase_retrieval_universal as u


class PoissonTests(unittest.TestCase):
    def test_zero_evidence_masks_and_fixed_secondary_mode(self):
        rng = np.random.default_rng(2)
        field = rng.normal(size=(2, 12, 12)) + 1j*rng.normal(size=(2, 12, 12))
        measured = np.zeros((12, 12)); measured[5:7, 5:7] = 2
        mask = np.zeros_like(measured); mask[0, 0] = 1
        result = p.refine_poisson(field, measured, np.ones_like(field.real), mask, steps=25, refine_modes=[0])
        np.testing.assert_array_equal(result.fields[1], field[1])
        self.assertTrue(np.all(np.diff(result.loss) <= 1e-12))
        self.assertLess(result.loss[-1], result.baseline_loss)
        self.assertEqual(result.zero_pixels, 139)
        altered = measured.copy(); altered[0, 0] = 1e12
        other = p.refine_poisson(field, altered, np.ones_like(field.real), mask, steps=25, refine_modes=[0])
        np.testing.assert_allclose(result.fields, other.fields)

    def test_blur_adjoint_and_directional_gradient(self):
        rng = np.random.default_rng(4)
        shape = (2, 10, 10)
        gamma = rng.uniform(.1, 1, size=shape)
        kfft = p._kernel_fft(gamma, shape)
        a, b = rng.normal(size=shape), rng.normal(size=shape)
        self.assertAlmostEqual(np.sum(p._blur(a, kfft)*b), np.sum(a*p._blur(b, kfft, True)), places=10)
        field = rng.normal(size=shape) + 1j*rng.normal(size=shape)
        direction = rng.normal(size=shape) + 1j*rng.normal(size=shape)
        y = rng.poisson(2, size=shape[-2:])
        def loss(f):
            mu = p.predicted_intensity(f, gamma) + 1e-10
            return np.sum(mu-y*np.log(mu))
        mu = p.predicted_intensity(field, gamma) + 1e-10
        grad = field*p._blur(np.broadcast_to(1-y/mu, shape), kfft, True)
        finite = (loss(field+1e-6*direction)-loss(field-1e-6*direction))/2e-6
        self.assertAlmostEqual(finite, 2*np.real(np.vdot(grad, direction)), places=5)
        result = p.refine_poisson(field, y, np.ones(shape), gamma=gamma, steps=15)
        self.assertLess(result.loss[-1], result.baseline_loss)

    def test_support_and_zero_steps(self):
        rng = np.random.default_rng(5)
        f = rng.normal(size=(16, 16)) + 1j*rng.normal(size=(16, 16))
        support = np.zeros((16, 16)); support[5:11, 5:11] = 1
        y = np.ones_like(support)
        np.testing.assert_array_equal(p.refine_poisson(f, y, support, steps=0).fields, f)
        r = p.refine_poisson(f, y, support, steps=10)
        obj = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(r.fields)))
        np.testing.assert_allclose(obj[support == 0], 0, atol=1e-12)
        self.assertTrue(np.all(np.diff(r.loss) <= 1e-12))

    def test_odd_grid_support_convention(self):
        rng = np.random.default_rng(12)
        field = rng.normal(size=(15, 15)) + 1j*rng.normal(size=(15, 15))
        support = np.zeros((15, 15)); support[4:10, 5:12] = 1
        result = p.refine_poisson(field, np.ones((15, 15)), support, steps=3)
        displayed = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(result.fields)))
        np.testing.assert_allclose(displayed[support == 0], 0, atol=1e-11)

    def test_object_roi_preserves_outside_and_relaxation(self):
        rng = np.random.default_rng(6)
        logs = rng.normal(size=(3, 12, 12)) + 1j*rng.normal(size=(3, 12, 12))
        thickness = np.zeros((12, 12)); thickness[:3, :4] = 1
        recipe = u.default_general_phase_retrieval_recipe()
        recipe['physical_projection_object_roi'] = True
        args = (logs, ['sat', 'a', 'b'], [783]*3, [1]*3, ['beam']*3)
        out, components = u.project_log_objects_physical(*args, recipe=recipe,
            material_thickness=thickness, relaxation=.1, saturated_states={'sat': 1},
            iterations=2, return_components=True)
        full = u.project_log_objects_physical(*args, recipe=recipe,
            material_thickness=thickness, relaxation=1, saturated_states={'sat': 1}, iterations=2)
        np.testing.assert_array_equal(out[:, thickness == 0], logs[:, thickness == 0])
        np.testing.assert_allclose(out[:, thickness > 0], (.9*logs+.1*full)[:, thickness > 0])
        self.assertEqual(components['magnetization'].shape, logs.shape)
        self.assertEqual(components['physical_projection_fit_pixels'], 12)

    def test_roi_phase_reference_leaves_vacuum_logs_unchanged(self):
        logs = np.ones((2, 8, 8), complex)
        logs[0] += 3.1j
        logs[1] -= 3.1j
        thickness = np.zeros((8, 8)); thickness[3:5, 3:5] = 1
        recipe = u.default_general_phase_retrieval_recipe()
        recipe.update(physical_projection_object_roi=True, physical_phase_reference=True)
        out, components = u.project_log_objects_physical(
            logs, ['a', 'b'], [783]*2, [1, -1], ['beam']*2,
            recipe=recipe, material_thickness=thickness, iterations=2,
            return_components=True)
        np.testing.assert_array_equal(out[:, thickness == 0], logs[:, thickness == 0])
        self.assertEqual(components['physical_projection_fit_pixels'], 4)

    def test_reference_phase_branch_preserves_an_exact_physical_model(self):
        y, x = np.indices((32, 32))
        aperture = (abs(x-16)<7) & (abs(y-16)<7)
        phase = 3.1 + .025*(x-16)
        objects = np.stack([np.where(aperture, np.exp(1j*(phase+m*.07)), 1e-12)
                            for m in [1, 0, -1]])
        logs = np.log(abs(objects)) + 1j*np.angle(objects)
        original = logs.copy()
        recipe = u.default_general_phase_retrieval_recipe()
        recipe.update(physical_projection_object_roi=True, physical_phase_reference=True)
        fitted = u.project_log_objects_physical(
            logs, ['sat','a','b'], [783]*3, [1]*3, ['beam']*3,
            material_thickness=aperture, recipe=recipe, saturated_states={'sat':1},
            iterations=10, relaxation=.15,
        )
        np.testing.assert_array_equal(logs, original)
        np.testing.assert_allclose(np.exp(fitted), objects, atol=1e-12)

    def test_roi_tracks_geometry(self):
        source = np.zeros((64, 64)); source[20:44, 24:40] = 1
        for crop, binning in ((0, 1), (8, 1), (8, 2)):
            recipe = dict(crop=crop, binning=binning, roi=None)
            images, mask, support, _, geo = g.prepare(np.ones((2, 64, 64)), np.zeros((64,64)), source, recipe)
            roi = g.support_bounding_roi(support)
            self.assertGreater(support[roi].sum(), 0)
            self.assertEqual(support[roi].sum(), support.sum())
            self.assertLessEqual(roi[0].stop, images.shape[-2])
