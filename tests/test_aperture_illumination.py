import unittest
import numpy as np
from library import phase_retrieval_universal as u


class ApertureIlluminationTests(unittest.TestCase):
    def setUp(self):
        self.support = np.zeros((24, 24))
        self.material = self.support.copy()
        self.material[8:16, 8:16] = 1.
        self.support[8:16, 8:16] = 1.
        for row, col in [(2,2), (2,20), (20,2), (20,20), (12,2)]:
            self.support[row:row+2, col:col+2] = 1.
        self.signs = np.array([1., .35, -1.])
        self.recipe = u.default_universal_phase_retrieval_recipe()
        self.recipe.update(align_global_phase=True, fit_aperture_illumination=True,
                           physical_projection_object_roi=True, clip_magnetization=False)
        self.logs = .3+.04j + self.signs[:,None,None]*np.fft.fftshift(self.material)*(.07+.03j)
        self.fields = u.object_log_to_fourier_field(self.logs)
        self.kwargs = dict(recipe=self.recipe, material_thickness=np.fft.fftshift(self.material),
                           projection_supportmask=np.fft.fftshift(self.support),
                           saturated_states={'p':1., 'n':-1.}, physical_iterations=100,
                           relaxation=1., return_components=True)

    def project(self, fields, initial=None):
        return u._project_physical_modes(fields, ['p','free','n'], [781.]*3,
            [1.]*3, ['beam']*3, initial_components=initial, **self.kwargs)

    def test_global_phase_gauge_does_not_change_magnetization_or_intensity(self):
        self.recipe.update(fit_aperture_illumination=False, fit_illumination_intensity=False)
        phases = np.array([.4, 2.7, -2.2])
        observed = self.fields*np.exp(1j*phases)[:,None,None]
        output, components = self.project(observed)
        np.testing.assert_allclose(output, observed, atol=1e-10)
        np.testing.assert_allclose(np.exp(1j*components['global_phase_offsets_rad']),
                                   np.exp(1j*(phases-phases[0])), atol=1e-10)
        np.testing.assert_allclose(components['magnetization'][:,np.fft.fftshift(self.material)>0],
                                   np.broadcast_to(self.signs[:,None], (3,64)), atol=1e-8)
        _, continued = self.project(observed, components)
        self.assertTrue(continued['physical_fit_reused'])
        np.testing.assert_allclose(continued['magnetization'], components['magnetization'], atol=1e-8)

    def test_six_aperture_gains_and_pooled_object_preserve_magnetic_signal(self):
        from scipy.ndimage import label
        labels, count = label(self.support)
        self.assertEqual(count, 6)
        labels = np.fft.ifftshift(labels)
        ref_ids = [i for i in range(1,7) if not np.any(self.material[np.fft.fftshift(labels)==i]>0)]
        gains = np.array([[1.]*5, [1.1,.9,1.05,.95,1.2], [.8,1.2,.9,1.1,1.]])
        phases = np.array([[0.]*5, [1.9,2.,1.8,2.1,1.95], [-2.2,-2.1,-2.3,-2.,-2.15]])
        factors = gains*np.exp(1j*phases)
        pooled = np.sqrt(np.mean(abs(factors)**2,axis=1))*np.exp(1j*np.angle(factors.sum(axis=1)))
        factor_map = np.broadcast_to(pooled[:,None,None], self.logs.shape).copy()
        for k, aperture in enumerate(ref_ids):
            factor_map[:,labels==aperture] = factors[:,k,None]
        observed_objects = np.exp(self.logs)*factor_map
        observed = np.fft.ifftshift(np.fft.ifft2(observed_objects),axes=(-2,-1))
        output, c = self.project(observed)
        np.testing.assert_allclose(output, observed, atol=1e-8)
        self.assertEqual(c['aperture_illumination_factors'].shape, (3,6))
        for k, aperture in enumerate(ref_ids):
            np.testing.assert_allclose(c['aperture_illumination_factors'][:,aperture-1], factors[:,k])
        object_id = c['aperture_illumination_kinds'].index('object_pooled')
        np.testing.assert_allclose(c['aperture_illumination_factors'][:,object_id], pooled)
        np.testing.assert_allclose(c['magnetization'][:,np.fft.fftshift(self.material)>0],
                                   np.broadcast_to(self.signs[:,None], (3,64)), atol=1e-8)

    def test_rh_only_keeps_oh_amplitude_fixed_and_phase_alignment_is_optional(self):
        self.recipe.update(object_illumination_mode='none', align_global_phase=True)
        phases = np.array([0., 1.8, -2.])
        gains = np.array([1., 1.3, .8])
        material = np.fft.fftshift(self.material)>0
        reference = (np.fft.fftshift(self.support)!=0)&~material
        objects = np.exp(self.logs)*np.exp(1j*phases)[:,None,None]
        objects[:,reference] *= gains[:,None]
        fields = np.fft.ifftshift(np.fft.ifft2(objects),axes=(-2,-1))
        output, c = self.project(fields)
        np.testing.assert_allclose(output, fields, atol=1e-8)
        np.testing.assert_allclose(c['illumination_intensity_factors'], 1.)
        oh_id = c['aperture_illumination_kinds'].index('object_none')
        np.testing.assert_allclose(c['aperture_illumination_factors'][:,oh_id], np.exp(1j*phases))
        np.testing.assert_allclose(c['magnetization'][:,material],
                                   np.broadcast_to(self.signs[:,None], (3,64)), atol=1e-8)
        # The notebook endpoint diagnostic must honor RH-only instead of
        # accidentally applying the pooled RH amplitude to the OH.
        import ast
        import json
        from pathlib import Path
        notebook = json.loads((Path(__file__).resolve().parents[1]/
                              '2601_XPCS/01_physical_hysteresis_263.ipynb').read_text())
        source = next(''.join(cell['source']) for cell in notebook['cells']
                      if 'def illumination_corrected_diagnostic_objects' in ''.join(cell.get('source', [])))
        node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                    and node.name == 'illumination_corrected_diagnostic_objects')
        ns = dict(np=np, universal=u, physical_recipe=self.recipe, ENERGY_EV=781.,
                  thickness_used=self.material, support_used=self.support,
                  mode_stack=lambda value: value[:,None] if value.ndim==3 else value,
                  display_objects=lambda value, stage: u.fourier_field_to_display_object(value))
        stage = dict(self.recipe, projection_focus_prop_um=0., projection_focus_phase_rad=0.,
                     projection_focus_setup=None)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'diagnostic', 'exec'), ns)
        diagnostic = ns['illumination_corrected_diagnostic_objects'](fields, stage)
        np.testing.assert_allclose(diagnostic, np.fft.fftshift(np.exp(self.logs),axes=(-2,-1)), atol=1e-10)
        self.recipe.update(align_global_phase=False, aperture_factor_type='amplitude')
        normalized, restore, details = u._reference_illumination_correction(
            fields, [781.]*3, ['beam']*3, self.recipe,
            self.kwargs['material_thickness'], self.kwargs['projection_supportmask'])
        self.assertFalse(details['global_phase_aligned'])
        np.testing.assert_array_equal(details['global_phase_offsets_rad'], 0.)
        norm_objects = np.fft.fft2(np.fft.fftshift(normalized,axes=(-2,-1)))
        np.testing.assert_allclose(norm_objects[:,material], objects[:,material], atol=1e-10)
        np.testing.assert_allclose(details['aperture_illumination_factors'][:,oh_id], 1.)
        np.testing.assert_allclose(restore(normalized), fields, atol=1e-10)

    def test_distance_weighting_and_nearest_use_centered_aperture_positions(self):
        from scipy.ndimage import label
        labels, count = label(self.support)
        labels = np.fft.ifftshift(labels)
        object_id = next(i for i in range(1,count+1)
                         if np.any(self.kwargs['material_thickness'][labels==i]>0))
        ref_ids = [i for i in range(1,count+1) if i!=object_id]
        # Different gains near +/-pi test gauge-relative interpolation too.
        ref_factors = np.array([.7, .9, 1.4, 1.1, 1.8])*np.exp(1j*np.array([3.05,-3.08,3.1,-3.12,3.]))
        objects = np.exp(self.logs)
        for k, aperture in enumerate(ref_ids):
            objects[1, labels==aperture] *= ref_factors[k]
        fields = np.fft.ifftshift(np.fft.ifft2(objects),axes=(-2,-1))
        for mode in ('distance_weighted', 'nearest'):
            self.recipe['object_illumination_mode'] = mode
            _, _, c = u._reference_illumination_correction(
                fields, [781.]*3, ['beam']*3, self.recipe,
                self.kwargs['material_thickness'], self.kwargs['projection_supportmask'])
            centers = c['aperture_centroids_yx']
            distance = np.linalg.norm(centers[np.array(ref_ids)-1]-centers[object_id-1],axis=1)
            expected_weights = 1./np.maximum(distance,1.)**2 if mode=='distance_weighted' else (distance==distance.min()).astype(float)
            expected_weights /= expected_weights.sum()
            np.testing.assert_allclose(c['object_reference_weights'][0], expected_weights)
            phase = c['global_phase_offsets_rad'][1]
            expected = np.exp(np.sum(expected_weights*(np.log(abs(ref_factors))
                              +1j*np.angle(ref_factors*np.exp(-1j*phase))))+1j*phase)
            np.testing.assert_allclose(c['aperture_illumination_factors'][1,object_id-1],expected)
            self.assertGreater(abs(expected), .5)

    def test_linear_plane_recovers_object_gain_and_magnetization(self):
        from scipy.ndimage import label
        labels, count = label(self.support)
        centers = np.asarray([np.argwhere(labels == k).mean(axis=0) for k in range(1,count+1)])
        labels = np.fft.ifftshift(labels)
        object_id = next(k for k in range(1,count+1)
                         if np.any(self.kwargs['material_thickness'][labels==k]>0))
        offsets = centers-centers[object_id-1]
        intercepts = np.array([0., .18+2.95j, -.12-2.8j])
        slopes = np.array([[0.,0.], [.012+.015j,-.007+.011j], [-.01+.012j,.008-.014j]])
        factors = np.exp(intercepts[:,None]+slopes@offsets.T)
        objects = np.exp(self.logs)
        for k in range(1,count+1):
            objects[:,labels==k] *= factors[:,k-1,None]
        fields = np.fft.ifftshift(np.fft.ifft2(objects),axes=(-2,-1))
        self.recipe['object_illumination_mode'] = 'linear_plane'
        output, c = self.project(fields)
        np.testing.assert_allclose(c['aperture_illumination_factors'], factors, atol=1e-10)
        np.testing.assert_allclose(output, fields, atol=1e-8)
        np.testing.assert_allclose(c['magnetization'][:,self.kwargs['material_thickness']>0],
                                  np.broadcast_to(self.signs[:,None], (3,64)), atol=1e-8)
        refs = np.asarray(c['reference_aperture_ids'])-1
        weights = c['object_reference_weights'][0]
        np.testing.assert_allclose(weights.sum(), 1.)
        np.testing.assert_allclose(weights@offsets[refs], 0., atol=1e-12)

    def test_linear_plane_rejects_collinear_reference_positions(self):
        self.recipe['object_illumination_mode'] = 'linear_plane'
        support = self.material.copy()
        support[2:4,2:4] = 1.
        support[2:4,10:12] = 1.
        support[2:4,20:22] = 1.
        with self.assertRaisesRegex(ValueError, 'non-collinear'):
            u._reference_illumination_correction(self.fields, [781.]*3, ['beam']*3,
                self.recipe, self.kwargs['material_thickness'], np.fft.fftshift(support))

    def test_primary_reference_profiles_are_constrained_even_with_unit_factors(self):
        self.recipe['object_illumination_mode'] = 'none'
        objects = np.exp(self.logs)
        reference_pixels = np.argwhere((np.fft.fftshift(self.support)!=0)
                                       & (np.fft.fftshift(self.material)==0))
        # Zero-mean shape error leaves the aperture scalar at exactly one.
        y, x = reference_pixels[0]
        yy, xx = reference_pixels[1]
        objects[1,y,x] += .08+.03j
        objects[1,yy,xx] -= .08+.03j
        fields = np.fft.ifftshift(np.fft.ifft2(objects),axes=(-2,-1))
        output, c = self.project(fields)
        self.assertTrue(c['reference_aperture_common_profile'])
        self.assertGreater(c['reference_aperture_profile_residual_rms'], 0.)
        np.testing.assert_allclose(c['aperture_illumination_factors'], 1., atol=1e-12)
        output_objects = np.fft.fft2(np.fft.fftshift(output,axes=(-2,-1)))
        reference = (np.fft.fftshift(self.support)!=0)&(np.fft.fftshift(self.material)==0)
        np.testing.assert_allclose(output_objects[:,reference],
                                   np.broadcast_to(output_objects[0,reference], (3,20)), atol=1e-10)
        self.assertGreater(np.linalg.norm(output-fields), 1e-6)
        self.kwargs['relaxation'] = 0.
        bypass, _ = self.project(fields)
        np.testing.assert_allclose(bypass, fields, atol=1e-10)

    def test_all_corrections_disabled_bypass_reference_estimation(self):
        from unittest.mock import patch
        self.recipe.update(align_global_phase=False, fit_aperture_illumination=False,
                           fit_illumination_intensity=False)
        with patch.object(u, '_reference_illumination_correction') as correction:
            _, c = self.project(self.fields)
            correction.assert_not_called()
        self.assertNotIn('aperture_illumination_factors', c)
        self.assertNotIn('global_phase_offsets_rad', c)

    def test_driver_exposes_phase_and_six_aperture_factors(self):
        import contextlib
        import io
        recipe = dict(self.recipe, crop=0, binning=1, modes=[1],
                      material_thickness=self.material,
                      projection_constraints_inside_support_only=True,
                      warmup_Nit=0, outer_iterations=0, final_fourier_constraint=False,
                      saturated_states={'p':1., 'n':-1.})
        observed = self.fields*np.exp(1j*np.array([0.,2.1,-2.]))[:,None,None]
        with contextlib.redirect_stdout(io.StringIO()):
            output, _, c, _, _ = u.universal_phase_retrieval_algorithm(
                abs(observed)**2, np.zeros_like(observed.real), self.support,
                ['p','free','n'], [781.]*3, [1.]*3, ['beam']*3,
                universal_recipe=recipe, start_fields=observed)
        self.assertTrue(c['global_phase_aligned'])
        self.assertEqual(c['aperture_illumination_factors'].shape, (3,6))
        np.testing.assert_allclose(output, observed, atol=1e-8)

    def test_reference_correction_requires_measurable_phase_overlap(self):
        objects = np.exp(self.logs)
        reference = (np.fft.fftshift(self.support)!=0)&(np.fft.fftshift(self.material)==0)
        objects[1,reference] = 0.
        fields = np.fft.ifftshift(np.fft.ifft2(objects),axes=(-2,-1))
        with self.assertRaisesRegex(ValueError, 'overlap|positive'):
            self.project(fields)


if __name__ == '__main__':
    unittest.main()
