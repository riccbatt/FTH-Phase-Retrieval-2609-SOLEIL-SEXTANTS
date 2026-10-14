"""Synthetic recovery and integration of chemical species physical constraints."""
import contextlib
import io
import unittest
import numpy as np
from library import phase_retrieval_universal as u


class ChemicalSpeciesTests(unittest.TestCase):
    def test_shared_intensity_weights_change_spectral_fit_as_weighted_least_squares(self):
        density=self.density[0]
        q=np.array([-.12,.04,.09,-.01],complex)
        noise=np.zeros(self.shape); noise[2,0]=1.
        noise_scale=np.array([.5,-.3,.1,-.3])
        logs=-2*noise[None]+q[:,None,None]*density+noise_scale[:,None,None]*noise
        recipe=self.recipe([dict(name='a',density_map=density)])
        recipe['physical_spatial_weighting']='exit_wave_intensity'
        _,c=self.project(logs,recipe,iterations=300)
        active=self.material>0
        expected_weights=np.mean(np.exp(2*logs.real),axis=0)
        expected_weights[~active]=0.
        expected_weights/=expected_weights[active].mean()
        np.testing.assert_allclose(c['physical_spatial_weights'],expected_weights)
        coefficient=np.sum(expected_weights*density*noise)/np.sum(expected_weights*density**2)
        expected=q-q.mean()+coefficient*(noise_scale-noise_scale.mean())
        np.testing.assert_allclose(c['species_charge_response'][0],expected,atol=1e-8)
        recipe['physical_spatial_weighting']=None
        _,uniform=self.project(logs,recipe,iterations=300)
        self.assertGreater(np.linalg.norm(uniform['charge_response']-c['charge_response']),.01)

    def test_species_use_one_combined_exit_wave_weight_map(self):
        species=[dict(name=str(i),charge_response=self.spectra[i]) for i in range(2)]
        for weighting in ('exit_wave_intensity','exit_wave_intensity_per_frame'):
            with self.subTest(weighting=weighting):
                recipe=self.recipe(species)
                recipe['physical_spatial_weighting']=weighting
                projected,c=self.project(self.logs,recipe,iterations=300)
                np.testing.assert_allclose(c['species_density_maps'],self.density,atol=1e-7)
                np.testing.assert_allclose(projected,self.logs,atol=1e-7)
                expected=np.exp(2*self.logs.real)
                if weighting=='exit_wave_intensity':
                    expected=expected.mean(axis=0)
                expected[...,self.material==0]=0.
                expected/=expected[...,self.material>0].mean(axis=-1)[...,None,None]
                np.testing.assert_allclose(c['physical_spatial_weights'],expected)
                self.assertLess(c['weighted_fit_residual_rms'],1e-7)

    def test_weighted_magnetic_response_fit_uses_species_saturation_anchors(self):
        q=np.array([.1+.05j,.13+.08j,.08-.03j,.2+.01j])
        m=np.array([1.,.2,-1.])
        logs=np.stack([self.logs[e]+self.density[0]*q[e]*m[s]
                       for s in range(3) for e in range(4)])
        species=[dict(name='magnetic',density_map=self.density[0],
                      charge_response=self.spectra[0],magnetic=True,
                      fit_magnetic_response=True,saturated_states={'p':1.,'n':-1.}),
                 dict(name='other',density_map=self.density[1],charge_response=self.spectra[1])]
        for weighting in ('exit_wave_intensity','exit_wave_intensity_per_frame'):
            recipe=self.recipe(species)
            recipe['physical_spatial_weighting']=weighting
            projected,c=self.project(logs,recipe,states=[s for s in ['p','f','n'] for _ in range(4)],
                                     energies=self.energies*3,iterations=300)
            np.testing.assert_allclose(projected,logs,atol=1e-8)
            np.testing.assert_allclose(c['magnetic_response'],q,atol=1e-8)
            np.testing.assert_allclose(c['magnetization'][1,self.material>0],.2,atol=1e-8)
            if weighting=='exit_wave_intensity':
                expected=np.exp(2*logs[np.r_[0:4,8:12]].real).mean(axis=0)
                expected[self.material==0]=0
                expected/=expected[self.material>0].mean()
                np.testing.assert_allclose(c['physical_spatial_weights'],expected)

    def setUp(self):
        self.shape = (6, 7)
        self.material = np.ones(self.shape); self.material[:2] = 0
        yy, xx = np.indices(self.shape)
        self.density = np.stack([(.3+.1*xx)*self.material, (.2+.12*yy)*self.material])
        self.energies = [700., 705., 710., 715.]
        self.spectra = np.array([[.1+.2j, .3+.1j, .2-.2j, -.1+.05j],
                                 [.2-.1j, -.1+.3j, .4+.1j, .1-.2j]])
        self.common = (.05+.08j)*np.ones(self.shape)
        self.logs = self.common+np.einsum('ie,iyx->eyx', self.spectra, self.density)

    def recipe(self, species):
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe.update(chemical_species=species, energy_values=np.array(self.energies),
                      chemical_reference_alignment=False, clip_magnetization=False)
        return recipe

    def project(self, logs, recipe, states=None, energies=None, polarizations=None, iterations=200):
        return u.project_log_objects_physical(logs, states or ['state']*len(logs),
            energies or self.energies, polarizations or [1.]*len(logs), ['beam']*len(logs),
            recipe=recipe, material_thickness=self.material,
            iterations=iterations, return_components=True)

    def test_known_spectra_recover_overlapping_density_maps(self):
        species = [dict(name=str(i), charge_response=self.spectra[i], fit_density=True) for i in range(2)]
        projected, c = self.project(self.logs, self.recipe(species))
        np.testing.assert_allclose(c['species_density_maps'], self.density, atol=2e-7)
        np.testing.assert_allclose(projected, self.logs, atol=2e-7)
        np.testing.assert_allclose(c['common_log_objects'][0], self.common, atol=2e-7)
        self.assertEqual(c['chemical_spectral_contrast_rank'], 2)
        self.assertEqual(c['density_scale_anchored'], [True, True])
        self.assertFalse(c['species_magnetic'].any())
        self.assertFalse(c['identifiable'])  # Does not certify an entire diffraction inverse problem.

    def test_known_maps_recover_spectral_contrasts(self):
        species = [dict(name=str(i), density_map=self.density[i]) for i in range(2)]
        projected, c = self.project(self.logs, self.recipe(species))
        expected = self.spectra-self.spectra.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(c['species_charge_response'], expected, atol=1e-8)
        np.testing.assert_allclose(projected, self.logs, atol=1e-8)
        np.testing.assert_allclose(c['species_density_maps'], self.density)

    def test_joint_blind_fit_is_consistent_but_not_certified_as_unique(self):
        recipe = self.recipe([dict(name='a'), dict(name='b')])
        projected, c = self.project(self.logs, recipe, iterations=400)
        self.assertLess(c['fit_residual_rms'], 1e-6)
        np.testing.assert_allclose(projected, self.logs, atol=1e-5)
        self.assertTrue(np.all(c['species_density_maps'] >= 0))
        self.assertFalse(c['identifiable'])
        self.assertTrue(any('mixing' in note for note in c['ambiguity_notes']))
        for i in range(2):
            self.assertAlmostEqual(c['species_density_maps'][i, self.material > 0].mean(), 1.)

    def test_vacuum_alignment_removes_observation_offsets_and_preserves_exterior(self):
        offsets = np.array([.03+.2j, -.02-.4j, .04+.5j, -.05-.3j])
        logs = self.logs+offsets[:, None, None]
        recipe = self.recipe([dict(name=str(i), charge_response=self.spectra[i]) for i in range(2)])
        recipe.update(chemical_reference_alignment=True, physical_projection_object_roi=True)
        projected, c = self.project(logs, recipe)
        np.testing.assert_allclose(c['species_density_maps'], self.density, atol=2e-7)
        np.testing.assert_allclose(c['chemical_reference_offsets'], offsets-offsets.mean(), atol=1e-10)
        np.testing.assert_allclose(projected, self.logs+offsets.mean(), atol=2e-7)
        self.assertEqual(c['chemical_reference_pixels'], 14)
        self.assertTrue(c['chemical_reference_alignment_applied'])
        recipe['chemical_reference_alignment'] = False
        projected, _ = self.project(logs, recipe)
        np.testing.assert_array_equal(projected[:, self.material == 0], logs[:, self.material == 0])

    def test_each_species_can_have_independent_magnetism_and_saturation(self):
        states = ['plus', 'middle', 'minus']
        m = np.array([1., .2, -1.])
        q = np.array([.1+.05j, .13+.08j, .08-.03j, .2+.01j])
        logs = np.stack([self.logs[e]+self.density[0]*q[e]*m[s]
                         for s in range(3) for e in range(4)])
        species = [dict(name='magnetic', density_map=self.density[0], charge_response=self.spectra[0],
                        magnetic=True, magnetic_response=q, saturated_states={'plus': 1., 'minus': -1.}),
                   dict(name='nonmagnetic', density_map=self.density[1], charge_response=self.spectra[1])]
        projected, c = self.project(logs, self.recipe(species),
            states=[s for s in states for _ in range(4)], energies=self.energies*3)
        np.testing.assert_allclose(projected, logs, atol=1e-8)
        active = self.material > 0
        np.testing.assert_allclose(c['species_magnetization'][0][:, active], np.broadcast_to(m[:, None], (3, active.sum())), atol=1e-8)
        np.testing.assert_array_equal(c['species_magnetization'][1], 0.)
        self.assertEqual(c['species']['magnetic']['saturated_states'], {'plus': 1., 'minus': -1.})

    def test_kk_is_applied_per_species(self):
        recipe = self.recipe([dict(name='a', density_map=self.density[0], charge_spectral_constraint='kk', kk_sign=-1.),
                              dict(name='b', density_map=self.density[1], charge_spectral_constraint='free')])
        _, c = self.project(self.logs, recipe, iterations=10)
        response = c['species']['a']['charge_response']
        factor = np.array(self.energies)*u._EV_TO_WAVENUMBER_PER_METRE
        constrained, _ = u.constrain_complex_spectrum(response/factor, spectral_constraint='kk',
            energy_values=self.energies, absorption_part='real', kk_sign=-1.)
        np.testing.assert_allclose(response, constrained*factor)
        self.assertEqual(c['species']['a']['spectral_info']['charge']['spectral_constraint'], 'kk')
        self.assertEqual(c['species']['b']['spectral_info']['charge']['spectral_constraint'], 'free')

    def test_unanchored_pure_phase_magnetism_is_not_lost(self):
        values = np.array([-.5, .1, .4])
        logs = np.stack([self.common + .15j*self.density[0]*m for m in values])
        recipe = self.recipe([dict(name='phase', density_map=self.density[0], magnetic=True)])
        projected, c = self.project(logs, recipe, states=['a', 'b', 'c'], energies=[700.]*3)
        np.testing.assert_allclose(projected, logs, atol=1e-8)
        self.assertGreater(abs(c['species_magnetic_response'][0, 0]), 0.)
        self.assertTrue(any('magnetic response' in note for note in c['ambiguity_notes']))

    def test_known_index_prior_constrains_beta_before_kt_factor(self):
        beta = np.array([.001, .002, .003, .0015])
        delta = np.array([.002, .001, -.001, .0005])
        d0 = 1e-8
        recipe = self.recipe([dict(name='a', density_map=self.density[0],
            reference_thickness_m=d0, charge_spectral_constraint='known_beta_kk',
            known_charge_beta_spectrum=beta, known_charge_delta_spectrum=delta,
            fit_known_spectrum_scale=False, fit_known_spectrum_offset=False)])
        _, c = self.project(self.logs, recipe, iterations=5)
        np.testing.assert_allclose(c['species']['a']['beta'], beta)
        # Existing KK helper retains a fitted dispersion offset. Test its shape.
        actual_delta = c['species']['a']['delta']
        np.testing.assert_allclose(actual_delta-actual_delta.mean(), delta-delta.mean())

    def test_zero_relaxation_preserves_fields_even_with_vacuum_alignment(self):
        recipe = self.recipe([dict(name='a')]); recipe['chemical_reference_alignment'] = True
        logs = self.logs + np.arange(4)[:, None, None]*(.02+.1j)
        projected = u.project_log_objects_physical(logs, ['s']*4, self.energies,
            [1.]*4, ['beam']*4, recipe=recipe, material_thickness=self.material,
            iterations=2, relaxation=0.)
        np.testing.assert_array_equal(projected, logs)

    def test_index_conversion_has_explicit_scale(self):
        recipe = self.recipe([dict(name='a', density_map=self.density[0], charge_response=self.spectra[0], reference_thickness_m=1e-8),
                              dict(name='b', density_map=self.density[1], charge_response=self.spectra[1])])
        _, c = self.project(self.logs, recipe, iterations=1)
        kt = np.array(self.energies)*u._EV_TO_WAVENUMBER_PER_METRE*1e-8
        np.testing.assert_allclose(c['species']['a']['beta'], self.spectra[0].real/kt)
        np.testing.assert_allclose(c['species']['a']['delta'], -self.spectra[0].imag/kt)
        self.assertNotIn('beta', c['species']['b'])

    def test_rank_deficiency_and_invalid_inputs(self):
        recipe = self.recipe([dict(charge_response=self.spectra[0]), dict(charge_response=self.spectra[0])])
        _, c = self.project(self.logs, recipe, iterations=2)
        self.assertLess(c['chemical_spectral_contrast_rank'], 2)
        self.assertTrue(any('rank deficient' in note for note in c['ambiguity_notes']))
        for species in ([], [dict(name='x'), dict(name='x')], [dict(magnetic_response=self.spectra[0])], [dict(unknown=1)]):
            with self.subTest(species=species), self.assertRaises(ValueError):
                self.project(self.logs, self.recipe(species), iterations=1)
        recipe = self.recipe([dict(name='a')]); recipe['chemical_reference_mask'] = self.material > 0
        with self.assertRaisesRegex(ValueError, 'zero-material'):
            self.project(self.logs, recipe)

    def test_direct_fourier_bridge_with_multiple_modes(self):
        fields = u.object_log_to_fourier_field(self.logs)
        modal_fields = np.stack((fields, fields*.1), axis=1)
        recipe = self.recipe([dict(name=str(i), charge_response=self.spectra[i], density_map=np.fft.ifftshift(self.density[i])) for i in range(2)])
        recipe.update(material_thickness=np.fft.ifftshift(self.material), modes=[1, 2], physical_iterations=5)
        output, c = u.project_fourier_fields_universal(modal_fields, ['s']*4, self.energies,
                    [1.]*4, ['beam']*4, universal_recipe=recipe, return_components=True)
        np.testing.assert_allclose(output[:, 0], fields, atol=1e-10)
        self.assertEqual(c['species_density_maps'].shape, (2, *self.shape))

    def test_retrieval_crop_bin_maps_species_like_material(self):
        from library.phase_retrieval_species import map_species_recipe
        from library import phase_retrieval_geometry as geometry
        shape = (24, 24)
        material = np.zeros(shape); material[8:16, 8:16] = 1
        recipe = self.recipe([dict(name='a', density_map=material, supportmask=material, charge_response=self.spectra[0])])
        recipe.update(crop=2, binning=2, material_thickness=material, warmup_Nit=0, inner_Nit=1,
                      outer_iterations=0, physical_iterations=2, modes=[1], final_fourier_constraint=False)
        data = np.ones((4, *shape)); mask = np.zeros_like(data)
        _, _, _, _, geom = geometry.prepare(data, mask, material, recipe)
        mapped = map_species_recipe(recipe, input_geometry=geom)
        expected = np.fft.fftshift(u._recipe_material_thickness(recipe, geom))
        np.testing.assert_allclose(mapped['chemical_species'][0]['density_map'], expected)
        with contextlib.redirect_stdout(io.StringIO()):
            fields, _, c, _, _ = u.universal_phase_retrieval_algorithm(data, mask, material,
                ['s']*4, self.energies, [1.]*4, ['beam']*4, universal_recipe=recipe)
        self.assertEqual(fields.shape[-2:], (10, 10))
        np.testing.assert_allclose(c['species_density_maps'][0], expected)


if __name__ == '__main__':
    unittest.main()
