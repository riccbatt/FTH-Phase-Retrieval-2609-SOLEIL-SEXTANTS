"""Mode count and repeated support factors throughout universal retrieval."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from library import phase_retrieval_universal as u
from library import phase_retrieval_universal_jit as jit
from library import phase_retrieval_core_unified as unified


class ArbitraryModesTests(unittest.TestCase):
    def run_modes(self, factors, *, partial=False, initialization='support_fft', workers=1):
        support=np.zeros((16,16));support[7:9,7:9]=1
        images=np.stack([np.ones((16,16))*v for v in [1.,2.]])
        recipe=u.default_universal_phase_retrieval_recipe()
        recipe.update(modes=factors, mode_initialization=initialization,
                      warmup_mode=['ER','ER'],warmup_Nit=[2,2],
                      warmup_RL_it=[0,1 if partial else 0],warmup_RL_freq=[100,1],
                      partial_coherence=partial,final_fourier_constraint=not partial,preserve_warmup_masked_intensity=True,
                      coherence_kernel_scope='shared',shared_coherence_update='average',
                      observation_workers=workers,inner_mode=['ER'],inner_Nit=[2],
                      RL_it=1 if partial else 0,RL_freq=1,
                      coherent_refresh_rounds=[2] if partial else [],coherent_refresh_iterations=2,
                      outer_iterations=2,physical_iterations=1,projection_every=2,
                      projection_model='physical_factorized',projection_relaxation=.1,
                      final_projection_relaxation=0,constrain_nonphysical_modes_common=True,
                      average_img=1,shuffle_observations=False)
        expected_factors = factors.tolist() if isinstance(factors, np.ndarray) else list(factors)
        with contextlib.redirect_stdout(io.StringIO()):
            result=u.universal_phase_retrieval_algorithm(images,np.zeros_like(images),support,
                ['a','b'],[780,780],[1,-1],['beam','beam'],universal_recipe=recipe)
        fields,warmup,components,_,_=result
        expected=(2,16,16) if len(expected_factors)==1 else (2,len(expected_factors),16,16)
        self.assertEqual(fields.shape,expected);self.assertEqual(warmup.shape,expected)
        self.assertTrue(np.isfinite(fields).all())
        self.assertEqual(components['modes'],expected_factors)
        if len(expected_factors)>1:
            np.testing.assert_allclose(fields[0,1:],fields[1,1:])
            self.assertEqual([g['support_factor'] for g in components['nonphysical_mode_common_groups']],expected_factors[1:])
        if partial:self.assertEqual(components['coherence_kernel'].shape,(len(factors),16,16))

    def test_all_requested_lists(self):
        for factors in [[1],[1,2],[1,1],[1,1,2],[1,2,3],[1,1,2,2],[1,0],[1,0,2]]:
            with self.subTest(modes=factors):self.run_modes(factors)

    def test_numpy_mode_factors_are_accepted(self):
        for factors in [np.array([1], dtype=float), np.array([1, 2], dtype=float), np.array([1, 1, 2], dtype=float)]:
            with self.subTest(modes=factors.tolist()):
                self.run_modes(factors)

    def test_zero_mode_has_no_support(self):
        support = np.zeros((8, 8), dtype=np.uint8)
        support[2:6, 2:6] = 1
        supports = unified._mode_supports(support, [1, 0, 2], (8, 8), gaussian_mode=[None, {"sigma_x": 4.0, "sigma_y": 4.0}, None])
        np.testing.assert_array_equal(supports[0], support)
        np.testing.assert_array_equal(supports[1], np.ones((8, 8), dtype=np.uint8))
        np.testing.assert_array_equal(supports[2], unified._expand_support_about_center(support, 2))

    def test_zero_mode_stays_real_in_object_space(self):
        support = np.zeros((16, 16), dtype=np.uint8)
        support[6:10, 6:10] = 1
        mask = np.stack([support, np.ones((16, 16), dtype=np.uint8)])
        gaussian = np.exp(-0.5 * ((np.arange(16)[:, None] - 7.5) ** 2 / 4.0 + (np.arange(16)[None, :] - 7.5) ** 2 / 4.0))
        phase = np.stack([
            np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(support.astype(np.complex128)))),
            gaussian * (1.0 + 1.0j),
        ])
        out, *_ = unified.PhaseRtrv_core_multimode(
            np.ones((16, 16), dtype=float),
            mask,
            mode='ER',
            Nit=2,
            beta_zero=0.5,
            beta_mode='const',
            alpha_zero=0.0,
            alpha_mode='const',
            Phase=phase,
            plot_every=1,
            average_img=1,
            Fourier_last=True,
            Nmodes=2,
        )
        self.assertEqual(out.shape, (2, 16, 16))
        self.assertTrue(np.allclose(np.imag(out[1]), 0.0, atol=1e-8))

    def test_zero_mode_fit_uses_measured_hologram(self):
        shape = (64, 64)
        fy = np.fft.fftfreq(shape[0], d=1.0 / shape[0])
        fx = np.fft.fftfreq(shape[1], d=1.0 / shape[1])
        kx, ky = np.meshgrid(fx, fy, indexing='xy')
        measured = 2.5 * np.exp(-0.5 * ((kx * 6.0) ** 2 + (ky * 8.0) ** 2))

        optimized = unified._fit_gaussian_zero_mode_to_measurement(shape, measured, bsmask=np.zeros(shape, dtype=bool))
        sigma_x, sigma_y, amplitude, _ = optimized
        baseline = 2.5 * np.exp(-0.5 * ((kx * 1.0) ** 2 + (ky * 1.0) ** 2))

        model = amplitude * np.exp(-0.5 * ((kx * sigma_x) ** 2 + (ky * sigma_y) ** 2))
        self.assertTrue(np.isfinite([sigma_x, sigma_y, amplitude]).all())
        self.assertLess(float(np.sum((model - measured) ** 2)), float(np.sum((baseline - measured) ** 2)))

    def test_zero_mode_fit_avoids_expensive_meshgrid_search(self):
        shape = (48, 48)
        fy = np.fft.fftfreq(shape[0], d=1.0 / shape[0])
        fx = np.fft.fftfreq(shape[1], d=1.0 / shape[1])
        kx, ky = np.meshgrid(fx, fy, indexing='xy')
        measured = 3.0 * np.exp(-0.5 * ((kx * 7.0) ** 2 + (ky * 9.0) ** 2))

        with patch.object(unified.np, "meshgrid", side_effect=AssertionError("meshgrid-based coarse search must be removed")):
            optimized = unified._fit_gaussian_zero_mode_to_measurement(shape, measured, bsmask=np.zeros(shape, dtype=bool))

        self.assertEqual(len(optimized), 4)
        self.assertTrue(np.isfinite(optimized[0]))
        self.assertTrue(np.isfinite(optimized[1]))
        self.assertTrue(np.isfinite(optimized[2]))

    def test_coherent_zero_mode_uses_support_center(self):
        shape = (32, 32)
        support = np.zeros(shape, dtype=np.uint8)
        support[8:24, 10:22] = 1
        support_center = unified._supportmask_center(support)
        measured = 1.7 * np.exp(-0.5 * (((np.arange(shape[0])[:, None] - support_center[0]) / 3.0) ** 2 + ((np.arange(shape[1])[None, :] - support_center[1]) / 4.0) ** 2))
        optimized = unified._gaussianize_zero_mode(
            np.zeros(shape),
            shape,
            measured_amplitude=measured,
            bsmask=np.zeros(shape, dtype=bool),
            center=((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0),
            coherent=True,
            reference_support=support,
        )
        row, col = np.unravel_index(np.argmax(np.abs(optimized)), optimized.shape)
        self.assertLess(abs(row - support_center[0]), 2.0)
        self.assertLess(abs(col - support_center[1]), 2.0)

    def test_partial_parallel_refresh_and_random_initialization(self):
        for init in ['support_fft','random_phase']:
            with self.subTest(init=init):self.run_modes([1,1,2,2],partial=True,workers=2,initialization=init)

    def test_spectral_dispatch_keeps_all_modes(self):
        support=np.zeros((16,16));support[7:9,7:9]=1
        for model in ['svd','rank1_spectral']:
            recipe=u.default_universal_phase_retrieval_recipe()
            recipe.update(modes=[1,1,2],projection_model=model,rank=1,
                          warmup_mode=['ER'],warmup_Nit=[1],inner_mode=['ER'],inner_Nit=[1],
                          outer_iterations=1,average_img=1,final_projection_relaxation=0)
            with contextlib.redirect_stdout(io.StringIO()):
                result=u.universal_phase_retrieval_algorithm(np.ones((2,16,16)),np.zeros((16,16)),support,
                    ['state']*2,[770,780],[1,1],['beam']*2,universal_recipe=recipe)
            self.assertEqual(result[0].shape,(2,3,16,16))
            self.assertEqual(result[2]['modes'],[1,1,2])

    def test_direct_projection_accepts_modal_stacks(self):
        fields=np.ones((2,4,8,8),complex)
        for model in ['physical_factorized','svd','rank1_spectral']:
            recipe=u.default_universal_phase_retrieval_recipe()
            recipe.update(modes=[1,1,2,2],projection_model=model,physical_iterations=1)
            out, components=u.project_fourier_fields_universal(fields,['s','s'],[770,780],
                [1,1],['beam','beam'],universal_recipe=recipe,return_components=True)
            self.assertEqual(out.shape,fields.shape)

    def test_general_projection_detects_single_beam_hysteresis_rank_deficiency(self):
        log_objects = np.ones((3, 8, 8), dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, 'physical_factorized|single-beam|single-energy'):
            u.project_log_objects_general(
                log_objects,
                state_labels=['s0', 's1', 's2'],
                energy_labels=[780.0, 780.0, 780.0],
                polarization_coefficients=[1.0, 1.0, 1.0],
                beam_labels=['beam0', 'beam0', 'beam0'],
                rank_deficient='error',
            )

    def test_jit_multimode_fallback(self):
        from unittest.mock import patch
        with patch('library.phase_retrieval_core_unified.PhaseRtrv_core',return_value='ok') as core:
            self.assertEqual(jit._configured_kernel(True)(Nmodes=4,Phase='field'),'ok')
            self.assertEqual(core.call_args.kwargs['Nmodes'],4)
        with self.assertRaises(NotImplementedError):jit._configured_kernel(False)(Nmodes=4)
