"""Mode count and repeated support factors throughout universal retrieval."""
import contextlib
import io
import unittest
import numpy as np
from library import phase_retrieval_universal as u
from library import phase_retrieval_universal_jit as jit


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
        with contextlib.redirect_stdout(io.StringIO()):
            result=u.universal_phase_retrieval_algorithm(images,np.zeros_like(images),support,
                ['a','b'],[780,780],[1,-1],['beam','beam'],universal_recipe=recipe)
        fields,warmup,components,_,_=result
        expected=(2,16,16) if len(factors)==1 else (2,len(factors),16,16)
        self.assertEqual(fields.shape,expected);self.assertEqual(warmup.shape,expected)
        self.assertTrue(np.isfinite(fields).all())
        self.assertEqual(components['modes'],factors)
        if len(factors)>1:
            np.testing.assert_allclose(fields[0,1:],fields[1,1:])
            self.assertEqual([g['support_factor'] for g in components['nonphysical_mode_common_groups']],factors[1:])
        if partial:self.assertEqual(components['coherence_kernel'].shape,(len(factors),16,16))

    def test_all_requested_lists(self):
        for factors in [[1],[1,2],[1,1],[1,1,2],[1,2,3],[1,1,2,2]]:
            with self.subTest(modes=factors):self.run_modes(factors)

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

    def test_jit_multimode_fallback(self):
        from unittest.mock import patch
        with patch('library.phase_retrieval_core_unified.PhaseRtrv_core',return_value='ok') as core:
            self.assertEqual(jit._configured_kernel(True)(Nmodes=4,Phase='field'),'ok')
            self.assertEqual(core.call_args.kwargs['Nmodes'],4)
        with self.assertRaises(NotImplementedError):jit._configured_kernel(False)(Nmodes=4)
