"""Check exact amplitudes and missing-data policies at the retrieval boundary."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'library'))
import phase_retrieval_core_unified as pr


class RecipePolicies(unittest.TestCase):
    def check_policy(self, policy, nmodes=2):
        shape=(8,8)
        yy,xx=np.mgrid[:8,:8]
        images={'+':xx+yy+10.,'-':2*xx+yy+30.}
        mask=np.zeros(shape,dtype=np.uint8);mask[3:5,3:5]=1
        recipe=pr.default_phase_retrieval_recipe()
        recipe.update(helicity=['+','+','-','+','+','-'],
                      coherence=['full']*3+['partial']*3,
                      algorithm_list=['ER']*6,number_iterations=[6]*6,
                      RL_its=[0,0,0,1,1,1],RL_freqs=[100,100,100,1,1,1],
                      Startimage=[None,'+','+','+','+','+'],
                      Startgamma=[None,None,None,None,'+','+'],
                      normalize_startimage_between_holograms=False,
                      modes=[1]*nmodes,hologram_offset={'+':-7.,'-':3.},
                      pc_missing_data=policy,hologram_intensity_cutoff_vmin=-1)
        outputs={};i=0
        def kernel(**kwargs):
            nonlocal i
            label=recipe['helicity'][i]
            measured=images[label]-recipe['hologram_offset'][label]
            expected=measured.copy()
            actual_policy=(policy[i] if isinstance(policy,list) else policy) if i>=3 else 'free'
            expected_mask=mask.copy()
            if actual_policy!='free':
                source=kwargs['Phase'] if actual_policy=='fixed_from_start' else outputs[label]
                filled=np.abs(source)**2
                if source.ndim==3:filled=filled.sum(axis=0)
                expected[mask!=0]=filled[mask!=0]
                expected_mask[:]=0
            else:expected[mask!=0]=0
            np.testing.assert_allclose(kwargs['diffract']**2,expected,rtol=1e-14,atol=1e-14)
            np.testing.assert_array_equal(kwargs['bsmask'],expected_mask)
            field=np.full(shape,complex(i+1,2))
            if nmodes>1:field=np.stack([field*(m+1) for m in range(nmodes)])
            outputs[label]=field.copy()
            gamma=np.zeros_like(field);gamma[...,4,4]=1
            i+=1
            return field,[.1],[.1],gamma if kwargs['RL_it'] else None
        with patch.object(pr,'PhaseRtrv_core',side_effect=kernel):
            result=pr.phase_retrieval_algorithm(images,mask,np.ones(shape),recipe)
        self.assertEqual(i,6)
        self.assertEqual([s['coherence'] for s in result['error']['steps']],['full']*3+['partial']*3)
        for step in result['error']['steps']:
            self.assertEqual(step['observed_pixels'],60)
            self.assertEqual(step['synthetic_target_pixels'],0 if step['pc_missing_data']=='free' else 4)

    def test_free_policy(self):self.check_policy('free')
    def test_fixed_start_single_mode(self):self.check_policy('fixed_from_start',1)
    def test_fixed_start_multimode_sums_intensities_before_sqrt(self):self.check_policy('fixed_from_start')
    def test_fixed_previous_uses_target_label_not_start_label(self):self.check_policy('fixed_from_previous')
    def test_per_stage_policy(self):self.check_policy(['free']*4+['fixed_from_start','fixed_from_previous'])

    def test_partial_without_kernel_updates_is_rejected(self):
        recipe=pr.default_phase_retrieval_recipe()
        recipe.update(coherence='partial',RL_its=0)
        with self.assertRaisesRegex(ValueError,'Partial stage'):
            pr.phase_retrieval_algorithm({'pos':np.ones((8,8)),'neg':np.ones((8,8))},
                                         np.zeros((8,8)),np.ones((8,8)),recipe)

    def test_invalid_policy_is_rejected(self):
        recipe=pr.default_phase_retrieval_recipe();recipe['pc_missing_data']='typo'
        with self.assertRaisesRegex(ValueError,'pc_missing_data'):
            pr.phase_retrieval_algorithm({'pos':np.ones((8,8)),'neg':np.ones((8,8))},
                                         np.zeros((8,8)),np.ones((8,8)),recipe)

if __name__=='__main__':unittest.main()
