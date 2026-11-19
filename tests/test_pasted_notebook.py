"""Verify the newly supplied notebook's recipe and diagnostics at the kernel boundary."""
from pathlib import Path
import json
import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR','/tmp/fth-test-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'library'))
import phase_retrieval_core_unified as pr
import fth_phase_workflow as wf
import fthcore as fth


def code(index):
    n=json.loads((ROOT/'04_phase_retrieval.ipynb').read_text())
    return ''.join(n['cells'][index]['source'])


class PastedNotebook(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_actual_notebook_recipe_passes_the_correct_helicity_and_start_fields(self):
        image=np.arange(256,dtype=float).reshape(16,16)+100.
        mask=np.zeros_like(image,dtype=np.uint8);mask[6:10,6:10]=1
        env=dict(np=np,plt=plt,wf=wf,pr=pr,mask_pixel=mask,
                 supportmask=np.ones_like(image),positive_label='+',reference_label='-',
                 holograms={'+':image,'-':image*1.2},phase_retrieval_labels=['+','-'],
                 data={'holo':{'+':{},'-':{}}})
        exec(code(9),env)
        recipe=env['phase_retrieval_recipe']
        self.assertEqual(recipe['number_iterations'],[700,90,90]*2)
        self.assertEqual(recipe['Startimage'][5],'-')
        self.assertEqual(recipe['crop'],0)
        exec(code(12),env)
        fields={};gammas={};step=0
        def kernel(**args):
            nonlocal step
            label=recipe['helicity'][step]
            measured = env['prepared_holograms'][label].copy()
            measured[env['prepared_bsmasks'][label] != 0] = 0
            np.testing.assert_allclose(args['diffract']**2, measured)
            np.testing.assert_array_equal(args['bsmask'],env['prepared_bsmasks'][label])
            source=recipe['Startimage'][step]
            if isinstance(source,str):
                np.testing.assert_array_equal(args['Phase'],fields[source])
            gamma_source=recipe['Startgamma'][step]
            if args['RL_it'] and isinstance(gamma_source,str):
                np.testing.assert_array_equal(args['gamma'],gammas[gamma_source])
            result=np.full((2,16,16),step+1+1j,dtype=complex)
            gamma=np.zeros((2,16,16));gamma[:,8,8]=1.
            fields[label]=result.copy()
            if args['RL_it']:gammas[label]=gamma.copy()
            step+=1
            return result,[.1],[.1],gamma if args['RL_it'] else None
        with patch.object(pr,'PhaseRtrv_core',side_effect=kernel):
            exec(code(15),env)
        self.assertEqual(step,6)
        self.assertEqual(env['retrieved_holograms_pc']['+'][0,0,0],5+1j)
        self.assertEqual(env['retrieved_holograms_pc']['-'][0,0,0],6+1j)
        exec(code(16),env)

    def test_exact_six_stage_recipe_completes_on_synthetic_data(self):
        n=32; yy,xx=np.mgrid[:n,:n]
        support=(((yy-16)**2+(xx-16)**2<36)|((yy-6)**2+(xx-25)**2<4)).astype(float)
        obj=support*(1+.1*np.sin(xx))*np.exp(.05j*xx)
        field=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(obj)))
        intensity=np.abs(field)**2*1e7
        mask=((yy-16)**2+(xx-16)**2<9).astype(np.uint8)
        env=dict(np=np,phase_retrieval_labels=['+','-'],positive_label='+',reference_label='-',
                 holograms={'+':intensity,'-':intensity*.98})
        exec(code(9),env)
        result=pr.phase_retrieval_algorithm(env['phase_retrieval_holograms'],mask,support,
                                            env['phase_retrieval_recipe'])
        self.assertEqual([(s['coherence'],s['helicity']) for s in result['error']['steps']],
                         [('full','+'),('full','+'),('full','-'),
                          ('partial','+'),('partial','+'),('partial','-')])
        for label in ('+','-'):
            self.assertEqual(result['partial_coherence'][label].shape,(2,32,32))
            self.assertTrue(np.isfinite(result['partial_coherence'][label]).all())
            self.assertTrue(np.isfinite(result['gamma'][label]).all())

    def test_partial_detector_diagnostic_includes_the_kernel(self):
        field=np.zeros((2,16,16),dtype=complex)
        field[:,8,8]=[2,3]
        gamma=np.zeros_like(field)
        gamma[:,8,9]=1
        predicted=wf.predicted_detector_intensity(field,gamma)
        self.assertAlmostEqual(predicted[8,9],13.)
        self.assertAlmostEqual(predicted[8,8],0.)
        self.assertAlmostEqual(predicted.sum(),13.)

    def test_negative_only_focus_operation_used_in_new_notebook(self):
        setup={'ccd_dist':.5,'energy':780.,'px_size':11e-6}
        positive=np.ones((16,16));negative=np.eye(16)
        actual=wf.reconstruct_fth_pair(positive,negative,setup,fth,operation='n',prop_dist=11.6,phase=-2.72)
        expected=wf.fth_reconstruct(negative,setup,fth,prop_dist=11.6,phase=-2.72)
        np.testing.assert_allclose(actual,expected)

if __name__=='__main__':unittest.main()
