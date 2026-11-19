"""Missing detector pixels must not become observations in partial coherence."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))
import phase_retrieval_core_unified as pr


class PartialCoherenceMasks(unittest.TestCase):
    def run_retrieval(self, nmodes, hidden_value):
        n = 16
        yy, xx = np.mgrid[:n, :n]
        support = ((yy-8)**2 + (xx-8)**2 < 16).astype(float)
        obj = support * (1 + .1*np.cos(xx)) * np.exp(.05j*xx)
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(obj)))
        intensity = np.abs(field)**2
        mask = ((yy-8)**2 + (xx-8)**2 < 4).astype(np.uint8)
        intensity[mask != 0] = hidden_value
        recipe = pr.default_phase_retrieval_recipe()
        recipe.update(algorithm_list=['ER']*6, number_iterations=[6]*6,
                      helicity=['+','+','-','+','+','-'],
                      RL_its=[0,0,0,1,1,1], RL_freqs=[100,100,100,1,1,1],
                      alpha_zero=0., average_img=1, plot_every=100,
                      Startgamma=[None,None,None,None,"+","+"],
                      Startimage=[field if nmodes == 1 else np.stack([field, field])/np.sqrt(2),
                                  '+','+','+','+','+'],
                      modes=[1] if nmodes == 1 else [1,2],
                      hologram_intensity_cutoff_vmin=-1,
                      normalize_startimage_between_holograms=False,
                      output=[False,True,True,False,True,True], return_format='dict')
        masks = []
        amplitudes = []
        starting_fields = []
        stage_index = 0
        kernel = pr.PhaseRtrv_core
        def checked_kernel(**kwargs):
            nonlocal stage_index
            label = recipe['helicity'][stage_index]
            expected_intensity = intensity * (1.0 if label == '+' else .95)
            expected_amplitude = np.sqrt(np.where(mask != 0, 0., expected_intensity))
            np.testing.assert_allclose(kwargs['diffract'], expected_amplitude,
                                       rtol=1e-14, atol=1e-16)
            stage_index += 1
            if kwargs['RL_it'] > 0:
                masks.append(np.asarray(kwargs['bsmask']).copy())
                amplitudes.append(np.asarray(kwargs['diffract']).copy())
                starting_fields.append(np.asarray(kwargs['Phase']).copy())
            return kernel(**kwargs)
        with patch.object(pr, 'PhaseRtrv_core', side_effect=checked_kernel):
            result = pr.phase_retrieval_algorithm(
                {'+':intensity, '-':intensity*.95}, mask_pixel=mask,
                supportmask=support, phase_retrieval_recipe=recipe)
        self.assertEqual(stage_index, 6)
        self.assertEqual(len(masks), 3)
        # Startimage='+' must resolve to the latest positive complex field,
        # independently of which helicity supplies the measured amplitude.
        for actual, source_step in zip(starting_fields, (1, 3, 4)):
            np.testing.assert_array_equal(actual, result['error']['steps'][source_step]['field_after'])
        for actual in masks:
            np.testing.assert_array_equal(actual, mask)
        for label in ('+','-'):
            self.assertIsNotNone(result['partial_coherence'][label])
            self.assertTrue(np.isfinite(result['partial_coherence'][label]).all())
        return result, amplitudes

    def test_single_mode_missing_values_do_not_constrain_partial_coherence(self):
        self.check_hidden_value_invariance(1)

    def test_multimode_missing_values_do_not_constrain_partial_coherence(self):
        self.check_hidden_value_invariance(2)

    def check_hidden_value_invariance(self, nmodes):
        _, original_constraints = self.run_retrieval(nmodes, 0.)
        _, changed_constraints = self.run_retrieval(nmodes, 1.e6)
        # Compare constraints to floating-point precision, independently of RL convergence.
        for before, after in zip(original_constraints, changed_constraints):
            np.testing.assert_allclose(before, after, rtol=1e-14, atol=1e-16)



if __name__ == '__main__':
    unittest.main()
