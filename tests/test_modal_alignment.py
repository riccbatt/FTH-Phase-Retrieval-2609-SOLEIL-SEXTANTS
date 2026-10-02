import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'library'))
from modal_alignment import align_modes

class ModalAlignmentTests(unittest.TestCase):
    def test_complex_unitary_and_intensity_invariance(self):
        rng = np.random.default_rng(4)
        ref = rng.normal(size=(2,9,8))+1j*rng.normal(size=(2,9,8))
        u = np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
        mixed = (u@ref.reshape(2,-1)).reshape(ref.shape)
        aligned, fit, report = align_modes(mixed, ref)
        np.testing.assert_allclose(aligned, ref, atol=1e-12)
        np.testing.assert_allclose(fit@fit.conj().T, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(np.sum(abs(aligned)**2,axis=0), np.sum(abs(mixed)**2,axis=0))
        self.assertLess(report['complex_relative_error'], 1e-12)

    def test_collapsed_mode_is_not_two_dimensional_recovery(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(size=(2,8,8)).astype(complex)
        collapsed = np.stack([ref[0], ref[0]])
        _, _, report = align_modes(collapsed, ref)
        self.assertEqual(report['ranks'], [1,2])
        self.assertGreater(report['complex_relative_error'], .1)

    def test_same_subspace_different_powers_is_not_unitary_match(self):
        ref = np.eye(2,8).reshape(2,2,4).astype(complex)
        _, _, report = align_modes(ref*np.array([2,.1])[:,None,None], ref)
        self.assertLess(max(report['principal_angles_degrees']), 1e-6)
        self.assertGreater(report['complex_relative_error'], .5)


class NotebookControlTests(unittest.TestCase):
    def test_single_and_two_mode_paths(self):
        import ast
        import json
        import phase_retrieval_core_unified as pr
        path = Path(__file__).resolve().parents[1]/'maxiv_phase_test/07_maxiv_modal_subspace_validation.ipynb'
        notebook = json.loads(path.read_text())
        tree = ast.parse(''.join(notebook['cells'][4]['source']))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        support = np.zeros((12,12)); support[4:8,4:8] = 1
        namespace = dict(np=np, pr=pr, support=support, VALIDATION_ITERATIONS=[2,2])
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'), namespace)
        target = abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(support))))**2
        for count in [1,2]:
            fields, residual = namespace['reconstruct_control'](target,np.zeros_like(support,bool),count,0)
            self.assertEqual(fields.shape,(count,12,12))
            self.assertTrue(np.isfinite(residual))

if __name__ == '__main__': unittest.main()
