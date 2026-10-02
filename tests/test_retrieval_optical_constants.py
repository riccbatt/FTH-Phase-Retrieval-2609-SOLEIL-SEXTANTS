import unittest
import numpy as np
from library.retrieval_optical_constants import calibrate_responses


class OpticalCalibrationTests(unittest.TestCase):
    def test_units_sign_and_magnetic_product(self):
        energy=np.array([770.,780.]);thickness=12.
        kt=2*np.pi*energy/1239.8419843320026*thickness
        delta=np.array([.001,.002]);beta=np.array([.003,.004])
        q=-kt*beta+1j*kt*delta
        qm=q*.1;m=np.array([[[-.2,.4],[0,.1]]])
        result=calibrate_responses(q,qm,m,energy,cobalt_thickness_nm=thickness,
                                  normalize_magnetization=True)
        np.testing.assert_allclose(result['charge_delta'],delta)
        np.testing.assert_allclose(result['charge_beta'],beta)
        np.testing.assert_allclose(result['charge_index_contrast'],-delta-1j*beta)
        np.testing.assert_allclose(np.exp(-1j*kt*result['charge_index_contrast']),np.exp(q))
        opposite=calibrate_responses(q.conj(),qm.conj(),m,energy,cobalt_thickness_nm=thickness,propagation_sign=1)
        np.testing.assert_allclose(opposite['charge_delta'],delta)
        np.testing.assert_allclose(opposite['charge_beta'],beta)
        np.testing.assert_allclose(result['magnetic_response'][:,None,None,None]*result['magnetization'],qm[:,None,None,None]*m)
        self.assertEqual(np.max(abs(result['magnetization'])),1.)
        self.assertEqual(np.min(result['magnetization']),-.5)
        np.testing.assert_allclose(m[0,0],[-.2,.4])

    def test_unknown_thickness_and_invalid_inputs(self):
        args=([1j],[2j],np.ones((1,2,2)),[780])
        self.assertNotIn('charge_delta',calibrate_responses(*args))
        for t in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):calibrate_responses(*args,cobalt_thickness_nm=t)
        with self.assertRaises(ValueError):
            calibrate_responses([1j],[2j],np.zeros((1,2,2)),[780],normalize_magnetization=True)
