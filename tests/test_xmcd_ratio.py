import unittest
import numpy as np
from library.retrieval_optical_constants import xmcd_ratio_indices


class XmcdRatioTests(unittest.TestCase):
    def test_thesis_sign_factor_two_and_global_phase_reference(self):
        shape=(8,8);d=np.zeros(shape);d[2:6,2:6]=12
        m=np.full(shape,.6);m[2:4]*=-1
        k=2*np.pi*780/1239.8419843320026;x=2*k*d*m
        beta,delta=.002,-.003
        ratio=np.exp(-x*beta-1j*x*delta+1.1j)
        estimate=xmcd_ratio_indices(ratio,np.ones(shape),780,d,m,
                                   phase_reference_mask=d==0)
        self.assertAlmostEqual(estimate['beta_m'],beta)
        self.assertAlmostEqual(estimate['delta_m'],delta)
        self.assertAlmostEqual(estimate['phase_offset'],1.1)
        self.assertEqual(estimate['valid_pixels'],16)
        self.assertAlmostEqual(estimate['beta_spatial_rms'],0)

    def test_missing_reference_and_zero_magnetization(self):
        out=xmcd_ratio_indices(np.ones((3,3)),np.ones((3,3)),780,10,1,
                               phase_reference_mask=np.zeros((3,3),bool))
        self.assertTrue(np.isnan(out['delta_m']))
        self.assertEqual(out['beta_m'],0)
        with self.assertRaises(ValueError):
            xmcd_ratio_indices(np.ones((3,3)),np.ones((3,3)),780,10,0)

    def test_low_signal_exclusion(self):
        p=np.ones((3,3),complex);p[0,0]=0;p[0,1]=np.nan
        out=xmcd_ratio_indices(p,np.ones((3,3)),780,10,1)
        self.assertEqual(out['valid_pixels'],7)
        self.assertEqual(out['phase_status'],'unreferenced')
