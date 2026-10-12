import unittest
import numpy as np
from library.cidi import cidi_separate, _fft


class CIDITests(unittest.TestCase):
    def fixture(self, nrefs=2):
        shape=(128,128)
        sample=np.zeros(shape,complex)
        sample[62:66,61:65]=np.arange(16).reshape(4,4)/20+.5j
        refs=[]
        for y,x,value in [(15,20,2+.4j),(95,28,1.1-.3j),(25,100,.8+.2j)][:nrefs]:
            r=np.zeros(shape,complex); r[y:y+2,x:x+2]=value; refs.append(r)
        inc=np.zeros(shape,complex)
        inc[62:66,61:65]=.1*np.exp(1j*np.arange(16).reshape(4,4))
        coherent=abs(_fft(sample+sum(refs)))**2
        incoherent=abs(_fft(inc))**2
        return sample,refs,coherent,incoherent

    def test_recovers_complex_finite_references_and_weak_incoherent_signal(self):
        for n in (2,3):
            sample,refs,coh,inc=self.fixture(n)
            result=cidi_separate(coh+inc,sample!=0,[r!=0 for r in refs])
            valid=result.valid_mask
            self.assertGreater(valid.sum(),10000)
            np.testing.assert_allclose(result.sample_intensity[valid],abs(_fft(sample))[valid]**2,atol=1e-8)
            np.testing.assert_allclose(result.reference_intensity[valid],sum(abs(_fft(r))**2 for r in refs)[valid],atol=1e-8)
            np.testing.assert_allclose(result.coherent_intensity[valid],coh[valid],atol=1e-8)
            np.testing.assert_allclose(result.incoherent_intensity[valid],inc[valid],atol=1e-8)
            self.assertFalse(valid[0,0])  # finite-reference diffraction zero
            intensity,mask=result.reconstruction_inputs()
            self.assertTrue(np.all(intensity>=0)); self.assertTrue(mask[0,0])

    def test_overlapping_geometry_is_rejected(self):
        sample,refs,coh,inc=self.fixture()
        refs[0]=np.roll(sample!=0,5,axis=0)
        with self.assertRaisesRegex(ValueError,'lobes overlap'):
            cidi_separate(coh+inc,sample!=0,[r!=0 for r in refs])

    def test_explicit_missing_pixel_fill(self):
        sample,refs,coh,inc=self.fixture()
        data=coh+inc; mask=np.zeros(data.shape,bool);mask[30,30]=True
        measured=data.copy(); measured[mask]=np.nan
        with self.assertRaisesRegex(ValueError,'filled_intensity'):
            cidi_separate(measured,sample!=0,[r!=0 for r in refs],invalid_mask=mask)
        result=cidi_separate(measured,sample!=0,[r!=0 for r in refs],invalid_mask=mask,filled_intensity=data)
        self.assertFalse(result.valid_mask[30,30])
        self.assertTrue(result.diagnostics['detector_filling_used'])

    def test_negative_residual_is_retained_and_excluded_from_retrieval(self):
        sample,refs,coh,inc=self.fixture()
        result=cidi_separate(coh+inc,sample!=0,[r!=0 for r in refs])
        result.incoherent_intensity[64,64]=-1.
        intensity,mask=result.reconstruction_inputs()
        self.assertTrue(mask[64,64]);self.assertEqual(intensity[64,64],0.)
        self.assertEqual(result.incoherent_intensity[64,64],-1.)

    def test_handoff_to_universal_driver(self):
        from library.phase_retrieval_universal import (
            default_universal_phase_retrieval_recipe, universal_phase_retrieval_algorithm)
        sample,refs,coh,inc=self.fixture()
        result=cidi_separate(coh+inc,sample!=0,[r!=0 for r in refs])
        intensity,mask=result.reconstruction_inputs()
        recipe=default_universal_phase_retrieval_recipe()
        recipe.update(projection_model='none',modes=[1],crop=0,binning=1,
                      warmup_Nit=0,outer_iterations=1,inner_Nit=1)
        output=universal_phase_retrieval_algorithm(
            np.stack([intensity,intensity]),np.stack([mask,mask]),sample!=0,
            ['signal_a','signal_b'],[781.,781.],[1.,1.],['beam','beam'],
            universal_recipe=recipe)
        self.assertTrue(np.all(np.isfinite(output[0])))
        self.assertEqual(output[0].shape[-2:],intensity.shape)

    def test_requires_two_references(self):
        sample,refs,coh,inc=self.fixture()
        with self.assertRaisesRegex(ValueError,'at least two'):
            cidi_separate(coh+inc,sample!=0,[refs[0]!=0])
