"""Radial starts use valid detector samples and preserve complex modal ratios."""
import unittest
from pathlib import Path
import numpy as np
from library import phase_retrieval_universal as u


class RadialInitializationTests(unittest.TestCase):
    def test_masked_annulus_interpolation_and_modal_phase(self):
        y, x = np.indices((9, 11))
        r = np.floor(np.hypot(y-4, x-5)).astype(int)
        measured = 2.0*r
        mask = r == 2
        measured[mask] = 1e12
        field = np.stack([np.full(r.shape, 1j), np.full(r.shape, 2+0j)])
        result = u._radially_normalize_startimage(field, measured, mask)
        np.testing.assert_allclose(np.sum(abs(result)**2, axis=0), 2*r, atol=1e-14)
        np.testing.assert_allclose(result[1], -2j*result[0])
        self.assertTrue(np.all(result[:, r == 0] == 0))
        np.testing.assert_array_equal(field[0], 1j)

    def test_zero_field_and_missing_outer_rings(self):
        image = np.full((8,8), np.nan)
        image[4,4] = 9
        out = u._radially_normalize_startimage(np.zeros((2,8,8)), image, 0)
        np.testing.assert_array_equal(out[0], 3)
        np.testing.assert_array_equal(out[1], 0)
        with self.assertRaises(ValueError):
            u._radially_normalize_startimage(out, image, 1)

    def test_driver_normalizes_both_fft_and_reference_seed(self):
        import contextlib
        import io
        y, x = np.indices((8, 8))
        rings = np.floor(np.hypot(y-4, x-4))
        holograms = np.stack([rings+1, 3*(rings+1)])
        calls = []
        def kernel(**kw):
            calls.append(kw["Phase"].copy())
            return kw["Phase"], np.zeros(1), np.zeros(1), kw.get("gamma")
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe.update(modes=[1, 2], mode_initialization="support_fft",
                      startimage_radial_normalization=True,
                      warmup_start_from_first=True, warmup_seed_scale_fit="sum",
                      warmup_mode=["ER"], warmup_Nit=[1], warmup_RL_it=[0],
                      warmup_RL_freq=[100], inner_mode=["ER"], inner_Nit=[1],
                      outer_iterations=1, projection_model="none",
                      final_projection_relaxation=0, final_fourier_constraint=False,
                      shuffle_observations=False, average_img=1)
        with contextlib.redirect_stdout(io.StringIO()):
            u.universal_phase_retrieval_algorithm(
                holograms, np.zeros_like(holograms), np.ones((8,8)),
                ["a", "b"], [1,1], [1,1], ["beam", "beam"],
                universal_recipe=recipe, phase_retrieval_kernel=kernel)
        for index in range(2):
            np.testing.assert_allclose(np.sum(abs(calls[index])**2, axis=0),
                                       holograms[index], atol=1e-12)

    def test_shared_library_links(self):
        root = Path(__file__).resolve().parents[1]
        for source in (root/'library').glob('phase_retrieval*.py'):
            self.assertEqual((root/'maxiv_phase_test/library'/source.name).resolve(), source)

    def test_common_mode_ignores_state_and_polarization_but_retains_energy(self):
        fields = np.ones((4,2,5,5), complex)
        fields[:,1] *= np.arange(1,5)[:,None,None]
        recipe=u.default_universal_phase_retrieval_recipe()
        recipe['nonphysical_modes_common_relaxation']=1
        out, groups=u._project_nonphysical_modes_common(fields,[1,1,2,2],['b']*4,recipe)
        np.testing.assert_array_equal(out[:,0], fields[:,0])
        np.testing.assert_array_equal(out[0,1], out[1,1])
        np.testing.assert_array_equal(out[2,1], out[3,1])
        self.assertFalse(np.array_equal(out[0,1],out[2,1]))

if __name__ == '__main__':
    unittest.main()
