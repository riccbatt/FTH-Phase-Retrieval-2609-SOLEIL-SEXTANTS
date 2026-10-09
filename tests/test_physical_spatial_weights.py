import unittest
import numpy as np
from library import phase_retrieval_universal as u


class SpatialWeightTests(unittest.TestCase):
    def test_weighting_preserves_projection_relaxation(self):
        logs = np.array([[[1.1], [1.8]], [[1.0], [1.4]],
                         [[.9], [.2]]], dtype=complex)
        for weighting in (None, 'exit_wave_intensity'):
            recipe = u.default_universal_phase_retrieval_recipe()
            recipe.update(physical_spatial_weighting=weighting,
                          physical_projection_object_roi=True)
            def project(relaxation):
                return u.project_log_objects_physical(
                    logs, ['p','free','n'], [781.]*3, [1.]*3, ['beam']*3,
                    recipe=recipe, material_thickness=np.ones((2,1)),
                    saturated_states={'p':1.,'n':-1.}, iterations=20,
                    relaxation=relaxation)
            target = project(1.)
            for relaxation in (0., .25, .45, 1.):
                np.testing.assert_allclose(project(relaxation),
                    (1-relaxation)*logs + relaxation*target)

    def test_intensity_weighted_anchor_response(self):
        # Bright accurate pixel and dim outlier: weighted q has an analytic solution.
        common = np.log(np.array([[10.], [1.]]))
        contrast = np.array([[.1], [1.]])
        logs = np.stack([common+contrast, common-contrast]).astype(complex)
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe['physical_spatial_weighting'] = 'exit_wave_intensity'
        _, c = u.project_log_objects_physical(
            logs, ['plus','minus'], [781.]*2, [1.]*2, ['beam']*2,
            recipe=recipe, material_thickness=np.ones((2,1)),
            saturated_states={'plus':1.,'minus':-1.}, iterations=5, return_components=True)
        expected_weights = np.mean(np.exp(2*logs.real), axis=0)
        expected_weights /= expected_weights.mean()
        np.testing.assert_allclose(c['physical_spatial_weights'], expected_weights)
        expected_response = np.sum(expected_weights*contrast)/np.sum(expected_weights)
        np.testing.assert_allclose(c['magnetic_response'], expected_response)
        self.assertLess(abs(c['magnetic_response'][0]-.1), .1)

    def test_roi_expansion_and_exact_free_copy(self):
        material=np.array([[0.,1.],[1.,0.]])
        common=np.array([[0.,1.],[2.,0.]])
        signs=np.array([1.,1.,-1.,-1.])
        logs=common[None]+signs[:,None,None]*material*(.1+.03j)
        recipe=u.default_universal_phase_retrieval_recipe()
        recipe.update(physical_spatial_weighting='exit_wave_intensity',physical_projection_object_roi=True)
        projected,c=u.project_log_objects_physical(
            logs,['p','pc','n','nc'],[781.]*4,[1.]*4,['beam']*4,
            recipe=recipe,material_thickness=material,saturated_states={'p':1.,'n':-1.},
            iterations=20,return_components=True)
        np.testing.assert_allclose(projected,logs)
        np.testing.assert_allclose(c['magnetization'][:,material>0],np.broadcast_to(signs[:,None],(4,2)))
        np.testing.assert_array_equal(c['physical_spatial_weights'][material==0],0.)
        self.assertAlmostEqual(c['physical_spatial_weights'][material>0].mean(),1.)

if __name__=='__main__': unittest.main()
