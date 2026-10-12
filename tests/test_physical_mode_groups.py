import unittest
import numpy as np
from library import phase_retrieval_universal as u


class PhysicalModeGroupTests(unittest.TestCase):
    def test_final_common_cleanup_preserves_selected_physical_modes(self):
        recipe=u.default_universal_phase_retrieval_recipe()
        recipe.update(modes=[1,1,2],magnetization_mode_groups=[[0,1]],
                      illumination_mode_groups=[[0],[1]],
                      nonphysical_modes_common_relaxation=1.)
        fields=np.ones((3,3,2,2),complex)
        fields[:,0] *= np.array([1.,2.,3.])[:,None,None]
        fields[:,1] *= np.array([3.,2.,1.])[:,None,None]
        fields[:,2] *= np.array([1.,2.,4.])[:,None,None]
        output,groups=u._project_nonphysical_modes_common(
            fields,[781.]*3,['beam']*3,recipe)
        np.testing.assert_array_equal(output[:,:2],fields[:,:2])
        np.testing.assert_allclose(output[:,2],7./3.)
        self.assertEqual([group['mode_index'] for group in groups],[2])

    def project(self, magnetic, illumination, free_values, common_values, initial=None, correction=False):
        material=np.zeros((6,6)); material[2:4,2:4]=1.
        support=material.copy(); support[0,0]=1.; support[5,5]=1.
        groups_m={mode:g for g,group in enumerate(magnetic) for mode in group}
        groups_i={mode:g for g,group in enumerate(illumination) for mode in group}
        logs=np.empty((3,3,6,6),complex)
        for mode in range(3):
            m=np.array([1.,free_values[groups_m[mode]],-1.])
            common=common_values[groups_i[mode]]+np.arange(36).reshape(6,6)*.001
            logs[:,mode]=common+m[:,None,None]*material*(.08+.03j)
        fields=np.stack([u.object_log_to_fourier_field(logs[:,mode]) for mode in range(3)],axis=1)
        recipe=u.default_universal_phase_retrieval_recipe()
        recipe.update(modes=[1,1,1],magnetization_mode_groups=magnetic,
                      illumination_mode_groups=illumination,clip_magnetization=False,
                      constrain_nonphysical_modes_common=True)
        if correction:
            recipe.update(align_global_phase=True, fit_aperture_illumination=True)
            phases=np.array([[.1,.6,-.4],[2.3,-1.9,.5],[-2.,1.3,-.7]])
            fields *= (np.array([1.,1.3,.8])[:,None]*np.exp(1j*phases))[:,:,None,None]
        output,c=u._project_physical_modes(fields,['p','f','n'],[781.]*3,[1.]*3,['beam']*3,
            recipe=recipe,material_thickness=material,projection_supportmask=support,
            saturated_states={'p':1.,'n':-1.},physical_iterations=100,
            initial_components=initial,return_components=True)
        return fields,output,c,material

    def test_shared_magnetization_with_distinct_illumination(self):
        fields,output,c,material=self.project([[0,1,2]],[[0],[1],[2]],[.4],[.2+.04j,.7+.3j,-.1+.1j])
        np.testing.assert_allclose(output,fields,atol=1e-8)
        np.testing.assert_allclose(c['magnetization'][1,material>0],.4,atol=1e-8)
        self.assertEqual(c['physical_model_modes'],[1,2,3])
        self.assertFalse(c['nonphysical_modes_common'])
        _,continued,c2,_=self.project([[0,1,2]],[[0],[1],[2]],[.4],[.2+.04j,.7+.3j,-.1+.1j],c)
        np.testing.assert_allclose(continued,fields,atol=1e-8)
        self.assertTrue(c2['physical_fit_reused'])

    def test_distinct_magnetization_with_shared_illumination(self):
        fields,output,c,_=self.project([[0],[1],[2]],[[0,1,2]],[.4,-.6,.8],[.2+.04j])
        np.testing.assert_allclose(output,fields,atol=1e-8)
        for group,value in enumerate([.4,-.6,.8]):
            np.testing.assert_allclose(c['magnetization_by_mode_group_packed'][str(group)][1],value,atol=1e-8)
        self.assertEqual(len(c['modal_physical_fit']['beam_names']),1)

    def test_combined_partitions(self):
        fields,output,c,_=self.project([[0,1],[2]],[[0],[1,2]],[.4,-.6],[.2+.04j,.7+.3j])
        np.testing.assert_allclose(output,fields,atol=1e-8)
        self.assertEqual(len(c['magnetization_by_mode_group_packed']),2)
        self.assertEqual(len(c['modal_physical_fit']['beam_names']),2)

    def test_shared_illumination_handles_independent_modal_phase_gauges_and_aperture_gains(self):
        fields,output,c,_=self.project([[0],[1],[2]],[[0,1,2]],
                                     [.4,-.6,.8],[.2+.04j],correction=True)
        np.testing.assert_allclose(output,fields,atol=1e-8)
        corrections=c['illumination_corrections_by_mode']
        for mode in ('1','2'):
            np.testing.assert_allclose(corrections[mode]['illumination_intensity_factors'],
                                       corrections['0']['illumination_intensity_factors'],atol=1e-10)
        self.assertEqual(len(c['modal_physical_fit']['beam_names']),1)

    def test_driver_applies_groups_and_final_projection_reuses_joint_fit(self):
        import contextlib
        import io
        magnetic=[[0,1,2]]; illumination=[[0],[1],[2]]
        fields,_,_,material=self.project(magnetic,illumination,[.4],[.2+.04j,.7+.3j,-.1+.1j])
        physical_support=material.copy();physical_support[0,0]=1.;physical_support[5,5]=1.
        recipe=u.default_universal_phase_retrieval_recipe()
        recipe.update(crop=0,binning=1,modes=[1,1,1],material_thickness=np.fft.fftshift(material),
                      magnetization_mode_groups=magnetic,illumination_mode_groups=illumination,
                      projection_constraints_inside_support_only=True,clip_magnetization=False,
                      warmup_Nit=0,outer_iterations=2,inner_Nit=1,physical_iterations=20,
                      projection_every=3,final_fourier_constraint=False,
                      saturated_states={'p':1.,'n':-1.})
        with contextlib.redirect_stdout(io.StringIO()):
            output,_,c,_,_=u.universal_phase_retrieval_algorithm(
                np.sum(abs(fields)**2,axis=1),np.zeros((3,6,6)),np.fft.fftshift(physical_support),
                ['p','f','n'],[781.]*3,[1.]*3,['beam']*3,
                universal_recipe=recipe,start_fields=fields)
        self.assertEqual(output.shape,fields.shape)
        self.assertEqual(c['physical_model_modes'],[1,2,3])
        self.assertTrue(c['physical_fit_reused'])
        self.assertEqual(c['magnetization'].shape,(3,6,6))
        self.assertEqual(len(c['modal_physical_fit']['state_names']),3)

    def test_groups_validate_and_missing_partition_defaults_to_independent(self):
        self.assertEqual(u._physical_mode_groups({'magnetization_mode_groups':[[0,1,2]]},3),
                         ([[0,1,2]],[[0],[1],[2]]))
        for magnetic,illumination in [([[0,0]],None),([[0,3]],None),([[1]],None),
                                      ([[0,1]],[[0],[2]])]:
            with self.assertRaises(ValueError):
                u._physical_mode_groups(dict(magnetization_mode_groups=magnetic,
                                             illumination_mode_groups=illumination),3)


if __name__=='__main__':unittest.main()
