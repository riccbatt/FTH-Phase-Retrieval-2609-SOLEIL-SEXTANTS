import unittest
import numpy as np
from library import phase_retrieval_universal as u


class MagnetizationTrendTests(unittest.TestCase):
    def test_custom_asymmetric_clipping_with_trend(self):
        for trend, sign in [('increasing', 1), ('decreasing', -1)]:
            maps = sign * np.array([[.8, .8], [-.4, -.4]])[:, None, :]
            u._constrain_magnetization_trend(
                maps, np.ones((1, 2), bool), ['a', 'b'], {}, trend,
                clip_range=(-.5, .9))
            self.assertTrue(np.all((maps >= -.5) & (maps <= .9)))
            self.assertGreaterEqual(sign * np.diff(maps.mean(axis=(1, 2)))[0], -1e-12)

    def test_mean_projection_retains_texture_and_anchors(self):
        for trend, sign in [('increasing', 1), ('decreasing', -1)]:
            maps = sign * np.array([[-1., -1., 0.], [.8, .4, 0.],
                                    [-.4, -.8, 0.], [1., 1., 0.]])[:, None, :]
            before = maps.copy()
            active = np.array([[True, True, False]])
            u._constrain_magnetization_trend(maps, active, ['a', 'b', 'c', 'd'],
                {'a': -sign, 'd': sign}, trend, False)
            np.testing.assert_allclose(maps[1:3, 0, :2].mean(axis=1), 0., atol=1e-15)
            np.testing.assert_allclose(np.diff(maps[1:3, 0, :2], axis=1),
                                       np.diff(before[1:3, 0, :2], axis=1))
            np.testing.assert_array_equal(maps[[0, 3]], before[[0, 3]])
            np.testing.assert_array_equal(maps[:, :, 2], 0.)

    def test_bounded_shift_and_independent_modal_sequences(self):
        maps = np.array([[1., .8], [-.8, -1.], [-.2, .2], [.8, 1.]])[:, None, :]
        u._constrain_magnetization_trend(maps, np.ones((1, 2), bool),
            list('abcd'), {}, 'increasing', True, [[0, 2], [1, 3]])
        means = maps.mean(axis=(1, 2))
        self.assertLessEqual(means[0], means[2] + 1e-12)
        self.assertLessEqual(means[1], means[3] + 1e-12)
        self.assertTrue(np.all(abs(maps) <= 1))

    def test_physical_projection_and_roi(self):
        names = ['n', 'b', 'c', 'p']
        thickness = np.array([[0., 1.], [1., 0.]])
        values = np.array([-1., .7, -.6, 1.])
        logs = .3 + values[:, None, None] * thickness * (.1 + .03j)
        for roi in [False, True]:
            for trend in ['none', 'increasing']:
                recipe = u.default_universal_phase_retrieval_recipe()
                recipe.update(magnetization_trend=trend, physical_projection_object_roi=roi)
                _, components = u.project_log_objects_physical(
                    logs, names, [781.]*4, [1.]*4, ['beam']*4, recipe=recipe,
                    material_thickness=thickness, saturated_states={'n': -1., 'p': 1.},
                    iterations=20, return_components=True)
                means = components['magnetization'][:, thickness > 0].mean(axis=1)
                if trend == 'increasing':
                    self.assertTrue(np.all(np.diff(means) >= -1e-12))
                else:
                    self.assertGreater(means[1], means[2])
                np.testing.assert_allclose(means[[0, 3]], [-1., 1.])
            recipe['magnetization_trend'] = 'decreasing'
            with self.assertRaisesRegex(ValueError, 'contradicts'):
                u.project_log_objects_physical(logs, names, [781.]*4, [1.]*4,
                    ['beam']*4, recipe=recipe, material_thickness=thickness,
                    saturated_states={'n': -1., 'p': 1.})

    def test_grouped_physical_projection(self):
        material = np.ones((2, 2))
        values = np.array([-1., .7, -.6, 1.])
        logs = .3 + values[:, None, None] * material * (.1 + .03j)
        fields = np.stack([u.object_log_to_fourier_field(logs)]*2, axis=1)
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe.update(modes=[1, 1], magnetization_trend='increasing',
                      magnetization_mode_groups=[[0], [1]],
                      illumination_mode_groups=[[0], [1]])
        _, components = u._project_physical_modes(
            fields, ['n', 'b', 'c', 'p'], [781.]*4, [1.]*4, ['beam']*4,
            recipe=recipe, material_thickness=material,
            saturated_states={'n': -1., 'p': 1.},
            physical_iterations=20, return_components=True)
        for maps in components['magnetization_by_mode_group_packed'].values():
            self.assertTrue(np.all(np.diff(maps.mean(axis=(1, 2))) >= -1e-12))

    def test_binary_means_allow_pixels_to_reverse(self):
        # Means increase, while pixel 0 decreases and then increases.
        maps = np.array([[1., -1., -1., -1.],
                         [-1., 1., 1., -1.],
                         [1., -1., 1., 1.]])[:, None, :]
        original = maps.copy()
        u._constrain_magnetization_trend(maps, np.ones((1, 4), bool),
            list('abc'), {}, 'increasing', discrete_values=np.array([-1., 1.]))
        np.testing.assert_array_equal(maps, original)
        self.assertTrue(np.any(np.diff(maps, axis=0) < 0))

    def test_discrete_violation_is_corrected_without_fractional_pixels(self):
        for levels in [np.array([-1., 1.]), np.array([-1., 0., 1.])]:
            for sign, trend in [(1, 'increasing'), (-1, 'decreasing')]:
                maps = sign*np.array([[-1.]*5, [1., 1., 1., 1., -1.],
                                      [-1., -1., -1., -1., 1.], [1.]*5])[:, None, :]
                u._constrain_magnetization_trend(maps, np.ones((1, 5), bool),
                    list('abcd'), {'a': -sign, 'd': sign}, trend,
                    discrete_values=levels)
                self.assertTrue(np.all(np.isin(maps, levels)))
                self.assertTrue(np.all(sign*np.diff(maps.mean(axis=(1, 2))) >= -1e-12))
                np.testing.assert_array_equal(maps[0], -sign)
                np.testing.assert_array_equal(maps[3], sign)

    def test_binary_physical_fit(self):
        names = ['n', 'b', 'c', 'p']
        thickness = np.ones((2, 2))
        values = np.array([-1., .8, -.8, 1.])
        logs = .3 + values[:, None, None]*thickness*(.1 + .03j)
        for roi in [False, True]:
            recipe = u.default_universal_phase_retrieval_recipe()
            recipe.update(magnetization_trend='increasing', binary_magnetization=True,
                          binary_magnetization_values=[-1., 1.],
                          physical_projection_object_roi=roi)
            _, components = u.project_log_objects_physical(
                logs, names, [781.]*4, [1.]*4, ['beam']*4, recipe=recipe,
                material_thickness=thickness, saturated_states={'n': -1., 'p': 1.},
                iterations=20, return_components=True)
            maps = components['magnetization']
            self.assertTrue(np.all(np.isin(maps, [-1., 1.])))
            self.assertTrue(np.all(np.diff(maps.mean(axis=(1, 2))) >= 0))

    def test_weighted_factor_retains_texture_and_ignores_dim_outliers(self):
        maps = np.array([[[.2, .4, 100.]], [[.1, .3, -100.]]])
        before = maps.copy()
        spatial = np.array([[1., 3., 0.]])
        u._constrain_magnetization_trend(maps, np.ones((1, 3), bool),
            ['a', 'b'], {}, 'increasing', clip=False, spatial_weights=spatial,
            state_weights=np.array([1., 3.]), correction='factor')
        means = np.array([np.average(m[0, :2], weights=[1., 3.]) for m in maps])
        np.testing.assert_allclose(means, [.275, .275])
        for i in range(2):
            np.testing.assert_allclose(maps[i, 0, :2]/before[i, 0, :2],
                                       means[i]/np.average(before[i, 0, :2], weights=[1., 3.]))
        np.testing.assert_array_equal(maps[:, 0, 2], before[:, 0, 2])

    def test_equal_neighbours_remain_unchanged(self):
        for trend in ['increasing', 'decreasing']:
            for correction in ['offset', 'factor']:
                maps = np.array([[[.1, .3]], [[.3, .1]]])
                before = maps.copy()
                u._constrain_magnetization_trend(maps, np.ones((1, 2), bool),
                    ['a', 'b'], {}, trend, correction=correction)
                np.testing.assert_array_equal(maps, before)

    def test_factor_falls_back_for_zero_mean_and_incompatible_clipping(self):
        for maps in [np.array([[[0., 0.]], [[-.4, -.2]]]),
                     np.array([[[.9, -.7]], [[.9, .7]]])]:
            u._constrain_magnetization_trend(maps, np.ones((1, 2), bool),
                ['a', 'b'], {}, 'decreasing', correction='factor')
            self.assertGreaterEqual(maps[0].mean()+1e-12, maps[1].mean())
            self.assertTrue(np.all(abs(maps) <= 1.))

    def test_physical_fit_cutoff_and_weighted_means(self):
        names = ['p', 'a', 'b', 'n']
        values = np.array([1., .2, .6, -1.])
        logs = np.array([.3, .3, -10.])[None, None, :] + values[:, None, None]*(.1+.03j)
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe.update(magnetization_trend='decreasing', magnetization_trend_correction='factor',
                      clip_magnetization=False)
        _, c = u.project_log_objects_physical(logs, names, [781.]*4, [1.]*4,
            ['beam']*4, recipe=recipe, material_thickness=np.ones((1, 3)),
            saturated_states={'p': 1., 'n': -1.}, iterations=20, return_components=True)
        self.assertEqual(c['magnetization_trend_spatial_weights'][0, 2], 0.)
        self.assertTrue(np.all(np.diff(c['magnetization_trend_means']) <= 1e-12))
        np.testing.assert_allclose(c['magnetization_trend_means'][[0, 3]], [1., -1.])

    def test_affine_domains_matches_spread_and_target_mean(self):
        maps = np.array([np.r_[np.full(20, -.2), np.full(80, .8)],
                         np.r_[np.full(80, -.8), np.full(20, .4)]])[:, None, :]
        before = maps.copy()
        u._constrain_magnetization_trend(maps, np.ones((1, 100), bool),
            ['a', 'b'], {}, 'increasing', clip=False, correction='affine_domains')
        np.testing.assert_allclose(maps.mean(axis=(1, 2)), [.02, .02], atol=1e-12)
        gaps = np.ptp(maps, axis=(1, 2))
        np.testing.assert_allclose(gaps, [1.1, 1.1])
        self.assertFalse(np.allclose(maps-before, (maps-before)[:, :, :1]))
        self.assertTrue(np.all(maps.min(axis=(1, 2)) < 0))
        self.assertTrue(np.all(maps.max(axis=(1, 2)) > 0))

    def test_affine_domains_rejects_unimodal_and_one_sided_populations(self):
        for values in [np.linspace(-1, 1, 100), np.ones(100)*.4,
                       np.r_[np.full(95, .4), np.full(5, -.4)]]:
            self.assertIsNone(u._two_domain_distribution(values, np.ones(100)))
        maps = np.array([np.full(100, .5), np.full(100, -.2)])[:, None, :]
        u._constrain_magnetization_trend(maps, np.ones((1, 100), bool),
            ['a', 'b'], {}, 'increasing', correction='affine_domains')
        np.testing.assert_allclose(maps, .15)

    def test_domain_median_trend_recovers_from_infeasible_offsets(self):
        for trend, sign in [('increasing', 1), ('decreasing', -1)]:
            maps = sign*np.array([[[-1., 1., 1., 1.]], [[-1., -1., -1., 1.]]])
            u._constrain_magnetization_trend(
                maps, np.ones((1, 4), bool), ['a', 'b'], {}, trend,
                clip=True, clip_range=(-2., 2.), median_bounds=True,
                median_scope='domains')
            means = maps.mean(axis=(1, 2))
            self.assertGreaterEqual(sign*(means[1]-means[0]), -1e-12)
            np.testing.assert_allclose(means, 0., atol=1e-12)
            for values in maps:
                for mask in (values < 0, values > 0):
                    if np.any(mask):
                        self.assertLessEqual(abs(np.median(values[mask])), 1.+1e-12)
            self.assertTrue(np.all(abs(maps) <= 2.))

    def test_separate_domain_medians_preserve_signs_and_anchors(self):
        maps = np.array([[[-3., -2., -.5, 0., .5, 2., 4.]],
                         [[-1., -1., -1., -1., -1., -1., -1.]]])
        before = maps.copy()
        u._constrain_magnetization_trend(
            maps, np.ones((1, 7), bool), ['a', 'anchor'], {'anchor': -1.},
            'none', clip=False, median_bounds=True, median_scope='domains')
        self.assertAlmostEqual(np.median(maps[0][maps[0] < 0]), -1.)
        self.assertAlmostEqual(np.median(maps[0][maps[0] > 0]), 1.)
        np.testing.assert_array_equal(np.sign(maps), np.sign(before))
        np.testing.assert_array_equal(maps[1], before[1])
        self.assertGreater(maps.max(), 1.)

    def test_domain_medians_skip_empty_domains_and_excluded_pixels(self):
        maps = np.array([[[2., 4., -100., 0.]]])
        u._constrain_magnetization_trend(
            maps, np.ones((1, 4), bool), ['a'], {}, 'none', clip=False,
            median_bounds=True, median_scope='domains',
            spatial_weights=np.array([[1., 1., 0., 1.]]))
        np.testing.assert_allclose(maps, [[[2./3., 4./3., -100., 0.]]])

    def test_median_bounds_without_a_trend_and_without_pixel_clipping(self):
        maps = np.array([[[1.4, 1.6, 1.8, 2.]], [[.5, .5, .5, 3.]]])
        before = maps.copy()
        u._constrain_magnetization_trend(maps, np.ones((1, 4), bool),
            ['a', 'b'], {}, 'none', clip=False, median_bounds=True)
        self.assertAlmostEqual(np.median(maps[0]), 1.)
        np.testing.assert_allclose(np.diff(maps[0]), np.diff(before[0]))
        np.testing.assert_array_equal(maps[1], before[1])
        self.assertGreater(maps[0].max(), 1.)

    def test_median_bounds_and_non_strict_trend_are_joint(self):
        for correction in ['offset', 'factor', 'affine_domains']:
            maps = np.array([np.full(40, 1.8), np.full(40, 1.2), np.full(40, -.4)])[:, None, :]
            u._constrain_magnetization_trend(maps, np.ones((1, 40), bool),
                ['a', 'b', 'c'], {}, 'increasing', clip=False,
                correction=correction, median_bounds=True)
            self.assertTrue(np.all(abs(np.median(maps, axis=(1, 2))) <= 1.+1e-12))
            self.assertTrue(np.all(np.diff(maps.mean(axis=(1, 2))) >= -1e-12))

    def test_median_bounds_use_retained_weighted_pixels(self):
        maps = np.array([[[1.5, 1.5, -100.]]])
        u._constrain_magnetization_trend(maps, np.ones((1, 3), bool),
            ['a'], {}, 'none', clip=False, spatial_weights=np.array([[1., 1., 0.]]), median_bounds=True)
        np.testing.assert_allclose(maps, [[[1., 1., -100.]]])
        self.assertAlmostEqual(u._weighted_magnetization_median(np.array([0., 2.]), np.ones(2)), 1.)

    def test_infeasible_median_and_trend_constraints_report_conflict(self):
        maps = np.array([np.r_[np.full(90, -.5), np.full(10, -100.)],
                         np.r_[np.full(90, .5), np.full(10, 100.)]])[:, None, :]
        with self.assertRaisesRegex(ValueError, "cannot be satisfied"):
            u._constrain_magnetization_trend(maps, np.ones((1, 100), bool),
                ['a', 'b'], {}, 'decreasing', clip=False, median_bounds=True)

    def test_invalid_trend(self):
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe['magnetization_trend'] = 'up'
        with self.assertRaisesRegex(ValueError, 'magnetization_trend'):
            u.project_log_objects_physical(np.zeros((2, 2, 2), complex),
                ['a', 'b'], [781.]*2, [1.]*2, ['beam']*2, recipe=recipe)
