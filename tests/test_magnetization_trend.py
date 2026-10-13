import unittest
import numpy as np
from library import phase_retrieval_universal as u


class MagnetizationTrendTests(unittest.TestCase):
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

    def test_invalid_trend(self):
        recipe = u.default_universal_phase_retrieval_recipe()
        recipe['magnetization_trend'] = 'up'
        with self.assertRaisesRegex(ValueError, 'magnetization_trend'):
            u.project_log_objects_physical(np.zeros((2, 2, 2), complex),
                ['a', 'b'], [781.]*2, [1.]*2, ['beam']*2, recipe=recipe)
