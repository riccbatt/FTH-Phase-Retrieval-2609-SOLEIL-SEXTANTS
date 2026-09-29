"""Saturation labels must not split otherwise identical physical states."""

import unittest
import numpy as np
from library import phase_retrieval_universal as universal


class SaturationInitializationTests(unittest.TestCase):
    def test_identical_plateaus_with_sparse_anchors(self):
        values = np.array([1., 1., 1., -1., -1., -1.])
        thickness = np.array([[0., 1.], [0.5, 2.]])
        for response in (.2+.1j, -.2-.1j, -.1+.2j):
            for roi in (False, True):
                for iterations in (1, 20):
                    with self.subTest(response=response, roi=roi, iterations=iterations):
                        logs = (.3+.4j) + values[:, None, None]*response*thickness
                        recipe = universal.default_universal_phase_retrieval_recipe()
                        recipe['physical_projection_object_roi'] = roi
                        projected, components = universal.project_log_objects_physical(
                            logs, list(range(6)), [710.]*6, [1.]*6, ['beam']*6,
                            saturated_states={1: 1., 4: -1.},
                            material_thickness=thickness, recipe=recipe,
                            iterations=iterations, return_components=True)
                        np.testing.assert_allclose(projected, logs, atol=1e-12)
                        np.testing.assert_allclose(
                            components['magnetization'][:, thickness > 0],
                            np.broadcast_to(values[:, None], (6, 3)), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
