"""Sequential notebook must reproduce the explicit notebook-01 recipe."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.ndimage import binary_dilation
from library import phase_retrieval_universal as universal
from library import phase_retrieval_core_unified as unified
from library import phase_retrieval_geometry as geometry

NOTEBOOK = Path(__file__).resolve().parents[1] / 'aperiodic_ajajas/03_sequential_hysteresis_warmup_poisson.ipynb'


def helpers():
    namespace = dict(np=np, unified=unified, binary_dilation=binary_dilation, painted_detector_mask=False)
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] == 'code':
            tree = ast.parse(''.join(cell['source']))
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    return namespace


class SequentialWarmupTests(unittest.TestCase):
    def test_editable_recipe_matches_01(self):
        ns = {}
        notebook = json.loads(NOTEBOOK.read_text())
        exec(''.join(notebook['cells'][1]['source']), ns)
        nb01 = json.loads(NOTEBOOK.with_name('01_phase_retrieval_2871_2872.ipynb').read_text())
        expected = dict(ns)
        # Evaluate 01's current settings assignments, independently of 03.
        for node in ast.parse(''.join(nb01['cells'][1]['source'])).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], 'id', '').isupper()
                    and getattr(node.targets[0], 'id', '') not in {'HERE', 'REPO'}):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<01>', 'exec'), expected)
        exec(''.join(nb01['cells'][6]['source']), expected)
        self.assertEqual(ns['recipe'], expected['recipe'])
        for name in ['SATURATED_SCAN', 'LOOP_SCAN', 'STATE_POINT_NUMBER',
                     'SUPPORT_EROSION', 'SUPPORT_DILATION', 'SUPPORT_CENTER_RADIUS',
                     'SUPPORT_SHIFT', 'HOLOGRAM_OFFSET', 'NORMALIZE_LOOP_INTENSITY']:
            self.assertEqual(ns[name], expected[name])
        self.assertFalse(ns['RUN_POISSON_REFINEMENT'])

    def test_focused_difference_matches_widget_with_nonzero_shift(self):
        from library import fthcore as fth
        from aperiodic_ajajas.library import fth_reconstruction as focus_fth
        ns = helpers()
        ns['fth'] = fth
        rng = np.random.default_rng(9)
        state = rng.normal(size=(24, 24)) + 1j * rng.normal(size=(24, 24))
        saturated = rng.normal(size=(24, 24)) + 1j * rng.normal(size=(24, 24))
        roi = (slice(3, 18), slice(5, 21))
        focus = dict(focus_um=3.22, phase_rad=-.76, dx=.35, dy=-.6)
        setup = dict(ccd_dist=.125, px_size=11e-6, energy=783.)
        # Reproduce the expressions in rec.focusCDI's callback.
        positive = fth.reconstructCDI(fth.propagate(
            state, 3.22e-6, experimental_setup=setup)) * np.exp(-.76j)
        negative = fth.reconstructCDI(fth.propagate(
            saturated, 3.22e-6, experimental_setup=setup)) * np.exp(-.76j)
        expected = focus_fth.sub_pixel_centering(
            np.nan_to_num(positive - negative, nan=0, posinf=0, neginf=0), .35, -.6)[roi]
        actual = ns['focused_difference'](state, saturated, roi, focus, setup)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ns['focused_difference'](
            saturated, saturated, roi, focus, setup), 0, atol=1e-12)

    def test_gif_keeps_global_limits_and_acquisition_order(self):
        import tempfile
        import h5py
        from PIL import Image, ImageDraw, GifImagePlugin
        ns = helpers()
        ns.update(h5py=h5py, Image=Image, ImageDraw=ImageDraw, GifImagePlugin=GifImagePlugin)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for point, scale in [(7, 1), (2, 5), (9, 2)]:
                path = Path(directory) / f"{point}.h5"
                with h5py.File(path, "w") as h:
                    h.attrs.update(point_number=point, current_A=point * .01)
                    grid = np.arange(120).reshape(10, 12) - 60
                    for stage, factor in [("warmup", 1), ("poisson", 2)]:
                        h.create_dataset("focused_differences/" + stage,
                                         data=scale * factor * (grid + 1j * grid[::-1]))
                paths.append(path)
            limits = ns['difference_color_limits'](paths)
            np.testing.assert_allclose(limits['real'], np.percentile(grid * 10, (1, 99)))
            destination = Path(directory) / 'sequence.gif'
            ns['write_difference_gif'](paths, destination, 'warmup', 'real', limits, 200)
            with Image.open(destination) as gif:
                self.assertEqual(gif.n_frames, 3)
                self.assertEqual(gif.size, (12, 10))
                self.assertEqual(gif.info['loop'], 0)
                self.assertEqual(gif.info['duration'], 200)
                for i, scale in enumerate([1, 5, 2]):
                    gif.seek(i)
                    actual = np.array(gif.convert('RGB'))[:, :, 0]
                    low, high = limits['real']
                    expected = np.rint(255 * np.clip((grid * scale - low) / (high - low), 0, 1))
                    np.testing.assert_array_equal(actual, expected.astype(np.uint8))

    def test_png_replaces_legacy_combined_masks_and_preserves_automatic(self):
        from aperiodic_ajajas.result_io import painted_cache_mask
        ns = helpers()
        painted = np.zeros((8, 8), bool)
        painted[3, 4] = True
        ns.update(painted_cache_mask=painted_cache_mask, painted_detector_mask=painted,
                  observation_indices=[None, 0])
        cache = {'saturated_mask': np.ones((8, 8)), 'loop_masks': np.ones((1, 8, 8))}
        for i in range(2):
            np.testing.assert_array_equal(painted_cache_mask(cache, [None, 0][i], painted), painted)
        automatic = np.zeros((1, 8, 8), bool)
        automatic[0, 1, 2] = True
        cache['loop_automatic_masks'] = automatic
        expected = painted | automatic[0]
        np.testing.assert_array_equal(painted_cache_mask(cache, 0, ns['painted_detector_mask']), expected)
        # Editing away a painted exclusion really removes it on reload.
        ns['painted_detector_mask'] = np.zeros_like(painted)
        np.testing.assert_array_equal(painted_cache_mask(cache, 0, ns['painted_detector_mask']), automatic[0])


if __name__ == '__main__':
    unittest.main()
