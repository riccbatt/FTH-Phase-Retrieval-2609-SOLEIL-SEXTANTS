"""Sequential notebook warmup must agree with the batch universal driver."""
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
    def test_reference_seeded_fields_match_batch_driver(self):
        ns = helpers()
        ns.update(MODES=[1, 2], WARMUP_START_FROM_REFERENCE=True,
                  WARMUP_POLICY='reference_and_others', REFERENCE_WARMUP_MODE=['HAPRE', 'ER'],
                  OTHER_WARMUP_MODE=['HAPRE', 'ER'], REFERENCE_WARMUP_NIT=[8, 3],
                  OTHER_WARMUP_NIT=[8, 3], AVERAGE_IMAGES=2)
        support = np.zeros((24, 24), dtype=np.uint8)
        support[10:14, 10:14] = 1
        supports, _ = geometry.recenter_modal_supports(support, [1, 2], center='components')
        rng = np.random.default_rng(23)
        images = rng.uniform(1, 5, (3, 24, 24))
        images[1] += images[0] * 2
        images[2] += images[0] * 3
        masks = np.zeros_like(images, dtype=bool)
        masks[:, 11:13, 11:13] = True
        masks[1, 4, 7] = True
        recipe = universal.default_general_phase_retrieval_recipe()
        recipe.update(modes=[1, 2], mode_initialization='support_fft',
                      recenter_modal_supports=True, mode_support_center='components',
                      warmup_start_from_first=True, warmup_seed_scale_fit='linear',
                      warmup_mode=['HAPRE', 'ER'], warmup_Nit=[8, 3],
                      warmup_beta_mode=['arctan', 'const'],
                      warmup_other_mode=['HAPRE', 'ER'], warmup_other_Nit=[8, 3],
                      warmup_other_beta_mode=['const', 'const'],
                      warmup_alpha_zero=[.4, .4], warmup_alpha_mode=['linear_to_0']*2,
                      warmup_TV_freq=[5, 5], average_img=2, beta_zero=.5,
                      outer_iterations=1, inner_Nit=1, inner_mode='ER',
                      plot_every=10**9, preserve_warmup_masked_intensity=True)
        _, expected, _, _, _ = universal.general_phase_retrieval_algorithm(
            images, masks, support, list(range(3)), [0]*3, [1]*3, [0]*3,
            general_recipe=recipe, phase_retrieval_kernel=unified.PhaseRtrv_core)
        reference = None
        for index in range(3):
            actual, _ = ns['run_warmup'](images[index], masks[index], supports, reference)
            np.testing.assert_allclose(actual, expected[index], rtol=1e-12, atol=1e-12)
            if index == 0:
                reference = (actual.copy(), images[0], masks[0])

    def test_focused_difference_matches_widget_with_nonzero_shift(self):
        from aperiodic_ajajas.library import fth_reconstruction as focus_fth
        ns = helpers()
        ns['focus_fth'] = focus_fth
        rng = np.random.default_rng(9)
        state = rng.normal(size=(24, 24)) + 1j * rng.normal(size=(24, 24))
        saturated = rng.normal(size=(24, 24)) + 1j * rng.normal(size=(24, 24))
        roi = (slice(3, 18), slice(5, 21))
        focus = dict(focus_um=3.22, phase_rad=-.76, dx=.35, dy=-.6)
        setup = dict(ccd_dist=.125, px_size=11e-6, energy=783.)
        # Reproduce the expressions in rec.focusCDI's callback.
        positive = focus_fth.reconstructCDI(focus_fth.propagate(
            state, 3.22e-6, setup) * np.exp(-.76j))
        negative = focus_fth.reconstructCDI(focus_fth.propagate(
            saturated, 3.22e-6, setup) * np.exp(-.76j))
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

    def test_support_discovery_filters_scans_and_ignores_point_defaults(self):
        import tempfile
        import h5py
        import os
        ns = helpers()
        ns.update(h5py=h5py, json=json, Path=Path)
        config = dict(saturated_scan=101, loop_scan=202, detector_center=[10, 10],
                      crop_raw_pixels=0, binning=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for point, scan, timestamp in [(4, 101, 100), (93, 101, 200), (50, 999, 300)]:
                path = root / f'two_state_point_{point:04d}.h5'
                with h5py.File(path, 'w') as h:
                    h.attrs['preprocess_config_json'] = json.dumps(dict(config, saturated_scan=scan))
                    h.create_dataset('support_after_adjustment', data=np.ones((2, 2)))
                os.utime(path, (timestamp, timestamp))
            self.assertEqual(ns['resolve_support_result'](root, config).name,
                             'two_state_point_0093.h5')
            with self.assertRaises(FileNotFoundError):
                ns['resolve_support_result'](root, config, root / 'two_state_point_0050.h5')
        notebook = json.loads(NOTEBOOK.read_text())
        settings = ''.join(notebook['cells'][1]['source'])
        tree = ast.parse(settings)
        key_assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                              and getattr(node.targets[0], 'id', '') == 'MATCH_02_KEYS')
        imported_keys = ast.literal_eval(key_assignment.value.func.value).split()
        for name in ['SATURATED_SCAN', 'LOOP_SCAN', 'RETRIEVAL_POINT_NUMBERS',
                     'LOOP_EXCLUDE_POINT_NUMBERS', 'MASK_REFERENCE_POINT_NUMBER']:
            self.assertNotIn(name, imported_keys)

    def test_png_replaces_legacy_combined_masks_and_preserves_automatic(self):
        from aperiodic_ajajas.result_io import painted_cache_mask
        ns = helpers()
        painted = np.zeros((8, 8), bool)
        painted[3, 4] = True
        ns.update(painted_cache_mask=painted_cache_mask, painted_detector_mask=painted,
                  observation_indices=[None, 0])
        cache = {'saturated_mask': np.ones((8, 8)), 'loop_masks': np.ones((1, 8, 8))}
        for i in range(2):
            np.testing.assert_array_equal(ns['observation_detector_mask'](cache, i), painted)
        automatic = np.zeros((1, 8, 8), bool)
        automatic[0, 1, 2] = True
        cache['loop_automatic_masks'] = automatic
        expected = painted | automatic[0]
        np.testing.assert_array_equal(ns['observation_detector_mask'](cache, 1), expected)
        # Editing away a painted exclusion really removes it on reload.
        ns['painted_detector_mask'] = np.zeros_like(painted)
        np.testing.assert_array_equal(ns['observation_detector_mask'](cache, 1), automatic[0])


if __name__ == '__main__':
    unittest.main()
