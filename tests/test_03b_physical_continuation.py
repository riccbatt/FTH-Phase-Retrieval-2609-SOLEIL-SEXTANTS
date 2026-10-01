"""Notebook 03b must resume saved fields without a warmup or changed geometry."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from library import phase_retrieval_geometry as geometry
from library import phase_retrieval_core_unified as unified

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'aperiodic_ajajas/03b_physical_from_sequential_warmup.ipynb'
CELLS = json.loads(NOTEBOOK.read_text())['cells']


def run_cell(index, namespace):
    exec(compile(''.join(CELLS[index]['source']), f'03b-cell-{index}', 'exec'), namespace)


def fixture(root, nmodes=2):
    run = root / 'source'
    run.mkdir()
    rng = np.random.default_rng(76)
    support = np.zeros((24, 24), np.uint8)
    support[10:14, 10:14] = 1
    factors = [1, 2][:nmodes]
    supports, _ = geometry.recenter_modal_supports(support, factors, center='image')
    fields = rng.normal(size=(3, nmodes, 24, 24)) + 1j*rng.normal(size=(3, nmodes, 24, 24))
    measured = np.sum(np.abs(fields)**2, axis=1) + 1
    rows = []
    for obs, point in enumerate([-1, 25, 37]):
        name = 'saturated' if point < 0 else f'loop_{point:04d}'
        folder = run / name
        folder.mkdir()
        with h5py.File(folder / 'result.h5', 'w') as h:
            h.attrs.update(point_number=point, current_A=obs*.1, saturated_scan=2923, loop_scan=2924,
                           source_scan=2923 if point < 0 else 2924,
                           energy_eV=783., field_domain='centered Fourier',
                           settings_json=json.dumps(dict(MODES=factors, MODE_SUPPORT_CENTER='image',
                                                         USE_PARTIAL_COHERENCE=False)),
                           recipe_json=json.dumps(dict(modes=factors, mode_support_center='image', RL_its=[0])))
            h['warmup_fields'] = fields[obs]
            h['poisson_fields'] = fields[obs] * 9  # Must never be used as the initial guess.
            h['measured'] = measured[obs]
            h['modal_supports'] = supports
            mask = np.zeros((24, 24), np.uint8)
            mask[4+obs, 6] = 1
            h['mask'] = mask
            h['effective_mask'] = mask
        rows.append(f'{point}\n')
    (run / 'summary.csv').write_text('point_number\n'+''.join(rows))
    return run, fields, supports


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        plt.close('all')
        self.output.__exit__(None, None, None)

    def namespace(self, run, root):
        ns = {}
        run_cell(1, ns)
        ns.update(SOURCE_RUN=run, OUTPUT_ROOT=root/'output', DISPLAY_PROP_UM=0.,
                  SHOW_ROUND_DIAGNOSTICS=False, SAVE_ROUND_FIELDS=True,
                  SAVE_NATIVE_PNGS=True, SHOW_EVERY=100, display=lambda *args: None)
        return ns

    def test_multimode_continuation_skips_warmup_and_saves_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, initial, supports = fixture(root)
            source_hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*/result.h5')}
            ns = self.namespace(run, root)
            run_cell(3, ns); run_cell(5, ns); run_cell(7, ns)
            np.testing.assert_array_equal(ns['start_fields'], initial)
            ns['recipe'].update(inner_mode=['ER'], inner_Nit=[2], average_img=1,
                                outer_iterations=2, physical_iterations=2)
            run_cell(9, ns)
            original_kernel = unified.PhaseRtrv_core
            with patch.object(unified, 'PhaseRtrv_core', wraps=original_kernel) as kernel:
                run_cell(11, ns)
                # Two non-frozen loop states x two rounds. No warmup calls.
                self.assertEqual(kernel.call_count, 4)
                self.assertTrue(all(call.kwargs['Nit'] == 2 for call in kernel.call_args_list))
            np.testing.assert_array_equal(ns['start_fields'], initial)
            np.testing.assert_array_equal(ns['components']['modal_supportmask_used'], supports)
            np.testing.assert_array_equal(ns['fields'][0, 0], initial[0, 0])
            np.testing.assert_allclose(ns['fields'][:, 1], np.broadcast_to(ns['fields'][0, 1], (3, 24, 24)))
            self.assertEqual(len(ns['round_records']), 2)
            self.assertTrue(np.isfinite(ns['fields']).all())
            run_cell(13, ns); run_cell(15, ns)
            with h5py.File(ns['OUTPUT']/'results.h5') as h:
                np.testing.assert_array_equal(h['warmup_fields'][()], initial)
                np.testing.assert_array_equal(h['fields'][()], ns['fields'])
                self.assertFalse(h.attrs['warmup_executed'])
                self.assertIn('magnetization', h['components'])
            for name in ['diagnostics/round_001.png', 'diagnostics/round_002.png',
                         'round_001.h5', 'round_002.h5', 'object_reconstructions.h5',
                         'hysteresis.csv', 'summary.csv', 'native_png/final_loop_0025_mode_00_real.png']:
                self.assertTrue((ns['OUTPUT']/name).is_file(), name)
            for path, digest in source_hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_single_mode_subset_and_missing_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, initial, _ = fixture(root, nmodes=1)
            ns = self.namespace(run, root)
            ns['POINT_NUMBERS'] = [37]
            run_cell(3, ns); run_cell(5, ns); run_cell(7, ns)
            np.testing.assert_array_equal(ns['start_fields'], initial[[0, 2], 0])
            ns['recipe'].update(inner_mode=['ER'], inner_Nit=[1], average_img=1,
                                outer_iterations=1, physical_iterations=1)
            run_cell(9, ns); run_cell(11, ns); run_cell(13, ns)
            self.assertEqual(ns['fields'].shape, (2, 24, 24))
            ns['POINT_NUMBERS'] = [99]
            with self.assertRaisesRegex(ValueError, 'not saved yet'):
                run_cell(3, ns)

    def test_inconsistent_saved_supports_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _, _ = fixture(root)
            ns = self.namespace(run, root)
            run_cell(3, ns)
            with h5py.File(run/'loop_0025/result.h5', 'r+') as h:
                h['modal_supports'][1, 0, 0] = 1
            with self.assertRaisesRegex(ValueError, 'modal supports differ'):
                run_cell(5, ns)


if __name__ == '__main__':
    unittest.main()
