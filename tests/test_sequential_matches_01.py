"""Exercise 03's actual notebook cells against a direct 01 reconstruction."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from library import phase_retrieval_core_unified as pr
from library.phase_retrieval_geometry import recenter_source_support

ROOT = Path(__file__).resolve().parents[1]
NB03 = ROOT / 'aperiodic_ajajas/03_sequential_hysteresis_warmup_poisson.ipynb'
NB01 = NB03.with_name('01_phase_retrieval_2871_2872.ipynb')


class Matches01Tests(unittest.TestCase):
    def test_two_states_match_direct_01_and_preserve_binary_masks(self):
        cells03 = json.loads(NB03.read_text())['cells']
        cells01 = json.loads(NB01.read_text())['cells']
        ns = {}
        with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as directory:
            exec(''.join(cells03[1]['source']), ns)
            directory = Path(directory)
            ns.update(CACHE=directory/'cache.h5', SUPPORT_PNG=directory/'support.png',
                      MASK_PIXEL_PNG=directory/'mask.png', OUTPUT_ROOT=directory/'out',
                      RETRIEVAL_CROP=2, RETRIEVAL_BINNING=2, SUPPORT_DILATION=1,
                      DISPLAY_ROI=np.s_[1:9, 1:9], SHOW_EVERY=100,
                      SAVE_INDIVIDUAL_MODE_PNGS=False, display=lambda *args: None)
            ns['recipe'].update(crop=2, binning=2, number_iterations=[8, 3, 3])
            shape = (32, 32)
            rng = np.random.default_rng(47)
            reference = rng.uniform(10, 20, shape)
            loops = np.stack([reference * 1.1 + rng.uniform(1, 2, shape),
                              reference * 1.3 + rng.uniform(1, 2, shape)])
            rgb = np.zeros((*shape, 3), np.uint8)
            rgb[15:17, 15:17] = (255, 0, 0)
            Image.fromarray(rgb).save(ns['SUPPORT_PNG'])
            rgb[:] = 0
            rgb[7, 9] = (255, 0, 0)  # One pixel must exclude a complete 2x2 bin.
            Image.fromarray(rgb).save(ns['MASK_PIXEL_PNG'])
            automatic = np.zeros((2, *shape), np.uint8)
            automatic[0, 8, 10] = 1
            automatic[1, 20, 20] = 1  # Fixed 01 reference mask, not union of all states.
            with h5py.File(ns['CACHE'], 'w') as h:
                h.attrs['config_json'] = json.dumps(dict(saturated_scan=2871, loop_scan=2872,
                                                         crop_raw_pixels=0, binning=1))
                h.attrs['photon_energy_eV'] = 783.
                h['saturated_hologram'] = reference
                h['loop_holograms'] = loops
                h['loop_file_names'] = np.array(['scan_25.nx', 'scan_26.nx'], dtype=h5py.string_dtype())
                h['loop_current_A'] = [0.1, 0.2]
                h['saturated_automatic_mask'] = np.zeros(shape, np.uint8)
                h['loop_automatic_masks'] = automatic
            exec(''.join(cells03[3]['source']), ns)
            # Run the original 01 input cell independently against the same fixture.
            one = dict(ns, recenter_source_support=recenter_source_support)
            exec(''.join(cells01[3]['source']), one)
            np.testing.assert_array_equal(ns['mask_pixel'], one['mask_pixel'])
            np.testing.assert_array_equal(ns['source_support'], one['support_used'])
            exec(''.join(cells03[5]['source']), ns)
            exec(''.join(cells03[7]['source']), ns)
            for point, loop in zip([25, 26], loops):
                expected = pr.phase_retrieval_algorithm(
                    dict(saturated=reference, loop=loop), one['mask_pixel'],
                    one['support_used'], ns['recipe'])
                with h5py.File(ns['OUTPUT']/f'loop_{point:04d}'/'result.h5') as h:
                    np.testing.assert_allclose(h['warmup_fields'][()], expected['full_coherence']['loop'],
                                               rtol=1e-12, atol=1e-12)
                    np.testing.assert_array_equal(h['poisson_fields'][()], h['warmup_fields'][()])
                    np.testing.assert_array_equal(h['mask'][()], expected['mask_pixel'])
                    np.testing.assert_array_equal(h['effective_mask'][()], expected['bsmasks']['loop'])
                    np.testing.assert_array_equal(h['modal_supports'][()], expected['supportmask'])
                    self.assertTrue(set(np.unique(h['mask'][()])) <= {0, 1})
                    self.assertTrue(set(np.unique(h['modal_supports'][()])) <= {0, 1})
                    self.assertEqual(h['mask'][2, 3], 1)
                    self.assertFalse(h.attrs['poisson_refinement_enabled'])
                    self.assertTrue((h.filename and (Path(h.filename).parent/'diagnostics.png').is_file()))
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
