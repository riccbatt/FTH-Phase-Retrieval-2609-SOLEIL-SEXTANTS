"""Paint masks and standalone result loading preserve pixels and provenance."""
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from PIL import Image
from aperiodic_ajajas.result_io import prepare_painted_detector_mask, result_catalog, load_result_image


class ResultIOTests(unittest.TestCase):
    def test_complex_reconstruction_roundtrip_and_viewer(self):
        from aperiodic_ajajas.result_io import save_complex_reconstructions
        rng = np.random.default_rng(12)
        fields = rng.normal(size=(2, 7, 9)) + 1j*rng.normal(size=(2, 7, 9))
        expected = fields * np.exp(.37j)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'result.h5'
            with h5py.File(path, 'w') as h:
                save_complex_reconstructions(h, {'warmup': fields, 'poisson': fields*2},
                                             lambda f: f*np.exp(.37j), {'focus_um': 0})
            with h5py.File(path) as h:
                self.assertEqual(h['reconstructions/warmup'].dtype, np.dtype('complex128'))
                np.testing.assert_array_equal(h['reconstructions/warmup'][()], expected)
                np.testing.assert_array_equal(h['reconstructions/poisson'][()], expected*2)
            # No Fourier fields are needed when the complex reconstruction is stored.
            image, _, _ = load_result_image(path, kind='reconstruction', mode=1)
            np.testing.assert_array_equal(image, expected[1])
            self.assertTrue(np.any(image.imag != 0))

    def test_viridis_log_template_is_native_size_and_preserves_input(self):
        from aperiodic_ajajas.result_io import save_detector_mask_template
        y, x = np.indices((40, 60))
        hologram = np.exp(15-x/5) + 20*(1+np.sin(y))
        original = hologram.copy()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'mask_pixel_123_pre.png'
            save_detector_mask_template(hologram, path, flatten_sigma=5)
            with Image.open(path) as image:
                self.assertEqual(image.size, (60, 40))
                rgb = np.array(image)
            self.assertTrue(np.any(rgb[..., 0] != rgb[..., 1]))
            self.assertFalse(np.any(np.all(rgb == (255, 0, 0), axis=-1)))
            np.testing.assert_array_equal(hologram, original)
            with self.assertRaises(ValueError):
                save_detector_mask_template(hologram, Path(root)/'mask_pixel_123.png')

    def test_paint_canvas_preserves_edits_and_exact_pixels(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'mask_pixel_123.png'
            template = path.with_name('mask_pixel_123_pre.png')
            detector = np.arange(35).reshape(5, 7)
            with self.assertRaises(FileNotFoundError):
                prepare_painted_detector_mask(detector, path)
            self.assertFalse(path.exists())
            self.assertTrue(template.exists())
            with Image.open(template) as image:
                self.assertEqual(image.size, (7, 5))
                rgb = np.array(image)
            rgb[2, 3] = (255, 0, 0)
            rgb[1, 1] = (255, 255, 255)
            rgb[3, 2] = (254, 0, 0)
            Image.fromarray(rgb).save(path)
            before = path.read_bytes()
            mask = prepare_painted_detector_mask(detector * 2, path)
            self.assertEqual(mask.sum(), 1)
            self.assertTrue(mask[2, 3])
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                prepare_painted_detector_mask(np.ones((7, 5)), path)

    def test_viewer_loads_current_and_saved_complex_roi_in_summary_order(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'summary.csv').write_text('point_number\n9\n2\n')
            expected = np.arange(12).reshape(3, 4) * (1+2j)
            for point in [2, 9]:
                folder = root/f'loop_{point:04d}';folder.mkdir()
                with h5py.File(folder/'result.h5', 'w') as h:
                    h.attrs.update(point_number=point, current_A=point*.1, source_scan=222)
                    h.create_dataset('warmup_fields', data=np.ones((2, 6, 6), complex))
                    group = h.create_group('focused_differences')
                    group.attrs['display_settings_json'] = json.dumps({'color_limits': {'real': [0, 11]}})
                    group.create_dataset('warmup', data=expected)
            records = result_catalog(root)
            self.assertEqual([r['point_number'] for r in records], [9, 2])
            image, metadata, settings = load_result_image(records[0]['path'])
            np.testing.assert_array_equal(image, expected)
            self.assertAlmostEqual(metadata['current_A'], .9)
            self.assertEqual(metadata['source_scan'], 222)
            self.assertEqual(settings['color_limits']['real'], [0, 11])


if __name__ == '__main__':
    unittest.main()
