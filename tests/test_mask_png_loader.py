from pathlib import Path

import numpy as np

from library.mask_store import load_red_mask_png, mask_png_paths, save_red_mask_png


def test_red_mask_png_round_trip(tmp_path):
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:5, 3:7] = True

    path = tmp_path / "painted_mask.png"
    save_red_mask_png(path, mask)
    loaded = load_red_mask_png(path, expected_shape=mask.shape)

    assert loaded.dtype == np.uint8
    assert loaded.shape == mask.shape
    assert np.array_equal(loaded, (mask.astype(np.uint8)))


def test_mask_png_paths_use_reference_and_painted_names(tmp_path):
    reference_png, painted_png = mask_png_paths(tmp_path, "supportmask", 613)

    assert reference_png == tmp_path / "supportmask_613_reference.png"
    assert painted_png == tmp_path / "supportmask_613.png"
