"""Execute preparation and reconstruction cells against actual synthetic HDF5/PNG inputs."""
import ast
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from library.mask_store import save_red_mask_png
from library.scan_io import DATASET

ROOT = Path(__file__).resolve().parents[1]


def execute_cell(source, namespace, settings):
    # Change only user settings; execute all I/O, metadata and numerical code as written.
    for statement in ast.parse(source).body:
        targets = statement.targets if isinstance(statement, ast.Assign) else []
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if len(names) == 1 and names[0] in settings:
            namespace[names[0]] = settings[names[0]]
        else:
            exec(compile(ast.Module(body=[statement], type_ignores=[]), '<notebook>', 'exec'), namespace)


def run_notebook(path, settings, short_RL=False):
    namespace = {}
    notebook = json.loads(path.read_text())
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            execute_cell(''.join(cell['source']), namespace, settings)
            if path.name == '01_reconstruct.ipynb' and index == 5 and short_RL:
                for stage in namespace['recipe']['stages']:
                    stage.update(it=4, physical_it=2, physical_every=2)
                    if stage['RL_it']:
                        stage.update(RL_it=2, RL_every=2)
    return namespace


def source_masks():
    support = np.zeros((64, 64), np.uint8)
    support[30:35, 30:35] = 1
    for row, col in [(23, 23), (23, 39), (39, 23)]:
        support[row:row+2, col:col+2] = 1
    mask = np.zeros_like(support); mask[0, 2] = 1
    return support, mask


def write_detector(path, frames, exposure=1.):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as handle:
        handle.create_dataset(DATASET, data=frames)
        handle.create_dataset('entry/instrument/detector/exposure', data=exposure)


@pytest.mark.parametrize('RL', [False, True])
@pytest.mark.parametrize('dataset', ['hyperspectral', '2601_XPCS'])
def test_preparation_to_reconstruction_export(tmp_path, monkeypatch, dataset, RL):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(plt, "show", lambda: None)
    support, mask = source_masks()
    masks = tmp_path/'masks'; masks.mkdir()
    cache = tmp_path/'prepared.h5'
    rng = np.random.default_rng(52)
    image = rng.uniform(100., 120., (64, 64))
    settings = dict(MASK_ROOT=masks, CACHE=cache)
    if dataset == 'hyperspectral':
        data_root = tmp_path/'maxiv'
        (data_root/'raw').mkdir(parents=True); (data_root/'proc').mkdir()
        pairs = [(10, 11, (12, 13), 1., 'sample'), (20, 21, (22, 23), 1., 'sample')]
        for pair, energy in zip(pairs, [700., 710.]):
            for scan in pair[:2]:
                with h5py.File(data_root/'raw'/f'2409_maxiv_{scan:05d}.h5', 'w') as handle:
                    snapshot = handle.create_group(f'entry{scan}/measurement/pre_scan_snapshot')
                    for name, value in dict(energy=energy, helicity=1 if scan == pair[0] else -1, magOOP=0.).items():
                        snapshot.create_dataset(name, data=[value])
            for scan in pair[:2] + pair[2]:
                with h5py.File(data_root/'proc'/f'scan_{scan:04d}_avg.h5', 'w') as handle:
                    handle.create_dataset('image', data=image if scan in pair[:2] else np.full((64,64), 10.))
        save_red_mask_png(masks/'supportmask.png', support)
        save_red_mask_png(masks/'mask_pixel.png', mask)
        settings.update(DATA_ROOT=data_root, PAIRS=pairs, SUPPORT_FILE=tmp_path/'unused.h5',
                        DETECTOR_CENTER=(32,32), BINNING=1, CROP=0, USE_SAVED_PIXEL_MASK=True)
    else:
        raw = tmp_path/'raw'; raw.mkdir()
        with h5py.File(raw/'00263.nxs', 'w') as handle:
            group = handle.create_group('scan/instrument/collection')
            for name, value in dict(point_nb=[0,1,2], m_caena=[1.,0.,-1.], mono=[710.]).items():
                group.create_dataset(name, data=value)
        for point in range(3):
            frames = np.stack([image*(1-.05*point), image*(1-.05*point)])
            if point == 1:
                frames[0, 1, 1] = 65535  # Must survive streaming, alignment and caching as an exclusion.
            write_detector(raw/'00263'/'marana'/f'frame_{point:04d}.nx', frames)
        write_detector(raw/'00262'/'marana'/'frame_0000.nx', np.full((2,64,64), 10.))
        save_red_mask_png(masks/'supportmask.png', support)
        save_red_mask_png(masks/'mask_pixel.png', mask)
        settings.update(RAW_ROOT=raw, ALIGNMENT_REFERENCE=0, ALIGNMENT_MAX_SHIFT=2.,
                        ALIGNMENT_UPSAMPLE_FACTOR=2, ALIGNMENT_ROI=None, CROP=0, BINNING=1,
                        AVERAGE_ROOT=tmp_path/'averaged', PREVIEW_ROOT=tmp_path/'previews')
    run_notebook(ROOT/'notebooks'/dataset/'00_prepare.ipynb', settings)
    assert cache.is_file()
    with h5py.File(cache) as handle:
        assert np.isfinite(handle['holograms'][()]).all()
        if dataset == '2601_XPCS':
            assert handle['mask_pixel'][1, :3, :3].any()
            assert handle['material_mask'][()].any() and 'state' in handle
            assert json.loads(handle.attrs['config_json'])['alignment']['enabled']
    output = tmp_path/'result.h5'
    reconstructed = run_notebook(ROOT/'notebooks'/dataset/'01_reconstruct.ipynb',
        dict(CACHE=cache, OUTPUT=output, POINT_INDICES=[0,1,2], RL=RL,
             SATURATED_POINTS={(263,0):1., (263,2):-1.}), short_RL=RL)
    assert np.isfinite(reconstructed['fields']).all()
    assert len(reconstructed['recipe']['stages']) == (4 if RL else 2)
    assert reconstructed['fields'].shape == (4 if dataset == 'hyperspectral' else 3, 2, 64, 64)
    assert output.is_file() and output.with_suffix('.pdf').is_file()
    np.testing.assert_allclose(reconstructed['detector_errors']['final'], reconstructed['errors']['detector'][-1])
    with h5py.File(output) as handle:
        assert {'fields', 'initial_fields', 'metadata', 'recipe', 'components', 'paper_analysis'} <= set(handle)
        assert {'state', 'energy', 'polarization', 'illumination'} <= (set(handle['metadata']) | set(handle['metadata'].attrs))
        if RL:
            assert 'RL_kernels' in handle['components']
    plt.close('all')
