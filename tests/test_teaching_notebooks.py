"""Execute the complete teaching sequence and check its scientific API contracts."""
import ast
import contextlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from library import phase_retrieval_universal as universal

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', [
    '01_simple_xmcd.ipynb', '02_hyperspectral.ipynb',
    '03_hysteresis_and_constraints.ipynb', '04_multimode_and_coherence.ipynb',
])
def test_notebook_runs_from_top_and_returns_expected_model(name, monkeypatch):
    monkeypatch.chdir(ROOT / 'notebooks')
    nb = json.loads((ROOT / 'notebooks' / name).read_text())
    ns = {'__name__': '__main__'}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                exec(compile(''.join(cell['source']), name, 'exec'), ns)
    fields = ns['fields']
    assert np.all(np.isfinite(fields))
    assert ns['bsmasks'].shape == ns['holograms'].shape
    assert fields.shape[0] == len(ns['holograms'])
    if name.startswith('01'):
        assert ns['components']['universal_projection_model'] == 'none'
        intensity = abs(fields)**2
        observed = ns['bsmasks'] == 0
        np.testing.assert_allclose(intensity[observed], ns['holograms'][observed],
                                   rtol=1e-6, atol=1e-10)
    elif name.startswith('02'):
        assert ns['components']['universal_projection_model'] == 'svd'
    elif name.startswith('03'):
        assert set(ns['magnetization']) == {'saturated', 'domains'}
        for image in ns['magnetization'].values():
            assert np.all(abs(image) <= 1 + 1e-12)
    else:
        assert fields.shape[1] == 2
        assert ns['recipe']['partial_coherence']
        assert ns['components']['partial_coherence']
        assert np.all(np.isfinite(ns['components']['coherence_kernels']))
        assert np.any(ns['bsmasks'] != 0)
    plt.close('all')


def test_retrieval_implementation_is_self_contained():
    tree = ast.parse(Path(universal.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all('phase_retrieval_core' not in a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert 'phase_retrieval_core' not in (node.module or '')
            assert all(not a.name.startswith('phase_retrieval_core') for a in node.names)
    assert not list((ROOT / 'library').glob('phase_retrieval_core*.py'))


@pytest.mark.parametrize('model', ['svd', 'rank1_spectral'])
def test_multimode_spectral_retrieval_preserves_observed_intensity(model):
    images = np.random.default_rng(4).uniform(1, 5, (2, 12, 12))
    support = np.zeros((12, 12)); support[4:8, 4:8] = 1
    recipe = universal.default_universal_phase_retrieval_recipe()
    recipe.update(projection_model=model, modes=[1, 1, 2],
                  warmup_mode=['ER'], warmup_Nit=[1],
                  inner_mode=['ER'], inner_Nit=[1], outer_iterations=1)
    with contextlib.redirect_stdout(io.StringIO()):
        fields, _, components, masks, _ = universal.universal_phase_retrieval_algorithm(
            images, np.zeros((12, 12)), support,
            ['sample'] * 2, [770., 780.], [+1] * 2, ['beam'] * 2,
            universal_recipe=recipe,
        )
    assert fields.shape == (2, 3, 12, 12)
    np.testing.assert_allclose(np.sum(abs(fields)**2, axis=1)[masks == 0],
                               images[masks == 0], rtol=1e-12, atol=1e-12)
    assert len(components['mode_components']) == 3
