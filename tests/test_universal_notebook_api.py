"""Check stage handoffs and active notebook imports through the universal API."""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from library import phase_retrieval_core_unified as stage
from library import phase_retrieval_universal as universal


@pytest.mark.parametrize('modes', [[1], [1, 2]])
def test_stage_recipe_preserves_sequential_coherence_handoffs(modes):
    support = np.zeros((12, 12))
    support[4:8, 4:8] = 1
    image = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(support)))) ** 2
    mask = np.zeros_like(support)
    mask[6, 6] = 1
    recipe = universal.default_phase_retrieval_recipe()
    recipe.update(
        algorithm_list=['ER'] * 4, number_iterations=[2] * 4,
        helicity=['pos', 'neg', 'pos', 'neg'],
        Startimage=[None, 'pos', 'pos', 'neg'],
        Startgamma=[None, None, None, 'pos'],
        RL_its=[0, 0, 1, 1], RL_freqs=[100, 100, 1, 1],
        beta_zero=.5, beta_mode='const', alpha_zero=0., alpha_mode='const',
        TV_freqs=100, plot_every=100, average_img=1, Fourier_last=True,
        output=[False, False, True, True], modes=modes, Nmodes=len(modes),
        return_format='dict',
    )
    with contextlib.redirect_stdout(io.StringIO()):
        expected = stage.phase_retrieval_algorithm(
            image, image * 1.2, mask, support, recipe)
        actual = universal.phase_retrieval_algorithm(
            image, image * 1.2, mask, support, recipe)
    def compare(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                compare(a, b)
        elif isinstance(left, np.ndarray):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
        else:
            assert left == right
    compare(expected, actual)


def test_active_notebooks_use_universal_imports():
    root = Path(__file__).resolve().parents[1]
    paths = list(root.glob('*.ipynb')) + list((root / 'maxiv_phase_test').glob('*.ipynb'))
    paths += list((root / '2601_XPCS').glob('*.ipynb'))
    for path in paths:
        notebook = json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            for line in ''.join(cell['source']).splitlines():
                if line.strip().startswith(('import ', 'from ')):
                    assert 'phase_retrieval_core_unified' not in line, path


def test_unified_is_a_compatibility_alias_and_kernel_is_shared():
    assert stage is universal
    assert universal.multimode_phase_retrieval_kernel is universal.PhaseRtrv_core
    assert universal.PhaseRtrv_core_single.__module__ == universal.__name__
    assert universal.PhaseRtrv_core_multimode.__module__ == universal.__name__


def test_universal_does_not_import_legacy_retrieval_engines():
    import ast
    tree = ast.parse(Path(universal.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all('phase_retrieval_core' not in a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert 'phase_retrieval_core' not in (node.module or '')
            assert all(not a.name.startswith('phase_retrieval_core') for a in node.names)


@pytest.mark.parametrize('model', ['svd', 'rank1_spectral'])
@pytest.mark.parametrize('nmodes', [1, 3])
def test_modal_spectral_api_preserves_observed_intensity(model, nmodes):
    from library import phase_retrieval_core_multienergy_multimode as legacy
    assert legacy.multi_energy_phase_retrieval_algorithm is (
        universal.multi_energy_multimode_phase_retrieval_algorithm
    )
    assert legacy._initialize_modal_fields.__module__ == universal.__name__
    support = np.zeros((12, 12))
    support[4:8, 4:8] = 1
    images = np.random.default_rng(4).uniform(1, 5, (2, 12, 12))
    mask = np.zeros((12, 12))
    mask[6, 6] = 1
    recipe = universal.default_multi_energy_multimode_phase_retrieval_recipe()
    recipe.update(
        Nmodes=nmodes, projection_model=model,
        warmup_mode=['ER'], warmup_Nit=[1],
        inner_mode=['ER'], inner_Nit=[1], outer_iterations=1,
        average_img=1, final_fourier_constraint=True,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        fields, warmup, components, masks, errors = (
            universal.multi_energy_multimode_phase_retrieval_algorithm(
                images, mask, support, multi_energy_recipe=recipe
            )
        )
    assert fields.shape == warmup.shape == ((2, 12, 12) if nmodes == 1 else (2, nmodes, 12, 12))
    assert components['Nmodes'] == nmodes
    assert len(components['mode_components']) == nmodes
    intensity = abs(fields)**2 if nmodes == 1 else np.sum(np.abs(fields)**2, axis=1)
    np.testing.assert_allclose(intensity[masks == 0], images[masks == 0],
                               rtol=1e-12, atol=1e-12)
    assert errors['energy_steps']
