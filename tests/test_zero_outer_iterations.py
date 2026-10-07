"""Regression checks for warmup-only sequential retrieval."""
import contextlib
import io
import numpy as np
import pytest
from library import phase_retrieval_universal as universal

# The former unified stage API now lives in the universal module.
stage = universal

@pytest.mark.parametrize('verify,defaults', [
    (universal._verify_recipe, universal.default_universal_phase_retrieval_recipe),
    (universal._verify_multi_energy_recipe,
     universal.default_multi_energy_phase_retrieval_recipe),
])
def test_outer_iterations_accepts_zero_and_rejects_invalid_counts(verify, defaults):
    recipe = defaults()
    recipe['outer_iterations'] = 0
    verify(recipe, 2)
    for value in (-1, 0.5, True):
        recipe['outer_iterations'] = value
        with pytest.raises(ValueError, match='outer_iterations'):
            verify(recipe, 2)


@pytest.mark.parametrize('modes', [[1], [1, 2]])
def test_zero_outer_loops_reproduces_unified_pos_neg_workflow(modes):
    rng = np.random.default_rng(14)
    pos = rng.uniform(0.5, 3., (12, 12))
    neg = 0.9 * pos + rng.uniform(0.05, 0.2, pos.shape)
    support = np.zeros_like(pos)
    support[4:8, 4:8] = 1
    mask = np.zeros_like(pos)
    mask[6, 6] = 1
    legacy_recipe = stage.default_phase_retrieval_recipe()
    legacy_recipe.update(
        algorithm_list=['HAPRE', 'ER', 'ER'],
        number_iterations=[750, 50, 50], helicity=['pos', 'pos', 'neg'],
        Startimage=[None, 'pos', 'pos'], Startgamma=[None] * 3,
        beta_zero=.5, beta_mode=['arctan', 'const', 'const'],
        alpha_zero=0., alpha_mode='const', RL_its=0, RL_freqs=1e9,
        TV_freqs=1e9, plot_every=1e9, average_img=1, Fourier_last=True,
        output=[False, True, True], modes=modes, return_format='dict',
    )
    recipe = universal.default_universal_phase_retrieval_recipe()
    recipe.update(
        modes=modes, mode_initialization='support_fft',
        startimage_scale_fit='through_origin', warmup_seed_scale_fit='linear',
        warmup_start_from_first=True, warmup_reference_observation=0,
        warmup_mode=['HAPRE', 'ER'], warmup_Nit=[750, 50],
        warmup_reference_mode=['HAPRE', 'ER'], warmup_reference_Nit=[750, 50],
        warmup_other_mode=['ER'], warmup_other_Nit=[50],
        warmup_reference_beta_mode=['arctan', 'const'],
        warmup_other_beta_mode='const', average_img=1,
        outer_iterations=0, projection_model='none',
        constrain_nonphysical_modes_common=False, final_fourier_constraint=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        expected = stage.phase_retrieval_algorithm(pos, neg, mask, support, legacy_recipe)
        fields, warmup, components, _, errors = universal.universal_phase_retrieval_algorithm(
            np.stack([pos, neg]), mask, support,
            ['pos', 'neg'], [783, 783], [1, -1], ['beam', 'beam'],
            universal_recipe=recipe,
        )
    for index, label in enumerate(['pos', 'neg']):
        np.testing.assert_allclose(fields[index], expected['full_coherence'][label],
                                   rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(fields, warmup)
    assert components['final_projection_skipped']
    assert [(s['observation'], s['mode'], s['Nit']) for s in errors['observation_steps']] == [
        (0, 'HAPRE', 750), (0, 'ER', 50), (1, 'ER', 50)]
    assert all(s['outer'] == -1 for s in errors['observation_steps'])


