"""Exercise the public initialization/main-loop interface and constraint interactions."""
import numpy as np
import pytest
from library import phase_retrieval_universal as u
from library import phase_retrieval_physical as physical
from library import magnitude_constraints as magnitude
from library import real_space_constraints as real_space


def prepare(initial, stages, **options):
    recipe = dict(state=['p', 'n'], energy=[700., 700.],
                  polarization=[1., 1.], illumination=['beam', 'beam'],
                  stages=[{'beta_mode': 'const', **stage} for stage in stages],
                  modes=[1.] if initial.ndim == 3 else [1.] * initial.shape[1])
    recipe.update(options)
    support = np.zeros((8, 8)); support[2:6, 2:6] = 1
    return u.initialize_phase_retrieval(np.full((2, 8, 8), 4.), np.zeros((8, 8)),
                                       support, recipe, start_fields=initial)


@pytest.mark.parametrize('nmodes', [1, 2])
def test_fourier_and_support_run_every_iteration(monkeypatch, nmodes):
    rng = np.random.default_rng(4)
    shape = (2, 8, 8) if nmodes == 1 else (2, nmodes, 8, 8)
    initial = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    state = prepare(initial, [dict(it=3, mode='ER', physical_it=False)])
    events = []
    fourier, support = magnitude.apply_fourier_constraint, real_space.PROJECTIONS['ER']

    def measured(*args, **kwargs):
        result = fourier(*args, **kwargs)
        np.testing.assert_allclose(np.sum(abs(result)**2, axis=1), 4., atol=1e-12)
        events.append('fourier')
        return result

    def supported(*args):
        result = support(*args)
        np.testing.assert_array_equal(result[..., state['object_support'][0] == 0], 0)
        events.append('support')
        return result

    monkeypatch.setattr(magnitude, 'apply_fourier_constraint', measured)
    monkeypatch.setitem(real_space.PROJECTIONS, 'ER', supported)
    _, snapshot, _, _, _ = u.universal_phase_retrieval_algorithm(state)
    np.testing.assert_array_equal(snapshot, initial)
    assert events == ['fourier', 'support'] * 3


@pytest.mark.parametrize('nmodes', [1, 2])
def test_fourier_leaves_excluded_detector_pixels_unchanged(nmodes):
    field = np.full((2, nmodes, 8, 8), 1+2j)
    invalid = np.zeros((2, 8, 8), bool); invalid[:, 0, 0] = True
    result = magnitude.apply_fourier_constraint(field, np.full((2, 8, 8), 2.), invalid)
    np.testing.assert_array_equal(result[:, :, 0, 0], field[:, :, 0, 0])
    np.testing.assert_allclose(np.sum(abs(result[:, :, 1:])**2, axis=1), 4., atol=1e-12)


@pytest.mark.parametrize('nmodes', [1, 2])
def test_tv_keeps_observations_and_modes_independent(nmodes):
    plane = np.arange(64, dtype=float).reshape(8, 8)
    fields = np.broadcast_to(plane, (2, nmodes, 8, 8))
    result = real_space.total_variation_gradient(fields)
    expected = real_space.total_variation_gradient(plane)
    np.testing.assert_allclose(result, np.broadcast_to(expected, fields.shape))


def test_initialization_does_not_apply_constraints():
    initial = np.full((2, 8, 8), 3+2j)
    reconstruction = prepare(initial, [dict(it=3, physical_it=False)])
    np.testing.assert_array_equal(reconstruction['fields'][:, 0], initial)
    np.testing.assert_array_equal(initial, 3+2j)
    assert len(reconstruction['schedule']) == 3


def test_feedback_persists_across_stage_boundaries():
    rng = np.random.default_rng(7)
    initial = rng.normal(size=(2, 8, 8)) + 1j*rng.normal(size=(2, 8, 8))
    one = prepare(initial, [dict(it=4, mode='HAPRE', physical_it=False)])
    two = prepare(initial, [dict(it=2, mode='HAPRE', physical_it=False)]*2)
    np.testing.assert_allclose(u.universal_phase_retrieval_algorithm(one)[0],
                               u.universal_phase_retrieval_algorithm(two)[0], atol=1e-12)


def test_RL_kernel_normalizes_each_mode():
    kernel = magnitude.RL_kernel(shape=(2, 8, 8))
    np.testing.assert_allclose(kernel.sum(axis=(-2, -1)), 1)
    np.testing.assert_allclose(magnitude.RL_kernel(kernel * np.array([2., 5.])[:, None, None]), kernel)
    with pytest.raises(ValueError): magnitude.RL_kernel(np.zeros((8, 8)))


@pytest.mark.parametrize('nmodes', [1, 2])
def test_RL_keeps_normalized_kernel(nmodes):
    rng = np.random.default_rng(5)
    shape = (2, 8, 8) if nmodes == 1 else (2, nmodes, 8, 8)
    initial = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    reconstruction = prepare(initial, [dict(it=5, mode='ER', RL_it=1, RL_every=1)])
    result, _, components, _, errors = u.universal_phase_retrieval_algorithm(reconstruction)
    gamma = components['RL_kernels']
    assert np.isfinite(result).all()
    assert np.isfinite(errors['detector']).all()
    np.testing.assert_allclose(gamma.sum(axis=(-2, -1)), 1, atol=1e-12)


def test_physical_projection_starts_on_recipe_iteration():
    state = prepare(np.ones((2, 8, 8), complex),
                    [dict(it=2, mode='ER'), dict(it=4, mode='ER', physical_it=2, physical_every=2)],
                    saturated_states={'p': 1., 'n': -1.})
    _, _, components, _, errors = u.universal_phase_retrieval_algorithm(state)
    assert [item['iteration'] for item in errors['projection_steps']] == [4, 6]
    assert np.isfinite(components['fit_residual_rms'])


@pytest.mark.parametrize('strategy', ['sequential', 'average', 'pooled'])
def test_shared_RL_driver(strategy):
    state = prepare(np.ones((2, 2, 8, 8), complex),
                    [dict(it=3, mode='ER', RL_it=1, RL_every=1)],
                    RL_scope='shared', RL_shared_update=strategy,
                    fft_workers=2)
    fields, _, components, _, _ = u.universal_phase_retrieval_algorithm(state)
    assert np.isfinite(fields).all()
    np.testing.assert_allclose(components['RL_kernels'].sum(axis=(-2, -1)), 1, atol=1e-12)


def test_later_stage_can_disable_physical_projection():
    state = prepare(np.ones((2, 8, 8), complex),
                    [dict(it=2, mode='ER', physical_it=2, physical_every=1),
                     dict(it=1, mode='ER', physical_it=False)],
                    saturated_states={'p': 1., 'n': -1.})
    fields, _, components, _, errors = u.universal_phase_retrieval_algorithm(state)
    obj = np.fft.fft2(np.fft.fftshift(fields, axes=(-2, -1)))
    np.testing.assert_allclose(obj[..., state['object_support'][0] == 0], 0, atol=1e-12)
    assert [item['iteration'] for item in errors['projection_steps']] == [1, 2]
    assert 'magnetization' in components


def test_RL_can_start_in_a_later_stage():
    state = prepare(np.ones((2, 8, 8), complex),
                    [dict(it=2, mode='ER', physical_it=False),
                     dict(it=2, mode='ER', RL_it=1, RL_every=1)])
    fields, _, components, _, errors = u.universal_phase_retrieval_algorithm(state)
    assert np.isfinite(fields).all() and np.isfinite(errors['detector']).all()
    np.testing.assert_allclose(components['RL_kernels'].sum(axis=(-2, -1)), 1)


@pytest.mark.parametrize('kind', ['physical', 'RL'])
@pytest.mark.parametrize('count,every', [(False, 1), (2, False), (False, False)])
def test_false_in_either_control_disables_update(monkeypatch, kind, count, every):
    def unexpected(*args, **kwargs):
        pytest.fail(f'{kind} must not run when its count or cadence is False')
    monkeypatch.setattr(u, 'apply_physical_constraint', unexpected)
    monkeypatch.setattr(magnitude, 'refine_RL', unexpected)
    state = prepare(np.ones((2, 8, 8), complex),
                    [dict(it=3, mode='ER', **{kind+'_it': count, kind+'_every': every})])
    _, _, components, _, errors = u.universal_phase_retrieval_algorithm(state)
    assert not errors['projection_steps'] and 'RL_kernels' not in components


def test_stage_counts_and_cadences_reach_the_constraints(monkeypatch):
    iteration = 0
    calls = dict(physical=[], RL=[], TV=[])
    original_fourier = magnitude.apply_fourier_constraint

    def fourier(*args):
        nonlocal iteration
        iteration += 1
        return original_fourier(*args)

    def physical_update(fields, reconstruction, components, relaxation, iterations):
        calls['physical'].append((iteration, iterations))
        return fields, dict(fit_residual_rms=0.)

    def RL_update(fields, targets, kernels, recipe, weights, iterations, fitted):
        calls['RL'].append((iteration, iterations))
        return kernels

    def TV_update(obj):
        calls['TV'].append(iteration)
        return np.zeros_like(obj)

    monkeypatch.setattr(magnitude, 'apply_fourier_constraint', fourier)
    monkeypatch.setattr(u, 'apply_physical_constraint', physical_update)
    monkeypatch.setattr(magnitude, 'refine_RL', RL_update)
    monkeypatch.setattr(u, 'total_variation_gradient', TV_update)
    state = prepare(np.ones((2, 8, 8), complex), [
        dict(it=5, mode='HAPRE', physical_it=3, physical_every=2, RL_it=4, RL_every=3, tv_every=2),
        dict(it=4, mode='ER', physical_it=2, physical_every=3, RL_it=6, RL_every=2),
    ])
    u.universal_phase_retrieval_algorithm(state)
    assert calls == dict(physical=[(2, 3), (4, 3), (8, 2)], RL=[(3, 4), (7, 6), (9, 6)], TV=[2, 4])


def test_false_stage_iteration_count_skips_stage():
    state = prepare(np.ones((2, 8, 8), complex),
                    [dict(it=False, RL_it=50, RL_every=20), dict(it=2, mode='ER')])
    _, _, components, _, errors = u.universal_phase_retrieval_algorithm(state)
    assert len(errors['detector']) == 2 and 'RL_kernels' not in components


def test_default_HAPRE_ER_sequence_and_optional_RL_round():
    from library.phase_retrieval_recipe import compile_stages
    base = u.default_universal_phase_retrieval_recipe()
    assert [(stage['it'], stage['mode']) for stage in base['stages']] == [(700, 'HAPRE'), (50, 'ER')]
    enabled = u.default_universal_phase_retrieval_recipe(RL=True)
    assert [(stage['it'], stage['mode']) for stage in enabled['stages']] == [
        (700, 'HAPRE'), (50, 'ER'), (700, 'HAPRE'), (50, 'ER')]
    assert all(stage['RL_it'] == 50 and stage['RL_every'] == 20 for stage in enabled['stages'][2:])
    controls = compile_stages(enabled)
    assert len(controls) == 1500 and not any(control['RL'] for control in controls[:750])
    assert sum(bool(control['RL_it']) for control in controls[750:]) == 37
    state = prepare(np.ones((2, 8, 8), complex), base['stages'])
    fields, _, _, _, errors = u.universal_phase_retrieval_algorithm(state)
    assert len(errors['detector']) == 750 and np.isfinite(fields).all()


def test_saturation_keys_are_state_values_not_observation_positions():
    state = prepare(np.ones((2, 8, 8), complex),
                    [dict(it=1, mode='ER', physical_it=2, physical_every=1)],
                    state=[10, 30], saturated_states={10: 1., 30: -1.})
    _, _, components, _, _ = u.universal_phase_retrieval_algorithm(state)
    active = state['object_support'][0] != 0
    assert components['state_names'] == [10, 30]
    np.testing.assert_allclose(components['magnetization'][0, active], 1.)
    np.testing.assert_allclose(components['magnetization'][1, active], -1.)
    with pytest.raises(ValueError, match='observed states'):
        prepare(np.ones((2, 8, 8), complex), [dict(it=1, mode='ER')],
                state=[10, 30], saturated_states={0: 1.})
