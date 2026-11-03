"""Recipe defaults and scalar schedules. All reconstruction timing is in stages."""
import numpy as np


STAGE_DEFAULTS = dict(mode='HAPRE', physical_it=False, physical_every=False,
                      tv_every=False, tv_step=1e-3, beta_zero=.5, beta_mode='arctan',
                      RL_it=False, RL_every=False)


def default_universal_phase_retrieval_recipe(RL=False):
    """700 HAPRE + 50 ER; optionally repeat both stages with RL fitting."""
    stages = [dict(STAGE_DEFAULTS, it=700), dict(STAGE_DEFAULTS, it=50, mode='ER')]
    if RL:
        stages += [dict(stage, RL_it=50, RL_every=20) for stage in stages]
    return {
        'RL_scope': 'per_observation',
        'RL_shared_update': 'sequential',
        'fft_workers': 1,
        'RL_freeze_after_reference': False,
        'random_seed': 0,
        'hologram_intensity_cutoff_vmin': -1,
        'binning': 1,
        'crop': 0,
        'roi': None,
        'modes': [1.0],
        'mode_support_center': 'image',
        'recenter_modal_supports': True,
        'mode_initialization': 'random_phase',
        'fit_illumination_intensity': False,
        'align_global_phase': False,
        'fit_aperture_illumination': False,
        'object_illumination_mode': 'pooled',
        'aperture_factor_type': 'complex',
        'illumination_intensity_reference_observation': 0,
        'magnetization_mode_groups': None,
        'illumination_mode_groups': None,
        'constrain_nonphysical_modes_common': False,
        'nonphysical_modes_common_relaxation': 1.0,
        'projection_relaxation': 1.0,
        'observation_weights': None,
        'clip_magnetization': True,
        'magnetization_clip_range': (-1.0, 1.0),
        'magnetization_trend': 'none',
        'magnetization_trend_intensity_cutoff': 0.01,
        'magnetization_trend_correction': 'offset',
        'bound_magnetization_median': False,
        'magnetization_median_scope': 'whole',
        'initial_binary_magnetization': False,
        'binary_magnetization': False,
        'binary_magnetization_values': [-1.0, 0.0, 1.0],
        'physical_projection_object_roi': False,
        'physical_phase_reference': False,
        'physical_spatial_weighting': 'exit_wave_intensity',
        'projection_focus_prop_um': 0.0,
        'projection_focus_phase_rad': 0.0,
        'projection_focus_setup': None,
        'projection_focus_integer_wavelength': True,
        'material_mask': None,
        'material_thickness': None,
        'fit_material_thickness': False,
        'saturated_states': None,
        'zero_magnetization_outside_support': True,
        'zero_thickness_outside_support': False,
        'projection_constraints_inside_support_only': True,
        'charge_spectral_constraint': 'free',
        'magnetic_spectral_constraint': 'free',
        'energy_values': None,
        'charge_absorption_part': 'real',
        'magnetic_absorption_part': 'real',
        'charge_response_real_range': None,
        'charge_response_imag_range': None,
        'magnetic_response_real_range': None,
        'magnetic_response_imag_range': None,
        'kk_sign': 1.0,
        'kk_subtract_baseline': True,
        'kk_normalize_input': False,
        'log_floor': 1e-12,
        'stages': stages,
        'state': None,
        'energy': None,
        'polarization': None,
        'illumination': None,
    }


def make_schedule(mode, count, value):
    """Evaluate a beta schedule, or use explicit per-iteration values."""
    if isinstance(mode, np.ndarray):
        return np.asarray(mode, float)
    step = np.arange(count, dtype=float)
    if mode == 'const':
        return np.full(count, value)
    if mode == 'arctan':
        return value + (.5 - np.arctan((step-min(count/2, 700))/(.15*count))/np.pi)*(.98-value)
    if mode == 'smoothstep':
        start, end = count//50, count-count//10
        y = (step-start)/max(end-start, 1.)
        beta = 1-(1-value)*(6*y**5-15*y**4+10*y**3)
        beta[:start], beta[end:] = 1, 0
        return beta
    if mode == 'sigmoid':
        return 1-(1-value)/(1+np.exp(-(step-count//20)/(count*.15)))
    if mode == 'exp':
        return value+(1-value)*(1-np.exp(-((step/7)**3)))
    if mode == 'linear_to_beta_zero':
        return 1+(value-1)*step/count
    if mode == 'linear_to_1':
        return value+(1-value)*step/count
    if mode == 'linear_to_0':
        return np.linspace(value, 0, count)
    if mode == 'steps':
        return np.clip(value*(1-np.floor(step*10/count)/9), 0, value)
    raise ValueError(f'Unknown beta schedule {mode!r}.')


def compile_stages(recipe):
    """Resolve stage-local counts/cadences. False in either member disables that update."""
    allowed = STAGE_DEFAULTS.keys() | {'it', 'projection_relaxation'}
    schedule = []
    for supplied in recipe['stages']:
        unknown = supplied.keys() - allowed
        if unknown:
            raise ValueError(f'Unknown stage keys: {sorted(unknown)}')
        stage = dict(STAGE_DEFAULTS, **supplied)
        count = stage['it']
        if not count:
            continue
        if count < 1:
            raise ValueError('it must be positive or False.')
        beta = make_schedule(stage['beta_mode'], count, stage['beta_zero'])
        relaxation = stage.get('projection_relaxation', recipe['projection_relaxation'])
        physical = bool(stage['physical_it'] and stage['physical_every'] and relaxation)
        RL = bool(stage['RL_it'] and stage['RL_every'])
        for i in range(1, count + 1):
            schedule.append(dict(
                mode=stage['mode'], beta=beta[i-1],
                physical_it=stage['physical_it'] if physical and i % stage['physical_every'] == 0 else False,
                projection_relaxation=relaxation,
                tv_step=stage['tv_step'] if stage['tv_every'] and i % stage['tv_every'] == 0 else 0,
                RL=RL, RL_start=RL and i == 1,
                RL_it=stage['RL_it'] if RL and i % stage['RL_every'] == 0 else False,
            ))
    return schedule
