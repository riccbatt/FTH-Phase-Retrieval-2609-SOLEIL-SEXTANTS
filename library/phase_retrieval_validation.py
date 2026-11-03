"""Small input contracts and observation-label encoding."""
import numpy as np


def normalize_mode_factors(factors):
    factors = np.atleast_1d(np.asarray(factors, dtype=float))
    if factors.ndim != 1 or not len(factors) or np.any(~np.isfinite(factors) | (factors < 0)):
        raise ValueError('modes must be finite nonnegative support factors.')
    if factors[0] != 1:
        raise ValueError('The first physical mode must use support factor 1.')
    return factors.tolist()


def ordered_unique(values):
    names = list(dict.fromkeys(values))
    lookup = {name: index for index, name in enumerate(names)}
    return names, np.array([lookup[value] for value in values])


def normalize_metadata(state, energy, polarization, illumination, n_observations):
    metadata = dict(states=np.asarray(state).reshape(n_observations),
                    energies=np.asarray(energy).reshape(n_observations),
                    polarizations=np.asarray(polarization, float).reshape(n_observations),
                    beams=np.asarray(illumination).reshape(n_observations))
    for key, singular in [('states', 'state'), ('energies', 'energy'), ('beams', 'beam')]:
        metadata[singular+'_names'], metadata[singular+'_indices'] = ordered_unique(metadata[key].tolist())
    return metadata


def observation_weights(weights, n_observations):
    weights = np.ones(n_observations) if weights is None else np.asarray(weights, float).reshape(n_observations)
    if np.any(~np.isfinite(weights) | (weights <= 0)):
        raise ValueError('Observation weights must be finite and positive.')
    return weights


def normalize_saturated_states(states, names):
    states = dict(states or {})
    if set(states) - set(names) or any(sign not in (-1, 1) for sign in states.values()):
        raise ValueError('Saturation anchors must map observed states to +1 or -1.')
    return states


def recipe_inputs(defaults, supplied):
    supplied = supplied or {}
    unknown = supplied.keys() - (defaults.keys() | {'Nmodes'})
    if unknown:
        raise ValueError(f'Unknown recipe keys: {sorted(unknown)}')
    recipe = dict(defaults, **supplied)
    recipe['modes'] = normalize_mode_factors(recipe['modes'])
    recipe['Nmodes'] = len(recipe['modes'])
    return recipe
