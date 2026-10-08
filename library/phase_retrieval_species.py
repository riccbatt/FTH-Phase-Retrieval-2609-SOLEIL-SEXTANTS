"""Nonnegative chemical mixtures for the physical log-transmission projector.

Species responses are dimensionless log-transmission coefficients, not bare
refractive indices. Vacuum anchors observation gauges, not the unrestricted
spatial common field inside material or the identity of blind mixture factors.
"""
import numpy as np


def _engine():
    try:
        from . import phase_retrieval_universal
    except ImportError:
        import phase_retrieval_universal
    return phase_retrieval_universal


_SPECIES_KEYS = {
    'name', 'magnetic', 'density_map', 'fit_density', 'supportmask',
    'charge_response', 'magnetic_response', 'fit_charge_response', 'fit_magnetic_response',
    'saturated_states', 'reference_thickness_m',
    'charge_spectral_constraint', 'magnetic_spectral_constraint',
    'known_charge_beta_spectrum', 'known_charge_delta_spectrum',
    'known_magnetic_beta_spectrum', 'known_magnetic_delta_spectrum',
    'charge_absorption_part', 'magnetic_absorption_part',
    'fit_known_spectrum_scale', 'fit_known_spectrum_offset', 'known_spectrum_normalization',
    'kk_sign', 'kk_subtract_baseline', 'kk_normalize_input',
    'charge_response_real_range', 'charge_response_imag_range',
    'magnetic_response_real_range', 'magnetic_response_imag_range',
}


def validate_species(species):
    if not isinstance(species, (list, tuple)) or not species:
        raise ValueError('chemical_species must be a nonempty list of species dictionaries, or None')
    names = []
    for index, item in enumerate(species):
        if not isinstance(item, dict):
            raise ValueError('Each chemical species must be a dictionary')
        unknown = set(item) - _SPECIES_KEYS
        if unknown:
            raise ValueError(f'Unknown chemical species key(s): {sorted(unknown)}')
        name = item.get('name', f'species_{index}')
        if not isinstance(name, str) or not name or name in names or '/' in name:
            raise ValueError('Chemical species names must be unique nonempty strings without /')
        names.append(name)
        for key in ('magnetic', 'fit_density', 'fit_charge_response', 'fit_magnetic_response',
                    'fit_known_spectrum_scale', 'fit_known_spectrum_offset', 'kk_subtract_baseline', 'kk_normalize_input'):
            if key in item and not isinstance(item[key], bool):
                raise ValueError(f'Species {name}: {key} must be bool')
        if not item.get('magnetic', False) and any(key in item for key in (
                'magnetic_response', 'known_magnetic_beta_spectrum', 'known_magnetic_delta_spectrum', 'saturated_states')):
            raise ValueError(f'Species {name}: magnetic settings require magnetic=True')
    return names


def map_species_recipe(recipe, input_geometry=None, shape=None):
    """Copy and map display-frame species inputs onto the log-object grid."""
    u = _engine()
    if recipe.get('chemical_species') is None:
        return recipe
    validate_species(recipe['chemical_species'])
    result = dict(recipe)
    def mapped(value, mask=False):
        if input_geometry is not None:
            local = dict(material_mask=None, material_thickness=value, fit_material_thickness=False)
            # Reuse the exact object-grid crop/bin transform used for thickness.
            array = u._recipe_material_thickness(local, input_geometry)
        else:
            array = u._validate_material_thickness(value, shape).copy()
        return np.fft.fftshift(array != 0 if mask else array)
    result['chemical_species'] = []
    for item in recipe['chemical_species']:
        item = dict(item)
        for key in ('density_map', 'supportmask'):
            if item.get(key) is not None:
                item[key] = mapped(item[key], mask=key == 'supportmask')
        result['chemical_species'].append(item)
    if recipe.get('chemical_reference_mask') is not None:
        result['chemical_reference_mask'] = mapped(recipe['chemical_reference_mask'], mask=True)
    return result


def project_species(log_objects, metadata, recipe, weights, material_thickness,
                    magnetization_supportmask, projection_supportmask, thickness_supportmask,
                    saturated_states, iterations, relaxation, return_components):
    u = _engine()
    binary_values = (u._binary_magnetization_values(recipe)
                     if recipe.get('binary_magnetization', False) else None)
    entries = recipe['chemical_species']
    names = validate_species(entries)
    if not isinstance(recipe['chemical_reference_alignment'], bool):
        raise ValueError('chemical_reference_alignment must be bool')
    seed = recipe['chemical_initialization_seed']
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('chemical_initialization_seed must be a nonnegative integer')
    nobs, *shape = log_objects.shape
    shape = tuple(shape)
    ne, ns, nb = (len(metadata[key]) for key in ('energy_names', 'state_names', 'beam_names'))
    ei, si, bi = (metadata[key] for key in ('energy_indices', 'state_indices', 'beam_indices'))
    polarization = metadata['polarizations']
    thickness = u._validate_material_thickness(material_thickness, shape)
    thickness = np.ones(shape) if thickness is None else thickness.copy()
    allowed = thickness > 0
    if recipe['zero_thickness_outside_support']:
        if thickness_supportmask is None:
            raise ValueError('thickness_supportmask is required for zero_thickness_outside_support')
        allowed &= u._normalize_projection_supportmask(thickness_supportmask, shape)
    projection = u._normalize_projection_supportmask(projection_supportmask, shape)
    if projection is not None:
        allowed &= projection
    magnetic_support = u._normalize_projection_supportmask(magnetization_supportmask, shape)
    if not np.any(allowed):
        raise ValueError('Chemical species aperture contains no material pixels')
    reference = recipe.get('chemical_reference_mask')
    if reference is None:
        reference = thickness == 0 if material_thickness is not None else np.zeros(shape, bool)
        if projection is not None:
            reference &= projection
    else:
        if not np.all(np.isfinite(reference)):
            raise ValueError('chemical_reference_mask must be finite')
        reference = u._normalize_projection_supportmask(reference, shape)
        if np.any(reference & (thickness > 0)):
            raise ValueError('chemical_reference_mask must contain only zero-material pixels')
        if projection is not None and np.any(reference & ~projection):
            raise ValueError('chemical_reference_mask must lie inside the projection support')
    if recipe['chemical_reference_alignment'] and recipe.get('chemical_reference_mask') is not None and not np.any(reference):
        raise ValueError('chemical_reference_mask is empty')
    target = np.asarray(log_objects, complex).copy()
    offsets = np.zeros(nobs, complex)
    # A spatially varying vacuum field is permitted. Only the scalar observation
    # gauge is aligned, within each shared beam group. Do not align different beams.
    if recipe['chemical_reference_alignment'] and np.any(reference):
        for beam in range(nb):
            obs = np.flatnonzero(bi == beam)
            base = target[obs[0], reference].imag
            for a in obs:
                # Choose equivalent log branches only on reference pixels;
                # chemical phase structure elsewhere is not rewrapped here.
                target[a].imag[reference] = base + np.angle(np.exp(1j*(target[a, reference].imag-base)))
            means = np.mean(target[obs][:, reference], axis=1)
            offsets[obs] = means - np.average(means, weights=weights[obs])
            target[obs] -= offsets[obs, None, None]
    elif recipe.get('physical_phase_reference', False):
        for beam in range(nb):
            for energy in range(ne):
                obs = np.flatnonzero((bi == beam) & (ei == energy))
                if obs.size:
                    base = target[obs[0]].imag
                    target.imag[obs] = base + np.angle(np.exp(1j*(target.imag[obs]-base)))

    count = len(entries)
    rng = np.random.default_rng(recipe['chemical_initialization_seed'])
    density = np.zeros((count, *shape))
    supports = np.zeros_like(density, bool)
    charge = np.zeros((count, ne), complex)
    magnetic = np.zeros_like(charge)
    magnetization = np.zeros((count, ns, *shape))
    is_magnetic = np.array([item.get('magnetic', False) for item in entries])
    fit_density, fit_charge, fit_magnetic, anchors, settings = [], [], [], [], []
    spectral_info = [{} for _ in entries]
    spectral_factors = []
    def spectrum(value, name):
        array = np.asarray(value, complex)
        if array.shape != (ne,) or not np.all(np.isfinite(array)):
            raise ValueError(f'{name} must be finite with shape ({ne},)')
        return array.copy()
    def constrain(values, index, kind):
        local = settings[index]
        factor = spectral_factors[index]
        # KK relates beta and delta, not k(E)*beta and k(E)*delta. Remove
        # the wave-number factor before imposing causality, then restore it.
        result, info = u.constrain_complex_spectrum(
            values/factor, spectral_constraint=local[f'{kind}_spectral_constraint'],
            energy_values=recipe['energy_values'],
            known_beta_spectrum=local.get(f'known_{kind}_beta_spectrum'),
            known_delta_spectrum=local.get(f'known_{kind}_delta_spectrum'),
            absorption_part=local[f'{kind}_absorption_part'], kk_sign=local['kk_sign'],
            kk_subtract_baseline=local['kk_subtract_baseline'], kk_normalize_input=local['kk_normalize_input'],
            known_beta_normalization=local['known_spectrum_normalization'],
            fit_known_beta_scale=local['fit_known_spectrum_scale'], fit_known_beta_offset=local['fit_known_spectrum_offset'])
        result *= factor
        result = u._constrain_response_values(result, local[f'{kind}_response_real_range'], local[f'{kind}_response_imag_range'])
        info['constraint_domain'] = 'refractive_index_response_before_wave_number_factor'
        spectral_info[index][kind] = info
        return result
    for i, item in enumerate(entries):
        supports[i] = allowed
        if item.get('supportmask') is not None:
            supports[i] &= u._normalize_projection_supportmask(item['supportmask'], shape)
        if not np.any(supports[i]):
            raise ValueError(f'Species {names[i]} has an empty material support')
        density[i] = (u._validate_material_thickness(item['density_map'], shape)
                      if item.get('density_map') is not None else thickness*(.5+rng.random(shape)))
        density[i] *= supports[i]
        fit_density.append(item.get('fit_density', item.get('density_map') is None))
        fit_charge.append(item.get('fit_charge_response', item.get('charge_response') is None))
        fit_magnetic.append(item.get('fit_magnetic_response', item.get('magnetic_response') is None))
        if not fit_charge[i] and item.get('charge_response') is None:
            raise ValueError(f'Species {names[i]}: fixed charge response requires charge_response')
        if is_magnetic[i] and not fit_magnetic[i] and item.get('magnetic_response') is None:
            raise ValueError(f'Species {names[i]}: fixed magnetic response requires magnetic_response')
        local = dict(recipe)
        local.update(item)
        # q = k*d0*(beta - i*delta), following the engine's index convention.
        local['kk_sign'] = item.get('kk_sign', -1.)
        if local['charge_absorption_part'] != 'real' or local['magnetic_absorption_part'] != 'real':
            raise ValueError('Species refractive-index constraints require absorption_part=real')
        d0 = item.get('reference_thickness_m')
        if d0 is not None and (isinstance(d0, bool) or not isinstance(d0, (int, float, np.number)) or not np.isfinite(d0) or d0 <= 0):
            raise ValueError('reference_thickness_m must be positive and finite')
        modes = [str(local[f'{kind}_spectral_constraint']).lower() for kind in ('charge', 'magnetic') if kind == 'charge' or is_magnetic[i]]
        needs_axis = any(mode not in ('free', 'none', 'unconstrained') for mode in modes) or d0 is not None
        if needs_axis:
            energies = np.asarray(recipe['energy_values'], float)
            if energies.shape != (ne,) or not np.all(np.isfinite(energies)) or np.any(energies <= 0):
                raise ValueError('Species spectral constraints require positive energy_values for every energy')
            factor = energies*u._EV_TO_WAVENUMBER_PER_METRE*(d0 if d0 is not None else 1.)
        else:
            factor = np.ones(ne)
        for kind in ('charge', 'magnetic'):
            for part in ('beta', 'delta'):
                key = f'known_{kind}_{part}_spectrum'
                if item.get(key) is not None:
                    if d0 is None:
                        raise ValueError('Species known refractive-index spectra require reference_thickness_m')
                elif local.get(key) is not None:
                    # Global settings have already been converted to q-products
                    # by the universal adapter; undo that conversion here.
                    sign = local['kk_sign'] if part == 'delta' else 1.
                    if not sign:
                        raise ValueError('kk_sign must be nonzero for known dispersion')
                    local[key] = np.asarray(local[key])/(factor*sign)
        spectral_factors.append(factor)
        settings.append(local)
        anchor = u._normalize_saturated_states(item.get('saturated_states', saturated_states), metadata['state_names']) if is_magnetic[i] else {}
        anchors.append(anchor)
        if item.get('charge_response') is not None:
            charge[i] = spectrum(item['charge_response'], 'charge_response')
        if is_magnetic[i]:
            magnetic[i] = spectrum(item['magnetic_response'], 'magnetic_response') if item.get('magnetic_response') is not None else 1.
            for s, state in enumerate(metadata['state_names']):
                if state in anchor:
                    magnetization[i, s] = anchor[state]*supports[i]
            if magnetic_support is not None:
                magnetization[i] *= magnetic_support
        if fit_charge[i]:
            charge[i] = constrain(charge[i], i, 'charge')
        if is_magnetic[i] and fit_magnetic[i]:
            magnetic[i] = constrain(magnetic[i], i, 'magnetic')

    def material_terms():
        output = np.zeros_like(target)
        for i in range(count):
            output += density[i][None]*(charge[i, ei, None, None] + polarization[:, None, None]*magnetic[i, ei, None, None]*magnetization[i, si])
        return output
    def fit_common(terms):
        return np.stack([np.average(target[bi == beam]-terms[bi == beam], axis=0, weights=weights[bi == beam]) for beam in range(nb)])
    def centered(values):
        result = values.copy()
        for beam in range(nb):
            obs = np.flatnonzero(bi == beam)
            result[obs] -= np.average(values[obs], axis=0, weights=weights[obs])
        return result
    # Seed free chemical spectra by regressing beam-centered observations on
    # distinct nonnegative spatial seeds. This avoids identical-species starts.
    centered_target = centered(target)
    free = [i for i in range(count) if fit_charge[i] and entries[i].get('charge_response') is None]
    if free:
        basis = density.reshape(count, -1).T
        for energy in range(ne):
            obs = np.flatnonzero(ei == energy)
            values = np.average(centered_target[obs], axis=0, weights=weights[obs]).ravel()
            fixed = [i for i in range(count) if i not in free]
            if fixed:
                values = values-basis[:, fixed] @ charge[fixed, energy]
            charge[free, energy] = np.linalg.lstsq(basis[:, free], values, rcond=None)[0]
        for i in free:
            charge[i] = constrain(charge[i], i, 'charge')

    # State/polarization contrasts provide a complex magnetic axis even when
    # there are no saturation anchors and absorption contrast is zero. Starting
    # every free magnetic response on the real axis would lose pure phase data.
    state_contrast = np.zeros_like(target)
    for beam in range(nb):
        for energy in range(ne):
            obs = np.flatnonzero((bi == beam) & (ei == energy))
            if obs.size:
                state_contrast[obs] = target[obs]-np.average(target[obs], axis=0, weights=weights[obs])
    for i in range(count):
        if not is_magnetic[i] or entries[i].get('magnetic_response') is not None:
            continue
        for energy in range(ne):
            obs = np.flatnonzero(ei == energy)
            values = state_contrast[obs][:, supports[i]]
            moment = np.sum(weights[obs, None]*values**2)
            if abs(moment) > 1e-30:
                axis = np.exp(.5j*np.angle(moment))
                amplitude = np.max(abs(np.real(axis.conjugate()*values)))
                magnetic[i, energy] = axis*max(amplitude, 1e-12)
            numerator, denominator = 0j, 0.
            for beam in range(nb):
                selected = [a for a in obs if bi[a] == beam and metadata['states'][a] in anchors[i]]
                if len(selected) < 2:
                    continue
                coefficients = np.array([polarization[a]*anchors[i][metadata['states'][a]] for a in selected])
                coefficients -= np.average(coefficients, weights=weights[selected])
                for a, coefficient in zip(selected, coefficients):
                    basis = coefficient*density[i]
                    numerator += weights[a]*np.vdot(basis, target[a])
                    denominator += weights[a]*np.vdot(basis, basis).real
            if denominator > 1e-30:
                magnetic[i, energy] = numerator/denominator
        if fit_magnetic[i]:
            magnetic[i] = constrain(magnetic[i], i, 'magnetic')

    history = []
    for iteration in range(iterations):
        terms = material_terms()
        common = fit_common(terms)
        # Nonnegative density coordinate updates eliminate the arbitrary common
        # field by centering BOTH residual and response separately in each beam.
        for i in range(count):
            if not fit_density[i]:
                continue
            response = charge[i, ei, None, None] + polarization[:, None, None]*magnetic[i, ei, None, None]*magnetization[i, si]
            own = density[i][None]*response
            residual = centered(target-terms+own)
            coefficient = centered(np.broadcast_to(response, target.shape))
            numerator = np.sum(weights[:, None, None]*np.real(coefficient.conj()*residual), axis=0)
            denominator = np.sum(weights[:, None, None]*abs(coefficient)**2, axis=0)
            updated = np.maximum(0., np.divide(numerator, denominator, out=density[i].copy(), where=denominator > 1e-30))*supports[i]
            terms += (updated-density[i])[None]*response
            density[i] = updated
        common = fit_common(terms)
        for i in range(count):
            if not is_magnetic[i]:
                continue
            for s, state in enumerate(metadata['state_names']):
                obs = np.flatnonzero(si == s)
                previous = magnetization[i, s].copy()
                if state in anchors[i]:
                    updated = anchors[i][state]*supports[i]
                else:
                    coefficient = polarization[obs, None, None]*magnetic[i, ei[obs], None, None]*density[i]
                    own = coefficient*previous
                    residual = target[obs]-common[bi[obs]]-terms[obs]+own
                    numerator = np.sum(weights[obs, None, None]*np.real(coefficient.conj()*residual), axis=0)
                    denominator = np.sum(weights[obs, None, None]*abs(coefficient)**2, axis=0)
                    updated = np.divide(numerator, denominator, out=previous.copy(), where=denominator > 1e-30)
                    if recipe.get('binary_magnetization', False):
                        updated = u._snap_magnetization(updated, binary_values)
                    elif recipe['clip_magnetization']:
                        updated = np.clip(updated, -1., 1.)
                    updated *= supports[i]
                if magnetic_support is not None:
                    updated *= magnetic_support
                updated *= density[i] > 0
                terms[obs] += polarization[obs, None, None]*magnetic[i, ei[obs], None, None]*density[i]*(updated-previous)
                magnetization[i, s] = updated
        common = fit_common(terms)
        # Joint complex spectral least squares over all fitted species columns.
        columns = [(i, 'charge') for i in range(count) if fit_charge[i]] + [(i, 'magnetic') for i in range(count) if is_magnetic[i] and fit_magnetic[i]]
        for energy in range(ne):
            gram = np.zeros((len(columns), len(columns)), complex)
            rhs = np.zeros(len(columns), complex)
            for a in np.flatnonzero(ei == energy):
                basis = np.array([density[i] if kind == 'charge' else density[i]*polarization[a]*magnetization[i, si[a]] for i, kind in columns])
                if not columns:
                    break
                b = basis.reshape(len(columns), -1)
                residual = target[a]-common[bi[a]]-terms[a]
                for j, (i, kind) in enumerate(columns):
                    residual += basis[j]*(charge[i, energy] if kind == 'charge' else magnetic[i, energy])
                gram += weights[a]*(b.conj() @ b.T)
                rhs += weights[a]*(b.conj() @ residual.ravel())
            if columns:
                coefficients = np.linalg.pinv(gram) @ rhs
                for value, (i, kind) in zip(coefficients, columns):
                    (charge if kind == 'charge' else magnetic)[i, energy] = value
        for i in range(count):
            if fit_charge[i]:
                if str(settings[i]['charge_spectral_constraint']).lower() in ('free', 'none', 'unconstrained'):
                    charge[i] -= np.mean(charge[i])  # Explicit common/chemical offset convention.
                charge[i] = constrain(charge[i], i, 'charge')
            if is_magnetic[i] and fit_magnetic[i]:
                magnetic[i] = constrain(magnetic[i], i, 'magnetic')
            # Fix only an unanchored density scale; never rescale fixed spectra.
            known_scale = str(settings[i]['charge_spectral_constraint']).lower().startswith('known') and not settings[i]['fit_known_spectrum_scale']
            if fit_density[i] and fit_charge[i] and not known_scale and (not is_magnetic[i] or fit_magnetic[i]):
                scale = np.mean(density[i, supports[i]])
                if scale > 1e-30:
                    density[i] /= scale
                    charge[i] *= scale
                    magnetic[i] *= scale
        terms = material_terms()
        common = fit_common(terms)
        history.append(float(np.sqrt(np.mean(abs((target-common[bi]-terms)[:, allowed])**2))))
    fitted = common[bi]+terms
    apply = allowed.copy()
    if recipe['chemical_reference_alignment']:
        apply |= reference
    projected = log_objects*(1-relaxation)+fitted*relaxation
    # Exterior/reference holes are preserved unless explicitly used as the
    # common vacuum reference. Alignment is likewise restricted to fitted pixels.
    projected = np.where(apply[None], projected, log_objects)
    if not return_components:
        return projected
    # Spectral rank after eliminating the energy-independent common field.
    contrast_design = centered(np.stack([charge[:, ei[a]] for a in range(nobs)]))
    rank_matrix = np.concatenate((contrast_design.real, contrast_design.imag), axis=0)
    rank = int(np.linalg.matrix_rank(rank_matrix))
    scale_anchored = [not fit_density[i] or not fit_charge[i] or (
        str(settings[i]['charge_spectral_constraint']).lower().startswith('known') and not settings[i]['fit_known_spectrum_scale']) for i in range(count)]
    ambiguities = []
    if any(fit_density[i] and fit_charge[i] for i in range(count)):
        ambiguities.append('Blind species mixing/identity is not guaranteed by nonnegativity, vacuum reference, or KK alone.')
    if not all(scale_anchored):
        ambiguities.append('Free density/spectrum scales use mean density = 1 inside each species support.')
    if rank < count:
        ambiguities.append('Chemical spectral contrast is rank deficient after eliminating the common field.')
    if any(is_magnetic[i] and not anchors[i] and fit_magnetic[i] for i in range(count)):
        ambiguities.append('Unanchored magnetic response and magnetization have a scale/sign ambiguity.')
    ambiguities.append('Vacuum fixes observation gauges there; an unrestricted common map inside material retains chemical spectral-offset ambiguity.')
    species = {}
    for i, name in enumerate(names):
        item = dict(density_map=density[i], density_fitted=fit_density[i], supportmask=supports[i],
                    charge_response=charge[i], magnetic=bool(is_magnetic[i]), magnetic_response=magnetic[i],
                    magnetization=magnetization[i], saturated_states=anchors[i],
                    magnetization_by_state={state: magnetization[i, s] for s, state in enumerate(metadata['state_names'])},
                    spectral_info=spectral_info[i], density_scale_anchored=scale_anchored[i],
                    response_units='dimensionless log-transmission coefficient')
        d0 = entries[i].get('reference_thickness_m')
        if d0 is not None:
            if isinstance(d0, bool) or not np.isfinite(d0) or d0 <= 0:
                raise ValueError('reference_thickness_m must be positive and finite')
            energies = np.asarray(recipe['energy_values'], float)
            if energies.shape != (ne,) or not np.all(np.isfinite(energies)) or np.any(energies <= 0):
                raise ValueError('Index conversion requires positive energy_values for every energy')
            kt = energies*u._EV_TO_WAVENUMBER_PER_METRE*d0
            item.update(beta=charge[i].real/kt, delta=-charge[i].imag/kt,
                        magnetic_beta=magnetic[i].real/kt, magnetic_delta=-magnetic[i].imag/kt,
                        refractive_index_scale_anchored=scale_anchored[i], reference_thickness_m=d0)
        species[name] = item
    first = next((i for i in range(count) if is_magnetic[i]), 0)
    components = dict(
        projection_model='physical_factorized', chemical_species_model=True,
        species_names=names, species=species, species_density_maps=density,
        species_charge_response=charge, species_magnetic_response=magnetic,
        species_magnetization=magnetization, species_magnetic=is_magnetic,
        common_log_objects=common, common_log_objects_by_beam={name: common[b] for b, name in enumerate(metadata['beam_names'])},
        common_exit_waves=np.exp(common), material_thickness=thickness,
        material_thickness_fitted=False, charge_response=charge[first], magnetic_response=magnetic[first],
        magnetization=magnetization[first], magnetization_by_state=species[names[first]]['magnetization_by_state'],
        legacy_component_species=names[first], state_names=metadata['state_names'], energy_names=metadata['energy_names'], beam_names=metadata['beam_names'],
        saturated_states=anchors[first], chemical_reference_mask=reference, chemical_reference_offsets=offsets,
        chemical_reference_pixels=int(reference.sum()), chemical_reference_alignment_applied=bool(recipe['chemical_reference_alignment'] and reference.any()),
        chemical_spectral_contrast_rank=rank, chemical_species_count=count,
        chemical_maps_conditionally_identifiable=bool(rank == count and not any(fit_charge)),
        density_scale_anchored=scale_anchored, identifiable=False, ambiguity_notes=ambiguities,
        fit_residual_rms=history[-1], species_fit_history=np.asarray(history),
        fitted_log_objects=fitted, physical_projection_roi_mask=apply,
        binary_magnetization=recipe.get('binary_magnetization', False),
        binary_magnetization_values=binary_values,
        magnetization_bounds=(-1., 1.) if (recipe['clip_magnetization'] or recipe.get('binary_magnetization', False)) else None)
    return projected, components
