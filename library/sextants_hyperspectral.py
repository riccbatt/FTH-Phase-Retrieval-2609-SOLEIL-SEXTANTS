"""SEXTANTS raw energy pairs -> joint physical-projection reconstruction.

Intensities remain measured counts (or counts/second); only the retrieval engine
converts them to amplitudes. No fixed detector PNG is read by this workflow.
"""
from pathlib import Path
import numpy as np
import h5py
from scipy.ndimage import shift, map_coordinates, label

try:
    from .data_loading import SextantsNexusLoader, Frame, _first_dataset, _text
    from .pipeline_io import acquisition_ids, first_id
    from .image_preprocessing import fit_dark_frame
    from .mask_store import load_red_mask_png
    from . import fth_phase_workflow as wf
    from .hyperspectral_engine import phase_retrieval_universal as engine
except ImportError:
    from data_loading import SextantsNexusLoader, Frame, _first_dataset, _text
    from pipeline_io import acquisition_ids, first_id
    from image_preprocessing import fit_dark_frame
    from mask_store import load_red_mask_png
    import fth_phase_workflow as wf
    from hyperspectral_engine import phase_retrieval_universal as engine


class StreamingSextantsLoader(SextantsNexusLoader):
    """Reduce a raw NeXus stack frame-by-frame instead of allocating it in RAM."""
    def load(self, image_id):
        source = self.path_for(image_id)
        with h5py.File(source, 'r') as handle:
            entries = [key for key in handle if isinstance(handle[key], h5py.Group)]
            expected = f'scan_{int(image_id):04d}'
            entry = expected if expected in handle else entries[0]
            image_path, dataset = self._detector_dataset(handle, entry)
            exposure_path, exposure_values = _first_dataset(handle, self.exposure_paths)
            frame_count = int(np.prod(dataset.shape[:-2]))
            if frame_count < 1:
                raise ValueError(f'Empty detector stack in {source}')
            exposures = np.asarray(exposure_values, dtype=float).reshape(-1)
            if exposures.size == 1:
                exposures = np.repeat(exposures, frame_count)
            if exposures.size != frame_count or not np.isfinite(exposures).all() or np.any(exposures <= 0):
                raise ValueError(f'Invalid frame exposures in {source}')
            image = np.zeros(dataset.shape[-2:], dtype=np.float64)
            for index in np.ndindex(dataset.shape[:-2]):
                image += np.asarray(dataset[index + (slice(None), slice(None))], dtype=np.float64)
            if self.frame_reduction == 'mean':
                image /= frame_count
                exposure = float(exposures.mean())
            elif self.frame_reduction == 'sum':
                exposure = float(exposures.sum())
            else:
                raise ValueError('frame_reduction must be mean or sum')
            metadata = dict(entry=entry, image_path=image_path, exposure_path=exposure_path,
                            detector=_text(dataset.attrs.get('long_name', image_path)),
                            frame_count=frame_count)
            distance = self._ccd_distance(handle[f'{entry}/scan_data'])
            if distance is not None:
                metadata['ccd_dist_path'], metadata['ccd_dist_m'] = distance
            for name, paths in [('energy_eV', self.ENERGY_PATHS), ('polarization_code', self.POLARIZATION_PATHS)]:
                try:
                    path, value = _first_dataset(handle, paths)
                except KeyError:
                    continue
                array = np.asarray(value).reshape(-1)
                if not array.size or not np.all(array == array[0]):
                    raise ValueError(f'{name} changes within raw acquisition {image_id}')
                metadata[name] = array[0].item() if isinstance(array[0], np.generic) else array[0]
                metadata[name + '_path'] = path
        return Frame(str(image_id), image, exposure, source, metadata)


def circle_support(shape, coordinates):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    result = np.zeros(shape, dtype=np.uint8)
    for row, column, radius in coordinates:
        if radius <= 0:
            raise ValueError('Support radii must be positive')
        result |= ((yy-row)**2 + (xx-column)**2 < radius**2).astype(np.uint8)
    if not result.any():
        raise ValueError('The support coordinates are outside the image')
    return result


def material_from_support(support):
    """Largest connected aperture is material; other apertures are references."""
    regions, count = label(np.asarray(support) != 0, np.ones((3, 3)))
    if not count:
        raise ValueError('Support must contain at least one aperture')
    sizes = np.bincount(regions.ravel()); sizes[0] = 0
    return (regions == np.argmax(sizes)).astype(np.uint8)


def resample_common_q(image, mask, scale):
    """Sample detector intensities on a common q grid, conserving pixel flux.

    ``scale`` is input pixels per output pixel. Missing interpolation neighbors
    and out-of-detector samples remain excluded; no beamstop fill is invented.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Reciprocal-space scale must be positive')
    image = np.asarray(image, dtype=float)
    mask = (np.asarray(mask) != 0) | ~np.isfinite(image)
    if np.isclose(scale, 1, rtol=0, atol=1e-12):
        return np.where(mask, 0, image), mask.astype(np.uint8)
    yy, xx = np.indices(image.shape, dtype=float)
    yy = image.shape[0]/2 + scale*(yy-image.shape[0]/2)
    xx = image.shape[1]/2 + scale*(xx-image.shape[1]/2)
    valid = map_coordinates((~mask).astype(float), [yy, xx], order=1,
                            mode='constant', cval=0, prefilter=False)
    values = map_coordinates(np.where(mask, 0, image), [yy, xx], order=1,
                             mode='constant', cval=0, prefilter=False) * scale**2
    missing = valid < 1 - 1e-7
    return np.where(missing, 0, values), missing.astype(np.uint8)


def _average(loader, ids, energy=None, tolerance=.35):
    ids = acquisition_ids(ids)
    image = None; energies = []; exposures = []; distances = []; sources = []; polarizations = []
    for image_id in ids:
        frame = loader.load(image_id)
        values = np.asarray(frame.image, dtype=float)
        if values.ndim != 2:
            raise ValueError(f'Expected a 2D reduced raw image for {image_id}')
        if image is None:
            image = np.zeros_like(values)
        if values.shape != image.shape:
            raise ValueError('Cannot average different detector shapes')
        image += values
        exposures.append(float(frame.exposure)); sources.append(str(frame.source))
        if energy is not None:
            actual = float(frame.metadata['energy_eV'])
            if abs(actual-energy) > tolerance:
                raise ValueError(f'Image {image_id}: energy {actual} != requested {energy} eV; do not average energies')
            energies.append(actual)
            polarizations.append(frame.metadata.get('polarization_code'))
            distances.append(float(frame.metadata['ccd_dist_m']))
    if not np.all(np.isfinite(exposures)) or min(exposures) <= 0:
        raise ValueError('Exposure times must be positive and finite')
    if energy is not None and len({repr(p) for p in polarizations if p is not None}) > 1:
        raise ValueError('Raw polarization metadata differs within an acquisition average')
    if distances and not np.allclose(distances, distances[0], rtol=0, atol=1e-6):
        raise ValueError('Detector distance differs within an average')
    return image/len(ids), dict(ids=ids, energies_eV=energies, exposure_s=float(np.mean(exposures)),
                                ccd_dist_m=distances[0] if distances else None, polarization_codes=polarizations, sources=sources)


def load_energy_scan(pairs, raw_folder, mask_folder, *, detector_center,
                     beamstop_mask_ids, dark_ids=None, dark_fit_recipe=None,
                     pixel_size_m=11e-6, focus=None, supportmask=None,
                     support_coordinates=None, energy_tolerance_eV=.35,
                     exposure_normalize=True, intensity_offset=0., common_q=True,
                     reference_energy_eV=None, reference_distance_m=None, dtype='float32'):
    """Prepare raw energy pairs using only raw files, masks and explicit settings.

    No previous reconstruction or notebook output is read. Geometry defaults to
    metadata in the first positive raw acquisition. Centers, focus, support and
    dark fitting are supplied here by the standalone notebook. Each plus/minus
    input may be one ID or a same-energy list; pair-local settings override defaults.
    """
    if not pairs:
        raise ValueError('Define at least one energy pair')
    loader = StreamingSextantsLoader(raw_folder)
    raw_metadata = dict(loader.load(first_id(pairs[0]['plus'])).metadata)
    reference_energy = float(raw_metadata['energy_eV'] if reference_energy_eV is None else reference_energy_eV)
    ref_distance = float(raw_metadata['ccd_dist_m'] if reference_distance_m is None else reference_distance_m)
    pixel_size_m = float(pixel_size_m)
    if any(not np.isfinite(v) or v <= 0 for v in (reference_energy, ref_distance, pixel_size_m)):
        raise ValueError('Reference energy, detector distance and pixel size must be positive and finite')
    setup = dict(energy=reference_energy, ccd_dist=ref_distance, px_size=pixel_size_m)
    records = []; images = []; masks = []; dark_cache = {}; source_shape = None
    dark_recipe = dark_fit_recipe or dict(rows=[0,600], columns=[0,200], percentile=100, stride=1)
    for point, pair in enumerate(pairs):
        nominal = float(pair['energy_eV'])
        if not np.isfinite(nominal) or nominal <= 0:
            raise ValueError('Energy coordinates must be positive and finite')
        actual_pair_energies = []
        for short, label_name, pol in [('plus','+',1.), ('minus','-',-1.)]:
            raw, meta = _average(loader, pair[short], nominal, energy_tolerance_eV)
            if source_shape is None:
                source_shape = raw.shape
            if raw.shape != source_shape:
                raise ValueError('All observations must share the detector shape')
            requested_dark = pair.get('dark_ids', dark_ids)
            requested_dark = None if requested_dark is None else requested_dark[label_name]
            dark_fit = None
            if requested_dark is not None:
                key = tuple(acquisition_ids(requested_dark))
                if key not in dark_cache:
                    dark_cache[key] = _average(loader, key)[0]
                raw, scale, offset, _, _ = fit_dark_frame(
                    raw, dark_cache[key], slice(*dark_recipe['rows']), slice(*dark_recipe['columns']),
                    percentile=dark_recipe['percentile'], stride=dark_recipe['stride'])
                dark_fit = dict(scale=scale, offset=offset)
            mask_id = pair.get('beamstop_mask_ids', beamstop_mask_ids)[label_name]
            path = Path(mask_folder)/f'mask_beamstop_{int(mask_id)}.png'
            raw_mask = load_red_mask_png(path, raw.shape) != 0
            raw_mask |= ~np.isfinite(raw)
            center = np.asarray(pair.get('center', detector_center), dtype=float)
            if center.shape != (2,) or not np.isfinite(center).all():
                raise ValueError('Detector center must be a finite row/column pair')
            delta = np.asarray(raw.shape)/2 - center
            image = shift(np.where(raw_mask, 0., raw), delta, order=1, mode='constant', cval=0., prefilter=False)
            missing = shift(raw_mask.astype(float), delta, order=1, mode='constant', cval=1., prefilter=False) > 1e-7
            if exposure_normalize:
                image /= meta['exposure_s']
            incident = pair.get('incident_intensity', 1.0)
            incident = float(incident[label_name] if isinstance(incident, dict) else incident)
            if not np.isfinite(incident) or incident <= 0:
                raise ValueError('Incident intensity normalization must be positive and finite')
            image /= incident
            image += float(intensity_offset)
            energy = float(np.mean(meta['energies_eV'])); actual_pair_energies.append(energy)
            q_scale = reference_energy * meta['ccd_dist_m'] / (energy * ref_distance)
            if common_q:
                image, missing = resample_common_q(image, missing, q_scale)
            elif (not np.isclose(q_scale, 1, rtol=1e-5)):
                raise ValueError('Different q sampling requires common_q=True for joint physical imaging')
            missing = (missing != 0) | ~np.isfinite(image) | (image < 0)
            if not np.any((~missing) & (image > 0)):
                raise ValueError(f'No usable positive intensity for pair {point}, {label_name}')
            images.append(np.where(missing, 0., image).astype(dtype))
            masks.append(missing.astype(np.uint8))
            records.append(dict(point=point, label=label_name, polarization=pol,
                energy_eV=nominal, measured_energy_eV=energy, state=pair.get('state','sample'),
                illumination=pair.get('illumination','beam0'), dark_ids=requested_dark,
                dark_fit=dark_fit, beamstop_mask_id=int(mask_id), beamstop_mask_file=str(path),
                center=center.tolist(), q_scale=float(q_scale), incident_intensity=incident, **meta))
        if abs(actual_pair_energies[0]-actual_pair_energies[1]) > energy_tolerance_eV:
            raise ValueError(f'Pair {point}: positive and negative energies differ')
    if supportmask is None:
        supportmask = (circle_support(source_shape, support_coordinates) if support_coordinates is not None
                       else None)
    support = np.asarray(supportmask)
    if support.shape != source_shape or not np.isfinite(support).all() or not np.any(support):
        raise ValueError('Provide a finite nonempty supportmask on the centered source object grid')
    return dict(holograms=np.stack(images), mask_pixel=np.stack(masks), supportmask=(support!=0).astype(np.uint8),
                records=records, im_id=first_id(pairs[0]['plus']),
                experimental_setup={**setup,'energy':reference_energy},
                focus=dict(focus or {}),
                preprocessing=dict(raw_folder=str(raw_folder), mask_folder=str(mask_folder),
                    detector_center=np.asarray(detector_center, dtype=float).tolist(),
                    dark_fit_recipe=dark_recipe, exposure_normalize=exposure_normalize,
                    intensity_offset=float(intensity_offset), common_q=common_q,
                    reference_energy_eV=reference_energy, dtype=dtype))


def default_hyperspectral_recipe(dataset, *, partial_coherence=False, binning=1, crop=0, modes=(1,2)):
    """Joint physical fit: independent coherent warmup, then joint spectral sweeps."""
    recipe = engine.default_universal_phase_retrieval_recipe()
    focus = dataset['focus']; setup = dataset['experimental_setup']
    recipe.update(projection_model='physical_factorized',
        warmup_mode=['HAPRE','ER'], warmup_Nit=[700,90],
        warmup_RL_it=[0,0], warmup_RL_freq=[1e9,1e9],
        inner_mode=['ER'], inner_Nit=[50 if partial_coherence else 5], outer_iterations=50,
        physical_iterations=20, projection_every=len(dataset['records']),
        projection_relaxation=.5, final_projection_relaxation=.5,
        final_fourier_constraint=not partial_coherence,
        modes=list(modes), binning=binning, crop=crop, roi=None,
        recenter_modal_supports=True, mode_support_center='image', mode_initialization='support_fft',
        constrain_nonphysical_modes_common=True,
        material_mask=dataset['supportmask'].astype(bool), fit_material_thickness=True,
        zero_thickness_outside_support=True,
        physical_projection_object_roi=True, physical_phase_reference=True,
        fit_illumination_intensity=False, align_global_phase=False,
        projection_constraints_inside_support_only=True, zero_magnetization_outside_support=True,
        charge_spectral_constraint='free', magnetic_spectral_constraint='free',
        energy_values=np.asarray(list(dict.fromkeys(r['energy_eV'] for r in dataset['records']))),
        clip_magnetization=False, partial_coherence=bool(partial_coherence),
        coherence_kernel_scope='per_observation', shared_coherence_update='sequential',
        RL_it=20 if partial_coherence else 0, RL_freq=20,
        preserve_warmup_masked_intensity=False,
        hologram_intensity_cutoff_vmin=-1, random_seed=0,
        observation_workers=1, fft_workers=1,
        projection_focus_prop_um=float(focus.get('prop_dist',0)),
        projection_focus_phase_rad=float(focus.get('phase',0)),
        projection_focus_setup=dict(ccd_dist=setup['ccd_dist'], px_size=setup['px_size']*binning,
                                    reference_energy_eV=setup['energy'],
                                    dx=float(focus.get('dx',0))/binning, dy=float(focus.get('dy',0))/binning))
    return recipe


def reconstruct(dataset, recipe, *, progress_callback=None):
    """Run the vendored joint engine; preserve full metadata for reproducibility."""
    records = dataset['records']
    if not records or len({r['energy_eV'] for r in records}) < 2:
        raise ValueError('Hyperspectral reconstruction requires at least two energies')
    recipe = dict(recipe)
    if any(recipe.get(key, False) for key in ('fit_illumination_intensity', 'align_global_phase', 'fit_aperture_illumination')):
        raise ValueError('This engine restricts reference-gain fitting to one energy. For this scan use measured incident_intensity per pair, not reference-gain fitting.')
    if recipe.get('projection_model') != 'physical_factorized':
        raise ValueError('Use the physical_factorized projection for this paired dataset')
    if recipe.get('material_mask') is not None and not np.any(recipe['material_mask']):
        raise ValueError('Provide a nonempty allowed material region; the full support is valid for thickness fitting')
    result = engine.universal_phase_retrieval_algorithm(
        dataset['holograms'], dataset['mask_pixel'], dataset['supportmask'],
        state_labels=[r['state'] for r in records], energy_labels=[r['energy_eV'] for r in records],
        polarization_coefficients=[r['polarization'] for r in records],
        illumination_labels=[r['illumination'] for r in records], universal_recipe=recipe,
        progress_callback=progress_callback)
    fields, initial, components, masks, errors = result
    if 'magnetization' not in components:
        raise ValueError('No physical fit was scheduled; enable a joint or final physical projection')
    focus_setup = recipe.get('projection_focus_setup') or {}
    applied_focus = dict(
        prop_dist=recipe['projection_focus_prop_um'], phase=recipe['projection_focus_phase_rad'],
        dx=focus_setup.get('dx',0)*recipe['binning'], dy=focus_setup.get('dy',0)*recipe['binning'],
        projection_setup=recipe.get('projection_focus_setup'))
    return dict(workflow='SEXTANTS_joint_hyperspectral_physical_projection', im_id=dataset['im_id'],
        fields=fields, warmup_fields=initial, components=components, bsmasks=masks, errors=errors,
        recipe=recipe, records=records, preprocessing=dataset['preprocessing'],
        supportmask_source=dataset['supportmask'],
        experimental_setup=dataset['experimental_setup'], focus=applied_focus)


def save_result(result, filename, *, overwrite=False):
    def portable(value):
        if isinstance(value, dict):
            return {str(key): portable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [portable(item) for item in value]
        if isinstance(value, np.ndarray) and value.dtype.kind in 'UO':
            return portable(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        return value
    return wf.save_data_dict(portable(result), filename, overwrite=overwrite)


def detector_prediction(result, index):
    """Forward-model one centered detector intensity, including modal blur."""
    components = result['components']
    gamma = None
    if components.get('partial_coherence', False):
        if 'coherence_kernel' in components:
            gamma = components['coherence_kernel']
        else:
            gamma = components['coherence_kernels'][index]
    return np.maximum(wf.predicted_detector_intensity(result['fields'][index], gamma), 0)


def detector_relative_errors(dataset, result):
    """Evaluate each observation on its measured pixels without duplicating the full cube."""
    recipe = dict(result['recipe'], Nmodes=1)
    errors = []
    for index in range(len(result['records'])):
        measured, mask, _, _, _ = engine.geometry.prepare(
            dataset['holograms'][index:index+1], dataset['mask_pixel'][index:index+1],
            dataset['supportmask'], recipe)
        # The imported preparation API requires at least two observations.
        # Broadcast this one measurement twice to reuse its exact baseline policy.
        # Only two planes are transiently prepared, never the whole energy cube.
        amplitudes, _, _ = engine._prepare_energy_amplitudes(
            np.broadcast_to(measured, (2, *measured.shape[-2:])),
            mask[0] if mask.ndim == 3 else mask,
            hologram_intensity_cutoff_vmin=recipe['hologram_intensity_cutoff_vmin'])
        target = amplitudes[0]**2
        valid = result['bsmasks'][index] == 0
        prediction = detector_prediction(result, index)
        if prediction.shape != target.shape:
            raise ValueError('Detector diagnostic grid differs from the retrieval grid')
        denominator = np.linalg.norm(target[valid])
        errors.append(float(np.linalg.norm((prediction-target)[valid]) / max(denominator, 1e-30)))
    return np.asarray(errors)


AUTO_FROM_DATASET = 'auto_from_dataset'


def resolve_hyperspectral_recipe(dataset, parameters, *, material_region='support'):
    """Require every engine key explicitly; resolve only four documented data bindings.

    This does not fill missing keys with defaults. The default recipe is consulted
    only to verify coverage, so engine updates cannot silently add hidden controls.
    """
    from copy import deepcopy
    expected = set(engine.default_universal_phase_retrieval_recipe())
    supplied = set(parameters)
    if expected != supplied:
        raise ValueError(f'Explicit recipe coverage mismatch: missing={sorted(expected-supplied)}, unknown={sorted(supplied-expected)}')
    recipe = deepcopy(parameters)
    binning = recipe['binning']
    if isinstance(binning, bool) or not isinstance(binning, (int, np.integer)) or binning < 1:
        raise ValueError('Recipe binning must be a positive integer')
    if material_region not in ('support', 'largest_aperture'):
        raise ValueError('material_region must be support or largest_aperture')
    def automatic(key):
        return isinstance(recipe[key], str) and recipe[key] == AUTO_FROM_DATASET
    if automatic('projection_every'):
        recipe['projection_every'] = len(dataset['records'])
    if automatic('energy_values'):
        recipe['energy_values'] = np.asarray(list(dict.fromkeys(r['energy_eV'] for r in dataset['records'])))
    if automatic('material_mask'):
        recipe['material_mask'] = (dataset['supportmask'].astype(bool) if material_region == 'support'
                                   else material_from_support(dataset['supportmask']).astype(bool))
    if automatic('projection_focus_setup'):
        setup, focus = dataset['experimental_setup'], dataset['focus']
        recipe['projection_focus_setup'] = dict(
            ccd_dist=setup['ccd_dist'], px_size=setup['px_size']*binning,
            reference_energy_eV=setup['energy'],
            dx=float(focus.get('dx',0))/binning, dy=float(focus.get('dy',0))/binning)
    unresolved = [key for key, value in recipe.items()
                  if isinstance(value, str) and value == AUTO_FROM_DATASET]
    if unresolved:
        raise ValueError(f'AUTO is not supported for these recipe keys: {unresolved}')
    return recipe
