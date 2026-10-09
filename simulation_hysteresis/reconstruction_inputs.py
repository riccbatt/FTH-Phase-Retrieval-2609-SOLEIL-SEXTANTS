"""Read simulation observations without loading exit waves or truth into retrieval."""
from pathlib import Path
import json
import h5py
import numpy as np

DEFAULT_DATA = Path('/home/riccardo/Desktop/Github/fomocid/outputs/18_magnetic_state_dataset/run_20261009T140226Z_20cf3d5b')


def load_dataset(root=DEFAULT_DATA, source='detected_no_beamstop', helicity='CR', indices=None,
                 beamstop_threshold=0.):
    """Beamstop opacity > threshold is excluded (True); soft edges are excluded too.

    Support is already on the centered object grid, as expected by the solver.
    All selected states retain their original intensity scale; no state normalization.
    """
    if source not in ('detected_no_beamstop', 'ideal', 'holograms_ideal'):
        raise ValueError('Choose detected_no_beamstop or ideal')
    if helicity not in ('CR', 'CL'): raise ValueError('Choose CR or CL')
    if not 0 <= beamstop_threshold < 1: raise ValueError('Threshold must be in [0, 1)')
    paths = sorted(Path(root).glob('*.h5'))
    if not paths: raise FileNotFoundError(f'No HDF5 files in {root}')
    selection = list(range(len(paths))) if indices is None else list(indices)
    if not selection or len(set(selection)) != len(selection) or any(i < 0 or i >= len(paths) for i in selection):
        raise ValueError('Invalid file indices')
    images, masks, labels, energies, truth, records = [], [], [], [], [], []
    saturated = {}; support = None; setup = None
    for i in sorted(selection):
        path = paths[i]
        with h5py.File(path) as f:
            key = f'holograms/{"ideal" if source == "holograms_ideal" else source}/{helicity}'
            if key not in f and source in ('ideal', 'holograms_ideal'):
                key = f'holograms_ideal/{helicity}'
            ds = f[key]
            image = np.asarray(ds[()], float) if ds.ndim == 2 else np.mean(ds[()], axis=0)
            if image.ndim != 2: raise ValueError(f'Expected 2D hologram: {path}')
            current_support = np.asarray(f['support_mask'][()], bool)
            opacity = np.asarray(f['masks/beamstop_mask'][()], float)
            if current_support.shape != image.shape or opacity.shape != image.shape:
                raise ValueError(f'Grid mismatch: {path}')
            if support is None:
                support = current_support
                configuration = json.loads(f.attrs['configuration_json'])
                detector = configuration['detector']
                setup = dict(ccd_dist=float(detector['sample_to_detector_distance']),
                             px_size=float(detector['pixel_size']), energy=float(f.attrs['energy_eV']))
            elif not np.array_equal(support, current_support):
                raise ValueError('Selected files must share support')
            label = str(f.attrs.get('state_name', path.stem))
            if label in labels: raise ValueError('Duplicate state label')
            # Only original file indices 0 and 2 calibrate the magnetic scale.
            # Replicate saturation files remain unconstrained validation states.
            if i == 0: saturated[label] = 1.
            elif i == 2: saturated[label] = -1.
            invalid = ~np.isfinite(image) | (image < 0) | ~np.isfinite(opacity)
            masks.append((opacity > beamstop_threshold) | invalid)
            images.append(np.where(np.isfinite(image), np.maximum(image, 0), 0))
            labels.append(label); energies.append(float(f.attrs['energy_eV']))
            truth.append(float(f.attrs.get('object_mean_mz', np.nan)))
            records.append(dict(index=i, path=str(path), dataset=key))
    if np.ptp(energies) > .25: raise ValueError('Expected single-energy dataset')
    return dict(holograms=np.stack(images), mask_pixel=np.stack(masks), support=support,
                state_labels=labels, saturated=saturated, energy_ev=float(np.mean(energies)),
                setup=setup, truth_mean_mz=np.asarray(truth), records=records)


def plot_input_diagnostics(data, positions=None, max_states=6):
    """Display centered FFT intensity (FTH/autocorrelation), not retrieved exit waves."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import label, center_of_mass
    from scipy.signal import fftconvolve
    images = data['holograms']; masks = data['mask_pixel']; support = data['support']
    if positions is None:
        anchors = [i for i, s in enumerate(data['state_labels']) if s in data['saturated']]
        others = [i for i in range(len(images)) if i not in anchors]
        positions = (anchors + others)[:max_states]
    if not positions or any(i < 0 or i >= len(images) for i in positions):
        raise ValueError('Choose loaded stack positions for diagnostics')
    def fth(image):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    def log_amplitude(array):
        amplitude = abs(array)
        return np.log10(np.maximum(amplitude, max(float(amplitude.max())*1e-8, 1e-30)))
    labels, count = label(support)
    print('Support components on the centered object grid (row, column):')
    for component in range(1, count+1):
        yy, xx = np.where(labels == component)
        print(f'  {len(yy)} pixels; centroid {center_of_mass(support, labels, component)}; '
              f'bounds rows {yy.min()}:{yy.max()+1}, columns {xx.min()}:{xx.max()+1}')
    # Difference vectors between aperture pixels predict the autocorrelation support.
    correlation = fftconvolve(support.astype(float), support[::-1, ::-1].astype(float), mode='full')
    cy, cx = np.array(support.shape)-1
    ny, nx = support.shape
    expected = correlation[cy-ny//2:cy+(ny+1)//2, cx-nx//2:cx+(nx+1)//2] > .5
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    axes[0].imshow(support, cmap='gray', interpolation='nearest')
    axes[0].set_title('Supplied support: object + reference holes')
    axes[1].imshow(support, cmap='gray', interpolation='nearest')
    yy, xx = np.where(support)
    axes[1].set_xlim(max(0, xx.min()-20), min(nx, xx.max()+21))
    axes[1].set_ylim(min(ny, yy.max()+21), max(0, yy.min()-20))
    axes[1].set_title('Support zoom (native grid)')
    axes[2].imshow(expected, cmap='gray', interpolation='nearest')
    axes[2].set_title('Expected FTH support: aperture autocorrelation')
    for ax in axes: ax.set(xlabel='Column', ylabel='Row')
    plt.show()
    fig, axes = plt.subplots(len(positions), 5, figsize=(22, 4*len(positions)),
                             squeeze=False, layout='constrained')
    for row, i in enumerate(positions):
        image = images[i]; invalid = masks[i]
        masked = np.where(invalid, 0., image)
        raw_fth = fth(image); masked_fth = fth(masked)
        panels = [np.log10(1+image), np.ma.array(np.log10(1+image), mask=invalid),
                  log_amplitude(raw_fth), log_amplitude(masked_fth), log_amplitude(raw_fth)]
        titles = ['Hologram log10(1+I)', 'Hologram with binary exclusion mask',
                  'FTH log10 amplitude: original intensity', 'FTH: excluded pixels set to zero',
                  'Original FTH + expected autocorrelation outline']
        for col, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[row, col]
            finite = np.asarray(panel)[np.isfinite(np.asarray(panel))]
            lo, hi = np.percentile(finite, [1, 99.8])
            im = ax.imshow(panel, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
            fig.colorbar(im, ax=ax, shrink=.7)
            ax.set_title(f'{data["state_labels"][i]}\n{title}', fontsize=9)
            ax.set(xlabel='Column', ylabel='Row')
        axes[row, 1].contour(invalid, levels=[.5], colors=['orange'], linewidths=.6)
        axes[row, 4].contour(expected, levels=[.5], colors=['cyan'], linewidths=.6)
        # Native exit-wave support is a separate overlay: FTH contains correlation peaks.
        axes[row, 3].contour(support, levels=[.5], colors=['orange'], linewidths=.6)
    plt.show()
    print('Orange: supplied support (FTH panel) or excluded pixels (hologram panel). Cyan: support autocorrelation.')
    print('FTH is the autocorrelation of the exit wave: displaced object images do not lie in the exit-wave support.')
    print('Masking introduces Fourier ringing; compare original and masked FTH before judging the support.')
    return dict(expected_fth_support=expected, component_count=count)


def audit_saturation_reconstruction(fields, state_labels, saturated, material, *, energy_ev=781.1386599767951, iterations=100, spatial_weighting=None):
    """Compare duplicate fields and constant-response vs local endpoint calibration.

    No fields or model settings are changed. Local endpoint maps are diagnostics,
    not independent validation, and pixels with weak endpoint contrast are omitted.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion
    from library import phase_retrieval_universal as u
    positive = next(i for i,s in enumerate(state_labels) if saturated.get(s) == 1.)
    negative = next(i for i,s in enumerate(state_labels) if saturated.get(s) == -1.)
    primary = fields[:, 0] if fields.ndim == 4 else fields
    objects = np.stack([u.fourier_field_to_display_object(f) for f in primary])
    material = np.asarray(material, bool)
    floor = max(float(abs(objects).max())*1e-12, 1e-30)
    logs = np.log(np.maximum(abs(objects), floor)) + 1j*np.angle(objects)
    # Same nearest-relative-phase branch convention as the physical projector.
    logs.imag[:] = logs[positive].imag + np.angle(np.exp(1j*(logs.imag-logs[positive].imag)))
    delta = (logs[positive]-logs[negative])/2
    relative = logs-(logs[positive]+logs[negative])/2
    reliable = material & (abs(delta) > .05*np.median(abs(delta[material]))) & (abs(delta)>1e-12)
    endpoint = np.full(logs.shape, np.nan)
    endpoint[:, reliable] = np.real(relative[:, reliable]*delta[reliable].conj())/abs(delta[reliable])**2
    print('Local endpoint diagnostic means (unclipped):', np.nanmean(endpoint[:,material],axis=1)[:4])
    print('Identical copies give identical local endpoint estimates even if the scalar-response model fails.')
    recipe = u.default_universal_phase_retrieval_recipe()
    recipe.update(physical_phase_reference=True, physical_spatial_weighting=spatial_weighting)
    rows=[]
    for erosion in (0, 4, 10, 20):
        active = material if erosion == 0 else binary_erosion(material, iterations=erosion)
        if not np.any(active): continue
        compact = logs[:, active, None]
        contrast = delta[active]
        print(f'Erosion {erosion}: anchor half-contrast mean={contrast.mean():.5g}, std={np.std(contrast):.5g}')
        for scope in ('all states', 'saturated observations'):
            selected = (list(range(len(state_labels))) if scope == 'all states' else
                        [i for i,s in enumerate(state_labels) if 'saturated' in s])
            if positive not in selected or negative not in selected: continue
            _, fit = u.project_log_objects_physical(
                compact[selected], [state_labels[i] for i in selected], [energy_ev]*len(selected),
                [1.]*len(selected), ['beam']*len(selected), recipe=recipe,
                material_thickness=np.ones(compact.shape[1:]), saturated_states=saturated,
                iterations=iterations, return_components=True)
            means=fit['magnetization'].mean(axis=(1,2))
            rows.append(dict(erosion=erosion, scope=scope, states=[state_labels[i] for i in selected],
                             mean_mz=means.tolist(), residual=fit['fit_residual_rms']))
            print(scope, 'first four mz:', means[:4], 'log RMS:', fit['fit_residual_rms'])
    roi = u.geometry.support_bounding_roi(material, padding_fraction=.1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for ax, image, title in zip(axes,
            [delta.real, delta.imag, abs(delta-delta[material].mean())],
            ['Anchor half-contrast: log amplitude', 'Anchor half-contrast: phase', 'Deviation from constant response']):
        im=ax.imshow(np.ma.array(image,mask=~material)[roi],cmap='viridis')
        ax.set_title(title);fig.colorbar(im,ax=ax)
    plt.show()
    return dict(refits=rows, endpoint_mean=np.nanmean(endpoint[:,material],axis=1),
                contrast_mean=delta[material].mean(), contrast_std=np.std(delta[material]))
