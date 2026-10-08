"""Streaming scan-263 preparation and saturation-normalized diagnostics."""
from pathlib import Path
import json
import re
import h5py
import numpy as np
from scipy.ndimage import shift, binary_dilation
from matplotlib.path import Path as Polygon
from library import phase_retrieval_geometry as geometry
from library.phase_retrieval_universal import largest_support_component

DATASET = 'entry/instrument/detector/data'

def manifest(raw_root, scans):
    records = []
    for scan in scans:
        stem = f'2601_xpcs_{scan:05d}'
        master = Path(raw_root)/f'{stem}.nxs'
        with h5py.File(master) as f:
            group = f['scan/instrument/collection']
            points = np.asarray(group['point_nb'][()], int)
            current = np.asarray(group['m_caena'][()], float)
            energy = float(np.asarray(group['mono'][()]).ravel()[0])
        if len(set(points)) != len(points) or len(points) != len(current):
            raise ValueError('Nonunique points or mismatched current metadata')
        paths = list((Path(raw_root)/stem/'marana').glob('*.nx'))
        lookup = {}
        for path in paths:
            number = int(re.search(r'_(\d+)\.nx$', path.name).group(1))
            if number in lookup: raise ValueError('Duplicate detector point')
            lookup[number] = path
        if set(lookup) != set(points): raise ValueError('Detector files and point metadata differ')
        for point, amp in zip(points, current):
            with h5py.File(lookup[point]) as f:
                data = f[DATASET]
                exposure = float(np.asarray(f['entry/instrument/detector/exposure'][()]).ravel()[0])
                shape, frames = data.shape[-2:], data.shape[0]
            if exposure <= 0 or frames < 1: raise ValueError('Invalid exposure/frame count')
            records.append(dict(scan=int(scan), point=int(point), current_A=float(amp), energy_ev=energy,
                                path=str(lookup[point]), exposure_s=exposure, frames=frames, shape=list(shape),
                                label=f'scan{scan}_point{point:04d}'))
    if not records: raise ValueError('No observations')
    if len({r['label'] for r in records}) != len(records): raise ValueError('Duplicate scan selection')
    return records


def frame_mean(path, saturation=65535):
    """Keep at most one camera frame in memory; mark saturation in ANY frame."""
    with h5py.File(path) as f:
        ds=f[DATASET]
        total=np.zeros(ds.shape[-2:],float); invalid=np.zeros(total.shape,bool)
        for i in range(ds.shape[0]):
            frame=np.asarray(ds[i],float)
            invalid |= ~np.isfinite(frame) | (frame >= saturation)
            total += np.nan_to_num(frame)
        return total/ds.shape[0], invalid, ds.shape[0]


def load_dark(paths, shape, exposure_s):
    """No paths means NO subtraction. Do not silently omit a mistyped path."""
    if not paths: return None, dict(applied=False, files=[], frame_count=0)
    total=np.zeros(shape,float); count=0
    for path in paths:
        with h5py.File(path) as f:
            exp=float(np.asarray(f['entry/instrument/detector/exposure'][()]).ravel()[0])
        if not np.isclose(exp,exposure_s,rtol=.01,atol=1e-6):
            raise ValueError(f'Dark exposure mismatch: {path}; supply matched darks')
        mean, _, frames=frame_mean(path)
        if mean.shape != tuple(shape): raise ValueError('Dark detector shape mismatch')
        total += mean*frames; count += frames
    return total/count, dict(applied=True,files=[str(p) for p in paths],frame_count=count)


def circles_mask(shape, circles):
    yy,xx=np.indices(shape); result=np.zeros(shape,bool)
    for row,col,radius in circles:
        if radius <= 0: raise ValueError('Circle radius must be positive')
        result |= (yy-row)**2+(xx-col)**2 <= radius**2
    return result


def detector_mask(shape, center, radius, polygons, extra_mask=None):
    yy,xx=np.indices(shape); mask=(yy-center[0])**2+(xx-center[1])**2 <= radius**2
    coords=np.column_stack([xx.ravel(),yy.ravel()])
    for vertices in polygons:
        mask |= Polygon(vertices).contains_points(coords).reshape(shape)
    if extra_mask is not None:
        if np.shape(extra_mask) != tuple(shape): raise ValueError('Extra mask must be on raw detector grid')
        mask |= np.asarray(extra_mask) != 0
    return mask


def estimate_hologram_shift(reference, moving, reference_mask, moving_mask, *,
                             max_shift=10., upsample_factor=20, roi=None):
    """Return corrective (row, column) raw-pixel shift and Pearson scores.

    Masked FFT cross-correlation gives an integer seed. Refine normalized
    cross-correlation on a fixed valid interior to subpixel precision. Intensities
    are linearly interpolated; no periodic wrapping or masked values enter the fit.
    """
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import minimum_filter
    from scipy.optimize import minimize
    if not np.isfinite(max_shift) or max_shift <= 0:
        raise ValueError('alignment max_shift must be positive and finite')
    if isinstance(upsample_factor, bool) or not isinstance(upsample_factor, (int, np.integer)) or upsample_factor < 1:
        raise ValueError('alignment upsample_factor must be a positive integer')
    reference, moving = np.asarray(reference, float), np.asarray(moving, float)
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError('Alignment requires matching 2D holograms')
    valid_ref = ~np.asarray(reference_mask, bool) & np.isfinite(reference)
    valid_mov = ~np.asarray(moving_mask, bool) & np.isfinite(moving)
    if roi is not None:
        if len(roi) != 4 or any(not isinstance(v, (int, np.integer)) for v in roi):
            raise ValueError('alignment roi must be (row_start, row_stop, col_start, col_stop)')
        r0, r1, c0, c1 = roi
        if not (0 <= r0 < r1 <= reference.shape[0] and 0 <= c0 < c1 <= reference.shape[1]):
            raise ValueError('alignment roi is outside the detector')
        region = np.s_[r0:r1, c0:c1]
        reference, moving, valid_ref, valid_mov = [a[region] for a in (reference, moving, valid_ref, valid_mov)]
    margin = int(np.ceil(max_shift)) + 1
    safe = valid_ref & minimum_filter(valid_mov.astype(np.uint8), size=2*margin+1, mode='constant', cval=0).astype(bool)
    rows, cols = np.nonzero(safe)
    if len(rows) < 64:
        raise ValueError('Too few valid pixels for alignment; enlarge ROI or reduce max_shift')
    target = reference[safe]; target = target-target.mean()
    norm = np.linalg.norm(target)
    if norm <= 1e-12:
        raise ValueError('Reference has no contrast for alignment')
    from scipy.ndimage import map_coordinates
    moving = np.where(valid_mov, moving, 0.)
    def score(delta):
        values = map_coordinates(moving, [rows-delta[0], cols-delta[1]], order=1, prefilter=False)
        values -= values.mean()
        denominator = norm*np.linalg.norm(values)
        return float(np.dot(target, values)/denominator) if denominator > 1e-12 else -1.
    if np.std(moving[valid_mov]) <= 1e-12:
        raise ValueError('Moving hologram has no contrast for alignment')
    seed = phase_cross_correlation(np.where(valid_ref, reference, 0.), moving,
        reference_mask=valid_ref, moving_mask=valid_mov, overlap_ratio=.5)[0]
    if np.any(np.abs(seed) > max_shift):
        raise ValueError(f'Estimated alignment shift {seed} exceeds max_shift={max_shift}')
    result = minimize(lambda delta: -score(delta), seed, method='Powell',
        bounds=[(max(-max_shift, v-1.5), min(max_shift, v+1.5)) for v in seed],
        options={'xtol': 1/upsample_factor, 'ftol': 1e-9})
    if not result.success or np.any(np.abs(result.x) >= max_shift-1/upsample_factor):
        raise ValueError('Alignment failed or reached maximum shift; inspect ROI and max_shift')
    before, after = score((0., 0.)), score(result.x)
    if after < before:
        return np.zeros(2), np.array([before, before])
    return result.x, np.array([before, after])


def prepare_cache(records, output, *, dark_paths, center, crop, binning, support_circles,
                  beamstop_radius, polygons, saturation=65535, extra_mask=None, support_override=None,
                  align_holograms=False, alignment_reference=0, alignment_max_shift=10.,
                  alignment_upsample_factor=20, alignment_roi=None):
    shape=tuple(records[0]['shape'])
    if any(tuple(r['shape']) != shape for r in records): raise ValueError('Inconsistent detector shapes')
    if dark_paths and not np.allclose([r['exposure_s'] for r in records], records[0]['exposure_s'],rtol=.01):
        raise ValueError('Use separately matched darks for different exposure groups')
    dark,dark_info=load_dark(dark_paths,shape,records[0]['exposure_s'])
    support=(circles_mask(shape,support_circles) if support_override is None else np.asarray(support_override,bool))
    if support.shape != shape: raise ValueError('Support override must use full object grid')
    support,translation=geometry.recenter_source_support(support,crop,binning)
    base=detector_mask(shape,center,beamstop_radius,polygons,extra_mask)
    movement=np.asarray(shape)//2-np.asarray(center)
    if not isinstance(align_holograms, bool): raise ValueError('align_holograms must be bool')
    if not isinstance(alignment_reference, int) or not 0 <= alignment_reference < len(records):
        raise ValueError('alignment_reference must index an acquisition')
    if align_holograms:
        ref_record = records[alignment_reference]
        ref_raw, ref_invalid, _ = frame_mean(ref_record['path'], saturation)
        reference = (ref_raw if dark is None else ref_raw-dark)/ref_record['exposure_s']
        reference_mask = base | ref_invalid

    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix('.partial.h5')
    config=dict(center=list(center),crop=crop,binning=binning,support_circles=support_circles,
                support_translation=translation,beamstop_radius=beamstop_radius,polygons=polygons,
                saturation=saturation,dark=dark_info,normalization='mean frame / exposure seconds; bins summed',
                support_override=support_override is not None,extra_mask=extra_mask is not None,
                alignment=dict(enabled=align_holograms, reference_index=alignment_reference,
                    max_shift_raw_pixels=alignment_max_shift, upsample_factor=alignment_upsample_factor,
                    roi=alignment_roi, shift_convention='corrective row, column; raw detector pixels',
                    method='masked FFT cross-correlation with subpixel normalized-correlation refinement'))
    with h5py.File(partial,'w') as h:
        h.attrs['config_json']=json.dumps(config)
        h.attrs['manifest_json']=json.dumps(records)
        h.create_dataset('point',data=[r['point'] for r in records]);h.create_dataset('scan',data=[r['scan'] for r in records])
        h.create_dataset('current_A',data=[r['current_A'] for r in records]);h.create_dataset('energy_ev',data=[r['energy_ev'] for r in records])
        h.create_dataset('frame_count',data=[r['frames'] for r in records])
        h.create_dataset('exposure_s',data=[r['exposure_s'] for r in records])
        h.create_dataset('state_labels',data=np.asarray([r['label'] for r in records],dtype=h5py.string_dtype()))
        h.create_dataset('raw_detector_mask',data=base,compression='gzip')
        h.create_dataset('source_supportmask',data=support,compression='gzip')
        if dark is not None: h.create_dataset('dark_mean',data=dark,compression='gzip')
        shifts = h.create_dataset('alignment_shifts_raw_pixels', data=np.zeros((len(records), 2)))
        scores = h.create_dataset('alignment_correlation', data=np.full((len(records), 2), np.nan))
        scores.attrs['columns'] = 'before, after; Pearson correlation on fixed valid ROI'
        for i,record in enumerate(records):
            raw,invalid,_=frame_mean(record['path'],saturation)
            corrected=raw if dark is None else raw-dark
            delta = np.zeros(2)
            if align_holograms:
                if i == alignment_reference:
                    scores[i] = [1., 1.]
                else:
                    try:
                        delta, quality = estimate_hologram_shift(reference, corrected/record['exposure_s'],
                            reference_mask, base|invalid, max_shift=alignment_max_shift,
                            upsample_factor=alignment_upsample_factor, roi=alignment_roi)
                    except ValueError as exc:
                        raise ValueError(f'Alignment failed for {record["label"]}: {exc}') from exc
                    scores[i] = quality
            shifts[i] = delta
            total_movement = movement + delta
            # Combine centering and alignment into one interpolation; shift masks too.
            centered=shift(corrected/record['exposure_s'],total_movement,order=1,mode='constant',cval=0,prefilter=False)
            excluded=shift((base|invalid).astype(float),total_movement,order=1,mode='constant',cval=1,prefilter=False)>0
            images,masks,used,_,_=geometry.prepare(centered[None],excluded,support,dict(crop=crop,binning=binning,roi=None))
            mask=(masks != 0)|~np.isfinite(images[0])|(images[0]<=0)
            if i==0:
                grid=images.shape[-2:]
                h.create_dataset('holograms',shape=(len(records),*grid),dtype='f4',chunks=(1,*grid),compression='gzip')
                h.create_dataset('mask_pixel',shape=(len(records),*grid),dtype='bool',chunks=(1,*grid),compression='gzip')
                h.create_dataset('supportmask',data=used)
                h.create_dataset('material_mask',data=largest_support_component(used))
                h.create_dataset('first_raw_mean',data=raw,compression='gzip')
            h['holograms'][i]=images[0];h['mask_pixel'][i]=mask
            print(f'Prepared {i+1}/{len(records)}: {record["label"]}',flush=True)
    partial.replace(output)
    return config


def endpoint_magnetization(objects, positive, negative, material, reference_mask, threshold=.05):
    """Unclipped post-hoc log-transmission estimate; no physical projection.

    Phase-align on reference holes, unwrap in acquisition order, and use the
    two reconstructed saturated endpoints to set the affine magnetic scale.
    Endpoints themselves are calibration, NOT independent validation.
    """
    aligned=np.asarray(objects,complex).copy()
    if not np.any(reference_mask): raise ValueError('Need reference-hole pixels for global phase alignment')
    for i in range(len(aligned)):
        phase=np.angle(np.vdot(aligned[positive][reference_mask],aligned[i][reference_mask]))
        aligned[i] *= np.exp(-1j*phase)
    floor=max(np.max(abs(aligned))*1e-10,1e-30)
    logs=np.log(np.maximum(abs(aligned),floor))+1j*np.unwrap(np.angle(aligned),axis=0)
    relative=logs-logs[positive]
    delta=relative[negative]
    signal=abs(delta); valid=np.asarray(material,bool)&(signal>threshold*np.max(signal[material]))
    denominator=abs(delta)**2
    estimate=np.full(relative.shape,np.nan)
    estimate[:,valid]=1-2*np.real(relative[:,valid]*delta[valid].conj())/denominator[valid]
    residual=relative-(1-estimate)/2*delta
    return estimate,valid,np.abs(residual)


def relative_magnetic_contrast(objects, material, reference_mask, reference=0):
    """Project phase-aligned log changes onto their dominant complex axis.

    Returns contrast in log-transmission arbitrary units, a valid-pixel mask,
    and the orthogonal residual magnitude. The reference state is zero; the
    axis sign is a numerical convention, not an absolute magnetic direction.
    No observed state is assumed to reach saturation.
    """
    aligned = np.asarray(objects, complex).copy()
    material = np.asarray(material, bool)
    reference_mask = np.asarray(reference_mask, bool)
    if not np.any(reference_mask):
        raise ValueError('Need reference-hole pixels for global phase alignment')
    for i in range(len(aligned)):
        phase = np.angle(np.vdot(aligned[reference][reference_mask], aligned[i][reference_mask]))
        aligned[i] *= np.exp(-1j * phase)
    floor = max(np.max(abs(aligned)) * 1e-10, 1e-30)
    logs = np.log(np.maximum(abs(aligned), floor)) + 1j * np.unwrap(np.angle(aligned), axis=0)
    relative = logs - logs[reference]
    valid = material & np.all(np.isfinite(relative), axis=0)
    contrast = np.full(relative.shape, np.nan)
    residual = np.full(relative.shape, np.nan)
    if np.any(valid):
        values = relative[:, valid]
        matrix = np.column_stack((values.real.ravel(), values.imag.ravel()))
        _, _, axes = np.linalg.svd(matrix, full_matrices=False)
        axis = axes[0]
        if axis[np.argmax(abs(axis))] < 0:
            axis = -axis
        direction = axis[0] + 1j * axis[1]
        contrast[:, valid] = np.real(values * direction.conjugate())
        residual[:, valid] = abs(values - contrast[:, valid] * direction)
    return contrast, valid, residual


def roi_curves(maps, material, centers, radius):
    yy,xx=np.indices(material.shape); means=[]; spreads=[]; masks=[]
    for row,col in centers:
        mask=((yy-row)**2+(xx-col)**2<=radius**2)&material
        if not np.any(mask): raise ValueError(f'ROI {(row,col)} misses material')
        vals=maps[:,mask]
        means.append(np.nanmean(vals,axis=1));spreads.append(np.nanstd(vals,axis=1));masks.append(mask)
    return np.asarray(means),np.asarray(spreads),np.asarray(masks)
