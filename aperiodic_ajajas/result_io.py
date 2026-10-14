"""Paint-edited detector masks and lightweight sequential-result loading."""
from pathlib import Path
import json
import h5py
import numpy as np
from PIL import Image


def save_detector_mask_template(image, path, *, flatten_sigma=30.0,
                                flatten_strength=0.75, percentiles=(0.5, 99.5)):
    """Native-size viridis view of log intensity, optionally flattened in log space.

    This is display processing only; the input hologram is never modified.
    Only *_pre.png paths are accepted to protect the final painted mask.
    """
    from scipy.ndimage import gaussian_filter
    from matplotlib import colormaps
    path = Path(path)
    if not path.name.endswith('_pre.png'):
        raise ValueError('Template output must end in _pre.png, never the painted mask filename')
    values = np.asarray(image, dtype=float)
    if values.ndim != 2 or not np.any(np.isfinite(values)):
        raise ValueError('Expected a 2-D hologram with finite pixels')
    if not np.isfinite(flatten_sigma) or flatten_sigma < 0:
        raise ValueError('flatten_sigma must be finite and nonnegative')
    if not np.isfinite(flatten_strength) or not 0 <= flatten_strength <= 1:
        raise ValueError('flatten_strength must be between 0 and 1')
    if len(percentiles) != 2 or not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError('Expected ordered percentiles between 0 and 100')
    finite = np.isfinite(values)
    logged = np.log1p(np.maximum(np.where(finite, values, 0), 0))
    logged[~finite] = np.median(logged[finite])
    view = logged.copy()
    if flatten_sigma > 0 and flatten_strength > 0:
        view -= flatten_strength * gaussian_filter(logged, sigma=flatten_sigma)
    low, high = np.percentile(view[finite], percentiles)
    scaled = np.clip((view-low) / max(high-low, 1e-12), 0, 1)
    rgb = colormaps['viridis'](scaled, bytes=True)[..., :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


def prepare_painted_detector_mask(image, path, *, template_path=None):
    """Save a *_pre.png Paint template; load pure red from the final *.png.

    Coordinates are the centered cached detector grid, before retrieval crop/bin.
    Existing edits are never overwritten or resized. Red means excluded.
    """
    path = Path(path)
    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2-D detector image")
    if path.stem.endswith("_pre"):
        raise ValueError("Pass the final mask filename without _pre")
    template = Path(template_path) if template_path is not None else path.with_name(path.stem + "_pre.png")
    if template.resolve() == path.resolve() or not template.name.endswith("_pre.png"):
        raise ValueError("Template must be a separate *_pre.png file")
    if not template.exists():
        save_detector_mask_template(values, template)
    if not path.is_file():
        raise FileNotFoundError(
            f"Paint pure-red exclusions on {template}, Save As {path}, then rerun this cell. "
            "Keep the original image dimensions.")
    with Image.open(path) as source:
        rgb = np.asarray(source.convert('RGB'))
    if rgb.shape[:2] != values.shape:
        raise ValueError(f"{path}: expected {values.shape}, found {rgb.shape[:2]}; do not resize in Paint")
    return np.all(rgb == (255, 0, 0), axis=-1)


def result_catalog(run_dir):
    """Read only metadata, in saved acquisition order when summary.csv exists."""
    import csv
    run_dir = Path(run_dir)
    summary = run_dir / 'summary.csv'
    if summary.is_file():
        with summary.open() as stream:
            rows = list(csv.DictReader(stream))
        paths = [run_dir / ('saturated' if int(row['point_number']) < 0
                           else f"loop_{int(row['point_number']):04d}") / 'result.h5'
                 for row in rows]
    else:
        paths = sorted(run_dir.glob('*/result.h5'))
    records = []
    for path in paths:
        if not path.is_file():
            continue
        with h5py.File(path, 'r') as h:
            records.append(dict(path=path, label=path.parent.name,
                                point_number=int(h.attrs.get('point_number', -1)),
                                current_A=float(h.attrs.get('current_A', np.nan)),
                                source_scan=int(h.attrs.get('source_scan', -1)),
                                has_difference='focused_differences' in h,
                                modes=h['warmup_fields'].shape[0]))
    return records


def load_result_image(path, stage='warmup', kind='difference', mode=0):
    """Read one ROI or one mode, using its saved focus and display settings."""
    from .library import fth_reconstruction as fth
    if stage not in ('warmup', 'poisson'):
        raise ValueError('stage must be warmup or poisson')
    path = Path(path)
    with h5py.File(path, 'r') as h:
        metadata = dict(h.attrs)
        if kind == 'difference':
            if 'focused_differences' not in h:
                raise ValueError('Saturation has no state-minus-saturation difference; select reconstruction')
            group = h['focused_differences']
            image = group[stage][()]
            settings = json.loads(group.attrs['display_settings_json'])
        elif kind == 'reconstruction':
            settings_file = path.parent.parent / 'reconstruction_png_settings.json'
            if settings_file.is_file():
                settings = json.loads(settings_file.read_text())
            elif 'reconstructions' in h:
                settings = json.loads(h['reconstructions'].attrs['display_settings_json'])
            else:
                saved = json.loads(h.attrs['settings_json'])
                settings = dict(focus_um=saved['DISPLAY_PROP_UM'], phase_rad=saved['DISPLAY_PHASE_RAD'],
                                experimental_setup=saved['DISPLAY_SETUP'])
            if 'reconstructions' in h:
                image = h[f'reconstructions/{stage}'][int(mode)]
            else:
                field = h[f'{stage}_fields'][int(mode)]
                image = fth.reconstructCDI(fth.propagate(field, settings['focus_um'] * 1e-6,
                                                       settings['experimental_setup'])) * np.exp(1j*settings['phase_rad'])
        else:
            raise ValueError('kind must be difference or reconstruction')
    return image, metadata, settings


def painted_cache_mask(cache, index, painted):
    """PNG is authoritative; retain only separately recorded automatic exclusions.

    Legacy combined masks may contain JSON polygons and cannot be disentangled.
    Never load them. New caches store automatic masks separately so repainting
    can remove old painted exclusions without losing saturation exclusions.
    """
    name = 'saturated_automatic_mask' if index is None else 'loop_automatic_masks'
    mask = np.asarray(painted, dtype=bool).copy()
    if name in cache:
        automatic = cache[name][()] if index is None else cache[name][index]
        if automatic.shape != mask.shape:
            raise ValueError('Automatic detector mask and painted PNG dimensions differ')
        mask |= automatic != 0
    return mask


def save_complex_reconstructions(handle, stage_fields, transform, settings):
    """Persist full focused object-space modes without discarding either component."""
    group = handle.create_group('reconstructions')
    group.attrs['field_domain'] = 'focused object space, full retrieval grid'
    group.attrs['axes'] = 'mode,row,column'
    group.attrs['display_settings_json'] = json.dumps(settings)
    for stage, fields in stage_fields.items():
        fields = np.asarray(fields)
        if fields.ndim != 3:
            raise ValueError('Expected fields with axes mode,row,column')
        dataset = group.create_dataset(stage, shape=fields.shape, dtype=np.complex128,
                                       chunks=(1, *fields.shape[-2:]), compression='gzip')
        for mode, field in enumerate(fields):
            dataset[mode] = np.asarray(transform(field), dtype=np.complex128)
    return group
