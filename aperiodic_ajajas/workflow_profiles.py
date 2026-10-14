"""Named sample settings and paths shared by notebooks 00–03 (stdlib only)."""
import copy
import json
import re
from pathlib import Path


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Names must contain only letters, numbers, dots, underscores or hyphens')
    return value


def select_numbers(available, requested=None, step=1, exclude=()):
    """Return positions in available, selecting acquisition numbers, not positions."""
    available = [int(v) for v in available]
    if len(set(available)) != len(available):
        raise ValueError('Duplicate acquisition point numbers')
    if requested is None:
        if type(step) is not int or step < 1:
            raise ValueError('Selection step must be a positive integer')
        result = [i for i, value in enumerate(available) if value not in exclude][::step]
    else:
        if not requested or any(type(v) is not int or v < 0 for v in requested):
            raise ValueError('Selection must be a nonempty list of nonnegative integers')
        if len(set(requested)) != len(requested):
            raise ValueError('Duplicate requested indices')
        missing = set(requested) - set(available)
        if missing:
            raise ValueError(f'Requested indices are unavailable: {sorted(missing)}')
        result = [available.index(v) for v in requested]
    if not result:
        raise ValueError('Selection is empty')
    return result


class WorkflowProfile:
    def __init__(self, root, name, data):
        self.root = Path(root).resolve()
        self.name = _name(name)
        self.data = data
        if data.get('schema_version') != 1:
            raise ValueError('Unsupported profile schema')
        self.run_name = _name(data['run_name'])
        self.output_dir = self.root / 'processed' / self.name / self.run_name
        self.cache = self.output_dir / 'preprocessed.h5'
        self.support_dir = self.root / 'supports' / self.name

    def path(self, value):
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def settings(self, stage):
        values = {**self.data.get('shared', {}), **self.data.get(stage, {})}
        for key in ('RAW_ROOT', 'SUPPORT_PNG', 'DETECTOR_MASK_PNG', 'REFERENCE_POLYGONS',
                    'SUPPORTMASK_OVERRIDE_FILE', 'MATERIAL_MASK_FILE', 'RELATIVE_THICKNESS_FILE'):
            if key in values:
                values[key] = self.path(values[key])
        return values

    def snapshot(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'profile.json').write_text(json.dumps(self.data, indent=2) + '\n')

    def validate_cache(self, config):
        expected = self.settings('preprocess')
        for key, setting in [('saturated_scan', 'SATURATED_SCAN'), ('loop_scan', 'LOOP_SCAN')]:
            if config.get(key) != expected[setting]:
                raise ValueError(f'Cache {key} does not match profile {self.name}; rerun 00')
        if config.get('profile_name') != self.name or config.get('profile') != self.data:
            raise ValueError('Cache profile has changed; rerun 00 or load its saved profile.json')


def load_profile(root, name=None):
    root = Path(root)
    if name is None:
        name = (root / 'profiles' / 'active.txt').read_text().strip()
    name = _name(name)
    return WorkflowProfile(root, name, json.loads((root / 'profiles' / f'{name}.json').read_text()))


def save_profile(root, name, data, *, activate=False, overwrite=False):
    """Save a new profile; explicit overwrite is required to replace an existing one."""
    profile = WorkflowProfile(root, name, copy.deepcopy(data))
    folder = profile.root / 'profiles'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'{profile.name}.json').open('w' if overwrite else 'x') as handle:
        json.dump(profile.data, handle, indent=2)
        handle.write('\n')
    if activate:
        (folder / 'active.txt').write_text(profile.name + '\n')
    return profile


def save_support(profile, mask, name, *, overwrite=False):
    """Save a binary painted support as red-channel PNG, reloadable by notebook 00."""
    import numpy as np
    from PIL import Image
    array = np.asarray(mask)
    if array.ndim != 2 or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError('Support must be a finite binary 2-D array')
    path = profile.support_dir / f'{_name(name)}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
    rgb[..., 0] = array.astype(np.uint8) * 255
    with path.open('wb' if overwrite else 'xb') as handle:
        Image.fromarray(rgb).save(handle, format='PNG')
    return path
