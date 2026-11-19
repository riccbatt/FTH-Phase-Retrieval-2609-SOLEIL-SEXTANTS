"""Portable naming and input helpers shared by the reconstruction notebooks."""
from pathlib import Path
import numpy as np


def acquisition_ids(value):
    """Flatten scalar/list/array ID inputs in acquisition order."""
    result = []
    def visit(item):
        if isinstance(item, (list, tuple)) or (isinstance(item, np.ndarray) and item.ndim != 0):
            for child in item:
                visit(child)
        else:
            number = int(item)
            if number != item or number < 0:
                raise ValueError(f'Invalid acquisition ID: {item!r}')
            result.append(number)
    visit(value)
    if not result:
        raise ValueError('At least one acquisition ID is required')
    return result


def first_id(value):
    return acquisition_ids(value)[0]


def result_path(base, ids, user, series=False):
    prefix = 'data_recon_series' if series else 'data_recon'
    return str(Path(base) / 'processed' / 'Logs' / f'{prefix}_ImId_{first_id(ids):04d}_{user}.hdf5')


def stitched_path(base, ids, polarization, user):
    name = {'+': 'plus', '-': 'minus'}[polarization]
    return str(Path(base) / 'processed' / 'stitched' / f'stitched_ImId_{first_id(ids):04d}_{name}_{user}.npz')


def resolve_input(base, filename):
    path = Path(filename)
    return path if path.is_absolute() else Path(base) / path


def stitched_ids(base, filename):
    with np.load(resolve_input(base, filename), allow_pickle=False) as saved:
        key = 'reference_image_ids' if 'reference_image_ids' in saved else 'reference_image_id'
        return acquisition_ids(saved[key])


def load_stitched(base, filename):
    """Read current or legacy stitching output; missing pixels stay excluded."""
    path = resolve_input(base, filename)
    with np.load(path, allow_pickle=False) as saved:
        image = np.asarray(saved['image'], dtype=float)
        mask_key = 'mask_pixel_stitched' if 'mask_pixel_stitched' in saved else 'mask_pixel'
        mask = (np.asarray(saved[mask_key]) != 0) | ~np.isfinite(image)
        detector = np.zeros(image.shape, dtype=bool)
        beamstop = np.asarray(saved['mask_beamstop']) != 0
        if image.ndim != 2 or any(x.shape != image.shape for x in (mask, detector, beamstop)):
            raise ValueError(f'Incompatible image/mask shapes in {path}')
        # Existing combined masks still mark pixels missing from old stitching.
        mask |= beamstop
        history = {key: np.asarray(saved[key]) for key in (
            'mask_pixel_stitched', 'mask_stitched_intersection', 'input_mask_detectors',
            'input_mask_beamstops_unshifted', 'input_mask_beamstops_shifted',
            'input_mask_pixels', 'aligned_input_mask_pixels') if key in saved}
        metadata = {'energy_eV': float(saved['energy_eV']),
                    'stitching_file': str(filename), 'stitching_masks': history}
        exposure = float(saved['reference_exposure'])
    return (np.where(mask, 0.0, image), detector.astype(np.uint8),
            beamstop.astype(np.uint8), mask.astype(np.uint8), metadata,
            exposure, str(path), None)


def stitched_setup_metadata(base, filename):
    """Geometry saved by stitching; empty for older files needing raw metadata."""
    with np.load(resolve_input(base, filename), allow_pickle=False) as saved:
        return {key: float(saved[key]) for key in ('energy_eV', 'ccd_dist_m', 'px_size_m')
                if key in saved}
