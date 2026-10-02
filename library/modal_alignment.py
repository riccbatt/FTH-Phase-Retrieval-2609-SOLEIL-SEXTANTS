"""Post-hoc constant-unitary diagnostics; never a reconstruction constraint."""
import numpy as np


def align_modes(modes, reference, mask=None):
    """Fit U on the specified pixels, return U @ modes and rank-aware metrics.

    No gain, spatial registration, conjugation, or pixel-dependent mixing is fit.
    Reference fields must use the same coordinates and intensity normalization.
    """
    x, y = np.asarray(modes), np.asarray(reference)
    if x.shape != y.shape or x.ndim < 2:
        raise ValueError('Modes and reference must have identical (mode, spatial...) shapes')
    selected = np.ones(x.shape[1:], bool) if mask is None else np.asarray(mask, bool)
    if selected.shape != x.shape[1:] or not selected.any():
        raise ValueError('Require a nonempty spatial mask')
    a, b = x[:, selected], y[:, selected]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite selected fields')
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        raise ValueError('Cannot compare zero fields')
    left, _, right = np.linalg.svd(b @ a.conj().T)
    unitary = left @ right
    aligned = (unitary @ x.reshape(x.shape[0], -1)).reshape(x.shape)
    sa, va = np.linalg.svd(a, full_matrices=False)[1:]
    sb, vb = np.linalg.svd(b, full_matrices=False)[1:]
    ra = int(np.sum(sa > sa[0]*1e-10))
    rb = int(np.sum(sb > sb[0]*1e-10))
    cosines = np.linalg.svd(va[:ra] @ vb[:rb].conj().T, compute_uv=False)
    metrics = dict(complex_relative_error=float(np.linalg.norm(unitary@a-b)/np.linalg.norm(b)),
                   ranks=[ra, rb],
                   mode_power_fractions=(sa**2/np.sum(sa**2)).tolist(),
                   reference_power_fractions=(sb**2/np.sum(sb**2)).tolist(),
                   principal_angles_degrees=np.degrees(np.arccos(np.clip(cosines,0,1))).tolist())
    per_mode = []
    for p, q in zip(unitary@a, b):
        norm = np.linalg.norm(p)*np.linalg.norm(q)
        valid = (abs(q) > .01*abs(q).max()) & (abs(p) > .01*abs(p).max())
        phase = np.angle(p[valid]*q[valid].conj())
        per_mode.append(dict(complex_correlation=float(abs(np.vdot(q,p))/norm) if norm else None,
            amplitude_relative_error=float(np.linalg.norm(abs(p)-abs(q))/np.linalg.norm(q)) if np.linalg.norm(q) else None,
            phase_rmse_radians=float(np.sqrt(np.mean(phase**2))) if phase.size else None,
            phase_pixel_count=int(phase.size)))
    metrics['per_mode'] = per_mode
    return aligned, unitary, metrics
