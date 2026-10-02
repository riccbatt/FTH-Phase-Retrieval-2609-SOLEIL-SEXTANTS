"""Postprocess fitted log responses without changing the retrieved exit waves."""
import numpy as np


def calibrate_responses(charge, magnetic, magnetization, energies_ev, *,
                        cobalt_thickness_nm=None, normalize_magnetization=False,
                        material_mask=None, propagation_sign=-1):
    """Convert dimensionless log responses using an explicit propagation sign.

    Convention: T=exp(sign*1j*k*(n-1)*d), n=1-delta+sign*1j*beta.
    Thus q=-k*d*beta-sign*1j*k*d*delta; beta>0 always attenuates.
    Default sign=-1 matches exp(-1j*k*n*d), up to the vacuum phase. Input thickness maps in the fit must be dimensionless relative
    to the supplied cobalt thickness. This operation does not fix illumination/
    charge offsets, unknown composition or magnetic-sign ambiguities.

    Optional normalization uses one global max(abs(m)) over material pixels;
    m_new=m/scale and q_m_new=q_m*scale preserve every magnetic product.
    It assumes the strongest observed magnetization represents the chosen unit,
    not that saturation was independently measured. No offset is subtracted.
    None thickness returns rescaled log coefficients without claiming indices.
    """
    if isinstance(propagation_sign, bool) or propagation_sign not in (-1, 1):
        raise ValueError("propagation_sign must be -1 or +1")
    charge=np.asarray(charge,dtype=complex).copy()
    magnetic=np.asarray(magnetic,dtype=complex).copy()
    m=np.asarray(magnetization,dtype=float).copy()
    energy=np.asarray(energies_ev,dtype=float)
    if charge.ndim != 1 or magnetic.shape != charge.shape or energy.shape != charge.shape:
        raise ValueError('Expected matching one-dimensional charge, magnetic and energy spectra')
    if m.ndim != 3 or not all(np.all(np.isfinite(v)) for v in (charge,magnetic,m,energy)) or np.any(energy<=0):
        raise ValueError('Require finite spectra, positive energies and a state-by-image magnetization stack')
    if not isinstance(normalize_magnetization,(bool,np.bool_)):
        raise ValueError('normalize_magnetization must be bool')
    active=np.ones(m.shape[-2:],bool) if material_mask is None else np.asarray(material_mask)!=0
    if active.shape != m.shape[-2:] or not active.any():
        raise ValueError('Material mask must match the object grid and contain material')
    scale=1.0
    if normalize_magnetization:
        scale=float(np.max(np.abs(m[:,active])))
        if scale==0: raise ValueError('Cannot normalize an all-zero magnetization map')
        m/=scale
        magnetic*=scale
    result=dict(charge_response=charge, magnetic_response=magnetic,
                magnetization=m, magnetization_scale=scale,
                magnetization_normalized=bool(normalize_magnetization),energies_ev=energy.copy(),
                propagation_sign=int(propagation_sign))
    if cobalt_thickness_nm is not None:
        if isinstance(cobalt_thickness_nm,bool) or not np.isfinite(cobalt_thickness_nm) or cobalt_thickness_nm<=0:
            raise ValueError('Cobalt thickness must be positive and finite in nm, or None')
        kt=2*np.pi*energy/1239.8419843320026*float(cobalt_thickness_nm)
        result.update(cobalt_thickness_nm=float(cobalt_thickness_nm),
                      wave_numbers_per_nm=kt/float(cobalt_thickness_nm),
                      charge_index_contrast=charge/(propagation_sign*1j*kt),
                      magnetic_index_contrast=magnetic/(propagation_sign*1j*kt),
                      charge_delta=-propagation_sign*charge.imag/kt,charge_beta=-charge.real/kt,
                      magnetic_delta=-propagation_sign*magnetic.imag/kt,magnetic_beta=-magnetic.real/kt)
    return result


def xmcd_ratio_indices(positive, negative, energy_ev, thickness_nm, mz, *,
                       material_mask=None, phase_reference_mask=None,
                       min_abs_mz=1e-3, relative_amplitude_floor=1e-6):
    """Evaluate the thesis helicity-ratio formula on one focused object pair.

    A=log|psi_plus/psi_minus|, theta=arg(psi_plus/psi_minus),
    beta=-A/(2*k*d*mz), delta=-theta/(2*k*d*mz). Inputs and masks
    must share the SAME object coordinate frame. d may be a spatial map in nm.
    Use a common, independently chosen m/d map when comparing retrieval stages.

    Report an x**2-weighted average of pixelwise estimates, x=2*k*d*mz.
    This is a through-origin slope fit and avoids cancellation between +/- domains.
    It is not a confidence interval. Low-amplitude, zero-thickness and near-zero-m
    pixels are excluded. With no phase reference the global phase is unanchored.
    A supplied reference mask must contain nonmagnetic illuminated reference
    pixels; remove their circular mean phase only (no amplitude renormalization).
    Principal-branch phases assume the physical ratio phase stays within +/-pi
    after referencing. No fitted phase ramp or spatial unwrapping is imposed.
    """
    pos, neg = np.asarray(positive,complex), np.asarray(negative,complex)
    if pos.ndim != 2 or neg.shape != pos.shape:
        raise ValueError('Exit waves must be matching 2D object arrays')
    if not np.isfinite(energy_ev) or energy_ev <= 0:
        raise ValueError('Energy must be positive in eV')
    if not np.isfinite(min_abs_mz) or min_abs_mz < 0 or not 0 <= relative_amplitude_floor < 1:
        raise ValueError('Invalid magnetization/amplitude exclusion thresholds')
    d=np.broadcast_to(np.asarray(thickness_nm,float),pos.shape)
    m=np.broadcast_to(np.asarray(mz,float),pos.shape)
    material=np.ones(pos.shape,bool) if material_mask is None else np.asarray(material_mask,bool)
    if material.shape != pos.shape: raise ValueError('Material mask shape mismatch')
    ap,an=abs(pos),abs(neg)
    finite=np.isfinite(pos)&np.isfinite(neg)
    pmax=float(np.max(ap[finite])) if finite.any() else 0.
    nmax=float(np.max(an[finite])) if finite.any() else 0.
    illuminated=finite&(ap>max(relative_amplitude_floor*pmax,0))&(an>max(relative_amplitude_floor*nmax,0))
    valid=illuminated&material&np.isfinite(d)&(d>0)&np.isfinite(m)&(abs(m)>min_abs_mz)
    if not valid.any(): raise ValueError('No valid material pixels for XMCD ratio (check amplitude, thickness and mz)')
    # Log differences avoid forming a potentially overflowing complex ratio.
    amplitude_log=np.zeros(pos.shape);theta=np.zeros(pos.shape)
    amplitude_log[illuminated]=np.log(ap[illuminated])-np.log(an[illuminated])
    theta[illuminated]=np.angle(np.exp(1j*(np.angle(pos[illuminated])-np.angle(neg[illuminated]))))
    phase_offset=0.; reference_count=0; phase_status='unreferenced'
    if phase_reference_mask is not None:
        ref=np.asarray(phase_reference_mask,bool)
        if ref.shape != pos.shape: raise ValueError('Phase-reference mask shape mismatch')
        ref=ref&illuminated
        reference_count=int(ref.sum())
        phasor=np.mean(np.exp(1j*theta[ref])) if ref.any() else 0j
        if abs(phasor) < 1e-8:
            phase_status='missing_or_incoherent_reference'
        else:
            phase_offset=float(np.angle(phasor));phase_status='referenced'
            theta[illuminated]=np.angle(np.exp(1j*(theta[illuminated]-phase_offset)))
    k=2*np.pi*float(energy_ev)/1239.8419843320026
    x=2*k*d[valid]*m[valid];weights=x*x
    beta_pixels=-amplitude_log[valid]/x
    delta_pixels=-theta[valid]/x
    beta=float(np.average(beta_pixels,weights=weights))
    delta=float(np.average(delta_pixels,weights=weights))
    if phase_status=='missing_or_incoherent_reference': delta=float('nan')
    return dict(beta_m=beta,delta_m=delta,valid_pixels=int(valid.sum()),
                reference_pixels=reference_count,phase_offset=phase_offset,
                phase_status=phase_status,valid_mask=valid,
                beta_spatial_rms=float(np.sqrt(np.average((beta_pixels-beta)**2,weights=weights))),
                delta_spatial_rms=float(np.sqrt(np.average((delta_pixels-delta)**2,weights=weights))),
                A_xmcd_mean=float(np.mean(amplitude_log[valid])),
                theta_xmcd_mean=float(np.mean(theta[valid])),wave_number_per_nm=k)
