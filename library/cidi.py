"""Coherence-isolated diffraction imaging (CIDI), optional analytic preprocessing.

Sarkar & Johnson, Mater. Adv. 5, 6378 (2024), doi:10.1039/D4MA00154K;
Non-linear X-ray Coherent Diffractive Imaging, arXiv:2512.15457, supplement 1.
All grids are centered. Cross terms retain their displacement; never recenter them.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np


def _fft(a):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))


def _ifft(a):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a)))


@dataclass
class CIDIResult:
    sample_intensity: np.ndarray
    reference_intensity: np.ndarray
    coherent_intensity: np.ndarray
    incoherent_intensity: np.ndarray
    valid_mask: np.ndarray
    diagnostics: dict

    def reconstruction_inputs(self, component='incoherent'):
        """Return nonnegative intensity and invalid mask for existing CDI drivers.

        Raw signed subtraction is retained in the result. Negative residual pixels
        are excluded here, rather than fitted as measured zeros.
        """
        if component not in ('incoherent', 'sample', 'coherent'):
            raise ValueError('component must be incoherent, sample, or coherent.')
        intensity = getattr(self, component+'_intensity')
        valid = self.valid_mask & np.isfinite(intensity) & (intensity >= 0)
        return np.where(valid, intensity, 0.), ~valid


def cidi_separate(intensity, sample_support, reference_supports, *,
                  incoherent_support=None, invalid_mask=None, filled_intensity=None,
                  relative_floor=1e-10):
    """Isolate reference-coherent and reference-incoherent detector intensities.

    Supports are centered binary footprints on the detector's conjugate grid.
    Supply *all* coherent reference apertures (at least two). The incoherent
    support defaults to sample_support; account for wavelength scaling explicitly.
    Exact circular cross-correlation footprints must be isolated from every other
    term. Overlapping geometry raises ValueError. No beam profile is required.

    Missing detector values affect every hologram pixel: an explicit finite
    filled_intensity is required when invalid_mask contains True. Such estimates
    are an approximation; their original detector pixels remain invalid.
    Returns intensity, not the phase of the incoherent field. Process observations
    separately to keep memory bounded; no reconstruction trajectory is retained.
    """
    y = np.asarray(intensity, dtype=float)
    if y.ndim != 2 or min(y.shape) < 2:
        raise ValueError('intensity must be a 2D detector image.')
    if not np.isfinite(relative_floor) or not 0 < relative_floor < 1:
        raise ValueError('relative_floor must lie strictly between zero and one.')
    def support(value):
        a = np.asarray(value)
        if a.shape != y.shape or not np.all(np.isfinite(a)) or not np.any(a):
            raise ValueError('Supports must be finite nonempty arrays matching intensity.')
        return a != 0
    obj = support(sample_support)
    refs = [support(r) for r in reference_supports]
    if len(refs) < 2:
        raise ValueError('CIDI requires at least two reference apertures.')
    inc = obj if incoherent_support is None else support(incoherent_support)
    occupied = obj.copy()
    for r in refs:
        if np.any(occupied & r):
            raise ValueError('Sample and reference supports must be disjoint.')
        occupied |= r
    invalid = np.zeros(y.shape, bool) if invalid_mask is None else np.broadcast_to(np.asarray(invalid_mask, bool), y.shape)
    if np.any(invalid):
        if filled_intensity is None:
            raise ValueError('Missing pixels require explicit filled_intensity for the hologram transform.')
        filled = np.asarray(filled_intensity, float)
        if filled.shape != y.shape:
            raise ValueError('filled_intensity must match intensity.')
        data = np.where(invalid, filled, y)
    else:
        data = y
    if not np.all(np.isfinite(data)) or np.any(data < 0):
        raise ValueError('Intensity used for CIDI must be finite and nonnegative.')
    def footprint(a, b):
        # FFT roundoff must not invent support pixels.
        return _ifft(_fft(a)*_fft(b).conj()).real > .5
    central = footprint(obj,obj) | footprint(inc,inc)
    for r in refs:
        central |= footprint(r,r)
    # Check both orientations of every coherent pair, including RH/RH terms.
    apertures = [obj]+refs
    terms = []
    coverage = central.astype(np.int32)
    for i,j in combinations(range(len(apertures)),2):
        for a,b in ((i,j),(j,i)):
            mask = footprint(apertures[a],apertures[b])
            terms.append((a,b,mask))
            coverage += mask
    if any(np.any(coverage[m] != 1) for _,_,m in terms):
        raise ValueError('CIDI cross-correlation lobes overlap (including circular wraparound); use separated reference geometry.')
    hologram = _ifft(data)
    sample_terms = {}
    pair_terms = {}
    interference = np.zeros(y.shape, float)
    for a,b,mask in terms:
        # Transform at its original lag to preserve detector interference phase.
        cross = _fft(np.where(mask,hologram,0))
        if a < b:
            interference += 2*cross.real
            if a == 0:
                sample_terms[b-1] = cross
            else:
                pair_terms[a-1,b-1] = cross
    sample = np.zeros(y.shape, float)
    count = np.zeros(y.shape, np.int32)
    for (i,j), cross in pair_terms.items():
        denominator = abs(cross)
        good = denominator > relative_floor*max(float(denominator.max()),np.finfo(float).tiny)
        estimate = np.divide(abs(sample_terms[i])*abs(sample_terms[j]), denominator,
                             out=np.zeros_like(sample), where=good)
        sample += np.where(good,estimate,0.)
        count += good
    sample = np.divide(sample,count,out=sample,where=count>0)
    valid = (count>0) & (sample > relative_floor*max(float(sample.max()),np.finfo(float).tiny))
    reference = np.zeros(y.shape,float)
    for cross in sample_terms.values():
        reference += np.divide(abs(cross)**2,sample,out=np.zeros_like(sample),where=valid)
    coherent = sample+reference+interference
    residual = data-coherent
    valid &= ~invalid & np.isfinite(coherent) & np.isfinite(residual)
    # Undefined divisions are explicit NaNs, never silently valid zeros.
    for a in (sample,reference,coherent,residual):
        a[~valid] = np.nan
    return CIDIResult(sample,reference,coherent,residual,valid,
        dict(reference_count=len(refs), reference_pairs=len(pair_terms),
             valid_pixels=int(valid.sum()), missing_pixels=int(invalid.sum()),
             negative_residual_pixels=int(np.sum(valid & (residual<0))),
             relative_floor=float(relative_floor), detector_filling_used=bool(np.any(invalid))))
