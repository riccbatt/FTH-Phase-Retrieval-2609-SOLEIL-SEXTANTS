# Optional coherence-isolated diffraction imaging (CIDI)

`library.cidi.cidi_separate` implements the analytic intensity separation in Sarkar and Johnson's CIDI formulation. It is an optional preprocessing step; existing reconstruction recipes do not enable it automatically. It needs no known illumination profile. It does need spatially isolated holographic cross-correlation lobes, at least two coherent reference apertures, and a contribution mutually incoherent with those references.

References:

- Arnab Sarkar and Allan S. Johnson, [Coherent X-ray imaging of stochastic dynamics](https://doi.org/10.1039/D4MA00154K), *Materials Advances* **5**, 6378–6387 (2024): original CIDI method.
- Arnab Sarkar and Allan S. Johnson, [Non-linear X-ray Coherent Diffractive Imaging](https://arxiv.org/abs/2512.15457) (2025), section II and supplementary section 1: application to nonlinear signals and the intensity identities implemented here.

## Model and implementation

For a sample field X and coherent reference fields R_i, the measured pattern is

```text
I = |X + sum_i R_i|² + S
```

S is the total intensity incoherent with the reference fields. The inverse Fourier transform of I contains a central autocorrelation and displaced cross-correlation lobes. Masking a complete isolated lobe and transforming it back, **without translating it**, gives C_i = X conjugate(R_i) or its conjugate. A reference-pair lobe gives D_ij = R_i conjugate(R_j) or its conjugate. Their magnitudes give

```text
|X|²   = |C_i| |C_j| / |D_ij|
|R_i|² = |C_i|² / |X|²
S      = I - |X|² - sum_i |R_i|² - all interference terms
```

The implementation uses every reference pair and averages the admissible estimates of |X|² equally. This extension to more than two references is our implementation choice. Reference diffraction zeros and small denominators are excluded using `relative_floor`, relative to each denominator's maximum. It retains the signed residual S and returns an explicit validity mask; undefined results are NaN.

All detector arrays and conjugate-plane support arrays are **centered**. Internally, centered FFT and inverse FFT use NumPy's default normalization. Intensities remain in the input detector units. Sample and reference supports need not be point apertures; their complete finite footprints determine the lobe masks. Every coherent reference present in the measurement must be supplied separately.

The routine computes all oriented support cross-correlations and checks their footprints against one another and against the central autocorrelation, including circular wraparound. An overlap raises an error rather than returning a misleading separation. `incoherent_support` specifies the conjugate-grid footprint of S's underlying field, defaulting to the sample support. For a different wavelength, supply the appropriately scaled footprint on the common detector grid. This is important for verifying separation geometry.

## Use with the existing workflow

```python
import numpy as np
from library import cidi_separate
from library.phase_retrieval_universal import (
    default_universal_phase_retrieval_recipe,
    universal_phase_retrieval_algorithm,
)

# Each support is a centered 2D mask on the same conjugate grid.
# reference_supports must include all RHs in the measured coherent field.
separated = cidi_separate(
    measured_intensity,
    sample_support=object_hole_support,
    reference_supports=reference_hole_supports,
    incoherent_support=nonlinear_support,
    relative_floor=1e-10,
)

# Excludes unstable divisions and negative residuals; raw S remains available.
signal, invalid = separated.reconstruction_inputs("incoherent")

# For the joint universal driver, repeat separation for each observation.
# It requires at least two observations; observation_results contains those results.
inputs = [r.reconstruction_inputs("incoherent") for r in observation_results]
signals = np.stack([item[0] for item in inputs])
invalid_masks = np.stack([item[1] for item in inputs])
n = len(inputs)
recipe = default_universal_phase_retrieval_recipe()
recipe.update(projection_model="none", modes=[1])
fields, warmup, components, masks, errors = universal_phase_retrieval_algorithm(
    holograms=signals,
    mask_pixel=invalid_masks,
    supportmask=nonlinear_support,
    state_labels=[f"signal_{i}" for i in range(n)],
    energy_labels=[nonlinear_energy_eV] * n,
    polarization_coefficients=[1.0] * n,
    illumination_labels=["beam"] * n,
    universal_recipe=recipe,
)
```

Use `reconstruction_inputs("sample")` with the OH support to reconstruct the separated coherent sample intensity. Use `"coherent"` with the union of OH and RH supports to retain the reference-coherent field. For quantitative magnetic fitting, isolate each observation first, then pass the coherent patterns and the original physical metadata to the physical driver. The incoherent residual has no automatic magnetic interpretation, so do not assign it the coherent transmission model by default. The separated patterns alone do not provide absolute phase.

## Missing pixels, noise, and memory

A missing detector pixel affects the whole hologram. If `invalid_mask` contains missing pixels, supply `filled_intensity` explicitly; the routine substitutes its values only at missing pixels and retains those pixels as invalid in the result. This is an approximation, and its errors can contaminate other detector pixels after separation. Obtain a suitable fill from preprocessing or a preliminary reconstruction; CIDI does not infer it. Nonfinite or negative unmasked intensities are rejected.

The raw incoherent residual includes detector noise and other reference-incoherent contributions. Negative residual values are preserved for diagnostics. `reconstruction_inputs` replaces invalid/negative values with zero **and masks them out**, so the CDI driver does not treat them as measured zeros. Inspect `separated.diagnostics` and the signed residual before retrieval; division thresholds and clipping are not a noise model. The method neither guarantees a unique decomposition for arbitrary partial coherence nor corrects drifting reference profiles automatically.

Process one observation at a time. The function retains no iteration history or observation stack. It allocates full-resolution correlation masks and terms proportional to the number of reference pairs, plus the returned intensity arrays. For five RHs this is a fixed per-image cost; it is not independent of detector size or reference count. Release each result after saving or consuming it for a long scan.
