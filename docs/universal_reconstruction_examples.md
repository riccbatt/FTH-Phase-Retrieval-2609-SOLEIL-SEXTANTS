# Reconstruction examples with the universal library

All examples import one implementation:

```python
import numpy as np
from library import phase_retrieval_universal as universal
from library import fthcore as fth
```

Use centered, background-corrected **intensities**, normalized for exposure and
incident flux. Arrays must share a detector grid. `mask_pixel` is zero for valid
measurements and nonzero for invalid pixels; `supportmask` is nonzero where the
object is allowed. Keep true measured zeros valid. Crop and bin through the
recipe rather than changing only one input. Iteration counts below are starting
examples; check convergence on your data.

## 1. Simple XMCD sets: phase retrieval without a joint projection

For one magnetic state measured with positive and negative helicity, let `pos`
and `neg` be 2D intensity arrays and `energy_eV` the photon energy. This uses the
metadata API with `projection_model="none"`: each observation receives ordinary
detector/support iterations, without fitting charge, spectra, or magnetization.
“No joint projection” still includes the detector-amplitude and support
constraints that define phase retrieval.

```python
holograms = np.stack([pos, neg])
recipe = universal.default_universal_phase_retrieval_recipe()
recipe.update(
    projection_model="none",
    modes=[1],
    mode_initialization="support_fft",
    warmup_mode=["HAPRE"], warmup_Nit=[700],
    warmup_start_from_first=False,
    inner_mode=["ER"], inner_Nit=[50], outer_iterations=1,
    shuffle_observations=False,
    partial_coherence=False,
    RL_it=0, warmup_RL_it=0,
    constrain_nonphysical_modes_common=False,
    saturated_states=None,
    final_projection_relaxation=0,
    final_fourier_constraint=True,
)
fields, warmup, components, bsmasks, errors = (
    universal.universal_phase_retrieval_algorithm(
        holograms, mask_pixel, supportmask,
        state_labels=["state_0", "state_0"],
        energy_labels=[energy_eV, energy_eV],
        polarization_coefficients=[+1, -1],
        illumination_labels=["beam_0", "beam_0"],
        universal_recipe=recipe,
    )
)
obj_pos = fth.reconstructCDI(fields[0])
obj_neg = fth.reconstructCDI(fields[1])
```

`fields` are Fourier-domain complex fields, not intensity images. `warmup`
contains the fields before the final ER sweep. For several XMCD sets, stack
`[pos_0, neg_0, pos_1, neg_1, ...]`, repeat each state label twice, and use
polarizations `[+1, -1, +1, -1, ...]`. Keep one metadata entry per observation.
With `none`, those labels describe the data but do not couple the objects.

To initialize neg from the retrieved pos instead of independently, use:

```python
recipe.update(
    warmup_reference_observation=0,
    warmup_start_from_first=True,
    warmup_reference_mode=["HAPRE", "ER"],
    warmup_reference_Nit=[700, 50],
    warmup_other_mode=["ER"], warmup_other_Nit=[50],
)
```

This seeds all nonreference observations from observation 0. For an explicit
pos-to-neg handoff separately within every pair, use universal's ordered-stage
`phase_retrieval_algorithm` with `default_phase_retrieval_recipe`, `helicity`,
`Startimage`, and `Startgamma`. That recipe format differs from the metadata
recipe above; both implementations live in universal.

### Turn the two objects into XMCD contrast

Align the objects and match their relative complex scale using a reliable common
nonmagnetic region before calculating quantitative contrast. Reference seeding
helps but does not remove all phase, translation, or scale ambiguity. Choose
`sample_roi` on the returned object grid and exclude low-amplitude pixels:

```python
amplitude_floor = 1e-6 * max(abs(obj_pos).max(), abs(obj_neg).max())
valid = (sample_roi.astype(bool)
         & (abs(obj_pos) > amplitude_floor)
         & (abs(obj_neg) > amplitude_floor))
xmcd = np.full(obj_pos.shape, np.nan + 1j*np.nan)
xmcd[valid] = np.log(obj_pos[valid] / obj_neg[valid])
log_amplitude_contrast = xmcd.real
phase_contrast = xmcd.imag
```

The real part is the logarithmic amplitude ratio; the imaginary part is the
relative phase, wrapped to the principal branch. The corresponding optical-density
difference is `-2 * log_amplitude_contrast`. Sign depends on the helicity convention.
This is helicity contrast, not an absolutely calibrated magnetization map.

## 2. Hyperspectral retrieval: choose the spectral projection

For a pure energy scan, `energy_holograms` has shape `(n_energy, rows, columns)`
and `energies_eV` contains one **distinct** energy per observation. State,
polarization, and illumination stay fixed. Repeated helicities or magnetic states
make it a mixed scan: use the physical example below instead of SVD/rank-one dispatch.

```python
n_energy = len(energies_eV)
spectral_recipe = universal.default_universal_phase_retrieval_recipe()
spectral_recipe.update(
    projection_model="svd", rank=1,
    modes=[1], mode_initialization="support_fft",
    warmup_mode=["HAPRE", "ER"], warmup_Nit=[700, 50],
    inner_mode=["ER"], inner_Nit=[50], outer_iterations=10,
    projection_every=n_energy,
    projection_relaxation=0.3,
    final_projection_relaxation=0,
    final_fourier_constraint=True,
    energy_values=np.asarray(energies_eV),
)
spectral_fields, spectral_warmup, spectral_components, spectral_masks, spectral_errors = (
    universal.universal_phase_retrieval_algorithm(
        energy_holograms, mask_pixel, supportmask,
        state_labels=["state_0"] * n_energy,
        energy_labels=energies_eV,
        polarization_coefficients=[+1] * n_energy,
        illumination_labels=["beam_0"] * n_energy,
        universal_recipe=spectral_recipe,
    )
)
```

Change these settings **before** the retrieval call to choose another projection:

| Model | Meaning | Main controls |
| --- | --- | --- |
| `none` | Independent retrieval at each energy | Warmup and inner schedules |
| `svd` | Low-rank model of complex log-objects across energies | `rank`, `projection_static_mode`, relaxation |
| `rank1_spectral` | A shared spatial component with a fitted complex spectral response | Spectral constraints and relaxation |

For a rank-one model with a free spectrum:

```python
spectral_recipe.update(
    projection_model="rank1_spectral",
    spectral_constraint="free",
)
```

Available `spectral_constraint` values are `free`, `kk`, `known_beta`, and
`known_beta_kk`. `kk` couples absorption and dispersion by the library's
Kramers–Kronig transform. It requires a numeric energy axis and suitable spectral
coverage; finite-window spectra can bias the transform. For a measured absorption
spectrum in the same unique-energy order:

```python
spectral_recipe.update(
    projection_model="rank1_spectral",
    spectral_constraint="known_beta_kk",
    known_beta_spectrum=beta_spectrum,
    absorption_part="real",
    fit_known_beta_scale=True,
    fit_known_beta_offset=True,
)
```

`absorption_part` specifies the absorption component of the fitted log-response;
it is not a raw refractive-index convention. SVD/rank-one models do not identify
magnetization. With `modes=[1, 2]`, the spectral projector acts separately on each
mode; corresponding mode indices across energies are an assumption.

## 3. Hysteresis, or combined energy/state/helicity scans

Flatten the measurements into `(n_observations, rows, columns)`. Use the same
state label for both helicities of a given field point, and different labels for
different field points. Reuse a state's label across energies only if it really
represents the same magnetic configuration. Label illumination groups according
to which observations share the same incoming complex field.

For example, two states with both helicities at one energy use:

```python
joint_holograms = np.stack([sat_pos, sat_neg, point_pos, point_neg])
state_labels = ["saturated", "saturated", "point_0", "point_0"]
energy_labels = [energy_eV] * 4
polarizations = [+1, -1, +1, -1]
beam_labels = ["beam_0"] * 4
```

Use the same call for a hyperspectral hysteresis scan, with the appropriate
energy label on every observation:

```python
physical_recipe = universal.default_universal_phase_retrieval_recipe()
physical_recipe.update(
    projection_model="physical_factorized",
    modes=[1], mode_initialization="support_fft",
    warmup_mode=["HAPRE", "ER"], warmup_Nit=[700, 50],
    warmup_start_from_first=False,
    inner_mode=["ER"], inner_Nit=[50], outer_iterations=10,
    projection_every=len(joint_holograms),
    projection_relaxation=0.3,
    final_projection_relaxation=0,
    saturated_states={"saturated": +1},
    clip_magnetization=True,
)
joint_fields, joint_warmup, physical_components, joint_masks, joint_errors = (
    universal.universal_phase_retrieval_algorithm(
        joint_holograms, mask_pixel, supportmask,
        state_labels=state_labels, energy_labels=energy_labels,
        polarization_coefficients=polarizations,
        illumination_labels=beam_labels,
        universal_recipe=physical_recipe,
    )
)
magnetization_maps = physical_components["magnetization_by_state"]
```

The physical model fits
`log(object) = log(illumination) + thickness * (charge(E) + polarization * magnetic(E) * magnetization(state))`.
Only the first mode enters this fit. Saturation fixes the specified state map to
±1 within material; it does not automatically freeze that observation's Fourier
field. `freeze_saturated_fields=True` additionally freezes its field after warmup.
The data must distinguish the fitted components. Saturation anchors the magnetic
scale, but absolute thickness and optical response still require calibration.

`state_energy_beam` is a less restrictive alternative: it fits additive
illumination and charge terms plus state-dependent magnetic terms in log-object
space, without the shared scalar-spectrum × real-magnetization factorization.
It does not provide the physical model's bounded magnetization maps or saturation
interpretation. Insufficient metadata diversity can make its linear fit rank deficient;
the default is to report an error rather than silently accept an ambiguous fit.

### Optional physical constraints

Apply these settings to `physical_recipe` **before** calling retrieval:

| Setting | Effect |
| --- | --- |
| `material_mask=material_aperture` | Known material/vacuum regions, including holes |
| `material_thickness=relative_thickness` | Fixed nonnegative relative thickness map |
| `fit_material_thickness=True` | Fit one shared nonnegative relative thickness map |
| `zero_magnetization_outside_support=True` | Restrict fitted magnetization to support |
| `zero_thickness_outside_support=True` | Restrict thickness to support |
| `projection_constraints_inside_support_only=True` | Preserve exterior fields during the joint projection |
| `physical_projection_object_roi=True` | Fit on the material aperture with a thickness map |
| `physical_phase_reference=True` | Select the nearest relative phase branch within an energy/beam group; assumes differences below π |

Charge and magnetic spectra have separate controls:
`charge_spectral_constraint` and `magnetic_spectral_constraint`, each accepting
`free`, `kk`, `known_beta`, or `known_beta_kk`. Supply the matching
`known_charge_beta_spectrum` or `known_magnetic_beta_spectrum` and `energy_values`
in unique-energy order. Known spectra can reduce ambiguities, but a single-energy
hysteresis scan cannot determine a spectral line shape. See the
[full guide](universal_phase_retrieval.md) for response bounds and explicit
refractive-index-to-log-response conversions.

For multiple modes with secondary fields shared across states/helicities within
each energy/illumination group:

```python
physical_recipe.update(
    modes=[1, 2],
    constrain_nonphysical_modes_common=True,
    nonphysical_modes_common_relaxation=1.0,
)
```

Assigning a separate illumination to every observation can remove the coupling
needed to distinguish illumination from magnetic contrast. Use shared groups
where experimentally justified; use `illumination_group_labels` for systematic
energy/state grouping.

## Coherence, projection cadence, and checking results

`partial_coherence=False` gives coherent retrieval. For the general driver
(`none`, `state_energy_beam`, or `physical_factorized`), this example keeps warmup
coherent and enables partial coherence in the inner loops:

```python
physical_recipe.update(
    partial_coherence=True,
    warmup_RL_it=0,
    preserve_warmup_masked_intensity=True,
    coherence_kernel_scope="per_observation",
    RL_it=20, RL_freq=10,       # Active with inner_Nit=[50]
    final_fourier_constraint=False,
)
```

Use shared gamma only when the same coherence kernel is justified across the
observations. Spectral SVD/rank-one schedules currently reject active RL updates;
use the general driver for partial-coherence retrieval. Coherent refreshes and
live callbacks also require the general driver.

`projection_every` counts completed observation updates; `None` means a full
sweep. `projection_relaxation` blends the repeated joint projection, while
`final_projection_relaxation=0` omits the final joint replacement. A final detector
constraint can move the fields away from the physical model. Compare detector
errors, physical/spectral residuals, and warmup versus final fields; a plausible
image alone does not establish convergence or identifiability.

Before a metadata-driven run, inspect the resolved schedule:

```python
universal.print_universal_workflow(physical_recipe, state_labels=state_labels)
```

All metadata runs return `(fields, warmup, components, bsmasks, errors)`.
Single-mode fields have shape `(observations, rows, columns)`; multimode fields
add a mode axis. Convert a selected mode with `fth.reconstructCDI`; do not sum
complex incoherent modes before computing detector intensity.
