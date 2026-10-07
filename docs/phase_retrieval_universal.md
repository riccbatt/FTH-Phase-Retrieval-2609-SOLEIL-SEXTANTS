# Universal phase retrieval API

Start with [task-based examples](universal_reconstruction_examples.md), then run
the four notebooks in order. All retrieval code lives in
`library/phase_retrieval_universal.py`; the other library files supply geometry,
gradient refinement and Kramers–Kronig numerics.

## One pattern for every metadata-driven reconstruction

```python
from library import phase_retrieval_universal as universal

recipe = universal.default_universal_phase_retrieval_recipe()
recipe.update(projection_model="none")
fields, warmup, components, bsmasks, errors = (
    universal.universal_phase_retrieval_algorithm(
        holograms, mask_pixel, supportmask,
        state_labels=states,
        energy_labels=energies,
        polarization_coefficients=polarizations,
        illumination_labels=beams,
        universal_recipe=recipe,
    )
)
```

Use measured intensities, not amplitudes. The observation stack has shape
`(n_observations, rows, columns)` and currently requires at least two observations.
Every metadata array has one entry per observation. A detector mask is either 2D
or observation-shaped; zero means valid. A support is nonzero at allowed object
pixels. The recipe applies crop before binning and transforms related arrays.

| Model | Use |
| --- | --- |
| `none` | Ordinary independent phase retrieval; no joint object projection |
| `svd` | Low-rank complex log-objects for a pure energy scan |
| `rank1_spectral` | Shared spatial component and fitted spectrum for a pure energy scan |
| `state_energy_beam` | Additive log-object model of illumination, energy and state-dependent magnetic components |
| `physical_factorized` | Shared charge/magnetic spectra, material thickness and bounded real magnetization maps |

Pure energy scans have fixed state, polarization and illumination, and one
observation per distinct energy. Mixed scans use the general models. State labels
identify magnetic configurations; illumination labels identify shared incoming
complex fields. Too many independent illumination groups can destroy identifiability.

## Schedules and projection controls

- `warmup_mode`, `warmup_Nit`: independent initialization/retrieval stages.
- `inner_mode`, `inner_Nit`: detector/support stages per observation per outer loop.
- `outer_iterations`: number of outer loops.
- `warmup_start_from_first`: seed other observations from a selected retrieved reference.
- `projection_every`: completed observation updates between joint projections;
  `None` means a full sweep.
- `projection_relaxation`: repeated joint-projection blend.
- `final_projection_relaxation`: final joint replacement; zero disables replacement.
- `final_fourier_constraint`: finish on detector constraints, which may alter physical consistency.

Inspect the schedule with `universal.print_universal_workflow(recipe, state_labels=states)`.
Unknown keys are errors. Use the default dictionary for the full option inventory;
not every option applies to every model.

## Physical and spectral constraints

The physical model uses
`log(object) = log(illumination) + thickness * (charge(E) + polarization * magnetic(E) * magnetization(state))`.

`material_mask` specifies material/vacuum. `material_thickness` supplies fixed
nonnegative relative thickness; `fit_material_thickness` fits it instead.
`saturated_states={"state": +1}` anchors that state's magnetization within material.
`clip_magnetization` imposes ±1 bounds. `freeze_saturated_fields` additionally fixes
its retrieved Fourier field after warmup. Absolute thickness and optical response
remain coupled without calibration.

`projection_constraints_inside_support_only` preserves exterior fields during
joint projection. `zero_magnetization_outside_support` and
`zero_thickness_outside_support` restrict the fitted maps. Focus transforms,
phase-reference selection and material-aperture fitting are available in the
recipe; apply them only when their experimental assumptions hold.

For rank-one spectral retrieval, `spectral_constraint` is `free`, `kk`,
`known_beta`, or `known_beta_kk`. Physical retrieval uses separate
`charge_spectral_constraint` and `magnetic_spectral_constraint` with matching
known-spectrum inputs. Supply `energy_values` and spectra in unique-energy order.
KK constraints require sufficient spectral coverage; a single-energy scan cannot
identify a spectral line shape. Log-response coefficients are not raw refractive
indices: conversion requires wave number and a calibrated thickness scale.

## Modes and partial coherence

`modes=[1, 2]` requests two fields with support factors 1 and 2. Duplicate factors
are allowed. Intensity is the sum of modal intensities, not the squared magnitude
of their complex sum. Only the first mode enters the physical fit;
`constrain_nonphysical_modes_common` shares secondary fields within energy/beam groups.
Spectral projectors instead act separately on corresponding mode indices.

For partial coherence, enable `partial_coherence`, set active `RL_it` and
`RL_freq <= Nit`, and choose a per-observation or experimentally justified shared
kernel. `preserve_warmup_masked_intensity` captures coherent masked-pixel fills.
Coherent warmup followed by partial inner loops is shown in notebook 04.
SVD/rank-one energy schedules reject active RL updates. Coherence refresh and
live callbacks require `none`, `state_energy_beam`, or `physical_factorized`.

## Outputs

`fields` and `warmup` are Fourier-domain complex fields. Single mode uses
`(observations, rows, columns)`; multiple modes add a modal axis.
`components` contains fitted maps, spectra, coherence and model diagnostics.
`bsmasks` records the effective invalid-pixel masks. `errors` records stages,
projection history, metadata and settings.

Convert a selected field to a centered object with
`universal.fourier_field_to_display_object(field)`. Inspect convergence and both
detector/model residuals. Align and normalize independently retrieved helicity
objects before computing quantitative XMCD. Phase is wrapped, and a visually
plausible image does not establish a unique physical solution.
