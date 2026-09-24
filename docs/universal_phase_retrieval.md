# Universal phase retrieval: user guide

## Implementation and entry points

Use `library/phase_retrieval_universal.py` for joint state, polarization, energy,
and illumination retrieval. Ajajas 02 and MAX IV 05 use this implementation.
MAX IV's `library/phase_retrieval*.py` entries are relative symbolic links to the
repository library. Keep the repository layout intact and restart kernels after
editing imported modules. The historical core modules remain for compatibility;
they are not separate copies of the universal physical workflow.

Main entry points:

- `default_universal_phase_retrieval_recipe()` returns a new default dictionary.
- `universal_phase_retrieval_algorithm(...)` validates metadata and dispatches.
- `format_universal_workflow(recipe, state_labels=...)` returns the resolved tree.
- `print_universal_workflow(...)` prints it.
- `plot_universal_workflow(...)` returns a Matplotlib figure; call `plt.show()` or
  `figure.savefig(...)`. Both main notebooks always print and plot before running.
- `general_phase_retrieval_algorithm(...)` is the lower-level joint driver.

Unknown recipe keys raise errors. Prefer the universal wrapper over calling
internal helpers. The default dictionary reference at the end lists every key;
not every option applies to every projection model.

## Arrays and metadata

`holograms` has shape `(observations, rows, columns)` and contains **intensities**,
not amplitudes. `mask_pixel` is either detector-shaped or observation-shaped:
zero means observed; nonzero means invalid/unmeasured. The algorithm's returned
`bsmasks` can additionally mark values rejected by the intensity cutoff. Use an
explicit mask and a negative cutoff when genuine zero observations should count.

`supportmask` describes allowed object-space pixels. The geometry layer applies
crop/binning and support transforms. Modal supports can be supplied where supported;
`modes=[1, 2]` requests two incoherent fields and enlarges the second support by
factor two. The entries describe support factors, not magnetic quantum numbers.

Supply one label/coefficient per observation:

| Argument | Meaning |
|---|---|
| `state_labels` | Which magnetization map applies |
| `energy_labels` | Which spectral response applies |
| `polarization_coefficients` | Magnetic sign/weight, often +1 or −1 |
| `illumination_labels` | Which common illumination applies |

One field returns as `(observations, rows, columns)`; two modes add a modal axis
`(observations, 2, rows, columns)`. Detector intensity is the **sum of modal
intensities**, not the magnitude squared of the summed complex fields.

## Working example

Assume centered measured intensities, mask and support are already loaded:

```python
import phase_retrieval_universal as universal
import matplotlib.pyplot as plt

recipe = universal.default_universal_phase_retrieval_recipe()
recipe.update(
    modes=[1, 2], mode_initialization="support_fft",
    startimage_scale_fit="through_origin",
    startimage_radial_normalization=True,
    projection_model="physical_factorized",
    constrain_nonphysical_modes_common=True,
    nonphysical_modes_common_relaxation=1.0,
    partial_coherence=True, coherence_kernel_scope="shared",
    shared_coherence_update="average",
    preserve_warmup_masked_intensity=True,
    warmup_mode=["HAPRE", "ER"], warmup_Nit=[200, 50],
    warmup_RL_it=[0, 20], warmup_RL_freq=[10**9, 10],
    inner_mode=["ER"], inner_Nit=[50], RL_it=20, RL_freq=10,
    outer_iterations=10,
    coherent_refresh_rounds=[4, 8],
    coherent_refresh_mode="ER", coherent_refresh_iterations=50,
    projection_every=len(holograms),
    final_projection_relaxation=0,
    observation_workers=2, fft_workers=1,
)
universal.print_universal_workflow(recipe, state_labels=states)
figure = universal.plot_universal_workflow(recipe, state_labels=states)
plt.show()
fields, warmup, components, bsmasks, errors = (
    universal.universal_phase_retrieval_algorithm(
        holograms, mask_pixel, supportmask,
        state_labels=states, energy_labels=energies,
        polarization_coefficients=polarizations,
        illumination_labels=beams, universal_recipe=recipe,
    )
)
```

Iteration counts here are examples, not convergence guarantees. Physical fitting
needs sufficient state/polarization/energy diversity. Inspect identifiability and
residuals in `components`, not just the appearance of a reconstruction.

## What happens, in order

1. Prepare detector/support geometry; validate labels and physical inputs.
2. Initialize fields or copy supplied `start_fields`.
3. Warm up observations, optionally seeding others from a retrieved reference.
4. When masked-fill preservation is enabled, finish **all coherent warmups** before
   starting partial-coherence warmups. Capture each coherent state's intensity.
5. For each outer round: optionally refresh coherently, run detector/support
   updates, then apply physical projections according to the configured cadence.
6. Apply configured final constraints and enforce the common secondary mode.
7. Return fields, warmup snapshot, components, masks and diagnostics.
8. Notebook-only optional quasi-Poisson refinement creates a separate result.

Warmup schedules with preservation must have coherent stages followed by partial
stages; alternating back and forth inside warmup is not supported. Use the explicit
outer-round refresh controls for repeated transitions.

## Initialization and the radial envelope

Both main notebooks default to independent coherent HAPRE × 750 + ER × 50
warmups for every observation, including opposite polarizations. Set
`WARMUP_START_FROM_REFERENCE=True` to seed from the selected reconstructed
reference instead. Ajajas additionally exposes `WARMUP_POLICY`: `same_for_all`
uses this schedule everywhere, while `reference_and_others` enables separate
reference/other schedules. Field seeding and shared-gamma calibration are separate.
Independent support starts do not imply random independent phases; physical
projections subsequently couple the fields but cannot guarantee resolution of
every phase ambiguity.

`mode_initialization="support_fft"` makes modal starts from inverse transforms of
supports; `random_phase` is the multimode alternative. Global normalization uses
`startimage_scale_fit`. `through_origin` avoids fitting an intercept; `linear`
uses a regression slope where supported by the initializer.

With `startimage_radial_normalization=True`, compute one-pixel annular means from
finite, nonnegative hologram samples with `mask_pixel==0`, including true zeros.
For each detector pixel set total modal intensity to its annulus's mean. Preserve
phase and relative modal amplitudes. At an exact zero-field pixel, put the target
amplitude into mode 1 at phase zero. Empty rings interpolate; beyond populated
rings use the nearest endpoint. The center is `(rows//2, columns//2)` on the
prepared detector grid, so first center the experimental diffraction pattern.

This applies after global normalization and to reference-seeded warmup starts.
It does not modify explicitly supplied `start_fields` on entry (later reference
seeding can replace another state's start). It is not a radial-symmetry constraint
on subsequent retrieval. Radial intensity matching can disturb exact initial
support membership; the iterations restore support consistency.

## Coherence, gamma and invalid pixels

`partial_coherence=False` is full coherence. A scheduled partial stage needs
`partial_coherence=True`, positive `RL_it`, and `RL_freq <= Nit`. `RL_it` controls
Richardson–Lucy kernel updates; `RL_freq` controls their cadence. Coherent stages
leave invalid pixels free. Partial stages use captured intensity fills at masked
pixels and fit through the blur model.

| Control | Effect |
|---|---|
| `coherence_kernel_scope="per_observation"` | Each observation has its own gamma |
| `coherence_kernel_scope="shared"` | One gamma shared over observations |
| `shared_coherence_update="sequential"` | Carry the updated gamma to the next observation |
| `shared_coherence_update="average"` | Fit each observation from one snapshot, normalize and weight-average kernels |
| `shared_coherence_update="pooled"` | Hold gamma during field updates, then fit all observations jointly |
| `coherence_round_iterations` | Iterations of the pooled kernel fit |
| `warmup_calibrate_shared_gamma` | Fit the reference kernel before synchronous partial warmup |
| `freeze_coherence_after_reference` | Keep the fitted reference kernel fixed; use shared sequential scope |
| `observation_weights` | Weights in applicable joint/kernel fits |

Average/pooled strategies require shared scope and masked warmup preservation.
A common kernel across energies is an experimental assumption; MAX IV defaults
to per-observation scope. Detector blur and partial coherence can be difficult to
separate; a fitted gamma is not automatically a calibrated detector PSF.

### Partial → full → partial, repeatedly

The notebooks expose `REFRESH_MASK_AFTER_OUTER_LOOPS=[3, 7]`: refresh after
completed loops 3 and 7. They translate this into library before-round indices
`coherent_refresh_rounds=[4, 8]`. Choose completed loops smaller than the total.

Set `coherent_refresh_rounds=[4, 8]` to refresh **before** outer rounds 4 and 8.
Numbers are one-based, unique and no larger than `outer_iterations`. Round 1 is
allowed (it follows warmup). An empty list disables refresh.

Each refresh starts from current fields, runs `coherent_refresh_mode` for
`coherent_refresh_iterations` with RL off, passes the original invalid mask, and
supplies neither gamma nor frozen intensity targets to that update. It inherits
beta/alpha/TV controls from the first inner stage. It then captures total modal
intensity and replaces the fills for subsequent partial updates. Actual observed
intensities are never replaced. Gamma is retained externally and resumes fitting
under the selected strategy. A refresh does not reset the field from support.

All states complete the coherent pass before the next partial round. Frozen
saturated observations stay frozen; their fills are captured unchanged. No
extra physical projection is inserted between refresh and capture. Mode 2 may
vary during this independent pass; the regular common-mode projection restores
its shared form. This is alternating projection, not a guarantee that all
constraints hold after every intermediate operation.

Refresh requires partial coherence enabled and the general driver
(`physical_factorized`, `state_energy_beam`, or `none`). Pure-energy `svd` and
`rank1_spectral` dispatch to a legacy driver and reject refresh. Configure active
partial inner stages to actually resume partial coherence after a refresh.

## Physical model and the secondary mode

The primary log-object is modeled as

`log(object[a]) = log(illumination[beam[a]]) + thickness * (charge[E[a]] + polarization[a] * magnetic[E[a]] * magnetization[state[a]])`.

Only mode 1 enters this model. With `constrain_nonphysical_modes_common=True` and
`nonphysical_modes_common_relaxation=1`, mode 2 is phase-aligned and complex-averaged
within each energy/illumination group. It has no independent magnetic map or
state/polarization variation at the common projection and final output. Different
energies or beams need not have identical secondary fields. Use these settings in
both notebooks; the generic library default leaves this constraint optional.

`projection_relaxation` sets how strongly the physical result replaces the current
primary field. `projection_every`/`projection_start` control cadence. Parallel
updates require projection boundaries at complete observation sweeps.
`physical_iterations` controls the internal physical fit. `final_projection_relaxation=0`
omits an additional final physical replacement. A final Fourier constraint can
alter physical consistency; inspect both detector and physical residuals.

Material controls include `material_mask`, `material_thickness`,
`fit_material_thickness`, known vacuum and support constraints. Set
`physical_projection_object_roi=True` with a thickness map to fit only the material
aperture, preserving the exterior. Full-frame FFT propagation remains necessary.
`physical_phase_reference=True` chooses the nearest relative phase branch within
each energy/beam group before log fitting; it assumes phase differences below π.
It is not a general phase-unwrapping algorithm.

Charge and magnetic spectral constraints, known spectra and bounds can constrain
ill-conditioned fits. Refractive-index spectra require thickness/wave-number
conversion; explicit `kt` products do not. See the complete option inventory and
the validation messages for mutually exclusive inputs. Absolute thickness and
spectral response scale remain coupled without external calibration.

## Performance, geometry and reproducibility

- `observation_workers` controls concurrent observations; `fft_workers` controls
  FFT threads per worker. Start with 2 and 1. More workers multiply field/history
  memory and can reduce performance. GPU execution uses one observation worker.
- `average_img`, support size, crop and binning strongly affect memory/runtime.
- `crop` removes detector edges before `binning`; binning sums intensities and
  propagates invalid masks. Object support transforms follow the resulting grid.
- `random_seed`, `mode_initialization_seed` and `shuffle_observations` control
  stochastic initialization/order. Save the resolved recipe and metadata.
- Notebook display ROIs use `support_bounding_roi` on the actual retrieval support.

## Outputs, diagnostics and refinement

`warmup` is the snapshot after the configured warmup, including partial warmup if
present. `components` contains fitted physical maps/spectra, geometry, modal
supports and fitted kernels. `errors` contains update/projection records and
resolved settings. `coherent_refresh_steps` records refresh rounds/observations
and frozen status. Set `projection_diagnostic_observation` to trace masked outer
intensity and observed-amplitude residual through projection stages and refreshes.
The workflow figure is a plan, not a convergence plot.

`poisson_refinement.refine_poisson` is a separate projected gradient polish using
a quasi-Poisson objective. For dark-subtracted, thresholded, averaged camera ADU,
it is not a calibrated photon likelihood or the cited paper's exact FIVS method.
Keep baseline fields; refine mode 1 only (`refine_modes=[0]`) to preserve a common
secondary mode; hold gamma fixed. The refined field need not satisfy the joint
physical model, so baseline component maps must not be relabeled as refined maps.

Compare object images with common color limits, predicted detector intensities,
valid-pixel residual/deviance, and optimization histories. Refinement initially
projects onto support; its first loss can exceed the unprojected baseline.

## Troubleshooting

| Symptom | Checks |
|---|---|
| Outer masked artifacts | Initialization, support geometry, phase branch, then coherent refresh; do not freeze masks in full coherence |
| Mode 2 differs across states | Enable common constraint at relaxation 1; compare final outputs within the same energy/beam group |
| Slow parallel run | Memory, worker count, FFT oversubscription, and actual scheduling strategy |
| No apparent gamma updates | Partial flag, RL cadence versus stage length, freeze option, kernel scope |
| Physical fit worsens detector agreement | Relaxation/cadence, material aperture, phase-reference assumption, model identifiability |
| Changes appear ignored | Rerun recipe construction; restart kernel after library edits; inspect printed tree |

## Complete recipe defaults

The following inventory is generated from `default_universal_phase_retrieval_recipe()`.
Arrays/known inputs defaulting to `None` must be provided on the appropriate
observation, energy or object grid. Defaults describe the API; notebooks override
many of them. Keys grouped by prefixes (warmup, known, charge, magnetic, thickness)
apply to their corresponding stage or physical quantity.

| Option | Default |
|---|---|
| `inner_mode` | `['HAPRE']` |
| `inner_Nit` | `[1]` |
| `outer_iterations` | `300` |
| `warmup_mode` | `['HAPRE']` |
| `warmup_Nit` | `[20]` |
| `warmup_reference_mode` | `None` |
| `warmup_reference_Nit` | `None` |
| `warmup_other_mode` | `None` |
| `warmup_other_Nit` | `None` |
| `warmup_start_from_first` | `False` |
| `warmup_reference_observation` | `0` |
| `startimage_scale_fit` | `'linear'` |
| `startimage_radial_normalization` | `False` |
| `warmup_seed_scale_fit` | `'sum'` |
| `warmup_seeded_stage_indices` | `None` |
| `freeze_saturated_fields` | `False` |
| `partial_coherence` | `False` |
| `coherence_kernel_scope` | `'per_observation'` |
| `shared_coherence_update` | `'sequential'` |
| `warmup_calibrate_shared_gamma` | `False` |
| `coherence_round_iterations` | `50` |
| `coherent_refresh_rounds` | `[]` |
| `coherent_refresh_iterations` | `50` |
| `coherent_refresh_mode` | `'ER'` |
| `observation_workers` | `1` |
| `fft_workers` | `1` |
| `preserve_warmup_masked_intensity` | `False` |
| `freeze_coherence_after_reference` | `False` |
| `shuffle_observations` | `True` |
| `random_seed` | `None` |
| `beta_zero` | `0.5` |
| `beta_mode` | `'arctan'` |
| `alpha_zero` | `0.0` |
| `alpha_mode` | `'const'` |
| `TV_freq` | `1000000000.0` |
| `RL_it` | `0` |
| `RL_freq` | `1000000000.0` |
| `warmup_beta_zero` | `None` |
| `warmup_beta_mode` | `None` |
| `warmup_alpha_zero` | `None` |
| `warmup_alpha_mode` | `None` |
| `warmup_TV_freq` | `None` |
| `warmup_RL_it` | `None` |
| `warmup_RL_freq` | `None` |
| `warmup_reference_beta_zero` | `None` |
| `warmup_reference_beta_mode` | `None` |
| `warmup_reference_alpha_zero` | `None` |
| `warmup_reference_alpha_mode` | `None` |
| `warmup_reference_TV_freq` | `None` |
| `warmup_reference_RL_it` | `None` |
| `warmup_reference_RL_freq` | `None` |
| `warmup_other_beta_zero` | `None` |
| `warmup_other_beta_mode` | `None` |
| `warmup_other_alpha_zero` | `None` |
| `warmup_other_alpha_mode` | `None` |
| `warmup_other_TV_freq` | `None` |
| `warmup_other_RL_it` | `None` |
| `warmup_other_RL_freq` | `None` |
| `plot_every` | `1000000000.0` |
| `average_img` | `1` |
| `Fourier_last` | `True` |
| `final_fourier_constraint` | `True` |
| `hologram_intensity_cutoff_vmin` | `-1` |
| `binning` | `1` |
| `crop` | `0` |
| `roi` | `None` |
| `Nmodes` | `1` |
| `modes` | `None` |
| `mode_support_center` | `'image'` |
| `recenter_modal_supports` | `False` |
| `mode_initialization` | `'random_phase'` |
| `mode_initialization_seed` | `0` |
| `constrain_nonphysical_modes_common` | `False` |
| `nonphysical_modes_common_relaxation` | `1.0` |
| `projection_model` | `'physical_factorized'` |
| `projection_every` | `None` |
| `projection_start` | `None` |
| `projection_relaxation` | `1.0` |
| `final_projection_relaxation` | `1.0` |
| `observation_weights` | `None` |
| `rank_deficient` | `'error'` |
| `physical_iterations` | `20` |
| `physical_projection_object_roi` | `False` |
| `physical_phase_reference` | `False` |
| `projection_diagnostic_observation` | `None` |
| `projection_focus_prop_um` | `0.0` |
| `projection_focus_phase_rad` | `0.0` |
| `projection_focus_setup` | `None` |
| `projection_focus_integer_wavelength` | `True` |
| `material_mask` | `None` |
| `material_thickness` | `None` |
| `fit_material_thickness` | `False` |
| `saturated_states` | `None` |
| `zero_magnetization_outside_support` | `False` |
| `zero_thickness_outside_support` | `False` |
| `projection_constraints_inside_support_only` | `False` |
| `physical_constraints_inside_support_only` | `False` |
| `charge_spectral_constraint` | `'free'` |
| `magnetic_spectral_constraint` | `'free'` |
| `energy_values` | `None` |
| `known_charge_beta_spectrum` | `None` |
| `known_charge_delta_spectrum` | `None` |
| `known_magnetic_beta_spectrum` | `None` |
| `known_magnetic_delta_spectrum` | `None` |
| `charge_absorption_part` | `'real'` |
| `magnetic_absorption_part` | `'real'` |
| `charge_response_real_range` | `None` |
| `charge_response_imag_range` | `None` |
| `magnetic_response_real_range` | `None` |
| `magnetic_response_imag_range` | `None` |
| `kk_sign` | `1.0` |
| `kk_subtract_baseline` | `True` |
| `kk_normalize_input` | `False` |
| `known_spectrum_normalization` | `'none'` |
| `fit_known_spectrum_scale` | `True` |
| `fit_known_spectrum_offset` | `True` |
| `log_floor` | `1e-12` |
| `rank` | `1` |
| `projection_static_mode` | `'mean'` |
| `spectral_constraint` | `'free'` |
| `known_beta_spectrum` | `None` |
| `known_delta_spectrum` | `None` |
| `absorption_part` | `'real'` |
| `known_beta_normalization` | `'none'` |
| `fit_known_beta_scale` | `True` |
| `fit_known_beta_offset` | `True` |
| `charge_kt_delta_range` | `None` |
| `charge_kt_beta_range` | `None` |
| `magnetic_kt_delta_range` | `None` |
| `magnetic_kt_beta_range` | `None` |
| `wave_numbers` | `None` |
| `thickness` | `None` |
| `known_charge_kt_beta_spectrum` | `None` |
| `known_charge_kt_delta_spectrum` | `None` |
| `known_magnetic_kt_beta_spectrum` | `None` |
| `known_magnetic_kt_delta_spectrum` | `None` |
