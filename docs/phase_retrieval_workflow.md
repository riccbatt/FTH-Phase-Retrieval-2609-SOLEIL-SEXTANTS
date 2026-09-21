# Phase retrieval API and workflow guide

This guide covers the current unified and universal phase retrieval APIs. The
Ajajas notebooks provide concrete two-image and hysteresis-loop examples; the
MaxIV hyperspectral notebook uses the same universal recipe system for an
energy series.

## Choose the driver

| Task | Driver | Example |
| --- | --- | --- |
| One related pair, including coherent, partially coherent, or multimode retrieval | `library.phase_retrieval_core_unified.phase_retrieval_algorithm` | `aperiodic_ajajas/01_phase_retrieval_2871_2872.ipynb` |
| Many states, energies, helicities, or illuminations with a shared physical model | `library.phase_retrieval_universal.universal_phase_retrieval_algorithm` | `aperiodic_ajajas/02_universal_hysteresis_2871_2872.ipynb` |
| Hyperspectral separation | Universal driver with energy labels and a spectral projection model | `maxiv_phase_test/05_maxiv_hyperspectral_phase_retrieval.ipynb` |

The universal driver first reconstructs each hologram during warmup. It then
alternates detector/support updates with a joint object-space projection.

## Data preparation

Prepare these arrays on one detector grid:

- `holograms`: `(n_observations, rows, columns)` measured intensities.
- `mask_pixel`: either `(rows, columns)` or one mask per observation. A nonzero
  pixel is excluded from the measured-amplitude constraint.
- `supportmask`: the real-space support. It may be on the source grid or on the
  final retrieval grid.
- Metadata arrays with one entry per hologram: state, energy, polarization
  coefficient, and illumination label.

The geometry operation is always **crop first, then bin**. The library applies
the same mapping to holograms, detector masks, support masks, material masks,
and thickness maps. Avoid manually binning one input without applying the same
geometry to the others.

Inspect dark subtraction, normalization, detector masks, and support overlays
before retrieval. Phase retrieval cannot repair a shifted mask or a support
defined on a different grid.

If an object-space filtering step crops the temporary inverse transform before
recalculating a hologram, the Fourier grid also changes shape. Center-crop the
object support without changing its pixel scale, and resize detector masks to
the new Fourier shape. The Ajajas helper
`adapt_retrieval_array_shapes(holograms, mask_pixel, supportmask)` performs
these checks and reports its changes.

## Minimal universal API

```python
import numpy as np
from library import phase_retrieval_universal as universal

recipe = universal.default_universal_phase_retrieval_recipe()
recipe.update(
    projection_model="physical_factorized",
    crop=400,
    binning=1,
    modes=[1],
    saturated_states={"saturated": +1},
    warmup_mode=["HAPRE", "ER"],
    warmup_Nit=[700, 50],
    inner_mode=["ER"],
    inner_Nit=[50],
    outer_iterations=10,
)

universal.print_universal_workflow(recipe, state_labels=state_labels)

fields, warmup_fields, components, bsmasks, errors = (
    universal.universal_phase_retrieval_algorithm(
        holograms,
        mask_pixel,
        supportmask,
        state_labels=state_labels,
        energy_labels=energy_eV,
        polarization_coefficients=polarization,
        illumination_labels=beam_labels,
        universal_recipe=recipe,
    )
)
```

`fields` and `warmup_fields` are Fourier-domain exit waves. Convert a field to
object space with the same centered FFT convention used by the notebooks:

```python
obj = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fields[index])))
```

For `modes=[1, 2]`, the field shape includes a mode axis. Array mode 0 is the
support-factor-1 mode and is the mode used by the physical projection.

## Warmup recipes

### One recipe for every hologram

Leave the role-specific settings as `None`:

```python
recipe.update(
    warmup_mode=["HAPRE", "ER"],
    warmup_Nit=[700, 50],
    warmup_reference_mode=None,
    warmup_reference_Nit=None,
    warmup_other_mode=None,
    warmup_other_Nit=None,
    warmup_start_from_first=False,
)
```

Every hologram uses HAPRE 700 followed by ER 50. Set
`warmup_start_from_first=False` so every hologram uses its independently
initialized support field, or set it to `True` to seed nonreference holograms
from the scaled reference result.

### A reference recipe and a different recipe for all other holograms

```python
recipe.update(
    warmup_reference_observation=0,
    warmup_start_from_first=True,
    warmup_seed_scale_fit="linear",  # alternatively "sum"
    warmup_reference_mode=["HAPRE", "ER"],
    warmup_reference_Nit=[700, 50],
    warmup_other_mode=["ER"],
    warmup_other_Nit=[50],
)
```

This reconstructs observation 0 with HAPRE+ER. Every other hologram starts
from an intensity-scaled copy of that reference field and runs only ER. This is
the default policy in the Ajajas universal hysteresis notebook.

Each role also accepts its own controls:

```text
warmup_{reference|other}_{beta_zero|beta_mode}
warmup_{reference|other}_{alpha_zero|alpha_mode}
warmup_{reference|other}_{TV_freq}
warmup_{reference|other}_{RL_it|RL_freq}
```

If a role-specific control is `None`, it inherits the corresponding shared
`warmup_*` value. The older `warmup_seeded_stage_indices` option remains
supported when no explicit `warmup_other_*` schedule is supplied.

## Partial coherence

Enable partial coherence with:

```python
recipe.update(
    partial_coherence=True,
    coherence_kernel_scope="shared",  # same gamma for every state
    final_fourier_constraint=False,
)
```

A stage performs Richardson-Lucy coherence updates only when `RL_it > 0` and
`RL_freq <= Nit`. For example:

```python
recipe.update(
    warmup_reference_mode=["HAPRE", "ER", "HAPRE", "ER"],
    warmup_reference_Nit=[700, 50, 700, 50],
    warmup_reference_RL_it=[0, 0, 50, 50],
    warmup_reference_RL_freq=[10**9, 10**9, 20, 20],
    warmup_other_mode=["ER", "ER"],
    warmup_other_Nit=[50, 50],
    warmup_other_RL_it=[0, 50],
    warmup_other_RL_freq=[10**9, 20],
)
```

This gives the reference a coherent HAPRE+ER pass followed by a partially
coherent HAPRE+ER pass. Other observations receive coherent ER followed by
partially coherent ER.

## Joint reconstruction and physical projection

`inner_mode` and `inner_Nit` describe the detector/support update applied to
each observation during every outer loop. `projection_every` counts completed
observation updates; `None` means once per full sweep.

```python
recipe.update(
    inner_mode=["ER"],
    inner_Nit=[50],
    outer_iterations=10,
    projection_every=len(state_labels),
    projection_model="physical_factorized",
    physical_iterations=10,
    projection_relaxation=0.1,
    final_projection_relaxation=0.0,
)
```

`projection_relaxation` controls the repeated projections. A value of zero for
`final_projection_relaxation` still calculates final physical components but
does not replace the final iterated fields. `final_fourier_constraint` controls
the last measured-amplitude projection independently.

### Focus the sample for the physical projection

The charge, thickness, and magnetization model should act in the focused
sample plane. If the retrieved object needs propagation for focus, configure a
reversible transform around every physical projection:

```python
recipe.update(
    projection_focus_prop_um=2.05,
    projection_focus_phase_rad=-0.76,
    projection_focus_setup={
        "ccd_dist": 0.125,
        "px_size": 11e-6,
        "energy": ENERGY_EV,
    },
)
```

The driver applies the same Fourier-plane propagation as `fth.propagate`,
rotates the focused object by the configured global phase, performs the
physical projection, and applies the exact inverse transform before returning
to detector/support iterations. The reported physical components therefore
belong to the focused sample plane, while `fields` remain in the original
detector-plane convention.

Set `projection_focus_prop_um=0` to skip both propagation and phase rotation.
If `projection_focus_setup` omits `energy`, each observation's numeric energy
label is used; this supports hyperspectral datasets. Set
`projection_focus_integer_wavelength=False` only when the propagation distance
should not be rounded to an integer wavelength as in the default
`fth.propagate` convention.

For hysteresis data, give every field point a distinct state label. Use
`saturated_states={"saturated": +1}` to anchor its magnetization, and
`freeze_saturated_fields=True` to preserve its reconstructed field during the
joint loops.

## Thickness, holes, and multimode retrieval

- `material_mask`: nonzero where material may be present and zero in known
  vacuum regions or reference holes.
- `material_thickness`: fixed nonnegative relative thickness.
- `fit_material_thickness=True`: fit a shared nonnegative thickness while
  respecting known zero pixels.
- `zero_thickness_outside_support=True`: force thickness to zero outside the
  phase-retrieval support.
- `largest_support_component(supportmask)`: create a binary thickness aperture
  from the largest support component.

Magnetization is constrained to zero where material thickness is zero.

Set `modes=[1, 2]` for two incoherent modes. Both modes receive detector and
support updates. The physical charge, magnetization, and thickness projection
acts only on support-factor-1 mode.

To prevent higher modes from absorbing state-dependent magnetic contrast, use:

```python
recipe.update(
    constrain_nonphysical_modes_common=True,
    nonphysical_modes_common_relaxation=1.0,
)
```

At every physical projection, modes other than support-factor 1 are replaced
by a state-independent common log-object. Observations are grouped by energy
and illumination, so a hyperspectral or multi-beam dataset may retain a
different common higher mode for each energy/beam pair. Polarization and state
do not create separate higher-mode components. A relaxation of 1 makes the
higher mode identical across states immediately; smaller values approach the
common object gradually. Detector/support updates between physical projections
may temporarily separate the modes again.

For a two-mode comparison with the unified driver, set
`mode_initialization="support_fft"`; both modes then start from the Fourier
transform of their support and use the same amplitude normalization as the
unified code. `mode_initialization="random_phase"` keeps the alternative
nondegenerate random modal start controlled by `mode_initialization_seed`.

## Inspect the workflow before running

```python
text = universal.print_universal_workflow(
    recipe,
    state_labels=state_labels,
    start_fields_provided=False,
)
```

Use `format_universal_workflow(...)` when the text should be saved rather than
printed. The tree shows:

- the reference observation;
- the source of every start field;
- reference and nonreference warmup stages;
- partial-coherence updates;
- inner updates and physical-projection cadence;
- modes, frozen saturated fields, and final constraints.

## Outputs and diagnostics

- `fields`: final retrieved Fourier fields.
- `warmup_fields`: fields immediately before joint physical iterations.
- `components`: charge, magnetization, thickness, mode, coherence, and model
  diagnostics.
- `bsmasks`: the detector masks actually used after geometry processing.
- `errors`: per-stage errors, recipe settings, metadata, and projection history.

Compare `warmup_fields` between notebook 01 and notebook 02 before diagnosing
the physical model. If warmup already differs, inspect recipe stages,
start-image handoffs, masks, crop/bin geometry, intensity normalization, and
partial coherence. If warmup agrees but final fields diverge, inspect the inner
schedule and projection settings.
