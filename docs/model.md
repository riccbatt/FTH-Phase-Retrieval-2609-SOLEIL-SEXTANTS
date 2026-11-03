# Physical model and algorithm

For observation a, the physical projector fits the complex object logarithm

    L_a(r) = c_b(a)(r) + t(r) [q_c(E_a) + p_a q_m(E_a) m_s(a)(r)].

Here c is a shared illumination/common object field, t is relative material thickness, p is the signed polarization coefficient, m is a real spatial magnetization factor, and q_c/q_m are complex energy responses. Observation metadata independently specify energy, polarization, magnetic state and illumination group. Relative thickness is zero in vacuum reference holes.

## Initialization, loop, output

`initialize_phase_retrieval` prepares geometry, masks, modal fields and the recipe schedule. `universal_phase_retrieval_algorithm` executes one loop. Every iteration updates all observations: Fourier magnitude, support/optional TV, scheduled physical fitting, then scheduled Richardson–Lucy (RL) kernel fitting.

The default recipe runs **700 HAPRE followed by 50 ER**. Physical fitting, TV and RL fitting are disabled by default. `default_universal_phase_retrieval_recipe(RL=True)` appends another **700 HAPRE + 50 ER**, both with `RL_it=50` and `RL_every=20`.

```python
from library.phase_retrieval_universal import (
    default_universal_phase_retrieval_recipe,
    initialize_phase_retrieval,
    universal_phase_retrieval_algorithm,
)

recipe = default_universal_phase_retrieval_recipe(RL=True)
recipe.update(state=state, energy=energy,
              polarization=polarizations, illumination=illumination)
reconstruction = initialize_phase_retrieval(holograms, detector_mask, support, recipe)
fields, initial, components, masks, errors = universal_phase_retrieval_algorithm(reconstruction)
```

Each stage is a dictionary. A custom stage can be written as:

```python
recipe['stages'] = [{
    'it': 300,
    'mode': 'HAPRE',
    'physical_it': 20,
    'physical_every': 10,
    'tv_every': False,
    'beta_zero': 0.5,
    'beta_mode': 'arctan',
    'RL_it': False,
    'RL_every': False,
}]
```

| Setting | Meaning |
| --- | --- |
| `it` | Retrieval iterations in this stage; `False` skips the stage. |
| `mode` | Support update: HAPRE, RAAR, ER, etc. |
| `physical_it`, `physical_every` | Physical-fit iterations per call, and retrieval iterations between calls. |
| `RL_it`, `RL_every` | RL-fit iterations per call, and retrieval iterations between calls. |
| `tv_every` | Apply one TV descent step every this many retrieval iterations; `False` disables it. |
| `tv_step` | TV step size, default `1e-3`; unused while `tv_every=False`. |
| `beta_zero`, `beta_mode` | Support-feedback coefficient and its stage-local schedule. |

If either member of an `it`/`every` pair is `False`, that operation never runs in the stage. Cadences start at their first multiple: `RL_every=20` means iterations 20, 40, 60, … within each stage. Fields, object feedback and fitted RL kernels carry across stages; beta schedules and cadence counters restart. No extra projection runs after the schedule.

Both RL controls must be enabled to use the blurred detector model in a stage. The current kernel is used on every Fourier step of that stage, while kernel fitting occurs only at `RL_every`. Missing-pixel intensity fills are captured at the start of each RL stage. `RL_scope` selects `per_observation` or `shared`; shared fitting uses `RL_shared_update` (`sequential`, `average`, or `pooled`). Output kernels are stored as `components['RL_kernels']`.

Physical components include `physical_fit_iteration`, identifying the iteration that produced their maps. Convergence output contains only detector residuals per iteration and residuals at physical-fit calls. CPU FFT threading is controlled by `fft_workers`.

The numerical modules have separate responsibilities:

- `phase_retrieval_universal.py`: the main loop and constraint order.
- `phase_retrieval_initialization.py`: initialization and geometry preparation.
- `magnitude_constraints.py`: Fourier magnitude and RL kernel fitting.
- `real_space_constraints.py`: support construction, support algorithms and TV.
- `phase_retrieval_physical.py`: physical fitting, object transforms and optional KK.
- `phase_retrieval_recipe.py` / `phase_retrieval_validation.py`: defaults, schedules and input contracts.

## Calibration and identifiability

Saturation anchors are a dictionary mapping **state values** to magnetization signs, for example `state=[10, 20, 30]` with `saturated_states={10: +1, 30: -1}`. These keys are not observation positions. An anchored state is fixed at that sign for all its energies, polarizations and illumination groups. `None` or `{}` leaves all states unanchored.

Use independently established saturation labels. Both notebooks start without saturation anchors. Hysteresis without anchors is a relative factor, and magnet current remains amperes until a measured current-to-field conversion is supplied. No monotonic trend or binary final magnetization is imposed by the notebooks.

For exp(-i k (n-1) d), use n=1-delta-i beta. With d=d0*t, q=-i*k*d0*(n-1), giving beta=-Re(q)/(k*d0) and delta=Im(q)/(k*d0). Conversion requires measured reference thickness. The common charge offset, magnetic scale and sign must also be fixed or reported as ambiguous. Thickness alone does not remove these gauges. Inspect `magnetic_scale_anchored`, `charge_absolute_gauge_anchored` and `identifiable` in returned components.

Use `charge_spectral_constraint="free"` or `"kk"` and the corresponding magnetic setting. KK uses fitted responses and the supplied energy axis. Its finite spectral window and unmeasured tails require a defensible treatment before interpreting absolute indices.

The returned errors contain only detector residuals per iteration and residuals at physical-fit iterations. Physical projection always returns projected fields and fitted components.

Reuse returned fields by passing `start_fields=fields` to initialization with matching observation order, mode count and retrieval grid. Retrieval-grid starts bypass detector crop/bin preparation; source-grid starts are prepared once. To seed several observations from one result, select and repeat that result along the observation axis. This reuses fields; physical components and RL kernels are fitted afresh.
