# Alternating summed-helicity retrieval

Both notebooks retain the existing retrieval architecture:

1. Warm up every measured sum with the configured HAPRE/ER recipe.
2. Apply `project_fourier_fields_general` jointly to the first two modes,
   labeled positive and negative helicity across energies or states.
3. Feed those projected fields into the existing phase-retrieval kernel,
   fitting the incoherent intensity sum with the configured modal supports.
4. Repeat steps 2 and 3, carrying forward the updated Fourier fields.

`MODES=[1,1]` fits two helicity candidates. `[1,1,2]` adds a nonmagnetic
nuisance mode, which does not enter the magnetic fit. Notebook 06 can make
that background common across states and fixes state 0 to uniform negative
saturation in the material aperture. Saturation does not mean zero magnetic
transmission. No separate measured helicity image is a retrieval target.

The physical model remains
`log psi = C + t * (qc(E) +/- qm(E) * m(state))`.
The projector fits it anew to the current fields at each cycle, as in the
original framework. No direct L-BFGS model optimization is used.

## Controls and diagnostics

`ITERATIONS` controls warmup, `INNER_ITERATIONS` controls phase-retrieval
steps between projections, and `OUTER_ITERATIONS` controls the number of
cycles. `PROJECTION_RELAXATION` sets the physical projection strength. The
shorter default schedule is a starting point for experimentation.

`history` records sum errors immediately after physical projection;
`retrieval_history` records errors after the following phase retrieval.
The saved final fields are after phase retrieval; the saved physical
components belong to the preceding fit and need not exactly describe the
final fields. `pre_detector_fields` retains its compatibility name but now
stores the last physical-stage fields before that full retrieval block.

Held-out difference error is compared with the zero-difference baseline of
1, using only a single global helicity swap for interpretation. A low sum
error alone is not evidence of separation. Tests exercise both notebooks,
both mode counts, the physical-then-retrieval ordering, fixed saturation,
diagnostics, and HDF5 export on small synthetic inputs. Successful full-data
separation has not been established.

## Literature audit and modal-subspace control (notebook 07)

The shared kernel already stores independent complex `(mode, y, x)` fields.
`_apply_modal_fourier_constraint` rescales every mode by the same factor
computed from `sqrt(sum(abs(field)**2))`. `_mode_supports` gives `[1,1]`
identical supports. HAPRE and ER act separately with the same support and
schedule; the HAPRE expression has no positivity condition. There is no
helicity labeling, mode sorting or SVD in this kernel. Initialization uses
independent random phases and a common total-power normalization. These
notebooks disable partial-coherence updates and use `average_img=1`.
The nonlinear physical projector does explicitly label the first two modes
as opposite helicities; this is a prior, not a consequence of `[1,1]`.
The extra support in `[1,1,2]` does not identify a unique background either.

The [PRL main paper](https://doi.org/10.1103/PhysRevLett.134.016704)
uses two object modes and ptychographic overlap. It discusses unitary basis
ambiguity and biases initialization toward linear charge/magnetic channels
using a positive-real first mode and complex second mode. It does not
establish uniqueness for an unscanned HAPRE experiment. Its supplementary
material could not be retrieved; no SI-specific algorithm change is made.
[Thibault and Menzel](https://doi.org/10.1038/nature11806) likewise concern
mixed-state ptychography. [Li et al.](https://doi.org/10.1364/OE.24.009038)
discuss mixed-state ambiguities and additional constraints.
The correct DOI for Ferrand/Mitov, Optics Letters 48, 5081 is
[10.1364/OL.498655](https://doi.org/10.1364/OL.498655); their vectorial
measurement ambiguities should not be treated as a HAPRE convergence result.

For fields `C+M` and `C-M`, the sum is `2|C|²+2|M|²`: the charge–magnetic
cross term cancels. Additional energies can constrain shared physical
parameters, but merely declaring two modes to be helicities does not add
measured information. A constant unitary rotation of the two fields leaves
the sum unchanged. Neither this ambiguity nor cancellation proves that all
spatial magnetic information is lost; additional constraints must actually
resolve the remaining freedom.

Notebook 07 is an isolated control, not a replacement solver. It uses the
05 loader, independent one-mode helicity reconstructions and blind two-mode
summed-data retrieval. Coherent reference fields enter only a post-hoc
constant-unitary Procrustes fit. It reports complex error, amplitude error,
phase RMS above a 1% amplitude threshold, principal angles, rank and modal
power fractions. Separate support and material fits are explicitly labeled.
No rescaling or spatial gauge correction is hidden in the comparison.
The reported matrix can vary between independently solved energies; this
does not demonstrate a common physical basis across the spectrum.
A low data residual or dominant-mode correlation is insufficient evidence
for helicity separation. Coherent-reference ambiguity and noise also limit
what failure of this benchmark can establish.

For row-stacked fields X (blind) and Y (reference), the implementation takes
`Y Xᴴ = L Σ Vᴴ` and uses `U = L Vᴴ`, minimizing `||U X - Y||F` on the
chosen pixels. The same U acts on every pixel. Singular values of the
row-space overlap give principal angles; the modal singular-value powers
are reported separately because identical spans need not have identical
powers and need not be related by a unitary matrix.

Control seed selection uses intensity error **after applying object support**
to the returned Fourier-constrained solution. With `Fourier_last=True`,
raw final detector error is approximately zero by construction and is not
a useful selection score. The existing 05/06 final-detector history has
this same limitation; compare it with physical-stage error and held-out
measurements rather than treating it as evidence of convergence.

### Reduced-resolution check

`modal_subspace_smoke_result.json` records energy index 5 (775.674 eV),
194×194 pixels, binning 8, 750 HAPRE + 50 ER iterations and two seeds.
Support-projected intensity errors were 0.472–0.473 for the blind sum and
0.477–0.496 for coherent references. Unitary-aligned complex errors were
1.13–1.19. This run did not recover matching modal fields, and the poor
coherent baselines prevent interpreting it as a test of uniqueness.
Support/preprocessing consistency and coherent convergence need investigation
before claiming a physical separation failure or success. This check is
not the full-resolution, jointly constrained multi-energy experiment.

## Hyperspectral initialization experiment

The supplied PRL supplement has now been read (sections II–IV). Notebook 05
uses a positive-real charge seed and a complex magnetic seed, shared across
energies. A unitary sum/difference transform produces the initial helicity
candidates before the existing warmup and alternating joint physical
projection / phase retrieval. The magnetic seed is zero outside the material.
Its initial norm is 0.1 times the charge norm; this is an amplitude ratio,
not a measured or enforced magnetic fraction. Subsequent fields remain complex.
The optional third mode retains its separate support and starts weakly.
`INITIALIZATION_BASIS='independent_helicity'` restores the previous control.

The joint physical model already provides the spectral redundancy: one real
magnetization map and common spatial transmission, with scalar complex charge
and magnetic responses at each energy. No individual helicity measurement
enters initialization or physical fitting. Merely adding energies without
this shared spatial model would not impose this coupling. Fitted spectra,
global sign ambiguity and absent scan overlap still limit identifiability.
`support_projected_history` is now exported alongside detector histories.

The 12-energy reduced-resolution comparison is recorded in
`hyperspectral_initialization_smoke_result.json` (194×194, one seed,
30 HAPRE + 10 ER warmup, five physical/10-ER cycles). Switching from
independent helicity to shared charge/magnetic initialization reduced the
held-out global-sign-adjusted difference error from 5.27 to 3.54. Both fail
the zero-difference baseline of 1. Support-projected errors remained about
0.50. This is an initialization experiment, not established separation;
the default longer full-resolution schedule has not been validated here.
