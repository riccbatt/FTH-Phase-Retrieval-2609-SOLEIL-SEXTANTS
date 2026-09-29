# MAX IV phase-retrieval sandbox

This folder is a temporary source snapshot. The original repository's Git
history, SOLEIL processed data, and old notebooks were excluded from the copy.
All new outputs are under this folder's `processed/` directory.

The test pair is averaged CMOS scans 1143 (+) and 1145 (-), with dark scans
1142/1144 and 1144/1146. These IDs and the CMOS path come from
`Quantitative_imaging_2ndharm_20260909.ipynb`. The data remain on the MAX IV
drive; the sandbox only saves centered, dark-corrected copies.

1. Run `00_prepare_maxiv.py` with the `myenv` Python environment. It loads and
   centers the two images and saves `processed/Logs/data_recon_ImId_1143_maxiv.hdf5`.
2. Open `03_maxiv_supportmask.ipynb`, inspect the amplitude overlay, adjust the
   `(row, column, radius)` circles, and run its save cell.
3. Open `04_maxiv_phase_retrieval.ipynb` and run its cells. Its 30/10/10 recipe
   is a short functional check on the full-resolution images; increase the
   iteration counts after the support mask is reviewed.

The initial sandbox run completed on both helicities. The returned amplitudes
match the measured amplitudes on unmasked detector pixels to about 5e-8
relative RMS. This confirms the Fourier constraint is aligned and executing;
the default Startimage/support transform round-trips within 7e-16 maximum
error. In the short run, 0.29% (+) and 0.61% (-) of object energy remains outside
the seeded support. These checks do not establish a final-quality image; the
support coordinates and longer recipe still need review.

The old notebook's visible two-mode example calls retrieval with
`partial_coherence=False` and plots the full-coherence mode. The sandbox
retrieval notebook now has `RETRIEVED_KIND` so the same result can be inspected
as full or partial coherence. With a reduced, two-mode test on the current tight
support mask, the RL continuation shows more ring structure than the full-only
result; see `processed/full_vs_partial_1143.png`. Its diagnostic compares the
measured amplitude with the convolved sum of modal intensities, which is the
appropriate quantity when `RL_its > 0`.

`06_maxiv_summed_helicities_saturated_reference.ipynb` tests separation of
summed helicities with two states at about 776.08 eV. State 0 uses +1136/-1134
and darks 1133/1135/1137 (25 frames, 0.6 s each); state 1 uses +1141/-1139
and darks 1138/1140/1142 (25 frames, 1.2 s each). The notebook verifies those
acquisition settings from the raw metadata. Averaged caches are divided by
exposure only; `CACHE_REDUCTION="sum"` handles summed-frame caches explicitly.
Both factor-1 modes enter the physical helicity fit. An optional factor-2
mode supplies common nonmagnetic background. State 0 is fixed to uniform
magnetization -1 inside the material aperture; it is not assumed to have zero
magnetic transmission. Set `USE_SATURATION_CONSTRAINT=False` for a control
with the same measured sums. Separate helicities and their difference are
used only for validation. Outputs go to
`processed/Logs/data_phase_summed_helicities_saturated_maxiv.hdf5`.

The summed-helicity notebooks 05 (12 energies) and 06 (two states) use the
existing alternating projection framework: initial HAPRE/ER warmup, then
joint physical projection of the two helicity modes followed by a block of
phase retrieval against each measured sum, repeated for `OUTER_ITERATIONS`.
`INNER_ITERATIONS` controls the spacing between physical projections and
`PROJECTION_RELAXATION` their strength. Default warmup is HAPRE 100 + ER 30;
30 physical/ER cycles follow with 10 ER iterations per cycle. These are
configurable schedules, not a convergence guarantee. The 05 loader prepares
one energy pair at a time to reduce loading memory.

Both notebooks report summed-data errors after each physical projection and
after the following phase retrieval. The final fields are after phase retrieval;
saved component maps describe the preceding physical fit. Held-out helicity
differences are compared to the no-separation error baseline of 1. The direct
L-BFGS optimizer experiment has been removed from this workflow. See
[summed-helicity workflow notes](summed_helicity_notes.md).

`07_maxiv_modal_subspace_validation.ipynb` checks whether blind two-mode
HAPRE recovers the same modal subspace as separate coherent +/− retrievals.
It reuses 05 preprocessing and reports constant-unitary alignment, ranks,
modal power fractions and complex/amplitude/phase errors. Defaults test
energy index 5 with two seeds and 750 HAPRE + 50 ER iterations. Expand
`VALIDATION_INDICES` for the full spectrum. This is a held-out diagnostic;
it does not change the alternating physical projections in 05/06.
