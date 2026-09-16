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
