# Permanent-magnet workflow

Run these four notebooks in order, from this folder or the repository root:

1. `01_average_raw_cmos.ipynb`: one frame-weighted, dark-corrected, photon-filtered average per scan index. Each index includes every Marana `.nx` file and every frame in its scan folder.
2. `02_align_holograms.ipynb`: masked cross-correlation with local subpixel refinement, followed by optional centering and edge cropping. Save holograms and conservatively shifted exclusion masks.
3. `03_warmup_phase_retrieval.ipynb`: independent HAPRE/ER warmup only; save fields together with their exact holograms, masks, support, material region, modes, and metadata.
4. `04_physical_projection.ipynb`: load the saved warmup, then run ER/physical rounds without repeating warmup. Set `OUTER_ITERATIONS = 0` for a final physical projection alone.

Edit the variables near the top of each notebook. Scan indices, energies, polarizations, and saturation signs are parallel arrays in acquisition order. Saturation anchors are optional; without opposite saturated endpoints, magnetization is relative contrast. Set `RAW_ROOT` in notebook 1 and `PROCESSED` at the start of each notebook to your external data directories. All stages save HDF5 under your selected `PROCESSED` folder, with diagnostic plots shown inline. Saved files are overwritten when that stage is rerun.

Supply `masks/mask_pixel.png` on the **raw detector grid** (red means excluded) and `masks/supportmask.png` on the **final centered object grid after notebook 2's crop** (red means allowed). Paint the support's object and reference holes as separate components. Its largest connected component defines material; inspect the displayed preview before retrieval. PNG sizes are checked, never silently resized. Nonred pixels are false; transparent pixels are false.

The photon threshold is in dark-corrected ADU. As in ajajas, filtering zeros each frame's values at or below the threshold before averaging; retained values remain analog, not integer photon counts. Dark and sample exposure/gain settings must match. Nonfinite raw pixels and values at or above saturation are replaced by zero per frame before averaging. No persistent invalid-pixel masks are produced. Notebook 2 uses the painted detector mask.

Alignment shifts are corrective `(row, column)` shifts in raw pixels. Interpolation does not wrap. Fractional shifts conservatively mask neighbouring invalid pixels and exposed edges. No flux normalization is applied during preprocessing; relative illumination can be fitted during the physical stage.

Keep retrieval on the prepared grid: crop=0, binning=1 in notebooks 3 and 4. Modal support centering is identical in both stages, and an assertion verifies the physical stage receives the exact saved warmup fields. Physical `components/` retain library fitting coordinates; the top-level `magnetization` dataset is centered for plotting. Dependencies are the repository's existing NumPy, SciPy, h5py, matplotlib, Pillow, and scikit-image packages.

Validation: `python -m unittest discover -s tests -p 'test_permanent_magnets_workflow.py'` runs the four actual notebooks on synthetic raw scans and checks filtering, frame weighting, alignment, HDF5 handoffs, and the physical start fields.

The alignment algorithm is self-contained in `workflow.py`. It uses only the images, masks, reference index, ROI, and shift limits supplied in notebook 2; it has no dependency on a beamtime-specific folder or dataset.

Notebook 2 requires a user-selected `ALIGNMENT_ROI = (row_start, row_stop, column_start, column_stop)` on the raw detector grid. Inspect the full holograms first, then the two selected holograms and their difference inside that region. Choose `DIAGNOSTIC_INDICES` for the pair comparison. After shifting, the notebook plots the pair difference and its FTH reconstruction, and saves these diagnostics in the alignment HDF5 file.
