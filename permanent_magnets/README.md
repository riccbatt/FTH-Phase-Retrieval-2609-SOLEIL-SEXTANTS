# Permanent-magnet workflow

Run these four notebooks in order, from this folder or the repository root:

1. `01_average_raw_cmos.ipynb`: one frame-weighted, dark-corrected, photon-filtered average per scan index. Each index includes every Marana `.nx` file and every frame in its scan folder.
2. `02_align_holograms.ipynb`: masked cross-correlation with local subpixel refinement, followed by optional centering and edge cropping. Save holograms and conservatively shifted exclusion masks.
3. `03_warmup_phase_retrieval.ipynb`: independent HAPRE/ER warmup only; save fields together with their exact holograms, masks, support, material region, modes, and metadata.
4. `04_physical_projection.ipynb`: load the saved warmup, then run ER/physical rounds without repeating warmup. Set `OUTER_ITERATIONS = 0` for a final physical projection alone.

Edit the variables near the top of each notebook. Scan indices, energies, polarizations, and saturation signs are parallel arrays in acquisition order. Saturation anchors are optional; without opposite saturated endpoints, magnetization is relative contrast. Set `RAW_ROOT` in notebook 1 and `PROCESSED` at the start of each notebook to your external data directories. All stages save HDF5 under your selected `PROCESSED` folder, with diagnostic plots shown inline. Saved files are overwritten when that stage is rerun.

Supply `masks/mask_pixel.png` on the **raw detector grid** (red means excluded) and `masks/supportmask.png` on the **final centered object grid after notebook 2's crop** (red means allowed). Paint the support's object and reference holes as separate components. Its largest connected component defines material; inspect the displayed preview before retrieval. PNG sizes are checked, never silently resized. Nonred pixels are false; transparent pixels are false.

The photon threshold is in dark-corrected ADU. As in ajajas, filtering zeros each frame's values at or below the threshold before averaging; retained values remain analog, not integer photon counts. Dark and sample exposure/gain settings must match. Nonfinite and saturated raw samples are ignored using a streaming nanmean. By default, only pixels with no valid samples remain NaN and are added to notebook 2's painted detector mask. `INVALID_PIXEL_POLICY = "any_frame"` instead invalidates a pixel if any sample is bad. Photon-thresholded samples are valid zeros and count toward the average. Per-pixel valid counts are saved in HDF5.

Alignment shifts are corrective `(row, column)` shifts in raw pixels. Interpolation does not wrap. Fractional shifts conservatively mask neighbouring invalid pixels and exposed edges. No flux normalization is applied during preprocessing; relative illumination can be fitted during the physical stage.

Keep retrieval on the prepared grid: crop=0, binning=1 in notebooks 3 and 4. Modal support centering is identical in both stages, and an assertion verifies the physical stage receives the exact saved warmup fields. Physical `components/` retain library fitting coordinates; the top-level `magnetization` dataset is centered for plotting. Dependencies are the repository's existing NumPy, SciPy, h5py, matplotlib, Pillow, and scikit-image packages.

Validation: `python -m unittest discover -s tests -p 'test_permanent_magnets_workflow.py'` runs the four actual notebooks on synthetic raw scans and checks filtering, frame weighting, alignment, HDF5 handoffs, and the physical start fields.

The alignment algorithm is self-contained in `workflow.py`. It uses only the images, masks, reference index, ROI, and shift limits supplied in notebook 2; it has no dependency on a beamtime-specific folder or dataset.

Notebook 2 requires a user-selected `ALIGNMENT_ROI = (row_start, row_stop, column_start, column_stop)` on the raw detector grid. Inspect the full holograms first, then the two selected holograms and their difference inside that region. Choose `DIAGNOSTIC_INDICES` for the pair comparison. After shifting, the notebook plots the pair difference and its FTH reconstruction, and saves these diagnostics in the alignment HDF5 file.

Notebooks 3 and 4 have separate **Plot saved results** cells that reload HDF5 without rerunning reconstruction. Set `DISPLAY_ROI` for an object-plane zoom; differences follow the saved XMCD pairs or reference index. Every mode has amplitude/phase plots with matched display scales across observations. Notebook 4 additionally shows fitted magnetization, mean magnetization, illumination, and warmup/physical amplitude and phase for `COMPARE_INDEX`. Independent warmups can carry arbitrary global phases, so their unaligned complex differences should be interpreted accordingly.

Define `STATE_LABELS` beside `POLARIZATIONS` in notebook 1. Observations of the same magnetic state have the same label; opposite helicities use +1 and -1. Labels are saved and carried through all stages. Notebook 3 only reconstructs independent observations; notebook 4 groups them using their saved state labels. Plotting automatically pairs opposite helicities within each state. With one polarization, choose `REFERENCE_INDEX` in each results-plotting cell. No separate DATA_MODE or XMCD_PAIRS settings are needed. Both notebooks show absolute and real complex differences; notebook 4 also shows magnetization per state.

Notebook 3 offers an optional ipywidgets focus preview. Set `ENABLE_FOCUS_SLIDERS = True`, enter measured detector pixel size and distance, and choose `FOCUS_SET_INDEX` and `FOCUS_MODE_INDEX`. Propagation and phase sliders preview one difference. **Save focus for all sets** records settings in warmup HDF5; rerun difference plots to apply them everywhere. Notebook 4 reads the saved focus for physical fitting and plotting. Saved detector fields are never overwritten by focusing.

Notebook 4 exposes `MAGNETIZATION_TREND_INTENSITY_CUTOFF` (default 0.01), `MAGNETIZATION_TREND_CORRECTION` (`"offset"`, `"factor"`, or `"affine_domains"`), and `BOUND_MAGNETIZATION_MEDIAN` (default False). Increasing/decreasing trends allow equal neighbouring weighted means. Median bounds also work without a trend. These controls concern physical fitting and are therefore absent from the warmup notebook. See `docs/universal_phase_retrieval.md` for correction fallback and distribution assumptions.
