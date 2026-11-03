# Two datasets, one reconstruction structure

Both `01_reconstruct.ipynb` files have the same eight pairs of markdown/code cells: setup and input, validate, model settings, independent warmup, physical continuation, detector/map diagnostics, paper observables, and export. To adapt a dataset, change preparation and the input metadata, then replace the observable extraction cell. Keep the reconstruction stages shared.

## Hyperspectral imaging

Preparation reads the scan pair table, verifies energy and field metadata, averages all specified dark averages, subtracts darks, divides by 500 frames times exposure, centers images, optionally registers energy, and propagates invalid pixels. **500 frames is inherited from the source workflow; verify it against actual acquisition metadata before quantitative analysis.** Averaged files cannot reveal saturation in an individual frame. For a final analysis obtain per-frame saturation flags or establish that this limitation is negligible.

The measured support and mask are included (44 KB), extracted from the source reconstruction without fields or other saved products. The support is an experimental input: `data/hyperspectral/support.h5` must contain 2D `supportmask` and `mask_pixel`. It is not a generated illustrative mask. If using the original saved support, copy these datasets from `maxiv_phase_test/processed/Logs/data_recon_ImId_1143_maxiv.hdf5`. Recheck centering and support after registration and crop/bin mapping.

The reconstruction treats pairs as opposite polarization at one state and multiple energies. Helicity assignment comes from the explicit scan table and must be checked against the beamline log. Enter measured magnetic-material thickness to convert log responses to optical constants; report unresolved gauges. The first preparation cell contains the experimental paths and paired scans.

## 2601_XPCS hysteresis

Preparation preserves acquisition order and repeated currents. It streams frame means, flags saturation in any frame, verifies matched dark exposures, centers, aligns, and writes holograms, exclusion masks, support, material, currents and scan/point labels to the cache. Geometry controls and wire vertices are explicit in the notebook. The adapter holds reusable streaming and alignment algorithms, without plot wrappers.

The reconstruction selects acquisition indices and accepts verified `(scan, point): +/-1` saturation anchors. The source working notebook used points 10 and 34; this export leaves the decision explicit rather than asserting their physical saturation. ROI means of fitted magnetization are shown against current. If both signs are anchored, a second observable uses the same warmup-derived complex contrast direction for both stages and endpoint-normalizes each stage. This is a comparison observable, not an independent absolute calibration or uncertainty estimate.

A smoother loop alone is not proof of quantitative improvement. Compare detector residuals, independent reference states, local stability across justified ROIs and uncertainty under repeat acquisitions. Save the analysis ROI and all calibration choices for final paper figures.

## Data and outputs

Raw paths are local settings. Prepared caches live under `data/<dataset>/`; output files live under `results/`. The HDF5 exports include source metadata, prepared measurements, masks, warmup/final fields, both recipes, physical components, errors and derived observables. Each dataset's final figure is also exported to PDF. For the paper archive additionally freeze raw-data checksums, the measured calibration sources and the software environment.
