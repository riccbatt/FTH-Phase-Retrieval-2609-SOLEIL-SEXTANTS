# Two datasets, one reconstruction structure

Both `01_reconstruct.ipynb` files have the same eight pairs of markdown/code cells: setup and input, validate, model settings, reconstruction metadata, joint reconstruction, detector/map diagnostics, paper observables, and export. To adapt a dataset, change preparation and the input metadata, then replace the observable extraction cell. Keep the initialization call and single main-loop call shared.

## Hyperspectral imaging

Preparation reads the scan pair table, verifies energy and field metadata, averages all specified dark averages, subtracts darks, divides by 500 frames times exposure, centers images, optionally registers energy, and propagates invalid pixels. **500 frames is inherited from the source workflow; verify it against actual acquisition metadata before quantitative analysis.** Averaged files cannot reveal saturation in an individual frame. For a final analysis obtain per-frame saturation flags or establish that this limitation is negligible.

Painted `supportmask.png` and `mask_pixel.png` files are read from `data/hyperspectral/masks/`. Detector dimensions come from the measured averages. An optional support HDF5 file with `supportmask` and `mask_pixel` can initialize missing PNGs. Set `DATA_ROOT` to the experiment folder containing `raw/` and `proc/`; preparation writes `prepared.h5`, including acquisition records and geometry settings. Recheck centering and support after registration and crop/bin mapping.

The reconstruction treats pairs as opposite polarization at one state and multiple energies. Helicity assignment comes from the explicit scan table and must be checked against the beamline log. Enter measured magnetic-material thickness to convert log responses to optical constants; report unresolved gauges. The first preparation cell contains the experimental paths and paired scans.

## 2601_XPCS hysteresis

Preparation preserves acquisition order and repeated currents. It reads painted `supportmask.png` and `mask_pixel.png` files from `data/2601_XPCS/masks/`, streams frame means, propagates saturation from any frame, verifies matched dark exposures, and optionally aligns holograms. It writes holograms, exclusion masks, support, material, currents and state labels to `prepared.h5`. The largest support aperture defines material; smaller apertures provide vacuum references. Crop/bin settings are stored in the cache and applied once during reconstruction. Raw scan layout is `<scan:05d>.nxs` with detector streams under `<scan:05d>/marana/`. Averaged HDF5 files and preview PNGs are also written.

The reconstruction selects acquisition indices and accepts verified `(scan, point): +/-1` saturation anchors. The source working notebook used points 10 and 34; this export leaves the decision explicit rather than asserting their physical saturation. ROI means of fitted magnetization are shown against current. If both signs are anchored, the final exit-wave contrast is also endpoint-normalized on the same ROI. This is derived from the same fit, not an independent absolute calibration or uncertainty estimate.

A smoother loop alone is not proof of quantitative improvement. Compare detector residuals, independent reference states, local stability across justified ROIs and uncertainty under repeat acquisitions. Save the analysis ROI and all calibration choices for final paper figures.

## Data and outputs

Raw paths are local settings. Prepared caches live under `data/<dataset>/`; output files live under `results/`. The HDF5 exports include source metadata, prepared measurements, masks, initial/final fields, the recipe, physical components, errors and derived observables. Each dataset's final figure is also exported to PDF. For the paper archive additionally freeze raw-data checksums, the measured calibration sources and the software environment.
