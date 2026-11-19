# SEXTANTS hyperspectral physical-projection workflow

Open **05_hyperspectral_physical.ipynb** from the repository root on the analysis PC.
Its libraries are local: no sibling checkout, VS Code, internet access, or GPU is required.
Python 3.10+ and the packages in `requirements-hyperspectral.txt` are sufficient for the new notebook.
CuPy is optional and must match the analysis PC's CUDA installation if used.

The configured scan is the active series from `01_FTH_series`: 20 positive/negative pairs,
positive IDs 1797–1835, negative IDs 1798–1836, and nominal energies 775–796 eV.
The raw folder remains `/nfs/ruche/sextants-soleil/com-sextants/COMET_20261110_Battistelli_laser_RF/`.
No measured data were available for local validation.

## Independent inputs and settings

No output from 01_FTH, 01_FTH_series, 04_phase_retrieval or any other notebook is
required. Do not run those notebooks first. The notebook defines and loads everything:

1. The raw NeXus energy/helicity pairs and dark IDs. The default dark IDs are 1468;
   verify them for this camera/beamstop session. Scalar and same-energy list inputs work.
2. `processed/mask_pixels/mask_beamstop_1795.png` by default. No fixed detector PNG is loaded.
3. `DETECTOR_CENTER`, `PIXEL_SIZE_M`, `FOCUS`, and `DARK_FIT_RECIPE` are explicit notebook
   settings. The starting center is `(962,894)` and focus is 28.55 um, -0.56 rad,
   dx=-0.24, dy=0. These are starting values to inspect on the raw scan, not an
   automatically validated focus or a dependency on an external calibration.
4. Reference energy and detector distance come directly from the first positive
   NeXus file unless overridden with `REFERENCE_ENERGY_EV` / `REFERENCE_DISTANCE_M`.
5. Support is drawn here from `(row,column,radius)` coordinates: `(1024,1024,1)`
   and `(789,1248,76)` initially. Alternatively set `SUPPORT_PNG` to a manually
   prepared inclusion mask; red pixels are included. No support HDF5 is needed.

The raw-detector preview overlays the selected center and beamstop. The subsequent
FTH-style first-pair preview overlays this notebook's support and uses its propagation,
phase, dx and dy so those settings can be checked before a long joint reconstruction.

By default `FIT_MATERIAL_THICKNESS=True` and `MATERIAL_REGION="support"` initialize
positive thickness over the complete support, including reference apertures. The fit
can drive thickness toward zero within that allowed region. Outside the support,
thickness stays zero. Nothing prescribes which supported aperture is material.
`MATERIAL_REGION="largest_aperture"` instead supplies a stronger material-location prior.
An initial thickness map may be supplied; initially zero pixels cannot become material,
so keep it positive wherever material occupancy should be inferred.

`PAIRS` is the full acquisition table. Each `plus`/`minus` accepts one ID or a list
of same-energy IDs. Different-energy averages and inconsistent raw energy/polarization
metadata are rejected. The nominal energies are checked against raw metadata.
The plus/minus assignment is explicit: verify the physical helicity signs on the analysis PC.

## Measurement and coordinate handling

Raw NeXus frame stacks are reduced frame-by-frame rather than loaded wholesale.
The loader fits and subtracts the average dark, centers the image and binary beamstop
mask together, optionally divides by mean exposure, and resamples each energy/distance
onto the common reciprocal-space grid defined by the raw reference file and notebook settings. Interpolation footprints crossing
missing data remain excluded, and pixel-flux Jacobians are applied. No beamstop intensity
is invented. The physical model uses per-observation masks. Intensities reach the engine
without an extra square or square root; the engine prepares amplitudes.

The common-q correction is a small-angle planar-detector approximation. It does not
replace Ewald-sphere correction, drift alignment, or remeasuring the center after
moving the detector. Set a per-pair `center` when required. The input is raw planar
detector intensity, not previously Ewald-projected reconstructions.

Exposure normalization does not measure incoming beam flux. Supply a positive
`incident_intensity` monitor value per pair (or a plus/minus dictionary) to correct that
variation. Do not normalize every sample hologram by its total intensity: that would
remove absorption information. The physical engine's separate reference-gain correction
currently supports one energy, so it is disabled for this multi-energy dataset.

## Reconstruction

The included joint engine is copied from the neighboring
`FTH-Phase-Retrieval-2609-SOLEIL-SEXTANTS` library used by its MAXIV hyperspectral notebooks.
`library/hyperspectral_engine/SOURCE.json` records source checksums and local adaptations.
There are no runtime imports from the parent folder. This engine is isolated from the
existing single-pair retrieval libraries.

All observations share one sample state by default. Independent coherent HAPRE/ER
warmups precede joint energy/helicity sweeps and physical projection:

`log(object) = common_illumination + thickness * (charge(E) + helicity * magnetic(E) * magnetization)`

The default fits nonnegative relative thickness throughout the allowed support,
free complex spectra, and a relative magnetization map. This is not guaranteed unique
material segmentation: thickness, illumination and response scales can trade off. It does not
infer absolute optical constants without thickness, flux and magnetic-scale anchors.
Inspect `identifiable`, the fit residuals, and mode-dependent artifacts. The phase-reference
option resolves complex-log branches within each energy pair; it is not an absolute phase calibration.

Use `MODES`, `BINNING`, `CROP`, `PARTIAL_COHERENCE` and the visible recipe to set the workload.
Coherent warmup remains coherent when partial coherence is enabled. PC uses per-observation
kernels by default and keeps missing detector pixels free; no final coherent-amplitude overwrite
is applied to PC fields. The forward blur and physical focus are separate operations.
The local focus adaptation accounts for each energy's pixel angle after common-q resampling.

Spectral KK constraints are optional, not enabled automatically for this finite energy window.
For KK, energy coordinates must be strictly increasing and match acquisition-label order.
Support geometry must retain the sample and reference apertures after crop/binning.
A 40-observation, two-mode, 2048-square reconstruction needs several large complex stacks;
inspect the reported input size and choose a feasible retrieval grid on the analysis PC.

## Results and validation

The first positive ID identifies every output:

- `processed/Logs/hyperspectral_prepared_ImId_1797_rb.hdf5`
- `processed/Logs/hyperspectral_physical_ImId_1797_rb.hdf5`
- `processed/hyperspectral_physical_ImId_1797_rb.png`

The result saves complex fields, warmup fields, charge/magnetic spectra, relative
magnetization, effective masks/support, physical components, recipe, errors, raw/dark
ID lists, energies, exposures and all preprocessing choices. Existing files are protected
unless `OVERWRITE=True`. No single-pair FTH result is overwritten.

Run the new workflow checks with:

```sh
python -m unittest discover -s tests -p test_sextants_hyperspectral.py -v
```

Fifteen tests cover actual synthetic NeXus/PNG I/O, same-energy averaging, energy rejection,
common-q masks/Jacobian, physical projection of known log-objects, coherent/PC and modal
retrieval, focus inversion including dx/dy, portable HDF5 saves, and all notebook cells through plotting/export.
The notebook test starts without any previous result HDF5 and rejects every attempted
`load_data_dict` call during execution, proving no other notebook output is needed.
They establish the data path and numerical contracts, not convergence of the unavailable measured scan.
Two older single-pair notebook tests currently fail against the separately pasted 01/04 revisions;
they are not part of this new workflow's passing checks.


## Complete recipe controls at the beginning

The opening settings cell contains a literal `RECIPE_PARAMETERS` dictionary with
all **160** parameters accepted by the included engine. It is grouped into schedules,
coherence, execution, numerical updates, warmup overrides, input geometry, modes,
physical constraints, thickness/material, chemical species, spectra/KK, and independent
optical anchors. No subsequent notebook cell overwrites these choices or inserts
unlisted defaults. Convenience controls such as `FOCUS`, `MODES`, `BINNING`, and
`FIT_MATERIAL_THICKNESS` appear above the dictionary and are referenced explicitly.
The raw intensity cube's storage dtype is also exposed as `RAW_DTYPE`.

Four parameters can explicitly use `AUTO` (`auto_from_dataset`):

- `projection_every`: number of observations, meaning one complete sweep.
- `energy_values`: unique nominal energies in acquisition order.
- `material_mask`: the selected `MATERIAL_REGION`, or supply an explicit array instead.
- `projection_focus_setup`: raw geometry with pixel size multiplied by recipe binning
  and dx/dy divided by binning, preserving their source-grid meaning.

These are resolved immediately after raw loading; the FTH preview uses the same
resolved focus controls. `None` retains the engine's documented automatic/inherited
meaning. For example, `warmup_beta_zero=None` inherits the shared beta setting,
whereas `projection_start=None` selects the first projection cadence boundary.

A coverage check rejects missing or unknown recipe keys. If the engine adds a new
parameter, the notebook must expose it before running; it cannot silently gain a new
hidden default. Tests check literal key coverage, edits reaching the resolved recipe,
absence of later overrides, AUTO restrictions, and end-to-end execution.

Not every setting is active for every model. The alternate projection-family controls
are labeled inactive with `physical_factorized`. The reference-gain controls remain
single-energy-only and must stay disabled here. Advanced chemical-species settings may
require species-aware result analysis in place of the notebook's single-species plots.
The engine validates combinations; backend-specific restrictions still apply, including serialized observation updates on GPU.
