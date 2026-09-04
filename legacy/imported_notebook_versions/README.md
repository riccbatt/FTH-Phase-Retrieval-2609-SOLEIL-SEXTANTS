# FTH-Phase-Retrieval-2609-SOLEIL-SEXTANTS

This folder contains the notebook sequence for going from raw BESSY/P04/SEXTANTS holograms to an FTH reconstruction, masks, support, and phase-retrieved CDI reconstruction. The numbered reconstruction notebooks share one HDF5 data dictionary in `processed/Logs/`.

All notebooks use the directory in which they are currently running as
`BASEFOLDER`. Raw SEXTANTS data are loaded from:

```python
RAW_FOLDER = "/home/experiences/sextants/com-sextants/ruche/sextants-soleil/com-sextants/COMET_20260902_Cocoons_Laser/"
RAW_FOLDER = "../COMET_20260902_Cocoons_Laser_raw/"
```

Every generated result is written below `BASEFOLDER/processed/`.

FTH and phase-retrieval result figures are written directly to `processed/`.
Paintable detector masks live in `processed/mask_pixels/`, and support masks
live in `processed/supportmask/`.

`Diode_Scans.ipynb` handles one or several diode scans, including energy
and magnetic-field scans. It discovers NeXus channels by name or metadata,
supports matrix-shaped continuous field traces by flattening them in acquisition
order, and saves both the numerical HDF5 result and plotted PNG under
`processed/diode_scans/`.

## Quick Start

Use `00_inspect_nexus_metadata.ipynb` whenever you need to browse the complete
NeXus tree, search dataset paths and attributes, or compare scalar metadata
across scans. It avoids loading large detector arrays unless explicitly enabled.

1. Optionally define raw-coordinate detector masks before FTH using either `00_define_mask_pixel_paint.ipynb` (PNG/Paint) or `00a_define_mask_pixel_napari.ipynb` (interactive Napari). Create one mask per raw image used directly or in stitching.
2. Optionally run `00b_stitch_beamstops.ipynb` to combine images acquired with different beamstops/exposures. Its output mask is the intersection of the aligned input masks, so only pixels covered in every input remain masked.
3. Open `01_FTH.ipynb`.
4. Set `USER`, the raw image IDs, and the `+`/`-` acquisition modes. For each input, `mask_id` selects a painted `mask_pixel_<image-id>.png` (or legacy `.npy`); `mask_id=None` uses no precise mask. Likewise, `dark_id=None` disables dark subtraction. Set `stitched_file` to a `00b` output to load a stitched image instead of the raw image.
5. Run the notebook through the final save cell.
6. Open `03_define_supportmask.ipynb` and define the support using one of three alternatives: Paint, Napari, or `(y, x, radius)` support coordinates.
7. Open `04_phase_retrieval.ipynb`, adjust the phase retrieval recipe if needed, run phase retrieval, focus the CDI reconstruction, and save.

### Image series and hysteresis loops

After calibrating one representative `+`/`-` pair with the original notebooks,
use `01_FTH_series.ipynb` to process parallel `PLUS_IMAGE_IDS` and
`MINUS_IMAGE_IDS` lists, with an optional `SERIES_POINTS` list for field or time
values. It reuses the calibrated center, pixel and smooth masks, Ewald setting, FTH focus, and ROI,
and checkpoints after every pair to
`processed/Logs/data_recon_series_<user>.hdf5`.

The current hysteresis calibration uses the center `[1005, 1035]`, measured
from image `13`, and the ROI `[597, 785, 526, 719]`. Rerun the original FTH and
support-mask notebooks after changing either value so all saved masks use the
same coordinate system.

The detector pixel size is `11.0e-6 m`. The sample-to-detector distance is not
hard-coded: `SextantsNexusLoader` reads the `ccd-ts` stage position from
`scan_data/data_03` in millimetres and calculates
`ccd_dist = (700 - data_03) / 1000`. Images 13 and 14 contain `data_03 = 200 mm`,
which gives `ccd_dist = 0.500 m`.

Then run `04_phase_retrieval_series.ipynb` to reuse the support mask,
phase-retrieval recipe, and CDI focus/ROI for every point. Its checkpoint is
`processed/Logs/data_phase_retrieval_series_<user>.hdf5`. Both notebooks save
one indexed PNG per series point. The FTH series reuses the calibration factor
and offset by default; set `REUSE_CALIBRATION_SCALING = False` to refit them for
each pair when detector intensity drifts.

Choose the calibration explicitly with `REFERENCE_IMAGE_ID` in
`01_FTH_series.ipynb`. The notebook loads the matching
`data_recon_ImId_####_<user>.hdf5` and uses its saved center, ROI, focus, masks,
and experimental setup. `CENTER_OVERRIDE` and `ROI_OVERRIDE` are available when
an intentional manual override is needed.

At the end of the FTH series, the notebook writes
`processed/FTH_series_<user>.gif`. Every frame uses the same intensity limits
across the full reconstruction stack; `GIF_FRAME_DURATION_MS` controls playback
speed. Plot and GIF labels show the paired IDs and progress, for example
`im_id=14,15 | 1/25`, instead of generic `point 0` labels.
Both series notebooks call `plt.close("all")` every `CLOSE_ALL_EVERY` pairs
(five by default) and once after their final overview to limit figure memory.

## Raw data library and optional preprocessing

`library/data_loading.py` defines a common `Frame` result and loader registry. `SextantsNexusLoader` supports SOLEIL filenames such as `scanx_0589.nxs`, discovers the detector dataset from its NeXus image metadata (the numbered `data_21`/`data_22` key varies by detector and scan), reads `scan_data/integration_times`, and records energy, polarization, and detector provenance. `NexusLoader` remains available for configurable HDF5/NeXus layouts, while `SpeLoader` adapts the existing SPE reader by dependency injection.

`library/mask_store.py` associates a raw-coordinate binary `mask_pixel` with an image ID. In every mask, `1` means unusable. It reads bright-red mask pixels from PNG and falls back to legacy NPY masks. It is deliberately centered only inside notebook 01, so changing the center does not invalidate the saved mask. `library/beamstop_stitching.py` normalizes by exposure, optionally registers and fits each input to the longest-exposure data, averages overlapping valid pixels at equal exposure, then fills gaps from progressively shorter exposures. Its output records the contributing count and exposure at every pixel.

The current notebooks are configured for:

```python
USER = "rb"
DATA_H5 = "processed/Logs/data_recon_ImId_0013_rb.hdf5"
```

Change these near the top of each notebook when switching to a new dataset.

## Shared HDF5 File

The HDF5 file is the handoff between notebooks. Notebook 01 creates it; later notebooks load it, add results, and save it back.

Important groups/keys:

- `experimental_setup`: geometry and detector metadata used by FTH/CDI propagation.
- `holo`: raw and processed holograms, keyed by the `+` and `-` acquisition modes.
- `center`: selected detector center.
- `mask_pixel`: centered detector mask for bad/saturated pixels and beamstop-like exclusions.
- `mask_pixel_smooth_recipe`: Butterworth smooth mask settings from notebook 01.
- `roi`: FTH reconstruction ROI.
- `focus`: FTH focus settings and ROI.
- `supportmask`: real-space support used by phase retrieval.
- `focus_cdi`: CDI focus settings, CDI ROI, selected mode, and retrieval metadata.
- `phase_retrieval_recipe`: exact recipe used for the last phase retrieval run.
- `phase_retrieval_errors`: per-step diffraction/support error traces.
- `recon_cdi`: final plotted CDI reconstruction.

## Notebook 01: FTH

Use `01_FTH.ipynb` to define paths, raw scan IDs, polarization labels, detector center, FTH mask recipe, FTH ROI, and FTH focus.

Main edits:

- `USER`: user suffix in output filenames.
- `RAW_DATA_KIND`: `"existing"` for the previous raw-data workflow or `"sextants_nexus"` for SOLEIL files.
- `hologram_inputs`: maps the `+` and `-` modes to raw image IDs and optional dark/mask inputs.
- `positive_label` and `reference_label`: labels used for the FTH difference hologram.
- `mask_pixel_smooth_recipe`: smooth Butterworth disk mask parameters.

Main outputs:

- `processed/Logs/data_recon_ImId_####_USER.hdf5`
- `processed/FTH_recon_ImId_####_USER.png`
- HDF5 keys for raw holograms, center, FTH reconstruction, `roi`, and `focus`.

The final save cell writes both the HDF5 file and PNG, so rerunning it always
keeps the numerical and viewable results together.

FTH and CDI focus propagation values are stored in micrometres, matching the
`focusCDI` slider label. The workflow converts micrometres to metres only when
calling the propagation kernel. Saved reconstructions also apply the selected
`dx` and `dy` sub-pixel shifts.

## Notebook 00/00a: Raw Pixel Mask

Before FTH, use either `00_define_mask_pixel_paint.ipynb` or
`00a_define_mask_pixel_napari.ipynb` to define precise bad-pixel and beamstop
masks in raw detector coordinates. They are alternative interfaces that produce
the same per-image PNG masks.

- `IMAGE_IDS` lists every raw image whose mask should be created.
- The Napari option supports `INITIAL_MASK_IDS` for loading compatible masks as templates.
- Create a separate mask for every frame that uses a different beamstop.

The canonical output is `processed/mask_pixels/mask_pixel_<image-id>.png`. Notebook 01 selects it with `mask_id`, centers it using the current center, and saves the centered mask into the reconstruction HDF5 file. Masked pixels are excluded from the FTH image and measured Fourier constraints during retrieval. See [PAINT_MASKS.md](PAINT_MASKS.md) for the Paint workflow and exact filenames.

## Notebook 03: Support Mask

Use `03_define_supportmask.ipynb` to create the real-space support for phase retrieval. It builds a preview reconstruction and lets you create or load a support mask.

Important steps:

- Build the support-mask preview reconstruction from the FTH result.
- Define `supportmask` by exactly one method: PNG/Paint, Napari labels, or `(y, x, radius)` support coordinates with the optional circle widget.
- Save the support mask into the shared HDF5 file.

Painted support masks are stored as `processed/supportmask/supportmask_<image-id>.png`; see [PAINT_MASKS.md](PAINT_MASKS.md).

Notebook 03 does not create `focus_cdi`. Notebook 04 uses `focus_cdi["roi"]` only if it already exists from a previous phase-retrieval save; otherwise `roi_cdi` defaults to the HDF5 FTH focus ROI, `focus["roi"]`.

## Notebook 04: Phase Retrieval

Use `04_phase_retrieval.ipynb` to run the unified recipe-driven phase retrieval and reconstruct the CDI image.

Main edits near the top:

- `USER`: output/user suffix.
- `DATA_H5`: shared HDF5 file to load.
- `pol1`, `pol2`: hologram labels used for phase retrieval and final CDI difference.

Main outputs:

- `processed/PhR_recon_ImId_####_USER.png`
- `data["holo"][label]["retrieved_full"]`
- `data["holo"][label]["retrieved_pc"]`
- `data["holo"][label]["retrieved_gradient"]`
- `data["holo"][label]["bsmask"]`
- `data["holo"][label]["gamma"]`
- `data["phase_retrieval_recipe"]`
- `data["phase_retrieval_errors"]`
- `data["focus_cdi"]`
- `data["recon_cdi"]`

The final phase-retrieval save cell writes both the updated HDF5 file and the
PNG reconstruction.

## Phase Retrieval Recipe

The recipe in notebook 04 is a dictionary named `phase_retrieval_recipe`. It is a step-by-step program for the phase retrieval engine. Each index of the list-valued entries defines one retrieval stage.

For example, with:

```python
"algorithm_list": ["HAPRE", "ER", "ER"],
"number_iterations": [700, 50, 50],
"helicity": ["+", "+", "-"],
```

the retrieval runs:

1. `HAPRE` on `+` for 700 iterations.
2. `ER` on `+` for 50 iterations.
3. `ER` on `-` for 50 iterations.

Many scalar values can be written once, and the code expands them internally to all steps. For clarity, step-specific values are usually written as lists.

### Common Recipe Keys

| Key | Meaning |
| --- | --- |
| `algorithm_list` | Algorithm for each stage. Allowed values include `ER`, `HAPRE`, `RAAR`, `HIO`, `HIOs`, `OSS`, `CHIO`, `HPR`, `SF`, and `gradient_descent`. |
| `number_iterations` | Number of iterations for each stage. Must be positive integers. |
| `helicity` | Hologram label to reconstruct at each stage. These must match keys in `data["holo"]`, here `+` and `-`. |
| `beta_zero` | Base beta parameter for projection algorithms. |
| `beta_mode` | Beta schedule. Common values are `const` and `arctan`. |
| `alpha_zero` | TV descent strength. `0.0` disables TV regularization. |
| `alpha_mode` | Schedule for `alpha_zero`; usually `const`. |
| `RL_its` | Richardson-Lucy iterations per update. `0` disables partial-coherence updates for that stage. |
| `RL_freqs` | Run RL updates every this many iterations. If `RL_freqs[i] > number_iterations[i]`, that stage is full-coherence. |
| `TV_freqs` | Frequency for TV regularization updates. Very large values effectively disable it. |
| `plot_every` | Diagnostic plotting interval during retrieval. |
| `average_img` | Number of best late-iteration images/gammas to average. |
| `Fourier_last` | If `True`, apply the measured Fourier amplitude constraint at the end of the stage. |
| `Startimage` | Initial complex field for each stage: `None`, an array, or a previous hologram label. |
| `Startgamma` | Initial partial-coherence kernel for each stage: `None`, an array, or a previous hologram label. |
| `output` | Whether this stage should be used as a saved result for the label. |
| `modes` | Modal support scaling factors. `len(modes)` is the number of incoherent modes. |
| `normalize_startimage_between_holograms` | If `True`, rescales a reused start image when switching labels. |
| `return_format` | Usually `auto`. Other supported values are `dict` and `legacy`. |
| `crop` | Number of pixels removed from every edge before retrieval. |
| `hologram_intensity_cutoff_vmin` | Low-intensity cutoff used when constructing masked diffraction constraints. |

### Full vs Partial Coherence

The coherence model is inferred per stage from `RL_its` and `RL_freqs`.

- Full coherence: `RL_its[i] == 0` or `RL_freqs[i] > number_iterations[i]`.
- Partial coherence: `RL_its[i] > 0` and `RL_freqs[i] <= number_iterations[i]`.

Partial-coherence stages update `gamma`, the coherence kernel. Later stages can reuse a previous label's gamma through `Startgamma`.

### Start Images and Start Gamma

`Startimage` controls where each stage starts:

- `None`: start from a new/randomized initial field.
- `"+"` or `"-"`: reuse the latest retrieved field for that mode.
- array: use the array directly.

`Startgamma` works similarly for the partial-coherence kernel:

- `None`: use the default initial gamma.
- `"+"` or `"-"`: reuse the latest gamma for that mode.
- array: use the array directly.

When `normalize_startimage_between_holograms` is `True`, the code rescales a reused start field when it moves from one mode to another. This is useful when `+` and `-` have different intensity levels.

### Modes

`modes` controls multimode retrieval. A 2D `supportmask` is converted into one support per mode using the listed scaling factors. For example:

```python
"modes": [1, 2]
```

runs two modes. The first uses the support as-is, and the second uses a scaled modal support. More modes can model incoherent mixtures, but they cost more memory and runtime and can make interpretation harder.

### Crop

`crop` removes pixels from every hologram edge before retrieval:

```python
"crop": 100
```

means the retrieval uses `image[100:-100, 100:-100]`. The pixel mask is cropped the same way. The support mask is resized and binarized to match the cropped hologram shape. Notebook 04 then rescales `roi_cdi` into the cropped coordinate system before reconstructing the CDI image.

Use a crop when edge artifacts dominate or when reducing runtime is useful. Do not set it so large that it removes meaningful diffraction information.

### Output Selection

The retrieval engine returns three result dictionaries:

- `full_coherence`: latest saved full-coherence result per label.
- `partial_coherence`: latest saved partial-coherence result per label.
- `gradient_descent`: latest saved gradient-descent result per label.

Notebook 04 stores these under each hologram label as:

```python
retrieved_full
retrieved_pc
retrieved_gradient
```

The CDI reconstruction section chooses one with:

```python
retrieved_type = "retrieved_full"
```

Change this to `retrieved_pc` or `retrieved_gradient` if you want the final CDI reconstruction to use those outputs instead.

### Practical Editing Patterns

To repeat the three-stage `HAPRE -> ER -> ER` block several times, change:

```python
times = 1
```

to a larger integer. Make sure any manually sliced list entries still have length `3 * times`.

To make a stage partial-coherence, set nonzero `RL_its` and an `RL_freqs` value smaller than or equal to that stage's iterations:

```python
"RL_its": [0, 0, 50],
"RL_freqs": [1e9, 1e9, 20],
```

To keep everything full-coherence, use:

```python
"RL_its": [0, 0, 0],
"RL_freqs": [1e9, 1e9, 1e9],
```

To save only the final stage for plotting/reconstruction, use `output=False` for intermediate stages and `output=True` for the desired final stage.

## Troubleshooting

`ValueError: Available labels are ...`

The `pol1` and `pol2` labels in notebook 04 do not match the labels saved by notebook 01. Check `data["holo"].keys()` or the labels printed in the load cell.

`Run 03_define_supportmask.ipynb before phase retrieval.`

Notebook 04 did not find `supportmask` in the HDF5 file. Run notebook 03 and save.

`Expected retrieved shape ..., got ...`

The CDI reconstruction cell expects the retrieved hologram shape to equal the support mask shape after applying `crop`. Re-run phase retrieval after changing `crop`, or make sure `phase_retrieval_recipe["crop"]` still matches the saved retrieved arrays.

`phase_retrieval_png` points to a missing file.

Run the plot/save cell in notebook 04 before the final HDF5 save cell. The expected filename is `processed/PhR_recon_ImId_####_USER.png`.

## File Naming Convention

For the starting dataset with positive image ID `13` and `USER = "rb"`:

- HDF5 data: `processed/Logs/data_recon_ImId_0013_rb.hdf5`
- FTH PNG: `processed/FTH_recon_ImId_0013_rb.png`
- Phase retrieval PNG: `processed/PhR_recon_ImId_0013_rb.png`
