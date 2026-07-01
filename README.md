# FTH and Phase Retrieval Notebook Workflow

This folder contains the notebook sequence for going from raw BESSY/P04 holograms to an FTH reconstruction, masks, support, and phase-retrieved CDI reconstruction. The notebooks share one HDF5 data dictionary in `processed/Logs/`, so the usual workflow is to run them in order.

## Quick Start

1. Open `01_FTH.ipynb`.
2. Set `USER`, the raw image IDs, and the polarization labels.
3. Run the notebook through the final save cell.
4. Open `02_define_mask_pixel.ipynb` or `02_define_mask_pixel_napari_fresh.ipynb` and define detector/bad-pixel masks.
5. Open `03_define_supportmask.ipynb` and define the support mask.
6. Open `04_phase_retrieval.ipynb`, adjust the phase retrieval recipe if needed, run phase retrieval, focus the CDI reconstruction, and save.

The current notebooks are configured for:

```python
USER = "rb"
DATA_H5 = "processed/Logs/data_recon_ImId_1269_rb.hdf5"
```

Change these near the top of each notebook when switching to a new dataset.

## Shared HDF5 File

The HDF5 file is the handoff between notebooks. Notebook 01 creates it; later notebooks load it, add results, and save it back.

Important groups/keys:

- `experimental_setup`: geometry and detector metadata used by FTH/CDI propagation.
- `holo`: raw and processed holograms, keyed by labels such as `LH` and `LV`.
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
- `hologram_inputs`: maps labels such as `LH`, `LV`, `CL`, `CR` to raw image IDs and dark/topography inputs.
- `positive_label` and `reference_label`: labels used for the FTH difference hologram.
- `mask_pixel_smooth_recipe`: smooth Butterworth disk mask parameters.

Main outputs:

- `processed/Logs/data_recon_ImId_####_USER.hdf5`
- `processed/FTH_recon_ImId_####_USER.png`
- HDF5 keys for raw holograms, center, FTH reconstruction, `roi`, and `focus`.

## Notebook 02: Pixel Mask

Use one of the notebook 02 variants to define precise bad-pixel masks.

- `02_define_mask_pixel.ipynb`: paint a PNG, draw polygons in notebook widgets, and optionally add thresholded saturated pixels.
- `02_define_mask_pixel_napari_fresh.ipynb`: paint the mask interactively in Napari.

The output is `mask_pixel`, saved into the same HDF5 file. Masked pixels are excluded from measured Fourier constraints during retrieval.

## Notebook 03: Support Mask

Use `03_define_supportmask.ipynb` to create the real-space support for phase retrieval. It builds a preview reconstruction and lets you create or load a support mask.

Important steps:

- Build the support-mask preview reconstruction from the FTH result.
- Define `supportmask` by PNG painting, the circle widget, or manual/loading options.
- Save the support mask into the shared HDF5 file.

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

## Phase Retrieval Recipe

The recipe in notebook 04 is a dictionary named `phase_retrieval_recipe`. It is a step-by-step program for the phase retrieval engine. Each index of the list-valued entries defines one retrieval stage.

For example, with:

```python
"algorithm_list": ["HAPRE", "ER", "ER"],
"number_iterations": [700, 50, 50],
"helicity": ["LH", "LH", "LV"],
```

the retrieval runs:

1. `HAPRE` on `LH` for 700 iterations.
2. `ER` on `LH` for 50 iterations.
3. `ER` on `LV` for 50 iterations.

Many scalar values can be written once, and the code expands them internally to all steps. For clarity, step-specific values are usually written as lists.

### Common Recipe Keys

| Key | Meaning |
| --- | --- |
| `algorithm_list` | Algorithm for each stage. Allowed values include `ER`, `HAPRE`, `RAAR`, `HIO`, `HIOs`, `OSS`, `CHIO`, `HPR`, `SF`, and `gradient_descent`. |
| `number_iterations` | Number of iterations for each stage. Must be positive integers. |
| `helicity` | Hologram label to reconstruct at each stage. These must match keys in `data["holo"]`, such as `LH` and `LV`. |
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
- `"LH"` or `"LV"`: reuse the latest retrieved field for that label.
- array: use the array directly.

`Startgamma` works similarly for the partial-coherence kernel:

- `None`: use the default initial gamma.
- `"LH"` or `"LV"`: reuse the latest gamma for that label.
- array: use the array directly.

When `normalize_startimage_between_holograms` is `True`, the code rescales a reused start field when it moves from one label to another. This is useful when `LH` and `LV` have different intensity levels.

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

For a dataset with image ID `1269` and `USER = "rb"`:

- HDF5 data: `processed/Logs/data_recon_ImId_1269_rb.hdf5`
- FTH PNG: `processed/FTH_recon_ImId_1269_rb.png`
- Phase retrieval PNG: `processed/PhR_recon_ImId_1269_rb.png`
