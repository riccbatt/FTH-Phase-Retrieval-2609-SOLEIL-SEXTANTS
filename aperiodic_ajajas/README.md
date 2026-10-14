# Explicit notebook inputs

Notebooks 00–03 no longer read `profiles/active.txt` or profile JSON files.
In 00, set `SATURATED_SCAN`, `LOOP_SCAN`, and `DARK_SCAN` in the first cell.
The raw directory, prefix, optional frame/point selections, photon threshold,
and detector-mask settings are also visible there.

1. Run the early **Create the image to paint** cell. It reconstructs the
   dark-subtracted saturated hologram with `fth.reconstruct` and writes
   `supportmask_<SATURATED_SCAN>_pre.png` beside the notebook using `plt.imsave`.
2. Open it in Paint, mark the support pure red `(255, 0, 0)`, and save as
   `supportmask_<SATURATED_SCAN>.png` without resizing.
3. Finish 00 without loading a support. It saves only prepared holograms,
   detector masks, and metadata to `processed/preprocessed_<LOOP_SCAN>_<SATURATED_SCAN>.h5`.
4. In 01, set `SUPPORT_PNG = HERE / f"supportmask_{SATURATED_SCAN}.png"`.
   Only pure-red pixels are selected. Set `SUPPORT_CENTER_RADIUS` in cached-grid
   pixels to add the central reference circle (0 disables it). The circle is
   added before `SUPPORT_EROSION` and `SUPPORT_DILATION`.
5. Check the before/after contours against `fth.reconstruct(reference)` in 01.
   The check uses the support before automatic retrieval-grid recentering.
   The result stores the adjusted mask, circle radius, morphology settings and
   PNG source, and saves `support_fth_overlay.png` beside reconstruction PNGs.
6. In 02/03, set `SUPPORT_FROM_01` to that saved 01 result. They load its
   adjusted support, independently of any old support dataset in the 00 cache.

In 02, adjust the focus sliders, then run the export cell immediately below
those widgets. It saves real and imaginary focused differences for every
loop state under `reconstructions_png/focused_differences/`, matching the
widget's ROI and 1–99 percentile contrast. Each stage uses its own open
widget's values, or printed display defaults when no widget was opened.

The notes below describe earlier scan-specific defaults; the explicit settings
and paths above supersede historical profile and output-path instructions.

# Aperiodic phase retrieval

The three reconstruction notebooks use the current repository libraries and
the loading conventions of `FTH_CDI_Basic_up_to_data-Copy2.ipynb`. Their
filenames retain the earlier scan numbers; their active scan settings are
2923 (saturated), 2924 (field loop), and 2926 (dark).

- `00_preprocess_2871_2872.ipynb` reads and averages the raw scans once and
  saves `aperiodic_ajajas/processed/preprocessed_2924_2923.h5` in this
  repository. It writes nothing to the external raw-data drive.
- `01_phase_retrieval_2871_2872.ipynb` retrieves saturated scan 2923 and one
  selected point from scan 2924 with the unified two-observation driver,
  reading only the prepared cache.
- `02_universal_hysteresis_2871_2872.ipynb` retrieves scan 2923 and all
  selected scan-2924 states jointly with the universal physical model,
  reading only the prepared cache.
- `FTH_CDI_Basic_up_to_data-Copy2_local_raw.ipynb` preserves the original
  reference functions and analysis cells, with the local raw-data path and
  sparse downloaded point indices. It writes processed files in this repository.

The repository-level
[`phase_retrieval_workflow.md`](../docs/phase_retrieval_workflow.md) is the API
and recipe guide for notebooks 01 and 02. Notebook 02 sets
`SHOW_RECIPE_WORKFLOW = True` by default and prints the actual reference,
start-image, warmup, joint-projection, mode, and final-constraint tree before
running the reconstruction.

The preprocessing notebook reads raw PETRA data under
`/media/riccardo/EXT_16TB1/DATA/P04/11022228/raw` by default. Set
`RAW_ROOT`, `PREFIX`, `DETECTOR_CENTER`, `DARK_SCAN`, and mask settings there.
Before loading and averaging the scans, its optional one-frame photon
diagnostic reads only the selected scan point and frame. Both the diagnostic
and bulk preprocessing retain signal strictly above `PHOTON_LEVEL` (5 in the
active profile), and zero signal at or below it. Set `PHOTON_LEVEL` to `null`
in the profile to disable thresholding. The early preview subtracts the corner
median; bulk preprocessing subtracts the configured dark correction.
Preprocessing centers at `(1021, 891)` and currently keeps the full 2048×2048
grid. It uses `supportmask_00_2.png`. Detector exclusion polygons come from
`detector_mask_polygons_20260214.json`; vertices are `(x, y)`, as in the
reference notebook's `mask_lib`. The reference mask widget shifts them by
`(2.5, -2)` pixels and erodes by 2 pixels before a final 2-pixel dilation.
Preprocessing saves `processed/support_overlay_2924.png` for visual inspection.
The 00 notebook also previews the support on the planned retrieval grid. It
recenters the cached object support if the requested retrieval crop/binning
would cut it, and reports the separate shifts needed by enlarged modes.
`PHOTON_LEVEL = -1000` follows the active Copy2 loop: the level is applied to
each dark-corrected frame before averaging. This level does not clip the
checked scan-2924 point 50 frame (minimum corrected count was −30); a higher
level changes the cached data and requires rerunning preprocessing.

The current repository cache, `processed/preprocessed_2924_2923.h5`, has one
saturated observation and 21 averaged scan-2924 points (100 CMOS frames per
point). The full scan-2924 and scan-2925 files are now available. Notebook 00
sets `LOOP_POINT_STEP = 5` and selects point numbers 0, 5, ..., 100 from
scan 2924; set it to 1 when all points are wanted. Preprocessing assigns
currents using each filename's point number, and retrieval labels states with
those numbers. The
current cache records dark scan 2926 and uses the corrected polygon mask.
Older caches with transposed polygon vertices are corrected on retrieval load.
The earlier comparison is saved at
`processed/mask_pixel_xy_comparison_2923.png`.

Notebook 02 optionally filters the holograms in object space before retrieval.
`FILTER_EDGE_PIXELS` zeros an edge strip without changing array dimensions.
`FILTER_TEMP_CROP_PIXELS` then removes that many pixels from every edge of the
temporary object-space stack before `holograms = real(reconstructCDI(temp))`.
A nonzero crop therefore reduces the recalculated hologram dimensions. The
notebook center-crops the object-space support to the new size and resizes the
detector-space `mask_pixel` with nearest-neighbour interpolation. Supplied
material and thickness maps are center-cropped or padded to the nearest valid
source/retrieval grid. Every automatic adjustment is printed before retrieval.
`SUPPORTMASK_OVERRIDE_FILE` selects the support before this check; set it to
`None` to use the support stored in the preprocessing cache.

Scan-2924 point 0 repeats the approximately +1 A saturated endpoint measured
in scan 2923. The universal hysteresis notebook defaults to `POINT_INDICES =
None`, selecting points 1–20: one magnetic state map for each of the 20
remaining cached field points, plus the separate scan-2923 saturated reference map.
Set `POINT_INDICES` explicitly to include point 0 if needed. Its current retrieval
settings crop 300 cached detector pixels from each edge without binning,
giving a 1448×1448 working grid. The enlarged second mode uses component-centered
support fitting to keep the apertures inside that grid.
The active Copy2 CDI loop calculates a correlation factor but passes the
unscaled field image to phase retrieval. Notebook 01 follows that behavior
with `NORMALIZE_LOOP_INTENSITY = False`. Notebook 02 now also defaults to
`False` so its warmup uses the same intensities as 01. The fitted factor and
intercept remain diagnostic in both notebooks. In notebook 02,
`WARMUP_POLICY = "reference_and_others"` reconstructs saturated scan 2923 with
its `REFERENCE_WARMUP_*` recipe, then seeds each field image from that result
using a fitted intensity slope and applies `OTHER_WARMUP_*`. The default is
HAPRE 700 + ER 50 for the saturated reference and ER 50 for every loop state.
Set `WARMUP_POLICY = "same_for_all"` to apply the reference recipe to every
hologram. `WARMUP_START_FROM_REFERENCE` independently selects a scaled
reference-field start or each hologram's own support-based initialization. Its
detector mask uses the same point-35 pair as the current notebook 01
`STATE_INDEX = 7`, selected by scan point number
rather than cache position. `FREEZE_SATURATED_FIELD = True` keeps the saturated
field fixed during the later joint physical iterations. Notebook 02 plots the
warmup pair's real and imaginary object images with the same region and display
range as 01. Notebook 02 currently uses `MODES = [1, 2]`,
`mode_initialization = "support_fft"`, and the same unified Fourier kernel as
01 so its coherent warmup can be compared directly. The same updates are
available in 01; `USE_PARTIAL_COHERENCE`
controls them in both notebooks. Notebook 02 defaults to one coherence kernel
shared across all field states, with one kernel per mode when `MODES = [1, 2]`.
`COHERENCE_KERNEL_SCOPE = "per_observation"` restores independent kernels.
With `FIX_COHERENCE_AFTER_SATURATED = True`, the shared kernel is learned from
the saturated warmup and then used without Richardson–Lucy updates for the
field states and joint iterations. `final_projection_relaxation = 0` in 02
skips the final physical projection, including its focus propagation and model
fit. Setting `projection_relaxation = 0` similarly skips every repeated
physical projection.
The fitted kernel is saved with the retrieval output. Nonpositive corrected
detector intensities are handled after crop and binning, as in 01. Set
`STATE_STRIDE = 2` to use every second field point, or `STATE_STRIDE = 3` to
use every third one. `FIRST_N_STATES = 10` then limits that strided selection
to its first 10 states. The point chosen by `MASK_REFERENCE_POINT_NUMBER` is
added automatically when `ALWAYS_INCLUDE_COMPARISON_STATE = True`, even when
it falls outside that selection. An explicit `POINT_INDICES` list and
`POINT_INDICES = "preview"` keep their exact selection and do not apply the
stride. The physical projector accumulates its
two-coefficient fit in constant memory instead of building a huge per-pixel
design matrix.

The physical projection operates in the same propagated, phase-rotated sample
plane used by `display_object` and the `focusCDI` sliders. Set
`DISPLAY_PROP_UM`, `DISPLAY_PHASE_RAD`, and `DISPLAY_SETUP_BASE` near the top
of notebook 02. `DISPLAY_PROP_UM` is in micrometres, matching the slider, and
defaults to 2.05. Before every repeated
or final physical projection, the driver focuses the Fourier fields, applies
the charge/magnetization/thickness model, and reverses the transform before the
next detector update. `DISPLAY_PROP_UM = 0` skips both propagation and
phase rotation. Saved physical components are in the focused sample plane;
saved Fourier fields remain in the original retrieval convention.

The two-observation notebook defaults to cache position 5 (about 0 A), since
position 0 repeats the saturated +1 A endpoint and has little magnetic
contrast. It now uses the Copy2 single-mode HAPRE 700 / ER 50 field sequence,
then starts the saturated reference from that retrieved field for 50 HAPRE
iterations. Optional
Richardson-Lucy partial-coherence refinement enabled by default. Set
`MODES = [1, 2]` there for multimode retrieval. Its diagnostic plots use
Copy2's 2.05 µm propagation and −0.76 rad phase before showing the real and
imaginary object difference. The universal notebook uses the same display
correction, but its physical refinement still uses a coherent detector model;
its result is not equivalent to Copy2's partial-coherence result.

Both retrieval notebooks now fit every requested modal support inside the
retrieval object grid before initializing Fourier fields. Mode 2 may move
independently; mode 0 stays at its measured position when already in view.
Notebook 02 uses component-centered mode expansion so the off-center apertures
remain in the retrieval grid. Source-grid shifts and
modal shifts are printed and saved with the retrieval result. Known material
and thickness maps in notebook 02 move with a shifted source support.

Both retrieval notebooks load the painted support from the preprocessing cache
with `SUPPORT_SHIFT = (0, 0)` by default. Check
`processed/support_overlay_2924.png` before changing the support position.

Set `MODES = [1]` for the original support or `MODES = [1, 2]` for two
incoherent modes. Ajajas uses `MODE_SUPPORT_CENTER = "components"`, so the
second support enlarges each off-center aperture 2× in place. Image-centered
enlargement would move most of this cropped sample outside the retrieval grid.
The detector intensity is the sum of the modal intensities. The shared
charge, magnetic, and thickness projection acts on support-factor-1 mode
(array mode 0) only; support-factor-2 mode still
gets detector and support updates. All selected field points retain separate
magnetization maps. The notebook displays array mode 0 (support factor 1) after
retrieval, and its diagnostic plots use that mode for object-space FFTs. The
result file stores the modal support stack.

Notebook 02 also exposes `CONSTRAIN_OTHER_MODES_COMMON`. When true, mode 2 is
projected to a state-independent common object after every physical projection,
so it cannot carry a separate hysteresis-dependent component. The grouping
retains separate common higher modes for different energy/illumination pairs.
`OTHER_MODES_COMMON_RELAXATION = 1` applies the constraint fully at each
projection; detector updates may separate the mode again until the next
physical projection.

For the universal retrieval, set `MATERIAL_MASK_FILE` to a binary PNG or
NumPy array on the cached object grid: nonzero means material, zero means a
known vacuum hole. `RELATIVE_THICKNESS_FILE` may supply a nonnegative relative
thickness map instead of assuming unit thickness in all material pixels. These
maps are object-space inputs and are distinct from the detector pixel mask and
the phase-retrieval support. Their absolute scale cannot be inferred from the
holograms alone; it is absorbed into the fitted spectral response.
Set `FIT_MATERIAL_THICKNESS = True` to fit a shared nonnegative relative
thickness map. A supplied binary mask then fixes known holes to zero while
allowing thickness to vary in the material region. With one energy and one
helicity, thickness and magnetization can remain correlated, so interpret the
fitted map as a model component rather than a quantitative thickness.
Set `ZERO_THICKNESS_OUTSIDE_SUPPORT = True` to force thickness to zero wherever
the phase-retrieval support is zero. This uses the support after crop/bin
mapping. A reference hole *inside* the support still needs an explicit zero in
`MATERIAL_MASK_FILE` or `RELATIVE_THICKNESS_FILE`.

Set `THICKNESS_APERTURE_MODE = "support"` in the universal notebook to use the
largest connected support aperture as the known material region, while fitting
the relative thickness inside it. Set it to `"fixed"` to hold thickness at 1
inside that aperture and 0 elsewhere. The reusable helper is
`universal.largest_support_component(support_used)`. Magnetization is zero
wherever the final thickness is zero.
