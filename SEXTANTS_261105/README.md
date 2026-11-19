# SEXTANTS_261105

Notebooks for SEXTANTS images, Fourier power animations, and detector radial and azimuthal averages.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run notebooks from the `SEXTANTS_261105/` directory. Experimental data and generated files are excluded from Git; provide your own data locally.

## MCP workflow

Place the MCP text images in `raw/MCP/` relative to this directory. You can also
symlink the original project's `raw/` directory here.

- [Detector profiles](mcp_detector_analysis.ipynb) calculates radial and azimuthal averages.
- [Radial profiles](mcp_images_radial.ipynb) processes MCP images by filename.
- [Mask template](prepare_mcp_mask_image09.ipynb) exports the scan 09 template.
  Paint the excluded pixels red and save it as
  `output/mcp/beamstop_mask_scan09.png` for
  [zone comparison](mcp_zone_comparison.ipynb), or reuse an existing painted mask.
- [Time traces](read_timetrace.ipynb) analyzes MCP time-trace files.

## Notebooks

- `view_images.ipynb`: create magnetic-field-labeled GIFs and Fourier power GIFs. Put `2026-11-06.rar` in `raw/`; archive extraction uses macOS `bsdtar`. Fourier display uses a linear scale excluding the central region when choosing its limit.
- `mcp_detector_analysis.ipynb`: load detector images, calculate radial and azimuthal averages, plot, and save results in ten cells. MCP text images belong in `raw/MCP/`. Nested entries in `image_list` are averaged pixel by pixel. For SEXTANTS CCD/CMOS NeXus files, set the beamtime folder and `input_type = "sextants"`; integers then identify scan numbers. Image datasets are identified by metadata, with an optional `dataset_path` override. Manual color limits are `mi` and `ma`.
- `prepare_beamstop_mask.ipynb`: preview one image and save a grayscale PNG at its original dimensions using `plt.imsave`. Paint excluded regions red and save without resizing. Set `mask_path` in the analysis notebook; red pixels are excluded from averaging and display block means. The saved template uses the same orientation as the detector plots.

Profiles use radius in pixels and raw intensity per valid pixel. They are not normalized by exposure or time-gate duration. Outputs are written under `output/`.

## CMOS field series

1. Run `prepare_cmos_mask.ipynb` to display scan 16 and save `output/cmos/paint_template.png` using manual `mi`, `ma` limits. Paint excluded regions red and save a separate `beamstop_mask.png` in that folder without resizing.
2. Run `cmos_field_series.ipynb`: set `ims`, one corresponding field per file, beam center, annulus and manual display limits. It saves a labeled GIF, line profiles, radius-versus-field and angle-versus-field PNG plots, CSVs and numerical arrays. Set `mask_path = None` explicitly to analyze without a mask.

The current CMOS files use `scan_NNNN/scan_data/data_22` (Dhyana image); `data_20` is scalar. Both notebooks expose `image_key` for other channel layouts. Multiple exposures in each file are averaged. Field values in the example are placeholders supplied by the user, not measured metadata. Profiles use raw intensity per valid pixel, without background/exposure correction. Radius remains in pixels.

### STXM Fourier views

`view_images.ipynb` saves both linear and logarithmic Fourier power GIFs in `output/fourier_gifs/`, and selected-frame PNGs in `output/fourier_images/`. Set `frame_index` for the PNGs. Both versions include colorbars and field labels. Logarithmic limits use the 1st–99.5th percentiles of positive power outside the central 3-pixel-radius disk, fixed across the animation. Already extracted inputs work without the RAR archive; sources without matching HDF5 field metadata are reported and skipped.

### Compare azimuthal asymmetry at three sample angles

`cmos_asymmetry_three_angles.ipynb` produces one plot of azimuthal std/mean versus field for the 0°, 22°, and 45° scans. It streams images, subtracts optional dark scan 416, excludes the painted beamstop, and uses common sufficiently sampled angular bins across all three scans. The entire detector contributes. The PNG goes to `output/cmos/` and numerical CSVs/settings to `output/cmos/azimuthal/`. Missing files remain NaN. This descriptive full-image metric includes background and detector geometry; it is not a direct ellipticity measurement.

### Select MCP or CMOS

The four newer detector notebooks (`cmos_field_series`, `cmos_asymmetry_three_angles`, `prepare_cmos_mask`, and `load_save_cmos_image`) now start with `detector = "CMOS"` or `"MCP"`. They share the small loader in `detector_io.py`; subsequent analysis is unchanged. MCP text matrices live in `raw/MCP/`; CMOS NeXus files live in `raw/CMOS/`. Outputs go to `output/mcp/` or `output/cmos/`, with numerical data in `radial/` and `azimuthal/`.

MCP integers identify zero-based positions in the printed, sorted `*_TDC_2D_*.txt` list, while CMOS integers identify scan numbers. Set the series scan lists and corresponding fields for your acquisition; existing CMOS ranges and settings are retained. MCP dark subtraction is disabled by default; select an MCP dark index if available. Provide a detector-sized painted mask in the chosen output directory, or explicitly disable masking with `None`. Center and bin settings remain user editable.
