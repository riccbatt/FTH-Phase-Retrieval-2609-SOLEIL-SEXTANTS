# November 2026 SOLEIL SEXTANTS beamtime

## MCP detector analysis

Run the [MCP and CMOS analysis notebooks](DETECTOR_ANALYSIS.md) from the
repository root. Place or link your MCP acquisition files under `raw/MCP/`.

## FTH and phase retrieval

Run these notebooks from the repository root. Start with the backend setup
cell before importing Matplotlib or Qt. The interactive focus tools use a Qt
backend. The NeXus input path defaults to the November beamtime directory at
SOLEIL. Set it to your local copy when running elsewhere.

1. Use [beamstop stitching](00b_stitch_beamstops_000.ipynb) if combining acquisitions with different beamstop positions. It saves inputs for the single-pair notebook in `processed/stitched/`.
2. Run [single-pair FTH](01_FTH.ipynb) to set the image IDs, detector center, masks, ROI, and focus. Save the calibration to `processed/Logs/`.
3. Run [series FTH](01_FTH_series.ipynb) to apply that calibration to a series. Set `REFERENCE_IMAGE_ID` to the calibration image ID.
4. Use [phase retrieval](04_phase_retrieval.ipynb) with `FTH_SOURCE = "fth"` for a single pair or `"series"` for a saved series point. Review the real-space support and retrieval recipe before running.

Supply the detector and beamstop PNG masks locally under
`processed/mask_pixels/`. The stitching notebook defaults to beamstop masks
835 and 1304. The single-pair FTH and hyperspectral notebooks default to 1795. See the
[archived notebooks](old_code/) for earlier workflow variants.

## Hyperspectral reconstruction

Run [05_hyperspectral_physical.ipynb](05_hyperspectral_physical.ipynb) directly
from the root using its raw energy/helicity pairs, dark frames, and beamstop
mask. The [hyperspectral guide](HYPERSPECTRAL.md) describes dependencies, data
inputs, and reconstruction settings.
