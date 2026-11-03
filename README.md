# Physical projection phase retrieval — paper workspace

This standalone folder contains the algorithm and two experimental workflows: hyperspectral imaging with optical-constant extraction, and 2601_XPCS scan-263 hysteresis reconstruction.

- `manuscript/paper.tex`: editable manuscript scaffold; experimental values, figures, citations and conclusions require author input.
- `docs/`: model, calibration, workflow and source provenance.
- `library/`: numerical reconstruction, geometry, Kramers–Kronig constraints, optical calibration and streaming XPCS preparation. Historical engines and facility library copies are omitted.
- `notebooks/hyperspectral/` and `notebooks/2601_XPCS/`: preparation followed by identically structured reconstruction notebooks.

Create an environment with Python 3.12+, install `requirements.txt`, and start JupyterLab from this folder. In each dataset folder run `00_prepare.ipynb`, inspect the geometry, then run `01_reconstruct.ipynb` from top to bottom. Edit data paths in the first preparation cell. No import or runtime path points back to the original repository.

Raw data are external and are not copied. Hyperspectral preparation expects the existing MAX IV `raw/` metadata and `proc/scan_NNNN_avg.h5` averages; it does not rebuild those facility averages. The included `data/hyperspectral/support.h5` contains `supportmask` and `mask_pixel` datasets on the full detector grid, extracted from the measured source support. XPCS preparation streams the configured Marana frames and builds the support from explicit circles. See `docs/workflows.md`.

`results/` receives HDF5 fields, components, settings, detector residuals and paper analysis, plus a PDF. Choose a new output filename for each run; reconstruction exports refuse to overwrite a result.

The manuscript is a scaffold, not a completed scientific report. No experimental reconstruction or claim of improvement has been validated in this export. Numerical and workflow validation is described in `docs/validation.md`.
