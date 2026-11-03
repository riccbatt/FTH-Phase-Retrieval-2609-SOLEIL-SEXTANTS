# Physical projection phase retrieval — paper workspace

This standalone folder contains the algorithm and two experimental workflows: hyperspectral imaging with optical-constant extraction, and 2601_XPCS scan-263 hysteresis reconstruction.

- `manuscript/paper.tex`: editable manuscript scaffold; experimental values, figures, citations and conclusions require author input.
- `docs/`: model, calibration, workflow and source provenance.
- `library/`: numerical reconstruction, geometry, Kramers–Kronig constraints, masks and streaming scan preparation.
- `notebooks/hyperspectral/` and `notebooks/2601_XPCS/`: preparation followed by identically structured reconstruction notebooks.

Create an environment with Python 3.12+, install `requirements.txt`, and start JupyterLab from this folder. In each dataset folder run `00_prepare.ipynb`, inspect the geometry, then run `01_reconstruct.ipynb` from top to bottom. Edit data paths in the first preparation cell. No import or runtime path points back to the original repository.

Raw data are external and are not copied. Hyperspectral preparation expects the existing MAX IV `raw/` metadata and `proc/scan_NNNN_avg.h5` averages; it does not rebuild those facility averages. Both preparation notebooks load the painted `supportmask.png` and `mask_pixel.png` files in their dataset’s `masks/` directory. Hyperspectral preparation can also initialize these PNGs from a supplied support HDF5 file. XPCS preparation streams the configured Marana frames and writes the prepared cache consumed by reconstruction. See `docs/workflows.md`.

`results/` receives HDF5 fields, components, settings, detector residuals and paper analysis, plus a PDF. Choose a new output filename for each run; reconstruction exports refuse to overwrite a result.

The manuscript is a scaffold, not a completed scientific report. No experimental reconstruction or claim of improvement has been validated in this export. Numerical and workflow validation is described in `docs/validation.md`.

Reconstruction has two explicit calls: `initialize_phase_retrieval(holograms, mask_pixel, supportmask, recipe)` prepares geometry, fields and the schedule; `universal_phase_retrieval_algorithm(reconstruction)` executes one loop and returns fields, initialization, physical components, masks and convergence errors. Fourier and support updates run every iteration. Recipe stages schedule physical projection, TV and RL fitting. See `docs/model.md` for an executable pattern and module responsibilities.

The default stages are **700 HAPRE + 50 ER**. Passing `RL=True` to `default_universal_phase_retrieval_recipe` appends the same pair with 50 RL-fit iterations every 20 retrieval iterations. Physical projection is enabled by setting both `physical_it` and `physical_every` in a stage. `False` in either count/cadence disables that update. Observation metadata use `polarization`, `energy`, `state` and `illumination`; saturation anchors map state values to signs, for example `{10: +1, 30: -1}`.
