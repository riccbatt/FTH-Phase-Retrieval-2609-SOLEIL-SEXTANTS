# Validation

Run from the repository root:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

**60 tests passed on 2026-11-03.** Plugin autoload was disabled to exclude an unrelated installed napari plugin.

The suite covers:

- Mandatory Fourier/support updates, detector-mask preservation and spatial-only TV.
- Stage-local physical/RL/TV timing, disabled `False` controls, and the 700 HAPRE + 50 ER default and repeated RL round.
- Single/multimode RL normalization and per-observation/shared kernel fitting.
- Saturation anchors keyed by state values, physical-fit reuse, grouped magnetic/illumination modes, reversible focusing, and exact multi-energy/multi-beam synthetic fits.
- Crop/bin geometry and direct initialization from previous reconstruction fields.
- Execution of both reconstruction notebooks with one/two modes and crop/bin variants, including HDF5/PDF export.
- Four complete preparation-to-export workflows using synthetic on-disk raw frames, MAX IV metadata/averages and painted masks, with RL on/off. Non-RL cases use the default 750 retrieval iterations; RL cases shorten iteration counts. These checks include alignment, dark subtraction, saturation-mask propagation and agreement between notebook and solver detector residuals.

All four notebook files pass nbformat validation and Python-cell compilation.

## Code audit

Every remaining top-level library function has a caller in runtime code or notebook code. Exports, tests and self-recursion alone were not counted as callers. Recipe-selected support algorithms are retained as callable dispatch entries. Unused geometry/output remapping, absorption conversion, duplicate mask writers and their orphaned constants were removed.

The beta schedules now share one evaluator. A numerical comparison against the baseline implementation passed for all nine supported modes across five iteration counts and three starting values (135 cases). Four further comparisons verified the unchanged KK sign convention after removing its sign-only wrapper.

## Environment and limits

The local run used Python 3.10.12, NumPy 1.26.4, SciPy 1.15.3, h5py 3.13.0, Matplotlib 3.7.0, scikit-image 0.19.3, nbformat 5.7.0 and pytest 7.1.2. `requirements.txt` recommends newer minimum versions for a fresh environment; this run is not a test of every allowed dependency version.

GPU execution and full experimental reconstructions were not run. Raw experimental data are external. The tests establish numerical contracts and working notebook handoffs, not experimental accuracy, uncertainty bounds or improved hysteresis results. Hyperspectral preparation assumes existing facility averages and a configured frame count (default 500); verify that count and the acquisition metadata before quantitative analysis.
