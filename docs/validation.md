# Validation

Run from the paper folder:

    python -m pytest tests -q

Result: **14 passed**. Tests cover physical-fit continuation and optional nearest-sign initialization, optical coefficient signs and thickness conversion, streaming frame means and per-frame saturation, and execution of the actual reconstruction notebook cells through HDF5/PDF export. Both datasets are exercised with single and two-mode fields, with and without crop/bin preprocessing. Cropped synthetic tests disable reference-hole illumination correction because their small synthetic grid excludes those apertures; full-grid cases exercise it. The cropped tests use component-centered modal support expansion to fit that small grid.

The measured hyperspectral support and configured XPCS support were also mapped through their default crop/bin settings and checked for valid factor-2 modal supports.

These checks validate code paths and data handoffs. They do not establish experimental performance, optical-constant accuracy, error bars or improved hysteresis accuracy. Full raw-data processing and reconstruction were not run. Hyperspectral preparation still assumes existing facility frame averages and the source workflow's 500-frame normalization. Verify those metadata before quantitative use. The manuscript has not been typeset or populated with experimental results.

Tested environment: Python 3.13.14; NumPy 2.4.6; SciPy 1.18.0; h5py 3.16.0; Matplotlib 3.11.1; scikit-image 0.26.0; nbformat 5.10.4; pytest 9.1.1. `requirements.txt` specifies lower bounds, not an environment lock. Freeze the environment used for final paper results.
