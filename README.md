# Universal phase retrieval

A linear introduction to XMCD, hyperspectral and hysteresis reconstruction.
Every notebook runs on small synthetic data without external files and imports
`library.phase_retrieval_universal`. Replace synthetic inputs with centered,
background-corrected intensities normalized for exposure and incident flux.

## Setup

```sh
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebooks in order and run every cell from the top:

1. [Simple XMCD](notebooks/01_simple_xmcd.ipynb): pos/neg phase retrieval without joint projections and logarithmic contrast.
2. [Hyperspectral](notebooks/02_hyperspectral.ipynb): SVD and rank-one spectral projections.
3. [Hysteresis and constraints](notebooks/03_hysteresis_and_constraints.ipynb): state/energy/helicity metadata, saturation and material constraints.
4. [Multimode and coherence](notebooks/04_multimode_and_coherence.ipynb): incoherent modes and partial-coherence retrieval.

[Task-based examples](docs/universal_reconstruction_examples.md) explain how to
adapt these workflows to real measurements. The [API guide](docs/phase_retrieval_universal.md)
details options, assumptions and diagnostics. Iteration counts in the notebooks
are illustrative; inspect convergence and physical identifiability on real data.

The library runs on CPU with NumPy/SciPy. CuPy is optional if you have a compatible
CUDA setup; it is not needed for the tutorials. No legacy retrieval engines are
required or shipped.

Run the checks with `python -m pytest -q`. They execute all four notebooks in order
and verify the public API, shapes and model-specific behavior.
