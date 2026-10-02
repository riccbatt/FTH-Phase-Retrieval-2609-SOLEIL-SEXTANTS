# Shared phase retrieval

The `phase_retrieval*.py`, `retrieval_progress.py` and `poisson_refinement.py`
entries are relative symbolic links to the repository's top-level `library/`.
Edit that implementation: both MAX IV and ajajas use it. Keep this directory
inside the repository when copying notebooks. Restart notebook kernels after
updating the library so Python does not retain an older imported module.
Facility-specific loading and preprocessing helpers remain local.

The prior local universal implementation, including the uncommitted Gaussian
start-gamma experiment, is preserved in
`../legacy/phase_retrieval_universal_before_unification.py` for reference only.
It is not imported by the notebooks.

Notebook 05 exposes radial start normalization, parallel observation updates,
shared/per-observation gamma, aperture-only physical fitting and optional final
quasi-Poisson refinement. Its secondary mode is shared across states and
polarizations within each energy/illumination group; only the primary mode is
fitted by the magnetic model. Independent warmup and detector substeps are
unconstrained until the joint common-mode projection.
