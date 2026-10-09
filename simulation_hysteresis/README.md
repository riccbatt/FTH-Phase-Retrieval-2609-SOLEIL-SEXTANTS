Open `01_physical_hysteresis.ipynb` from the repository root or this directory and run its cells in order using the same scientific Python kernel as the XPCS notebooks.

- `DATA_ROOT` points to the provided simulation run.
- `HOLOGRAM_SOURCE` selects `detected_no_beamstop` or `ideal` (`holograms_ideal` is also accepted).
- `HELICITY` selects `CR` or `CL`. The fitted magnetic response absorbs the helicity sign in this single-helicity workflow.
- `FILE_INDICES=None` loads all 24 files in filename order. A subset must include original file indices 0 (+1) and 2 (-1). Only these two files become anchors; all other states, including saturation replicates 1 and 3, are fitted freely.
- `BEAMSTOP_THRESHOLD=0` excludes every nonzero opacity pixel, including soft edges. True means excluded; the mask remains applied to ideal data as requested.
- The largest connected support component supplies the object-hole material region; reference holes remain outside the magnetic projection. No crop or binning is applied.

The algorithm is adapted from the current `2601_XPCS/01_physical_hysteresis_263.ipynb`: HAPRE/ER warmup, then ER blocks alternating with the joint `physical_factorized` projection. Physical-stage reruns reuse immutable warmup fields. The default final projection enforces the fitted model fully. Detector errors report its compatibility with the observations.

The data contains saturated states, skyrmions, and labyrinths rather than a field sweep. Plots therefore show state order, not magnet current. Ground-truth sample-aperture means are diagnostic only and can differ from retrieval-aperture means at boundaries. Multislice propagation, detector footprints, and noise can limit agreement with the factorized model.

Results are saved to `processed/physical_<source>_<helicity>.h5`, including warmup/final fields, magnetization, source paths, masks, fitted components, and histories. Re-running the save cell replaces that output.

Before warmup, the input-diagnostic cell displays support components and their coordinates, original/masked holograms, original/masked FTH reconstructions, and predicted support autocorrelation outlines. `DIAGNOSTIC_POSITIONS` selects loaded stack positions. FTH peaks are aperture cross correlations, so the supplied exit-wave support should not enclose all displaced FTH images.

`USE_EXIT_WAVE_INTENSITY_WEIGHTS=True` enables spatial least-squares weighting in the single-species physical projector. At every projection it uses the mean squared exit-wave amplitude of the selected saturation references, normalized to mean 1 inside the material aperture. The same map applies to every state. This weights shared response estimation; common fields and individual magnetization maps are pixelwise fits, where a shared positive spatial weight cancels. It does not multiply magnetization or alter the detector constraint. Set it to `False` to recover the unweighted fit. The weight map and weighted residual are displayed and exported in `physical_components`.

Intensity weighting changes the fitted target, while `PROJECTION_RELAXATION` still controls the update: `L_new=(1-r)*L_old+r*L_fit` in complex-log space. `FINAL_PROJECTION_RELAXATION` independently controls the final projection. Fitted component/magnetization maps describe the target before relaxation; displayed exit waves show the actual returned fields.
