# Physical model and algorithm

For observation a, the physical projector fits the complex object logarithm

    L_a(r) = c_b(a)(r) + t(r) [q_c(E_a) + p_a q_m(E_a) m_s(a)(r)].

Here c is a shared illumination/common object field, t is relative material thickness, p is the signed polarization coefficient, m is a real spatial magnetization factor, and q_c/q_m are complex energy responses. The observation metadata independently specify energy, polarization, magnetic state and illumination group. Relative thickness is zero in vacuum reference holes.

The notebooks first reconstruct each observation independently with HAPRE/ER. They then alternate joint physical projection and ER detector/support updates. The warmup is immutable and can be reused when changing physical-stage settings. Continuation embeds the cached fields on the source grid and verifies that crop/bin preparation recovers exactly the original warmup.

The final full physical projection is enabled. Detector intensity agreement can therefore change after that last projection. Both stages are measured on the same valid detector pixels, summing the intensity of all incoherent modes. Physical-component maps correspond to the physical fit; they should not be silently attributed to a later unconstrained polishing step.

Only numerical operations belong in libraries. Acquisition selection, plotting, comparisons, ROI selection, calibration and export are visible notebook cells. `phase_retrieval_universal.py` retains the established numerical implementation and dependencies reachable from the paper API, including shared kernel machinery; it is still a substantial solver. This export does not claim a new independently rewritten minimal solver. Optional species decomposition, gradient refinement and pure spectral baselines are rejected explicitly. No Poisson postprocessing is included.

## Calibration and identifiability

Use independently established saturation labels, never infer saturation from endpoint order. Both notebooks start without saturation anchors. Hysteresis without anchors is a relative factor, and magnet current remains amperes until a measured current-to-field conversion is supplied. No monotonic trend or binary final magnetization is imposed.

For exp(-i k (n-1) d), use n=1-delta-i beta. With d=d0*t, q=-i*k*d0*(n-1), giving beta=-Re(q)/(k*d0) and delta=Im(q)/(k*d0). The conversion requires a measured reference thickness. The common charge offset, magnetic scale and sign must also be fixed or reported as ambiguous. Thickness alone does not remove these gauges. Inspect `magnetic_scale_anchored`, `charge_absolute_gauge_anchored` and `identifiable` in returned components.

Kramers–Kronig constraints over the measured energy range require a defensible treatment of the finite spectral window and unmeasured tails. Spectral sampling and these assumptions must be documented before interpreting absolute indices.
