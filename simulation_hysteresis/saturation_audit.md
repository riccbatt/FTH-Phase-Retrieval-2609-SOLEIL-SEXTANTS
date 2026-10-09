Saturation-copy audit of `processed/physical_ideal_CR.h5`

The input ideal CR holograms and saved simulated exit waves for files 0/1 and 2/3 are exactly identical (relative differences zero). Saved warmup fields also give identical complex log objects for each pair. Thus motion, noise, or label errors do not explain the discrepancy in this run.

The saved physical magnetization means for the first four states were `[1, 0.44861, -1, -0.52920]`. A fresh physical fit to all 24 warmup log objects gave `[1, 0.36783, -1, -0.37602]` after 6 iterations and `[1, 0.36455, -1, -0.34992]` after 500. Increasing physical fitting iterations does not resolve this example.

The model assumes `L_s(r) = C(r) + t(r)*q*m_s(r)`, with a spatially constant complex q and fixed uniform thickness inside the material mask. Hence the two forced endpoints require `L_plus(r)-L_minus(r) = 2*t(r)*q`. The reconstructed anchor contrast violates this condition. Anchors stay at +/-1 even where the data disagrees; free copies minimize the least-squares residual with pixelwise magnetization instead. Clipping and averaging those maps can reduce their mean substantially. Identical inputs therefore need not produce identical fitted magnetization when only one copy is constrained and the model cannot fit its anchor exactly. This is a biased estimator under model mismatch, not evidence for unsaturated input states.

Controls:

- Exact synthetic objects following the model recovered `[1, 1, -1, -1]` even after one fitting iteration.
- The actual saved simulation exit waves, evaluated on the sample object-hole mask, recovered `[1, 0.99999959, -1, -0.99999959]`; log fit RMS was 6.47e-7. Their anchor log contrast was spatially constant to about 1.34e-6. This supports the assumed magnetic model for the actual saturated sample exit waves.
- Fitting only the four reconstructed warmup saturated observations gave free means +/-0.73234. Eroding the dilated retrieval material mask by 4, 10, and 20 pixels improved those means to +/-0.81526, +/-0.83673, and +/-0.85872. Edge/background inclusion contributes but is not the entire error.
- After the physical stage the pairs no longer have identical reconstructed log objects: pair log RMS differences were 0.17808 and 0.22741. The asymmetric anchor constraints have altered the initially identical copies differently.

The retrieval material mask currently derives from a support dilated by four pixels, so its extra aperture rim is treated as uniform magnetic material too. The current final projection relaxation is zero, meaning saved component maps describe the last fit rather than an enforced final physical exit wave. Neither observation alone explains the warmup refit failure above.

The notebook now includes an optional saturation audit for warmup and returned fields. It compares all-state and saturated-observation refits, aperture erosions, spatial anchor contrast, and unclipped local endpoint estimates. Local endpoint normalization gives identical copies exactly +/-1 by construction, so it is a diagnostic of the constant-response assumption and must not be mistaken for independent accuracy validation.

Next reconstruction work should compare retrieved anchor exit waves to simulated exit waves with consistent sampling/phase and check whether the detector footprint integration and propagation used in simulation are captured by the coherent FFT forward model. The simulation's `ideal` intensity is noise-free, but its configuration still enables detector pixel-footprint integration and flat-detector geometry. Its name does not imply an exact match to the inverse model. Do not fix the observed bias simply by forcing the free copies to saturation.
