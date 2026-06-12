# Phase-Retrieval Libraries

This page describes the theory, implementation, and main usage of:

- `phase_retrieval_core.py`
- `phase_retrieval_core_multimode.py`
- `phase_retrieval_core_multienergy.py`
- `phase_retrieval_core_multienergy_multimode.py`
- `phase_retrieval_core_dichroic.py`

The examples focus on the reconstruction calls. Data preparation, centering,
support construction, and detector corrections are intentionally outside their
scope.

## 1. Shared Phase-Retrieval Model

For a coherent measurement, the detector records an intensity

$$
I(\mathbf q) = \left|\Psi(\mathbf q)\right|^2,
$$

but not the Fourier phase of $\Psi$. Phase retrieval alternates between two
constraint spaces:

1. In Fourier space, replace the amplitude at measured pixels by $\sqrt{I}$.
2. Transform to real space and apply a support-based projection.
3. Transform back and repeat.

The available real-space algorithms include ER, HAPRE, RAAR, HIO, SF, OSS,
CHIO, and HPR. Their feedback parameter is generated from `beta_zero` and
`beta_mode`.

An optional total-variation step is controlled by:

```python
alpha_zero
alpha_mode
TV_freq
```

`alpha_zero=0` disables TV descent. Partial-coherence reconstruction is
available in the base and multimode recipe drivers through Richardson-Lucy
updates of the mutual-coherence estimate.

### Array conventions

Measured holograms are intensities. The low-level kernels receive their square
root as `diffract`.

`mask_pixel` and the internally generated `bsmask` follow this convention:

```text
0       measured pixel, enforce the Fourier amplitude
nonzero unconstrained/invalid Fourier pixel
```

Zero measured intensity remains a valid constraint unless explicitly masked.

The reconstruction functions return Fourier-domain complex fields in the
convention used internally by `PhaseRtrv_core`. In the multi-energy libraries,
the corresponding real-space complex log-exit-wave is obtained by:

```python
L = fourier_field_to_object_log(fields)
```

and converted back with:

```python
fields = object_log_to_fourier_field(L)
```

The complex log representation is

$$
L(\mathbf r)
= \log\!\left|O(\mathbf r)\right|
+ i\,\arg O(\mathbf r),
$$

where $O(\mathbf r)$ is the sample exit wave.

## 2. Base Library

File: `library/phase_retrieval_core.py`

### Purpose

This is the standard single-mode, two-hologram implementation. The inputs are
named `pos` and `neg`, but the base algorithm does not impose a physical
relationship between their reconstructed objects. It runs a user-defined
sequence of independent reconstruction steps and permits one result to
initialize the next.

The main entry points are:

```python
default_phase_retrieval_recipe()
phase_retrieval_algorithm(...)
PhaseRtrv_core(...)
plot_phase_retrieval_errors(...)
```

### Flowchart

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 18, "rankSpacing": 24, "curve": "linear"}, "themeVariables": {"fontSize": "14px"}}}%%
flowchart TD
    A[Load pos/neg<br/>intensities] --> B[Prepare amplitudes<br/>and masks]
    B --> C[Initialize<br/>Fourier field]
    C --> D[Read next recipe step]
    D --> E{Partial<br/>coherence?}
    E -- No --> F[Apply measured<br/>Fourier amplitude]
    E -- Yes --> G[Apply convolved<br/>intensity constraint]
    G --> H[Richardson-Lucy<br/>coherence update]
    F --> I[Transform to<br/>real space]
    H --> I
    I --> J[Optional TV update]
    J --> K[Apply selected<br/>support projection]
    K --> L{More iterations<br/>in this step?}
    L -- Yes --> E
    L -- No --> M[Store helicity result]
    M --> N{More recipe<br/>steps?}
    N -- Yes --> D
    N -- No --> O[Return fields, masks,<br/>coherence, and errors]
```

### Recipe structure

Every list index defines one complete reconstruction step:

```python
recipe = {
    "algorithm_list": ["HAPRE", "ER", "ER"],
    "number_iterations": [700, 50, 50],
    "helicity": ["pos", "pos", "neg"],
    "beta_zero": [0.5, 0.5, 0.5],
    "beta_mode": ["arctan", "const", "const"],
    "alpha_zero": [0.0, 0.0, 0.0],
    "alpha_mode": ["const", "const", "const"],
    "TV_freqs": [1e9, 1e9, 1e9],
    "RL_its": [0, 0, 0],
    "RL_freqs": [1e9, 1e9, 1e9],
    "plot_every": [350, 25, 25],
    "average_img": [30, 30, 30],
    "Fourier_last": [True, True, True],
    "Startimage": [None, "pos", "pos"],
    "Startgamma": [None, None, None],
}
```

`Startimage` may be `None`, an explicit array, `"pos"`, or `"neg"`. Reusing the
other hologram's reconstruction applies a masked amplitude normalization before
the next step.

Richardson-Lucy partial coherence is active when:

```text
RL_its[i] > 0 and RL_freqs[i] <= number_iterations[i].
```

### Usage

```python
from library import phase_retrieval_core as pr

recipe = pr.default_phase_retrieval_recipe()
recipe.update({
    "algorithm_list": ["HAPRE", "ER", "ER"],
    "number_iterations": [700, 50, 50],
    "helicity": ["pos", "pos", "neg"],
})

(
    retrieved_pos,
    retrieved_neg,
    retrieved_pos_partial,
    retrieved_neg_partial,
    bsmask_pos,
    bsmask_neg,
    gamma_pos,
    gamma_neg,
    errors,
) = pr.phase_retrieval_algorithm(
    pos_hologram,
    neg_hologram,
    mask_pixel,
    supportmask,
    phase_retrieval_recipe=recipe,
)
```

## 3. Multimode Library

File: `library/phase_retrieval_core_multimode.py`

### Theory

An incoherent modal mixture produces

$$
I(\mathbf q)
= \sum_m \left|\Psi_m(\mathbf q)\right|^2.
$$

The modes do not interfere. The Fourier constraint rescales all modes by a
common factor so their summed intensity matches the measurement. Real-space
support projections are then applied to each mode independently.

This model can absorb partial spatial coherence or other incoherent field
components, but modal decompositions have gauge freedom: degenerate modes may
rotate or exchange order without changing the measured intensity.

### Compatibility

With:

```python
"Nmodes": 1
```

the library reproduces the single-mode API. For multiple modes, reconstructed
fields have shape:

```text
(Nmodes, nx, ny)
```

while measured holograms remain two-dimensional.

### Flowchart

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 18, "rankSpacing": 24, "curve": "linear"}, "themeVariables": {"fontSize": "14px"}}}%%
flowchart TD
    A[Load pos/neg<br/>intensities] --> B[Initialize<br/>Nmodes fields]
    B --> C[Sum modal<br/>intensities]
    C --> D[Jointly rescale modes<br/>to measured intensity]
    D --> E[Transform all modes<br/>to real space]
    E --> F[Project each mode<br/>inside support]
    F --> G[Transform modes<br/>to Fourier space]
    G --> H{More iterations?}
    H -- Yes --> C
    H -- No --> I[Store modal<br/>reconstruction]
    I --> J[Return Nmodes x nx x ny<br/>field arrays]
```

### Usage

```python
from library import phase_retrieval_core_multimode as pr_multi

recipe = pr_multi.default_phase_retrieval_recipe()
recipe.update({
    "Nmodes": 3,
    "algorithm_list": ["HAPRE", "ER"],
    "number_iterations": [700, 50],
    "helicity": ["pos", "pos"],
})

results = pr_multi.phase_retrieval_algorithm(
    pos_hologram,
    neg_hologram,
    mask_pixel,
    supportmask,
    phase_retrieval_recipe=recipe,
)

retrieved_pos = results[0]  # shape (3, nx, ny)
```

The measured-amplitude consistency check is:

```python
reconstructed_amplitude = np.sqrt(
    np.sum(np.abs(retrieved_pos) ** 2, axis=0)
)
```

## 4. Multi-Energy Library

File: `library/phase_retrieval_core_multienergy.py`

### Alternating reconstruction

Input holograms have shape:

```text
(nE, nx, ny)
```

The driver performs:

1. An optional independent warmup schedule at every energy.
2. The full inner update schedule at every energy.
3. A projection that couples the reconstructed exit waves across energy.
4. Repetition for `outer_iterations`.

The output remains a stack of Fourier fields with shape `(nE, nx, ny)`.

### Flowchart

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 16, "rankSpacing": 22, "curve": "linear"}, "themeVariables": {"fontSize": "14px"}}}%%
flowchart TD
    A[Load energy-resolved<br/>holograms] --> B[Initialize one field<br/>per energy]
    B --> C[Optional warmup<br/>at every energy]
    C --> D[Start outer iteration]
    D --> E[Run inner schedule<br/>at every energy]
    E --> G{Projection model}
    G -- none --> H[Keep independent fields]
    G -- SVD --> I[Common component<br/>plus low-rank residual]
    G -- rank1 --> J[Fit C plus M a_E]
    J --> K{Spectral<br/>constraint}
    K -- free --> L[Retrieve complex<br/>spectrum]
    K -- KK --> M[Infer dispersion<br/>from absorption]
    K -- known beta --> N[Impose supplied<br/>beta spectrum]
    K -- beta + KK --> O[Impose beta-delta<br/>relation]
    H --> P{More outer iterations?}
    I --> P
    L --> P
    M --> P
    N --> P
    O --> P
    P -- Yes --> D
    P -- No --> Q[Apply final projection<br/>and measured amplitudes]
```

### Staged update recipes

The inner schedule uses parallel scalar-or-list settings:

```python
recipe = {
    "inner_mode": ["HAPRE", "ER"],
    "inner_Nit": [700, 50],
    "beta_zero": [0.5, 0.9],
    "beta_mode": ["arctan", "const"],
    "alpha_zero": [0.0, 0.0],
    "alpha_mode": ["const", "const"],
    "TV_freq": [1e9, 1e9],
}
```

This executes 700 HAPRE iterations and then 50 ER iterations for each energy
during every outer iteration. Scalars are broadcast to all stages.

Warmup is configured independently:

```python
recipe.update({
    "warmup_mode": ["HAPRE", "ER"],
    "warmup_Nit": [700, 50],
})
```

Set `"warmup_Nit": 0` to disable it. The optional `warmup_beta_zero`,
`warmup_beta_mode`, `warmup_alpha_zero`, `warmup_alpha_mode`, and
`warmup_TV_freq` keys override inherited inner controls.

### Projection models

#### No coupling

```python
"projection_model": "none"
```

The energy channels are reconstructed independently.

#### SVD low rank

$$
L_E(\mathbf r) = C(\mathbf r) + \Delta_E(\mathbf r),
\qquad
\operatorname{rank}_E(\Delta) \le K.
$$

Use:

```python
{
    "projection_model": "svd",
    "rank": K,
    "projection_static_mode": "mean",
    "energy_weights": None,
    "projection_relaxation": 1.0,
}
```

This is a data-driven coupling between the energy channels. It assumes that,
after removing an energy-independent component, the energy-dependent changes
can be represented by only a few spatial-spectral components. It does not
require a known absorption spectrum, a KK relation, or a prescribed line
shape.

##### Matrix representation

After each energy has undergone its independent phase-retrieval updates, the
Fourier fields are converted to complex real-space log-objects. Each
two-dimensional image is flattened, and the energy channels are placed in the
columns of a matrix:

$$
\mathbf L =
\begin{bmatrix}
L_{1}(\mathbf r_1) & \cdots & L_{n_E}(\mathbf r_1) \\
\vdots              &        & \vdots \\
L_{1}(\mathbf r_P) & \cdots & L_{n_E}(\mathbf r_P)
\end{bmatrix}
\in \mathbb C^{P\times n_E},
$$

where $P=n_xn_y$ is the number of pixels. A row therefore contains the energy
dependence of one pixel, while a column contains the reconstructed log-object
at one energy.

The static component $\mathbf C\in\mathbb C^P$ is estimated according to
`projection_static_mode`:

- `"mean"`: weighted mean across energy, which is the default.
- `"first"`: use the first energy channel as the static reference.
- `"none"`: do not remove a static component.

For the default weighted mean,

$$
C(\mathbf r_p)
= \frac{\sum_E w_E L_E(\mathbf r_p)}
       {\sum_E w_E}.
$$

The residual matrix is then

$$
\boldsymbol\Delta
= \mathbf L-\mathbf C\mathbf 1^{\mathsf T}.
$$

##### Weighted truncated SVD

If `energy_weights` are provided, the code normalizes them to have unit mean
and forms:

$$
\boldsymbol\Delta_w
= \boldsymbol\Delta\,\operatorname{diag}(\sqrt{w_E}).
$$

Consequently, channels with larger weights influence the low-rank fit more
strongly. The code requires every weight to be finite and strictly positive.

It then calculates:

$$
\boldsymbol\Delta_w
= \mathbf U\boldsymbol\Sigma\mathbf V^\dagger.
$$

Only the first $K$ singular components are retained:

$$
\boldsymbol\Delta_{w,K}
= \mathbf U_K\boldsymbol\Sigma_K\mathbf V_K^\dagger.
$$

This is the best rank-$K$ approximation to the weighted residual in the
least-squares Frobenius norm. The code then removes the weighting and restores
the static component:

$$
\boldsymbol\Delta_K
= \boldsymbol\Delta_{w,K}
  \operatorname{diag}(1/\sqrt{w_E}),
\qquad
\mathbf L_{\mathrm{SVD}}
= \mathbf C\mathbf 1^{\mathsf T}+\boldsymbol\Delta_K.
$$

Finally, `projection_relaxation` mixes the projected and unprojected
log-objects:

$$
\mathbf L_{\mathrm{new}}
= (1-\lambda)\mathbf L
+ \lambda\mathbf L_{\mathrm{SVD}},
\qquad 0\le\lambda\le1.
$$

Here $\lambda$ is `projection_relaxation`. A value of `1.0` applies the full
projection; smaller values impose the common structure more gradually.

The projected log-objects are exponentiated and transformed back to the
Fourier-field convention before the next phase-retrieval iteration.

##### Meaning of the rank

The retained SVD terms can be written as

$$
\Delta_E(\mathbf r)
\approx
\sum_{k=1}^{K} M_k(\mathbf r)a_{k,E}.
$$

Each term contains:

- a complex spatial map $M_k(\mathbf r)$;
- a complex energy-dependent coefficient $a_{k,E}$.

Therefore:

- `rank=0` keeps only the energy-independent component.
- `rank=1` permits one dominant energy-dependent spatial pattern.
- `rank=2` permits two independently varying spectral-spatial patterns.
- Larger ranks approach independent reconstruction of every energy.

The implementation caps the effective rank at the number of energy channels.
Once the requested rank is large enough to retain the full residual matrix,
the SVD step no longer removes any residual component.

The decomposition is empirical. Individual SVD components are not guaranteed
to equal particular chemical species or physical mechanisms. Their ordering is
set by explained weighted variance, and components with similar singular
values may rotate within their shared subspace.

Because $\mathbf L$ is complex, the SVD treats absorption-like and phase-like
variations jointly. It does not perform one SVD for the real part and another
for the imaginary part. A retained component may therefore contain correlated
changes in both exit-wave amplitude and phase.

The split between the static component and the residual is also a convention.
For example, with `"mean"`, any energy-independent part of a resonant signal is
absorbed into $C(\mathbf r)$. The complete projected stack is meaningful, but
the individual static and SVD components should not automatically be assigned
a unique physical interpretation.

##### Why it improves multi-energy retrieval

Independent phase retrieval gives every energy channel enough freedom to
develop unrelated phase errors, stagnation artefacts, branch choices, and
noise. In a real energy scan, however, most object structure is shared:

- the support and sample geometry are common;
- the nonresonant charge contribution changes slowly or remains nearly fixed;
- resonant changes usually occupy only a small number of spatial regions and
  spectral line shapes;
- detector noise and algorithmic artefacts are less likely to be correlated
  across all energies.

The low-rank projection keeps variations that recur coherently across the
energy stack and suppresses channel-specific variations that require many
singular components. Information from strong or high-signal channels therefore
helps stabilize weak channels. Repeating the independent Fourier constraints
and the joint low-rank projection alternates between consistency with the
measured data and consistency with the shared sample model.

This is useful only when the low-rank assumption is approximately correct. A
rank that is too small can erase genuine spectral structure or force different
materials into the same component. A rank that is too large provides little
regularization. Inspect:

```python
components["singular_values"]
components["static_log_object"]
components["energy_dependent_log_object"]
```

A practical starting point is to compare reconstructions over several ranks
and look for a stable result whose discarded singular values form a lower,
noise-like tail.

Before interpreting that spectrum, make sure that:

- all energies are spatially registered to the same pixels;
- the support describes the same object region at every energy;
- Fourier-field normalization is reasonably consistent across the stack;
- complex-log phases use compatible branches.

A subpixel drift produces derivative-like components and increases the apparent
rank. Likewise, unrelated $2\pi$ phase wraps can dominate the singular values.
The SVD projection can reduce random inconsistencies, but it cannot determine
whether a large variation is physical, a registration error, or a phase-branch
error.

A complete SVD recipe can look like:

```python
recipe = {
    "inner_mode": ["HAPRE", "ER"],
    "inner_Nit": [700, 50],
    "outer_iterations": 100,
    "warmup_Nit": 0,
    "projection_model": "svd",
    "rank": 1,
    "projection_static_mode": "mean",
    "energy_weights": np.ones(len(energies_eV)),
    "projection_relaxation": 0.5,
}
```

For a first test, equal weights and a relaxation between `0.3` and `1.0` are
reasonable numerical choices. They are not universal physical defaults:
rank, weights, and relaxation should be checked against reconstruction
stability and preservation of known spectral features.

##### SVD projection sequence

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 16, "rankSpacing": 22, "curve": "linear"}, "themeVariables": {"fontSize": "14px"}}}%%
flowchart LR
    A[Fourier fields<br/>at all energies] --> B[Convert to complex<br/>log-objects]
    B --> C[Flatten images into<br/>pixels x energies]
    C --> D[Estimate and subtract<br/>static component]
    D --> E[Apply square-root<br/>energy weights]
    E --> F[Compute SVD]
    F --> G[Keep first K<br/>components]
    G --> H[Undo weighting and<br/>restore static part]
    H --> I[Relax toward<br/>projected stack]
    I --> J[Convert back to<br/>Fourier fields]
```

#### Explicit rank-one spectrum

$$
L_E(\mathbf r) = C(\mathbf r) + M(\mathbf r)\,a_E.
$$

Use:

```python
"projection_model": "rank1_spectral"
```

$C(\mathbf r)$ is energy independent, $M(\mathbf r)$ is the spatial map of the
energy-dependent component, and $a_E$ is its complex spectrum.

Available spectral constraints are:

```text
free            retrieve both parts of a_E
kk              retrieve absorption-like part and calculate dispersion
known_beta      impose supplied beta(E), retain retrieved dispersion
known_beta_kk   impose beta(E) and supplied or KK-calculated delta(E)
```

`known_beta_spectrum` means the imaginary part `beta(E)` of the refractive
index, not raw absorption data. Convert measured absorption to beta and extend
or KK-transform it beforehand with `library/kramers_kronig.py`.

`absorption_part` specifies whether the absorption-like component occupies the
real or imaginary part of the fitted coefficient `a_E`. It is a factorization
convention and should be kept consistent between reconstruction and analysis.

### Usage

```python
from library import phase_retrieval_core_multienergy as pr_energy

recipe = {
    "inner_mode": ["HAPRE", "ER"],
    "inner_Nit": [700, 50],
    "outer_iterations": 100,
    "warmup_Nit": 0,
    "projection_model": "rank1_spectral",
    "spectral_constraint": "known_beta_kk",
    "energy_values": energies_eV,
    "known_beta_spectrum": beta_spectrum,
    "known_delta_spectrum": delta_spectrum,
    "projection_relaxation": 1.0,
}

fields, components, bsmasks, errors = (
    pr_energy.multi_energy_phase_retrieval_algorithm(
        holograms,
        mask_pixel,
        supportmask,
        multi_energy_recipe=recipe,
    )
)
```

Important outputs include:

```python
components["static_log_object"]
components["spectral_spatial_map"]
components["spectral_coefficients"]
```

## 5. Multi-Energy Multimode Library

File: `library/phase_retrieval_core_multienergy_multimode.py`

This combines the incoherent modal Fourier constraint with the multi-energy
object projection.

Fields have shape:

```text
(nE, Nmodes, nx, ny)
```

or `(nE, nx, ny)` when `Nmodes=1`.

The multi-energy projection is applied independently to each mode:

```text
mode 0 across all energies
mode 1 across all energies
...
```

Consequently, mode indices must remain physically corresponding across energy.
The code preserves mode order during updates, but it cannot identify arbitrary
mode permutations or unitary mixing between degenerate modes.

### Flowchart

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 18, "rankSpacing": 24, "curve": "linear"}, "themeVariables": {"fontSize": "14px"}}}%%
flowchart TD
    A[Load energy stack] --> B[Initialize Nmodes fields<br/>at every energy]
    B --> C[Run multimode updates<br/>at every energy]
    C --> D[Collect mode m<br/>across energies]
    D --> E[Apply multi-energy<br/>projection to mode m]
    E --> F{More modes?}
    F -- Yes --> D
    F -- No --> G[Reassemble energy-mode<br/>field stack]
    G --> H{More outer<br/>iterations?}
    H -- Yes --> C
    H -- No --> I[Enforce measured<br/>summed intensity]
    I --> J[Return nE x Nmodes<br/>x nx x ny fields]
```

### Usage

```python
from library import phase_retrieval_core_multienergy_multimode as pr_em

recipe = {
    "Nmodes": 3,
    "mode_initialization_seed": 0,
    "inner_mode": ["HAPRE", "ER"],
    "inner_Nit": [700, 50],
    "outer_iterations": 100,
    "warmup_Nit": 0,
    "projection_model": "svd",
    "rank": 1,
}

fields, components, bsmasks, errors = (
    pr_em.multi_energy_phase_retrieval_algorithm(
        holograms,
        mask_pixel,
        supportmask,
        multi_energy_recipe=recipe,
    )
)
```

Per-mode projection results are available in:

```python
components["mode_components"]
```

The measured intensity is reconstructed from:

```python
intensity = np.sum(np.abs(fields) ** 2, axis=1)
```

## 6. Dichroic Multi-State Library

File: `library/phase_retrieval_core_dichroic.py`

### General model

For observation `j`, magnetic state `s(j)`, and polarization coefficient `p_j`,
the shared-charge model is:

$$
L_j(\mathbf r)
= C(\mathbf r)
+ p_j M_{s(j)}(\mathbf r).
$$

`state_labels[j]` identifies the sample state, while
`polarization_signs[j]` is normally `+1` or `-1`. These names deliberately
separate sample state from helicity.

### Flowchart

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 15, "rankSpacing": 22, "curve": "linear"}, "themeVariables": {"fontSize": "14px"}}}%%
flowchart TD
    A[Load holograms, states,<br/>and polarization signs] --> B[Initialize one field<br/>per observation]
    B --> C[Optional warmup]
    C --> D[Run inner schedule<br/>for every observation]
    D --> E[Convert to complex<br/>log-exit waves]
    E --> F{Dichroic projection}
    F -- none --> G[Keep independent<br/>reconstructions]
    F -- shared charge --> H[Fit common charge and<br/>state magnetic terms]
    F -- saturated --> I[Fit charge and<br/>magnetic terms]
    I --> J[Infer q from<br/>saturated states]
    J --> K{Optional kt-delta<br/>or kt-beta bounds?}
    K -- No --> L[Keep data-inferred q]
    K -- Yes --> M[Clip corresponding<br/>q components]
    L --> N[Fit real mz maps<br/>for all states]
    M --> N
    G --> O{More outer<br/>iterations?}
    H --> O
    N --> O
    O -- Yes --> D
    O -- No --> P[Apply measured amplitudes<br/>and return components]
```

### Shared-charge projection

```python
"projection_model": "shared_charge"
```

This fits a common complex charge log-object and an unconstrained complex
magnetic log-object for every state. The weighted linear system is solved
pixel-by-pixel in vectorized form.

#### Relation to low-rank/SVD coupling

The same low-dimensional idea that motivates the multi-energy SVD also applies
to multiple magnetic states. If every reconstructed state were independent,
each image could acquire a different charge background, phase offset, or
reconstruction artefact. The physical model instead says that all observations
are generated from a smaller collection of latent images:

$$
L_j(\mathbf r)
= C(\mathbf r)
+ p_jM_{s(j)}(\mathbf r).
$$

For $n_s$ magnetic states, the observation stack is therefore generated by at
most $1+n_s$ complex spatial components: one shared charge component and one
magnetic component per state. In matrix form:

$$
\mathbf L
= \mathbf X\mathbf A^{\mathsf T},
$$

where:

- $\mathbf L\in\mathbb C^{P\times n_{\mathrm{obs}}}$ contains flattened
  log-objects;
- $\mathbf X\in\mathbb C^{P\times(1+n_s)}$ contains the unknown charge and
  state magnetic maps;
- $\mathbf A\in\mathbb R^{n_{\mathrm{obs}}\times(1+n_s)}$ is fixed by
  `state_labels` and `polarization_signs`.

The first column of $\mathbf A$ is one. Observation $j$ has
$p_j$ in the column corresponding to state $s(j)$ and zero in the other state
columns. The code solves this linear model by weighted least squares at every
pixel and rebuilds all observations from the fitted shared components.

#### Identifiability of the shared-charge model

For $S$ distinct magnetic states, the unknown latent images are:

$$
C(\mathbf r),\ M_1(\mathbf r),\ldots,M_S(\mathbf r).
$$

There are therefore $S+1$ unknown complex values at each pixel. Observation
$j$, belonging to state $s(j)$, contributes the design-matrix row

$$
\mathbf A_j
=
\begin{bmatrix}
1 & 0 & \cdots & p_j & \cdots & 0
\end{bmatrix},
$$

where $p_j$ appears in the column for state $s(j)$. The decomposition is unique
only when:

$$
\operatorname{rank}(\mathbf A)=S+1.
$$

For the state-specific design used by this library, this has a simple practical
interpretation:

1. Every state must appear in at least one observation.
2. At least one state must be observed with two distinct polarization
   coefficients, normally $+1$ and $-1$.
3. At least $S+1$ total observations are required.

The paired state fixes the common charge component. Once $C(\mathbf r)$ is
anchored, a single-polarization observation of another state is sufficient to
determine that state's unconstrained complex magnetic term:

$$
M_s(\mathbf r)
= \frac{L_j(\mathbf r)-C(\mathbf r)}{p_j}.
$$

The state carrying the opposite-polarization pair can be any state. It does not
have to be saturated. Conversely, simply observing many states does not help
if every state is seen at only one polarization: the charge can still be
shifted while compensating all magnetic terms.

The following configurations summarize the requirement:

| Measurements | Identifiable by `shared_charge`? | Reason |
|---|---:|---|
| One state at $+1$ and $-1$ | Yes | Two observations determine $C$ and one $M_s$. |
| $S$ states; one state at both signs, every other state at one sign | Yes | This is the minimum $S+1$-observation design. |
| Every state at both signs | Yes | Overdetermined and usually better conditioned. |
| One observation per state, all at the same sign | No | No observation anchors the charge/magnetic separation. |
| Domain state and saturated state, both only at $+1$ | No | Saturation metadata does not repair the rank-deficient first-stage fit in the current implementation. |

The code calculates and returns:

```python
components["design_matrix"]
components["design_rank"]
components["design_condition_number"]
components["identifiable"]
```

`design_rank == S + 1` establishes algebraic uniqueness. It does not guarantee
good numerical conditioning. A large `design_condition_number` means that
small reconstruction errors can produce large changes in the separated charge
and magnetic maps. Balanced $+1/-1$ measurements and repeated observations
generally improve conditioning.

The minimum design is not automatically a strong regularizer. For example, a
single state measured once at each polarization gives two equations for two
complex unknowns:

$$
C = \frac{L_+ + L_-}{2},
\qquad
M = \frac{L_+ - L_-}{2}.
$$

Every pair of reconstructed images can be represented this way, so the
`shared_charge` projection has no residual to reject. Stability improves when
the system is overdetermined, when several observations share the same latent
components, or when the stronger saturated-reference model is valid.

This is closely related to a low-rank projection because the observation
matrix is restricted to the column space of a small number of latent maps.
However, the current dichroic library does **not** calculate an unconstrained
SVD for this step. It uses the known state/polarization design matrix, which is
more informative:

- SVD would discover an arbitrary low-dimensional basis.
- The dichroic design matrix assigns the basis explicitly to charge and
  magnetic-state terms.
- Polarization reversal fixes the sign with which a magnetic term enters each
  observation.
- Saturated states can additionally fix the magnetization scale.

The advantage is the same as in multi-energy retrieval: information shared by
several measurements is pooled, while variations inconsistent with the joint
model are rejected. In particular:

- repeated observations constrain the same charge exit wave;
- opposite polarizations separate charge-like and magnetic-like terms;
- a well-constrained state can stabilize another state measured with lower
  signal;
- uncorrelated reconstruction artefacts are less likely to fit the prescribed
  state/polarization structure.

The restriction also creates an identifiability requirement. The columns of
$\mathbf A$ must be linearly independent. If they are not, multiple sets of
charge and magnetic maps reproduce exactly the same observations. The code
checks the design rank and rejects such a geometry by default.

The `saturated_reference` model imposes a still stronger pointwise
factorization:

$$
M_s(\mathbf r) = q(\mathbf r)m_{z,s}(\mathbf r),
$$

where the complex response $q(\mathbf r)$ is shared and each $m_{z,s}$ is real.
At each pixel, all state-dependent magnetic terms must therefore lie on the
same line in the complex plane, with real coefficients $m_{z,s}(\mathbf r)$.
This is not necessarily a global rank-one matrix across pixels and states,
because every state may have a different spatial magnetization map. Unlike a
generic SVD basis, the saturated observations provide the normalization and
complex direction of $q$, making the pointwise factors physically
interpretable when the model assumptions hold.

### Saturated-reference projection

```python
"projection_model": "saturated_reference"
```

This estimates the magnetic response from the reconstructed saturated state and
then enforces:

$$
L_j(\mathbf r)
= C(\mathbf r)
+ p_j q(\mathbf r)m_{z,s(j)}(\mathbf r),
$$

where $q(\mathbf r)$ is inferred from the data and $m_z$ is real. With

$$
n_m = -(\delta_m+i\beta_m)m_z,
\qquad
\phi = \phi_c\exp(-ikt\,n_m),
$$

the inferred unit-magnetization log response is:

$$
q = ikt(\delta_m+i\beta_m)
  = -kt\beta_m + i\,kt\delta_m.
$$

The user does not need to supply $\delta_m$, $\beta_m$, thickness, or $q$.
Instead, the user flags at least one state with known saturated magnetization:

```python
saturated_states = ["saturated_up"]  # interpreted as mz=+1

# Explicit signs are also supported:
saturated_states = {
    "saturated_up": +1,
    "saturated_down": -1,
}
```

Supplying `saturated_states` to the reconstruction driver automatically selects
the saturated-reference projection unless `projection_model` is explicitly
overridden.

#### How the saturated-reference projection works

The current implementation applies the constraint in two stages.

First, it performs the full-rank shared-charge fit:

$$
L_j(\mathbf r)
= C(\mathbf r)+p_jM_{s(j)}(\mathbf r).
$$

This stage must already satisfy the design-rank conditions described above.
Marking a state as saturated does not bypass this requirement.

Second, for each saturated state $s$, whose magnetization is supplied as
$m_{z,s}^{\mathrm{sat}}=+1$ or $-1$, it forms a response estimate:

$$
q_s(\mathbf r)
= \frac{M_s(\mathbf r)}
       {m_{z,s}^{\mathrm{sat}}}.
$$

With multiple saturated states, the current code averages these estimates:

$$
q(\mathbf r)
= \frac{1}{N_{\mathrm{sat}}}
  \sum_{s\in\mathcal S_{\mathrm{sat}}} q_s(\mathbf r).
$$

Optional `kt_delta_m_range` and `kt_beta_m_range` bounds are then applied to
the imaginary and negative-real parts of $q$, respectively.

For every nonsaturated state, the closest real magnetization coefficient is
obtained by projecting its complex magnetic term onto the complex direction
defined by $q$:

$$
m_{z,s}(\mathbf r)
=
\frac{
  \operatorname{Re}\!\left[
    q^*(\mathbf r)M_s(\mathbf r)
  \right]
}{
  \left|q(\mathbf r)\right|^2
}.
$$

This is a real least-squares fit of
$M_s(\mathbf r)\approx q(\mathbf r)m_{z,s}(\mathbf r)$. If
`clip_magnetization=True`, the result is additionally clipped to
$[-1,1]$. At pixels where $\lvert q\rvert$ is numerically zero, the code cannot
determine a magnetization coefficient and returns zero.

The projected observations are finally rebuilt as:

$$
L_j^{\mathrm{proj}}(\mathbf r)
= C(\mathbf r)
+ p_jq(\mathbf r)m_{z,s(j)}(\mathbf r).
$$

This real-$m_z$ constraint is what gives the saturated-reference model more
stabilizing power than the unconstrained `shared_charge` model. Complex
state-dependent fluctuations that do not lie along the response direction
$q(\mathbf r)$ are rejected.

#### Required number of states and holograms

Only one saturated state is mathematically required. More saturated states are
optional and provide multiple estimates of $q$, but they are useful only if
they genuinely share the same response.

The minimum configurations are:

| Goal | Minimum identifiable measurements |
|---|---|
| Estimate charge and the response of one saturated state | The saturated state measured at two distinct polarizations, normally $+1$ and $-1$: 2 holograms. |
| Reconstruct one nonsaturated state using one saturated reference | Two states and at least 3 holograms: one state measured at two polarizations, the other measured at least once, with either state flagged saturated. |
| Reconstruct $S-1$ nonsaturated states using a saturated reference | $S$ states and at least $S+1$ holograms: every state once, one state at a second polarization, and at least one of the $S$ states saturated. |
| Add redundancy and stronger noise rejection | Measure several or all states at both polarizations; for $S$ states this gives up to $2S$ holograms. |

For example, this three-hologram design is sufficient for a saturated state and
one domain state:

```python
state_labels = ["saturated", "saturated", "domains"]
polarization_signs = [+1, -1, +1]
saturated_states = {"saturated": +1}
```

The opposite-polarization pair determines $C$ and the saturated magnetic term.
The remaining observation determines the domain-state magnetic term, which is
then constrained to the real-$m_z$ direction.

This alternative is also algebraically identifiable:

```python
state_labels = ["domains", "domains", "saturated"]
polarization_signs = [+1, -1, +1]
saturated_states = {"saturated": +1}
```

Here the domain pair anchors $C$, and the single saturated observation provides
$M_{\mathrm{sat}}$ once $C$ is known. The saturated state itself therefore does
not have to be the state measured at both polarizations.

By contrast, this is not sufficient for the current implementation:

```python
state_labels = ["domains", "saturated"]
polarization_signs = [+1, +1]
saturated_states = {"saturated": +1}
```

There are two observations but three unknown first-stage maps:
$C$, $M_{\mathrm{domains}}$, and $M_{\mathrm{sat}}$. The shared-charge design
is rank deficient, so the code rejects it even though one state is marked
saturated.

#### Physical conditions for validity

The algebraic rank condition is necessary but not sufficient. The
saturated-reference model also assumes:

1. **Common charge exit wave.** The nonmagnetic complex transmission
   $C(\mathbf r)$ must be the same for every state and polarization. Structural
   motion, changing illumination, damage, or charge-state changes violate this
   assumption.
2. **Known polarization coefficients.** The supplied signs must correctly
   describe the magnetic coupling. They are normally $+1$ and $-1$, and the
   convention must be used consistently.
3. **Common magnetic response.** The same complex $q(\mathbf r)$ must relate
   magnetization to the magnetic log-object in every state. This assumes the
   same photon energy, sample thickness distribution, composition, and magnetic
   optical constants across the state series.
4. **Real scalar magnetization.** State changes must be representable by a real
   out-of-plane factor $m_z(\mathbf r)$. A changing complex response, additional
   magnetic components, or polarization-dependent nonmagnetic effects are not
   represented by this model.
5. **Correct saturation metadata.** A state marked `+1` or `-1` must have that
   known uniform magnetization over the reconstructed magnetic region. Partial
   saturation biases both $q$ and every subsequently retrieved $m_z$ map.
6. **Spatial registration.** Corresponding pixels must represent the same
   sample location in all holograms. Drift is otherwise interpreted as a
   state-dependent magnetic signal.
7. **Compatible complex-log branches.** The reconstructed phases must use
   compatible branches. Unrelated $2\pi$ wraps violate the additive log-object
   equations.
8. **Nonzero response.** Magnetization is identifiable only where
   $\lvert q(\mathbf r)\rvert$ is sufficiently large compared with noise.

Multiple saturated states should produce mutually consistent estimates
$q_s(\mathbf r)$. Their disagreement is a useful model diagnostic: it can
indicate incomplete saturation, drift, state-dependent optical constants, or
phase-branch errors. The current implementation averages the estimates rather
than fitting separate responses.

#### Why the joint reconstruction can be more stable

The algorithm alternates between two different sources of information:

1. **Per-observation phase retrieval** enforces each hologram's measured
   Fourier amplitudes and the common real-space support.
2. **Dichroic projection** forces the reconstructed log-exit waves to share the
   same charge term and to follow the supplied state/polarization structure.

The Fourier data alone do not determine the phase uniquely, and independent
reconstructions may settle into different stagnation points. The dichroic
projection transfers information between observations:

- a polarization pair helps determine the charge term used by every state;
- repeated states constrain one magnetic map from several holograms;
- a saturated state determines the complex magnetic-response direction used
  to constrain all nonsaturated states;
- deviations that cannot be represented by the joint model are removed before
  the next Fourier-amplitude update.

This does not make phase retrieval convex and does not guarantee convergence to
the true object. The benefit depends on the joint model being more accurate
than the independent reconstruction errors. If the shared-charge or
shared-response assumptions are wrong, the same projection can force a
consistent but biased answer.

The amount of actual regularization also depends on the measurement design:

- A single nonsaturated state measured at $+1$ and $-1$ is identifiable under
  `shared_charge`, but every image pair has an exact mean/difference
  decomposition. The joint projection adds essentially no extra constraint.
- A single saturated state measured at both polarizations identifies $C$ and
  $q$, but contains no unknown magnetization state to stabilize.
- One saturated state plus at least one nonsaturated state lets the
  saturated-reference projection impose the real-$m_z$ condition on the
  nonsaturated reconstruction. This is the smallest design that uses the
  saturated response to constrain an unknown state.
- Additional polarizations, repeated acquisitions, or additional states make
  the system overdetermined and allow the projection residual to reject noise
  and model-inconsistent reconstruction features.

Monitor the following diagnostics rather than relying only on the final image:

```python
components["identifiable"]
components["design_rank"]
components["design_condition_number"]
components["fit_residual_rms"]
components["magnetic_response_unconstrained"]
components["magnetic_response"]
```

A low residual is meaningful only together with a full-rank, reasonably
conditioned design and physically credible saturated-state assumptions.

The reconstruction returns:

```python
components["magnetic_response"]
components["magnetic_log_attenuation"]  # -real(q) = k*t*beta_m
components["magnetic_phase_shift"]      #  imag(q) = k*t*delta_m
components["magnetization_by_state"]
```

These are the dimensionless refractive-index-thickness products measured by
the exit wave. Even though `k` is known from the illumination energy, the
holograms determine only:

$$
kt\delta_m
\qquad\text{and}\qquad
kt\beta_m.
$$

They cannot determine $t$, $\delta_m$, and $\beta_m$ independently. Absolute
$\delta_m$ and $\beta_m$ require an independently known sample thickness:

$$
\delta_m
= \frac{\text{magnetic phase shift}}{kt},
\qquad
\beta_m
= \frac{\text{magnetic log attenuation}}{kt}.
$$

Without independently known $t$, the retrieval cannot distinguish refractive
index from thickness.

### Optional product ranges

The default recipe is exclusively data driven and contains no required
refractive-index prior:

```python
{
    "kt_delta_m_range": None,
    "kt_beta_m_range": None,
}
```

If approximate ranges for the observable products are available, they may be
imposed directly inside the joint projection:

```python
recipe = {
    "kt_delta_m_range": (kt_delta_min, kt_delta_max),
    "kt_beta_m_range": (kt_beta_min, kt_beta_max),
}
```

These dimensionless bounds map directly to the inferred complex response:

$$
\operatorname{Im}(q) \in \texttt{kt\_delta\_m\_range},
\qquad
-\operatorname{Re}(q) \in \texttt{kt\_beta\_m\_range}.
$$

The projection clips the two components independently before refitting the
real magnetization maps. This gives a conservative rectangular constraint in
complex response space. `projection_relaxation` controls how strongly the
bounded projection is mixed into the current reconstruction.

The returned diagnostics distinguish the raw and constrained estimates:

```python
components["magnetic_response_unconstrained"]
components["magnetic_response"]
components["physical_response_bounds_applied"]
components["response_bounds"]
```

Either range may be supplied independently. For example, to constrain only the
magnetic attenuation product:

```python
recipe = {
    "kt_beta_m_range": (0.01, 0.02),
}
```

This constrains attenuation while leaving the magnetic phase shift data driven.
If both entries remain `None`, no such constraint is applied.

### Opposite-polarization pair

```python
from library import phase_retrieval_core_dichroic as pr_dichroic

recipe = {
    "inner_mode": ["HAPRE", "ER"],
    "inner_Nit": [700, 50],
    "outer_iterations": 100,
    "warmup_Nit": 0,

    # Optional; omit both entries for a data-only reconstruction.
    "kt_delta_m_range": (kt_delta_min, kt_delta_max),
    "kt_beta_m_range": (kt_beta_min, kt_beta_max),
}

fields, components, bsmasks, errors = (
    pr_dichroic.dichroic_phase_retrieval_algorithm(
        np.stack([hologram_pos, hologram_neg]),
        mask_pixel,
        supportmask,
        state_labels=["state_A", "state_A"],
        polarization_signs=[+1, -1],
        saturated_states=["state_A"],
        dichroic_recipe=recipe,
    )
)

mz = components["magnetization_by_state"]["state_A"]
charge = components["charge_log_object"]
response = components["magnetic_response"]
```

Mark a state as saturated only when its magnetization is known to be uniformly
`+1` or `-1`. A non-saturated opposite-polarization pair can still be
reconstructed with `shared_charge`, but it cannot establish the absolute
magnetization scale.

### Multiple states

```python
recipe = {
    "inner_mode": ["HAPRE", "ER"],
    "inner_Nit": [700, 50],
}

fields, components, bsmasks, errors = (
    pr_dichroic.dichroic_phase_retrieval_algorithm(
        np.stack([
            hologram_saturated_pos,
            hologram_saturated_neg,
            hologram_domains_pos,
        ]),
        mask_pixel,
        supportmask,
        state_labels=["saturated", "saturated", "domains"],
        polarization_signs=[+1, -1, +1],
        saturated_states=["saturated"],
        dichroic_recipe=recipe,
    )
)

mz_domains = components["magnetization_by_state"]["domains"]
```

The opposite-polarization saturated pair anchors the shared charge and complex
magnetic response. Additional states may then be supplied with only one
polarization.

### Identifiability limitation

A same-polarization domain image plus a same-polarization saturated image is
not sufficient by itself:

```python
state_labels = ["domains", "saturated"]
polarization_signs = [+1, +1]
```

Marking the second state as saturated fixes `mz=1`, but the data still permit a
joint offset/scale transformation between charge and magnetic response. The
library rejects this rank-deficient geometry instead of returning an arbitrary
decomposition. At least one opposite-polarization partner, or another
independent physical normalization, is required.

## 7. Choosing a Library

| Data/model | Library |
|---|---|
| Two standard coherent holograms | `phase_retrieval_core` |
| Two holograms with incoherent modes | `phase_retrieval_core_multimode` |
| Energy stack with shared spectral structure | `phase_retrieval_core_multienergy` |
| Energy stack with incoherent modes | `phase_retrieval_core_multienergy_multimode` |
| Multiple magnetic states or helicities with shared charge | `phase_retrieval_core_dichroic` |

Start with the simplest model supported by the data. Increasing rank, mode
count, or the number of unconstrained state components increases flexibility
but also increases gauge freedom and the risk of unstable decompositions.

## 8. Practical Checks

Before interpreting a reconstruction:

1. Confirm that measured Fourier amplitudes are recovered outside `bsmask`.
2. Compare several random starts or initialization seeds.
3. Inspect error histories and joint-projection residuals.
4. Test stability against support changes.
5. For multimode fits, inspect modal intensity fractions and mode mixing.
6. For multi-energy fits, inspect singular values or fitted spectra.
7. For dichroic fits, check `identifiable`, design rank, and fit residuals.
8. Treat phase unwrapping and complex-log branch choices carefully.

The physical projections improve stability only when their assumptions match
the experiment. A numerically excellent constrained fit can still be biased by
incorrect spectral data, polarization signs, response-product bounds, magnetic
response, or state labels.
