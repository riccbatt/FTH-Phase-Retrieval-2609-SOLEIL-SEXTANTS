# CUDA-JIT Phase Retrieval

This document describes the optional runtime-compiled GPU libraries:

- `library/phase_retrieval_core_jit.py`
- `library/phase_retrieval_universal_jit.py`

They are additional libraries. The ordinary CPU/CuPy phase-retrieval modules
remain available and retain their existing APIs.

## What JIT Can Accelerate

The phase-retrieval loop contains two different kinds of GPU work.

The global Fourier transforms are already executed by NVIDIA cuFFT through
CuPy. cuFFT is a highly optimized compiled vendor library. Replacing it with a
Python JIT FFT would not normally improve performance.

The operations around the FFTs are elementwise:

- impose measured Fourier amplitudes;
- preserve unconstrained beamstop pixels;
- apply support masks;
- apply HAPRE, ER, HIO, RAAR, and related projections;
- combine previous and current iterates.

Writing these operations as ordinary CuPy expressions can launch several CUDA
kernels and allocate several temporary arrays per iteration. The JIT library
uses `cupy.ElementwiseKernel`, compiled at runtime by NVIDIA NVRTC, to fuse each
logical operation into one CUDA kernel.

The accelerated iteration is therefore:

```text
fused amplitude constraint
        |
        v
      cuFFT
        |
        v
fused support projection
        |
        v
    inverse cuFFT
```

The largest gains are expected when:

- the reconstruction runs many iterations;
- arrays are large enough for GPU execution;
- the ordinary implementation is limited by temporary arrays or launch
  overhead;
- input data remain on one GPU during the complete iterative stage.

The FFTs may still dominate large reconstructions, so JIT fusion should not be
expected to multiply performance in every workload. Benchmark the actual image
size, algorithm, and GPU.

## Requirements

The JIT backend requires:

- an NVIDIA CUDA-capable GPU;
- a working NVIDIA driver;
- CuPy built for the installed CUDA runtime;
- NVRTC, normally included with the CUDA toolkit or compatible CuPy package.

The repository `environment.yml` currently specifies CUDA 11.8 and CuPy 12.
The actual CuPy package must match the CUDA installation on the reconstruction
machine.

The module can be imported without CuPy. In that case:

```python
from library import phase_retrieval_core_jit as jit

print(jit.cuda_jit_status())
```

returns an unavailable status instead of failing during import.

## Supported Algorithms

The following support projections have fused CUDA-JIT kernels:

| Algorithm | JIT support |
|---|---|
| `ER` | Yes |
| `SF` | Yes |
| `HAPRE` | Yes |
| `RAAR` | Yes |
| `HIOs` | Yes |
| `HIO` | Yes |
| `CHIO` | Yes |
| `HPR` | Yes |
| `OSS` | Ordinary-kernel fallback |

The first JIT version accelerates full-coherence iterations.

The following features currently use the ordinary universal kernel:

- Richardson-Lucy partial-coherence updates;
- active total-variation descent;
- OSS, because its Gaussian filtering adds another FFT-dependent operation.

This fallback is deliberate. Those paths need additional global transforms or
finite-difference operations and do not benefit from the current elementwise
fusion in the same way.

Set `fallback=False` to raise an error instead of using an ordinary path.

## Checking the GPU

```python
from library import phase_retrieval_core_jit as jit

status = jit.cuda_jit_status()
print(status)
```

When CUDA is available, the dictionary includes:

- `available`;
- `device_id`;
- `device_name`;
- `compute_capability`;
- `backend`.

## Warming Up NVRTC

The first call compiles each kernel and includes NVRTC compilation latency.
Compile before timing:

```python
from library import phase_retrieval_core_jit as jit

jit.warm_up_jit(
    shape=(2048, 2048),
    modes=["HAPRE", "ER"],
)
```

Kernels are cached in the process after their first compilation. CuPy may also
use its on-disk kernel cache.

Do not include the warmup call when measuring steady-state reconstruction
time.

## Direct Core Usage

```python
from library.phase_retrieval_core_jit import PhaseRtrv_core_jit

field, diffraction_error, support_error, gamma = PhaseRtrv_core_jit(
    diffract=measured_amplitude,
    mask=supportmask,
    mode="HAPRE",
    Nit=700,
    beta_zero=0.5,
    beta_mode="arctan",
    Phase=start_field,
    bsmask=invalid_pixel_mask,
    average_img=30,
    Fourier_last=True,
    fallback=False,
)
```

Its arguments match `PhaseRtrv_core`, with the additional `fallback` argument.

## Universal JIT Usage

The universal JIT front end accepts the same arguments and recipe as
`universal_phase_retrieval_algorithm`.

```python
from library import phase_retrieval_universal_jit as universal_jit

fields, components, bsmasks, errors = (
    universal_jit.universal_phase_retrieval_algorithm_jit(
        holograms,
        mask_pixel,
        supportmask,
        state_labels=state_labels,
        energy_labels=energy_labels,
        polarization_coefficients=polarizations,
        illumination_labels=illumination_labels,
        saturated_states=saturated_states,
        universal_recipe={
            "inner_mode": ["HAPRE", "ER"],
            "inner_Nit": [700, 50],
            "warmup_mode": ["HAPRE", "ER"],
            "warmup_Nit": [700, 50],
            "outer_iterations": 100,
            "projection_model": "physical_factorized",
        },
        start_fields=start_fields,
        fallback=False,
    )
)
```

The cross-observation physical, SVD, and spectral projections are unchanged.
The accelerated kernel replaces the repeated per-observation
Fourier/support-update stages.

## Pure Multi-Energy JIT Usage

```python
fields, components, bsmasks, errors = (
    universal_jit.multi_energy_phase_retrieval_algorithm_jit(
        holograms,
        mask_pixel,
        supportmask,
        multi_energy_recipe=recipe,
        start_fields=start_fields,
        fallback=False,
    )
)
```

## Metadata-Aware Physical JIT Usage

```python
fields, components, bsmasks, errors = (
    universal_jit.general_phase_retrieval_algorithm_jit(
        holograms,
        mask_pixel,
        supportmask,
        state_labels,
        energy_labels,
        polarizations,
        illumination_labels,
        saturated_states=saturated_states,
        general_recipe=recipe,
        start_fields=start_fields,
        fallback=False,
    )
)
```

## Fallback Behavior

The default is:

```python
fallback=True
```

This makes notebook code portable:

- supported algorithm plus CUDA: use JIT;
- no CUDA: use the ordinary universal kernel;
- unsupported JIT feature: use the ordinary universal kernel.

For controlled GPU production runs, prefer:

```python
fallback=False
```

This prevents an unnoticed CPU or ordinary-kernel fallback.

## Benchmarking Correctly

GPU execution is asynchronous. Synchronize before stopping a timer:

```python
import time
import cupy as cp

universal_jit.warm_up_jit((2048, 2048), ["HAPRE", "ER"])
cp.cuda.Stream.null.synchronize()

start = time.perf_counter()
result = universal_jit.universal_phase_retrieval_algorithm_jit(
    # reconstruction arguments
)
cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - start
```

Compare against the ordinary CuPy path on the same machine, using:

- identical start fields;
- identical recipes;
- identical array dtypes;
- identical final synchronization;
- multiple repetitions after warmup.

Monitor GPU memory as well as time. Kernel fusion is useful partly because it
reduces intermediate allocations.

## Precision

The JIT kernels currently use:

- `complex128` for complex fields;
- `float64` for amplitudes, masks, beta, and support values.

This matches the high-precision behavior of the existing NumPy-oriented code
more closely than a forced single-precision implementation.

A future performance mode could add `complex64` kernels. That can be faster on
many consumer GPUs and use half the memory, but it requires separate numerical
validation because iterative phase retrieval can amplify precision
differences.

## Current Hardware Validation

The repository tests verify:

- import without CUDA;
- clear availability reporting;
- ordinary-kernel fallback;
- strict no-fallback errors;
- explicit JIT-kernel injection into the universal driver;
- compatibility of the unchanged reconstruction APIs.

CUDA execution and performance must be benchmarked on an NVIDIA machine. A
machine without CuPy or an NVIDIA GPU cannot compile or execute the NVRTC
kernels, even though it can test the fallback and integration paths.

