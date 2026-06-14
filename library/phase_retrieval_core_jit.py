"""
CUDA-JIT phase-retrieval kernel using CuPy NVRTC elementwise kernels.

The FFTs are executed by cuFFT through CuPy. The operations surrounding the
FFTs are fused into runtime-compiled CUDA kernels, reducing temporary arrays
and kernel launches compared with composing many individual CuPy operations.

This module is optional. It imports cleanly without CuPy and can fall back to
the ordinary universal phase-retrieval kernel when CUDA JIT is unavailable or
when a feature is not yet accelerated.
"""

import logging

import numpy as np

from . import phase_retrieval_universal as universal


log = logging.getLogger(__name__)

try:
    import cupy as cp

    try:
        CUDA_JIT_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
        CUDA_JIT_IMPORT_ERROR = None
    except Exception as exc:
        CUDA_JIT_AVAILABLE = False
        CUDA_JIT_IMPORT_ERROR = exc
except ImportError as exc:
    cp = None
    CUDA_JIT_AVAILABLE = False
    CUDA_JIT_IMPORT_ERROR = exc


_AMPLITUDE_CONSTRAINT = None
_PROJECTION_KERNELS = {}


def cuda_jit_status():
    """Return availability and device information for the CUDA-JIT backend."""
    status = {
        "available": CUDA_JIT_AVAILABLE,
        "backend": "cupy_nvrtc",
        "import_error": (
            None
            if CUDA_JIT_IMPORT_ERROR is None
            else str(CUDA_JIT_IMPORT_ERROR)
        ),
    }
    if CUDA_JIT_AVAILABLE:
        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        status.update({
            "device_id": int(device.id),
            "device_name": name,
            "compute_capability": device.compute_capability,
        })
    return status


def _require_cuda_jit():
    """Raise an actionable error when CuPy NVRTC execution is unavailable."""
    if not CUDA_JIT_AVAILABLE:
        detail = (
            ""
            if CUDA_JIT_IMPORT_ERROR is None
            else f" Original error: {CUDA_JIT_IMPORT_ERROR}"
        )
        raise RuntimeError(
            "CUDA JIT is unavailable. Install a CuPy build matching the CUDA "
            f"runtime and use an NVIDIA GPU.{detail}"
        )


def _get_amplitude_constraint_kernel():
    """Create the fused measured-amplitude constraint on first use."""
    global _AMPLITUDE_CONSTRAINT
    _require_cuda_jit()
    if _AMPLITUDE_CONSTRAINT is None:
        _AMPLITUDE_CONSTRAINT = cp.ElementwiseKernel(
            "complex128 field, float64 amplitude, bool invalid",
            "complex128 constrained",
            """
            if (invalid) {
                constrained = field;
            } else {
                const double magnitude = abs(field);
                constrained = magnitude > 1.0e-30
                    ? amplitude * field / magnitude
                    : complex<double>(amplitude, 0.0);
            }
            """,
            "phase_retrieval_amplitude_constraint",
        )
    return _AMPLITUDE_CONSTRAINT


def _projection_operation(mode):
    """Return the CUDA expression for one support-space projection."""
    operations = {
        "ER": "projected = inverse * support;",
        "SF": "projected = inverse * (2.0 * support - 1.0);",
        "HAPRE": """
            projected = inverse
                + beta * (previous - 2.0 * inverse) * (1.0 - support);
        """,
        "RAAR": """
            const bool condition = real(2.0 * inverse - previous) < 0.0;
            projected = inverse
                + beta * (previous - 2.0 * inverse)
                * (1.0 - support) * condition;
        """,
        "HIOs": """
            projected = inverse
                + (1.0 - support)
                * (previous - (beta + 1.0) * inverse);
        """,
        "HIO": """
            const complex<double> correction =
                previous - (beta + 1.0) * inverse;
            projected = inverse
                + (1.0 - support) * correction
                + support * correction * (real(inverse) < 0.0);
        """,
        "CHIO": """
            const double alpha2 = 0.4;
            const bool condition1 =
                real(inverse - alpha2 * previous) >= 0.0;
            const bool condition2 =
                real(-inverse + alpha2 * previous) >= 0.0;
            const bool condition3 = real(inverse) >= 0.0;
            projected = previous - beta * inverse
                + support * condition1
                * (-previous + (beta + 1.0) * inverse)
                + condition2 * condition3
                * (beta - (1.0 - alpha2) / alpha2) * inverse;
        """,
        "HPR": """
            const complex<double> correction =
                previous - (beta + 1.0) * inverse;
            const bool condition =
                real(previous - (beta - 3.0) * inverse) > 0.0;
            projected = inverse
                + (1.0 - support) * correction
                + support * correction * condition;
        """,
    }
    try:
        return operations[mode]
    except KeyError as exc:
        raise ValueError(
            f"JIT projection does not support mode {mode!r}. "
            f"Supported modes are {sorted(operations)}."
        ) from exc


def _get_projection_kernel(mode):
    """Create one fused support projection for the selected algorithm."""
    _require_cuda_jit()
    if mode not in _PROJECTION_KERNELS:
        _PROJECTION_KERNELS[mode] = cp.ElementwiseKernel(
            (
                "complex128 inverse, complex128 previous, "
                "float64 support, float64 beta"
            ),
            "complex128 projected",
            _projection_operation(mode),
            f"phase_retrieval_projection_{mode.lower()}",
        )
    return _PROJECTION_KERNELS[mode]


def _error_db(guess, target):
    """Calculate the diffraction error on the GPU and return a Python float."""
    numerator = cp.sum(cp.abs(target - guess) ** 2)
    denominator = cp.sum(cp.abs(target) ** 2)
    if float(denominator.get()) == 0.0:
        return np.inf
    error = 10.0 * cp.log10(numerator / denominator)
    return float(error.get())


def _fallback_or_raise(fallback, reason, args, kwargs):
    """Run the ordinary kernel when requested, otherwise raise an error."""
    if fallback:
        log.info("CUDA-JIT fallback: %s", reason)
        return universal.PhaseRtrv_core(*args, **kwargs)
    raise RuntimeError(reason)


def PhaseRtrv_core_jit(
    diffract,
    mask,
    mode="ER",
    Nit=500,
    beta_zero=0.5,
    beta_mode="const",
    alpha_zero=0.0,
    alpha_mode="const",
    Phase=None,
    seed=False,
    plot_every=20,
    bsmask=None,
    real_object=False,
    average_img=10,
    Fourier_last=True,
    gamma=None,
    RL_freq=None,
    RL_it=0,
    TV_freq=2e9,
    fallback=True,
):
    """
    Run full-coherence phase retrieval with fused runtime-compiled CUDA kernels.

    ER, SF, HAPRE, RAAR, HIOs, HIO, CHIO, and HPR are accelerated. OSS,
    Richardson-Lucy partial coherence, and active TV descent currently use the
    ordinary kernel when ``fallback=True`` because they require additional FFT
    or finite-difference kernels and are not launch-bound in the same way.

    The signature matches ``phase_retrieval_universal.PhaseRtrv_core`` with
    the additional ``fallback`` control.
    """
    original_args = (diffract, mask)
    original_kwargs = {
        "mode": mode,
        "Nit": Nit,
        "beta_zero": beta_zero,
        "beta_mode": beta_mode,
        "alpha_zero": alpha_zero,
        "alpha_mode": alpha_mode,
        "Phase": Phase,
        "seed": seed,
        "plot_every": plot_every,
        "bsmask": bsmask,
        "real_object": real_object,
        "average_img": average_img,
        "Fourier_last": Fourier_last,
        "gamma": gamma,
        "RL_freq": RL_freq,
        "RL_it": RL_it,
        "TV_freq": TV_freq,
    }
    if not CUDA_JIT_AVAILABLE:
        return _fallback_or_raise(
            fallback,
            "CUDA JIT is unavailable.",
            original_args,
            original_kwargs,
        )
    if mode == "OSS":
        return _fallback_or_raise(
            fallback,
            "OSS is not implemented by the CUDA-JIT projection kernel.",
            original_args,
            original_kwargs,
        )
    if gamma is not None and RL_it > 0 and (RL_freq or Nit + 1) <= Nit:
        return _fallback_or_raise(
            fallback,
            "Richardson-Lucy partial coherence is not JIT accelerated.",
            original_args,
            original_kwargs,
        )

    beta = universal.make_beta_schedule(beta_mode, Nit, beta_zero)
    alpha = universal.make_alpha_schedule(alpha_mode, Nit, alpha_zero)
    if np.any(alpha > 0):
        return _fallback_or_raise(
            fallback,
            "Active total-variation descent is not JIT accelerated.",
            original_args,
            original_kwargs,
        )

    diffract = np.asarray(diffract)
    mask = np.asarray(mask)
    if diffract.ndim != 2 or mask.ndim != 2:
        raise ValueError("diffract and mask must be two-dimensional.")
    if diffract.shape != mask.shape:
        raise ValueError("diffract and mask must have the same shape.")
    if Nit <= 0 or average_img <= 0 or plot_every <= 0:
        raise ValueError("Nit, average_img, and plot_every must be positive.")

    if bsmask is None:
        bsmask = np.zeros_like(diffract, dtype=bool)
    else:
        bsmask = np.asarray(bsmask, dtype=bool)
        if bsmask.shape != diffract.shape:
            raise ValueError("bsmask must have the same shape as diffract.")

    if seed:
        np.random.seed(0)
    if Phase is None:
        phase = np.exp(2j * np.pi * np.random.rand(*diffract.shape))
        Phase = np.where(
            bsmask,
            phase,
            diffract * np.exp(1j * np.angle(phase)),
        )
    Phase = np.asarray(Phase)
    if Phase.shape != diffract.shape:
        raise ValueError("Phase must have the same shape as diffract.")

    # Move each static array to the GPU once and retain all iterates there.
    amplitude_gpu = cp.asarray(np.fft.fftshift(diffract), dtype=cp.float64)
    support_gpu = cp.asarray(np.fft.fftshift(mask), dtype=cp.float64)
    invalid_gpu = cp.asarray(np.fft.fftshift(bsmask), dtype=cp.bool_)
    field_gpu = cp.asarray(
        np.fft.fftshift(Phase),
        dtype=cp.complex128,
    )

    amplitude_constraint = _get_amplitude_constraint_kernel()
    projection = _get_projection_kernel(mode)
    field_gpu = amplitude_constraint(field_gpu, amplitude_gpu, invalid_gpu)
    previous_gpu = cp.fft.fft2(field_gpu)

    n_best = min(int(average_img), int(Nit))
    best_fields = cp.zeros(
        (n_best,) + diffract.shape,
        dtype=cp.complex128,
    )
    best_errors = np.full(n_best, np.inf, dtype=float)
    start_best_at = max(0, Nit - 2 * n_best)
    diffraction_errors = []
    observed_gpu = ~invalid_gpu

    for step in range(Nit):
        # One fused kernel applies the measured-amplitude constraint.
        field_gpu = amplitude_constraint(
            field_gpu,
            amplitude_gpu,
            invalid_gpu,
        )

        # cuFFT handles the global transforms; the support update is fused.
        inverse_gpu = cp.fft.fft2(field_gpu)
        inverse_gpu = projection(
            inverse_gpu,
            previous_gpu,
            support_gpu,
            float(beta[step]),
        )
        previous_gpu = inverse_gpu
        field_gpu = cp.fft.ifft2(inverse_gpu)

        if (
            step <= 2
            or step % plot_every == 0
            or step >= start_best_at
        ):
            error = _error_db(
                cp.abs(field_gpu) * observed_gpu,
                amplitude_gpu * observed_gpu,
            )
            diffraction_errors.append(error)
            if step >= start_best_at:
                worst = int(np.argmax(best_errors))
                if error < best_errors[worst]:
                    best_errors[worst] = error
                    best_fields[worst] = field_gpu

    field_gpu = cp.mean(best_fields, axis=0)
    if Fourier_last:
        field_gpu = amplitude_constraint(
            field_gpu,
            amplitude_gpu,
            invalid_gpu,
        )

    result = np.fft.ifftshift(cp.asnumpy(field_gpu))
    return result, diffraction_errors, [], None


def warm_up_jit(shape=(256, 256), modes=None):
    """
    Compile the CUDA kernels before a timed reconstruction.

    NVRTC compilation happens once per process and kernel signature. Calling
    this function removes compilation latency from subsequent benchmarks.
    """
    _require_cuda_jit()
    if modes is None:
        modes = ["ER", "HAPRE", "RAAR", "HIO"]
    field = cp.ones(shape, dtype=cp.complex128)
    amplitude = cp.ones(shape, dtype=cp.float64)
    invalid = cp.zeros(shape, dtype=cp.bool_)
    support = cp.ones(shape, dtype=cp.float64)
    _get_amplitude_constraint_kernel()(field, amplitude, invalid)
    for mode in modes:
        _get_projection_kernel(mode)(
            field,
            field,
            support,
            0.5,
        )
    cp.cuda.Stream.null.synchronize()

