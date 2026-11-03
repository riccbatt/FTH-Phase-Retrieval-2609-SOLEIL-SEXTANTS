"""
Universal joint phase retrieval for energy, polarization, state, and beam scans.

Each observation ``a`` is described by four pieces of metadata:

``energy_labels[a]``
    Selects the photon energy and therefore the charge and magnetic response.
``polarization_coefficients[a]``
    Usually +1 or -1. It changes the sign of the magnetic interaction.
``state_labels[a]``
    Selects the real reduced-magnetization map ``mz_s(r)``.
``illumination_labels[a]``
    Selects the illumination field ``C_m(r)`` without changing the sample.

Writing ``c_m = log(C_m)``, the physical projection uses log-object space,

    L_a(r) = c_m(a)(r) + t(r)[q_c(E_a) + p_a q_m(E_a) mz_s(a)(r)],

Here t(r) is a dimensionless relative thickness and the fitted q values are
log-transmission coefficients, NOT refractive indices. With physical thickness
d(r)=d0*t(r) and transmission exp(-1j*k*(n-1)*d), define
q_c(E)=-1j*k(E)*d0*(n_c(E)-1), q_m(E)=-1j*k(E)*d0*n_m(E),
where k(E)=2*pi/lambda(E). This absorbs k*d0 exactly once. Dropping the vacuum
reference replaces n_c-1 by n_c and moves the vacuum phase into the common field.

with ``-1 <= mz <= 1``. By default ``t=1``; optional material inputs set known
vacuum holes to ``t=0``, supply relative thickness, or fit a shared nonnegative
relative thickness map. The scalar spectral responses absorb the reference
thickness and wave number. Optional saturated states fix an entire ``mz`` map
to +1 or -1 within material. Absolute thickness and refractive index remain
coupled unless external information fixes their scale.

The module contains its own phase-retrieval kernel, schedules, multi-energy
projectors, ordered observation stages, and single/multimode kernels.
The public paper workflow is a single recipe-driven loop that alternates
Fourier/support updates with the physical projection when enabled.
"""
import logging
from contextlib import nullcontext
log = logging.getLogger(__name__)

from functools import partial
import os
import time
import numpy as np
try:
    from . import phase_retrieval_geometry as geometry
except ImportError:
    import phase_retrieval_geometry as geometry

from numpy.typing import ArrayLike

__all__ = [
    "default_general_phase_retrieval_recipe",
    "default_universal_phase_retrieval_recipe",
    "general_phase_retrieval_algorithm",
    "universal_phase_retrieval_algorithm",
    "project_log_objects_physical",
    "largest_support_component",
    "PhaseRtrv_core_multimode",
    "PhaseRtrv_core",
    "PhaseRtrv_core_single",
]

from scipy import stats

from .magnitude_constraints import _apply_modal_fourier_constraint, _modal_convolved_intensities
from .real_space_constraints import _expand_support_about_center, _mode_supports, _project_support_space
from .phase_retrieval_physical import (
    _binary_magnetization_values,
    _component_centered_modal_supports,
    _constrain_magnetization_trend,
    _constrain_response_values,
    _normalize_metadata,
    _normalize_range,
    _normalize_saturated_states,
    _observation_weights,
    _ordered_unique,
    _physical_mode_groups,
    _projection_focus_transform,
    _project_grouped_physical_modes,
    _project_nonphysical_modes_common,
    _project_physical_modes,
    _reference_illumination_correction,
    _snap_magnetization,
    _two_domain_distribution,
    _weighted_magnetization_median,
    largest_support_component,
    project_fourier_fields_general,
    project_log_objects_physical,
)

try:
    from . import kramers_kronig as kk
except ImportError:
    import kramers_kronig as kk


_EV_TO_WAVENUMBER_PER_METRE = 5.067730716156395e6


#############################################################
#       GPU handling
# ############################################################

try:
    import cupy as cp

    GPU = cp.is_available()
    log.info("CUDA GPU available.")
except ImportError:
    log.warning(
        "Could not import cupy module (is it installed?). "
        "Proceeding with CPU support only."
    )
    GPU = False
except Exception as ex:
    log.warning(
        f"Error determining GPU availability: {ex}. "
        "Proceeding with CPU support only."
    )
    GPU = False

# Use cupyx or scipy fft functions
if GPU:
    import cupy as xp
    from cupyx.scipy.fft import fft2, ifft2    
else:
    import numpy as xp
    import scipy.fft as fft

    # Respect scipy.fft.set_workers in each observation worker.
    fft2 = fft.fft2
    ifft2 = fft.ifft2


def to_numpy(array, xp):
    """
    Convert xp array to NumPy safely.
    """
    if xp.__name__ == "cupy":
        return array.get()
    return array


# -----------------------------------------------------------------------------
# Public API and modular solver layout
# -----------------------------------------------------------------------------
#
# The implementation is intentionally kept in one file for a single paper-ready
# algorithmic core, but the code is segmented into thematic blocks so that the
# reader can follow the flow without chasing a long list of unrelated helpers.
# -----------------------------------------------------------------------------

# Ordered observation stages and modal kernel share the universal implementation.




def _as_modes(arr, Nmodes, shape_2d, name, dtype=None):
    """
    Convert a 2D or 3D array into modal shape ``(Nmodes, nx, ny)``.

    Accepted inputs are:
      - ``(nx, ny)``: copied into all modes
      - ``(1, nx, ny)``: copied into all modes
      - ``(Nmodes, nx, ny)``: used directly

    Context:
    Internal shape contract: kernels work with an explicit leading mode axis even when the
    public single-mode API supplies a 2D image.
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        if arr.shape != shape_2d:
            raise ValueError(f"{name} must have shape {shape_2d}.")
        out = np.repeat(arr[None, :, :], Nmodes, axis=0)

    elif arr.ndim == 3:
        if arr.shape[1:] != shape_2d:
            raise ValueError(
                f"{name} has incompatible spatial shape {arr.shape[1:]}; "
                f"expected {shape_2d}."
            )
        if arr.shape[0] == Nmodes:
            out = arr.copy()
        elif arr.shape[0] == 1:
            out = np.repeat(arr, Nmodes, axis=0)
        else:
            raise ValueError(
                f"{name} first dimension must be 1 or Nmodes={Nmodes}."
            )
    else:
        raise ValueError(f"{name} must be 2D or 3D.")

    if dtype is not None:
        out = out.astype(dtype, copy=False)

    return out

def _normalize_mode_factors(mode_factors, *, default_nmodes=1):
    """Normalize recipe mode factors into a finite, nonnegative list."""
    if mode_factors is None:
        return [1.0] * int(default_nmodes)

    if isinstance(mode_factors, np.ndarray):
        raw_values = list(mode_factors.ravel())
    elif isinstance(mode_factors, (list, tuple)):
        raw_values = list(mode_factors)
    elif isinstance(mode_factors, (int, float, np.number)) and not isinstance(mode_factors, bool):
        raw_values = [mode_factors]
    else:
        try:
            raw_values = list(mode_factors)
        except TypeError as exc:
            raise ValueError(
                "modes must be a nonempty 1D list of nonnegative support factors."
            ) from exc

    if not raw_values or any(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise ValueError("modes entries must be finite nonnegative support factors.")

    try:
        values = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("modes entries must be finite nonnegative support factors.") from exc

    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("modes entries must be finite nonnegative support factors.")
    if float(values[0]) != 1.0:
        raise ValueError("The first physical mode must use support factor 1.")

    return [float(value) for value in values]


def _maybe_squeeze_modes(arr, Nmodes):
    """Return 2D output for ``Nmodes == 1`` and 3D output otherwise."""
    if arr is None:
        return None
    if Nmodes == 1 and np.asarray(arr).ndim == 3:
        return arr[0]
    return arr


def _refine_modes_gradient(
    phase,
    measured_amplitude,
    supportmask,
    bsmask,
    *,
    nmodes,
    image_shape,
    Nit,
    learning_rate,
    support_weight,
    Fourier_last,
):
    raise ValueError("Gradient refinement is outside the paper implementation")






def PhaseRtrv_core_multimode(
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
    Nmodes=1,
):
    """
    Unified single-mode / multimode full- and partial-coherence retrieval.

    ``diffract`` is the measured 2D diffraction amplitude. For ``Nmodes > 1``,
    ``Phase``, ``mask``, and ``gamma`` may be 3D arrays of shape
    ``(Nmodes, nx, ny)``. The Fourier constraint is then applied to the
    incoherent modal intensity sum:

        ``diffract**2 ≈ sum_m |mode_m|**2``

    or, with RL enabled,

        ``diffract**2 ≈ sum_m convolution(|mode_m|**2, gamma_m)``.

    For ``Nmodes == 1``, outputs are squeezed back to 2D to preserve the
    previous API.

    Context:
    Single-observation multimode kernel. All modes contribute to the detector intensity, but
    each has its own support. State-independent secondary-mode projection is a
    responsibility of the universal driver.
    """
    diffract = np.asarray(diffract)
    mask = np.asarray(mask)
    if isinstance(Nmodes, bool) or not isinstance(Nmodes, (int, np.integer)):
        raise ValueError("Nmodes must be a positive integer.")
    Nmodes = int(Nmodes)
    if Nmodes <= 0:
        raise ValueError("Nmodes must be a positive integer.")
    if diffract.ndim != 2:
        raise ValueError("diffract must be a 2D measured diffraction amplitude.")
    if Nit <= 0:
        raise ValueError("Nit must be > 0.")
    if average_img <= 0:
        raise ValueError("average_img must be > 0.")
    if plot_every <= 0:
        raise ValueError("plot_every must be > 0.")
    if RL_freq is not None and RL_freq <= 0:
        raise ValueError("RL_freq must be > 0.")
    if RL_it < 0:
        raise ValueError("RL_it must be >= 0.")
    if TV_freq <= 0:
        raise ValueError("TV_freq must be > 0.")

    l, n = diffract.shape
    shape_2d = (l, n)

    mask_modes = _as_modes(mask, Nmodes, shape_2d, "mask")

    if bsmask is None:
        bsmask = np.zeros(shape_2d, dtype=np.float32)
    else:
        bsmask = np.asarray(bsmask)
        if bsmask.shape != shape_2d:
            raise ValueError("bsmask must have same 2D shape as diffract.")

    proj_fn = PROJECTIONS.get(mode)
    if proj_fn is None:
        raise ValueError(f"Invalid mode '{mode}'. Allowed: {sorted(PROJECTIONS)}")

    Beta = make_beta_schedule(beta_mode, Nit, beta_zero)
    Alpha = make_alpha_schedule(alpha_mode, Nit, alpha_zero)

    if seed:
        np.random.seed(0)

    if Phase is None:
        phase_random = np.exp(
            1j * np.random.rand(Nmodes, l, n) * np.pi * 2
        )
        Phase = (
            (1 - bsmask)[None, :, :]
            * diffract[None, :, :]
            / np.sqrt(Nmodes)
            * phase_random
            + phase_random * bsmask[None, :, :]
        )

    phase_modes = _as_modes(Phase, Nmodes, shape_2d, "Phase", dtype=np.complex128)

    if RL_freq is None:
        RL_freq = Nit + 1

    use_RL = gamma is not None and RL_it > 0 and RL_freq <= Nit

    if use_RL:
        gamma_modes = _as_modes(gamma, Nmodes, shape_2d, "gamma", dtype=np.complex128)
        gamma_modes = np.fft.fftshift(gamma_modes, axes=(-2, -1))
    else:
        gamma_modes = None

    # Shift to corner convention. Modes are shifted only over the spatial axes.
    bsmask = np.fft.fftshift(bsmask)
    mask_modes = np.fft.fftshift(mask_modes, axes=(-2, -1))
    diffract = np.fft.fftshift(diffract)
    guess = np.fft.fftshift(phase_modes, axes=(-2, -1))

    BSmask_cp = xp.asarray(bsmask).astype(xp.bool_)
    obs = ~BSmask_cp

    guess_cp = xp.asarray(guess)
    mask_cp = xp.asarray(mask_modes)
    diffract_cp = xp.asarray(diffract)

    if use_RL:
        gamma_cp = xp.asarray(gamma_modes)
        for m in range(Nmodes):
            gamma_sum = xp.sum(gamma_cp[m])
            if not bool(xp.isfinite(gamma_sum)) or bool(xp.abs(gamma_sum) == 0):
                raise ValueError(
                    f"gamma mode {m} must have a finite, non-zero sum."
                )
            gamma_cp[m] /= gamma_sum
    else:
        gamma_cp = None

    convolved = _modal_convolved_intensities(guess_cp, gamma_cp, Nmodes)
    prev = xp.zeros_like(guess_cp, dtype=xp.complex128)
    initial_constrained = _apply_modal_fourier_constraint(
        guess_cp,
        convolved,
        diffract_cp,
        obs,
        BSmask_cp,
    )
    for m in range(Nmodes):
        prev[m] = fft2(initial_constrained[m])

    if not use_RL:
        guess_cp = initial_constrained
        convolved = _modal_convolved_intensities(
            guess_cp,
            gamma_cp,
            Nmodes,
        )

    Error_diffr_list = []
    Error_supp_list = []

    n_best = min(average_img, Nit)
    Best_guess = xp.zeros((n_best, Nmodes, l, n), dtype=xp.complex128)
    Best_error = xp.full((n_best,), xp.inf, dtype=xp.float64)

    if use_RL:
        Best_gamma = xp.zeros((n_best, Nmodes, l, n), dtype=xp.complex128)

    start_best_at = max(0, Nit - n_best * 2)

    for s in range(Nit):
        beta = float(Beta[s])
        alpha = float(Alpha[s])

        # Recipe-driven loop in a single pass:
        #   1) Fourier constraint: match the measured detector magnitude
        #   2) support projection: impose the support in the selected mode
        #   3) optional physical projection: follow the recipe-defined projection
        #   4) optional RL coherence refinement: adapt the gamma kernel
        #   5) keep the iteration state in real space before the next pass
        convolved = _modal_convolved_intensities(
            guess_cp,
            gamma_cp,
            Nmodes,
        )
        guess_cp = _apply_modal_fourier_constraint(
            guess_cp,
            convolved,
            diffract_cp,
            obs,
            BSmask_cp,
        )

        new_guess = xp.zeros_like(guess_cp)

        # Constraint 2: project the current guess into the real-space support.
        for m in range(Nmodes):
            inv = fft2(guess_cp[m])
            inv = _project_support_space(
                inv,
                prev[m],
                mask_cp[m],
                proj_fn,
                beta,
                s,
                Nit,
                alpha if (s % TV_freq == 0) else 0.0,
            )
            prev[m] = inv.copy()
            new_guess[m] = ifft2(inv)

        # Constraint 3: if enabled, update the coherence model on top of the
        # same Fourier/support cycle. This part keeps the same numerical update
        # as the original implementation but now reads as a separate constraint
        # layer rather than a hidden side effect.
        if use_RL and s > 2 and (s % RL_freq == 0):
            for m in range(Nmodes):
                convolved_new_m = ifft2(
                    fft2(xp.abs(new_guess[m]) ** 2) * fft2(gamma_cp[m])
                )
                Idelta = 2 * xp.abs(new_guess[m]) ** 2 - xp.abs(guess_cp[m]) ** 2
                I_exp = obs * (xp.abs(diffract_cp) ** 2) + convolved_new_m * BSmask_cp
                gamma_cp[m] = RL(
                    Idelta=Idelta,
                    Iexp=I_exp,
                    gamma_cp=gamma_cp[m],
                    RL_it=RL_it,
                )

        guess_cp = new_guess
        convolved = _modal_convolved_intensities(
            guess_cp,
            gamma_cp,
            Nmodes,
        )

        total_intensity = xp.sum(convolved, axis=0)
        if use_RL:
            err_guess = obs * total_intensity
            err_target = obs * xp.abs(diffract_cp) ** 2
        else:
            err_guess = obs * xp.sqrt(total_intensity)
            err_target = obs * diffract_cp

        # Record errors and retain the best late-iteration candidates.
        if s <= 2 or (s % plot_every == 0) or (s >= start_best_at):
            err = Error_diffract_cp(err_guess, err_target)
            Error_diffr_list.append(err)

            if s >= start_best_at:
                j = int(xp.argmax(Best_error).item())
                if err < Best_error[j]:
                    Best_error[j] = err
                    Best_guess[j, :, :, :] = guess_cp

                    if use_RL:
                        Best_gamma[j, :, :, :] = gamma_cp

    guess_cp = xp.mean(Best_guess, axis=0)

    if use_RL:
        gamma_cp = xp.mean(Best_gamma, axis=0)

    if Fourier_last:
        convolved = _modal_convolved_intensities(
            guess_cp,
            gamma_cp,
            Nmodes,
        )
        guess_cp = _apply_modal_fourier_constraint(
            guess_cp,
            convolved,
            diffract_cp,
            obs,
            BSmask_cp,
        )

    guess = to_numpy(guess_cp, xp)
    guess = np.fft.ifftshift(guess, axes=(-2, -1))
    guess = _maybe_squeeze_modes(guess, Nmodes)

    if use_RL:
        gamma = to_numpy(gamma_cp, xp)
        gamma = np.fft.ifftshift(gamma, axes=(-2, -1))
        gamma = _maybe_squeeze_modes(gamma, Nmodes)
    else:
        gamma = None

    return guess, Error_diffr_list, Error_supp_list, gamma

def PhaseRtrv_core(*args, Nmodes=1, **kwargs):
    """
    Dispatch to the fast single-mode or full multimode phase-retrieval kernel.

    ``Nmodes == 1`` uses :func:`PhaseRtrv_core_single`, the 2D implementation
    from the standard core. ``Nmodes > 1`` uses
    :func:`PhaseRtrv_core_multimode`, where the Fourier constraint is applied to
    the summed modal intensity.

    Context:
    Public kernel dispatcher: select single/multimode implementation through Nmodes while
    preserving the shared call contract expected by observation workers.
    """
    if isinstance(Nmodes, bool) or not isinstance(Nmodes, (int, np.integer)):
        raise ValueError("Nmodes must be a positive integer.")
    Nmodes = int(Nmodes)
    if Nmodes <= 0:
        raise ValueError("Nmodes must be a positive integer.")
    if Nmodes == 1:
        return PhaseRtrv_core_single(*args, **kwargs)
    return PhaseRtrv_core_multimode(*args, Nmodes=Nmodes, **kwargs)






# -----------------------------------------------------------------------------
# Solver helpers: modal geometry, schedules, and projection operators
# -----------------------------------------------------------------------------

# Beta schedule (CPU-side)
def _beta_const(Nit, beta_zero):
    """Return a constant beta schedule of length ``Nit``."""
    return np.full(Nit, beta_zero, dtype=np.float64)


def _beta_arctan(Nit, beta_zero):
    """Return the legacy arctangent beta schedule."""
    step = np.arange(Nit, dtype=np.float64)
    return beta_zero + (
        0.5 - np.arctan((step - min(Nit / 2, 700)) / (0.15 * Nit)) / np.pi
    ) * (0.98 - beta_zero)


def _beta_smoothstep(Nit, beta_zero):
    """Return a smoothstep beta schedule that decays to zero near the end."""
    step = np.arange(Nit, dtype=np.float64)
    start = Nit // 50
    end = Nit - Nit // 10
    denom = max(end - start, 1.0)
    y = (step - start) / denom
    beta = 1 - (1 - beta_zero) * (6 * y**5 - 15 * y**4 + 10 * y**3)
    beta[:start] = 1
    beta[end:] = 0
    return beta


def _beta_sigmoid(Nit, beta_zero):
    """Return a sigmoid-shaped beta schedule."""
    step = np.arange(Nit, dtype=np.float64)
    x0 = Nit // 20
    alpha = 1 / (Nit * 0.15)
    return 1 - (1 - beta_zero) / (1 + np.exp(-(step - x0) * alpha))


def _beta_exp(Nit, beta_zero):
    """Return an exponential beta schedule."""
    step = np.arange(Nit, dtype=np.float64)
    return beta_zero + (1 - beta_zero) * (1 - np.exp(-((step / 7) ** 3)))


def _beta_linear_to_beta_zero(Nit, beta_zero):
    """Return a linear schedule from 1 to ``beta_zero``."""
    step = np.arange(Nit, dtype=np.float64)
    return 1 + (beta_zero - 1) / Nit * step


def _beta_linear_to_1(Nit, beta_zero):
    """Return a linear schedule from ``beta_zero`` to 1."""
    step = np.arange(Nit, dtype=np.float64)
    return beta_zero + (1 - beta_zero) / Nit * step

def _beta_linear_to_0(Nit, beta_zero):
    """Return a linear schedule from ``beta_zero`` to 0."""
    return np.linspace(beta_zero, 0, Nit, dtype=np.float64)


def _beta_steps(Nit, beta_zero):
    """Return a ten-step schedule from ``beta_zero`` to 0."""
    n_steps = 10
    step_ids = np.floor(np.arange(Nit) * n_steps / Nit)
    beta = beta_zero * (1 - step_ids / (n_steps - 1))
    return np.clip(beta, 0, beta_zero).astype(np.float64)


BETA_SCHEDULES = {
    "const": _beta_const,
    "arctan": _beta_arctan,
    "smoothstep": _beta_smoothstep,
    "sigmoid": _beta_sigmoid,
    "exp": _beta_exp,
    "linear_to_beta_zero": _beta_linear_to_beta_zero,
    "linear_to_1": _beta_linear_to_1,
    "linear_to_0": _beta_linear_to_0,
    "steps": _beta_steps,
}
ALPHA_SCHEDULES = BETA_SCHEDULES.copy()


def make_beta_schedule(beta_mode, Nit, beta_zero):
    """
    Create beta schedule of length Nit.

    beta_mode can be:
      - a numpy array of length Nit (returned as float64 view/copy),
      - or a string key in BETA_SCHEDULES.
    -------
    author: CK 2026
    """
    if Nit <= 0:
        raise ValueError("Nit must be > 0.")

    if isinstance(beta_mode, np.ndarray):
        if beta_mode.shape[0] != Nit:
            raise ValueError("If beta_mode is an array, it must have length Nit.")
        return beta_mode.astype(np.float64, copy=False)

    try:
        fn = BETA_SCHEDULES[beta_mode]
    except KeyError:
        raise ValueError(
            f"Unknown beta_mode '{beta_mode}'. Allowed: {sorted(BETA_SCHEDULES)}"
        )

    beta = fn(Nit, beta_zero)
    if beta.shape[0] != Nit:
        raise RuntimeError("Beta schedule returned wrong length.")
    return beta

def make_alpha_schedule(alpha_mode, Nit, alpha_zero):
    """Create an alpha schedule using the same schedule definitions as beta."""
    return make_beta_schedule(alpha_mode, Nit, alpha_zero)




# Real-space projection steps
def _proj_ER(inv, prev, mask, beta, step_idx, Nit):
    """Error-reduction projection: keep only values inside the support."""
    return inv * mask


def _proj_SF(inv, prev, mask, beta, step_idx, Nit):
    """Solvent-flipping projection."""
    return inv * (2 * mask - 1)


def _proj_hapre(inv, prev, mask, beta, step_idx, Nit):
    """HAPRE projection step."""
    return inv + beta * (prev - 2 * inv) * (1 - mask)


def _proj_RAAR(inv, prev, mask, beta, step_idx, Nit):
    """Relaxed averaged alternating reflections projection."""
    return inv + beta * (prev - 2 * inv) * (1 - mask) * (2 * inv - prev < 0)


def _proj_HIOs(inv, prev, mask, beta, step_idx, Nit):
    """Simplified hybrid-input-output projection."""
    return inv + (1 - mask) * (prev - (beta + 1) * inv)


def _proj_HIO(inv, prev, mask, beta, step_idx, Nit):
    """Hybrid-input-output projection with positivity condition."""
    return (
        inv
        + (1 - mask) * (prev - (beta + 1) * inv)
        + mask * (prev - (beta + 1) * inv) * (xp.real(inv) < 0)
    )


def _proj_OSS(inv, prev, mask, beta, step_idx, Nit):
    """Oversampling-smoothness projection with iteration-dependent Gaussian filtering."""
    inv2 = (
        inv
        + (1 - mask) * (prev - (beta + 1) * inv)
        + mask * (prev - (beta + 1) * inv) * (xp.real(inv) < 0)
    )
    l = inv2.shape[0]
    alpha2 = l - (l - 1 / l) * xp.floor(step_idx / Nit * 10) / 10
    smoothed = ifft2(W(inv2.shape[0], inv2.shape[1], alpha2) * fft2(inv2))
    return inv2 * mask + (1 - mask) * smoothed


def _proj_CHIO(inv, prev, mask, beta, step_idx, Nit):
    """Constrained hybrid-input-output projection using a fixed alpha2 parameter."""
    alpha2 = 0.4
    return (
        (prev - beta * inv)
        + mask * (xp.real(inv - alpha2 * prev) >= 0) * (-prev + (beta + 1) * inv)
        + (xp.real(-inv + alpha2 * prev) >= 0)
        * (xp.real(inv) >= 0)
        * ((beta - (1 - alpha2) / alpha2) * inv)
    )


def _proj_HPR(inv, prev, mask, beta, step_idx, Nit):
    """Hybrid projection-reflection update."""
    return (
        inv
        + (1 - mask) * (prev - (beta + 1) * inv)
        + mask * (prev - (beta + 1) * inv) * (xp.real(prev - (beta - 3) * inv) > 0)
    )


PROJECTIONS = {
    "ER": _proj_ER,
    "SF": _proj_SF,
    "HAPRE": _proj_hapre,
    "RAAR": _proj_RAAR,
    "HIOs": _proj_HIOs,
    "HIO": _proj_HIO,
    "OSS": _proj_OSS,
    "CHIO": _proj_CHIO,
    "HPR": _proj_HPR,
}

# ----------------------------
# Other Subprocesses
# ----------------------------
def RL(Idelta, Iexp, gamma_cp, RL_it):
    """
    Richardson–Lucy update loop (CuPy).

    Parameters
    ----------
    Idelta : cupy.ndarray
        Intensity delta term (2D).
    Iexp : cupy.ndarray
        Expected intensity / measurement term (2D).
    gamma_cp : cupy.ndarray
        Current PSF / mutual optical intensity estimate (2D).
    RL_it : int
        Number of RL iterations.

    Returns
    -------
    gamma_cp : cupy.ndarray
        Updated and normalized gamma estimate.
    --------
    author: RB 2020
    """
    # Precompute FFTs of Idelta and its flipped version
    Id = fft2(Idelta)
    Id_flip = fft2(Idelta[::-1, ::-1])

    for _ in range(RL_it):
        # update
        gamma_cp = gamma_cp * ifft2(Id_flip * fft2(Iexp / (ifft2(Id * fft2(gamma_cp)))))

        # normalize (once per iteration)
        gamma_cp /= xp.nansum(gamma_cp)

    return gamma_cp

# -----------------------------------------------------------------------------
# Main kernel implementations
# -----------------------------------------------------------------------------


def _enforce_single_mode_fourier_constraint(guess_cp, diffract_cp, gamma_cp, obs, BSmask_cp, use_RL):
    """Apply the measured Fourier magnitude while preserving the existing RL branch."""
    if use_RL:
        convolved = ifft2(fft2(xp.abs(guess_cp) ** 2) * fft2(gamma_cp))
        factor = diffract_cp / xp.sqrt(convolved)
        guess_cp = guess_cp.copy()
        guess_cp[obs] *= factor[obs]
        return guess_cp, convolved

    return xp.where(
        BSmask_cp,
        guess_cp,
        diffract_cp * xp.exp(1j * xp.angle(guess_cp)),
    ), None


def _update_partial_coherence_kernel(guess_cp, diffract_cp, gamma_cp, obs, BSmask_cp, RL_freq, RL_it, s):
    """Update the RL coherence kernel using the same logic as the legacy implementation."""
    if s <= RL_freq or (s % RL_freq != 0):
        return guess_cp, gamma_cp

    convolved_new = ifft2(fft2(xp.abs(guess_cp) ** 2) * fft2(gamma_cp))
    Idelta = 2 * xp.abs(guess_cp) ** 2 - xp.abs(guess_cp) ** 2
    I_exp = obs * (xp.abs(diffract_cp) ** 2) + convolved_new * BSmask_cp
    gamma_cp = RL(
        Idelta=Idelta,
        Iexp=I_exp,
        gamma_cp=gamma_cp,
        RL_it=RL_it,
    )
    return guess_cp, gamma_cp


def _track_best_candidate(best_guess, best_error, field, error_value, best_index, use_RL, gamma_cp):
    """Keep the best late-iteration candidate using the same selection rule as the legacy loop."""
    index = int(xp.argmax(best_error).item())
    if error_value < best_error[index]:
        best_error[index] = error_value
        best_guess[index, :, :] = field
        if use_RL:
            best_guess[index, :, :] = field
    return best_error, best_guess


def PhaseRtrv_core_single(
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
):
    """
    Unified full/partial-coherence phase-retrieval kernel.

    The function applies the Fourier constraint, transforms to real space,
    applies an optional TV descent controlled by the alpha schedule, applies the
    selected real-space projection, and tracks the best reconstructions near the
    end of the run. Richardson-Lucy partial-coherence updates are enabled only
    when ``gamma`` is provided, ``RL_it > 0``, and ``RL_freq <= Nit``. Otherwise
    the same loop behaves as a full-coherence reconstruction.
    """
    diffract = np.asarray(diffract)
    mask = np.asarray(mask)

    if diffract.ndim != 2 or mask.ndim != 2:
        raise ValueError("diffract and mask must be 2D arrays.")
    if diffract.shape != mask.shape:
        raise ValueError("diffract and mask must have the same shape.")
    if Nit <= 0:
        raise ValueError("Nit must be > 0.")
    if average_img <= 0:
        raise ValueError("average_img must be > 0.")
    if plot_every <= 0:
        raise ValueError("plot_every must be > 0.")
    if RL_freq is not None and RL_freq <= 0:
        raise ValueError("RL_freq must be > 0.")
    if RL_it < 0:
        raise ValueError("RL_it must be >= 0.")
    if TV_freq <= 0:
        raise ValueError("TV_freq must be > 0.")

    l, n = diffract.shape

    if bsmask is None:
        bsmask = np.zeros((l, n), dtype=np.float32)
    else:
        bsmask = np.asarray(bsmask)
        if bsmask.shape != (l, n):
            raise ValueError("bsmask must have same shape as diffract.")

    proj_fn = PROJECTIONS.get(mode)
    if proj_fn is None:
        raise ValueError(f"Invalid mode '{mode}'. Allowed: {sorted(PROJECTIONS)}")

    Beta = make_beta_schedule(beta_mode, Nit, beta_zero)
    Alpha = make_alpha_schedule(alpha_mode, Nit, alpha_zero)

    if seed:
        np.random.seed(0)

    if Phase is None:
        Phase = np.exp(1j * np.random.rand(l, n) * np.pi * 2)
        Phase = (1 - bsmask) * diffract * np.exp(1j * np.angle(Phase)) + Phase * bsmask

    Phase = np.asarray(Phase)
    if Phase.shape != (l, n):
        raise ValueError("Phase must have same shape as diffract.")

    if RL_freq is None:
        RL_freq = Nit + 1

    use_RL = gamma is not None and RL_it > 0 and RL_freq <= Nit

    # Shift to corner convention
    bsmask = np.fft.fftshift(bsmask)
    mask = np.fft.fftshift(mask)
    diffract = np.fft.fftshift(diffract)
    guess = np.fft.fftshift(np.array(Phase, copy=True))

    BSmask_cp = xp.asarray(bsmask).astype(xp.bool_)
    obs = ~BSmask_cp

    guess_cp = xp.asarray(guess)
    mask_cp = xp.asarray(mask)
    diffract_cp = xp.asarray(diffract)

    if use_RL:
        gamma = np.fft.fftshift(np.asarray(gamma))
        gamma_cp = xp.asarray(gamma)
        gamma_sum = xp.sum(gamma_cp)
        if not bool(xp.isfinite(gamma_sum)) or bool(xp.abs(gamma_sum) == 0):
            raise ValueError("gamma must have a finite, non-zero sum.")
        gamma_cp /= gamma_sum

        convolved = ifft2(fft2(xp.abs(guess_cp) ** 2) * fft2(gamma_cp))

        prev = fft2(
            obs * diffract_cp / xp.sqrt(convolved) * guess_cp
            + guess_cp * BSmask_cp
        )
    else:
        gamma_cp = None
        convolved = None

        guess_cp = xp.where(
            BSmask_cp,
            guess_cp,
            diffract_cp * xp.exp(1j * xp.angle(guess_cp)),
        )

        prev = fft2(guess_cp)

    Error_diffr_list = []
    Error_supp_list = []

    n_best = min(average_img, Nit)
    Best_guess = xp.zeros((n_best, l, n), dtype=xp.complex64)
    Best_error = xp.full((n_best,), xp.inf, dtype=xp.float64)

    if use_RL:
        Best_gamma = xp.zeros((n_best, l, n), dtype=xp.complex64)

    start_best_at = max(0, Nit - n_best * 2)

    for s in range(Nit):
        beta = float(Beta[s])
        alpha = float(Alpha[s])

        # Recipe-driven loop in a single pass:
        #   1) Fourier constraint: match measured magnitude
        #   2) support projection: enforce the object support in real space
        #   3) optional physical projection: apply the recipe-selected projection
        #   4) optional RL coherence refinement: update partial-coherence kernel
        #   5) keep the state in real space at the end of the iteration
        # This keeps the numerical order readable without reordering the actual math.
        guess_cp, convolved = _enforce_single_mode_fourier_constraint(
            guess_cp,
            diffract_cp,
            gamma_cp,
            obs,
            BSmask_cp,
            use_RL,
        )

        # Constraint 2: support projection in real space.
        inv = fft2(guess_cp)
        if ((s % TV_freq) == 0) and alpha > 0:
            inv = inv + alpha * TV(inv, 1)
        inv = proj_fn(inv, prev, mask_cp, beta, s, Nit)
        prev = inv.copy()
        new_guess = ifft2(inv)

        # Constraint 3: optional RL coherence update on the projected result.
        if use_RL:
            if s > RL_freq and (s % RL_freq == 0):
                convolved_new = ifft2(fft2(xp.abs(new_guess) ** 2) * fft2(gamma_cp))
                Idelta = 2 * xp.abs(new_guess) ** 2 - xp.abs(guess_cp) ** 2
                I_exp = obs * (xp.abs(diffract_cp) ** 2) + convolved_new * BSmask_cp
                gamma_cp = RL(
                    Idelta=Idelta,
                    Iexp=I_exp,
                    gamma_cp=gamma_cp,
                    RL_it=RL_it,
                )

            guess_cp = new_guess
            convolved = ifft2(fft2(xp.abs(guess_cp) ** 2) * fft2(gamma_cp))
            err_guess = obs * convolved
            err_target = obs * xp.abs(diffract_cp) ** 2
        else:
            guess_cp = new_guess
            err_guess = xp.abs(guess_cp) * obs
            err_target = diffract_cp * obs

        if s <= 2 or (s % plot_every == 0) or (s >= start_best_at):
            err = Error_diffract_cp(err_guess, err_target)
            Error_diffr_list.append(err)

            if s >= start_best_at:
                if err < Best_error[int(xp.argmax(Best_error).item())]:
                    j = int(xp.argmax(Best_error).item())
                    Best_error[j] = err
                    Best_guess[j, :, :] = guess_cp
                    if use_RL:
                        Best_gamma[j, :, :] = gamma_cp

    guess_cp = xp.mean(Best_guess, axis=0)

    if use_RL:
        gamma_cp = xp.mean(Best_gamma, axis=0)

        if Fourier_last:
            convolved_last = ifft2(fft2(xp.abs(guess_cp) ** 2) * fft2(gamma_cp))
            factor_last = diffract_cp / xp.sqrt(convolved_last)
            guess_cp[obs] *= factor_last[obs]

        guess = to_numpy(guess_cp, xp)
        gamma = to_numpy(gamma_cp, xp)

        return np.fft.ifftshift(guess),Error_diffr_list,Error_supp_list,np.fft.ifftshift(gamma)

    else:
        if Fourier_last:
            guess_cp = xp.where(
                BSmask_cp,
                guess_cp,
                diffract_cp * xp.exp(1j * xp.angle(guess_cp)),
            )

        guess = to_numpy(guess_cp, xp)

        return np.fft.ifftshift(guess), Error_diffr_list, Error_supp_list, None


#############################################################
#    TOTAL VARIATION FUNCTION
# ############################################################

def _grad_backward_1D_cp(u, ax):
    """Return the backward finite difference along one array axis."""
    out = xp.empty_like(u, dtype=u.dtype)

    s1 = [slice(None)] * u.ndim
    s2 = [slice(None)] * u.ndim

    s1[ax] = slice(1, None)
    s2[ax] = slice(None, -1)

    out[tuple(s1)] = u[tuple(s1)] - u[tuple(s2)]

    s1[ax] = 0
    #s2[ax] = -1

    out[tuple(s1)] = 0
    #out[tuple(s2)] = 0

    return out


def _grad_forward_1D_cp(u, ax):
    """Return the forward finite difference along one array axis."""
    out = xp.empty_like(u, dtype=u.dtype)

    s1 = [slice(None)] * u.ndim
    s2 = [slice(None)] * u.ndim

    s1[ax] = slice(1, None)
    s2[ax] = slice(None, -1)

    out[tuple(s2)] = u[tuple(s1)] - u[tuple(s2)]

    s1[ax] = -1
    #s2[ax] = 0

    out[tuple(s1)] = 0
    #out[tuple(s2)] = 0

    return out


def _tv_grad_malm_cp(u, ep=1e-4):
    """
    Equivalent to Erik Malm's _tv_grad().
    This is the true TV gradient direction.
    """

    grad_u = xp.array([
        _grad_backward_1D_cp(u, ax)
        for ax in range(u.ndim)
    ])

    grad_mag = xp.sqrt(xp.sum(xp.abs(grad_u)**2, axis=0))
    grad_mag = ep + grad_mag

    tv_grad = xp.zeros_like(u)

    for ax in range(u.ndim):
        tv_grad -= _grad_forward_1D_cp(grad_u[ax] / grad_mag, ax)

    return tv_grad


def TV(u, mask=1, ep=1e-4):
    """
    TV descent direction compatible with your current update:

        u = u + alpha * TV(u, mask)

    This is equivalent to Erik's:

        u = u - stepsize * _tv_grad(u)

    Therefore:

        TV = -_tv_grad
    """

    tv_descent_direction = -_tv_grad_malm_cp(u, ep=ep)

    return tv_descent_direction * mask

#############################################################
#    FILTER FOR OSS
# ############################################################
def W(npx,npy,alpha=0.1):
    '''
    Simple generator of a gaussian, used for filtering in OSS
    INPUT:  npx,npy: number of pixels on the image
            alpha: width of the gaussian

    OUTPUT: gaussian matrix

    --------
    author: RB 2020
    '''
    Y,X = xp.meshgrid(xp.arange(npy),xp.arange(npx))
    k=(xp.sqrt((X-npx//2)**2+(Y-npy//2)**2))
    return xp.fft.fftshift(xp.exp(-0.5*(k/alpha)**2))

#############################################################
#    ERROR FUNCTIONS
# ############################################################

def Error_diffract_cp(guess, diffract):
    '''
    Error on the diffraction attern of retrieved data.
    INPUT:  guess, diffract: retrieved and experimental diffraction patterns

    OUTPUT: Error between the two

    --------
    author: RB 2020
    '''
    Num=xp.abs(diffract-guess)**2
    Den=xp.abs(diffract)**2
    Error = Num.sum()/Den.sum()
    try:
        # CuPy does not consistently expose numpy.errstate across versions.
        with np.errstate(divide="ignore", invalid="ignore") if xp is np else nullcontext():
            Error=10*xp.log10(Error)
    except FloatingPointError:
        Error = xp.inf
    return to_numpy(Error, xp)



#############################################################
#       MULTI-ENERGY PHASE RETRIEVAL EXTENSION
#############################################################


def _as_energy_stack(holograms, name="holograms"):
    """
    Validate and return an array with shape (nE, nx, ny).
    """
    arr = np.asarray(holograms)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (nE, nx, ny).")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two energies.")
    return arr


def _as_energy_mask(mask_pixel, nE, image_shape):
    """
    Return an energy-dependent mask stack with shape (nE, nx, ny).

    Accepts either:
      - mask_pixel.shape == (nx, ny), shared by all energies
      - mask_pixel.shape == (nE, nx, ny), energy dependent
    """
    mask_pixel = np.asarray(mask_pixel)
    if mask_pixel.shape == image_shape:
        return np.broadcast_to(mask_pixel, (nE,) + image_shape).copy()
    if mask_pixel.shape == (nE,) + image_shape:
        return mask_pixel.copy()
    raise ValueError(
        "mask_pixel must have shape (nx, ny) or (nE, nx, ny). "
        f"Got {mask_pixel.shape}, expected {image_shape} or {(nE,) + image_shape}."
    )


def _prepare_energy_amplitudes(
    holograms,
    mask_pixel,
    hologram_intensity_cutoff_vmin=-1,
):
    """
    Prepare measured amplitudes and energy-specific beamstop/invalid masks.

    Parameters
    ----------
    holograms : array, shape (nE, nx, ny)
        Intensity holograms / diffraction intensities, one per energy.
    mask_pixel : array, shape (nx, ny) or (nE, nx, ny)
        External Fourier mask(s). Nonzero values are treated as unconstrained.
    hologram_intensity_cutoff_vmin : float
        If >= 0, subtract this percentile independently from each energy before
        clipping negative values to zero.

    Returns
    -------
    amplitudes, intensities, bsmasks : arrays, shape (nE, nx, ny)

    Context:
    Input boundary: convert measured intensities into amplitude targets and effective
    invalid masks. Keep the explicit detector mask distinct from cutoff-based exclusions
    when estimating radial means or low-count statistics.
    """
    intensities = _as_energy_stack(holograms).astype(float, copy=True)
    nE, nx, ny = intensities.shape
    bsmasks = _as_energy_mask(mask_pixel, nE=nE, image_shape=(nx, ny))

    for j in range(nE):
        img = intensities[j]

        if hologram_intensity_cutoff_vmin >= 0:
            finite = img[(img != 0) & np.isfinite(img)]
            if finite.size:
                mi = np.nanpercentile(finite, hologram_intensity_cutoff_vmin)
                img = img - mi

        img = np.where(np.isnan(img), 0, img)
        bsmasks[j, img <= 0] = 1
        intensities[j] = np.clip(img, 0, None)

    amplitudes = np.sqrt(intensities)
    return amplitudes, intensities, bsmasks


def fourier_field_to_object_log(
    phase_stack,
    log_floor=1e-12,
    unwrap_energy_phase=True,
):
    """
    Convert Fourier-domain fields returned by PhaseRtrv_core to real-space
    complex log-objects.

    Inside PhaseRtrv_core, the support-domain object estimate is approximately

        object = fft2(fftshift(phase))

    for the returned field convention.
    """
    phase_stack = _as_energy_stack(phase_stack, name="phase_stack")
    if not np.isfinite(log_floor) or log_floor <= 0:
        raise ValueError("log_floor must be finite and > 0.")
    obj = np.empty_like(phase_stack, dtype=np.complex128)

    for j in range(phase_stack.shape[0]):
        obj[j] = np.fft.fft2(np.fft.fftshift(phase_stack[j]))

    amp = np.maximum(np.abs(obj), log_floor)
    phase = np.angle(obj)
    if unwrap_energy_phase:
        phase = np.unwrap(phase, axis=0)

    return np.log(amp) + 1j * phase








def object_log_to_fourier_field(L_stack):
    """
    Convert real-space log-objects back to Fourier-domain fields.
    """
    L_stack = _as_energy_stack(L_stack, name="L_stack")
    phase_stack = np.empty_like(L_stack, dtype=np.complex128)

    for j in range(L_stack.shape[0]):
        phase_stack[j] = np.fft.ifftshift(np.fft.ifft2(np.exp(L_stack[j])))

    return phase_stack




def _validate_material_thickness(value, shape):
    """Validate a relative material thickness on the object grid."""
    if value is None:
        return None
    thickness = np.asarray(value, dtype=float)
    if thickness.shape != tuple(shape) or not np.all(np.isfinite(thickness)):
        raise ValueError(f"material_thickness must be finite with shape {tuple(shape)}.")
    if np.any(thickness < 0):
        raise ValueError("material_thickness must be nonnegative.")
    return thickness


def _recipe_material_thickness(recipe, input_geometry):
    """
    Map optional object-grid material inputs through crop/bin geometry.

    Context:
    Geometry bridge: bring the supplied material/thickness information onto the object grid
    used by the physical projection, rather than treating detector binning as object-space
    averaging.
    """
    mask = recipe.get("material_mask")
    thickness = recipe.get("material_thickness")
    if mask is None and thickness is None and not recipe.get("fit_material_thickness", False):
        return None
    source_shape = input_geometry.source_shape
    used_shape = input_geometry.used_shape
    crop = input_geometry.crop
    scale = (np.asarray(source_shape, dtype=float)
             / (np.asarray(source_shape, dtype=float) - 2 * crop))

    def map_object(value, name, order):
        array = np.asarray(value)
        if array.shape == used_shape:
            return array.copy()
        if array.shape != source_shape:
            raise ValueError(
                f"{name} must have source shape {source_shape} or retrieval shape {used_shape}."
            )
        if source_shape == used_shape:
            return array.copy()
        if crop:
            return geometry._resample_spatial(
                array, used_shape, order=order, coordinate_scale=scale,
            )
        # Match geometry.prepare: binning alone keeps object pixel scale.
        origin = tuple((n - u) // 2 for n, u in zip(source_shape, used_shape))
        return array[tuple(slice(o, o + u) for o, u in zip(origin, used_shape))]

    result = (np.ones(used_shape, dtype=float) if thickness is None else
              _validate_material_thickness(map_object(thickness, "material_thickness", 1), used_shape))
    if mask is not None:
        material = map_object(mask, "material_mask", 0)
        if not np.all(np.isfinite(material)):
            raise ValueError("material_mask must be finite.")
        result = result * (material != 0)
    if not np.any(result > 0):
        raise ValueError("material_mask/material_thickness contains no material pixels.")
    return result






# -------------------------------------------------------------------------
#  Explicit spectral model: L_E(r) = C(r) + M(r) a_E
# -------------------------------------------------------------------------

def _energy_axis_for_kk(energy_values, nE):
    """Validate the positive, increasing energy axis used by KK constraints."""
    if energy_values is None:
        raise ValueError("energy_values must be provided for KK-based constraints.")
    energy_values = np.asarray(energy_values, dtype=float)
    if energy_values.shape != (nE,):
        raise ValueError("energy_values must have shape (nE,).")
    if np.any(~np.isfinite(energy_values)):
        raise ValueError("energy_values contains non-finite values.")
    if np.any(energy_values <= 0):
        raise ValueError("energy_values must be strictly positive for the KK transform.")
    if np.any(np.diff(energy_values) <= 0):
        raise ValueError("energy_values must be strictly increasing.")
    return energy_values


def _normalize_vector(v, mode="l2"):
    """Return a real vector normalized with the requested scale convention."""
    v = np.asarray(v, dtype=float)
    if mode == "none":
        return v.copy()
    if mode == "maxabs":
        scale = np.nanmax(np.abs(v))
    elif mode == "l2":
        scale = np.linalg.norm(v)
    elif mode == "std":
        scale = np.nanstd(v)
    else:
        raise ValueError("normalization mode must be 'none', 'maxabs', 'l2', or 'std'.")
    if not np.isfinite(scale) or scale <= 0:
        return v.copy()
    return v / scale


def _fit_scale_offset(reference, target, fit_offset=True):
    """
    Fit target ≈ scale * reference + offset using real least squares.
    """
    reference = np.asarray(reference, dtype=float)
    target = np.asarray(target, dtype=float)
    ok = np.isfinite(reference) & np.isfinite(target)
    if np.count_nonzero(ok) < 2:
        return 1.0, 0.0
    x = reference[ok]
    y = target[ok]
    if fit_offset:
        A = np.column_stack([x, np.ones_like(x)])
        scale, offset = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        denom = np.dot(x, x)
        scale = 1.0 if denom <= 0 else np.dot(x, y) / denom
        offset = 0.0
    return float(scale), float(offset)




def _complex_spectrum_from_parts(absorption, dispersion, absorption_part="imag"):
    """
    Build a complex spectrum from absorptive and dispersive real vectors.

    absorption_part defines where the absorptive part lives in the log-object
    spectral coefficient:
      - 'real': a_E = absorption + 1j * dispersion
      - 'imag': a_E = dispersion + 1j * absorption

    For an exit-wave log-object, absorption often appears in the real part of
    log(O), while phase/refraction appears in the imaginary part. Therefore
    'real' is the default.
    """
    absorption = np.asarray(absorption, dtype=float)
    dispersion = np.asarray(dispersion, dtype=float)
    if absorption_part == "real":
        return absorption + 1j * dispersion
    if absorption_part == "imag":
        return dispersion + 1j * absorption
    raise ValueError("absorption_part must be 'real' or 'imag'.")


def _extract_absorption_part(a, absorption_part="imag"):
    """Extract the designated absorptive component of a complex spectrum."""
    if absorption_part == "real":
        return np.real(a)
    if absorption_part == "imag":
        return np.imag(a)
    raise ValueError("absorption_part must be 'real' or 'imag'.")


def _extract_dispersion_part(a, absorption_part="imag"):
    """Extract the designated dispersive component of a complex spectrum."""
    if absorption_part == "real":
        return np.imag(a)
    if absorption_part == "imag":
        return np.real(a)
    raise ValueError("absorption_part must be 'real' or 'imag'.")


def constrain_complex_spectrum(
    a_initial,
    spectral_constraint="free",
    energy_values=None,
    absorption_part="imag",
    kk_sign=1.0,
    kk_subtract_baseline=True,
    kk_normalize_input=False,
):
    """Apply the optional Kramers-Kronig spectral constraint to a complex spectrum."""
    a_initial = np.asarray(a_initial, dtype=np.complex128)
    nE = a_initial.size
    mode = str(spectral_constraint).lower()
    if mode in {"none", "free", "unconstrained"}:
        return a_initial.copy(), {
            "spectral_constraint": "free",
            "spectrum_initial": a_initial.copy(),
            "spectrum_constrained": a_initial.copy(),
        }

    if mode not in {"kk", "kramers-kronig", "kramers_kronig"}:
        raise ValueError("spectral_constraint must be 'free' or a Kramers-Kronig option.")

    retrieved_abs = _extract_absorption_part(a_initial, absorption_part)
    retrieved_disp = _extract_dispersion_part(a_initial, absorption_part)
    E = _energy_axis_for_kk(energy_values, nE)
    absorption = retrieved_abs.copy()
    dispersion = kk_sign * kk.beta_to_delta(
        E,
        absorption,
        subtract_baseline=kk_subtract_baseline,
        normalize_input=kk_normalize_input,
        output_mean=np.mean(retrieved_disp),
    )
    a = _complex_spectrum_from_parts(absorption, dispersion, absorption_part)
    return a.astype(np.complex128), {
        "spectral_constraint": mode,
        "spectrum_initial": a_initial.copy(),
        "spectrum_constrained": a.copy(),
        "absorption_part": absorption.copy(),
        "dispersion_part": dispersion.copy(),
        "absorption_part_location": absorption_part,
    }






# -------------------------------------------------------------------------
#  Multi-energy reconstruction driver
# -------------------------------------------------------------------------



def _normalize_update_schedule(modes, iterations, name, allow_disabled=False):
    """Validate and return a list of ``(mode, Nit)`` update stages."""
    allowed_algorithms = {
        "ER",
        "SF",
        "HAPRE",
        "RAAR",
        "HIOs",
        "HIO",
        "OSS",
        "CHIO",
        "HPR",
        "gradient_descent",
    }

    if isinstance(modes, str):
        modes = [modes]
    elif isinstance(modes, (list, tuple)):
        modes = list(modes)
    else:
        raise ValueError(f"{name}_mode must be a string or a list of strings.")

    if isinstance(iterations, (int, np.integer)) and not isinstance(
        iterations, bool
    ):
        if allow_disabled and iterations == 0:
            return []
        iterations = [int(iterations)]
    elif isinstance(iterations, (list, tuple)):
        iterations = list(iterations)
    else:
        raise ValueError(f"{name}_Nit must be an integer or a list of integers.")

    if not modes:
        raise ValueError(f"{name}_mode cannot be empty.")
    if len(modes) != len(iterations):
        raise ValueError(
            f"{name}_mode and {name}_Nit must have the same length."
        )

    schedule = []
    for stage_index, (mode, Nit) in enumerate(zip(modes, iterations)):
        if mode not in allowed_algorithms:
            raise ValueError(
                f"Invalid {name}_mode[{stage_index}]={mode!r}. "
                f"Allowed modes are {sorted(allowed_algorithms)}."
            )
        if (
            isinstance(Nit, bool)
            or not isinstance(Nit, (int, np.integer))
            or Nit <= 0
        ):
            raise ValueError(
                f"{name}_Nit[{stage_index}] must be a positive integer."
            )
        schedule.append((mode, int(Nit)))
    return schedule


def _broadcast_stage_parameter(value, stage_count, key):
    """Return one value per stage, broadcasting scalar settings."""
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) != stage_count:
            raise ValueError(
                f"{key} must be a scalar or have the same length as the "
                f"corresponding mode/iteration schedule "
                f"({stage_count}); got {len(values)}."
            )
        return values
    return [value] * stage_count


def _build_update_schedule(recipe, name, allow_disabled=False):
    """Build a fully specified stage dictionary sequence for one update pass."""
    mode_key = f"{name}_mode"
    Nit_key = f"{name}_Nit"
    base_schedule = _normalize_update_schedule(
        recipe[mode_key],
        recipe[Nit_key],
        name=name,
        allow_disabled=allow_disabled,
    )
    if not base_schedule:
        return []

    control_keys = [
        "beta_zero",
        "beta_mode",
        "alpha_zero",
        "alpha_mode",
        "TV_freq",
        "RL_it",
        "RL_freq",
    ]
    controls = {}
    for key in control_keys:
        recipe_key = f"{key}"
        value = recipe[recipe_key]
        controls[key] = _broadcast_stage_parameter(
            value,
            len(base_schedule),
            recipe_key,
        )

    schedule = []
    for stage_index, (mode, Nit) in enumerate(base_schedule):
        stage = {
            "mode": mode,
            "Nit": Nit,
            **{
                key: controls[key][stage_index]
                for key in control_keys
            },
        }

        for key in ("beta_zero", "alpha_zero"):
            value = stage[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"{key}[{stage_index}] must be a finite number."
                )

        for key in ("beta_mode", "alpha_mode"):
            value = stage[key]
            if isinstance(value, str):
                if value not in BETA_SCHEDULES:
                    raise ValueError(
                        f"Invalid {key}[{stage_index}]={value!r}. "
                        f"Allowed modes are {sorted(BETA_SCHEDULES)}."
                    )
            elif isinstance(value, np.ndarray):
                if value.ndim != 1 or value.shape[0] != Nit:
                    raise ValueError(
                        f"{key}[{stage_index}] array must have "
                        f"length Nit={Nit}."
                    )
                if np.any(~np.isfinite(value)):
                    raise ValueError(
                        f"{key}[{stage_index}] contains non-finite values."
                    )
            else:
                raise ValueError(
                    f"{key}[{stage_index}] must be a schedule name "
                    "or a one-dimensional NumPy array."
                )

        TV_freq = stage["TV_freq"]
        if (
            isinstance(TV_freq, bool)
            or not isinstance(TV_freq, (int, float, np.number))
            or not np.isfinite(TV_freq)
            or TV_freq <= 0
        ):
            raise ValueError(
                f"TV_freq[{stage_index}] must be a positive number."
            )

        RL_it = stage["RL_it"]
        if (
            isinstance(RL_it, bool)
            or not isinstance(RL_it, (int, np.integer))
            or RL_it < 0
        ):
            raise ValueError(
                f"RL_it[{stage_index}] must be a non-negative integer."
            )

        RL_freq = stage["RL_freq"]
        if (
            isinstance(RL_freq, bool)
            or not isinstance(RL_freq, (int, float, np.number))
            or not np.isfinite(RL_freq)
            or RL_freq <= 0
        ):
            raise ValueError(
                f"RL_freq[{stage_index}] must be a positive number."
            )

        schedule.append(stage)
    return schedule





def _compile_iteration_recipe(
    schedule,
    recipe,
    *,
    field,
    amplitude,
    bsmask,
    gamma,
    fixed_masked_intensity=None,
    freeze_gamma=False,
):
    """Build a single explicit recipe for the next observation-level update pass.

    Each entry describes exactly what the solver should enforce in that stage:
    Fourier amplitude, support projection, and optional partial-coherence RL
    refinement. The recipe is compiled once so the update loop reads as a clear
    sequence instead of hidden branching logic.
    """
    compiled = []
    filled_intensity = None
    for stage_index, stage in enumerate(schedule):
        stage = dict(stage)
        iterations = int(stage["Nit"])

        if freeze_gamma and stage["RL_it"] > 0 and stage["RL_freq"] <= iterations:
            if gamma is None:
                raise ValueError("A fitted reference gamma is required before freezing coherence.")
            stage["RL_freq"] = iterations

        use_rl = stage["RL_it"] > 0 and stage["RL_freq"] <= iterations
        if use_rl:
            if gamma is None:
                gamma_shape = (
                    (recipe["Nmodes"], *amplitude.shape)
                    if recipe["Nmodes"] > 1 else amplitude.shape
                )
                gamma = np.full(gamma_shape, 2e-6, dtype=float)
                gamma[..., amplitude.shape[0] // 2, amplitude.shape[1] // 2] = 0.7
            if filled_intensity is None:
                current = (
                    np.sum(np.abs(field) ** 2, axis=0)
                    if recipe["Nmodes"] > 1 else np.abs(field) ** 2
                )
                masked_values = current if fixed_masked_intensity is None else np.asarray(fixed_masked_intensity)
                filled_intensity = np.where(bsmask != 0, masked_values, amplitude**2)
            stage["amplitude"] = np.sqrt(np.maximum(filled_intensity, 0))
            stage["bsmask"] = np.zeros_like(bsmask)
            stage["coherence"] = "partial"
        else:
            stage["amplitude"] = amplitude
            stage["bsmask"] = bsmask
            stage["coherence"] = "full"
            filled_intensity = None
        stage["use_rl"] = use_rl
        stage["schedule_stage"] = stage_index
        compiled.append(stage)
    return compiled, gamma


def _projection_is_due(completed_updates, projection_start, projection_every,
                       total_updates=None):
    """Return whether a joint projection is scheduled after this update.

    Both controls use completed observation or energy updates, not outer-loop
    indices. A positive start value is itself an eligible boundary. The last
    update is reserved for the separately controlled final projection.
    """
    if total_updates is not None and completed_updates >= total_updates:
        return False
    if completed_updates < projection_start:
        return False
    offset = (
        completed_updates
        if projection_start == 0
        else completed_updates - projection_start
    )
    return offset % projection_every == 0


def _resolve_projection_cadence(recipe, default_every):
    """Return positive integer projection_every and projection_start values."""
    projection_every = (
        int(default_every)
        if recipe["projection_every"] is None
        else int(recipe["projection_every"])
    )
    projection_start = (
        projection_every
        if recipe["projection_start"] is None
        else int(recipe["projection_start"])
    )
    return projection_every, projection_start






# -----------------------------------------------------------------------------
# Recipe defaults and high-level driver configuration
# -----------------------------------------------------------------------------


def default_general_phase_retrieval_recipe():
    """Return defaults for joint state/energy/polarization/beam retrieval."""
    return {
        # Phase-retrieval stages applied to each observation in the main loop.
        "inner_mode": ["HAPRE"],
        "inner_Nit": [1],
        "outer_iterations": 300,
        "startimage_scale_fit": "linear",  # or "through_origin"
        "startimage_radial_normalization": False,
        "freeze_saturated_fields": False,
        "partial_coherence": False,
        "coherence_kernel_scope": "per_observation",  # or "shared" across states
        # sequential: hand gamma to the next state; average: independent fits
        # from one snapshot; pooled: fixed gamma per sweep then joint RL fit.
        "shared_coherence_update": "sequential",
        "coherence_round_iterations": 50,
        "coherent_refresh_rounds": [],  # 1-based outer rounds, before their PC updates
        "coherent_refresh_iterations": 50,
        "coherent_refresh_mode": "ER",
        "observation_workers": 1,
        "fft_workers": 1,
        "freeze_coherence_after_reference": False,
        "shuffle_observations": True,
        "random_seed": 0,
        "beta_zero": 0.5,
        "beta_mode": "arctan",
        "alpha_zero": 0.0,
        "alpha_mode": "const",
        "TV_freq": 1e9,
        "RL_it": 0,
        "RL_freq": 1e9,
        "plot_every": 1e9,
        "average_img": 1,
        "Fourier_last": True,
        "final_fourier_constraint": True,
        "hologram_intensity_cutoff_vmin": -1,
        "binning": 1,
        "crop": 0,
        "roi": None,
        "Nmodes": 1,
        "modes": None,
        "gaussian_mode_sigma_x": 4.0,
        "gaussian_mode_sigma_y": 4.0,
        "gaussian_mode_angle": 0.0,
        "gaussian_mode_amplitude": 1.0,
        "gaussian_mode_center": None,
        "gaussian_mode_coherent": False,
        "mode_support_center": "image",
        "recenter_modal_supports": True,
        "mode_initialization": "random_phase",  # or "support_fft"
        "mode_initialization_seed": 0,
        # Optional incident-power ratios from zero-thickness reference apertures.
        "fit_illumination_intensity": False,
        "align_global_phase": False,
        "fit_aperture_illumination": False,
        "object_illumination_mode": "pooled",
        "aperture_factor_type": "complex",
        "illumination_intensity_reference_observation": 0,
        # Optionally remove state dependence from support-factor modes other
        # than 1 by projecting them to one common object per energy/beam.
        "magnetization_mode_groups": None,
        "illumination_mode_groups": None,
        "constrain_nonphysical_modes_common": False,
        "nonphysical_modes_common_relaxation": 1.0,
        # General log-object projection settings.
        "projection_model": "physical_factorized",
        # Number of completed observation updates between projections. None
        # means one full observation sweep.
        "projection_every": None,
        # None starts projection at the first projection_every boundary.
        "projection_start": None,
        "projection_relaxation": 1.0,
        "final_projection_relaxation": 1.0,
        "observation_weights": None,
        "rank_deficient": "error",
        # Physical factorization L = C_m + q_c(E) + p*q_m(E)*mz_s.
        "physical_iterations": 20,
        "clip_magnetization": True,
        "magnetization_clip_range": (-1.0, 1.0),
        "magnetization_trend": "none",
        "magnetization_trend_intensity_cutoff": 0.01,
        "magnetization_trend_correction": "offset",
        "bound_magnetization_median": False,
        "magnetization_median_scope": "whole",
        # Snap to the nearest of -1, 0, +1.
        "initial_binary_magnetization": False,
        "binary_magnetization": False,
        "binary_magnetization_values": [-1.0, 0.0, 1.0],
        "physical_projection_object_roi": False,
        "physical_phase_reference": False,
        # Shared spatial least-squares weights from mean anchor |exit wave|^2.
        "physical_spatial_weighting": "exit_wave_intensity",
        "projection_diagnostic_observation": None,
        # Optional reversible focus transform around every object-space
        # physical projection. Zero propagation skips propagation and phase.
        "projection_focus_prop_um": 0.0,
        "projection_focus_phase_rad": 0.0,
        "projection_focus_setup": None,
        "projection_focus_integer_wavelength": True,
        "material_mask": None,
        "material_thickness": None,
        "fit_material_thickness": False,
        # None preserves the legacy one-chemical/one-magnetic physical model.
        # A list enables nonnegative chemical mixtures with optional magnetism
        # and independent spectral constraints for each species.
        "chemical_species": None,
        "chemical_reference_mask": None,
        "chemical_reference_alignment": True,
        "chemical_initialization_seed": 0,
        "saturated_states": None,
        # If True, force the fitted magnetic state maps to zero outside the
        # real-space support used by the phase-retrieval kernel.
        "zero_magnetization_outside_support": True,
        # If True, force the material thickness to zero outside that support.
        # A supplied material mask can additionally mark holes inside it.
        "zero_thickness_outside_support": False,
        # If True, apply the selected joint log-object projection only inside
        # the real-space support; outside it, keep each observation unchanged.
        "projection_constraints_inside_support_only": True,
        # Backward-compatible alias for older physical-only recipes.
        "physical_constraints_inside_support_only": False,
        "charge_spectral_constraint": "free",
        "magnetic_spectral_constraint": "free",
        "energy_values": None,
        "charge_absorption_part": "real",
        "magnetic_absorption_part": "real",
        "charge_response_real_range": None,
        "charge_response_imag_range": None,
        "magnetic_response_real_range": None,
        "magnetic_response_imag_range": None,
        "kk_sign": 1.0,
        "kk_subtract_baseline": True,
        "kk_normalize_input": False,
        "log_floor": 1e-12,
    }


def _verify_recipe(recipe, n_observations, n_energies=None):
    """Validate the general retrieval recipe and observation weights."""
    if (isinstance(recipe["Nmodes"], bool)
            or not isinstance(recipe["Nmodes"], (int, np.integer))
            or recipe["Nmodes"] < 1):
        raise ValueError("Nmodes must be a positive integer.")
    if recipe["mode_support_center"] not in {"image", "components"}:
        raise ValueError("mode_support_center must be 'image' or 'components'.")
    if not isinstance(recipe["recenter_modal_supports"], bool):
        raise ValueError("recenter_modal_supports must be bool.")
    seed = recipe["mode_initialization_seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise ValueError("mode_initialization_seed must be an integer or None.")
    if recipe["mode_initialization"] not in {"random_phase", "support_fft"}:
        raise ValueError("mode_initialization must be 'random_phase' or 'support_fft'.")
    if not isinstance(recipe["fit_illumination_intensity"], bool):
        raise ValueError("fit_illumination_intensity must be bool.")
    if recipe["fit_illumination_intensity"] and recipe["projection_model"] != "physical_factorized":
        raise ValueError("Intensity fitting requires physical_factorized projection.")
    for key in ("align_global_phase", "fit_aperture_illumination"):
        if not isinstance(recipe[key], bool):
            raise ValueError(f"{key} must be bool.")
        if recipe[key] and recipe["projection_model"] != "physical_factorized":
            raise ValueError(f"{key} requires physical_factorized projection.")
    if recipe["object_illumination_mode"] not in {"none", "pooled", "distance_weighted", "nearest", "linear_plane"}:
        raise ValueError("object_illumination_mode must be none, pooled, distance_weighted, nearest, or linear_plane.")
    if recipe["aperture_factor_type"] not in {"complex", "amplitude"}:
        raise ValueError("aperture_factor_type must be complex or amplitude.")
    if recipe.get("magnetization_mode_groups") is not None or recipe.get("illumination_mode_groups") is not None:
        if recipe["projection_model"] != "physical_factorized":
            raise ValueError("Physical mode groups require the physical_factorized model.")
        _physical_mode_groups(recipe, recipe["Nmodes"])
    if not isinstance(recipe["constrain_nonphysical_modes_common"], bool):
        raise ValueError("constrain_nonphysical_modes_common must be bool.")
    common_relaxation = recipe["nonphysical_modes_common_relaxation"]
    if (isinstance(common_relaxation, bool)
            or not isinstance(common_relaxation, (int, float, np.number))
            or not np.isfinite(common_relaxation)
            or not 0 <= common_relaxation <= 1):
        raise ValueError("nonphysical_modes_common_relaxation must be between 0 and 1.")
    if not isinstance(recipe["fit_material_thickness"], bool):
        raise ValueError("fit_material_thickness must be bool.")
    if recipe.get("chemical_species") is not None:
        raise ValueError("Chemical species decomposition is outside the paper implementation")
        validate_species(recipe["chemical_species"])
        if recipe["projection_model"] != "physical_factorized":
            raise ValueError("chemical_species requires projection_model='physical_factorized'.")
        if recipe["fit_material_thickness"]:
            raise ValueError("With chemical_species, fit species density maps instead of fit_material_thickness.")
    if not isinstance(recipe["chemical_reference_alignment"], bool):
        raise ValueError("chemical_reference_alignment must be bool.")
    seed = recipe["chemical_initialization_seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("chemical_initialization_seed must be a nonnegative integer.")
    focus_prop = recipe["projection_focus_prop_um"]
    focus_phase = recipe["projection_focus_phase_rad"]
    if (isinstance(focus_prop, bool) or not isinstance(focus_prop, (int, float, np.number))
            or not np.isfinite(focus_prop)):
        raise ValueError("projection_focus_prop_um must be a finite number.")
    if (isinstance(focus_phase, bool) or not isinstance(focus_phase, (int, float, np.number))
            or not np.isfinite(focus_phase)):
        raise ValueError("projection_focus_phase_rad must be a finite number.")
    if not isinstance(recipe["projection_focus_integer_wavelength"], bool):
        raise ValueError("projection_focus_integer_wavelength must be bool.")
    focus_setup = recipe["projection_focus_setup"]
    if focus_prop != 0:
        if not isinstance(focus_setup, dict):
            raise ValueError("Nonzero projection focus requires projection_focus_setup.")
        for key in ("ccd_dist", "px_size"):
            value = focus_setup.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"projection_focus_setup[{key!r}] must be positive and finite.")
        if "energy" in focus_setup:
            energy = focus_setup["energy"]
            if (isinstance(energy, bool) or not isinstance(energy, (int, float, np.number))
                    or not np.isfinite(energy) or energy <= 0):
                raise ValueError("projection_focus_setup['energy'] must be positive and finite.")
    for key in ("observation_workers", "fft_workers", "coherence_round_iterations"):
        value = recipe[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if recipe["shared_coherence_update"] not in {"sequential", "average", "pooled"}:
        raise ValueError("shared_coherence_update must be sequential, average, or pooled.")
    if recipe["shared_coherence_update"] != "sequential":
        if recipe["coherence_kernel_scope"] != "shared":
            raise ValueError("Round coherence updates require coherence_kernel_scope='shared'.")
        if recipe["freeze_coherence_after_reference"]:
            raise ValueError("Round coherence updates cannot freeze gamma after the reference.")
    for key in ("freeze_coherence_after_reference", "startimage_radial_normalization"):
        if not isinstance(recipe[key], bool):
            raise ValueError(f"{key} must be bool.")
    refresh = recipe["coherent_refresh_rounds"]
    if (not isinstance(refresh, (list, tuple)) or
            any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                or v < 1 or v > recipe["outer_iterations"] for v in refresh)
            or len(set(refresh)) != len(refresh)):
        raise ValueError("coherent_refresh_rounds must contain distinct 1-based outer round numbers.")
    if refresh and not recipe["partial_coherence"]:
        raise ValueError("Coherent refresh rounds require partial_coherence=True.")
    if refresh and not any(stage["RL_it"] > 0 and stage["RL_freq"] <= stage["Nit"]
                           for stage in _build_update_schedule(recipe, "inner")):
        raise ValueError("Coherent refresh requires an active partial-coherence inner stage to resume.")
    if (isinstance(recipe["coherent_refresh_iterations"], bool) or
            not isinstance(recipe["coherent_refresh_iterations"], (int, np.integer)) or
            recipe["coherent_refresh_iterations"] < 1):
        raise ValueError("coherent_refresh_iterations must be a positive integer.")
    _normalize_update_schedule([recipe["coherent_refresh_mode"]],
                               [recipe["coherent_refresh_iterations"]], name="coherent refresh")
    if recipe["startimage_scale_fit"] not in {"linear", "through_origin"}:
        raise ValueError("startimage_scale_fit must be 'linear' or 'through_origin'.")
    if not isinstance(recipe["freeze_saturated_fields"], bool):
        raise ValueError("freeze_saturated_fields must be bool.")
    if not isinstance(recipe["partial_coherence"], bool):
        raise ValueError("partial_coherence must be bool.")
    if recipe["coherence_kernel_scope"] not in {"per_observation", "shared"}:
        raise ValueError("coherence_kernel_scope must be 'per_observation' or 'shared'.")
    _build_update_schedule(recipe, name="inner")
    for key in (
        "outer_iterations",
        "average_img",
    ):
        value = recipe[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{key} must be an integer.")
    if recipe["outer_iterations"] < 0:
        raise ValueError("outer_iterations must be >= 0.")
    projection_start = recipe["projection_start"]
    if projection_start is not None and (
        isinstance(projection_start, bool)
        or not isinstance(projection_start, (int, np.integer))
        or projection_start < 0
    ):
        raise ValueError("projection_start must be None or a non-negative integer.")
    projection_every = recipe["projection_every"]
    if projection_every is not None and (
        isinstance(projection_every, bool)
        or not isinstance(projection_every, (int, np.integer))
        or projection_every <= 0
    ):
        raise ValueError("projection_every must be None or a positive integer.")
    if recipe["average_img"] <= 0:
        raise ValueError("average_img must be > 0.")
    if recipe["plot_every"] <= 0:
        raise ValueError("plot_every must be > 0.")
    if not isinstance(recipe["Fourier_last"], bool):
        raise ValueError("Fourier_last must be bool.")
    if not isinstance(recipe["final_fourier_constraint"], bool):
        raise ValueError("final_fourier_constraint must be bool.")
    if recipe["partial_coherence"] and recipe["final_fourier_constraint"]:
        raise ValueError(
            "final_fourier_constraint must be False with partial_coherence: "
            "a coherent amplitude overwrite would undo the blurred-intensity fit."
        )
    if not (0 <= recipe["projection_relaxation"] <= 1):
        raise ValueError("projection_relaxation must be between 0 and 1.")
    if not (0 <= recipe["final_projection_relaxation"] <= 1):
        raise ValueError("final_projection_relaxation must be between 0 and 1.")
    if recipe["projection_model"] not in {"physical_factorized", "none"}:
        raise ValueError("projection_model must be 'physical_factorized' or 'none'.")
    if recipe["rank_deficient"] not in {"error", "minimum_norm"}:
        raise ValueError(
            "rank_deficient must be 'error' or 'minimum_norm'."
        )
    if not isinstance(recipe["clip_magnetization"], bool):
        raise ValueError("clip_magnetization must be bool.")
    if not isinstance(recipe["initial_binary_magnetization"], bool):
        raise ValueError("initial_binary_magnetization must be bool.")
    if not isinstance(recipe["binary_magnetization"], bool):
        raise ValueError("binary_magnetization must be bool.")
    if recipe["binary_magnetization"]:
        _binary_magnetization_values(recipe)
    physical_iterations = recipe["physical_iterations"]
    if (
        isinstance(physical_iterations, bool)
        or not isinstance(physical_iterations, (int, np.integer))
        or physical_iterations <= 0
    ):
        raise ValueError("physical_iterations must be a positive integer.")
    if not isinstance(recipe["zero_magnetization_outside_support"], bool):
        raise ValueError("zero_magnetization_outside_support must be bool.")
    if not isinstance(recipe["zero_thickness_outside_support"], bool):
        raise ValueError("zero_thickness_outside_support must be bool.")
    if not isinstance(recipe["projection_constraints_inside_support_only"], bool):
        raise ValueError("projection_constraints_inside_support_only must be bool.")
    if not isinstance(recipe["physical_constraints_inside_support_only"], bool):
        raise ValueError("physical_constraints_inside_support_only must be bool.")
    for key in (
        "charge_response_real_range",
        "charge_response_imag_range",
        "magnetic_response_real_range",
        "magnetic_response_imag_range",
    ):
        _normalize_range(recipe[key], key)
    allowed_spectral = {
        "none",
        "free",
        "unconstrained",
        "kk",
        "kramers-kronig",
        "kramers_kronig",
    }
    for kind in ("charge", "magnetic"):
        mode = str(recipe[f"{kind}_spectral_constraint"]).lower()
        if mode not in allowed_spectral:
            raise ValueError(
                f"{kind}_spectral_constraint has an unsupported value."
            )
        if n_energies is not None and mode not in {"none", "free", "unconstrained"}:
            constrain_complex_spectrum(
                np.zeros(n_energies, dtype=np.complex128),
                spectral_constraint=recipe[f"{kind}_spectral_constraint"],
                energy_values=recipe["energy_values"],
                absorption_part=recipe[f"{kind}_absorption_part"],
                kk_sign=recipe["kk_sign"],
                kk_subtract_baseline=recipe["kk_subtract_baseline"],
                kk_normalize_input=recipe["kk_normalize_input"],
            )
    _observation_weights(recipe["observation_weights"], n_observations)


def _normalize_coherence_kernel(gamma):
    """Project a coherence kernel onto the nonnegative, unit-mass simplex."""
    kernel = np.maximum(np.asarray(gamma, dtype=float).real, 0.0).copy()
    mass = kernel.sum(axis=(-2, -1), keepdims=True)
    if np.any(~np.isfinite(kernel)) or np.any(mass <= 0):
        raise ValueError("Coherence kernels must have finite positive mass.")
    return kernel / mass


def _pooled_coherence_update(fields, targets, gamma, iterations, weights=None):
    """
    RL fit of one kernel per mode to the summed, weighted state likelihood.

    Fields and targets use centered detector coordinates. All states contribute,
    including states whose fields are frozen. Unlike averaging separate fits,
    every RL step uses the joint residual and the sum of modal predictions.

    Context:
    Called at synchronous round boundaries with fields held fixed. targets already contains
    measured intensities plus the currently frozen masked fills. The leading observation
    dimension is pooled; modal kernels remain separate.
    """
    from scipy.fft import fft2 as cpu_fft2, ifft2 as cpu_ifft2

    fields = np.asarray(fields)
    if fields.ndim == 3:
        fields = fields[:, None]
    squeeze = np.ndim(gamma) == 2
    kernel = _normalize_coherence_kernel(gamma)
    if squeeze:
        kernel = kernel[None]
    kernel = np.fft.ifftshift(kernel, axes=(-2, -1))
    weights = np.ones(len(fields)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(fields),) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Pooled coherence needs nonnegative weights with positive total.")
    # Stream states instead of caching an extra complex FFT stack for the loop.
    for _ in range(iterations):
        correction = np.zeros_like(kernel)
        kernel_fft = cpu_fft2(kernel)
        for index, field in enumerate(fields):
            if weights[index] == 0:
                continue
            intensity_fft = cpu_fft2(np.fft.ifftshift(np.abs(field) ** 2, axes=(-2, -1)))
            prediction = cpu_ifft2(intensity_fft * kernel_fft).real.sum(axis=0)
            target = np.fft.ifftshift(np.maximum(targets[index], 0))
            ratio_fft = cpu_fft2(target / np.maximum(prediction, 1e-30))
            correction += weights[index] * np.maximum(
                cpu_ifft2(np.conj(intensity_fft) * ratio_fft).real, 0,
            )
        updated = kernel * correction
        mass = updated.sum(axis=(-2, -1), keepdims=True)
        # A mode with no illumination cannot inform its kernel.
        kernel = np.divide(updated, mass, out=kernel.copy(), where=mass > 0)
    kernel = np.fft.fftshift(kernel, axes=(-2, -1))
    return kernel[0] if squeeze else kernel


def _observation_map(function, observations, workers):
    """
    Bound in-flight reconstructions and preserve deterministic result order.

    Context:
    Scheduling boundary for independent work: return results in requested observation order
    even when workers finish out of order. Bounded submission limits the number of large
    reconstruction buffers alive at once.
    """
    observations = list(observations)
    if workers == 1:
        for observation in observations:
            yield observation, function(observation)
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(observations), workers):
            batch = observations[start:start + workers]
            for observation, result in zip(batch, executor.map(function, batch)):
                yield observation, result


def _run_observation_update(*args, **kwargs):
    """
    Use thread-local FFT settings; never change module globals in workers.

    Context:
    Worker wrapper around one observation schedule. Configure FFT threads locally and attach
    execution timing; update shared gamma only after collecting worker results in the parent
    driver.
    """
    from scipy.fft import set_workers
    recipe = args[5]
    import threading
    started = time.perf_counter()
    with set_workers(int(recipe["fft_workers"])):
        output = _run_update_schedule(*args, **kwargs)
    finished = time.perf_counter()
    if output[1]:
        output[1][0]["execution"] = {
            "started": started, "finished": finished,
            "wall_seconds": finished - started,
            "thread_id": threading.get_ident(), "fft_workers": int(recipe["fft_workers"]),
        }
    return output


def _run_update_schedule(
    field,
    amplitude,
    supportmask,
    bsmask,
    schedule,
    recipe,
    phase_retrieval_kernel=None,
    gamma=None,
    return_gamma=False,
    fixed_masked_intensity=None,
    freeze_gamma=False,
):
    """
    Run the configured phase-retrieval stages for one observation.

    Context:
    Numerical stage runner, not a multi-state driver. Coherent stages pass the invalid mask
    through unchanged. Partial stages substitute masked intensity fills and use a fully
    observed effective target; callers own the lifetime of those fills.
    """
    has_rl = any(stage["RL_it"] > 0 and stage["RL_freq"] <= stage["Nit"]
                 for stage in schedule)
    if has_rl and not recipe.get("partial_coherence", False):
        raise ValueError(
            "This observation schedule does not support partial-coherence "
            "RL updates because no coherence kernel is supplied."
        )
    if (recipe["Nmodes"] > 1 and phase_retrieval_kernel is None
            and recipe.get("observation_workers", 1) == 1
            and not has_rl and not recipe.get("partial_coherence", False)):
        result = _run_modal_energy_update_schedule(
            field, amplitude, supportmask, bsmask, schedule, recipe,
            nmodes=recipe["Nmodes"], image_shape=amplitude.shape,
        )
        return (*result, gamma) if return_gamma else result
    if recipe["Nmodes"] > 1:
        if phase_retrieval_kernel is None:
            phase_retrieval_kernel = PhaseRtrv_core
    if phase_retrieval_kernel is None:
        phase_retrieval_kernel = PhaseRtrv_core
    stage_results = []
    recipe_steps, gamma = _compile_iteration_recipe(
        schedule,
        recipe,
        field=field,
        amplitude=amplitude,
        bsmask=bsmask,
        gamma=gamma,
        fixed_masked_intensity=fixed_masked_intensity,
        freeze_gamma=freeze_gamma,
    )

    for stage in recipe_steps:
        iterations = stage["Nit"]
        if stage["mode"] == "gradient_descent":
            raise ValueError("Gradient refinement is outside the paper implementation")

        kernel_args = {"Nmodes": recipe["Nmodes"]} if recipe["Nmodes"] > 1 else {}
        field, error, support_error, gamma_out = phase_retrieval_kernel(
            diffract=stage["amplitude"],
            mask=supportmask,
            mode=stage["mode"],
            Nit=iterations,
            beta_zero=stage["beta_zero"],
            beta_mode=stage["beta_mode"],
            alpha_zero=stage["alpha_zero"],
            alpha_mode=stage["alpha_mode"],
            Phase=field,
            seed=False,
            plot_every=recipe["plot_every"],
            bsmask=stage["bsmask"],
            real_object=False,
            average_img=min(max(1, recipe["average_img"]), iterations),
            Fourier_last=recipe["Fourier_last"],
            gamma=gamma if stage["use_rl"] else None,
            RL_freq=stage["RL_freq"],
            RL_it=stage["RL_it"],
            TV_freq=stage["TV_freq"],
            **kernel_args,
        )
        if gamma_out is not None and not freeze_gamma:
            gamma = gamma_out

        stage_results.append({
            **stage,
            "error": np.asarray(error),
            "support_error": np.asarray(support_error),
            "coherence": stage["coherence"],
        })

    return (field, stage_results, gamma) if return_gamma else (field, stage_results)


def _apply_measured_amplitudes(fields, amplitudes, bsmasks):
    """
    Reapply measured Fourier amplitudes outside invalid-pixel regions.

    Context:
    Final detector projection: replace amplitudes only at observed pixels. For multiple
    incoherent modes use one common scale factor based on summed modal power, preserving
    their relative weights.
    """
    constrained = np.asarray(fields).copy()
    if constrained.ndim == 4:
        total = np.sqrt(np.sum(np.abs(constrained) ** 2, axis=1))
        scale = np.divide(amplitudes, total,
                          out=np.ones_like(amplitudes), where=total > 1e-30)
        return constrained * np.where(bsmasks == 0, scale, 1.0)[:, None]
    observed = bsmasks == 0
    constrained[observed] = (
        amplitudes[observed] * np.exp(1j * np.angle(constrained[observed]))
    )
    return constrained


def _radially_normalize_startimage(field, hologram, mask_pixel):
    """Match total modal intensity to the valid detector's radial mean.

    One-pixel annuli are centered on the central detector pixel after crop/binning.
    Masked/nonfinite/negative data are excluded; measured zeros remain samples.
    Empty annuli interpolate between populated rings (nearest at either end).
    Phases and relative modal amplitudes are retained wherever power is nonzero.
    At exact field zeros, put the target amplitude in the first mode at phase zero.
    This is an initialization only, not a constraint on subsequent iterations.
    """
    measured = np.asarray(hologram, dtype=float)
    array = np.asarray(field, dtype=np.complex128)
    if measured.ndim != 2 or array.ndim not in (2, 3) or array.shape[-2:] != measured.shape:
        raise ValueError("Radial initialization requires a 2D hologram and matching field.")
    rows, columns = np.ogrid[:measured.shape[0], :measured.shape[1]]
    rings = np.floor(np.hypot(rows - measured.shape[0] // 2,
                             columns - measured.shape[1] // 2)).astype(int)
    valid = (np.asarray(mask_pixel) == 0) & np.isfinite(measured) & (measured >= 0)
    if not np.any(valid):
        raise ValueError("Radial initialization needs at least one valid detector pixel.")
    size = int(rings.max()) + 1
    counts = np.bincount(rings[valid], minlength=size)
    sums = np.bincount(rings[valid], weights=measured[valid], minlength=size)
    populated = counts > 0
    profile = np.interp(np.arange(size), np.flatnonzero(populated),
                        sums[populated] / counts[populated])
    target_amplitude = np.sqrt(profile[rings])
    modes = array[None] if array.ndim == 2 else array
    amplitude = np.sqrt(np.sum(np.abs(modes) ** 2, axis=0))
    scale = np.divide(target_amplitude, amplitude, out=np.zeros_like(amplitude),
                      where=amplitude > 0)
    result = modes * scale
    empty = amplitude == 0
    result[0, empty] = target_amplitude[empty]
    return result[0] if array.ndim == 2 else result


def _initialize_fields(
    supportmask,
    amplitudes,
    intensities,
    mask_stack,
    scale_fit="linear",
):
    """
    Create and approximately normalize one initial field per observation.

    Context:
    Single-mode support-based initialization. Each observation starts with the support
    transform and receives its own measured-amplitude scaling; the caller optionally applies
    radial normalization afterward.
    """
    n_observations = amplitudes.shape[0]
    start = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(supportmask)))
    fields = np.repeat(start[None], n_observations, axis=0).astype(
        np.complex128
    )
    for observation in range(n_observations):
        valid = (
            (mask_stack[observation] == 0)
            & (intensities[observation] > 0)
        )
        if not np.any(valid):
            continue
        measured = amplitudes[observation][valid].ravel()
        current = np.abs(fields[observation][valid]).ravel()
        if scale_fit == "through_origin":
            denominator = np.dot(measured, measured)
            scale = np.dot(measured, current) / denominator if denominator > 0 else 0
        elif measured.size >= 2 and np.ptp(measured) > 0 and np.ptp(current) > 0:
            scale = stats.linregress(measured, current).slope
        else:
            scale = 0
        if np.isfinite(scale) and scale > 1e-12:
            fields[observation] /= scale
    return fields


def _initialize_physical_modal_fields(supportmask, amplitudes, intensities,
                                      mask_stack, recipe):
    """
    Multimode initialization branch used by the general driver. support_fft transforms each
    modal support; the alternative delegates to the random-phase initializer. Scaling uses
    total modal power, not a coherent field sum.
    """
    if recipe["mode_initialization"] == "support_fft":
        support_modes = np.asarray(supportmask)
        base_modes = np.empty_like(support_modes, dtype=np.complex128)
        for mode_index, support_mode in enumerate(support_modes):
            base_modes[mode_index] = np.fft.ifftshift(
                np.fft.ifft2(np.fft.fftshift(support_mode))
            )
        fields = np.repeat(base_modes[None], amplitudes.shape[0], axis=0)
        for observation in range(amplitudes.shape[0]):
            valid = ((mask_stack[observation] == 0)
                     & (intensities[observation] > 0))
            if not np.any(valid):
                continue
            measured = amplitudes[observation][valid].ravel()
            current = np.sqrt(np.sum(np.abs(fields[observation]) ** 2, axis=0))[
                valid
            ].ravel()
            denominator = np.dot(measured, measured)
            scale = (np.dot(measured, current) / denominator
                     if denominator > 0 else 0)
            if np.isfinite(scale) and scale > 1e-12:
                fields[observation] /= scale
        return fields
    return _initialize_modal_fields(
        supportmask, amplitudes, intensities, mask_stack, recipe["Nmodes"],
        recipe["mode_initialization_seed"],
    )

# -----------------------------------------------------------------------------
# High-level retrieval algorithms
# -----------------------------------------------------------------------------


def general_phase_retrieval_algorithm(
    holograms,
    mask_pixel,
    supportmask,
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    saturated_states=None,
    general_recipe=None,
    start_fields=None,
    phase_retrieval_kernel=None,
    progress_callback=None,
):
    """
    Jointly retrieve arbitrary states, energies, polarizations, and beams.

    The algorithm alternates independent Fourier/support updates with the
    projection ``L_a = C_beam(a) + p_a R_state(a),energy(a)``.

    ``energy_labels`` may represent energy or any illumination category
    expected to change the material response. ``beam_labels`` represent
    conditions that change only the common illumination field. Polarization
    coefficients are normally +1 and -1 but may be any finite nonzero values.
    ``saturated_states`` optionally maps state labels to known +1/-1 reduced
    magnetization; a sequence is interpreted as +1 saturation.

    Returns
    -------
    fields : ndarray, shape (n_observations, nx, ny)
        Final Fourier-domain fields.
    components : dict
        Common beam fields, state-energy responses, and design diagnostics.
    bsmasks : ndarray, shape (n_observations, nx, ny)
        Invalid-pixel masks used by the reconstruction.
    errors : dict
        Per-stage errors and projection diagnostics.

    Context:
    Main orchestration for mixed metadata. Owns coherent masked-fill snapshots, per-
    observation/shared gamma, coherent refreshes, worker synchronization and physical-
    projection cadence. The low-level kernel only sees one observation at a time.
    """
    # Optional observer, outside the numerical recipe and never serialized.
    # Callbacks must not mutate the borrowed reconstruction arrays.
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    recipe = default_general_phase_retrieval_recipe()
    if general_recipe is not None:
        if not isinstance(general_recipe, dict):
            raise TypeError("general_recipe must be a dictionary.")
        unknown = set(general_recipe) - set(recipe)
        if unknown:
            raise ValueError(
                f"Unknown general recipe key(s): {sorted(unknown)}"
            )
        recipe.update(general_recipe)
    if saturated_states is not None:
        recipe["saturated_states"] = saturated_states

    mode_factors = _normalize_mode_factors(
        recipe["modes"],
        default_nmodes=int(recipe["Nmodes"]),
    )
    recipe["modes"] = list(mode_factors)
    recipe["Nmodes"] = len(mode_factors)

    # The physical object support is shared by all coherent modes. Keep its
    # geometry 2D; the modal Fourier kernel broadcasts it to each mode.
    preparation_recipe = dict(recipe, Nmodes=1)
    holograms, mask_pixel, supportmask, start_fields, input_geometry = geometry.prepare(
        holograms, mask_pixel, supportmask, preparation_recipe, start_fields
    )
    holograms = _as_energy_stack(holograms, name="holograms")
    n_observations, nx, ny = holograms.shape
    material_thickness = _recipe_material_thickness(recipe, input_geometry)
    if material_thickness is not None:
        material_thickness = np.fft.fftshift(material_thickness)
    if recipe.get("chemical_species") is not None:
        raise ValueError("Chemical species decomposition is outside the paper implementation")
        recipe = map_species_recipe(recipe, input_geometry=input_geometry)
    metadata = _normalize_metadata(
        state_labels,
        energy_labels,
        polarization_coefficients,
        beam_labels,
        n_observations,
    )
    _verify_recipe(
        recipe,
        n_observations,
        n_energies=len(metadata["energy_names"]),
    )

    supportmask = np.asarray(supportmask)
    if supportmask.shape != (nx, ny):
        raise ValueError("supportmask must have shape (nx, ny).")
    mode_support_shifts = [(0, 0)] * int(recipe["Nmodes"])
    if recipe["Nmodes"] > 1:
        if recipe["recenter_modal_supports"]:
            modal_supportmask, mode_support_shifts = geometry.recenter_modal_supports(
                supportmask, mode_factors, center=recipe["mode_support_center"],
            )
        elif recipe["mode_support_center"] == "components":
            modal_supportmask = _component_centered_modal_supports(
                supportmask, mode_factors,
            )
        else:
            modal_supportmask = _mode_supports(
                supportmask, mode_factors, (nx, ny),
            )
    else:
        modal_supportmask = supportmask
    thickness_supportmask = None
    if recipe["zero_thickness_outside_support"]:
        if recipe["projection_model"] != "physical_factorized":
            raise ValueError(
                "zero_thickness_outside_support requires projection_model='physical_factorized'."
            )
        if material_thickness is None:
            material_thickness = np.ones((nx, ny), dtype=float)
        thickness_supportmask = np.fft.fftshift(supportmask) != 0
        material_thickness = material_thickness * thickness_supportmask
        if not np.any(material_thickness > 0):
            raise ValueError("The support contains no material pixels for thickness fitting.")
    # PhaseRtrv_core applies support constraints in the shifted object frame.
    # The log-object projection uses the same frame via fft2(fftshift(field)).
    magnetization_supportmask = (
        np.fft.fftshift(supportmask)
        if recipe["zero_magnetization_outside_support"]
        else None
    )
    projection_supportmask = (
        np.fft.fftshift(supportmask)
        if (
            recipe["projection_constraints_inside_support_only"]
            or recipe["physical_constraints_inside_support_only"]
        )
        else None
    )

    # Convert measured intensities to amplitudes and invalid-pixel masks.
    amplitudes, intensities, bsmasks = _prepare_energy_amplitudes(
        holograms,
        mask_pixel,
        hologram_intensity_cutoff_vmin=recipe[
            "hologram_intensity_cutoff_vmin"
        ],
    )
    mask_stack = _as_energy_mask(
        mask_pixel,
        nE=n_observations,
        image_shape=(nx, ny),
    )

    # Initialize each observation independently unless fields are supplied.
    if start_fields is None:
        if recipe["Nmodes"] > 1:
            fields = _initialize_physical_modal_fields(
                modal_supportmask, amplitudes, intensities, mask_stack, recipe,
            )
        else:
            fields = _initialize_fields(
                supportmask, amplitudes, intensities, mask_stack,
                scale_fit=recipe["startimage_scale_fit"],
            )
        if recipe["startimage_radial_normalization"]:
            for observation in range(n_observations):
                fields[observation] = _radially_normalize_startimage(
                    fields[observation], holograms[observation], mask_stack[observation],
                )
    else:
        fields = np.asarray(start_fields, dtype=np.complex128).copy()
        expected = ((n_observations, recipe["Nmodes"], nx, ny) if recipe["Nmodes"] > 1
                    else holograms.shape)
        if fields.shape != expected:
            raise ValueError(
                f"start_fields must have shape {expected}."
            )

    errors = {
        "observation_steps": [],
        "projection_steps": [],
        "settings": recipe.copy(),
    }
    diagnostic_observation = recipe["projection_diagnostic_observation"]
    if diagnostic_observation is not None and (
            isinstance(diagnostic_observation, bool)
            or not isinstance(diagnostic_observation, (int, np.integer))
            or not 0 <= diagnostic_observation < n_observations):
        raise ValueError("projection_diagnostic_observation must index a selected observation.")
    errors["detector_constraint_diagnostics"] = []
    if diagnostic_observation is not None:
        yy, xx = np.indices((nx, ny))
        outer_pixels = np.hypot(yy - nx // 2, xx - ny // 2) > min(nx, ny) / 4
        explicit_outer = outer_pixels & (mask_stack[diagnostic_observation] != 0)
        observed_pixels = bsmasks[diagnostic_observation] == 0

    def record_checkpoint(stage, checkpoint_fields=None):
        if diagnostic_observation is None:
            return
        current = (fields if checkpoint_fields is None else checkpoint_fields)[diagnostic_observation]
        power = np.abs(current) ** 2
        primary = power[0] if power.ndim == 3 else power
        total = power.sum(axis=0) if power.ndim == 3 else power
        target = amplitudes[diagnostic_observation][observed_pixels]
        residual = np.sqrt(total[observed_pixels]) - target
        errors["detector_constraint_diagnostics"].append({
            "stage": stage, "observation": int(diagnostic_observation),
            "outer_masked_pixels": int(explicit_outer.sum()),
            "primary_outer_masked_mean": float(primary[explicit_outer].mean()) if explicit_outer.any() else None,
            "total_outer_masked_mean": float(total[explicit_outer].mean()) if explicit_outer.any() else None,
            "observed_amplitude_relative_error": float(np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-30)),
        })

    coherence_kernels = [None] * n_observations
    shared_coherence_kernel = None
    share_coherence = recipe["coherence_kernel_scope"] == "shared"
    start_time = time.time()
    outer_iterations = int(recipe["outer_iterations"])
    print(
        "Universal phase retrieval: starting physical reconstruction "
        f"with {n_observations} observations, "
        f"{len(metadata['energy_names'])} energies, and "
        f"{outer_iterations} outer loops",
        flush=True,
    )

    workers = min(int(recipe["observation_workers"]), n_observations)
    import inspect
    kernel_module = inspect.getmodule(phase_retrieval_kernel) if phase_retrieval_kernel else None
    if workers > 1 and (GPU if kernel_module is None else getattr(kernel_module, "GPU", GPU)):
        print("GPU backend: using one observation worker to avoid competing GPU jobs", flush=True)
        workers = 1
    gamma_strategy = recipe["shared_coherence_update"]
    round_coherence = (share_coherence and recipe["partial_coherence"]
                       and gamma_strategy != "sequential")
    print(f"Observation workers: {workers}; FFT threads per worker: {recipe['fft_workers']}; "
          f"shared gamma update: {gamma_strategy}", flush=True)
    projection_every, projection_start = _resolve_projection_cadence(
        recipe, default_every=n_observations,
    )
    parallel_joint = (round_coherence or (workers > 1 and
                      (not recipe["partial_coherence"] or not share_coherence
                       or recipe["freeze_coherence_after_reference"])))
    if parallel_joint and recipe["projection_model"] != "none" and recipe["projection_relaxation"] > 0:
        if projection_every % n_observations or projection_start % n_observations:
            raise ValueError("Parallel rounds require physical projection boundaries at complete observation sweeps.")
    state_weights = _observation_weights(recipe["observation_weights"], n_observations)

    fixed_masked_intensities = [None] * n_observations
    if recipe["freeze_coherence_after_reference"] and not share_coherence:
        raise ValueError("Freezing reference coherence requires coherence_kernel_scope='shared'.")

    fields_initial = fields.copy()
    frozen_indices = []
    if recipe["freeze_saturated_fields"]:
        saturated_names = recipe["saturated_states"]
        if not saturated_names:
            raise ValueError("freeze_saturated_fields requires saturated_states.")
        frozen_indices = [index for index, state in enumerate(metadata["states"])
                          if state in saturated_names]
        if not frozen_indices:
            raise ValueError("No observations match saturated_states for freezing.")
    frozen_fields = fields[frozen_indices].copy()

    def restore_frozen_observations(current_fields):
        if not frozen_indices:
            return
        if (np.asarray(current_fields).ndim == 4
                and recipe["constrain_nonphysical_modes_common"]):
            # Freeze the physically modelled primary mode. Higher modes must
            # remain common across states, including the saturated state.
            current_fields[frozen_indices, 0] = frozen_fields[:, 0]
        else:
            current_fields[frozen_indices] = frozen_fields

    inner_schedule = _build_update_schedule(recipe, name="inner")
    rng = np.random.default_rng(recipe["random_seed"])
    components = {"projection_model": recipe["projection_model"]}
    projection_every, projection_start = _resolve_projection_cadence(
        recipe,
        default_every=n_observations,
    )
    completed_updates = 0
    fixed_targets = (np.where(bsmasks != 0, np.asarray(fixed_masked_intensities), amplitudes ** 2)
                     if round_coherence else None)

    # Alternate detector/support updates with the metadata-aware object model.
    # Every projection sees the complete current observation stack.
    for outer in range(outer_iterations):
        if outer + 1 in recipe["coherent_refresh_rounds"]:
            # Complete an all-state coherent sweep before capturing new fills.
            # Retain gamma separately: it must not blur the coherent update.
            refresh_stage = dict(inner_schedule[0],
                                 mode=recipe["coherent_refresh_mode"],
                                 Nit=recipe["coherent_refresh_iterations"], RL_it=0)
            def refresh_observation(observation):
                if observation in frozen_indices:
                    return fields[observation], []
                return _run_observation_update(
                    fields[observation], amplitudes[observation], modal_supportmask,
                    bsmasks[observation], [refresh_stage], recipe,
                    phase_retrieval_kernel=phase_retrieval_kernel,
                    gamma=None, return_gamma=False, fixed_masked_intensity=None,
                )
            for observation, (field, results) in _observation_map(
                    refresh_observation, range(n_observations), workers):
                print(f"  Coherent refresh observation {observation + 1}/{n_observations}", flush=True)
                fields[observation] = field
                power = np.abs(field) ** 2
                fixed_masked_intensities[observation] = (
                    power.sum(axis=0) if power.ndim == 3 else power)
                errors.setdefault("coherent_refresh_steps", []).append({
                    "outer_round": outer + 1, "observation": observation,
                    "frozen": observation in frozen_indices, "steps": results,
                })
            if round_coherence:
                fixed_targets = np.where(bsmasks != 0,
                                         np.asarray(fixed_masked_intensities), amplitudes ** 2)
            record_checkpoint("coherent_refresh_before_round_" + str(outer + 1))
            print(f"Refreshed masked intensities before partial-coherence round {outer + 1}", flush=True)
        print(
            f"Universal phase retrieval: outer loop "
            f"{outer + 1}/{outer_iterations}",
            flush=True,
        )
        order = (
            rng.permutation(n_observations)
            if recipe["shuffle_observations"]
            else np.arange(n_observations)
        )
        round_results = {}
        if parallel_joint:
            round_gamma = shared_coherence_kernel
            if round_coherence and round_gamma is None:
                round_kernel = np.full(np.shape(fields[0]), 2e-6, dtype=float)
                round_kernel[..., round_kernel.shape[-2] // 2, round_kernel.shape[-1] // 2] = 0.7
                round_gamma = _normalize_coherence_kernel(round_kernel)

            def update_joint(observation):
                gamma = round_gamma if share_coherence else coherence_kernels[observation]
                if observation in frozen_indices:
                    # Frozen fields still inform the shared illumination estimate.
                    if round_coherence and gamma_strategy == "average":
                        fitted_gamma = _pooled_coherence_update(
                            fields[observation:observation + 1],
                            fixed_targets[observation:observation + 1], gamma,
                            recipe["coherence_round_iterations"],
                        )
                        return fields[observation], [], fitted_gamma
                    return fields[observation], [], gamma
                return _run_observation_update(
                    fields[observation], amplitudes[observation], modal_supportmask,
                    bsmasks[observation], inner_schedule, recipe,
                    phase_retrieval_kernel=phase_retrieval_kernel,
                    gamma=None if gamma is None else gamma.copy(), return_gamma=True,
                    fixed_masked_intensity=fixed_masked_intensities[observation],
                    freeze_gamma=(recipe["freeze_coherence_after_reference"] or
                                  (round_coherence and gamma_strategy == "pooled")),
                )

            gamma_sum = None
            for position, (observation, (field, results, gamma)) in enumerate(
                    _observation_map(update_joint, order, workers), start=1):
                print(f"  Updating observation {position}/{n_observations} "
                      f"(index {int(observation)}, parallel round)", flush=True)
                fields[observation] = field
                round_results[observation] = results
                if round_coherence and gamma_strategy == "average":
                    contribution = state_weights[observation] * _normalize_coherence_kernel(gamma)
                    gamma_sum = contribution if gamma_sum is None else gamma_sum + contribution
                elif not share_coherence:
                    coherence_kernels[observation] = gamma
            if round_coherence:
                if gamma_strategy == "average":
                    shared_coherence_kernel = _normalize_coherence_kernel(gamma_sum)
                else:
                    shared_coherence_kernel = _pooled_coherence_update(
                        fields, fixed_targets, round_gamma,
                        recipe["coherence_round_iterations"], state_weights,
                    )
                errors.setdefault("coherence_rounds", []).append({
                    "outer": outer, "strategy": gamma_strategy,
                    "observations": n_observations,
                })
        for position, observation in enumerate(order, start=1):
            if not parallel_joint:
                print(
                    "  Updating observation "
                    f"{position}/{n_observations} "
                    f"(index {int(observation)}, "
                    f"energy={metadata['energies'][observation]}, "
                    f"polarization={metadata['polarizations'][observation]:g})",
                    flush=True,
                )
            if parallel_joint:
                results = round_results[observation]
            elif observation in frozen_indices:
                results = []
            else:
                fields[observation], results, updated_gamma = _run_observation_update(
                    fields[observation],
                    amplitudes[observation],
                    modal_supportmask,
                    bsmasks[observation],
                    inner_schedule,
                    recipe,
                    phase_retrieval_kernel=phase_retrieval_kernel,
                    gamma=(shared_coherence_kernel if share_coherence
                           else coherence_kernels[observation]),
                    return_gamma=True,
                    fixed_masked_intensity=fixed_masked_intensities[observation],
                    freeze_gamma=recipe["freeze_coherence_after_reference"],
                )
                if share_coherence:
                    shared_coherence_kernel = updated_gamma
                else:
                    coherence_kernels[observation] = updated_gamma
            for result in results:
                errors["observation_steps"].append({
                    "outer": outer,
                    "observation": int(observation),
                    "state": metadata["states"][observation],
                    "energy": metadata["energies"][observation],
                    "polarization": metadata["polarizations"][observation],
                    "beam": metadata["beams"][observation],
                    "stage": "joint",
                    **result,
                })

            if recipe["projection_model"] == "none":
                continue
            completed_updates += 1
            if not _projection_is_due(
                completed_updates,
                projection_start,
                projection_every,
                total_updates=outer_iterations * n_observations,
            ):
                continue
            if recipe["projection_relaxation"] == 0:
                continue

            record_checkpoint("joint_detector_support")
            print("  Applying joint physical projection", flush=True)
            fields, components = _project_physical_modes(
                fields,
                metadata["states"],
                metadata["energies"],
                metadata["polarizations"],
                metadata["beams"],
                weights=recipe["observation_weights"],
                relaxation=recipe["projection_relaxation"],
                rank_deficient=recipe["rank_deficient"],
                projection_model=recipe["projection_model"],
                saturated_states=recipe["saturated_states"],
                physical_iterations=recipe["physical_iterations"],
                recipe=recipe,
                log_floor=recipe["log_floor"],
                magnetization_supportmask=magnetization_supportmask,
                projection_supportmask=projection_supportmask,
                material_thickness=material_thickness,
                fit_material_thickness=recipe["fit_material_thickness"],
                thickness_supportmask=thickness_supportmask,
                initial_components=(components if "magnetization" in components
                                    and recipe.get("chemical_species") is None
                                    and recipe["projection_model"] == "physical_factorized" else None),
                return_components=True,
                diagnostic_callback=record_checkpoint,
            )
            restore_frozen_observations(fields)
            if progress_callback is not None:
                progress_callback(dict(stage="physical_projection", outer_round=outer + 1,
                    fields=fields, components=components, recipe=recipe,
                    supportmask=supportmask, state_labels=metadata["states"],
                    energy_labels=metadata["energies"]))
            errors["projection_steps"].append({
                "outer": outer,
                "observation": int(observation),
                "completed_update": completed_updates,
                "fit_residual_rms": components["fit_residual_rms"],
                "projection_model": components["projection_model"],
                "design_rank": components.get("design_rank"),
                "identifiable": components.get("identifiable"),
            })
            print(
                "  Joint physical projection complete "
                f"(fit residual RMS={components['fit_residual_rms']:.6g})",
                flush=True,
            )

    # A zero relaxation is a true bypass: avoid the potentially expensive
    # focus, physical fit, and inverse propagation when no result is applied.
    if (recipe["projection_model"] != "none"
            and recipe["final_projection_relaxation"] > 0):
        print(
            "Universal phase retrieval: computing final physical model",
            flush=True,
        )
        final_projected_fields, components = _project_physical_modes(
            fields,
            metadata["states"],
            metadata["energies"],
            metadata["polarizations"],
            metadata["beams"],
            weights=recipe["observation_weights"],
            relaxation=recipe["final_projection_relaxation"],
            rank_deficient=recipe["rank_deficient"],
            projection_model=recipe["projection_model"],
            saturated_states=recipe["saturated_states"],
            physical_iterations=recipe["physical_iterations"],
            recipe=recipe,
            log_floor=recipe["log_floor"],
            magnetization_supportmask=magnetization_supportmask,
            projection_supportmask=projection_supportmask,
            material_thickness=material_thickness,
            fit_material_thickness=recipe["fit_material_thickness"],
            thickness_supportmask=thickness_supportmask,
            initial_components=(components if "magnetization" in components
                                and recipe.get("chemical_species") is None
                                and recipe["projection_model"] == "physical_factorized" else None),
            return_components=True,
        )
        fields = final_projected_fields
        restore_frozen_observations(fields)
        if progress_callback is not None:
            progress_callback(dict(stage="final_physical_projection", outer_round=outer_iterations,
                fields=fields, components=components, recipe=recipe,
                supportmask=supportmask, state_labels=metadata["states"],
                energy_labels=metadata["energies"]))
    components["final_projection_relaxation"] = recipe["final_projection_relaxation"]
    components["final_projection_skipped"] = bool(
        recipe["projection_model"] == "none"
        or recipe["final_projection_relaxation"] == 0
    )

    # Optionally finish exactly on the measured Fourier amplitudes.
    record_checkpoint("before_final_fourier")
    if recipe["final_fourier_constraint"]:
        fields = _apply_measured_amplitudes(fields, amplitudes, bsmasks)
        restore_frozen_observations(fields)
        components["final_fourier_constraint_applied"] = True
    else:
        components["final_fourier_constraint_applied"] = False
    if (np.asarray(fields).ndim == 4
            and recipe["constrain_nonphysical_modes_common"]):
        # A final observation-specific amplitude projection can separate the
        # higher modes again. Finish on the requested common-mode constraint.
        gain_scale = np.sqrt(components.get("illumination_intensity_factors", np.ones(len(fields))))[:, None, None, None]
        fields, common_groups = _project_nonphysical_modes_common(
            fields / gain_scale,
            metadata["energies"],
            metadata["beams"],
            recipe,
            weights=recipe["observation_weights"],
        )
        fields *= gain_scale
        restore_frozen_observations(fields)
        components["nonphysical_mode_common_groups"] = common_groups
        components["common_mode_applied_after_final_fourier"] = bool(
            recipe["final_fourier_constraint"]
        )
    record_checkpoint("returned_final")
    components["Nmodes"] = int(recipe["Nmodes"])
    components["modes"] = list(mode_factors)
    components["modal_supportmask_used"] = np.asarray(modal_supportmask).copy()
    components["mode_support_shifts"] = mode_support_shifts
    components["frozen_saturated_observations"] = frozen_indices
    components["observation_workers"] = workers
    components["shared_coherence_update"] = gamma_strategy
    components["coherence_kernel_scope"] = recipe["coherence_kernel_scope"]
    components["partial_coherence"] = (
        shared_coherence_kernel is not None if share_coherence
        else any(kernel is not None for kernel in coherence_kernels)
    )
    if components["partial_coherence"]:
        if share_coherence:
            components["coherence_kernel"] = shared_coherence_kernel
        else:
            components["coherence_kernels"] = np.stack(coherence_kernels)

    executions = [record["execution"] for record in errors["observation_steps"] if "execution" in record]
    events = sorted([(item["started"], 1) for item in executions]
                    + [(item["finished"], -1) for item in executions])
    active = maximum = 0
    for _, change in events:
        active += change
        maximum = max(maximum, active)
    errors["execution_summary"] = {
        "requested_workers": int(recipe["observation_workers"]),
        "effective_workers": workers, "max_concurrent_updates": maximum,
        "worker_threads_used": len({item["thread_id"] for item in executions}),
        "summed_update_wall_seconds": sum(item["wall_seconds"] for item in executions),
    }
    print(f"Measured concurrent observation updates: {maximum} (configured {workers})", flush=True)
    errors["runtime_seconds"] = float(np.round(time.time() - start_time, 3))
    print(
        "Universal phase retrieval: complete in "
        f"{errors['runtime_seconds']:.3f} s",
        flush=True,
    )
    if progress_callback is not None:
        progress_callback(dict(stage="final", outer_round=outer_iterations,
            fields=fields, components=components, recipe=recipe,
            supportmask=supportmask, state_labels=metadata["states"],
            energy_labels=metadata["energies"]))
    return input_geometry.finish(fields, fields_initial, components, bsmasks, errors)


_PHYSICAL_MODELS = {
    "physical_factorized",
    "none",
}


def default_universal_phase_retrieval_recipe():
    """Return defaults for the universal reconstruction driver."""
    recipe = default_general_phase_retrieval_recipe()
    recipe.update({
        # The physical model is the general mixed-dataset default.
        "projection_model": "physical_factorized",
        # Multi-energy-only projection controls.
        "rank": 1,
        "projection_static_mode": "mean",
        "spectral_constraint": "free",
        "absorption_part": "real",
    })
    return recipe


def _coerce_mode_factors(factors):
    """Normalize mode factors to a non-empty 1D list and ensure a valid mode count."""
    if factors is None:
        return None, 1
    if isinstance(factors, np.ndarray):
        if factors.ndim == 0:
            factors = [factors.item()]
        elif factors.ndim != 1:
            raise ValueError("modes must be a nonempty 1D list of nonnegative support factors")
        else:
            factors = list(factors)
    elif isinstance(factors, (int, float, np.number)) and not isinstance(factors, bool):
        factors = [factors]
    elif not isinstance(factors, (list, tuple)):
        try:
            factors = list(factors)
        except TypeError as exc:
            raise ValueError(
                "modes must be a nonempty list of nonnegative support factors"
            ) from exc
    if not factors or any(
        isinstance(v, bool) or not isinstance(v, (int, float, np.number))
        or not np.isfinite(v) or v < 0 for v in factors
    ):
        raise ValueError("modes must be a nonempty list of nonnegative support factors")
    return list(factors), len(factors)


def universal_phase_retrieval_algorithm(
    holograms,
    mask_pixel,
    supportmask,
    state_labels,
    energy_labels,
    polarization_coefficients,
    illumination_labels,
    saturated_states=None,
    universal_recipe=None,
    start_fields=None,
    phase_retrieval_kernel=None,
    progress_callback=None,
):
    """
    Reconstruct a metadata-described list of diffraction measurements.

    Use ``physical_factorized`` for mixed energy/polarization/state/beam data.
    Use ``none`` for independent phase retrieval without a joint object model.

    Returns
    -------
    fields : ndarray
        Final Fourier-domain fields, one per input hologram.
    fields_initial : ndarray
        Fourier-domain fields before the main joint projection loop.
    components : dict
        Components and identifiability diagnostics from the selected model.
    bsmasks : ndarray
        Observation-specific invalid-pixel masks.
    errors : dict
        Per-stage errors, settings, metadata, and projection diagnostics.

    Context:
    Public notebook entry point. Supply intensity images plus
    state/energy/polarization/illumination metadata. Returns final fields,
    the pre-projection snapshot, fitted components, effective masks and diagnostics.
    """
    recipe = default_universal_phase_retrieval_recipe()
    if universal_recipe is not None:
        if not isinstance(universal_recipe, dict):
            raise TypeError("universal_recipe must be a dictionary.")
        unknown = set(universal_recipe) - set(recipe)
        if unknown:
            raise ValueError(
                f"Unknown universal recipe key(s): {sorted(unknown)}"
            )
        recipe.update(universal_recipe)
    if saturated_states is not None:
        recipe["saturated_states"] = saturated_states

    if recipe["modes"] is not None:
        recipe["modes"], recipe["Nmodes"] = _coerce_mode_factors(recipe["modes"])
    if (
        isinstance(recipe["Nmodes"], bool)
        or not isinstance(recipe["Nmodes"], (int, np.integer))
        or recipe["Nmodes"] < 1
    ):
        raise ValueError("Nmodes must be a positive integer")

    holograms = _as_energy_stack(
        holograms,
        name="holograms",
    )
    metadata = _normalize_metadata(
        state_labels,
        energy_labels,
        polarization_coefficients,
        illumination_labels,
        holograms.shape[0],
    )
    model = str(recipe["projection_model"]).lower()
    if recipe.get("chemical_species") is not None and model != "physical_factorized":
        raise ValueError("chemical_species requires projection_model='physical_factorized'.")
    print(
        "Universal phase retrieval: dispatching "
        f"{holograms.shape[0]} observations with "
        f"projection_model={model!r}",
        flush=True,
    )

    if model not in _PHYSICAL_MODELS:
        raise ValueError("projection_model must be 'physical_factorized' or 'none'.")

    physical_recipe = recipe.copy()
    general_recipe = {key: physical_recipe[key] for key in default_general_phase_retrieval_recipe()}
    fields, fields_initial, components, bsmasks, errors = (
        general_phase_retrieval_algorithm(
            holograms,
            mask_pixel,
            supportmask,
            metadata["states"],
            metadata["energies"],
            metadata["polarizations"],
            metadata["beams"],
            saturated_states=physical_recipe["saturated_states"],
            general_recipe=general_recipe,
            progress_callback=progress_callback,
            start_fields=start_fields,
            phase_retrieval_kernel=phase_retrieval_kernel,
        )
    )

    # Preserve the complete observation description in every result.
    metadata_summary = {
        "state_labels": metadata["states"].copy(),
        "energy_labels": metadata["energies"].copy(),
        "polarization_coefficients": metadata["polarizations"].copy(),
        "illumination_labels": metadata["beams"].copy(),
    }
    components["universal_projection_model"] = model
    components["observation_metadata"] = metadata_summary
    errors["observation_metadata"] = metadata_summary
    errors["universal_settings"] = recipe.copy()
    return fields, fields_initial, components, bsmasks, errors



# Public names used by notebooks that construct explicit modal supports.
mode_supports = _mode_supports
multimode_phase_retrieval_kernel = PhaseRtrv_core


# Multimode spectral projections and retrieval use the same local kernels.







def _initialize_modal_fields(
    supportmask,
    amplitudes,
    intensities,
    mask_stack,
    nmodes,
    random_seed,
):
    """Create normalized, nondegenerate modal fields at every energy."""
    n_energy, nx, ny = amplitudes.shape
    support_modes = _as_modes(
        supportmask,
        nmodes,
        (nx, ny),
        "supportmask",
        dtype=np.asarray(supportmask).dtype,
    )
    base_modes = np.empty((nmodes, nx, ny), dtype=np.complex128)
    for mode_index in range(nmodes):
        base_modes[mode_index] = np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(support_modes[mode_index]))
        )

    if nmodes > 1:
        rng = np.random.default_rng(random_seed)
        modal_phase = np.exp(
            1j
            * rng.uniform(
                -np.pi,
                np.pi,
                (nmodes, nx, ny),
            )
        )
        return (
            amplitudes[:, None, :, :]
            * modal_phase[None, :, :, :]
            / np.sqrt(nmodes)
        )

    fields = np.repeat(base_modes[None, :, :, :], n_energy, axis=0)

    for energy_index in range(n_energy):
        valid = (
            (mask_stack[energy_index] == 0)
            & (intensities[energy_index] > 0)
        )
        if not np.any(valid):
            continue
        measured = amplitudes[energy_index][valid].ravel()
        current = np.sqrt(
            np.sum(np.abs(fields[energy_index]) ** 2, axis=0)
        )[valid].ravel()
        if (
            measured.size >= 2
            and np.ptp(measured) > 0
            and np.ptp(current) > 0
        ):
            fit = stats.linregress(measured, current)
            if abs(fit.slope) > 1e-12:
                fields[energy_index] /= fit.slope

    return fields


def _run_modal_energy_update_schedule(
    field,
    amplitude,
    supportmask,
    bsmask,
    schedule,
    recipe,
    nmodes,
    image_shape,
    phase_retrieval_kernel=None,
):
    """Run all scheduled multimode updates sequentially for one energy."""
    stage_results = []
    for stage_index, stage in enumerate(schedule):
        mode = stage["mode"]
        Nit = stage["Nit"]
        if stage["RL_it"] > 0 and stage["RL_freq"] <= Nit:
            raise ValueError(
                "This energy schedule does not support partial-coherence RL "
                "updates because no coherence kernel is supplied."
            )
        if mode == "gradient_descent":
            result, err_d, err_s, _ = _refine_modes_gradient(
                field,
                amplitude,
                supportmask,
                bsmask,
                nmodes=nmodes,
                image_shape=image_shape,
                Nit=Nit,
                learning_rate=make_beta_schedule(
                    stage["beta_mode"],
                    Nit,
                    stage["beta_zero"],
                ),
                support_weight=make_alpha_schedule(
                    stage["alpha_mode"],
                    Nit,
                    stage["alpha_zero"],
                ),
                Fourier_last=recipe["Fourier_last"],
            )
        else:
            result, err_d, err_s, _ = (phase_retrieval_kernel or PhaseRtrv_core_multimode)(
                diffract=amplitude,
                mask=supportmask,
                mode=mode,
                Nit=Nit,
                beta_zero=stage["beta_zero"],
                beta_mode=stage["beta_mode"],
                alpha_zero=stage["alpha_zero"],
                alpha_mode=stage["alpha_mode"],
                Phase=field,
                seed=False,
                plot_every=recipe["plot_every"],
                bsmask=bsmask,
                real_object=False,
                average_img=min(max(1, recipe["average_img"]), Nit),
                Fourier_last=recipe["Fourier_last"],
                gamma=None,
                RL_freq=stage["RL_freq"],
                RL_it=stage["RL_it"],
                TV_freq=stage["TV_freq"],
                Nmodes=nmodes,
            )
        field = _as_modes(
            result,
            nmodes,
            image_shape,
            "retrieved fields",
            dtype=np.complex128,
        )
        stage_results.append(
            {
                "schedule_stage": stage_index,
                "mode": mode,
                "Nit": Nit,
                "beta_zero": stage["beta_zero"],
                "beta_mode": stage["beta_mode"],
                "alpha_zero": stage["alpha_zero"],
                "alpha_mode": stage["alpha_mode"],
                "TV_freq": stage["TV_freq"],
                "RL_it": stage["RL_it"],
                "RL_freq": stage["RL_freq"],
                "Nmodes": nmodes,
                "error": np.asarray(err_d),
                "support_error": np.asarray(err_s),
            }
        )
    return field, stage_results

