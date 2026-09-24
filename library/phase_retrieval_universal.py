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

with ``-1 <= mz <= 1``. By default ``t=1``; optional material inputs set known
vacuum holes to ``t=0``, supply relative thickness, or fit a shared nonnegative
relative thickness map. The scalar spectral responses absorb the reference
thickness and wave number. Optional saturated states fix an entire ``mz`` map
to +1 or -1 within material. Absolute thickness and refractive index remain
coupled unless external information fixes their scale.

The module contains its own phase-retrieval kernel, schedules, multi-energy
projectors, and metadata-aware physical projectors. It does not import any
other phase-retrieval library. Selecting ``svd`` or ``rank1_spectral`` for a
pure energy scan remains numerically equivalent to the corresponding
multi-energy implementation, while ``physical_factorized`` handles mixed
state/polarization/energy/illumination datasets.
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

import matplotlib.pyplot as plt

from scipy import stats

try:
    from . import kramers_kronig as kk
except ImportError:
    import kramers_kronig as kk

try:
    from . import phase_retrieval_gradient as gradient
except ImportError:
    import phase_retrieval_gradient as gradient

    
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


#############################################################
#       PHASE RETRIEVAL Algorithm
# ############################################################


def default_phase_retrieval_recipe():
    """
    Return the default flat phase-retrieval recipe.

    Each index of the list-valued entries defines one reconstruction step. The
    coherence model is not specified explicitly: it is inferred inside
    ``PhaseRtrv_core`` from ``RL_its`` and ``RL_freqs``. A step is full-coherence
    when ``RL_its[i] == 0`` or ``RL_freqs[i] > number_iterations[i]``; otherwise
    Richardson-Lucy updates are enabled.
    """
    recipe = {
        "algorithm_list": ["HAPRE", "ER", "ER", "HAPRE", "ER", "ER"],
        "number_iterations": [700, 50, 50, 700, 50, 50],
        "helicity": ["pos", "pos", "neg", "pos", "pos", "neg"],

        "beta_zero": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "beta_mode": ["arctan", "const", "const", "arctan", "const", "const"],

        # alpha controls the optional TV descent step in PhaseRtrv_core.
        # alpha_zero=0 disables TV regularization for the default recipe.
        "alpha_zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "alpha_mode": ["const", "const", "const", "const", "const", "const"],

        # RL_its=0 or RL_freq>Nit means full coherence. Nonzero RL_its with
        # RL_freq<=Nit means partial coherence with Richardson-Lucy updates.
        "RL_its": [0, 0, 0, 50, 50, 50],
        "RL_freqs": [1e9, 1e9, 1e9, 20, 20, 20],
        "TV_freqs": [1e9, 1e9, 1e9, 1e9, 1e9, 1e9],

        "plot_every": [349, 24, 24, 349, 24, 24],
        "average_img": [30, 30, 30, 30, 30, 30],
        "Fourier_last": [True, True, True, True, True, True],
        "output": [False, False, False, False, True, True],

        "hologram_intensity_cutoff_vmin": -1,
        "Startimage": [None, "pos", "pos", "pos", "pos", "pos"],
        "Startgamma": [None,  None,  None,  None, "pos", "pos"],
    }
    return recipe


def _default_output_flags(helicity):
    """Return True for the last recipe step of each helicity."""
    flags = [False] * len(helicity)
    for h in ("pos", "neg"):
        for index in range(len(helicity) - 1, -1, -1):
            if helicity[index] == h:
                flags[index] = True
                break
    return flags


def _resolve_start_field(start_spec, default_field, latest, name):
    """
    Resolve a recipe Startimage/Startgamma entry.

    start_spec can be:
      - None: use the default support-based initialization
      - np.ndarray: use this array directly
      - "pos": use latest["pos"]
      - "neg": use latest["neg"]
    """

    if start_spec is None:
        return default_field.copy()

    if isinstance(start_spec, np.ndarray):
        return start_spec.copy()

    if isinstance(start_spec, str) and start_spec in {"pos", "neg"}:
        if latest[start_spec] is None:
            raise ValueError(
                f"{name} requested latest '{start_spec}', "
                f"but no previous {start_spec} result exists."
            )
        return latest[start_spec].copy()

    raise ValueError(
        f"Invalid {name} entry: {start_spec!r}. "
        "Allowed values are None, np.ndarray, 'pos', or 'neg'."
    )


def _verify_valid_phase_retrieval_recipe(recipe):
    """
    Validate the flat phase-retrieval recipe.

    The reconstruction recipe is represented by parallel lists. This function
    checks that all step-wise lists have identical length and that all requested
    algorithms, helicities, iteration counts, RL parameters, and bookkeeping
    options are valid before a long reconstruction starts.
    """
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

    required_list_keys = [
        "algorithm_list",
        "number_iterations",
        "helicity",
        "beta_zero",
        "beta_mode",
        "alpha_zero",
        "alpha_mode",
        "RL_its",
        "RL_freqs",
        "TV_freqs",
        "plot_every",
        "average_img",
        "Fourier_last",
        "output",
    ]

    lengths = []

    for key in required_list_keys:
        if key not in recipe:
            raise ValueError(f"Missing recipe key: {key}")

        if not isinstance(recipe[key], list):
            raise ValueError(f"Recipe key '{key}' must be a list.")

        lengths.append(len(recipe[key]))

    if len(set(lengths)) != 1:
        raise ValueError(
            "All recipe list entries must have the same length. "
            f"Got lengths: {dict(zip(required_list_keys, lengths))}"
        )

    if lengths[0] == 0:
        raise ValueError("The recipe must contain at least one reconstruction step.")

    invalid_algorithms = [
        alg for alg in recipe["algorithm_list"] if alg not in allowed_algorithms
    ]

    if invalid_algorithms:
        raise ValueError(
            f"Invalid algorithm(s): {invalid_algorithms}. "
            f"Allowed algorithms are: {sorted(allowed_algorithms)}"
        )

    invalid_helicity = [h for h in recipe["helicity"] if h not in {"pos", "neg"}]

    if invalid_helicity:
        raise ValueError(
            f"Invalid helicity value(s): {invalid_helicity}. "
            "Allowed values are 'pos' and 'neg'."
        )

    if not all(isinstance(n, int) and n > 0 for n in recipe["number_iterations"]):
        raise ValueError("All number_iterations values must be positive integers.")

    if not all(n >= 0 for n in recipe["RL_its"]):
        raise ValueError("All RL_its values must be >= 0.")

    if not all(f > 0 for f in recipe["RL_freqs"]):
        raise ValueError("All RL_freqs values must be > 0.")
    
    if not all(f > 0 for f in recipe["TV_freqs"]):
        raise ValueError("All TV_freqs values must be > 0.")
    
    if not all(n > 0 for n in recipe["plot_every"]):
        raise ValueError("All plot_every values must be > 0.")

    if not all(n > 0 for n in recipe["average_img"]):
        raise ValueError("All average_img values must be > 0.")

    if not all(isinstance(flag, bool) for flag in recipe["Fourier_last"]):
        raise ValueError("All Fourier_last values must be bool.")

    if not all(isinstance(flag, bool) for flag in recipe["output"]):
        raise ValueError("All output values must be bool.")

    for key in ["Startimage", "Startgamma"]:
        if key not in recipe:
            raise ValueError(f"Missing recipe key: {key}")

        if not isinstance(recipe[key], list):
            raise ValueError(f"Recipe key '{key}' must be a list.")

        if len(recipe[key]) != len(recipe["algorithm_list"]):
            raise ValueError(
                f"Recipe key '{key}' must have same length as algorithm_list."
            )
        
def _scale_phase_between_helicities(
    phase,
    source_helicity,
    target_helicity,
    intensity_data,
    bsmasks,
    use_offset=False,
    crop=0,
    plot=False,
    verbose=False,
):
    """
    Rescale a reconstruction when it is reused as the starting guess for the
    opposite helicity.

    The scale factor is estimated from a linear fit of the measured hologram
    intensities, while excluding invalid pixels from both helicities using
    bsmask_p/bsmask_n.

    Parameters
    ----------
    phase : np.ndarray
        Previous reconstructed complex Fourier-domain field.
    source_helicity, target_helicity : {"pos", "neg"}
        Helicity of the previous reconstruction and the target step.
    intensity_data : dict
        {"pos": pos_input, "neg": neg_input}
    bsmasks : dict
        {"pos": bsmask_p, "neg": bsmask_n}
    use_offset : bool
        If True, also apply the fitted offset approximately at intensity level.
        For phase/amplitude fields, the offset treatment is approximate and is
        usually better left False.
    crop : int
        Number of pixels excluded from every image edge before fitting.
    plot, verbose : bool
        Optionally display or print the fitted intensity relation.

    Returns
    -------
    scaled_phase : np.ndarray
        Rescaled complex reconstruction.
    """

    if phase is None or source_helicity is None or source_helicity == target_helicity:
        return phase

    if source_helicity not in {"pos", "neg"} or target_helicity not in {"pos", "neg"}:
        raise ValueError(
            f"Invalid helicity conversion: {source_helicity!r} -> {target_helicity!r}"
        )

    source_intensity = intensity_data[source_helicity].copy()
    target_intensity = intensity_data[target_helicity].copy()

    valid = (bsmasks[source_helicity] == 0) & (bsmasks[target_helicity] == 0)

    if not isinstance(crop, int) or crop < 0:
        raise ValueError("crop must be a non-negative integer.")
    if crop:
        if 2 * crop >= min(source_intensity.shape):
            raise ValueError("crop removes the complete intensity image.")
        interior = np.zeros_like(valid, dtype=bool)
        interior[crop:-crop, crop:-crop] = True
        valid &= interior

    xdata = source_intensity[valid].astype(float, copy=False)
    ydata = target_intensity[valid].astype(float, copy=False)
    finite = np.isfinite(xdata) & np.isfinite(ydata)
    xdata = xdata[finite]
    ydata = ydata[finite]

    if xdata.size < 2:
        raise ValueError(
            "At least two jointly valid pixels are required to scale helicities."
        )

    source_variance = np.var(xdata)
    if source_variance > np.finfo(float).eps:
        factor = np.cov(xdata, ydata, ddof=0)[0, 1] / source_variance
        offset = np.mean(ydata) - factor * np.mean(xdata)
    else:
        source_sum = np.sum(xdata)
        if abs(source_sum) <= np.finfo(float).eps:
            raise ValueError("Cannot scale from zero source intensity.")
        factor = np.sum(ydata) / source_sum
        offset = 0.0

    if not np.isfinite(factor) or factor < 0:
        raise ValueError(
            f"Negative helicity scaling factor obtained: {factor}. "
            "Check masks/intensities before using this result."
        )

    # The reconstruction field amplitude scales as sqrt(intensity).
    factor = max(factor, 1e-12)
    scaled_phase = phase * np.sqrt(factor)

    if verbose:
        print(f"Linear fit: {factor:.4f}*x + {offset:.4f}")

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(xdata, ydata, s=5)
        order = np.argsort(xdata)
        ax.plot(xdata[order], factor * xdata[order] + offset, "r-")
        ax.set_xlabel("Source intensity")
        ax.set_ylabel("Target intensity")
        ax.set_title(f"Linear fit: {factor:.4f}*x + {offset:.4f}")

    if use_offset:
        # Approximate amplitude-level offset correction.
        # Usually keep this disabled because adding an intensity offset to a
        # complex Fourier-domain field is not uniquely defined.
        amp = np.abs(scaled_phase)
        phase_angle = np.exp(1j * np.angle(scaled_phase))
        corrected_intensity = np.maximum(amp**2 + offset, 0)
        scaled_phase = np.sqrt(corrected_intensity) * phase_angle

    return scaled_phase


def phase_retrieval_algorithm(
    pos: ArrayLike,
    neg: ArrayLike,
    mask_pixel: ArrayLike,
    supportmask: ArrayLike,
    phase_retrieval_recipe=None,
):
    """
    Run a recipe-driven two-helicity phase retrieval.

    The recipe is a flat sequence of steps. Each step selects an algorithm,
    iteration count, helicity (``"pos"`` or ``"neg"``), beta schedule, optional TV
    schedule through alpha, and optional Richardson-Lucy parameters.

    The starting phase of each step is controlled by recipe["Startimage"].
    Startimage[i] may be:
        None        -> use the default support-based start image
        "pos"       -> use the latest positive-helicity reconstruction
        "neg"       -> use the latest negative-helicity reconstruction
        np.ndarray  -> use the supplied array directly

    When a reconstruction from a different helicity is reused, it is rescaled
    using a masked correlation-based linear normalization.


    Returns
    -------
    retrieved_p, retrieved_n : np.ndarray or None
        Final reconstruction for positive/negative helicity, if requested.
    bsmask_p, bsmask_n : np.ndarray
        Beamstop/floating-pixel masks used for full-coherence steps.
    gamma_p, gamma_n : np.ndarray
        Latest mutual-coherence estimates for each helicity. If no RL step was
        performed for a helicity, this is the initial gamma estimate.
    error : dict
        Step-wise error information stored under ``error["steps"]``.
    """

    # initializing the phase retrieval recipe
    recipe = default_phase_retrieval_recipe()
    user_supplied_output = False
    if phase_retrieval_recipe is not None:
        if not isinstance(phase_retrieval_recipe, dict):
            raise TypeError("phase_retrieval_recipe must be a dictionary.")
        user_supplied_output = "output" in phase_retrieval_recipe
        unknown_keys = set(phase_retrieval_recipe) - set(recipe)
        if unknown_keys:
            raise ValueError(
                "Unknown phase-retrieval recipe key(s): "
                f"{sorted(unknown_keys)}"
            )
        recipe.update(phase_retrieval_recipe)
    if not user_supplied_output:
        recipe["output"] = _default_output_flags(recipe["helicity"])
    _verify_valid_phase_retrieval_recipe(recipe)

    pos_input = np.asarray(pos).copy()
    neg_input = np.asarray(neg).copy()
    mask_pixel = np.asarray(mask_pixel)
    supportmask = np.asarray(supportmask)

    if pos_input.shape != neg_input.shape:
        raise ValueError("pos and neg must have the same shape.")
    if mask_pixel.shape != pos_input.shape:
        raise ValueError("mask_pixel must have the same shape as pos/neg.")
    if supportmask.shape != pos_input.shape:
        raise ValueError("supportmask must have the same shape as pos/neg.")

    # Baseline-subtract each helicity using the requested lower percentile,
    # ignoring zeros because zeros usually represent invalid/masked pixels.
    vmin = recipe["hologram_intensity_cutoff_vmin"]
    if vmin >= 0:
        vals = pos_input[(pos_input != 0) & np.isfinite(pos_input)]
        if vals.size:
            mi = np.nanpercentile(vals, vmin)
            pos_input = pos_input - mi

        vals = neg_input[(neg_input != 0) & np.isfinite(neg_input)]
        if vals.size:
            mi = np.nanpercentile(vals, vmin)
            neg_input = neg_input - mi

        
    # Replace NaNs and negative intensities before taking square roots.
    pos_input = np.where(np.isnan(pos_input), 0, pos_input)
    neg_input = np.where(np.isnan(neg_input), 0, neg_input)

    # Full-coherence masks inherit the external mask and mark nonpositive
    # corrected intensities as unconstrained.
    bsmask_p = mask_pixel.copy()
    bsmask_n = mask_pixel.copy()
    bsmask_p[pos_input <= 0] = 1
    bsmask_n[neg_input <= 0] = 1


    # clip positive intensities to zero to avoid NaNs in the square root.
    pos_input = np.clip(pos_input, 0, None)
    neg_input = np.clip(neg_input, 0, None)

    pos_amp = np.sqrt(pos_input)
    neg_amp = np.sqrt(neg_input)

    # Initial Fourier-domain guess. The supportmask-based default follows the
    # convention of the original implementation.
    first_startimage = recipe["Startimage"][0]
    if first_startimage is None:
        Startimage = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(supportmask)))
    elif isinstance(first_startimage, np.ndarray):
        Startimage = np.asarray(first_startimage).copy()
    else:
        raise ValueError(
            "The first Startimage entry cannot be 'pos' or 'neg' "
            "because no reconstruction exists yet."
        )

    if Startimage.shape != pos_input.shape:
        raise ValueError("Startimage must have the same shape as pos/neg.")

    # Initial mutual-coherence estimate for RL-enabled steps.
    first_startgamma = recipe["Startgamma"][0]

    if first_startgamma is None:
        Startgamma = np.ones(pos_input.shape, dtype=float) * 1e-6 * 2
        Startgamma[pos_input.shape[0] // 2, pos_input.shape[1] // 2] = 0.7
        # new startgamma
        #xx,yy=np.meshgrid(pos_input.shape)-pos_input.shape[0]//2
        #Startgamma = np.exp(-(xx*2+yy*2)/2)
    elif isinstance(first_startgamma, np.ndarray):
        Startgamma = np.asarray(first_startgamma).copy()
    else:
        raise ValueError(
            "The first Startgamma entry cannot be 'pos' or 'neg' "
            "because no coherence estimate exists yet."
        )


    if Startgamma.shape != pos_input.shape:
        raise ValueError("Startgamma must have the same shape as pos/neg.")


    first_helicity = recipe["helicity"][0]

    if first_helicity == "pos":
        first_input = pos_input
    else:
        first_input = neg_input

    valid_pix = (mask_pixel == 0) & (first_input > 0)

    if np.any(valid_pix):
        x = np.sqrt(first_input[valid_pix]).ravel()
        y = np.abs(Startimage[valid_pix]).ravel()

        if x.size >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            res = stats.linregress(x, y)
            if abs(res.slope) > 1e-12:
                Startimage = Startimage / res.slope

    data = {
        "pos": {"amp": pos_amp, "input": pos_input, "bsmask": bsmask_p},
        "neg": {"amp": neg_amp, "input": neg_input, "bsmask": bsmask_n},
    }

    retrieved = {"pos": None, "neg": None}
    retrieved_fc = {"pos": None, "neg": None}
    retrieved_pc = {"pos": None, "neg": None}
    retrieved_gradient = {"pos": None, "neg": None}
    gamma = {"pos": Startgamma.copy(), "neg": Startgamma.copy()}
    pc_diffract = {}


    default_start_image = Startimage.copy()
    default_start_gamma = Startgamma.copy()


    error = {"steps": [], "outputs": [], "outputs_by_helicity": {"pos": [], "neg": []}}
    start_time = time.time()

    for i, mode in enumerate(recipe["algorithm_list"]):
        h = recipe["helicity"][i]
        Nit = recipe["number_iterations"][i]
        RL_it = int(recipe["RL_its"][i])
        RL_freq = recipe["RL_freqs"][i]

        use_RL = RL_it > 0 and RL_freq <= Nit

        # Full-coherence steps use the measured amplitude and the beamstop mask.
        # RL-enabled steps use the beamstop-filled intensity estimate and no
        # Fourier-domain beamstop mask, matching the old partial-coherence logic.
        if use_RL:
            if h not in pc_diffract:
                source = retrieved_fc[h]
                if source is None:
                    pc_input = data[h]["input"]
                else:
                    pc_input = (
                        np.abs(source) ** 2 * data[h]["bsmask"]
                        + data[h]["input"] * (1 - data[h]["bsmask"])
                    )
                pc_diffract[h] = np.sqrt(pc_input)
            diffract = pc_diffract[h]
            bsmask = np.zeros_like(data[h]["bsmask"])
            gamma_in = _resolve_start_field(
                recipe["Startgamma"][i],
                default_start_gamma,
                gamma,
                name="Startgamma",
            )

        else:
            diffract = data[h]["amp"]
            bsmask = data[h]["bsmask"]
            gamma_in = None


        #### PHASE DEFINITION
        start_spec = recipe["Startimage"][i]

        Phase = _resolve_start_field(
            start_spec,
            default_start_image,
            retrieved,
            name="Startimage",
        )

        if isinstance(start_spec, str) and start_spec in {"pos", "neg"}:
            Phase = _scale_phase_between_helicities(
                Phase,
                source_helicity=start_spec,
                target_helicity=h,
                intensity_data={key: data[key]["input"] for key in ["pos", "neg"]},
                bsmasks={key: data[key]["bsmask"] for key in ["pos", "neg"]},
            )
            

        if mode == "gradient_descent":
            if use_RL:
                raise ValueError(
                    "gradient_descent recipe stages do not support "
                    "Richardson-Lucy partial-coherence updates."
                )

            refined = gradient.refine_field_gradient(
                Phase,
                diffract,
                supportmask=supportmask,
                mask_pixel=bsmask,
                n_steps=Nit,
                learning_rate=make_beta_schedule(
                    recipe["beta_mode"][i],
                    Nit,
                    recipe["beta_zero"][i],
                ),
                support_weight=make_alpha_schedule(
                    recipe["alpha_mode"][i],
                    Nit,
                    recipe["alpha_zero"][i],
                ),
                loss_mode="amplitude",
                support_projection=False,
                fourier_projection=False,
            )
            result = refined.fields
            if recipe["Fourier_last"][i]:
                result = _apply_measured_amplitude(result, diffract, bsmask)
            Error_diff = refined.diffraction_loss
            Error_supp = refined.support_loss
            gamma_out = None
            extra_diagnostics = {"loss": refined.loss}
        else:
            result, Error_diff, Error_supp, gamma_out = PhaseRtrv_core(
                diffract=diffract,
                mask=supportmask,
                mode=mode,
                Nit=Nit,
                beta_zero=recipe["beta_zero"][i],
                beta_mode=recipe["beta_mode"][i],
                alpha_zero=recipe["alpha_zero"][i],
                alpha_mode=recipe["alpha_mode"][i],
                Phase=Phase,
                seed=False,
                plot_every=recipe["plot_every"][i],
                bsmask=bsmask,
                real_object=False,
                average_img=recipe["average_img"][i],
                Fourier_last=recipe["Fourier_last"][i],
                gamma=gamma_in,
                RL_freq=RL_freq,
                RL_it=RL_it,
                TV_freq=recipe["TV_freqs"][i],
            )
            extra_diagnostics = {}

        retrieved[h] = result

        if mode == "gradient_descent":
            retrieved_gradient[h] = result
            if retrieved_fc[h] is None:
                retrieved_fc[h] = result
        elif use_RL:
            retrieved_pc[h] = result
        else:
            retrieved_fc[h] = result
            pc_diffract.pop(h, None)


        if gamma_out is not None:
            gamma[h] = gamma_out

        step_info = {
            "step": i,
            "helicity": h,
            "mode": mode,
            "Nit": Nit,
            "RL_it": RL_it,
            "RL_freq": RL_freq,
            "coherence": "partial" if use_RL else "full",
            "output": recipe["output"][i],
            "error": np.asarray(Error_diff),
            "support_error": np.asarray(Error_supp),
            "field_after": result.copy(),
            **extra_diagnostics,
        }
        error["steps"].append(step_info)

        if recipe["output"][i]:
            output_info = {
                "step": i,
                "helicity": h,
                "mode": mode,
                "coherence": step_info["coherence"],
                "field": result.copy(),
            }
            error["outputs"].append(output_info)
            error["outputs_by_helicity"][h].append(output_info)

        print(
            f"Step {i}: helicity={h}, mode={mode}, Nit={Nit}, "
            f"{'partial coherence' if use_RL else 'full coherence'}"
        )

    print("--- %s seconds ---" % np.round((time.time() - start_time), 2))
    print("Phase Retrieval Done!")

    error["latest"] = {
        "full_coherence": retrieved_fc.copy(),
        "partial_coherence": retrieved_pc.copy(),
        "gradient_descent": retrieved_gradient.copy(),
    }

    return (
        retrieved_fc["pos"],
        retrieved_fc["neg"],
        retrieved_pc["pos"],
        retrieved_pc["neg"],
        bsmask_p,
        bsmask_n,
        gamma["pos"],
        gamma["neg"],
        error,
    )

def plot_phase_retrieval_errors(error, phase_retrieval_recipe=None, ax=None):
    """
    Plot the step-wise diffraction errors stored by ``phase_retrieval_algorithm``.

    The current error format is ``error["steps"]``, where each entry contains the
    step index, helicity, algorithm, coherence type, and sampled error array. The
    function returns a concatenated list of all plotted error values together
    with the Matplotlib figure and axis.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if "steps" not in error:
        raise ValueError(
            "Unsupported error dictionary. Expected the new format with "
            "error['steps']."
        )

    counter = 0
    full_error_list = []

    for step_info in error["steps"]:
        error_list = np.asarray(step_info["error"])

        if error_list.size == 0:
            continue

        full_error_list.extend(error_list.tolist())

        x = counter + np.arange(len(error_list))
        counter = x[-1] + 1

        label = (
            f"step {step_info['step']} - {step_info['helicity']} - "
            f"{step_info['mode']} - {step_info['coherence']}"
        )
        ax.plot(x, error_list, label=label)

    if full_error_list:
        ax.set_title(
            f"Smallest Error: {np.min(full_error_list):.2f} dB, "
            f"Final error: {full_error_list[-1]:.2f} dB"
        )

    ax.legend()
    ax.set_xlabel("Tracked errors")
    ax.set_ylabel("log(Error) [dB]")
    ax.grid(True)

    return full_error_list, fig, ax



#############################################################
#       PHASE RETRIEVAL FUNCTIONS HELPER
# ############################################################



# ----------------------------
# Beta schedule (CPU-side)
# ----------------------------
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


def _apply_measured_amplitude(field, amplitude, bsmask):
    """Apply measured Fourier amplitudes outside invalid-pixel regions."""
    constrained = np.asarray(field).copy()
    observed = np.asarray(bsmask) == 0
    constrained[observed] = (
        amplitude[observed] * np.exp(1j * np.angle(constrained[observed]))
    )
    return constrained


# ----------------------------
# Real-space projection steps
# ----------------------------
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

#############################################################
#       MAIN PHASE RETRIEVAL FUNCTIONS
# ############################################################


def PhaseRtrv_core(
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

        # Apply the measured Fourier constraint, including partial coherence.
        if use_RL:
            factor = diffract_cp / xp.sqrt(convolved)
            guess_cp[obs] *= factor[obs]
        else:
            guess_cp = xp.where(
                BSmask_cp,
                guess_cp,
                diffract_cp * xp.exp(1j * xp.angle(guess_cp)),
            )

        # Transform to support space and apply TV plus the selected projection.
        inv = fft2(guess_cp)

        if ((s%TV_freq)==0) and alpha > 0:
            inv=  (inv + alpha* TV(inv, 1) )

        inv = proj_fn(inv, prev, mask_cp, beta, s, Nit)
        prev = inv.copy()

        new_guess = ifft2(inv)

        # Optionally update the coherence kernel from the projected field.
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

        # Record errors and retain the best late-iteration candidates.
        if s <= 2 or (s % plot_every == 0) or (s >= start_best_at):
            err = Error_diffract_cp(err_guess, err_target)
            Error_diffr_list.append(err)

            if s >= start_best_at:
                j = int(xp.argmax(Best_error).item())
                if err < Best_error[j]:
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


def fourier_field_to_display_object(phase_stack):
    """
    Convert returned Fourier fields to centered real-space objects for display.

    The phase-retrieval kernels and log-object projections use the shifted
    support-space frame returned by ``fft2(fftshift(field))``. For plotting with
    ``imshow`` against the usual centered support mask, shift that object once
    more so the sample appears in the middle instead of at the array corners.
    """
    phase_stack = np.asarray(phase_stack)
    single_field = phase_stack.ndim == 2
    if single_field:
        phase_stack = phase_stack[None, ...]
    else:
        phase_stack = _as_energy_stack(phase_stack, name="phase_stack")

    obj = np.empty_like(phase_stack, dtype=np.complex128)

    for j in range(phase_stack.shape[0]):
        obj[j] = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(phase_stack[j]))
        )

    return obj[0] if single_field else obj


def fourier_field_to_display_object_log(
    phase_stack,
    log_floor=1e-12,
    unwrap_energy_phase=True,
):
    """
    Convert returned Fourier fields to centered complex log-objects for display.

    This is the display-frame counterpart of :func:`fourier_field_to_object_log`.
    Use it for plotting or notebook inspection when the support/object should be
    visually centered.
    """
    if not np.isfinite(log_floor) or log_floor <= 0:
        raise ValueError("log_floor must be finite and > 0.")

    obj = fourier_field_to_display_object(phase_stack)
    amp = np.maximum(np.abs(obj), log_floor)
    phase = np.angle(obj)
    if unwrap_energy_phase:
        phase = np.unwrap(phase, axis=0)

    return np.log(amp) + 1j * phase


def display_object_log_to_fourier_field(L_stack):
    """
    Convert centered display-frame log-objects back to returned Fourier fields.
    """
    L_stack = _as_energy_stack(L_stack, name="L_stack")
    phase_stack = np.empty_like(L_stack, dtype=np.complex128)

    for j in range(L_stack.shape[0]):
        phase_stack[j] = np.fft.ifftshift(
            np.fft.ifft2(np.fft.ifftshift(np.exp(L_stack[j])))
        )

    return phase_stack


def object_log_to_fourier_field(L_stack):
    """
    Convert real-space log-objects back to Fourier-domain fields.
    """
    L_stack = _as_energy_stack(L_stack, name="L_stack")
    phase_stack = np.empty_like(L_stack, dtype=np.complex128)

    for j in range(L_stack.shape[0]):
        phase_stack[j] = np.fft.ifftshift(np.fft.ifft2(np.exp(L_stack[j])))

    return phase_stack


def _normalize_projection_supportmask(projection_supportmask, image_shape):
    """Validate an optional log-object-frame mask for limiting projections."""
    if projection_supportmask is None:
        return None
    mask = np.asarray(projection_supportmask) != 0
    if mask.shape != tuple(image_shape):
        raise ValueError("projection_supportmask must have shape (nx, ny).")
    return mask


def _apply_projection_supportmask(projected, original, projection_supportmask):
    """Keep projected values only inside projection_supportmask."""
    if projection_supportmask is None:
        return projected
    return np.where(projection_supportmask[None], projected, original)


# -------------------------------------------------------------------------
#  Generic SVD low-rank projection: L_E(r) = C(r) + low-rank residual
# -------------------------------------------------------------------------

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


def _material_thickness_on_grid(recipe, shape):
    """Resolve material options for a direct projection on its current grid."""
    mask = recipe.get("material_mask")
    thickness = _validate_material_thickness(recipe.get("material_thickness"), shape)
    if mask is None and thickness is None and not recipe.get("fit_material_thickness", False):
        return None
    result = np.ones(shape, dtype=float) if thickness is None else thickness.copy()
    if mask is not None:
        material = np.asarray(mask)
        if material.shape != tuple(shape) or not np.all(np.isfinite(material)):
            raise ValueError(f"material_mask must be finite with shape {tuple(shape)}.")
        result *= material != 0
    if not np.any(result > 0):
        raise ValueError("material_mask/material_thickness contains no material pixels.")
    return result


def project_log_object_low_rank(
    L_stack,
    rank=1,
    static_mode="mean",
    weights=None,
    relaxation=1.0,
    projection_supportmask=None,
    material_thickness=None,
    fit_material_thickness=False,
    return_components=False,
):
    """
    Project a multi-energy log-object stack onto

        L_E(r) = C(r) + Delta_E(r),

    where C(r) is energy independent and Delta_E(r) has rank ``rank`` over the
    energy axis. This is the unconstrained SVD option.
    """
    L_stack = _as_energy_stack(L_stack, name="L_stack")

    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be a non-negative integer.")
    if rank < 0:
        raise ValueError("rank must be a non-negative integer.")
    if not (0 <= relaxation <= 1):
        raise ValueError("relaxation must be between 0 and 1.")

    nE, nx, ny = L_stack.shape
    projection_supportmask = _normalize_projection_supportmask(
        projection_supportmask,
        (nx, ny),
    )
    rank = min(int(rank), nE)
    Lmat = L_stack.reshape(nE, -1).T  # pixels x energies

    if weights is None:
        w = np.ones(nE, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (nE,):
            raise ValueError("weights must have shape (nE,).")
        if np.any(w <= 0):
            raise ValueError("weights must be strictly positive.")

    w = w / np.mean(w)
    sqrt_w = np.sqrt(w)

    if static_mode == "mean":
        C = np.sum(Lmat * w[None, :], axis=1) / np.sum(w)
    elif static_mode == "first":
        C = Lmat[:, 0].copy()
    elif static_mode == "none":
        C = np.zeros(Lmat.shape[0], dtype=Lmat.dtype)
    else:
        raise ValueError("static_mode must be 'mean', 'first', or 'none'.")

    Delta = Lmat - C[:, None]
    thickness = _validate_material_thickness(material_thickness, (nx, ny))
    if fit_material_thickness:
        raise ValueError("Fitting thickness requires projection_model='rank1_spectral'.")
    if thickness is not None:
        # A generic low-rank model has free spatial factors. A known vacuum
        # region can still be constrained to have no energy-dependent term.
        Delta = Delta.copy()
        Delta[thickness.ravel() == 0] = 0

    if rank == 0:
        Delta_rank = np.zeros_like(Delta)
        singular_values = np.array([], dtype=float)
    else:
        Delta_w = Delta * sqrt_w[None, :]
        U, s, Vh = np.linalg.svd(Delta_w, full_matrices=False)
        Delta_rank_w = (U[:, :rank] * s[:rank]) @ Vh[:rank, :]
        Delta_rank = Delta_rank_w / sqrt_w[None, :]
        singular_values = s
    if thickness is not None:
        Delta_rank[thickness.ravel() == 0] = 0

    Lproj_mat = C[:, None] + Delta_rank
    Lproj = Lproj_mat.T.reshape(nE, nx, ny)

    if relaxation < 1:
        Lproj = (1 - relaxation) * L_stack + relaxation * Lproj
    if thickness is not None:
        Lproj[:, thickness == 0] = C.reshape(nx, ny)[thickness == 0]
    if projection_supportmask is not None and thickness is not None:
        projection_supportmask = projection_supportmask | (thickness == 0)
    Lproj = _apply_projection_supportmask(
        Lproj,
        L_stack,
        projection_supportmask,
    )

    if return_components:
        components = {
            "projection_model": "svd",
            "static_log_object": C.reshape(nx, ny),
            "energy_dependent_log_object": Delta_rank.T.reshape(nE, nx, ny),
            "singular_values": singular_values,
            "material_thickness": thickness,
            "projection_supportmask_applied": (
                projection_supportmask is not None
            ),
        }
        return Lproj, components

    return Lproj


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


def _canonicalize_rank1_factors(
    spatial_factor,
    spectral_factor,
    known_beta_spectrum=None,
    absorption_part="imag",
):
    """
    Fix the arbitrary complex phase/sign of a rank-1 factorization.

    The factors M and a can be transformed as
    ``M -> M*exp(-i*theta)``, ``a -> a*exp(i*theta)`` without changing M*a.
    Choose theta so that M is as real-valued as possible. If a known absorption
    spectrum is available, use it to select the remaining sign.
    """
    spatial_factor = np.asarray(spatial_factor, dtype=np.complex128).copy()
    spectral_factor = np.asarray(spectral_factor, dtype=np.complex128).copy()

    phase_moment = np.sum(spatial_factor**2)
    if np.isfinite(phase_moment) and abs(phase_moment) > 0:
        theta = 0.5 * np.angle(phase_moment)
        spatial_factor *= np.exp(-1j * theta)
        spectral_factor *= np.exp(1j * theta)

    flip = False
    if known_beta_spectrum is not None:
        known = np.asarray(known_beta_spectrum, dtype=float)
        retrieved = _extract_absorption_part(
            spectral_factor,
            absorption_part=absorption_part,
        )
        if known.shape == retrieved.shape:
            known_centered = known - np.mean(known)
            retrieved_centered = retrieved - np.mean(retrieved)
            if np.dot(known_centered, retrieved_centered) < 0:
                flip = True
    elif spatial_factor.size:
        anchor = spatial_factor[np.argmax(np.abs(spatial_factor))]
        if np.real(anchor) < 0:
            flip = True

    if flip:
        spatial_factor *= -1
        spectral_factor *= -1

    return spatial_factor, spectral_factor


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
    known_beta_spectrum=None,
    known_delta_spectrum=None,
    absorption_part="imag",
    kk_sign=1.0,
    kk_subtract_baseline=True,
    kk_normalize_input=False,
    known_beta_normalization="none",
    fit_known_beta_scale=True,
    fit_known_beta_offset=True,
    preserve_retrieved_dispersion_for_known_beta=True,
):
    """
    Constrain the complex energy dependence a_E.

    Supported constraints
    ---------------------
    free / none:
        Leave a_E unconstrained.

    kk:
        Take the absorption-like part of the retrieved a_E and compute the
        dispersion-like part from a discrete Kramers-Kronig transform.

    known_beta:
        Replace the absorption-like part of a_E by a supplied beta spectrum.
        The dispersion-like part is kept from the retrieved a_E by default.

    known_beta_kk:
        Replace the absorption-like part by a supplied beta spectrum and
        compute the dispersion-like part from Kramers-Kronig. If
        ``known_delta_spectrum`` is supplied, use that externally prepared
        delta spectrum instead. This is the preferred route when beta was
        extended with broad-range Henke/CXRO data before the transform.

    Notes
    -----
    ``known_beta_spectrum`` is beta(E), the imaginary part of the refractive
    index, not raw absorbance or arbitrary-unit XAS. Conversion and broad-range
    spectral extension live in ``library.kramers_kronig``. KK integration uses
    the exact piecewise-linear specialization of Watts' piecewise
    Laurent-polynomial method.
    """
    a_initial = np.asarray(a_initial, dtype=np.complex128)
    nE = a_initial.size
    mode = str(spectral_constraint).lower()
    if mode in {"none", "free", "unconstrained"}:
        return a_initial.copy(), {
            "spectral_constraint": "free",
            "spectrum_initial": a_initial.copy(),
            "spectrum_constrained": a_initial.copy(),
        }

    retrieved_abs = _extract_absorption_part(a_initial, absorption_part)
    retrieved_disp = _extract_dispersion_part(a_initial, absorption_part)

    if mode in {"kk", "kramers-kronig", "kramers_kronig"}:
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

    elif mode in {"known_beta", "known-beta"}:
        if known_beta_spectrum is None:
            raise ValueError("known_beta_spectrum is required.")
        known_beta = np.asarray(known_beta_spectrum, dtype=float)
        if known_beta.shape != (nE,):
            raise ValueError("known_beta_spectrum must have shape (nE,).")
        if np.any(~np.isfinite(known_beta)):
            raise ValueError("known_beta_spectrum contains non-finite values.")
        known_beta = _normalize_vector(known_beta, mode=known_beta_normalization)
        if fit_known_beta_scale:
            scale, offset = _fit_scale_offset(
                known_beta,
                retrieved_abs,
                fit_offset=fit_known_beta_offset,
            )
            absorption = scale * known_beta + offset
        else:
            absorption = known_beta

        if preserve_retrieved_dispersion_for_known_beta:
            dispersion = retrieved_disp.copy()
        else:
            dispersion = np.zeros_like(absorption)
        a = _complex_spectrum_from_parts(absorption, dispersion, absorption_part)

    elif mode in {"known_beta_kk", "known-beta-kk"}:
        E = _energy_axis_for_kk(energy_values, nE)
        if known_beta_spectrum is None:
            raise ValueError("known_beta_spectrum is required.")
        known_beta = np.asarray(known_beta_spectrum, dtype=float)
        if known_beta.shape != (nE,):
            raise ValueError("known_beta_spectrum must have shape (nE,).")
        if np.any(~np.isfinite(known_beta)):
            raise ValueError("known_beta_spectrum contains non-finite values.")
        known_beta = _normalize_vector(known_beta, mode=known_beta_normalization)
        if fit_known_beta_scale:
            scale, offset = _fit_scale_offset(
                known_beta,
                retrieved_abs,
                fit_offset=fit_known_beta_offset,
            )
            absorption = scale * known_beta + offset
        else:
            scale = 1.0
            absorption = known_beta

        if known_delta_spectrum is None:
            dispersion = kk_sign * kk.beta_to_delta(
                E,
                absorption,
                subtract_baseline=kk_subtract_baseline,
                output_mean=np.mean(retrieved_disp),
            )
        else:
            known_delta = np.asarray(known_delta_spectrum, dtype=float)
            if known_delta.shape != (nE,):
                raise ValueError("known_delta_spectrum must have shape (nE,).")
            if np.any(~np.isfinite(known_delta)):
                raise ValueError("known_delta_spectrum contains non-finite values.")
            dispersion = kk_sign * scale * known_delta
            dispersion = dispersion - np.mean(dispersion) + np.mean(retrieved_disp)
        a = _complex_spectrum_from_parts(absorption, dispersion, absorption_part)

    else:
        raise ValueError(
            "spectral_constraint must be one of 'free', 'kk', "
            "'known_beta', or 'known_beta_kk'."
        )

    return a.astype(np.complex128), {
        "spectral_constraint": mode,
        "spectrum_initial": a_initial.copy(),
        "spectrum_constrained": a.copy(),
        "absorption_part": absorption.copy(),
        "dispersion_part": dispersion.copy(),
        "absorption_part_location": absorption_part,
    }


def project_log_object_rank1_spectral(
    L_stack,
    static_mode="mean",
    weights=None,
    relaxation=1.0,
    spectral_constraint="free",
    energy_values=None,
    known_beta_spectrum=None,
    known_delta_spectrum=None,
    absorption_part="imag",
    kk_sign=1.0,
    kk_subtract_baseline=True,
    kk_normalize_input=False,
    known_beta_normalization="none",
    fit_known_beta_scale=True,
    fit_known_beta_offset=True,
    projection_supportmask=None,
    material_thickness=None,
    fit_material_thickness=False,
    return_components=False,
):
    """
    Project a log-object stack onto the explicit model

        L_E(r) = C(r) + M(r) a_E.

    This is a rank-1 spectral factorization with an explicit complex spectral
    vector a_E. The vector can be unconstrained, KK-constrained, constrained by a
    known beta spectrum, or constrained by known beta + KK.
    """
    L_stack = _as_energy_stack(L_stack, name="L_stack")
    if not (0 <= relaxation <= 1):
        raise ValueError("relaxation must be between 0 and 1.")

    if (fit_material_thickness and
            str(spectral_constraint).lower() in {"known_beta", "known-beta", "known_beta_kk", "known-beta-kk"}
            and not fit_known_beta_scale):
        raise ValueError("Fitting relative thickness requires a free spectral scale.")

    nE, nx, ny = L_stack.shape
    thickness = _validate_material_thickness(material_thickness, (nx, ny))
    if fit_material_thickness and thickness is None:
        thickness = np.ones((nx, ny), dtype=float)
    projection_supportmask = _normalize_projection_supportmask(
        projection_supportmask,
        (nx, ny),
    )
    Lmat = L_stack.reshape(nE, -1).T  # pixels x energies

    if weights is None:
        w = np.ones(nE, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (nE,):
            raise ValueError("weights must have shape (nE,).")
        if np.any(w <= 0):
            raise ValueError("weights must be strictly positive.")
    w = w / np.mean(w)
    sqrt_w = np.sqrt(w)

    if static_mode == "mean":
        C = np.sum(Lmat * w[None, :], axis=1) / np.sum(w)
    elif static_mode == "first":
        C = Lmat[:, 0].copy()
    elif static_mode == "none":
        C = np.zeros(Lmat.shape[0], dtype=Lmat.dtype)
    else:
        raise ValueError("static_mode must be 'mean', 'first', or 'none'.")

    Delta = Lmat - C[:, None]
    if thickness is not None:
        # The supplied map is a relative thickness, shared by all energies.
        # Vacuum pixels have zero material response by construction.
        t = thickness.ravel()
        denominator = np.sum(t * t)
        if denominator <= 0:
            raise ValueError("material_thickness must contain material pixels.")
        a_initial = np.sum(t[:, None] * Delta, axis=0) / denominator
        if fit_material_thickness:
            allowed = t > 0
            for _ in range(12):
                spectral_power = np.sum(w * np.abs(a_initial) ** 2)
                if spectral_power <= 1e-30:
                    raise ValueError("Cannot fit thickness without energy-dependent contrast.")
                updated = np.maximum(
                    0, np.real(np.sum(Delta * (w * np.conj(a_initial))[None, :], axis=1))
                    / spectral_power,
                )
                updated[~allowed] = 0
                scale = np.mean(updated[allowed])
                if scale <= 1e-30:
                    raise ValueError("Fitted material thickness vanished everywhere.")
                t = updated / scale
                a_initial = np.sum(t[:, None] * Delta, axis=0) / np.sum(t * t)
            thickness = t.reshape(nx, ny)
        a_constrained, spectral_info = constrain_complex_spectrum(
            a_initial,
            spectral_constraint=spectral_constraint,
            energy_values=energy_values,
            known_beta_spectrum=known_beta_spectrum,
            known_delta_spectrum=known_delta_spectrum,
            absorption_part=absorption_part,
            kk_sign=kk_sign,
            kk_subtract_baseline=kk_subtract_baseline,
            kk_normalize_input=kk_normalize_input,
            known_beta_normalization=known_beta_normalization,
            fit_known_beta_scale=fit_known_beta_scale,
            fit_known_beta_offset=fit_known_beta_offset,
        )
        if fit_material_thickness:
            spectral_power = np.sum(w * np.abs(a_constrained) ** 2)
            if spectral_power > 1e-30:
                t = np.maximum(
                    0, np.real(np.sum(Delta * (w * np.conj(a_constrained))[None, :], axis=1))
                    / spectral_power,
                )
                t[~allowed] = 0
                scale = np.mean(t[allowed])
                if scale > 1e-30:
                    t /= scale
                    a_constrained *= scale
                    thickness = t.reshape(nx, ny)
        # The energy-independent field is free at each pixel. Refit it after
        # any spectral prior changes the energy mean of the response.
        C = np.sum((Lmat - t[:, None] * a_constrained[None, :])
                   * w[None, :], axis=1) / np.sum(w)
        Delta_rank = t[:, None] * a_constrained[None, :]
        Lproj = (C[:, None] + Delta_rank).T.reshape(nE, nx, ny)
        if relaxation < 1:
            Lproj = (1 - relaxation) * L_stack + relaxation * Lproj
        Lproj[:, thickness == 0] = C.reshape(nx, ny)[thickness == 0]
        if projection_supportmask is not None:
            projection_supportmask = projection_supportmask | (thickness == 0)
        Lproj = _apply_projection_supportmask(Lproj, L_stack, projection_supportmask)
        if return_components:
            components = {
                "projection_model": "rank1_spectral",
                "static_log_object": C.reshape(nx, ny),
                "material_thickness": thickness.copy(),
                "material_thickness_fitted": bool(fit_material_thickness),
                "spectral_spatial_map": thickness.astype(np.complex128),
                "spectral_coefficients_initial": a_initial,
                "spectral_coefficients": a_constrained,
                "energy_dependent_log_object": Delta_rank.T.reshape(nE, nx, ny),
                "projection_supportmask_applied": projection_supportmask is not None,
            }
            components.update(spectral_info)
            return Lproj, components
        return Lproj

    # Initial weighted rank-1 factorization.
    Delta_w = Delta * sqrt_w[None, :]
    U, s, Vh = np.linalg.svd(Delta_w, full_matrices=False)
    if s.size == 0:
        M = np.zeros(Lmat.shape[0], dtype=np.complex128)
        a_initial = np.zeros(nE, dtype=np.complex128)
    else:
        M_w = U[:, 0] * s[0]
        a_w = Vh[0, :]
        a_initial = a_w / sqrt_w
        M = M_w.copy()

    M, a_initial = _canonicalize_rank1_factors(
        M,
        a_initial,
        known_beta_spectrum=known_beta_spectrum,
        absorption_part=absorption_part,
    )

    a_constrained, spectral_info = constrain_complex_spectrum(
        a_initial,
        spectral_constraint=spectral_constraint,
        energy_values=energy_values,
        known_beta_spectrum=known_beta_spectrum,
        known_delta_spectrum=known_delta_spectrum,
        absorption_part=absorption_part,
        kk_sign=kk_sign,
        kk_subtract_baseline=kk_subtract_baseline,
        kk_normalize_input=kk_normalize_input,
        known_beta_normalization=known_beta_normalization,
        fit_known_beta_scale=fit_known_beta_scale,
        fit_known_beta_offset=fit_known_beta_offset,
    )

    # Refit M(r) for the constrained spectral vector a_E by weighted complex LS:
    # minimize_E sum w_E |Delta_pE - M_p a_E|^2.
    denom = np.sum(w * np.abs(a_constrained) ** 2)
    if denom <= 0 or not np.isfinite(denom):
        M = np.zeros(Lmat.shape[0], dtype=np.complex128)
    else:
        M = np.sum(
            Delta * (w * np.conj(a_constrained))[None, :],
            axis=1,
        ) / denom

    Delta_rank = M[:, None] * a_constrained[None, :]
    Lproj_mat = C[:, None] + Delta_rank
    Lproj = Lproj_mat.T.reshape(nE, nx, ny)

    if relaxation < 1:
        Lproj = (1 - relaxation) * L_stack + relaxation * Lproj
    Lproj = _apply_projection_supportmask(
        Lproj,
        L_stack,
        projection_supportmask,
    )

    if return_components:
        components = {
            "projection_model": "rank1_spectral",
            "static_log_object": C.reshape(nx, ny),
            "spectral_spatial_map": M.reshape(nx, ny),
            "spectral_coefficients_initial": a_initial,
            "spectral_coefficients": a_constrained,
            "energy_dependent_log_object": Delta_rank.T.reshape(nE, nx, ny),
            "singular_values": s,
            "projection_supportmask_applied": (
                projection_supportmask is not None
            ),
        }
        components.update(spectral_info)
        return Lproj, components

    return Lproj


def project_fourier_fields_multi_energy(
    phase_stack,
    projection_model="svd",
    rank=1,
    static_mode="mean",
    weights=None,
    relaxation=1.0,
    log_floor=1e-12,
    spectral_constraint="free",
    energy_values=None,
    known_beta_spectrum=None,
    known_delta_spectrum=None,
    absorption_part="imag",
    kk_sign=1.0,
    kk_subtract_baseline=True,
    kk_normalize_input=False,
    known_beta_normalization="none",
    fit_known_beta_scale=True,
    fit_known_beta_offset=True,
    projection_supportmask=None,
    material_thickness=None,
    fit_material_thickness=False,
    return_components=False,
):
    """
    Apply a selected multi-energy projection to Fourier-domain fields.

    This is the representation boundary used by the reconstruction drivers:
    it converts Fourier fields to complex log-objects once, applies the chosen
    log-object projection, and converts the result back once.

    projection_model options
    ------------------------
    'none':
        No multi-energy constraint.
    'svd' or 'low_rank':
        Generic SVD projection, L_E = C + rank-K residual.
    'rank1_spectral':
        Explicit physical model, L_E = C + M*a_E, with optional spectral
        constraints on the complex energy dependence a_E.
    """
    model = str(projection_model).lower()
    if model in {"none", "no", "off", "unconstrained"}:
        if return_components:
            return phase_stack, {"projection_model": "none"}
        return phase_stack

    if model in {"svd", "low_rank", "low-rank"}:
        projection_kind = "svd"
    elif model in {
        "rank1_spectral",
        "spectral",
        "explicit",
        "cma",
        "c+m*a",
    }:
        projection_kind = "rank1_spectral"
    else:
        raise ValueError(
            "projection_model must be 'none', 'svd'/'low_rank', "
            "or 'rank1_spectral'."
        )

    log_objects = fourier_field_to_object_log(
        phase_stack,
        log_floor=log_floor,
    )

    if projection_kind == "svd":
        projected = project_log_object_low_rank(
            log_objects,
            rank=rank,
            static_mode=static_mode,
            weights=weights,
            relaxation=relaxation,
            projection_supportmask=projection_supportmask,
            material_thickness=material_thickness,
            fit_material_thickness=fit_material_thickness,
            return_components=return_components,
        )
    else:
        projected = project_log_object_rank1_spectral(
            log_objects,
            static_mode=static_mode,
            weights=weights,
            relaxation=relaxation,
            spectral_constraint=spectral_constraint,
            energy_values=energy_values,
            known_beta_spectrum=known_beta_spectrum,
            known_delta_spectrum=known_delta_spectrum,
            absorption_part=absorption_part,
            kk_sign=kk_sign,
            kk_subtract_baseline=kk_subtract_baseline,
            kk_normalize_input=kk_normalize_input,
            known_beta_normalization=known_beta_normalization,
            fit_known_beta_scale=fit_known_beta_scale,
            fit_known_beta_offset=fit_known_beta_offset,
            projection_supportmask=projection_supportmask,
            material_thickness=material_thickness,
            fit_material_thickness=fit_material_thickness,
            return_components=return_components,
        )

    if return_components:
        projected_logs, components = projected
        return object_log_to_fourier_field(projected_logs), components

    return object_log_to_fourier_field(projected)


# -------------------------------------------------------------------------
#  Multi-energy reconstruction driver
# -------------------------------------------------------------------------

def default_multi_energy_phase_retrieval_recipe():
    """
    Return default settings for joint multi-energy phase retrieval.

    The reconstruction alternates short single-energy updates with an optional
    projection that couples the object estimates across energy.
    """
    return {
        # Single-energy update schedule repeated once per outer iteration.
        # A scalar inner_mode/inner_Nit pair is accepted as a one-stage shorthand.
        "inner_mode": ["HAPRE", "ER"],
        "inner_Nit": [700,50],
        "outer_iterations": 100,

        # Optional independent warmup schedule, run once at every energy.
        # Set warmup_Nit=0 to disable it.
        "warmup_mode": ["HAPRE", "ER"],
        "warmup_Nit": [700,50],

        "shuffle_energies": True,
        "random_seed": None,
        # Scalars are broadcast to every stage; lists customize each stage.
        "beta_zero": 0.5,
        "beta_mode": "arctan",
        "alpha_zero": 0.0,
        "alpha_mode": "const",
        "TV_freq": 1e9,
        "RL_it": 0,
        "RL_freq": 1e9,

        # Optional warmup overrides. None inherits the corresponding joint
        # setting. If the schedule lengths differ, the first joint-stage value
        # is broadcast across warmup.
        "warmup_beta_zero": None,
        "warmup_beta_mode": None,
        "warmup_alpha_zero": None,
        "warmup_alpha_mode": None,
        "warmup_TV_freq": None,
        "warmup_RL_it": None,
        "warmup_RL_freq": None,
        "plot_every": 1e9,
        "average_img": 1,
        "Fourier_last": True,
        "final_fourier_constraint": True,
        "hologram_intensity_cutoff_vmin": -1,
        "binning": 1,
        "crop": 0,
        "roi": None,

        # Multi-energy projection settings.
        "projection_model": "svd",  # 'none', 'svd', or 'rank1_spectral'
        "rank": 1,
        # Number of completed energy updates between projections. None means
        # one full energy sweep, preserving the historical default cadence.
        "projection_every": None,
        "projection_relaxation": 1.0,
        "final_projection_relaxation": 1.0,
        # None starts projection at the first projection_every boundary.
        "projection_start": None,
        # If True, apply cross-object projections only inside supportmask.
        "projection_constraints_inside_support_only": False,
        # Backward-compatible alias for older recipes.
        "physical_constraints_inside_support_only": False,
        "projection_static_mode": "mean",
        # Optional relative object thickness. A binary material_mask is enough
        # to mark known vacuum holes; material_thickness may give relative
        # nonnegative thickness where the sample is present.
        "material_mask": None,
        "material_thickness": None,
        "fit_material_thickness": False,
        "energy_weights": None,
        "log_floor": 1e-12,

        # Explicit C + M*a_E spectral constraint settings.
        "spectral_constraint": "free",  # 'free', 'kk', 'known_beta', 'known_beta_kk'
        "energy_values": None,
        "known_beta_spectrum": None,
        "known_delta_spectrum": None,
        "absorption_part": "real",  # absorption-like part of log-object spectrum: 'real' or 'imag'
        "kk_sign": 1.0,
        "kk_subtract_baseline": True,
        "kk_normalize_input": False,
        "known_beta_normalization": "none",  # 'none', 'maxabs', 'l2', 'std'
        "fit_known_beta_scale": True,
        "fit_known_beta_offset": True,
    }


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
    """
    Build fully specified stage dictionaries from a recipe.

    Context:
    Schedule compiler used before executing observations. Expand scalar controls to one
    value per stage so the numerical runner does not need to interpret notebook shorthand.
    """
    prefix = "" if name == "inner" else "warmup_"
    mode_key = "inner_mode" if name == "inner" else "warmup_mode"
    Nit_key = "inner_Nit" if name == "inner" else "warmup_Nit"
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
        recipe_key = f"{prefix}{key}"
        value = recipe[recipe_key]
        if name == "warmup" and value is None:
            value = recipe[key]
            if (
                isinstance(value, (list, tuple))
                and len(value) != len(base_schedule)
            ):
                if not value:
                    raise ValueError(f"{key} cannot be empty.")
                value = value[0]
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
                    f"{prefix}{key}[{stage_index}] must be a finite number."
                )

        for key in ("beta_mode", "alpha_mode"):
            value = stage[key]
            if isinstance(value, str):
                if value not in BETA_SCHEDULES:
                    raise ValueError(
                        f"Invalid {prefix}{key}[{stage_index}]={value!r}. "
                        f"Allowed modes are {sorted(BETA_SCHEDULES)}."
                    )
            elif isinstance(value, np.ndarray):
                if value.ndim != 1 or value.shape[0] != Nit:
                    raise ValueError(
                        f"{prefix}{key}[{stage_index}] array must have "
                        f"length Nit={Nit}."
                    )
                if np.any(~np.isfinite(value)):
                    raise ValueError(
                        f"{prefix}{key}[{stage_index}] contains non-finite values."
                    )
            else:
                raise ValueError(
                    f"{prefix}{key}[{stage_index}] must be a schedule name "
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
                f"{prefix}TV_freq[{stage_index}] must be a positive number."
            )

        RL_it = stage["RL_it"]
        if (
            isinstance(RL_it, bool)
            or not isinstance(RL_it, (int, np.integer))
            or RL_it < 0
        ):
            raise ValueError(
                f"{prefix}RL_it[{stage_index}] must be a non-negative integer."
            )

        RL_freq = stage["RL_freq"]
        if (
            isinstance(RL_freq, bool)
            or not isinstance(RL_freq, (int, float, np.number))
            or not np.isfinite(RL_freq)
            or RL_freq <= 0
        ):
            raise ValueError(
                f"{prefix}RL_freq[{stage_index}] must be a positive number."
            )

        schedule.append(stage)
    return schedule


def _build_role_warmup_schedule(recipe, role):
    """
    Build an optional reference/other warmup schedule.

    A role schedule inherits controls from the shared ``warmup_*`` schedule.
    Leaving its mode and iteration settings unset preserves the legacy shared
    schedule behaviour.

    Context:
    Resolve the reference or other-observation override. This selects algorithms/iterations;
    whether a field is copied from the reference is controlled separately by
    warmup_start_from_first.
    """
    mode = recipe[f"warmup_{role}_mode"]
    Nit = recipe[f"warmup_{role}_Nit"]
    if mode is None and Nit is None:
        return None
    if mode is None or Nit is None:
        raise ValueError(
            f"warmup_{role}_mode and warmup_{role}_Nit must be set together."
        )

    role_recipe = recipe.copy()
    role_recipe["warmup_mode"] = mode
    role_recipe["warmup_Nit"] = Nit
    for key in (
        "beta_zero", "beta_mode", "alpha_zero", "alpha_mode",
        "TV_freq", "RL_it", "RL_freq",
    ):
        role_value = recipe[f"warmup_{role}_{key}"]
        if role_value is not None:
            role_recipe[f"warmup_{key}"] = role_value
    return _build_update_schedule(
        role_recipe, name="warmup", allow_disabled=True,
    )


def format_universal_workflow(
    universal_recipe=None,
    state_labels=None,
    start_fields_provided=False,
):
    """
    Return an ASCII tree describing a universal retrieval recipe.

    The tree reports which warmup schedule is applied to the reference and
    nonreference holograms, where their starting fields come from, and how the
    later joint phase-retrieval and physical-projection loop is arranged.
    This function only inspects the recipe; it does not allocate image arrays.

    Context:
    Notebook-facing plan, not a run log: render the recipe without reconstructing any
    fields. Use this before launching a long calculation to inspect initialization,
    schedules and coherence policy.
    """
    recipe = default_universal_phase_retrieval_recipe()
    if universal_recipe is not None:
        if not isinstance(universal_recipe, dict):
            raise TypeError("universal_recipe must be a dictionary.")
        unknown = set(universal_recipe) - set(recipe)
        if unknown:
            raise ValueError(f"Unknown universal recipe key(s): {sorted(unknown)}")
        recipe.update(universal_recipe)

    labels = list(state_labels) if state_labels is not None else None
    reference_index = int(recipe["warmup_reference_observation"])
    if labels is not None and not 0 <= reference_index < len(labels):
        raise ValueError("warmup_reference_observation exceeds state_labels.")
    reference_label = (
        repr(labels[reference_index]) if labels is not None else f"observation {reference_index}"
    )
    other_labels = (
        [label for index, label in enumerate(labels) if index != reference_index]
        if labels is not None else None
    )

    shared_schedule = _build_update_schedule(
        recipe, name="warmup", allow_disabled=True,
    )
    reference_schedule = _build_role_warmup_schedule(recipe, "reference")
    other_schedule = _build_role_warmup_schedule(recipe, "other")
    reference_schedule = shared_schedule if reference_schedule is None else reference_schedule
    if other_schedule is None:
        indices = recipe["warmup_seeded_stage_indices"]
        other_schedule = (
            [shared_schedule[index] for index in indices]
            if indices is not None else shared_schedule
        )
    inner_schedule = _build_update_schedule(recipe, name="inner")

    def stage_text(stage):
        text = f"{stage['mode']} × {stage['Nit']}"
        if stage["RL_it"] > 0 and stage["RL_freq"] <= stage["Nit"]:
            text += f" + partial coherence (RL {stage['RL_it']} every {stage['RL_freq']})"
        if stage["alpha_zero"] != 0:
            text += f", support weight {stage['alpha_zero']} ({stage['alpha_mode']})"
        if stage["TV_freq"] <= stage["Nit"]:
            text += f", TV every {stage['TV_freq']:g}"
        return text

    def branch(lines, prefix, title, start, schedule):
        lines.append(f"{prefix}├─ {title}")
        lines.append(f"{prefix}│  ├─ start: {start}")
        if schedule:
            for index, stage in enumerate(schedule):
                connector = "└" if index == len(schedule) - 1 else "├"
                lines.append(f"{prefix}│  {connector}─ {stage_text(stage)}")
        else:
            lines.append(f"{prefix}│  └─ warmup disabled")

    modes = (recipe["modes"] if recipe["modes"] is not None
             else list(range(1, recipe["Nmodes"] + 1)))
    if start_fields_provided:
        initial = "supplied start_fields"
    elif len(modes) > 1 and recipe["mode_initialization"] == "random_phase":
        initial = "measured amplitude with seeded random phase in each mode"
    else:
        initial = "FFT(support), scaled to measured intensity"
    if recipe["startimage_radial_normalization"] and not start_fields_provided:
        initial += "; radial intensity normalization"
    if labels is None:
        others_title = "Other holograms"
    elif len(other_labels) <= 6:
        others_title = "Other holograms: " + ", ".join(map(repr, other_labels))
    else:
        others_title = (
            f"Other holograms: {len(other_labels)} observations "
            f"({other_labels[0]!r} … {other_labels[-1]!r})"
        )
    other_start = (
        f"scaled final field from reference {reference_label}"
        if recipe["warmup_start_from_first"] else initial
    )

    if recipe["startimage_radial_normalization"] and recipe["warmup_start_from_first"]:
        other_start += "; radial intensity normalization for this observation"

    lines = ["Universal phase-retrieval workflow"]
    if recipe["coherent_refresh_rounds"]:
        lines.append(f"├─ Coherent refresh before rounds {recipe['coherent_refresh_rounds']}: "
                     f"{recipe['coherent_refresh_mode']} × {recipe['coherent_refresh_iterations']}; "
                     "refresh masked fills, retain gamma, resume partial coherence")
    lines.append(
        f"├─ Geometry: crop {recipe['crop']} pixels, then bin ×{recipe['binning']}"
    )
    branch(lines, "", f"Reference hologram: {reference_label}", initial, reference_schedule)
    branch(lines, "", others_title, other_start, other_schedule)
    lines.append(f"├─ Joint reconstruction × {recipe['outer_iterations']} outer loops")
    lines.append(
        "│  ├─ observation order: "
        + ("shuffled each loop" if recipe["shuffle_observations"] else "input order")
    )
    for index, stage in enumerate(inner_schedule):
        lines.append(f"│  ├─ each hologram: {stage_text(stage)}")
    projection_every = recipe["projection_every"]
    cadence = "one observation sweep" if projection_every is None else f"{projection_every} observation updates"
    if recipe["projection_relaxation"] == 0:
        lines.append("│  └─ Repeated physical projection: skipped (relaxation=0)")
    else:
        lines.append(
            f"│  └─ {recipe['projection_model']} projection every {cadence}; "
            f"relaxation={recipe['projection_relaxation']}"
        )
    if float(recipe["projection_focus_prop_um"]) != 0:
        lines.append(
            "│     focus before projection: "
            f"{recipe['projection_focus_prop_um']} µm, "
            f"phase {recipe['projection_focus_phase_rad']} rad; reverse afterward"
        )
    lines.append(f"├─ Modes: {modes}; physical constraints act on support-factor-1 mode")
    if len(modes) > 1 and recipe["constrain_nonphysical_modes_common"]:
        lines.append(
            "├─ Other modes: state-independent common object per energy/beam; "
            f"relaxation={recipe['nonphysical_modes_common_relaxation']}"
        )
    if recipe["preserve_warmup_masked_intensity"]:
        lines.append("├─ Warmup order: ALL coherent states, then partial-coherence states")
        lines.append("│  Masked intensities remain fixed to each state's coherent warmup")
    if recipe["freeze_coherence_after_reference"]:
        lines.append("├─ Gamma: fit reference, then reuse unchanged for every state")
    elif recipe["coherence_kernel_scope"] == "shared":
        descriptions = {
            "sequential": "each state passes its updated gamma to the next state (serial)",
            "average": "independent fits from one snapshot; normalized weighted average after each round",
            "pooled": "fixed gamma per round; pooled fit against all states afterward",
        }
        lines.append("├─ Gamma: " + descriptions[recipe["shared_coherence_update"]])
    if recipe["warmup_calibrate_shared_gamma"]:
        lines.append("├─ Partial warmup: calibrate reference gamma first, then seed all loop fits with it")
    lines.append(f"├─ CPU workers: {recipe['observation_workers']}; FFT threads each: {recipe['fft_workers']}")
    if recipe["freeze_saturated_fields"]:
        lines.append("├─ Saturated observations remain fixed after warmup")
    if recipe["final_projection_relaxation"] == 0:
        lines.append("├─ Final physical projection: skipped (relaxation=0)")
    else:
        lines.append(
            f"├─ Final physical projection: relaxation={recipe['final_projection_relaxation']}"
        )
    lines.append(
        "├─ Final measured-amplitude projection: "
        + ("enabled" if recipe["final_fourier_constraint"] else "disabled")
    )
    lines.append("└─ Results")
    lines.append("   ├─ warmup_fields: fields before joint physical iterations")
    lines.append("   ├─ fields: final retrieved fields")
    lines.append("   ├─ components: fitted physical/coherence/mode quantities")
    lines.append("   ├─ bsmasks: detector masks after crop/bin geometry")
    lines.append("   └─ errors: stage history, settings, and diagnostics")
    return "\n".join(lines)


def print_universal_workflow(
    universal_recipe=None,
    state_labels=None,
    start_fields_provided=False,
):
    """Print :func:`format_universal_workflow` and return its text."""
    text = format_universal_workflow(
        universal_recipe,
        state_labels=state_labels,
        start_fields_provided=start_fields_provided,
    )
    print(text)
    return text


def plot_universal_workflow(universal_recipe=None, state_labels=None,
                            start_fields_provided=False):
    """Display the resolved workflow as a saveable Matplotlib figure."""
    text = format_universal_workflow(universal_recipe, state_labels=state_labels,
                                    start_fields_provided=start_fields_provided)
    lines = text.splitlines()
    fig, ax = plt.subplots(figsize=(max(12, min(24, max(map(len, lines)) * .095)),
                                    max(4, len(lines) * .24)))
    ax.axis("off")
    ax.text(0, 1, text, va="top", ha="left", family="monospace", fontsize=9,
            transform=ax.transAxes)
    fig.tight_layout()
    return fig


def _run_energy_update_schedule(
    field,
    amplitude,
    supportmask,
    bsmask,
    schedule,
    recipe,
    phase_retrieval_kernel=None,
):
    """Run all phase-retrieval stages sequentially for one energy."""
    if phase_retrieval_kernel is None:
        phase_retrieval_kernel = PhaseRtrv_core
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
            refined = gradient.refine_field_gradient(
                field,
                amplitude,
                supportmask=supportmask,
                mask_pixel=bsmask,
                n_steps=Nit,
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
                loss_mode="amplitude",
                support_projection=False,
                fourier_projection=False,
            )
            field = refined.fields
            if recipe["Fourier_last"]:
                observed = bsmask == 0
                field = np.asarray(field).copy()
                field[observed] = (
                    amplitude[observed] * np.exp(1j * np.angle(field[observed]))
                )
            err_d = refined.diffraction_loss
            err_s = refined.support_loss
        else:
            field, err_d, err_s, _ = phase_retrieval_kernel(
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
                "error": np.asarray(err_d),
                "support_error": np.asarray(err_s),
            }
        )
    return field, stage_results


def _projection_is_due(completed_updates, projection_start, projection_every):
    """Return whether a joint projection is scheduled after this update.

    Both controls use completed observation or energy updates, not outer-loop
    indices. A positive start value is itself an eligible boundary.
    """
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


def _verify_multi_energy_recipe(recipe, nE):
    """Validate schedules, projection settings, weights, and spectral inputs."""
    allowed_models = {
        "none",
        "no",
        "off",
        "unconstrained",
        "svd",
        "low_rank",
        "low-rank",
        "rank1_spectral",
        "spectral",
        "explicit",
        "cma",
        "c+m*a",
    }
    allowed_spectral_constraints = {
        "none",
        "free",
        "unconstrained",
        "kk",
        "kramers-kronig",
        "kramers_kronig",
        "known_beta",
        "known-beta",
        "known_beta_kk",
        "known-beta-kk",
    }

    model = str(recipe["projection_model"]).lower()
    if model not in allowed_models:
        raise ValueError(
            "projection_model must be 'none', 'svd'/'low_rank', "
            "or 'rank1_spectral'."
        )
    if not isinstance(recipe["fit_material_thickness"], bool):
        raise ValueError("fit_material_thickness must be bool.")
    if recipe["fit_material_thickness"] and model not in {
        "rank1_spectral", "spectral", "explicit", "cma", "c+m*a",
    }:
        raise ValueError("Fitting thickness requires projection_model='rank1_spectral'.")

    _build_update_schedule(
        recipe,
        name="inner",
    )
    _build_update_schedule(
        recipe,
        name="warmup",
        allow_disabled=True,
    )

    integer_keys = ["outer_iterations", "average_img"]
    for key in integer_keys:
        value = recipe[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{key} must be an integer.")

    projection_every = recipe["projection_every"]
    if projection_every is not None and (
        isinstance(projection_every, bool)
        or not isinstance(projection_every, (int, np.integer))
        or projection_every <= 0
    ):
        raise ValueError("projection_every must be None or a positive integer.")
    if recipe["outer_iterations"] <= 0:
        raise ValueError("outer_iterations must be > 0.")
    projection_start = recipe["projection_start"]
    if projection_start is not None and (
        isinstance(projection_start, bool)
        or not isinstance(projection_start, (int, np.integer))
        or projection_start < 0
    ):
        raise ValueError("projection_start must be None or a non-negative integer.")
    if recipe["average_img"] <= 0:
        raise ValueError("average_img must be > 0.")
    if not isinstance(recipe["Fourier_last"], bool):
        raise ValueError("Fourier_last must be bool.")
    if not isinstance(recipe["final_fourier_constraint"], bool):
        raise ValueError("final_fourier_constraint must be bool.")
    if recipe["plot_every"] <= 0:
        raise ValueError("plot_every must be > 0.")
    if not (0 <= recipe["projection_relaxation"] <= 1):
        raise ValueError("projection_relaxation must be between 0 and 1.")
    if not (0 <= recipe["final_projection_relaxation"] <= 1):
        raise ValueError("final_projection_relaxation must be between 0 and 1.")
    if not isinstance(recipe["projection_constraints_inside_support_only"], bool):
        raise ValueError("projection_constraints_inside_support_only must be bool.")
    if not isinstance(recipe["physical_constraints_inside_support_only"], bool):
        raise ValueError("physical_constraints_inside_support_only must be bool.")

    if isinstance(recipe["rank"], bool) or not isinstance(
        recipe["rank"], (int, np.integer)
    ):
        raise ValueError("rank must be a non-negative integer.")
    if recipe["rank"] < 0:
        raise ValueError("rank must be a non-negative integer.")

    weights = recipe["energy_weights"]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (nE,) or np.any(~np.isfinite(weights)):
            raise ValueError("energy_weights must be finite with shape (nE,).")
        if np.any(weights <= 0):
            raise ValueError("energy_weights must be strictly positive.")

    if model in {"rank1_spectral", "spectral", "explicit", "cma", "c+m*a"}:
        spectral_mode = str(recipe["spectral_constraint"]).lower()
        if spectral_mode not in allowed_spectral_constraints:
            raise ValueError(
                "spectral_constraint must be one of 'free', 'kk', "
                "'known_beta', or 'known_beta_kk'."
            )
        if spectral_mode in {
            "kk",
            "kramers-kronig",
            "kramers_kronig",
            "known_beta_kk",
            "known-beta-kk",
        }:
            _energy_axis_for_kk(recipe["energy_values"], nE)
        if spectral_mode in {
            "known_beta",
            "known-beta",
            "known_beta_kk",
            "known-beta-kk",
        }:
            known = np.asarray(recipe["known_beta_spectrum"], dtype=float)
            if known.shape != (nE,) or np.any(~np.isfinite(known)):
                raise ValueError(
                    "known_beta_spectrum must be finite with shape (nE,)."
                )
            known_delta = recipe["known_delta_spectrum"]
            if known_delta is not None:
                known_delta = np.asarray(known_delta, dtype=float)
                if known_delta.shape != (nE,) or np.any(~np.isfinite(known_delta)):
                    raise ValueError(
                        "known_delta_spectrum must be finite with shape (nE,)."
                    )


def multi_energy_phase_retrieval_algorithm(
    holograms,
    mask_pixel,
    supportmask,
    multi_energy_recipe=None,
    start_fields=None,
    phase_retrieval_kernel=None,
):
    """
    Jointly reconstruct several same-sample holograms measured at different
    photon energies.

    Each outer iteration runs the complete ``inner_mode``/``inner_Nit``
    schedule at every energy. The selected cross-energy projection is applied
    after each configured number of completed energy updates. For example::

        inner_mode = ["HAPRE", "ER"]
        inner_Nit = [700, 50]
        beta_zero = [0.5, 0.9]
        beta_mode = ["arctan", "const"]
        alpha_zero = [0.0, 0.0]
        alpha_mode = ["const", "const"]
        TV_freq = [1e9, 1e9]

    performs 700 HAPRE updates followed by 50 ER updates at each energy during
    every outer iteration, using the corresponding beta, alpha, and TV values
    for each stage. Any of those controls can instead be a scalar, which is
    broadcast to all stages. ``warmup_mode`` and ``warmup_Nit`` define an
    independent schedule run once before the joint iterations; optional
    ``warmup_beta_*``, ``warmup_alpha_*``, and ``warmup_TV_freq`` settings
    override its inherited controls.

    Selectable projection models (check Eckart–Young–Mirsky theorem for optimal low-rank approximation)
    ----------------------------
    projection_model='none'
        Independent single-energy updates with no cross-energy constraint.

    projection_model='svd'
        Generic static + low-rank model:
            L_E(r) = C(r) + Delta_E(r), rank(Delta) <= K.

    projection_model='rank1_spectral'
        Explicit spectral model:
            L_E(r) = C(r) + M(r) a_E.
        Here a_E is a complex energy dependence that can be constrained by
        Kramers-Kronig or by a known absorption spectrum.

    Spectral constraints for projection_model='rank1_spectral'
    ----------------------------------------------------------
    spectral_constraint='free'
        Fit arbitrary complex a_E.

    spectral_constraint='kk'
        Use the retrieved absorption-like part of a_E and compute the
        dispersion-like part via a discrete Kramers-Kronig transform.

    spectral_constraint='known_beta'
        Replace the absorption-like part of a_E by supplied beta(E), the
        imaginary part of the refractive index.

    spectral_constraint='known_beta_kk'
        Use supplied beta(E) and either supplied delta(E), or compute delta via
        the standalone Kramers-Kronig library.

    Parameters
    ----------
    holograms : array, shape (nE, nx, ny)
        Intensity holograms / diffraction intensities at different energies.
    mask_pixel : array, shape (nx, ny) or (nE, nx, ny)
        Fourier-domain invalid-pixel mask. Nonzero values are unconstrained.
    supportmask : array, shape (nx, ny)
        Real-space support mask used by PhaseRtrv_core.
    multi_energy_recipe : dict or None
        Overrides entries from default_multi_energy_phase_retrieval_recipe().
    start_fields : array or None, shape (nE, nx, ny)
        Optional initial Fourier-domain fields.

    Returns
    -------
    retrieved : array, shape (nE, nx, ny)
        Final Fourier-domain reconstructions, one per energy.
    components : dict
        Static/energy-dependent components from the final cross-energy
        projection. If ``final_fourier_constraint`` is True, these components
        describe the fields immediately before the final measured-amplitude
        constraint; the dictionary records that constraint under
        ``final_fourier_constraint_applied``.
    bsmasks : array, shape (nE, nx, ny)
        Energy-specific beamstop/invalid masks.
    error : dict
        Error history and projection diagnostics.
    """
    recipe = default_multi_energy_phase_retrieval_recipe()
    if multi_energy_recipe is not None:
        if not isinstance(multi_energy_recipe, dict):
            raise TypeError("multi_energy_recipe must be a dictionary.")
        unknown_keys = set(multi_energy_recipe) - set(recipe)
        if unknown_keys:
            raise ValueError(
                f"Unknown multi-energy recipe key(s): {sorted(unknown_keys)}"
            )
        recipe.update(multi_energy_recipe)

    holograms, mask_pixel, supportmask, start_fields, input_geometry = geometry.prepare(
        holograms, mask_pixel, supportmask, recipe, start_fields
    )
    holograms = _as_energy_stack(holograms)
    supportmask = np.asarray(supportmask)

    nE, nx, ny = holograms.shape
    _verify_multi_energy_recipe(recipe, nE)
    material_thickness = _recipe_material_thickness(recipe, input_geometry)
    if material_thickness is not None:
        material_thickness = np.fft.fftshift(material_thickness)
    if supportmask.shape != (nx, ny):
        raise ValueError("supportmask must have shape (nx, ny).")
    projection_supportmask = (
        np.fft.fftshift(supportmask)
        if (
            recipe["projection_constraints_inside_support_only"]
            or recipe["physical_constraints_inside_support_only"]
        )
        else None
    )

    amplitudes, intensities, bsmasks = _prepare_energy_amplitudes(
        holograms,
        mask_pixel,
        hologram_intensity_cutoff_vmin=recipe["hologram_intensity_cutoff_vmin"],
    )

    mask_stack = _as_energy_mask(mask_pixel, nE=nE, image_shape=(nx, ny))

    if start_fields is None:
        start = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(supportmask)))
        fields = np.repeat(start[None, :, :], nE, axis=0).astype(np.complex128)

        # Rough per-energy amplitude normalization, analogous to the original
        # single-energy initialization.
        for j in range(nE):
            valid_j = (mask_stack[j] == 0) & (intensities[j] > 0)
            if np.any(valid_j):
                x = amplitudes[j][valid_j].ravel()
                y = np.abs(fields[j][valid_j]).ravel()
                if x.size >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
                    res = stats.linregress(x, y)
                    if abs(res.slope) > 1e-12:
                        fields[j] = fields[j] / res.slope
    else:
        fields = _as_energy_stack(start_fields, name="start_fields").astype(
            np.complex128,
            copy=True,
        )
        if fields.shape != holograms.shape:
            raise ValueError("start_fields must have the same shape as holograms.")

    rng = np.random.default_rng(recipe["random_seed"])
    errors = {
        "energy_steps": [],
        "projection_steps": [],
        "settings": recipe.copy(),
    }
    start_time = time.time()
    print(
        "Universal phase retrieval: preparing pure-energy reconstruction "
        f"with {nE} energies",
        flush=True,
    )

    warmup_schedule = _build_update_schedule(
        recipe,
        name="warmup",
        allow_disabled=True,
    )
    if warmup_schedule:
        print(
            "Universal phase retrieval: starting independent warmup "
            f"for {nE} energies",
            flush=True,
        )
        for j in range(nE):
            print(f"  Warmup energy {j + 1}/{nE}", flush=True)
            fields[j], stage_results = _run_energy_update_schedule(
                fields[j],
                amplitudes[j],
                supportmask,
                bsmasks[j],
                warmup_schedule,
                recipe,
                phase_retrieval_kernel=phase_retrieval_kernel,
            )
            for stage_result in stage_results:
                errors["energy_steps"].append({
                    "outer": -1,
                    "energy": j,
                    "stage": "warmup",
                    **stage_result,
                })
        print("Universal phase retrieval: warmup complete", flush=True)
    else:
        print("Universal phase retrieval: warmup disabled", flush=True)

    fieldswarmup=fields.copy()
    outer_iterations = int(recipe["outer_iterations"])
    inner_schedule = _build_update_schedule(
        recipe,
        name="inner",
    )
    projection_every, projection_start = _resolve_projection_cadence(
        recipe,
        default_every=nE,
    )
    completed_updates = 0

    components = {"projection_model": recipe["projection_model"]}

    print(
        "Universal phase retrieval: starting pure-energy reconstruction "
        f"with {nE} energies and {outer_iterations} outer loops",
        flush=True,
    )
    # A sweep visits every energy once. Projection cadence can be finer than
    # a sweep, so it is driven by the completed-update counter.
    for outer in range(outer_iterations):
        print(
            f"Universal phase retrieval: outer loop "
            f"{outer + 1}/{outer_iterations}",
            flush=True,
        )
        if recipe["shuffle_energies"]:
            energy_order = rng.permutation(nE)
        else:
            energy_order = np.arange(nE)

        for position, j in enumerate(energy_order, start=1):
            print(
                "  Updating energy "
                f"{position}/{nE} (index {int(j)})",
                flush=True,
            )
            fields[j], stage_results = _run_energy_update_schedule(
                fields[j],
                amplitudes[j],
                supportmask,
                bsmasks[j],
                inner_schedule,
                recipe,
                phase_retrieval_kernel=phase_retrieval_kernel,
            )
            for stage_result in stage_results:
                errors["energy_steps"].append({
                    "outer": outer,
                    "energy": int(j),
                    "stage": "joint",
                    **stage_result,
                })

            completed_updates += 1
            if not _projection_is_due(
                completed_updates,
                projection_start,
                projection_every,
            ):
                continue
            if recipe["projection_relaxation"] == 0:
                continue

            # The projection sees the complete current energy stack.
            print("  Applying joint energy projection", flush=True)
            fields, components = project_fourier_fields_multi_energy(
                fields,
                projection_model=recipe["projection_model"],
                rank=recipe["rank"],
                static_mode=recipe["projection_static_mode"],
                weights=recipe["energy_weights"],
                relaxation=recipe["projection_relaxation"],
                log_floor=recipe["log_floor"],
                spectral_constraint=recipe["spectral_constraint"],
                energy_values=recipe["energy_values"],
                known_beta_spectrum=recipe["known_beta_spectrum"],
                known_delta_spectrum=recipe["known_delta_spectrum"],
                absorption_part=recipe["absorption_part"],
                kk_sign=recipe["kk_sign"],
                kk_subtract_baseline=recipe["kk_subtract_baseline"],
                kk_normalize_input=recipe["kk_normalize_input"],
                known_beta_normalization=recipe["known_beta_normalization"],
                fit_known_beta_scale=recipe["fit_known_beta_scale"],
                fit_known_beta_offset=recipe["fit_known_beta_offset"],
                projection_supportmask=projection_supportmask,
                material_thickness=material_thickness,
                fit_material_thickness=recipe["fit_material_thickness"],
                return_components=True,
            )
            errors["projection_steps"].append(
                {
                    "outer": outer,
                    "energy": int(j),
                    "completed_update": completed_updates,
                    "projection_model": components.get("projection_model"),
                    "rank": recipe["rank"],
                    "spectral_constraint": components.get(
                        "spectral_constraint",
                        recipe["spectral_constraint"],
                    ),
                    "relaxation": recipe["projection_relaxation"],
                    "singular_values": components.get("singular_values"),
                    "spectral_coefficients": components.get(
                        "spectral_coefficients"
                    ),
                }
            )
            print("  Joint energy projection complete", flush=True)

    # Final projection, unless the user explicitly selected no projection.
    print("Universal phase retrieval: applying final energy projection", flush=True)
    projected_fields, components = project_fourier_fields_multi_energy(
        fields,
        projection_model=recipe["projection_model"],
        rank=recipe["rank"],
        static_mode=recipe["projection_static_mode"],
        weights=recipe["energy_weights"],
        relaxation=recipe["final_projection_relaxation"],
        log_floor=recipe["log_floor"],
        spectral_constraint=recipe["spectral_constraint"],
        energy_values=recipe["energy_values"],
        known_beta_spectrum=recipe["known_beta_spectrum"],
        known_delta_spectrum=recipe["known_delta_spectrum"],
        absorption_part=recipe["absorption_part"],
        kk_sign=recipe["kk_sign"],
        kk_subtract_baseline=recipe["kk_subtract_baseline"],
        kk_normalize_input=recipe["kk_normalize_input"],
        known_beta_normalization=recipe["known_beta_normalization"],
        fit_known_beta_scale=recipe["fit_known_beta_scale"],
        fit_known_beta_offset=recipe["fit_known_beta_offset"],
        projection_supportmask=projection_supportmask,
        material_thickness=material_thickness,
        fit_material_thickness=recipe["fit_material_thickness"],
        return_components=True,
    )
    if recipe["final_projection_relaxation"] > 0:
        fields = projected_fields
    components["final_projection_relaxation"] = recipe["final_projection_relaxation"]

    if recipe["final_fourier_constraint"]:
        for j in range(nE):
            constrained = amplitudes[j] * np.exp(1j * np.angle(fields[j]))
            fields[j] = np.where(bsmasks[j] != 0, fields[j], constrained)
        components["final_fourier_constraint_applied"] = True
    else:
        components["final_fourier_constraint_applied"] = False

    errors["runtime_seconds"] = float(np.round(time.time() - start_time, 3))
    print(
        "Universal phase retrieval: complete in "
        f"{errors['runtime_seconds']:.3f} s",
        flush=True,
    )

    return input_geometry.finish(fields, fieldswarmup, components, bsmasks, errors)


def default_general_phase_retrieval_recipe():
    """Return defaults for joint state/energy/polarization/beam retrieval."""
    return {
        # Phase-retrieval stages applied independently to every observation.
        "inner_mode": ["HAPRE"],
        "inner_Nit": [1],
        "outer_iterations": 300,
        "warmup_mode": ["HAPRE"],
        "warmup_Nit": [20],
        # Optional explicit schedules for the reference observation and all
        # other observations.  None keeps the shared warmup schedule above.
        "warmup_reference_mode": None,
        "warmup_reference_Nit": None,
        "warmup_other_mode": None,
        "warmup_other_Nit": None,
        "warmup_start_from_first": False,
        "warmup_reference_observation": 0,
        "startimage_scale_fit": "linear",  # or "through_origin"
        "startimage_radial_normalization": False,
        "warmup_seed_scale_fit": "sum",  # or "linear" intensity slope
        "warmup_seeded_stage_indices": None,  # None repeats the full schedule
        "freeze_saturated_fields": False,
        "partial_coherence": False,
        "coherence_kernel_scope": "per_observation",  # or "shared" across states
        # sequential: hand gamma to the next state; average: independent fits
        # from one snapshot; pooled: fixed gamma per sweep then joint RL fit.
        "shared_coherence_update": "sequential",
        "warmup_calibrate_shared_gamma": False,
        "coherence_round_iterations": 50,
        "coherent_refresh_rounds": [],  # 1-based outer rounds, before their PC updates
        "coherent_refresh_iterations": 50,
        "coherent_refresh_mode": "ER",
        "observation_workers": 1,
        "fft_workers": 1,
        # Complete all coherent warmups first; freeze their missing-pixel targets
        # for every later partial-coherence stage (sum of intensities for modes).
        "preserve_warmup_masked_intensity": False,
        # Shared scope only: calibrate gamma in reference warmup, then hold it.
        "freeze_coherence_after_reference": False,
        "shuffle_observations": True,
        "random_seed": None,
        "beta_zero": 0.5,
        "beta_mode": "arctan",
        "alpha_zero": 0.0,
        "alpha_mode": "const",
        "TV_freq": 1e9,
        "RL_it": 0,
        "RL_freq": 1e9,
        "warmup_beta_zero": None,
        "warmup_beta_mode": None,
        "warmup_alpha_zero": None,
        "warmup_alpha_mode": None,
        "warmup_TV_freq": None,
        "warmup_RL_it": None,
        "warmup_RL_freq": None,
        "warmup_reference_beta_zero": None,
        "warmup_reference_beta_mode": None,
        "warmup_reference_alpha_zero": None,
        "warmup_reference_alpha_mode": None,
        "warmup_reference_TV_freq": None,
        "warmup_reference_RL_it": None,
        "warmup_reference_RL_freq": None,
        "warmup_other_beta_zero": None,
        "warmup_other_beta_mode": None,
        "warmup_other_alpha_zero": None,
        "warmup_other_alpha_mode": None,
        "warmup_other_TV_freq": None,
        "warmup_other_RL_it": None,
        "warmup_other_RL_freq": None,
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
        "mode_support_center": "image",
        "recenter_modal_supports": False,
        "mode_initialization": "random_phase",  # or "support_fft"
        "mode_initialization_seed": 0,
        # Optionally remove state dependence from support-factor modes other
        # than 1 by projecting them to one common object per energy/beam.
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
        "physical_projection_object_roi": False,
        "physical_phase_reference": False,
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
        "saturated_states": None,
        # If True, force the fitted magnetic state maps to zero outside the
        # real-space support used by the phase-retrieval kernel.
        "zero_magnetization_outside_support": False,
        # If True, force the material thickness to zero outside that support.
        # A supplied material mask can additionally mark holes inside it.
        "zero_thickness_outside_support": False,
        # If True, apply the selected joint log-object projection only inside
        # the real-space support; outside it, keep each observation unchanged.
        "projection_constraints_inside_support_only": False,
        # Backward-compatible alias for older physical-only recipes.
        "physical_constraints_inside_support_only": False,
        "charge_spectral_constraint": "free",
        "magnetic_spectral_constraint": "free",
        "energy_values": None,
        "known_charge_beta_spectrum": None,
        "known_charge_delta_spectrum": None,
        "known_magnetic_beta_spectrum": None,
        "known_magnetic_delta_spectrum": None,
        "charge_absorption_part": "real",
        "magnetic_absorption_part": "real",
        "charge_response_real_range": None,
        "charge_response_imag_range": None,
        "magnetic_response_real_range": None,
        "magnetic_response_imag_range": None,
        "kk_sign": 1.0,
        "kk_subtract_baseline": True,
        "kk_normalize_input": False,
        "known_spectrum_normalization": "none",
        "fit_known_spectrum_scale": True,
        "fit_known_spectrum_offset": True,
        "log_floor": 1e-12,
    }


def _ordered_unique(values):
    """Return unique hashable values in first-occurrence order."""
    unique = []
    lookup = {}
    indices = np.empty(len(values), dtype=int)
    for index, value in enumerate(values):
        try:
            value_index = lookup[value]
        except KeyError:
            lookup[value] = len(unique)
            unique.append(value)
            value_index = lookup[value]
        except TypeError as exc:
            raise ValueError("Metadata labels must be hashable.") from exc
        indices[index] = value_index
    return unique, indices


def _normalize_metadata(
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    n_observations,
):
    """
    Validate and encode all observation metadata used by the model.

    Context:
    Observation labels define which fitted quantities are shared. Preserve encounter order
    when assigning group indices so component arrays remain aligned with the input
    observations.
    """
    states = np.asarray(state_labels)
    energies = np.asarray(energy_labels)
    polarizations = np.asarray(polarization_coefficients, dtype=float)
    beams = np.asarray(beam_labels)
    expected_shape = (n_observations,)
    if states.shape != expected_shape:
        raise ValueError("state_labels must have shape (n_observations,).")
    if energies.shape != expected_shape:
        raise ValueError("energy_labels must have shape (n_observations,).")
    if polarizations.shape != expected_shape:
        raise ValueError(
            "polarization_coefficients must have shape (n_observations,)."
        )
    if beams.shape != expected_shape:
        raise ValueError("beam_labels must have shape (n_observations,).")
    if np.any(~np.isfinite(polarizations)) or np.any(polarizations == 0):
        raise ValueError(
            "polarization_coefficients must be finite and nonzero."
        )

    state_names, state_indices = _ordered_unique(states.tolist())
    energy_names, energy_indices = _ordered_unique(energies.tolist())
    beam_names, beam_indices = _ordered_unique(beams.tolist())
    response_names, response_indices = _ordered_unique(
        list(zip(states.tolist(), energies.tolist()))
    )
    return {
        "states": states,
        "energies": energies,
        "polarizations": polarizations,
        "beams": beams,
        "state_names": state_names,
        "state_indices": state_indices,
        "energy_names": energy_names,
        "energy_indices": energy_indices,
        "beam_names": beam_names,
        "beam_indices": beam_indices,
        "response_names": response_names,
        "response_indices": response_indices,
    }


def _observation_weights(weights, n_observations):
    """Return validated positive weights, defaulting to equal weighting."""
    if weights is None:
        return np.ones(n_observations, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_observations,):
        raise ValueError(
            "observation_weights must have shape (n_observations,)."
        )
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("observation_weights must be finite and positive.")
    return weights


def _normalize_saturated_states(saturated_states, state_names):
    """Return validated ``{state: +1/-1}`` saturated-state metadata."""
    if saturated_states is None:
        return {}
    if isinstance(saturated_states, dict):
        saturated = dict(saturated_states)
    elif isinstance(saturated_states, (list, tuple, set, np.ndarray)):
        saturated = {state: 1.0 for state in list(saturated_states)}
    else:
        raise ValueError(
            "saturated_states must be a sequence or a state-to-sign mapping."
        )
    unknown = set(saturated) - set(state_names)
    if unknown:
        raise ValueError(
            "saturated_states contains unknown states: "
            f"{sorted(unknown, key=str)}"
        )
    for state, value in saturated.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or abs(float(value)) != 1.0
        ):
            raise ValueError(
                f"Saturation value for state {state!r} must be +1 or -1."
            )
        saturated[state] = float(value)
    return saturated


def _normalize_range(value, name):
    """Validate an optional finite ``(minimum, maximum)`` interval."""
    if value is None:
        return None
    values = np.asarray(value, dtype=float)
    if values.shape != (2,) or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be a finite (minimum, maximum) pair.")
    minimum, maximum = map(float, values)
    if minimum > maximum:
        raise ValueError(f"{name} minimum must not exceed its maximum.")
    return minimum, maximum


def _constrain_response_values(response, real_range=None, imag_range=None):
    """Clip real and imaginary response components to optional intervals."""
    constrained = np.asarray(response, dtype=np.complex128).copy()
    real_range = _normalize_range(real_range, "response_real_range")
    imag_range = _normalize_range(imag_range, "response_imag_range")
    if real_range is not None:
        constrained.real = np.clip(constrained.real, *real_range)
    if imag_range is not None:
        constrained.imag = np.clip(constrained.imag, *imag_range)
    return constrained


def project_log_objects_general(
    log_objects,
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    weights=None,
    relaxation=1.0,
    rank_deficient="error",
    projection_supportmask=None,
    return_components=False,
):
    """
    Fit ``L_a = C_beam(a) + p_a R_state(a),energy(a)``.

    The design is identifiable only when the observation rows separate every
    requested beam component and state/energy response. By default a
    rank-deficient geometry is rejected instead of choosing an arbitrary gauge.
    """
    log_objects = _as_energy_stack(log_objects, name="log_objects")
    if not (0 <= relaxation <= 1):
        raise ValueError("relaxation must be between 0 and 1.")
    if rank_deficient not in {"error", "minimum_norm"}:
        raise ValueError(
            "rank_deficient must be 'error' or 'minimum_norm'."
        )

    n_observations, nx, ny = log_objects.shape
    if projection_supportmask is not None:
        projection_supportmask = np.asarray(projection_supportmask) != 0
        if projection_supportmask.shape != (nx, ny):
            raise ValueError(
                "projection_supportmask must have shape (nx, ny)."
            )
    metadata = _normalize_metadata(
        state_labels,
        energy_labels,
        polarization_coefficients,
        beam_labels,
        n_observations,
    )
    weights = _observation_weights(weights, n_observations)

    # Build one explicit linear model for beam fields and material responses.
    n_beams = len(metadata["beam_names"])
    n_responses = len(metadata["response_names"])
    design = np.zeros(
        (n_observations, n_beams + n_responses),
        dtype=float,
    )
    rows = np.arange(n_observations)
    design[rows, metadata["beam_indices"]] = 1.0
    design[
        rows,
        n_beams + metadata["response_indices"],
    ] = metadata["polarizations"]

    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    design_rank = np.linalg.matrix_rank(weighted_design)
    n_components = design.shape[1]
    if design_rank < n_components and rank_deficient == "error":
        raise ValueError(
            "The general state/energy/polarization/beam design is rank "
            "deficient. Add observations that connect beam conditions to "
            "shared state-energy responses, or add opposite-polarization "
            "measurements. Use rank_deficient='minimum_norm' only if an "
            "arbitrary pseudoinverse solution is acceptable."
        )

    data = log_objects.reshape(n_observations, -1)
    coefficients = np.linalg.pinv(weighted_design) @ (
        data * sqrt_weights[:, None]
    )
    fitted_data = design @ coefficients
    fitted = fitted_data.reshape(n_observations, nx, ny)
    projected = (
        fitted
        if relaxation == 1
        else (1 - relaxation) * log_objects + relaxation * fitted
    )
    if projection_supportmask is not None:
        projected = np.where(
            projection_supportmask[None],
            projected,
            log_objects,
        )

    if not return_components:
        return projected

    common_log_objects = coefficients[:n_beams].reshape(n_beams, nx, ny)
    response_log_objects = coefficients[n_beams:].reshape(
        n_responses,
        nx,
        ny,
    )
    response_energy_lookup = {
        label: index
        for index, label in enumerate(metadata["energy_names"])
    }
    response_energy_indices = np.asarray(
        [
            response_energy_lookup[energy]
            for _, energy in metadata["response_names"]
        ],
        dtype=int,
    )
    residual = data - fitted_data
    components = {
        "projection_model": "state_energy_beam",
        "common_log_objects": common_log_objects,
        "common_log_objects_by_beam": {
            label: common_log_objects[index]
            for index, label in enumerate(metadata["beam_names"])
        },
        "common_exit_waves": np.exp(common_log_objects),
        "response_log_objects": response_log_objects,
        "response_log_objects_by_state_energy": {
            pair: response_log_objects[index]
            for index, pair in enumerate(metadata["response_names"])
        },
        "state_names": metadata["state_names"],
        "energy_names": metadata["energy_names"],
        "beam_names": metadata["beam_names"],
        "response_names": metadata["response_names"],
        "response_energy_indices": response_energy_indices,
        "state_indices": metadata["state_indices"],
        "energy_indices": metadata["energy_indices"],
        "beam_indices": metadata["beam_indices"],
        "response_indices": metadata["response_indices"],
        "polarization_coefficients": metadata["polarizations"],
        "design_matrix": design,
        "design_rank": design_rank,
        "design_condition_number": float(np.linalg.cond(weighted_design)),
        "identifiable": design_rank == n_components,
        "fit_residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        "projection_supportmask_applied": (
            projection_supportmask is not None
        ),
    }
    return projected, components


def largest_support_component(supportmask):
    """Binary thickness template for the largest connected support aperture.

    The input is an unshifted 2D object-space support. Diagonal neighbors
    belong to the same aperture. The result has the same shape and uint8 type.
    """
    from scipy.ndimage import label

    support = np.asarray(supportmask)
    if support.ndim != 2:
        raise ValueError("supportmask must be a 2D array.")
    labels, count = label(support != 0, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        raise ValueError("supportmask has no nonzero aperture.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return (labels == np.argmax(sizes)).astype(np.uint8)


def project_log_objects_physical(
    log_objects,
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    weights=None,
    relaxation=1.0,
    saturated_states=None,
    iterations=20,
    recipe=None,
    magnetization_supportmask=None,
    projection_supportmask=None,
    material_thickness=None,
    fit_material_thickness=False,
    thickness_supportmask=None,
    return_components=False,
):
    """
    Fit ``L = C_beam + t(r)[q_charge(E) + p*q_magnetic(E)*mz_state(r)]``.

    ``mz_state`` is real and clipped to ``[-1, 1]``. Saturated states, when
    supplied, are fixed to +1 or -1. Charge and magnetic response spectra can
    be constrained through the local free/KK/known-beta spectral options,
    followed by optional rectangular value bounds.

    Context:
    Physical-fit boundary: input is a stack of complex object logarithms, not detector
    intensities. With object ROI enabled, pack only positive-thickness pixels before phase
    adjustment and fitting, then scatter them back while preserving the exterior.
    """
    log_objects = _as_energy_stack(log_objects, name="log_objects")
    if not (0 <= relaxation <= 1):
        raise ValueError("relaxation must be between 0 and 1.")
    if isinstance(iterations, bool) or not isinstance(
        iterations,
        (int, np.integer),
    ) or iterations <= 0:
        raise ValueError("physical_iterations must be a positive integer.")
    recipe = default_general_phase_retrieval_recipe() if recipe is None else recipe
    if recipe.get("physical_projection_object_roi", False):
        if material_thickness is None:
            raise ValueError("Object-ROI physical projection requires a material thickness/aperture map.")
        aperture = np.asarray(material_thickness) > 0
        if projection_supportmask is not None:
            aperture &= np.asarray(projection_supportmask) != 0
        if not np.any(aperture):
            raise ValueError("Physical projection aperture is empty.")
        # A compact (active_pixels, 1) image reuses the spatial fitting code
        # without allocating observations × modes × the full detector area.
        def pack(array):
            return None if array is None else np.asarray(array)[aperture][:, None]
        local_recipe = dict(recipe, physical_projection_object_roi=False)
        local, components = project_log_objects_physical(
            log_objects[:, aperture, None], state_labels, energy_labels,
            polarization_coefficients, beam_labels, weights=weights,
            relaxation=relaxation, saturated_states=saturated_states,
            iterations=iterations, recipe=local_recipe,
            magnetization_supportmask=pack(magnetization_supportmask),
            projection_supportmask=np.ones((int(aperture.sum()), 1), dtype=bool),
            material_thickness=pack(material_thickness),
            fit_material_thickness=fit_material_thickness,
            thickness_supportmask=pack(thickness_supportmask), return_components=True,
        )
        # Fit only active aperture pixels. Vacuum and reference holes are not
        # silently overwritten by the zero-thickness common-field constraint.
        projected = log_objects.copy()
        projected[:, aperture] = local[..., 0]
        # Restore map shapes for plotting/saving, but do not label unused
        # exterior component values as physically fitted information.
        def expand(value):
            if isinstance(value, dict):
                return {key: expand(item) for key, item in value.items()}
            if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[-2:] == local.shape[-2:]:
                full = np.zeros(value.shape[:-2] + aperture.shape, dtype=value.dtype)
                full[..., aperture] = value[..., 0]
                return full
            return value
        components = expand(components)
        components["physical_phase_reference"] = bool(recipe.get("physical_phase_reference", False))
        components["physical_projection_roi_mask"] = aperture.copy()
        components["physical_projection_fit_pixels"] = int(aperture.sum())
        components["physical_projection_full_pixels"] = int(aperture.size)
        return (projected, components) if return_components else projected
    if recipe.get("physical_phase_reference", False):
        # Pick equivalent complex-log branches relative to the first state in
        # each energy/beam group. This preserves exp(log_object) exactly while
        # preventing a +/- pi wrap from masquerading as a large magnetic phase.
        # Assumption: relative state phases lie on the nearest reference branch.
        log_objects = log_objects.copy()
        phase_groups = {}
        for index, key in enumerate(zip(energy_labels, beam_labels)):
            phase_groups.setdefault(key, []).append(index)
        for indices in phase_groups.values():
            reference_phase = log_objects[indices[0]].imag.copy()
            log_objects.imag[indices] = reference_phase + np.angle(np.exp(
                1j * (log_objects.imag[indices] - reference_phase),
            ))
    if (fit_material_thickness and not recipe["fit_known_spectrum_scale"] and
            any(str(recipe[key]).lower() in {"known_beta", "known-beta", "known_beta_kk", "known-beta-kk"}
                for key in ("charge_spectral_constraint", "magnetic_spectral_constraint"))):
        raise ValueError("Fitting relative thickness requires free charge and magnetic spectral scales.")

    n_observations, nx, ny = log_objects.shape
    thickness = _validate_material_thickness(material_thickness, (nx, ny))
    if thickness is None:
        thickness = np.ones((nx, ny), dtype=float)
    thickness = thickness.copy()
    if recipe["zero_thickness_outside_support"]:
        if thickness_supportmask is None:
            raise ValueError(
                "thickness_supportmask is required when "
                "zero_thickness_outside_support=True."
            )
        thickness_supportmask = np.asarray(thickness_supportmask) != 0
        if thickness_supportmask.shape != (nx, ny):
            raise ValueError("thickness_supportmask must have shape (nx, ny).")
        thickness *= thickness_supportmask
    allowed_material = thickness > 0
    metadata = _normalize_metadata(
        state_labels,
        energy_labels,
        polarization_coefficients,
        beam_labels,
        n_observations,
    )
    weights = _observation_weights(weights, n_observations)
    saturated = _normalize_saturated_states(
        saturated_states,
        metadata["state_names"],
    )
    n_states = len(metadata["state_names"])
    n_energies = len(metadata["energy_names"])
    n_beams = len(metadata["beam_names"])
    if magnetization_supportmask is not None:
        magnetization_supportmask = np.asarray(magnetization_supportmask) != 0
        if magnetization_supportmask.shape != (nx, ny):
            raise ValueError(
                "magnetization_supportmask must have shape (nx, ny)."
            )
    if projection_supportmask is not None:
        projection_supportmask = np.asarray(projection_supportmask) != 0
        if projection_supportmask.shape != (nx, ny):
            raise ValueError(
                "projection_supportmask must have shape (nx, ny)."
            )

    # Initialize beam fields from weighted observation means.
    common = np.zeros((n_beams, nx, ny), dtype=np.complex128)
    beam_weight = np.zeros(n_beams, dtype=float)
    for observation in range(n_observations):
        beam = metadata["beam_indices"][observation]
        common[beam] += weights[observation] * log_objects[observation]
        beam_weight[beam] += weights[observation]
    common /= beam_weight[:, None, None]

    charge = np.zeros(n_energies, dtype=np.complex128)
    magnetic = np.ones(n_energies, dtype=np.complex128)
    magnetization = np.zeros((n_states, nx, ny), dtype=float)
    for state, sign in saturated.items():
        state_index = metadata["state_names"].index(state)
        magnetization[state_index] = sign
    # During thickness fitting, keep the known saturated sign at every pixel
    # where material is allowed. At zero fitted thickness it has no effect on
    # the exit wave, but retains a direction for a later thickness update.
    magnetization *= allowed_material if fit_material_thickness else thickness > 0
    if magnetization_supportmask is not None:
        magnetization *= magnetization_supportmask

    # Estimate an initial complex magnetic direction independently per energy.
    for energy in range(n_energies):
        selection = metadata["energy_indices"] == energy
        residuals = []
        for observation in np.flatnonzero(selection):
            beam = metadata["beam_indices"][observation]
            polarization = metadata["polarizations"][observation]
            residuals.append(
                (log_objects[observation] - common[beam]) / polarization
            )
        if residuals:
            residuals = np.asarray(residuals)
            moment = np.sum(residuals**2)
            if np.isfinite(moment) and abs(moment) > 1e-30:
                direction = np.exp(0.5j * np.angle(moment))
                amplitude = np.max(
                    np.abs(
                        np.real(np.conj(direction) * residuals)
                    )
                )
                magnetic[energy] = direction * max(amplitude, 1e-12)

    charge_spectral_info = {}
    magnetic_spectral_info = {}
    for _ in range(int(iterations)):
        # Fit one common complex field for every beam condition.
        common.fill(0.0)
        beam_weight.fill(0.0)
        for observation in range(n_observations):
            beam = metadata["beam_indices"][observation]
            energy = metadata["energy_indices"][observation]
            state = metadata["state_indices"][observation]
            polarization = metadata["polarizations"][observation]
            modeled_material = (
                charge[energy]
                + polarization
                * magnetic[energy]
                * magnetization[state]
            )
            common[beam] += weights[observation] * (
                log_objects[observation] - thickness * modeled_material
            )
            beam_weight[beam] += weights[observation]
        common /= beam_weight[:, None, None]

        # Probe zero-thickness pixels that are still allowed to contain
        # material. Using the exact zero there makes the magnetic update
        # singular and can turn zero thickness into an absorbing state.
        magnetization_fit_thickness = thickness
        if fit_material_thickness:
            positive = thickness[allowed_material & (thickness > 0)]
            probe_thickness = 1e-3 * np.mean(positive) if positive.size else 1e-3
            magnetization_fit_thickness = np.where(
                allowed_material & (thickness == 0), probe_thickness, thickness,
            )

        # Fit real reduced-magnetization maps while enforcing |mz| <= 1.
        for state, state_name in enumerate(metadata["state_names"]):
            if state_name in saturated:
                magnetization[state].fill(saturated[state_name])
                magnetization[state] *= (
                    allowed_material if fit_material_thickness else thickness > 0
                )
                if magnetization_supportmask is not None:
                    magnetization[state] *= magnetization_supportmask
                continue
            numerator = np.zeros((nx, ny), dtype=float)
            denominator = np.zeros((nx, ny), dtype=float)
            for observation in np.flatnonzero(
                metadata["state_indices"] == state
            ):
                beam = metadata["beam_indices"][observation]
                energy = metadata["energy_indices"][observation]
                coefficient = (
                    metadata["polarizations"][observation]
                    * magnetic[energy]
                    * magnetization_fit_thickness
                )
                residual = (
                    log_objects[observation]
                    - common[beam]
                    - thickness * charge[energy]
                )
                numerator += weights[observation] * np.real(
                    np.conj(coefficient) * residual
                )
                denominator += weights[observation] * abs(coefficient) ** 2
            magnetization[state] = np.clip(
                np.divide(numerator, denominator,
                          out=(magnetization[state].copy() if fit_material_thickness
                               else np.zeros_like(numerator)),
                          where=denominator > 1e-30),
                -1.0, 1.0,
            )
            if magnetization_supportmask is not None:
                magnetization[state] *= magnetization_supportmask

        # Fit scalar charge and magnetic responses independently at each energy.
        for energy in range(n_energies):
            observations = np.flatnonzero(
                metadata["energy_indices"] == energy
            )
            # The model has only two scalar coefficients per energy. Accumulate
            # its 2x2 normal equations instead of materializing one enormous
            # (observations * pixels, 2) complex design matrix.
            gram = np.zeros((2, 2), dtype=np.complex128)
            rhs = np.zeros(2, dtype=np.complex128)
            for observation in observations:
                beam = metadata["beam_indices"][observation]
                state = metadata["state_indices"][observation]
                magnetic_basis = (
                    metadata["polarizations"][observation]
                    * magnetization[state]
                    * thickness
                )
                residual = log_objects[observation] - common[beam]
                weight = weights[observation]
                gram[0, 0] += weight * np.vdot(thickness, thickness)
                gram[0, 1] += weight * np.vdot(thickness, magnetic_basis)
                gram[1, 1] += weight * np.vdot(magnetic_basis, magnetic_basis)
                rhs[0] += weight * np.vdot(thickness, residual)
                rhs[1] += weight * np.vdot(magnetic_basis, residual)
            gram[1, 0] = np.conj(gram[0, 1])
            coefficients = np.linalg.pinv(gram) @ rhs
            charge[energy], magnetic[energy] = coefficients

        if fit_material_thickness:
            # Eliminate the free beam field by centering observations within
            # each illumination condition, then solve for a shared real map.
            numerator = np.zeros((nx, ny), dtype=float)
            denominator = np.zeros((nx, ny), dtype=float)
            for beam in range(n_beams):
                observations = np.flatnonzero(metadata["beam_indices"] == beam)
                total_weight = np.sum(weights[observations])
                responses = np.stack([
                    charge[metadata["energy_indices"][a]]
                    + metadata["polarizations"][a]
                    * magnetic[metadata["energy_indices"][a]]
                    * magnetization[metadata["state_indices"][a]]
                    for a in observations
                ])
                mean_response = np.sum(
                    responses * weights[observations, None, None], axis=0,
                ) / total_weight
                mean_log = np.sum(
                    log_objects[observations]
                    * weights[observations, None, None], axis=0,
                ) / total_weight
                for index, observation in enumerate(observations):
                    direction = responses[index] - mean_response
                    residual = log_objects[observation] - mean_log
                    numerator += weights[observation] * np.real(
                        np.conj(direction) * residual
                    )
                    denominator += weights[observation] * np.abs(direction) ** 2
            if np.any(denominator[allowed_material] > 1e-30):
                updated = np.maximum(
                    0, np.divide(numerator, denominator,
                                 out=thickness.copy(), where=denominator > 1e-30),
                )
                updated[~allowed_material] = 0
                scale = np.mean(updated[allowed_material])
                if scale > 1e-30:
                    thickness = updated / scale
                    charge *= scale
                    magnetic *= scale

        # Fix the charge/common-field offset gauge before applying priors.
        charge_offset = np.mean(charge)
        charge -= charge_offset
        common += charge_offset * thickness

        charge, charge_spectral_info = constrain_complex_spectrum(
            charge,
            spectral_constraint=recipe["charge_spectral_constraint"],
            energy_values=recipe["energy_values"],
            known_beta_spectrum=recipe["known_charge_beta_spectrum"],
            known_delta_spectrum=recipe["known_charge_delta_spectrum"],
            absorption_part=recipe["charge_absorption_part"],
            kk_sign=recipe["kk_sign"],
            kk_subtract_baseline=recipe["kk_subtract_baseline"],
            kk_normalize_input=recipe["kk_normalize_input"],
            known_beta_normalization=recipe["known_spectrum_normalization"],
            fit_known_beta_scale=recipe["fit_known_spectrum_scale"],
            fit_known_beta_offset=recipe["fit_known_spectrum_offset"],
        )
        magnetic, magnetic_spectral_info = constrain_complex_spectrum(
            magnetic,
            spectral_constraint=recipe["magnetic_spectral_constraint"],
            energy_values=recipe["energy_values"],
            known_beta_spectrum=recipe["known_magnetic_beta_spectrum"],
            known_delta_spectrum=recipe["known_magnetic_delta_spectrum"],
            absorption_part=recipe["magnetic_absorption_part"],
            kk_sign=recipe["kk_sign"],
            kk_subtract_baseline=recipe["kk_subtract_baseline"],
            kk_normalize_input=recipe["kk_normalize_input"],
            known_beta_normalization=recipe["known_spectrum_normalization"],
            fit_known_beta_scale=recipe["fit_known_spectrum_scale"],
            fit_known_beta_offset=recipe["fit_known_spectrum_offset"],
        )
        charge = _constrain_response_values(
            charge,
            recipe["charge_response_real_range"],
            recipe["charge_response_imag_range"],
        )
        magnetic = _constrain_response_values(
            magnetic,
            recipe["magnetic_response_real_range"],
            recipe["magnetic_response_imag_range"],
        )

    # A fitted thickness may have reached zero on the last iteration. Keep the
    # returned magnetic object consistent with the final material domain.
    magnetization *= thickness > 0

    if fit_material_thickness:
        # Keep the illumination estimate consistent with the final fitted
        # thickness and spectral responses.
        common.fill(0.0)
        beam_weight.fill(0.0)
        for observation in range(n_observations):
            beam = metadata["beam_indices"][observation]
            energy = metadata["energy_indices"][observation]
            state = metadata["state_indices"][observation]
            material = thickness * (
                charge[energy]
                + metadata["polarizations"][observation]
                * magnetic[energy] * magnetization[state]
            )
            common[beam] += weights[observation] * (
                log_objects[observation] - material
            )
            beam_weight[beam] += weights[observation]
        common /= beam_weight[:, None, None]

    fitted = np.stack([
        common[metadata["beam_indices"][observation]]
        + thickness * (
            charge[metadata["energy_indices"][observation]]
            + metadata["polarizations"][observation]
            * magnetic[metadata["energy_indices"][observation]]
            * magnetization[metadata["state_indices"][observation]]
        )
        for observation in range(n_observations)
    ])
    projected = (
        fitted
        if relaxation == 1
        else (1 - relaxation) * log_objects + relaxation * fitted
    )
    if material_thickness is not None or thickness_supportmask is not None:
        projected[:, thickness == 0] = fitted[:, thickness == 0]
    if projection_supportmask is not None:
        if material_thickness is not None or thickness_supportmask is not None:
            projection_supportmask = projection_supportmask | (thickness == 0)
        projected = np.where(
            projection_supportmask[None],
            projected,
            log_objects,
        )
    if not return_components:
        return projected

    residual = log_objects - fitted
    charge_mode = str(recipe["charge_spectral_constraint"]).lower()
    magnetic_mode = str(recipe["magnetic_spectral_constraint"]).lower()
    known_modes = {"known_beta", "known-beta", "known_beta_kk", "known-beta-kk"}
    charge_absolute_anchored = (
        charge_mode in known_modes
        and not recipe["fit_known_spectrum_scale"]
        and not recipe["fit_known_spectrum_offset"]
    )
    magnetic_scale_anchored = (
        bool(saturated)
        or (
            magnetic_mode in known_modes
            and not recipe["fit_known_spectrum_scale"]
        )
    )
    components = {
        "projection_model": "physical_factorized",
        "common_log_objects": common,
        "common_log_objects_by_beam": {
            name: common[index]
            for index, name in enumerate(metadata["beam_names"])
        },
        "common_exit_waves": np.exp(common),
        "charge_response": charge,
        "magnetic_response": magnetic,
        "magnetization": magnetization,
        "material_thickness": thickness.copy(),
        "material_thickness_fitted": bool(fit_material_thickness),
        "thickness_supportmask_applied": thickness_supportmask is not None,
        "magnetization_by_state": {
            name: magnetization[index]
            for index, name in enumerate(metadata["state_names"])
        },
        "magnetization_supportmask_applied": (
            magnetization_supportmask is not None
        ),
        "physical_projection_supportmask_applied": (
            projection_supportmask is not None
        ),
        "saturated_states": saturated,
        "state_names": metadata["state_names"],
        "energy_names": metadata["energy_names"],
        "beam_names": metadata["beam_names"],
        "charge_gauge": "zero_mean_before_optional_constraints",
        "magnetic_scale_anchored": magnetic_scale_anchored,
        "charge_absolute_gauge_anchored": charge_absolute_anchored,
        "identifiable": (
            magnetic_scale_anchored and charge_absolute_anchored
            and not fit_material_thickness
        ),
        "fit_residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        "charge_spectral_info": charge_spectral_info,
        "magnetic_spectral_info": magnetic_spectral_info,
        "magnetization_bounds": (-1.0, 1.0),
    }
    return projected, components


def _projection_focus_transform(
    fields,
    energy_labels,
    propagation_um,
    phase_rad,
    setup,
    inverse=False,
    integer_wavelength=True,
):
    """
    Apply the reversible Fourier-plane transform used by ``fth.propagate``.

    Context:
    Projection-only propagation/alignment: move fields to the calibrated physical-model
    plane and undo that transform afterward. This operation still needs a full Fourier grid
    even when the subsequent physical fit uses only an aperture.
    """
    array = _as_energy_stack(fields, name="fields").astype(np.complex128, copy=True)
    if float(propagation_um) == 0:
        return array
    if not isinstance(setup, dict):
        raise ValueError("projection_focus_setup is required for nonzero propagation.")
    ccd_dist = float(setup["ccd_dist"])
    px_size = float(setup["px_size"])
    labels = np.asarray(energy_labels)
    if labels.shape != (array.shape[0],):
        raise ValueError("energy_labels must match the number of focused fields.")
    energies = (np.full(array.shape[0], float(setup["energy"]))
                if "energy" in setup else labels.astype(float))
    if np.any(~np.isfinite(energies)) or np.any(energies <= 0):
        raise ValueError(
            "Focusing requires positive numeric energy labels or "
            "projection_focus_setup['energy']."
        )

    rows, columns = array.shape[-2:]
    yy, xx = np.mgrid[0:rows, 0:columns]
    radius_squared = (yy - rows / 2) ** 2 + (xx - columns / 2) ** 2
    direction = -1.0 if inverse else 1.0
    distance = direction * float(propagation_um) * 1e-6
    global_phase = direction * float(phase_rad)
    hc_over_e = 1.2398419843320026e-6  # Planck*c/e, in eV metre.
    angular_term = 1 - (px_size / ccd_dist) ** 2 * radius_squared
    if np.any(angular_term < 0):
        raise ValueError("Projection-focus geometry produces evanescent detector pixels.")
    angular_root = np.sqrt(angular_term)
    for observation, energy in enumerate(energies):
        wavelength = hc_over_e / energy
        used_distance = (
            np.round(distance / wavelength) * wavelength
            if integer_wavelength else distance
        )
        propagation_phase = 2 * used_distance * np.pi / wavelength * angular_root
        array[observation] *= np.exp(1j * (propagation_phase + global_phase))
    return array


def project_fourier_fields_general(
    fields,
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    weights=None,
    relaxation=1.0,
    rank_deficient="error",
    projection_model="physical_factorized",
    saturated_states=None,
    physical_iterations=20,
    recipe=None,
    log_floor=1e-12,
    magnetization_supportmask=None,
    projection_supportmask=None,
    material_thickness=None,
    fit_material_thickness=False,
    thickness_supportmask=None,
    return_components=False,
):
    """
    Apply the selected general model to Fourier-domain fields.

    Context:
    Bridge detector fields to object-space constraints. Fourier transforms and complex
    logarithms happen here; the downstream projector fits model parameters in log-object
    space.
    """
    focus_prop_um = 0.0 if recipe is None else recipe.get("projection_focus_prop_um", 0.0)
    focus_phase_rad = 0.0 if recipe is None else recipe.get("projection_focus_phase_rad", 0.0)
    focus_setup = None if recipe is None else recipe.get("projection_focus_setup")
    focus_integer_wavelength = (
        True if recipe is None else recipe.get("projection_focus_integer_wavelength", True)
    )
    focused_fields = _projection_focus_transform(
        fields,
        energy_labels,
        focus_prop_um,
        focus_phase_rad,
        focus_setup,
        inverse=False,
        integer_wavelength=focus_integer_wavelength,
    )
    log_objects = fourier_field_to_object_log(
        focused_fields,
        log_floor=log_floor,
        unwrap_energy_phase=False,
    )
    if projection_model == "physical_factorized":
        projected = project_log_objects_physical(
            log_objects,
            state_labels,
            energy_labels,
            polarization_coefficients,
            beam_labels,
            weights=weights,
            relaxation=relaxation,
            saturated_states=saturated_states,
            iterations=physical_iterations,
            recipe=recipe,
            magnetization_supportmask=magnetization_supportmask,
            projection_supportmask=projection_supportmask,
            material_thickness=material_thickness,
            fit_material_thickness=fit_material_thickness,
            thickness_supportmask=thickness_supportmask,
            return_components=return_components,
        )
    else:
        projected = project_log_objects_general(
            log_objects,
            state_labels,
            energy_labels,
            polarization_coefficients,
            beam_labels,
            weights=weights,
            relaxation=relaxation,
            rank_deficient=rank_deficient,
            projection_supportmask=projection_supportmask,
            return_components=return_components,
        )
    if return_components:
        projected_log_objects, components = projected
        projected_fields = object_log_to_fourier_field(projected_log_objects)
        projected_fields = _projection_focus_transform(
            projected_fields,
            energy_labels,
            focus_prop_um,
            focus_phase_rad,
            focus_setup,
            inverse=True,
            integer_wavelength=focus_integer_wavelength,
        )
        components["projection_focus_prop_um"] = float(focus_prop_um)
        components["projection_focus_phase_rad"] = float(focus_phase_rad)
        components["projection_focus_setup"] = focus_setup
        components["physical_components_frame"] = (
            "focused_object_plane" if float(focus_prop_um) != 0 else "retrieval_object_plane"
        )
        return (
            projected_fields,
            components,
        )
    projected_fields = object_log_to_fourier_field(projected)
    return _projection_focus_transform(
        projected_fields,
        energy_labels,
        focus_prop_um,
        focus_phase_rad,
        focus_setup,
        inverse=True,
        integer_wavelength=focus_integer_wavelength,
    )


def response_to_refractive_index(
    response_log_objects,
    wave_numbers,
    thickness,
    response_energy_indices,
):
    """
    Convert ``R_je = i*k_e*t*n_je`` to ``n_je`` when ``k_e`` and ``t`` are known.

    This conversion is deliberately separate from phase retrieval because the
    diffraction data determine the product ``i*k_e*t*n_je``, not its factors.
    """
    response = np.asarray(response_log_objects, dtype=np.complex128)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    response_energy_indices = np.asarray(
        response_energy_indices,
        dtype=int,
    )
    if response.ndim != 3:
        raise ValueError(
            "response_log_objects must have shape (n_responses, nx, ny)."
        )
    if response_energy_indices.shape != (response.shape[0],):
        raise ValueError(
            "response_energy_indices must have shape (n_responses,)."
        )
    if wave_numbers.ndim != 1 or np.any(~np.isfinite(wave_numbers)):
        raise ValueError("wave_numbers must be a finite one-dimensional array.")
    if (
        isinstance(thickness, bool)
        or not np.isscalar(thickness)
        or not np.isfinite(thickness)
        or thickness <= 0
    ):
        raise ValueError("thickness must be a finite positive scalar.")
    if (
        np.any(response_energy_indices < 0)
        or np.any(response_energy_indices >= wave_numbers.size)
    ):
        raise ValueError("response_energy_indices contains an invalid index.")
    selected_wave_numbers = wave_numbers[response_energy_indices]
    if np.any(selected_wave_numbers == 0):
        raise ValueError("wave_numbers must be nonzero.")
    return response / (
        1j * selected_wave_numbers[:, None, None] * float(thickness)
    )


def _verify_recipe(recipe, n_observations, n_energies=None):
    """Validate the general retrieval recipe and observation weights."""
    if (isinstance(recipe["Nmodes"], bool)
            or not isinstance(recipe["Nmodes"], (int, np.integer))
            or recipe["Nmodes"] not in (1, 2)):
        raise ValueError("Nmodes must be 1 or 2 for the physical driver.")
    if recipe["mode_support_center"] not in {"image", "components"}:
        raise ValueError("mode_support_center must be 'image' or 'components'.")
    if not isinstance(recipe["recenter_modal_supports"], bool):
        raise ValueError("recenter_modal_supports must be bool.")
    seed = recipe["mode_initialization_seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise ValueError("mode_initialization_seed must be an integer or None.")
    if recipe["mode_initialization"] not in {"random_phase", "support_fft"}:
        raise ValueError("mode_initialization must be 'random_phase' or 'support_fft'.")
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
        if not recipe["preserve_warmup_masked_intensity"]:
            raise ValueError("Round coherence updates require preserve_warmup_masked_intensity=True.")
    for key in ("preserve_warmup_masked_intensity", "freeze_coherence_after_reference",
                "warmup_calibrate_shared_gamma", "startimage_radial_normalization"):
        if not isinstance(recipe[key], bool):
            raise ValueError(f"{key} must be bool.")
    if not isinstance(recipe["warmup_start_from_first"], bool):
        raise ValueError("warmup_start_from_first must be bool.")
    if (isinstance(recipe["warmup_reference_observation"], bool)
            or not isinstance(recipe["warmup_reference_observation"], (int, np.integer))
            or recipe["warmup_reference_observation"] < 0):
        raise ValueError("warmup_reference_observation must be a non-negative integer.")
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
    if recipe["warmup_seed_scale_fit"] not in {"sum", "linear"}:
        raise ValueError("warmup_seed_scale_fit must be 'sum' or 'linear'.")
    seeded_indices = recipe["warmup_seeded_stage_indices"]
    if seeded_indices is not None:
        if (not isinstance(seeded_indices, (list, tuple)) or not seeded_indices
                or any(isinstance(index, bool)
                       or not isinstance(index, (int, np.integer))
                       or index < 0 for index in seeded_indices)):
            raise ValueError("warmup_seeded_stage_indices must be a nonempty list of non-negative integers or None.")
    if not isinstance(recipe["partial_coherence"], bool):
        raise ValueError("partial_coherence must be bool.")
    if recipe["coherence_kernel_scope"] not in {"per_observation", "shared"}:
        raise ValueError("coherence_kernel_scope must be 'per_observation' or 'shared'.")
    _build_update_schedule(recipe, name="inner")
    _build_update_schedule(recipe, name="warmup", allow_disabled=True)
    _build_role_warmup_schedule(recipe, "reference")
    _build_role_warmup_schedule(recipe, "other")
    for key in (
        "outer_iterations",
        "average_img",
    ):
        value = recipe[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{key} must be an integer.")
    if recipe["outer_iterations"] <= 0:
        raise ValueError("outer_iterations must be > 0.")
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
    if recipe["projection_model"] not in {
        "physical_factorized",
        "state_energy_beam",
        "none",
    }:
        raise ValueError(
            "projection_model must be 'physical_factorized', "
            "'state_energy_beam', or 'none'."
        )
    if recipe["rank_deficient"] not in {"error", "minimum_norm"}:
        raise ValueError(
            "rank_deficient must be 'error' or 'minimum_norm'."
        )
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
        "known_beta",
        "known-beta",
        "known_beta_kk",
        "known-beta-kk",
    }
    for kind in ("charge", "magnetic"):
        mode = str(recipe[f"{kind}_spectral_constraint"]).lower()
        if mode not in allowed_spectral:
            raise ValueError(
                f"{kind}_spectral_constraint has an unsupported value."
            )
        if n_energies is not None and mode not in {
            "none",
            "free",
            "unconstrained",
        }:
            constrain_complex_spectrum(
                np.zeros(n_energies, dtype=np.complex128),
                spectral_constraint=recipe[f"{kind}_spectral_constraint"],
                energy_values=recipe["energy_values"],
                known_beta_spectrum=recipe[f"known_{kind}_beta_spectrum"],
                known_delta_spectrum=recipe[f"known_{kind}_delta_spectrum"],
                absorption_part=recipe[f"{kind}_absorption_part"],
                kk_sign=recipe["kk_sign"],
                kk_subtract_baseline=recipe["kk_subtract_baseline"],
                kk_normalize_input=recipe["kk_normalize_input"],
                known_beta_normalization=recipe[
                    "known_spectrum_normalization"
                ],
                fit_known_beta_scale=recipe["fit_known_spectrum_scale"],
                fit_known_beta_offset=recipe["fit_known_spectrum_offset"],
            )
    _observation_weights(recipe["observation_weights"], n_observations)


def _normalized_coherence_kernel(gamma):
    """
    Return a real, nonnegative kernel with unit mass per spatial plane.

    Context:
    Called after shared-kernel fitting/averaging. Unit mass is enforced separately for each
    modal spatial plane; otherwise the blur kernel could absorb an arbitrary intensity
    scale.
    """
    kernel = np.maximum(np.asarray(gamma).real, 0).copy()
    mass = kernel.sum(axis=(-2, -1), keepdims=True)
    if np.any(~np.isfinite(kernel)) or np.any(mass <= 0):
        raise ValueError("Coherence kernels must have finite positive mass.")
    return kernel / mass


def _initial_coherence_kernel(field):
    """
    Create a centered, nearly delta-like initial blur on the same spatial/modal grid as one
    observation. This is an initial guess, not a fitted detector point-spread function.
    """
    kernel = np.full(np.shape(field), 2e-6, dtype=float)
    kernel[..., kernel.shape[-2] // 2, kernel.shape[-1] // 2] = 0.7
    return _normalized_coherence_kernel(kernel)


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
    kernel = _normalized_coherence_kernel(gamma)
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
    if (recipe["Nmodes"] == 2 and phase_retrieval_kernel is None
            and recipe.get("observation_workers", 1) == 1
            and not has_rl and not recipe.get("partial_coherence", False)):
        try:
            from . import phase_retrieval_core_multienergy_multimode as modal
        except ImportError:
            import phase_retrieval_core_multienergy_multimode as modal
        result = modal._run_energy_update_schedule(
            field, amplitude, supportmask, bsmask, schedule, recipe,
            nmodes=2, image_shape=amplitude.shape,
        )
        return (*result, gamma) if return_gamma else result
    if recipe["Nmodes"] == 2:
        if phase_retrieval_kernel is None:
            try:
                from . import phase_retrieval_core_unified as unified
            except ImportError:
                import phase_retrieval_core_unified as unified
            phase_retrieval_kernel = unified.PhaseRtrv_core
    if phase_retrieval_kernel is None:
        phase_retrieval_kernel = PhaseRtrv_core
    stage_results = []
    filled_intensity = None
    for stage_index, stage in enumerate(schedule):
        stage = dict(stage)
        iterations = stage["Nit"]
        if freeze_gamma and stage["RL_it"] > 0 and stage["RL_freq"] <= iterations:
            if gamma is None:
                raise ValueError("A fitted reference gamma is required before freezing coherence.")
            # Keep convolution active, but no RL update occurs in range(Nit).
            stage["RL_freq"] = iterations
        use_rl = stage["RL_it"] > 0 and stage["RL_freq"] <= iterations
        if use_rl:
            if gamma is None:
                gamma_shape = ((recipe["Nmodes"], *amplitude.shape)
                               if recipe["Nmodes"] == 2 else amplitude.shape)
                gamma = np.full(gamma_shape, 2e-6, dtype=float)
                gamma[..., amplitude.shape[0] // 2, amplitude.shape[1] // 2] = 0.7
            if filled_intensity is None:
                current = (np.sum(np.abs(field) ** 2, axis=0)
                           if recipe["Nmodes"] == 2 else np.abs(field) ** 2)
                masked_values = (current if fixed_masked_intensity is None
                                 else np.asarray(fixed_masked_intensity))
                filled_intensity = np.where(bsmask != 0, masked_values, amplitude ** 2)
            stage_amplitude = np.sqrt(np.maximum(filled_intensity, 0))
            stage_bsmask = np.zeros_like(bsmask)
        else:
            stage_amplitude = amplitude
            stage_bsmask = bsmask
            filled_intensity = None
        if stage["mode"] == "gradient_descent":
            if use_rl:
                raise ValueError("gradient_descent does not support RL updates")
            refined = gradient.refine_field_gradient(
                field,
                amplitude,
                supportmask=supportmask,
                mask_pixel=bsmask,
                n_steps=iterations,
                learning_rate=make_beta_schedule(
                    stage["beta_mode"],
                    iterations,
                    stage["beta_zero"],
                ),
                support_weight=make_alpha_schedule(
                    stage["alpha_mode"],
                    iterations,
                    stage["alpha_zero"],
                ),
                loss_mode="amplitude",
                support_projection=False,
                fourier_projection=False,
            )
            field = refined.fields
            if recipe["Fourier_last"]:
                observed = bsmask == 0
                field = np.asarray(field).copy()
                field[observed] = (
                    amplitude[observed] * np.exp(1j * np.angle(field[observed]))
                )
            error = refined.diffraction_loss
            support_error = refined.support_loss
        else:
            kernel_args = {"Nmodes": recipe["Nmodes"]} if recipe["Nmodes"] == 2 else {}
            field, error, support_error, gamma_out = phase_retrieval_kernel(
                diffract=stage_amplitude,
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
                bsmask=stage_bsmask,
                real_object=False,
                average_img=min(max(1, recipe["average_img"]), iterations),
                Fourier_last=recipe["Fourier_last"],
                gamma=gamma if use_rl else None,
                RL_freq=stage["RL_freq"],
                RL_it=stage["RL_it"],
                TV_freq=stage["TV_freq"],
                **kernel_args,
            )
            if gamma_out is not None and not freeze_gamma:
                gamma = gamma_out
        stage_results.append({
            "schedule_stage": stage_index,
            **stage,
            "error": np.asarray(error),
            "support_error": np.asarray(support_error),
            "coherence": "partial" if use_rl else "full",
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
    try:
        from . import phase_retrieval_core_multienergy_multimode as modal
    except ImportError:
        import phase_retrieval_core_multienergy_multimode as modal
    return modal._initialize_modal_fields(
        supportmask, amplitudes, intensities, mask_stack, 2,
        recipe["mode_initialization_seed"],
    )


def _project_nonphysical_modes_common(
    fields,
    energy_labels,
    beam_labels,
    recipe,
    weights=None,
):
    """
    Remove state dependence from modes outside the physical model.

    Context:
    Secondary-mode projection at joint boundaries and final output. Group by energy and
    illumination, ignoring state and polarization; align arbitrary modal global phases
    before complex averaging to avoid cancellation.
    """
    array = np.asarray(fields)
    if array.ndim != 4 or array.shape[1] < 2:
        return array.copy(), []
    n_observations = array.shape[0]
    energies = np.asarray(energy_labels)
    beams = np.asarray(beam_labels)
    if energies.shape != (n_observations,) or beams.shape != (n_observations,):
        raise ValueError("Energy and beam labels must match the observation count.")
    if weights is None:
        observation_weights = np.ones(n_observations, dtype=float)
    else:
        observation_weights = np.asarray(weights, dtype=float)
        if observation_weights.shape != (n_observations,):
            raise ValueError("observation_weights must match the observation count.")

    relaxation = float(recipe["nonphysical_modes_common_relaxation"])
    output = array.copy()
    group_records = []
    if relaxation == 0:
        return output, group_records
    groups = {}
    for observation, key in enumerate(zip(energies.tolist(), beams.tolist())):
        groups.setdefault(key, []).append(observation)
    for mode_index in range(1, array.shape[1]):
        for key, indices in groups.items():
            group_weights = observation_weights[indices]
            total = float(np.sum(group_weights))
            if total <= 0:
                raise ValueError("Each common-mode energy/beam group needs positive weight.")
            # Incoherent modes have arbitrary independent global phases. Align
            # that gauge before taking a linear complex-field average. Averaging
            # wrapped log phases creates artificial phase edges and diffraction.
            fields_group = array[indices, mode_index].copy()
            anchor = fields_group[0]
            if not np.any(anchor):
                anchor = fields_group[np.argmax(np.sum(np.abs(fields_group) ** 2, axis=(-2, -1)))]
            for index in range(len(fields_group)):
                overlap = np.vdot(anchor, fields_group[index])
                if abs(overlap) > 0:
                    fields_group[index] *= np.exp(-1j * np.angle(overlap))
            common = np.sum(fields_group * group_weights[:, None, None], axis=0) / total
            output[indices, mode_index] = (1 - relaxation) * fields_group + relaxation * common
            # A common focus transform within each energy/beam group is linear
            # and unitary, so averaging here is equivalent to averaging in focus.
            group_records.append({
                "mode_index": mode_index,
                "support_factor": mode_index + 1,
                "energy": key[0], "beam": key[1],
                "observations": list(indices),
                "averaging": "phase_aligned_complex_field",
            })
    return output, group_records


def _project_physical_modes(
    fields,
    state_labels,
    energy_labels,
    polarization_coefficients,
    beam_labels,
    **kwargs,
):
    """
    Apply the physical model to mode 1 and optional common-mode constraints.

    Context:
    Dispatch the primary mode to the physical model, then enforce the optional shared
    secondary mode. The secondary field never enters the charge/magnetization fit.
    """
    diagnostic_callback = kwargs.pop("diagnostic_callback", None)
    if np.asarray(fields).ndim == 3:
        result = project_fourier_fields_general(
            fields, state_labels, energy_labels, polarization_coefficients,
            beam_labels, **kwargs,
        )
        if diagnostic_callback is not None:
            diagnostic_callback("physical_primary", result[0] if kwargs.get("return_components") else result)
        return result
    projected_primary, components = project_fourier_fields_general(
        fields[:, 0], state_labels, energy_labels, polarization_coefficients,
        beam_labels, **kwargs,
    )
    output = np.asarray(fields).copy()
    output[:, 0] = projected_primary
    if diagnostic_callback is not None:
        diagnostic_callback("physical_primary", output)
    recipe = kwargs.get("recipe")
    common_groups = []
    if recipe is not None and recipe["constrain_nonphysical_modes_common"]:
        output, common_groups = _project_nonphysical_modes_common(
            output,
            energy_labels,
            beam_labels,
            recipe,
            weights=kwargs.get("weights"),
        )
    if diagnostic_callback is not None:
        diagnostic_callback("common_secondary", output)
    components["Nmodes"] = output.shape[1]
    components["physical_model_modes"] = [1]
    components["nonphysical_modes_common"] = bool(
        recipe is not None and recipe["constrain_nonphysical_modes_common"]
    )
    components["nonphysical_modes_common_relaxation"] = (
        float(recipe["nonphysical_modes_common_relaxation"])
        if recipe is not None else 0.0
    )
    components["nonphysical_mode_common_groups"] = common_groups
    return output, components


def _component_centered_modal_supports(supportmask, factors):
    """Expand each disconnected aperture in place for off-center samples."""
    from scipy.ndimage import affine_transform, center_of_mass, label

    support = np.asarray(supportmask) != 0
    labels, count = label(support, structure=np.ones((3, 3), dtype=int))
    centers = center_of_mass(support, labels, range(1, count + 1))
    result = []
    for factor in factors:
        if float(factor) == 1.0:
            result.append(support.astype(np.uint8))
            continue
        expanded = np.zeros(support.shape, dtype=bool)
        matrix = np.eye(2) / float(factor)
        for component, center in enumerate(centers, start=1):
            center = np.asarray(center, dtype=float)
            offset = center - matrix @ center
            expanded |= affine_transform(
                (labels == component).astype(np.uint8), matrix,
                offset=offset, output_shape=support.shape, order=0,
                mode="constant", cval=0, prefilter=False,
            ) != 0
        result.append(expanded.astype(np.uint8))
    return np.stack(result)


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
    Main orchestration for mixed metadata. Owns warmup, coherent masked-fill snapshots, per-
    observation/shared gamma, coherent refreshes, worker synchronization and physical-
    projection cadence. The low-level kernel only sees one observation at a time.
    """
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

    mode_factors = recipe["modes"]
    if mode_factors is not None:
        if not isinstance(mode_factors, (list, tuple)) or len(mode_factors) not in (1, 2):
            raise ValueError("modes must be [1] or a two-entry support-factor list.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float, np.number))
               or not np.isfinite(value) or value <= 0 for value in mode_factors):
            raise ValueError("modes entries must be finite positive support factors.")
        if float(mode_factors[0]) != 1.0:
            raise ValueError("The first physical mode must use support factor 1.")
        recipe["Nmodes"] = len(mode_factors)
    else:
        mode_factors = [1] * int(recipe["Nmodes"])

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
    if recipe["Nmodes"] == 2:
        if recipe["recenter_modal_supports"]:
            modal_supportmask, mode_support_shifts = geometry.recenter_modal_supports(
                supportmask, mode_factors, center=recipe["mode_support_center"],
            )
        elif recipe["mode_support_center"] == "components":
            modal_supportmask = _component_centered_modal_supports(
                supportmask, mode_factors,
            )
        else:
            try:
                from . import phase_retrieval_core_unified as unified
            except ImportError:
                import phase_retrieval_core_unified as unified
            modal_supportmask = unified._mode_supports(
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
        if recipe["Nmodes"] == 2:
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
        expected = ((n_observations, 2, nx, ny) if recipe["Nmodes"] == 2
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

    preserve_masked = recipe["preserve_warmup_masked_intensity"]
    fixed_masked_intensities = [None] * n_observations
    if recipe["freeze_coherence_after_reference"] and not share_coherence:
        raise ValueError("Freezing reference coherence requires coherence_kernel_scope='shared'.")

    # Optional independent warmup before coupling observations.
    warmup_schedule = _build_update_schedule(
        recipe,
        name="warmup",
        allow_disabled=True,
    )
    reference_schedule = _build_role_warmup_schedule(recipe, "reference")
    other_schedule = _build_role_warmup_schedule(recipe, "other")
    if reference_schedule is not None or other_schedule is not None:
        # An omitted role inherits the shared schedule. This also permits an
        # explicit empty schedule for either role by setting its Nit to zero.
        reference_schedule = (warmup_schedule if reference_schedule is None
                              else reference_schedule)
        other_schedule = (warmup_schedule if other_schedule is None
                          else other_schedule)
    else:
        reference_schedule = warmup_schedule
        other_schedule = None
    if reference_schedule or other_schedule or warmup_schedule:
        seeded_indices = recipe["warmup_seeded_stage_indices"]
        if (other_schedule is None and seeded_indices is not None
                and max(seeded_indices) >= len(warmup_schedule)):
            raise ValueError("warmup_seeded_stage_indices exceeds warmup stage count.")
        reference_observation = int(recipe["warmup_reference_observation"])
        if reference_observation >= n_observations:
            raise ValueError("warmup_reference_observation exceeds observation count.")
        warmup_order = [reference_observation] + [
            index for index in range(n_observations) if index != reference_observation
        ]
        print(
            "Universal phase retrieval: starting independent warmup "
            f"for {n_observations} observations",
            flush=True,
        )
        warmup_passes = ("coherent", "partial") if preserve_masked else ("original",)
        for pass_kind in warmup_passes:
            pass_started = time.perf_counter()
            synchronous = round_coherence and pass_kind == "partial"
            round_gamma = shared_coherence_kernel
            if synchronous and round_gamma is None:
                round_gamma = _initial_coherence_kernel(fields[reference_observation])

            calibrate_reference = synchronous and recipe["warmup_calibrate_shared_gamma"]

            def update_warmup(observation):
                field = fields[observation]
                if observation == reference_observation:
                    observation_schedule = reference_schedule
                elif other_schedule is not None:
                    observation_schedule = other_schedule
                elif seeded_indices is not None:
                    observation_schedule = [warmup_schedule[index] for index in seeded_indices]
                else:
                    observation_schedule = warmup_schedule
                if preserve_masked:
                    partial_flags = [stage["RL_it"] > 0 and stage["RL_freq"] <= stage["Nit"]
                                     for stage in observation_schedule]
                    if pass_kind == "coherent" and any(partial_flags):
                        first_partial = partial_flags.index(True)
                        if not all(partial_flags[first_partial:]):
                            raise ValueError("Preserved warmup requires coherent stages before partial stages.")
                    observation_schedule = [stage for stage, partial in zip(observation_schedule, partial_flags)
                                            if partial == (pass_kind == "partial")]
                    if pass_kind == "coherent" and not observation_schedule:
                        raise ValueError("Every state needs a coherent warmup to preserve masked intensities.")
                if not observation_schedule:
                    return None
                if (pass_kind != "partial" and observation != reference_observation
                        and recipe["warmup_start_from_first"]):
                    if recipe["warmup_seed_scale_fit"] == "linear":
                        valid = ((bsmasks[reference_observation] == 0)
                                 & (bsmasks[observation] == 0))
                        source = intensities[reference_observation][valid]
                        target = intensities[observation][valid]
                        if source.size < 2:
                            raise ValueError("Cannot scale warmup field: fewer than two shared valid pixels.")
                        source_centered = source - np.mean(source)
                        denominator = np.dot(source_centered, source_centered)
                        factor = (np.dot(source_centered, target - np.mean(target))
                                  / denominator if denominator > 0 else 0)
                        if not np.isfinite(factor) or factor <= 0:
                            raise ValueError("Cannot scale warmup field: non-positive intensity slope.")
                        scale = np.sqrt(factor)
                    else:
                        reference_sum = float(np.sum(intensities[reference_observation]))
                        current_sum = float(np.sum(intensities[observation]))
                        scale = (np.sqrt(current_sum / reference_sum)
                                 if reference_sum > 0 and current_sum > 0 else 1.0)
                    field = fields[reference_observation] * scale
                    if recipe["startimage_radial_normalization"]:
                        field = _radially_normalize_startimage(
                            field, holograms[observation], mask_stack[observation],
                        )
                gamma = (round_gamma if synchronous else shared_coherence_kernel
                         if share_coherence else coherence_kernels[observation])
                return _run_observation_update(
                    field, amplitudes[observation], modal_supportmask,
                    bsmasks[observation], observation_schedule, recipe,
                    phase_retrieval_kernel=phase_retrieval_kernel,
                    gamma=None if gamma is None else gamma.copy(), return_gamma=True,
                    fixed_masked_intensity=fixed_masked_intensities[observation],
                    freeze_gamma=((synchronous and gamma_strategy == "pooled"
                                   and not (calibrate_reference and observation == reference_observation)) or
                                  (recipe["freeze_coherence_after_reference"]
                                   and observation != reference_observation)),
                )

            # Seeded coherent states depend only on the completed reference.
            # Sequential shared-gamma warmup retains its legacy ordering.
            parallel_pass = (pass_kind == "coherent" or synchronous
                             or not recipe["partial_coherence"] or not share_coherence)
            pass_workers = workers if parallel_pass else 1
            batches = ([warmup_order[:1], warmup_order[1:]]
                       if (recipe["warmup_start_from_first"] and pass_kind != "partial")
                       or calibrate_reference else [warmup_order])
            gamma_sum = None
            gamma_weight = 0.0
            for batch in batches:
                for observation, result in _observation_map(update_warmup, batch, pass_workers):
                    if result is None:
                        continue
                    print(f"  Warmup observation {observation + 1}/{n_observations} ({pass_kind})", flush=True)
                    fields[observation], results, updated_gamma = result
                    if preserve_masked and pass_kind == "coherent":
                        fixed_masked_intensities[observation] = (
                            np.sum(np.abs(fields[observation]) ** 2, axis=0)
                            if recipe["Nmodes"] == 2 else np.abs(fields[observation]) ** 2
                        )
                    if synchronous:
                        if calibrate_reference and observation == reference_observation:
                            round_gamma = _normalized_coherence_kernel(updated_gamma)
                        if updated_gamma is not None:
                            contribution = state_weights[observation] * _normalized_coherence_kernel(updated_gamma)
                            gamma_sum = contribution if gamma_sum is None else gamma_sum + contribution
                            gamma_weight += state_weights[observation]
                    elif share_coherence:
                        shared_coherence_kernel = updated_gamma
                    else:
                        coherence_kernels[observation] = updated_gamma
                    for result in results:
                        errors["observation_steps"].append({
                            "outer": -1, "observation": observation,
                            "state": metadata["states"][observation],
                            "energy": metadata["energies"][observation],
                            "polarization": metadata["polarizations"][observation],
                            "beam": metadata["beams"][observation],
                            "stage": "warmup", **result,
                        })
            if synchronous:
                if gamma_strategy == "average":
                    if gamma_weight <= 0:
                        raise ValueError("Average gamma requires partial-coherence warmup stages.")
                    shared_coherence_kernel = _normalized_coherence_kernel(gamma_sum / gamma_weight)
                else:
                    targets = np.where(bsmasks != 0, np.asarray(fixed_masked_intensities), amplitudes ** 2)
                    shared_coherence_kernel = _pooled_coherence_update(
                        fields, targets, round_gamma, recipe["coherence_round_iterations"], state_weights,
                    )
            record_checkpoint("warmup_" + pass_kind)
            elapsed = time.perf_counter() - pass_started
            errors.setdefault("warmup_passes", []).append({"pass": pass_kind, "wall_seconds": elapsed})
            print(f"  Warmup {pass_kind} pass complete in {elapsed:.2f} seconds", flush=True)
        print("Universal phase retrieval: warmup complete", flush=True)
    else:
        print("Universal phase retrieval: warmup disabled", flush=True)

    if preserve_masked and any(value is None for value in fixed_masked_intensities):
        raise ValueError("Every state needs a coherent warmup to preserve masked intensities.")
    fieldswarmup=fields.copy()
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
                round_gamma = _initial_coherence_kernel(fields[0])

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
                    contribution = state_weights[observation] * _normalized_coherence_kernel(gamma)
                    gamma_sum = contribution if gamma_sum is None else gamma_sum + contribution
                elif not share_coherence:
                    coherence_kernels[observation] = gamma
            if round_coherence:
                if gamma_strategy == "average":
                    shared_coherence_kernel = _normalized_coherence_kernel(gamma_sum)
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
                return_components=True,
                diagnostic_callback=record_checkpoint,
            )
            restore_frozen_observations(fields)
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
            return_components=True,
        )
        fields = final_projected_fields
        restore_frozen_observations(fields)
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
        fields, common_groups = _project_nonphysical_modes_common(
            fields,
            metadata["energies"],
            metadata["beams"],
            recipe,
            weights=recipe["observation_weights"],
        )
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
    return input_geometry.finish(fields, fieldswarmup, components, bsmasks, errors)


_PHYSICAL_MODELS = {
    "physical_factorized",
    "state_energy_beam",
    "none",
}
_EV_TO_WAVENUMBER_PER_METRE = 5.067730716156395e6


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
        "known_beta_spectrum": None,
        "known_delta_spectrum": None,
        "absorption_part": "real",
        "known_beta_normalization": "none",
        "fit_known_beta_scale": True,
        "fit_known_beta_offset": True,
        # Physical bounds are stated in identifiable dimensionless products.
        "charge_kt_delta_range": None,
        "charge_kt_beta_range": None,
        "magnetic_kt_delta_range": None,
        "magnetic_kt_beta_range": None,
        # Optional refractive-index spectra are converted to q(E)=-ikt*n(E).
        "wave_numbers": None,
        "thickness": None,
        "known_charge_kt_beta_spectrum": None,
        "known_charge_kt_delta_spectrum": None,
        "known_magnetic_kt_beta_spectrum": None,
        "known_magnetic_kt_delta_spectrum": None,
    })
    return recipe


def _canonical_projection_model(model):
    """Return the canonical spelling of a supported projection model."""
    model = str(model).lower()
    aliases = {
        "low_rank": "svd",
        "low-rank": "svd",
        "spectral": "rank1_spectral",
        "explicit": "rank1_spectral",
        "cma": "rank1_spectral",
        "c+m*a": "rank1_spectral",
    }
    return aliases.get(model, model)


def _apply_kt_product_bounds(recipe):
    """
    Translate ``kt*delta`` and ``kt*beta`` bounds into complex-response bounds.

    For ``q = -i k t (delta + i beta)``,

        Re(q) = k t beta
        Im(q) = -k t delta.
    """
    translated = recipe.copy()
    mappings = {
        "charge_kt_beta_range": "charge_response_real_range",
        "charge_kt_delta_range": "charge_response_imag_range",
        "magnetic_kt_beta_range": "magnetic_response_real_range",
        "magnetic_kt_delta_range": "magnetic_response_imag_range",
    }
    for source, target in mappings.items():
        value = _normalize_range(recipe[source], source)
        if value is None:
            continue
        if recipe[target] is not None:
            raise ValueError(
                f"Specify either {source} or {target}, not both."
            )
        if "delta" in source:
            lower, upper = value
            value = (-upper, -lower)
        translated[target] = value
    return translated


def _kt_factors(recipe, n_energies):
    """
    Return ``k(E)*t`` for converting supplied refractive-index spectra.

    ``wave_numbers`` are interpreted in inverse metres and ``thickness`` in
    metres. If wave numbers are omitted, ``energy_values`` in eV are converted
    using ``k = E/(hbar*c)``.
    """
    thickness = recipe["thickness"]
    if thickness is None:
        raise ValueError(
            "thickness is required when known delta/beta refractive-index "
            "spectra are supplied."
        )
    if (
        isinstance(thickness, bool)
        or not isinstance(thickness, (int, float, np.number))
        or not np.isfinite(thickness)
        or thickness <= 0
    ):
        raise ValueError("thickness must be a positive finite value in metres.")

    if recipe["wave_numbers"] is not None:
        wave_numbers = np.asarray(recipe["wave_numbers"], dtype=float)
    else:
        energies = np.asarray(recipe["energy_values"], dtype=float)
        if energies.shape != (n_energies,):
            raise ValueError(
                "energy_values must have shape (n_energies,) when converting "
                "known refractive-index spectra."
            )
        wave_numbers = energies * _EV_TO_WAVENUMBER_PER_METRE

    if (
        wave_numbers.shape != (n_energies,)
        or np.any(~np.isfinite(wave_numbers))
        or np.any(wave_numbers <= 0)
    ):
        raise ValueError(
            "wave_numbers must be positive and have shape (n_energies,)."
        )
    return wave_numbers * float(thickness)


def _known_product_spectrum(recipe, kind, part, n_energies):
    """
    Resolve a known ``kt`` product, optionally converting delta(E) or beta(E).

    The explicit ``known_*_kt_*_spectrum`` form needs no thickness. The
    refractive-index form ``known_*_*_spectrum`` is multiplied by ``k(E)t``.
    """
    product_key = f"known_{kind}_kt_{part}_spectrum"
    index_key = f"known_{kind}_{part}_spectrum"
    product = recipe[product_key]
    refractive_index = recipe[index_key]
    if product is not None and refractive_index is not None:
        raise ValueError(
            f"Specify either {product_key} or {index_key}, not both."
        )
    if product is not None:
        values = np.asarray(product, dtype=float)
    elif refractive_index is not None:
        values = (
            np.asarray(refractive_index, dtype=float)
            * _kt_factors(recipe, n_energies)
        )
    else:
        return None
    if values.shape != (n_energies,) or np.any(~np.isfinite(values)):
        raise ValueError(
            f"{product_key} or {index_key} must be finite with shape "
            "(n_energies,)."
        )
    return values


def _prepare_physical_recipe(recipe, n_energies):
    """Translate universal bounds and spectra for the local physical model."""
    translated = _apply_kt_product_bounds(recipe)
    for kind in ("charge", "magnetic"):
        beta = _known_product_spectrum(
            translated,
            kind,
            "beta",
            n_energies,
        )
        delta = _known_product_spectrum(
            translated,
            kind,
            "delta",
            n_energies,
        )
        if beta is not None:
            translated[f"known_{kind}_beta_spectrum"] = beta
        if delta is not None:
            # q=-ikt*n makes the dispersion-like response equal to -kt*delta.
            translated[f"known_{kind}_delta_spectrum"] = -delta
        translated[f"{kind}_absorption_part"] = "real"
    return translated


def _pure_energy_scan(metadata):
    """Return True when only energy changes and every energy occurs once."""
    return (
        len(metadata["state_names"]) == 1
        and len(metadata["beam_names"]) == 1
        and np.all(
            metadata["polarizations"] == metadata["polarizations"][0]
        )
        and len(metadata["energy_names"]) == len(metadata["energies"])
    )


def _physical_driver_recipe(recipe):
    """
    Select recipe entries understood by the local physical driver.

    Context:
    Adapter at universal dispatch: retain the keys accepted by the general physical driver
    so universal-only spectrum aliases do not leak into lower-level validation.
    """
    defaults = default_general_phase_retrieval_recipe()
    return {key: recipe[key] for key in defaults}


def _energy_driver_recipe(recipe):
    """
    Translate universal settings for the local pure-energy driver.

    Context:
    Compatibility adapter for pure-energy low-rank retrieval. This dispatch is distinct from
    the mixed-state physical driver and does not support every newer joint-workflow option.
    """
    defaults = default_multi_energy_phase_retrieval_recipe()
    translated = {
        key: recipe[key]
        for key in defaults
        if key in recipe
    }
    translated["shuffle_energies"] = recipe["shuffle_observations"]
    translated["energy_weights"] = recipe["observation_weights"]
    translated["projection_model"] = _canonical_projection_model(
        recipe["projection_model"]
    )
    return translated


def project_fourier_fields_universal(
    fields,
    state_labels,
    energy_labels,
    polarization_coefficients,
    illumination_labels,
    universal_recipe=None,
    saturated_states=None,
    return_components=False,
    supportmask=None,
):
    """
    Apply one universal object-space projection to Fourier-domain fields.

    Physical and linear metadata-aware models accept arbitrary observation
    lists. SVD and rank-one spectral models require a pure energy scan because
    those projectors interpret the complete first axis as the energy axis.
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

    fields = _as_energy_stack(fields, name="fields")
    metadata = _normalize_metadata(
        state_labels,
        energy_labels,
        polarization_coefficients,
        illumination_labels,
        fields.shape[0],
    )
    model = _canonical_projection_model(recipe["projection_model"])

    material_thickness = _material_thickness_on_grid(recipe, fields.shape[-2:])
    if material_thickness is not None:
        material_thickness = np.fft.fftshift(material_thickness)
    thickness_supportmask = None
    if recipe["zero_thickness_outside_support"]:
        if model != "physical_factorized":
            raise ValueError(
                "zero_thickness_outside_support requires projection_model='physical_factorized'."
            )
        if supportmask is None:
            raise ValueError(
                "supportmask is required when zero_thickness_outside_support=True "
                "in a direct projection."
            )
        support = np.asarray(supportmask)
        if support.shape != fields.shape[-2:]:
            raise ValueError("supportmask must match the Fourier-field spatial shape.")
        if material_thickness is None:
            material_thickness = np.ones(fields.shape[-2:], dtype=float)
        thickness_supportmask = np.fft.fftshift(support) != 0
        material_thickness = material_thickness * thickness_supportmask
        if not np.any(material_thickness > 0):
            raise ValueError("The support contains no material pixels for thickness fitting.")

    if model == "none":
        unchanged = np.asarray(fields).copy()
        if return_components:
            return unchanged, {"projection_model": "none"}
        return unchanged

    if model in _PHYSICAL_MODELS:
        physical_recipe = _prepare_physical_recipe(
            recipe,
            len(metadata["energy_names"]),
        )
        return project_fourier_fields_general(
            fields,
            metadata["states"],
            metadata["energies"],
            metadata["polarizations"],
            metadata["beams"],
            weights=physical_recipe["observation_weights"],
            relaxation=physical_recipe["projection_relaxation"],
            rank_deficient=physical_recipe["rank_deficient"],
            projection_model=model,
            saturated_states=physical_recipe["saturated_states"],
            physical_iterations=physical_recipe["physical_iterations"],
            recipe=physical_recipe,
            log_floor=physical_recipe["log_floor"],
            material_thickness=material_thickness,
            fit_material_thickness=recipe["fit_material_thickness"],
            thickness_supportmask=thickness_supportmask,
            return_components=return_components,
        )

    if model in {"svd", "rank1_spectral"}:
        if recipe["coherent_refresh_rounds"]:
            raise ValueError("Coherent refresh requires the general driver (physical_factorized, state_energy_beam, or none).")
        if not _pure_energy_scan(metadata):
            raise ValueError(
                f"projection_model={model!r} requires a pure energy scan: "
                "one state, one illumination condition, constant polarization, "
                "and one observation per energy. Use 'physical_factorized' "
                "for mixed datasets."
            )
        result = project_fourier_fields_multi_energy(
            fields,
            projection_model=model,
            rank=recipe["rank"],
            static_mode=recipe["projection_static_mode"],
            weights=recipe["observation_weights"],
            relaxation=recipe["projection_relaxation"],
            log_floor=recipe["log_floor"],
            spectral_constraint=recipe["spectral_constraint"],
            energy_values=recipe["energy_values"],
            known_beta_spectrum=recipe["known_beta_spectrum"],
            known_delta_spectrum=recipe["known_delta_spectrum"],
            absorption_part=recipe["absorption_part"],
            kk_sign=recipe["kk_sign"],
            kk_subtract_baseline=recipe["kk_subtract_baseline"],
            kk_normalize_input=recipe["kk_normalize_input"],
            known_beta_normalization=recipe["known_beta_normalization"],
            fit_known_beta_scale=recipe["fit_known_beta_scale"],
            fit_known_beta_offset=recipe["fit_known_beta_offset"],
            material_thickness=material_thickness,
            fit_material_thickness=recipe["fit_material_thickness"],
            return_components=return_components,
        )
        return result

    raise ValueError(
        "projection_model must be 'physical_factorized', "
        "'state_energy_beam', 'none', 'svd', or 'rank1_spectral'."
    )


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
):
    """
    Reconstruct a metadata-described list of diffraction measurements.

    Use ``physical_factorized`` for mixed energy/polarization/state/beam data.
    Use ``svd`` or ``rank1_spectral`` for a pure energy scan; these choices
    use the self-contained pure-energy implementation. Use ``none`` for
    independent phase retrieval without a joint object model.

    Returns
    -------
    fields : ndarray
        Final Fourier-domain fields, one per input hologram.
    fieldswarmup : ndarray
        Fourier-domain fields after warmup and before joint projections.
    components : dict
        Components and identifiability diagnostics from the selected model.
    bsmasks : ndarray
        Observation-specific invalid-pixel masks.
    errors : dict
        Per-stage errors, settings, metadata, and projection diagnostics.

    Context:
    Public notebook entry point. Supply intensity images plus
    state/energy/polarization/illumination metadata. Returns final fields, warmup snapshot,
    fitted components, effective masks and diagnostics; it does not run the optional
    notebook Poisson polish.
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
    model = _canonical_projection_model(recipe["projection_model"])
    print(
        "Universal phase retrieval: dispatching "
        f"{holograms.shape[0]} observations with "
        f"projection_model={model!r}",
        flush=True,
    )

    # Pure energy modes intentionally use the original multi-energy driver.
    if model in {"svd", "rank1_spectral"}:
        if recipe["coherent_refresh_rounds"]:
            raise ValueError("Coherent refresh requires the general driver (physical_factorized, state_energy_beam, or none).")
        if not _pure_energy_scan(metadata):
            raise ValueError(
                f"projection_model={model!r} requires a pure energy scan. "
                "Use 'physical_factorized' for mixed metadata."
            )
        fields, fieldswarmup,components, bsmasks, errors = (
            multi_energy_phase_retrieval_algorithm(
                holograms,
                mask_pixel,
                supportmask,
                multi_energy_recipe=_energy_driver_recipe(recipe),
                start_fields=start_fields,
                phase_retrieval_kernel=phase_retrieval_kernel,
            )
        )
    elif model in _PHYSICAL_MODELS:
        physical_recipe = _prepare_physical_recipe(
            recipe,
            len(metadata["energy_names"]),
        )
        fields, fieldswarmup,components, bsmasks, errors = (
            general_phase_retrieval_algorithm(
                holograms,
                mask_pixel,
                supportmask,
                metadata["states"],
                metadata["energies"],
                metadata["polarizations"],
                metadata["beams"],
                saturated_states=physical_recipe["saturated_states"],
                general_recipe=_physical_driver_recipe(physical_recipe),
                start_fields=start_fields,
                phase_retrieval_kernel=phase_retrieval_kernel,
            )
        )
    else:
        raise ValueError(
            "projection_model must be 'physical_factorized', "
            "'state_energy_beam', 'none', 'svd', or 'rank1_spectral'."
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
    return fields, fieldswarmup,components, bsmasks, errors
