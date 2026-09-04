"""
Single-mode phase retrieval for a pair of positive/negative holograms.

The low-level :func:`PhaseRtrv_core` kernel alternates between the measured
Fourier-amplitude constraint and a selected real-space projection (ER, HAPRE,
RAAR, HIO, and related algorithms). The high-level
:func:`phase_retrieval_algorithm` executes a recipe across the two input
holograms and can optionally include total-variation and partial-coherence
Richardson-Lucy updates.

Arrays returned by the reconstruction functions use the Fourier-field
convention expected by :func:`PhaseRtrv_core`; they are not directly the
real-space sample exit wave.

Functions were taken and adapted from the code base associated with:

Riccardo Battistelli, Daniel Metternich, Michael Schneider, Lisa-Marie Kern, Kai Litzius, Josefin Fuchs, Christopher Klose, Kathinka Gerlinger, Kai Bagschik, Christian M. Günther, Dieter Engel, Claus Ropers, Stefan Eisebitt, Bastian Pfau, Felix Büttner, and Sergey Zayko, "Coherent x-ray magnetic imaging with 5 nm resolution," Optica 11, 234-237 (2024)

2020 - Original Code
@authors:   RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)

2026 - Refactoring
@authors: Christopher Klose (christopher.klose@mbi-berlin.de)

2026 - Further refactoring and documentation
@authors: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
"""

import logging
from contextlib import nullcontext
log = logging.getLogger(__name__)

from functools import partial
import os
import time
import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt

from scipy import stats

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

    # Use all available CPU workers for FFT operations.
    fft2 = partial(fft.fft2, workers=os.cpu_count())
    ifft2 = partial(fft.ifft2, workers=os.cpu_count())


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


def _apply_measured_amplitude(field, amplitude, bsmask):
    """Apply measured Fourier amplitudes outside invalid-pixel regions."""
    constrained = np.asarray(field).copy()
    observed = np.asarray(bsmask) == 0
    constrained[observed] = (
        amplitude[observed] * np.exp(1j * np.angle(constrained[observed]))
    )
    return constrained


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
    ``algorithm_list`` may also contain ``"gradient_descent"``. In that case
    ``number_iterations`` is the number of gradient steps, ``beta_zero`` and
    ``beta_mode`` define the learning-rate schedule, and ``alpha_zero`` plus
    ``alpha_mode`` define the support-loss weight schedule. Gradient stages use
    the same centered supportmask convention as normal ``PhaseRtrv_core`` stages.

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
        Latest full-coherence reconstruction for each helicity, if requested.
    retrieved_p_pc, retrieved_n_pc : np.ndarray or None
        Latest partial-coherence reconstruction for each helicity, if requested.
    bsmask_p, bsmask_n : np.ndarray
        Beamstop/floating-pixel masks used for full-coherence steps.
    gamma_p, gamma_n : np.ndarray
        Latest mutual-coherence estimates for each helicity. If no RL step was
        performed for a helicity, this is the initial gamma estimate.
    error : dict
        Step-wise error information stored under ``error["steps"]``.
        Every step dictionary contains the ``field_after`` Fourier field.
        ``error["latest"]`` stores the latest
        ``full_coherence``, ``partial_coherence``, and ``gradient_descent``
        outputs separately for each helicity. For ``gradient_descent`` stages,
        the step dictionary also contains the combined gradient ``loss``
        history. Recipe steps with ``output=True`` are collected in
        ``error["outputs"]`` and grouped in ``error["outputs_by_helicity"]``.
        If the recipe omits ``output``, only the last step for each helicity is
        selected.
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

    # Full-coherence beamstop masks inherit the external mask and mark negative
    # corrected intensities as unconstrained. Zero intensity remains constrained.
    bsmask_p = mask_pixel.copy()
    bsmask_n = mask_pixel.copy()
    bsmask_p[pos_input < 0] = 1
    bsmask_n[neg_input < 0] = 1


    # Clip negative corrected intensities to zero before taking square roots.
    pos_input = np.clip(pos_input, 0, None)
    neg_input = np.clip(neg_input, 0, None)

    pos_amp = np.sqrt(pos_input)
    neg_amp = np.sqrt(neg_input)

    # Initial Fourier-domain guess. The supportmask-based default follows the
    # convention of the original implementation.
    first_startimage = recipe["Startimage"][0]
    if first_startimage is None:
        Startimage = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(supportmask))
        )
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
            if retrieved[h] is not None:
                pc_input = (
                    np.abs(retrieved[h]) ** 2 * data[h]["bsmask"]
                    + data[h]["input"] * (1 - data[h]["bsmask"])
                )
            else:
                pc_input = data[h]["input"]

            diffract = np.sqrt(pc_input)
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
            extra_diagnostics = {
                "loss": refined.loss,
            }
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

        return (
            np.fft.ifftshift(guess),
            Error_diffr_list,
            Error_supp_list,
            np.fft.ifftshift(gamma),
        )

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
    # CuPy does not consistently expose numpy.errstate across versions.
    with np.errstate(divide="ignore", invalid="ignore") if xp is np else nullcontext():
        Error=10*xp.log10(Error)
    return to_numpy(Error, xp)
