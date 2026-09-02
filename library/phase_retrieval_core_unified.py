"""
Unified single-mode/multimode phase retrieval for arbitrary hologram labels.

For ``Nmodes > 1``, the measured intensity is modeled as

    I(q) = sum_m |Psi_m(q)|**2.

The Fourier constraint therefore rescales all modal fields together so that
their summed intensity matches the measured hologram. Real-space projections
are then applied independently to each mode. With ``Nmodes == 1`` the public
driver keeps the single-mode array path to avoid the extra modal dimension.

The high-level :func:`phase_retrieval_algorithm` accepts either the legacy
``pos, neg`` inputs or a dictionary such as ``{"pos": pos, "neg": neg,
"LH": LH, "LV": LV}``. The recipe key ``"helicity"`` is kept for backward
compatibility, but it now means "hologram key selected at this step".

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

        # Number of incoherent reconstruction modes. Nmodes=1 reproduces
        # the standard single-mode phase retrieval. Nmodes>1 uses a modal
        # intensity sum for the Fourier constraint.
        "Nmodes": 1,
        # Public mode labels. Their count selects Nmodes; [1] is single-mode.
        # Nmodes remains available for backward compatibility.
        "modes": None,
        # Remove this many pixels from every edge of returned spatial arrays.
        "crop": 0,
        "normalize_startimage_between_holograms": True,
        "return_format": "auto",

        "hologram_intensity_cutoff_vmin": -1,
        "Startimage": [None, "pos", "pos", "pos", "pos", "pos"],
        "Startgamma": [None,  None,  None,  None, "pos", "pos"],
    }
    return recipe


def _default_output_flags(labels):
    """Return True for the last recipe step of each requested hologram label."""
    flags = [False] * len(labels)
    for label in dict.fromkeys(labels):
        for index in range(len(labels) - 1, -1, -1):
            if labels[index] == label:
                flags[index] = True
                break
    return flags


def _broadcast_recipe_scalars(recipe):
    """Expand scalar per-step settings to the number of algorithm stages."""
    stage_count = len(recipe["algorithm_list"])
    scalar_keys = (
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
    )
    for key in scalar_keys:
        value = recipe[key]
        if isinstance(value, tuple):
            recipe[key] = list(value)
        elif not isinstance(value, list):
            recipe[key] = [value] * stage_count


def _resolve_start_field(start_spec, default_field, latest, name):
    """
    Resolve a recipe Startimage/Startgamma entry.

    start_spec can be:
      - None: use the default support-based initialization
      - np.ndarray: use this array directly
      - str: use latest[str]
    """

    if start_spec is None:
        return default_field.copy()

    if isinstance(start_spec, np.ndarray):
        return start_spec.copy()

    if isinstance(start_spec, str):
        if start_spec not in latest:
            raise ValueError(
                f"{name} requested unknown hologram label {start_spec!r}."
            )
        if latest[start_spec] is None:
            raise ValueError(
                f"{name} requested latest '{start_spec}', "
                f"but no previous {start_spec} result exists."
            )
        return latest[start_spec].copy()

    raise ValueError(
        f"Invalid {name} entry: {start_spec!r}. "
        "Allowed values are None, np.ndarray, or a hologram label string."
    )



def _as_modes(arr, Nmodes, shape_2d, name, dtype=None):
    """
    Convert a 2D or 3D array into modal shape ``(Nmodes, nx, ny)``.

    Accepted inputs are:
      - ``(nx, ny)``: copied into all modes
      - ``(1, nx, ny)``: copied into all modes
      - ``(Nmodes, nx, ny)``: used directly
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


def _maybe_squeeze_modes(arr, Nmodes):
    """Return 2D output for ``Nmodes == 1`` and 3D output otherwise."""
    if arr is None:
        return None
    if Nmodes == 1 and np.asarray(arr).ndim == 3:
        return arr[0]
    return arr


def _modal_intensity_numpy(arr):
    """Return the total intensity of a 2D single-mode or 3D multimode field."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return np.abs(arr) ** 2
    if arr.ndim == 3:
        return np.sum(np.abs(arr) ** 2, axis=0)
    raise ValueError("Expected a 2D or 3D reconstruction array.")


def _modal_amplitude_numpy(arr):
    """Return sqrt(total modal intensity) for a 2D or 3D complex field."""
    return np.sqrt(_modal_intensity_numpy(arr))


def _modal_convolved_intensities(current_guess, current_gamma, nmodes):
    """Return one coherent or partially coherent intensity per mode."""
    convolved = xp.zeros_like(current_guess, dtype=xp.complex128)
    for mode_index in range(nmodes):
        intensity = xp.abs(current_guess[mode_index]) ** 2
        if current_gamma is None:
            convolved[mode_index] = intensity
        else:
            convolved[mode_index] = ifft2(
                fft2(intensity) * fft2(current_gamma[mode_index])
            )
    return convolved


def _apply_modal_fourier_constraint(
    current_guess,
    current_convolved,
    measured_amplitude,
    observed,
    invalid,
):
    """Jointly rescale all modes to match the measured total intensity."""
    total_intensity = xp.sum(current_convolved, axis=0)
    modal_amplitude = xp.sqrt(total_intensity)
    modal_amplitude = xp.where(
        xp.abs(modal_amplitude) > 1e-30,
        modal_amplitude,
        1e-30,
    )
    factor = measured_amplitude / modal_amplitude
    return current_guess * (
        observed[None, :, :] * factor[None, :, :]
        + invalid[None, :, :]
    )


def _apply_modal_fourier_constraint_numpy(field_modes, measured_amplitude, bsmask):
    """NumPy version of the summed-intensity modal Fourier constraint."""
    observed = bsmask == 0
    invalid = ~observed
    total_amplitude = _modal_amplitude_numpy(field_modes)
    total_amplitude = np.where(total_amplitude > 1e-30, total_amplitude, 1e-30)
    factor = measured_amplitude / total_amplitude
    return field_modes * (
        observed[None, :, :] * factor[None, :, :]
        + invalid[None, :, :]
    )


def _apply_measured_amplitude(field, amplitude, bsmask):
    """Apply measured Fourier amplitudes outside invalid-pixel regions."""
    constrained = np.asarray(field).copy()
    observed = np.asarray(bsmask) == 0
    constrained[observed] = (
        amplitude[observed] * np.exp(1j * np.angle(constrained[observed]))
    )
    return constrained


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
    """Refine every coherent mode and preserve summed-amplitude convention."""
    phase_modes = _as_modes(
        phase,
        nmodes,
        image_shape,
        "Phase",
        dtype=np.complex128,
    )
    support_modes = _as_modes(
        supportmask,
        nmodes,
        image_shape,
        "supportmask",
        dtype=np.asarray(supportmask).dtype,
    )
    refined_modes = np.empty_like(phase_modes, dtype=np.complex128)
    diffraction_losses = []
    support_losses = []
    total_losses = []
    modal_target = measured_amplitude / np.sqrt(max(nmodes, 1))

    for mode_index in range(nmodes):
        result = gradient.refine_field_gradient(
            phase_modes[mode_index],
            modal_target,
            supportmask=support_modes[mode_index],
            mask_pixel=bsmask,
            n_steps=Nit,
            learning_rate=learning_rate,
            support_weight=support_weight,
            loss_mode="amplitude",
            support_projection=False,
            fourier_projection=False,
        )
        refined_modes[mode_index] = result.fields
        diffraction_losses.append(result.diffraction_loss)
        support_losses.append(result.support_loss)
        total_losses.append(result.loss)

    if Fourier_last:
        refined_modes = _apply_modal_fourier_constraint_numpy(
            refined_modes,
            measured_amplitude,
            bsmask,
        )

    return (
        _maybe_squeeze_modes(refined_modes, nmodes),
        np.mean(np.stack(diffraction_losses), axis=0),
        np.mean(np.stack(support_losses), axis=0),
        np.mean(np.stack(total_losses), axis=0),
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

    if not all(isinstance(h, str) for h in recipe["helicity"]):
        raise ValueError("All helicity entries must be hologram label strings.")

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

    if not isinstance(recipe["normalize_startimage_between_holograms"], bool):
        raise ValueError("normalize_startimage_between_holograms must be bool.")

    if recipe["return_format"] not in {"auto", "legacy", "dict"}:
        raise ValueError("return_format must be 'auto', 'legacy', or 'dict'.")

    for key in ["Startimage", "Startgamma"]:
        if key not in recipe:
            raise ValueError(f"Missing recipe key: {key}")

        if not isinstance(recipe[key], list):
            raise ValueError(f"Recipe key '{key}' must be a list.")

        if len(recipe[key]) != len(recipe["algorithm_list"]):
            raise ValueError(
                f"Recipe key '{key}' must have same length as algorithm_list."
            )
        
def _scale_phase_between_holograms(
    phase,
    source_label,
    target_label,
    intensity_data,
    bsmasks,
    use_offset=False,
    crop=0,
    plot=False,
    verbose=False,
):
    """
    Rescale a reconstruction when it is reused as the starting guess for a
    different hologram label.

    The scale factor is estimated from a linear fit of the measured hologram
    intensities, while excluding invalid pixels from both helicities using
    bsmask_p/bsmask_n.

    Parameters
    ----------
    phase : np.ndarray
        Previous reconstructed complex Fourier-domain field.
    source_label, target_label : str
        Label of the previous reconstruction and the target step.
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

    if phase is None or source_label is None or source_label == target_label:
        return phase

    if source_label not in intensity_data or target_label not in intensity_data:
        raise ValueError(
            f"Invalid hologram conversion: {source_label!r} -> {target_label!r}"
        )

    source_intensity = intensity_data[source_label].copy()
    target_intensity = intensity_data[target_label].copy()

    valid = (bsmasks[source_label] == 0) & (bsmasks[target_label] == 0)

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


def _normalize_phase_retrieval_inputs(
    holograms,
    neg=None,
    mask_pixel=None,
    supportmask=None,
    phase_retrieval_recipe=None,
):
    """Accept both legacy positional inputs and the new hologram dictionary."""
    legacy_call = not isinstance(holograms, dict)

    if legacy_call:
        if neg is None or mask_pixel is None or supportmask is None:
            raise TypeError(
                "Legacy calls require pos, neg, mask_pixel, supportmask."
            )
        return (
            {"pos": holograms, "neg": neg},
            mask_pixel,
            supportmask,
            phase_retrieval_recipe,
            True,
        )

    if not holograms:
        raise ValueError("holograms dictionary must not be empty.")

    if phase_retrieval_recipe is None and isinstance(supportmask, dict):
        phase_retrieval_recipe = supportmask
        supportmask = mask_pixel
        mask_pixel = neg
    elif supportmask is None:
        supportmask = mask_pixel
        mask_pixel = neg

    if mask_pixel is None or supportmask is None:
        raise TypeError(
            "Dictionary calls require holograms, mask_pixel, supportmask."
        )

    return holograms, mask_pixel, supportmask, phase_retrieval_recipe, False


def _legacy_tuple_from_result(result):
    """Return the historical ``pos/neg`` tuple from a structured result."""
    fc = result["full_coherence"]
    pc = result["partial_coherence"]
    bsmasks = result["bsmasks"]
    gamma = result["gamma"]
    return (
        fc.get("pos"),
        fc.get("neg"),
        pc.get("pos"),
        pc.get("neg"),
        bsmasks.get("pos"),
        bsmasks.get("neg"),
        gamma.get("pos"),
        gamma.get("neg"),
        result["error"],
    )


def phase_retrieval_algorithm(
    holograms: ArrayLike,
    neg: ArrayLike = None,
    mask_pixel: ArrayLike = None,
    supportmask: ArrayLike = None,
    phase_retrieval_recipe=None,
):
    """
    Run a recipe-driven phase retrieval for any number of hologram labels.

    Legacy use is still supported::

        phase_retrieval_algorithm(pos, neg, mask, support, recipe)

    New generalized use passes a dictionary::

        phase_retrieval_algorithm(
            {"pos": pos, "neg": neg, "LH": LH, "LV": LV},
            mask,
            support,
            recipe,
        )

    ``recipe["helicity"]`` lists the dictionary keys to reconstruct in order.
    ``recipe["Startimage"]`` and ``recipe["Startgamma"]`` may refer to any
    previous key. If ``normalize_startimage_between_holograms`` is true, a
    masked linear intensity scaling is applied when a start image is reused
    across different labels. Set it to false to copy the complex field exactly.
    """
    (
        holograms,
        mask_pixel,
        supportmask,
        phase_retrieval_recipe,
        legacy_call,
    ) = _normalize_phase_retrieval_inputs(
        holograms,
        neg=neg,
        mask_pixel=mask_pixel,
        supportmask=supportmask,
        phase_retrieval_recipe=phase_retrieval_recipe,
    )

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
    _broadcast_recipe_scalars(recipe)
    _verify_valid_phase_retrieval_recipe(recipe)

    labels = list(holograms.keys())
    requested = list(dict.fromkeys(recipe["helicity"]))
    missing = [label for label in requested if label not in holograms]
    if missing:
        raise ValueError(
            f"Recipe requests hologram label(s) absent from input: {missing}"
        )

    mode_labels = recipe["modes"]
    if mode_labels is not None:
        if not isinstance(mode_labels, (list, tuple)) or not mode_labels:
            raise ValueError("recipe['modes'] must be a non-empty list or tuple.")
        if len(set(mode_labels)) != len(mode_labels):
            raise ValueError("recipe['modes'] entries must be unique.")
        Nmodes = len(mode_labels)
    else:
        Nmodes = recipe["Nmodes"]
    if isinstance(Nmodes, bool) or not isinstance(Nmodes, (int, np.integer)):
        raise ValueError("recipe['Nmodes'] must be a positive integer.")
    Nmodes = int(Nmodes)
    if Nmodes <= 0:
        raise ValueError("recipe['Nmodes'] must be a positive integer.")
    crop = recipe["crop"]
    if isinstance(crop, bool) or not isinstance(crop, (int, np.integer)):
        raise ValueError("recipe['crop'] must be a non-negative integer.")
    crop = int(crop)
    if crop < 0:
        raise ValueError("recipe['crop'] must be a non-negative integer.")

    inputs = {label: np.asarray(value).copy() for label, value in holograms.items()}
    first_shape = next(iter(inputs.values())).shape
    if len(first_shape) != 2:
        raise ValueError("All holograms must be 2D measured intensities.")
    for label, value in inputs.items():
        if value.ndim != 2:
            raise ValueError(f"Hologram {label!r} must be 2D.")
        if value.shape != first_shape:
            raise ValueError("All holograms must have the same shape.")

    mask_pixel = np.asarray(mask_pixel)
    supportmask = np.asarray(supportmask)
    shape_2d = first_shape
    if mask_pixel.shape != shape_2d:
        raise ValueError("mask_pixel must have the same shape as the holograms.")
    if supportmask.ndim == 2:
        if supportmask.shape != shape_2d:
            raise ValueError("2D supportmask must match the hologram shape.")
    elif supportmask.ndim == 3:
        if supportmask.shape[1:] != shape_2d:
            raise ValueError("3D supportmask must have shape (Nmodes, nx, ny).")
        if supportmask.shape[0] not in {1, Nmodes}:
            raise ValueError("supportmask first axis must be 1 or Nmodes.")
    else:
        raise ValueError("supportmask must be 2D or 3D.")

    data = {}
    vmin = recipe["hologram_intensity_cutoff_vmin"]
    for label, intensity in inputs.items():
        if vmin >= 0:
            vals = intensity[(intensity != 0) & np.isfinite(intensity)]
            if vals.size:
                intensity = intensity - np.nanpercentile(vals, vmin)
        intensity = np.where(np.isnan(intensity), 0, intensity)
        bsmask = mask_pixel.copy()
        bsmask[intensity < 0] = 1
        intensity = np.clip(intensity, 0, None)
        data[label] = {
            "amp": np.sqrt(intensity),
            "input": intensity,
            "bsmask": bsmask,
        }

    support_modes = _as_modes(
        supportmask,
        Nmodes,
        shape_2d,
        "supportmask",
        dtype=np.asarray(supportmask).dtype,
    )

    first_startimage = recipe["Startimage"][0]
    if first_startimage is None:
        Startimage_modes = np.empty_like(support_modes, dtype=np.complex128)
        for mode_index in range(Nmodes):
            Startimage_modes[mode_index] = np.fft.fftshift(
                np.fft.ifft2(np.fft.ifftshift(support_modes[mode_index]))
            )
        Startimage = _maybe_squeeze_modes(Startimage_modes, Nmodes)
    elif isinstance(first_startimage, np.ndarray):
        Startimage = _maybe_squeeze_modes(
            _as_modes(
                first_startimage,
                Nmodes,
                shape_2d,
                "Startimage",
                dtype=np.complex128,
            ),
            Nmodes,
        )
    else:
        raise ValueError(
            "The first Startimage entry cannot be a label because no "
            "reconstruction exists yet."
        )

    first_startgamma = recipe["Startgamma"][0]
    if first_startgamma is None:
        Startgamma_modes = np.ones((Nmodes, *shape_2d), dtype=float) * 1e-6 * 2
        Startgamma_modes[:, shape_2d[0] // 2, shape_2d[1] // 2] = 0.7
        Startgamma = _maybe_squeeze_modes(Startgamma_modes, Nmodes)
    elif isinstance(first_startgamma, np.ndarray):
        Startgamma = _maybe_squeeze_modes(
            _as_modes(
                first_startgamma,
                Nmodes,
                shape_2d,
                "Startgamma",
                dtype=float,
            ),
            Nmodes,
        )
    else:
        raise ValueError(
            "The first Startgamma entry cannot be a label because no "
            "coherence estimate exists yet."
        )

    first_label = recipe["helicity"][0]
    valid_pix = (mask_pixel == 0) & (data[first_label]["input"] > 0)
    if np.any(valid_pix):
        x = data[first_label]["amp"][valid_pix].ravel()
        y = _modal_amplitude_numpy(Startimage)[valid_pix].ravel()
        if x.size >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            res = stats.linregress(x, y)
            if abs(res.slope) > 1e-12:
                Startimage = Startimage / res.slope

    retrieved = {label: None for label in labels}
    retrieved_fc = {label: None for label in labels}
    retrieved_pc = {label: None for label in labels}
    retrieved_gradient = {label: None for label in labels}
    gamma = {label: Startgamma.copy() for label in labels}
    default_start_image = Startimage.copy()
    default_start_gamma = Startgamma.copy()

    error = {
        "steps": [],
        "outputs": [],
        "outputs_by_helicity": {label: [] for label in labels},
    }
    start_time = time.time()

    for i, mode in enumerate(recipe["algorithm_list"]):
        label = recipe["helicity"][i]
        Nit = recipe["number_iterations"][i]
        RL_it = int(recipe["RL_its"][i])
        RL_freq = recipe["RL_freqs"][i]
        use_RL = RL_it > 0 and RL_freq <= Nit

        if use_RL:
            if retrieved[label] is not None:
                retrieved_intensity = _modal_intensity_numpy(retrieved[label])
                pc_input = (
                    retrieved_intensity * data[label]["bsmask"]
                    + data[label]["input"] * (1 - data[label]["bsmask"])
                )
            else:
                pc_input = data[label]["input"]
            diffract = np.sqrt(pc_input)
            bsmask = np.zeros_like(data[label]["bsmask"])
            gamma_in = _resolve_start_field(
                recipe["Startgamma"][i],
                default_start_gamma,
                gamma,
                name="Startgamma",
            )
        else:
            diffract = data[label]["amp"]
            bsmask = data[label]["bsmask"]
            gamma_in = None

        start_spec = recipe["Startimage"][i]
        Phase = _resolve_start_field(
            start_spec,
            default_start_image,
            retrieved,
            name="Startimage",
        )
        if (
            recipe["normalize_startimage_between_holograms"]
            and isinstance(start_spec, str)
            and start_spec != label
        ):
            Phase = _scale_phase_between_holograms(
                Phase,
                source_label=start_spec,
                target_label=label,
                intensity_data={key: data[key]["input"] for key in labels},
                bsmasks={key: data[key]["bsmask"] for key in labels},
            )

        if mode == "gradient_descent":
            if use_RL:
                raise ValueError(
                    "gradient_descent recipe stages do not support "
                    "Richardson-Lucy partial-coherence updates."
                )
            if Nmodes == 1:
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
                Error_loss = refined.loss
            else:
                result, Error_diff, Error_supp, Error_loss = _refine_modes_gradient(
                    Phase,
                    diffract,
                    supportmask,
                    bsmask,
                    nmodes=Nmodes,
                    image_shape=shape_2d,
                    Nit=Nit,
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
                    Fourier_last=recipe["Fourier_last"][i],
                )
            gamma_out = None
        else:
            Error_loss = None
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
                Nmodes=Nmodes,
            )

        retrieved[label] = result
        if mode == "gradient_descent":
            retrieved_gradient[label] = result
            if retrieved_fc[label] is None:
                retrieved_fc[label] = result
        elif use_RL:
            retrieved_pc[label] = result
        else:
            retrieved_fc[label] = result
        if gamma_out is not None:
            gamma[label] = gamma_out

        step_info = {
            "step": i,
            "helicity": label,
            "mode": mode,
            "Nit": Nit,
            "RL_it": RL_it,
            "RL_freq": RL_freq,
            "coherence": "partial" if use_RL else "full",
            "output": recipe["output"][i],
            "error": np.asarray(Error_diff),
            "support_error": np.asarray(Error_supp),
            "field_after": result.copy(),
        }
        if Error_loss is not None:
            step_info["loss"] = np.asarray(Error_loss)
        error["steps"].append(step_info)

        if recipe["output"][i]:
            output_info = {
                "step": i,
                "helicity": label,
                "mode": mode,
                "coherence": step_info["coherence"],
                "field": result.copy(),
            }
            error["outputs"].append(output_info)
            error["outputs_by_helicity"][label].append(output_info)

        print(
            f"Step {i}: helicity={label}, mode={mode}, Nit={Nit}, "
            f"{'partial coherence' if use_RL else 'full coherence'}, "
            f"Nmodes={Nmodes}"
        )

    print("--- %s seconds ---" % np.round((time.time() - start_time), 2))
    print("Phase Retrieval Done!")

    if crop:
        if 2 * crop >= min(shape_2d):
            raise ValueError("recipe['crop'] removes the complete reconstruction.")

        def crop_spatial(value):
            if value is None:
                return None
            return np.asarray(value)[..., crop:-crop, crop:-crop]

        for collection in (retrieved_fc, retrieved_pc, retrieved_gradient, gamma):
            for label, value in collection.items():
                collection[label] = crop_spatial(value)
        for label in labels:
            data[label]["bsmask"] = crop_spatial(data[label]["bsmask"])
        for step in error["steps"]:
            step["field_after"] = crop_spatial(step["field_after"])
        for output_info in error["outputs"]:
            output_info["field"] = crop_spatial(output_info["field"])

    result = {
        "full_coherence": retrieved_fc,
        "partial_coherence": retrieved_pc,
        "gradient_descent": retrieved_gradient,
        "bsmasks": {label: data[label]["bsmask"] for label in labels},
        "gamma": gamma,
        "error": error,
        "recipe": recipe,
    }

    return_format = recipe["return_format"]
    if return_format == "legacy" or (return_format == "auto" and legacy_call):
        return _legacy_tuple_from_result(result)
    return result


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


ALPHA_SCHEDULES = BETA_SCHEDULES.copy()
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
    Single-mode full/partial-coherence phase-retrieval kernel.

    This is the lean 2D implementation used by the original
    ``phase_retrieval_core``. It deliberately does not create a modal axis, so
    it is the fast path selected whenever ``Nmodes == 1``.
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
        Phase = (
            (1 - bsmask) * diffract * np.exp(1j * np.angle(Phase))
            + Phase * bsmask
        )

    Phase = np.asarray(Phase)
    if Phase.shape != (l, n):
        raise ValueError("Phase must have same shape as diffract.")

    if RL_freq is None:
        RL_freq = Nit + 1

    use_RL = gamma is not None and RL_it > 0 and RL_freq <= Nit

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

        if use_RL:
            factor = diffract_cp / xp.sqrt(convolved)
            guess_cp[obs] *= factor[obs]
        else:
            guess_cp = xp.where(
                BSmask_cp,
                guess_cp,
                diffract_cp * xp.exp(1j * xp.angle(guess_cp)),
            )

        inv = fft2(guess_cp)

        if (s % TV_freq == 0) and alpha > 0:
            inv = inv + alpha * TV(inv, 1)

        inv = proj_fn(inv, prev, mask_cp, beta, s, Nit)
        prev = inv.copy()

        new_guess = ifft2(inv)

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

    if Fourier_last:
        guess_cp = xp.where(
            BSmask_cp,
            guess_cp,
            diffract_cp * xp.exp(1j * xp.angle(guess_cp)),
        )

    guess = to_numpy(guess_cp, xp)
    return np.fft.ifftshift(guess), Error_diffr_list, Error_supp_list, None


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

        # Apply one measured-intensity constraint jointly to all modes.
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

        # Update every mode independently in support space.
        for m in range(Nmodes):
            inv = fft2(guess_cp[m])

            if (s % TV_freq == 0) and alpha > 0:
                inv = inv + alpha * TV(inv, 1)

            inv = proj_fn(inv, prev[m], mask_cp[m], beta, s, Nit)
            prev[m] = inv.copy()
            new_guess[m] = ifft2(inv)

        # Update partial-coherence kernels only at the requested RL interval.
        if use_RL and s > RL_freq and (s % RL_freq == 0):
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
    """
    if isinstance(Nmodes, bool) or not isinstance(Nmodes, (int, np.integer)):
        raise ValueError("Nmodes must be a positive integer.")
    Nmodes = int(Nmodes)
    if Nmodes <= 0:
        raise ValueError("Nmodes must be a positive integer.")
    if Nmodes == 1:
        return PhaseRtrv_core_single(*args, **kwargs)
    return PhaseRtrv_core_multimode(*args, Nmodes=Nmodes, **kwargs)



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
    with xp.errstate(divide="ignore", invalid="ignore"):
        Error=10*xp.log10(Error)
    return to_numpy(Error, xp)
