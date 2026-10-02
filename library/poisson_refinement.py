"""Optional quasi-Poisson polishing of CDI fields with fixed coherence kernels.

Camera ADU after thresholding/averaging are not independent photon counts: this
is a likelihood-shaped objective, not a calibrated statistical likelihood.
Inspired by Diederichs, Filbir & Römer (2024), doi:10.1088/1361-6420/ad97d7;
this is projected Poisson optimization, not their FIVS algorithm.
"""
from dataclasses import dataclass
import numpy as np
from scipy.fft import fft2, ifft2


@dataclass
class PoissonRefinementResult:
    fields: np.ndarray
    loss: np.ndarray
    step_sizes: np.ndarray
    baseline_loss: float
    valid_pixels: int
    zero_pixels: int
    status: str


def _modes(field):
    """
    Shape adapter: represent a single field as a one-mode stack so the same intensity and
    gradient code supports either case.
    """
    array = np.asarray(field, dtype=complex)
    if array.ndim not in (2, 3):
        raise ValueError("Expected a field or a stack of modes.")
    return array[None] if array.ndim == 2 else array


def _kernel_fft(gamma, shape):
    """
    Prepare fixed blur for the optimizer. Normalize each modal kernel and move its centered
    origin into FFT order before transforming. None denotes full coherence.
    """
    if gamma is None:
        return None
    kernel = np.broadcast_to(np.asarray(gamma).real, shape).copy()
    if not np.all(np.isfinite(kernel)) or np.any(kernel < -1e-10):
        raise ValueError("Gamma must be finite and nonnegative.")
    kernel = np.maximum(kernel, 0)
    mass = kernel.sum(axis=(-2, -1), keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("Gamma must have positive mass per mode.")
    return fft2(np.fft.ifftshift(kernel / mass, axes=(-2, -1)))


def _blur(values, kernel_fft, adjoint=False):
    """
    Forward/adjoint circular convolution on centered detector arrays. The adjoint conjugates
    the kernel spectrum and is needed to back-propagate intensity residuals correctly.
    """
    if kernel_fft is None:
        return values
    kernel = np.conj(kernel_fft) if adjoint else kernel_fft
    return np.fft.fftshift(ifft2(fft2(np.fft.ifftshift(values, axes=(-2, -1))) * kernel).real,
                          axes=(-2, -1))


def predicted_intensity(field, gamma=None):
    """
    Forward detector model used for diagnostics and objective comparisons: blur each modal
    intensity, then sum incoherently. No object-space or magnetic projection is applied.
    """
    modes = _modes(field)
    return np.maximum(_blur(np.abs(modes) ** 2, _kernel_fft(gamma, modes.shape)).sum(axis=0), 0)


def valid_pixels(measured, mask):
    """
    Likelihood sample selection: retain finite nonnegative measurements including zeros, and
    exclude explicit invalid pixels. Negative dark-corrected values are not usable counts.
    """
    measured = np.asarray(measured, dtype=float)
    # True zeros are evidence. Negative dark-corrected values are not counts.
    return np.isfinite(measured) & (measured >= 0) & ~np.broadcast_to(np.asarray(mask, dtype=bool), measured.shape)


def poisson_deviance(prediction, measured, mask=0, floor=1e-10):
    """
    Scalar diagnostic on observed pixels. Compare stages using the same mask and forward
    model; in processed camera units this is a quasi-Poisson score rather than a calibrated
    statistical test.
    """
    valid = valid_pixels(measured, mask)
    if not np.any(valid):
        raise ValueError("No valid nonnegative measurements.")
    y = np.asarray(measured)[valid]
    mu = np.maximum(np.asarray(prediction)[valid], 0) + floor
    terms = mu - y
    positive = y > 0
    terms[positive] += y[positive] * np.log(y[positive] / mu[positive])
    return float(2 * np.mean(terms))


def refine_poisson(field, measured, support, mask=0, *, gamma=None, steps=50,
                   learning_rate=1., refine_modes=None, tolerance=1e-7):
    """
    Projected Wirtinger descent with backtracking; no amplitude overwrite.

    Support is centered and may be modal. Only ``refine_modes`` are changed;
    use [0] to preserve a shared secondary mode. Missing-pixel fills are ignored.
    The returned history starts AFTER enforcing support on the starting field;
    baseline_loss records the original, potentially unsupported starting field.

    Context:
    Optional post-retrieval optimizer, separate from the joint physical model. Keep
    secondary modes fixed with refine_modes=[0]. The returned fields may improve detector
    agreement without preserving the baseline fitted magnetic/charge maps.
    """
    original = np.asarray(field)
    current = _modes(field).copy()
    y = np.asarray(measured, dtype=float)
    if y.shape != current.shape[-2:]:
        raise ValueError("Measurement and field grids must match.")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or tolerance < 0:
        raise ValueError("Invalid optimization controls.")
    if not np.all(np.isfinite(current)):
        raise ValueError("Fields must be finite.")
    valid = valid_pixels(y, mask)
    count = int(valid.sum())
    if count == 0:
        raise ValueError("No valid nonnegative measurements.")
    y = np.where(valid, y, 0.)
    floor = max(float(y[valid].mean()), 1.) * 1e-10
    kfft = _kernel_fft(gamma, current.shape)
    support = np.fft.fftshift(np.broadcast_to(np.asarray(support) != 0, current.shape), axes=(-2, -1))
    selected = np.ones(len(current), dtype=bool) if refine_modes is None else np.isin(np.arange(len(current)), refine_modes)
    if not selected.any():
        raise ValueError("Select at least one mode to refine.")

    def project(array):
        return np.fft.ifftshift(ifft2(fft2(np.fft.fftshift(array, axes=(-2, -1))) * support), axes=(-2, -1))

    def evaluate(array, gradient=False):
        mu = np.maximum(_blur(np.abs(array) ** 2, kfft).sum(axis=0), 0) + floor
        loss = float(np.sum((mu - y * np.log(mu))[valid]) / count)
        if not gradient:
            return loss
        # Unmeasured pixels supply no likelihood gradient; measured zeros do.
        residual = np.where(valid, 1 - y / mu, 0)
        grad = array * _blur(np.broadcast_to(residual, array.shape), kfft, adjoint=True)
        grad[~selected] = 0
        return loss, project(grad)

    # Keep the pre-support baseline so plots expose any initial projection cost.
    baseline = evaluate(current)
    if steps:
        projected = project(current)
        current[selected] = projected[selected]
    loss, grad = evaluate(current, True)
    history, rates = [loss], []
    status = "iteration_limit"
    rate = learning_rate
    for _ in range(steps):
        norm = float(np.sum(np.abs(grad) ** 2) / count)
        if norm == 0:
            status = "stationary"
            break
        trial_rate = min(rate * 1.5, learning_rate)
        # Backtracking accepts only a finite sufficient decrease; a failed search
        # returns the last accepted field, not the rejected trial.
        for _ in range(30):
            candidate = current - trial_rate * grad
            candidate_loss = evaluate(candidate)
            if np.isfinite(candidate_loss) and candidate_loss <= loss - 1e-4 * 2 * trial_rate * norm:
                break
            trial_rate *= .5
        else:
            status = "line_search_stalled"
            break
        improvement = loss - candidate_loss
        current = candidate
        loss, grad = evaluate(current, True)
        history.append(loss)
        rates.append(trial_rate)
        rate = trial_rate
        if improvement <= tolerance * max(1., abs(loss)):
            status = "converged"
            break
    result = current[0] if original.ndim == 2 else current
    return PoissonRefinementResult(result, np.asarray(history), np.asarray(rates), baseline,
                                   count, int(np.sum(valid & (y == 0))), status)
