"""Shared NumPy/CuPy arrays and spatial FFTs."""
import numpy as np

try:
    import cupy as cp
    GPU = cp.is_available()
except Exception:
    GPU = False

if GPU:
    xp = cp
    from cupyx.scipy.fft import fft2, ifft2
else:
    xp = np
    from scipy.fft import fft2, ifft2

to_numpy = xp.asnumpy if GPU else np.asarray
