"""Shared pixel loading for the MCP/CMOS notebooks."""
from pathlib import Path
import h5py
import numpy as np


def load_detector_image(index, detector, input_folder, image_key="data_22"):
    """MCP: filename or zero-based sorted file index. CMOS: scan number."""
    folder = Path(input_folder)
    if detector == "MCP":
        files = sorted(folder.glob("*_TDC_2D_*.txt"))
        path = files[int(index)] if isinstance(index, (int, np.integer)) else folder / index
        return np.loadtxt(path, dtype=float)
    if detector != "CMOS":
        raise ValueError('Choose detector = "MCP" or "CMOS"')
    index = int(index)
    with h5py.File(folder / f"scanxx_{index:04d}.nxs", "r") as file:
        data = np.asarray(file[f"scan_{index:04d}/scan_data/{image_key}"], dtype=float)
    return data.reshape(-1, *data.shape[-2:]).mean(axis=0)
