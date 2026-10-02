"""Load, plot, and save one-dimensional diode/energy scans."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _text(value.item())
    return str(value)


def scan_channels(path: str | Path) -> dict[str, str]:
    """Return searchable channel aliases mapped to dataset paths in a NeXus file."""
    channels: dict[str, str] = {}
    with h5py.File(path, "r") as handle:
        def visit(name, item):
            if not isinstance(item, h5py.Dataset) or item.ndim == 0:
                return
            aliases = {name, name.rsplit("/", 1)[-1]}
            for attr in ("long_name", "title", "name", "signal"):
                if attr in item.attrs:
                    value = _text(item.attrs[attr])
                    aliases.update((value, value.rsplit("/", 1)[-1]))
            for alias in aliases:
                channels.setdefault(alias.casefold(), name)

        handle.visititems(visit)
    return channels


def load_scan_channel(path: str | Path, channel: str) -> np.ndarray:
    """Load a scan channel by full path, dataset name, or metadata alias."""
    channels = scan_channels(path)
    key = channel.casefold()
    if key not in channels:
        matches = sorted({value for alias, value in channels.items() if key in alias})
        if len(matches) != 1:
            available = sorted(k for k in channels if "/" not in k)
            raise KeyError(
                f"Channel {channel!r} has {len(matches)} matches in {path}. "
                f"Available aliases include: {available[:30]}"
            )
        dataset_path = matches[0]
    else:
        dataset_path = channels[key]
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[dataset_path]).squeeze()


def save_diode_scans(
    output_folder: str | Path,
    scan_ids: Sequence[int | str],
    xdata: Sequence[np.ndarray],
    ydata: Sequence[np.ndarray],
    *,
    x_channel: str,
    y_channel: str,
    normalization: Sequence[np.ndarray] | None = None,
    normalization_channel: str | None = None,
    user: str = "",
) -> tuple[Path, Path]:
    """Save diode scan arrays to HDF5 and their common plot to PNG."""
    if not scan_ids or not (len(scan_ids) == len(xdata) == len(ydata)):
        raise ValueError("scan_ids, xdata, and ydata must have the same non-zero length")
    if normalization is not None and len(normalization) != len(scan_ids):
        raise ValueError("normalization must contain one array per scan")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    ids = [str(value) for value in scan_ids]
    id_part = ids[0] if len(ids) == 1 else f"{ids[0]}-{ids[-1]}"
    user_part = f"_{user}" if user else ""
    stem = f"DiodeScan_{x_channel}_ScanId_{id_part}{user_part}"
    png_path = output_folder / f"{stem}.png"
    h5_path = output_folder / f"{stem}.hdf5"
    axis_name = x_channel.casefold()
    if "energy" in axis_name:
        scan_type = "diode_energy_scan"
    elif "field" in axis_name or "tesla" in axis_name:
        scan_type = "diode_field_scan"
    else:
        scan_type = "diode_scan"

    fig, ax = plt.subplots(figsize=(7, 5))
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["scan_type"] = scan_type
        handle.attrs["x_channel"] = x_channel
        handle.attrs["y_channel"] = y_channel
        handle.attrs["normalization_channel"] = normalization_channel or ""
        handle.attrs["png_file"] = str(png_path)
        for index, (scan_id, x_values, y_values) in enumerate(zip(ids, xdata, ydata)):
            x_values = np.asarray(x_values).squeeze()
            y_values = np.asarray(y_values).squeeze()
            if x_values.shape != y_values.shape:
                raise ValueError(
                    f"Scan {scan_id}: x and y must have matching shapes, got "
                    f"{x_values.shape} and {y_values.shape}"
                )
            # Continuous field/diode trace scans are commonly stored as a
            # point-by-sample matrix. Flatten in acquisition order so rising
            # and falling hysteresis branches remain in their measured order.
            x_values = x_values.ravel()
            y_values = y_values.ravel()
            norm = (
                np.ones_like(y_values, dtype=float)
                if normalization is None
                else np.asarray(normalization[index]).squeeze().ravel()
            )
            if norm.shape != y_values.shape:
                raise ValueError(f"Scan {scan_id}: normalization shape does not match y")
            normalized = np.divide(
                y_values, norm, out=np.full(y_values.shape, np.nan, dtype=float),
                where=norm != 0,
            )
            group = handle.create_group(f"scan_{scan_id}")
            group["x"] = x_values
            group["diode"] = y_values
            group["normalization"] = norm
            group["diode_normalized"] = normalized
            ax.plot(x_values, normalized, "o-", markersize=2, label=f"Scan {scan_id}")

    ax.set_title(f"Diode scan {id_part}")
    ax.set_xlabel(x_channel)
    ylabel = y_channel
    if normalization_channel:
        ylabel += f" / {normalization_channel}"
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return h5_path, png_path
