"""Safe, tabular inspection helpers for HDF5/NeXus files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


def _readable(value: Any, max_items: int = 8) -> Any:
    """Convert small HDF5 values to display-friendly Python values."""
    array = np.asarray(value)
    if array.size > max_items:
        return f"<{array.size} values>"
    result = array.tolist()

    def decode(item: Any) -> Any:
        if isinstance(item, bytes):
            return item.decode(errors="replace")
        if isinstance(item, list):
            return [decode(child) for child in item]
        return item

    return decode(result)


def inspect_nexus(path: str | Path, max_preview_items: int = 8) -> list[dict[str, Any]]:
    """Inventory every object and attribute without loading large datasets."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            attributes = {key: _readable(value, max_preview_items) for key, value in obj.attrs.items()}
            row: dict[str, Any] = {
                "file": path.name,
                "path": f"/{name}",
                "kind": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                "shape": tuple(obj.shape) if isinstance(obj, h5py.Dataset) else None,
                "dtype": str(obj.dtype) if isinstance(obj, h5py.Dataset) else None,
                "value": None,
                "attributes": attributes,
                "long_name": attributes.get("long_name"),
                "units": attributes.get("units"),
                "description": attributes.get("description"),
            }
            if isinstance(obj, h5py.Dataset) and obj.size <= max_preview_items:
                row["value"] = _readable(obj[()], max_preview_items)
            rows.append(row)

        handle.visititems(visit)
    return rows


def search_inventory(rows: Iterable[dict[str, Any]], query: str = "") -> list[dict[str, Any]]:
    """Case-insensitively search paths, values, and attributes."""
    query = query.strip().lower()
    if not query:
        return list(rows)
    return [row for row in rows if query in repr(row).lower()]


def scalar_metadata(path: str | Path) -> dict[str, Any]:
    """Return scalar/single-value datasets keyed by their absolute paths."""
    return {
        row["path"]: row["value"]
        for row in inspect_nexus(path, max_preview_items=1)
        if row["kind"] == "dataset" and row["value"] is not None
    }


__all__ = ["inspect_nexus", "scalar_metadata", "search_inventory"]
