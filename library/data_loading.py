"""Facility-independent raw detector data loading.

Loaders return the same :class:`Frame` object, so notebooks do not need to know
whether an image came from SPE or NeXus.  NeXus dataset paths are configurable;
the SEXTANTS defaults are based on the previous beamtime loader and can be
updated after inspecting representative files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import h5py
import numpy as np


@dataclass(frozen=True)
class Frame:
    """A detector image plus the acquisition metadata needed downstream."""

    image_id: str
    image: np.ndarray
    exposure: float
    source: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RawLoader(Protocol):
    def load(self, image_id: int | str) -> Frame: ...


def _first_dataset(handle: h5py.File, candidates: tuple[str, ...]) -> tuple[str, Any]:
    roots = ("",) + tuple(f"{key}/" for key in handle.keys())
    for candidate in candidates:
        candidate = candidate.lstrip("/")
        for root in roots:
            path = f"{root}{candidate}"
            if path in handle and isinstance(handle[path], h5py.Dataset):
                return path, handle[path][()]
    raise KeyError(f"None of these datasets exists: {', '.join(candidates)}")


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value).squeeze()
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    result = float(array)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive, got {result}")
    return result


class NexusLoader:
    """Load detector frames from HDF5/NeXus files.

    ``filename`` may be a format string (for example ``"scan_{id:05d}.nxs"``)
    or a callable returning a path. Multiple candidate dataset paths make the
    loader tolerant of a top-level NeXus entry group.
    """

    SEXTANTS_IMAGE_PATHS = (
        "scan_data/data_22",
        "entry/scan_data/data_22",
        "entry/data/data",
    )
    SEXTANTS_EXPOSURE_PATHS = (
        "scan_data/integration_times",
        "entry/scan_data/integration_times",
        "entry/instrument/detector/count_time",
        "entry/instrument/detector/exposure_time",
    )

    def __init__(
        self,
        raw_folder: str | Path,
        filename: str | Callable[[int | str], str | Path] = "{id}.nxs",
        image_paths: tuple[str, ...] = SEXTANTS_IMAGE_PATHS,
        exposure_paths: tuple[str, ...] = SEXTANTS_EXPOSURE_PATHS,
        frame_reduction: str = "mean",
    ) -> None:
        self.raw_folder = Path(raw_folder)
        self.filename = filename
        self.image_paths = tuple(image_paths)
        self.exposure_paths = tuple(exposure_paths)
        self.frame_reduction = frame_reduction

    def path_for(self, image_id: int | str) -> Path:
        if callable(self.filename):
            path = Path(self.filename(image_id))
        else:
            try:
                name = self.filename.format(id=int(image_id))
            except (TypeError, ValueError):
                name = self.filename.format(id=image_id)
            path = Path(name)
        return path if path.is_absolute() else self.raw_folder / path

    def load(self, image_id: int | str) -> Frame:
        source = self.path_for(image_id)
        with h5py.File(source, "r") as handle:
            image_path, raw_image = _first_dataset(handle, self.image_paths)
            exposure_path, raw_exposure = _first_dataset(handle, self.exposure_paths)

        image = np.asarray(raw_image, dtype=float).squeeze()
        if image.ndim > 2:
            if self.frame_reduction == "mean":
                image = image.mean(axis=0)
            elif self.frame_reduction == "sum":
                image = image.sum(axis=0)
            else:
                raise ValueError("frame_reduction must be 'mean' or 'sum'")
        if image.ndim != 2:
            raise ValueError(f"Expected a 2-D detector image, got shape {image.shape}")
        exposure = _scalar(raw_exposure, "exposure")
        return Frame(
            image_id=str(image_id),
            image=image,
            exposure=exposure,
            source=source,
            metadata={"image_path": image_path, "exposure_path": exposure_path},
        )


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


class SextantsNexusLoader(NexusLoader):
    """Loader for SOLEIL SEXTANTS ``scanx_NNNN.nxs`` files.

    Detector keys such as ``data_21`` and ``data_22`` are scan-dependent. This
    loader identifies the image using dataset metadata instead of the number.
    """

    ENERGY_PATHS = ("SEXTANTS/mono/energy", "SEXTANTS/hu80.2_energy/energy")
    POLARIZATION_PATHS = ("SEXTANTS/hu80.2_energy/polarisation",)

    @staticmethod
    def _ccd_distance(scan_data: h5py.Group) -> tuple[str, float] | None:
        """Derive sample-to-CCD distance from the ``ccd-ts`` position in mm."""
        # At SEXTANTS the CCD translation stage is stored in data_03. Prefer
        # that stable beamtime location; retain the metadata search as fallback.
        candidates = []
        if "data_03" in scan_data:
            candidates.append(scan_data["data_03"])
        candidates.extend(
            dataset for key, dataset in scan_data.items() if key != "data_03"
        )
        for dataset in candidates:
            if not isinstance(dataset, h5py.Dataset):
                continue
            long_name = _text(dataset.attrs.get("long_name", "")).lower()
            if dataset.name.rsplit("/", 1)[-1] != "data_03" and "ccd-ts" not in long_name:
                continue
            values = np.asarray(dataset[()]).squeeze()
            if values.size != 1:
                raise ValueError(f"CCD distance dataset {dataset.name} is not scalar")
            ccd_ts_mm = float(values)
            return dataset.name.lstrip("/"), (700.0 - ccd_ts_mm) / 1000.0
        return None

    def __init__(self, raw_folder: str | Path, frame_reduction: str = "mean") -> None:
        super().__init__(
            raw_folder,
            filename="scanx_{id:04d}.nxs",
            exposure_paths=("scan_data/integration_times",),
            frame_reduction=frame_reduction,
        )

    @staticmethod
    def _detector_dataset(handle: h5py.File, entry: str) -> tuple[str, h5py.Dataset]:
        scan_data = handle[f"{entry}/scan_data"]
        candidates = []
        for key, dataset in scan_data.items():
            if not isinstance(dataset, h5py.Dataset):
                continue
            interpretation = _text(dataset.attrs.get("interpretation", "")).lower()
            long_name = _text(dataset.attrs.get("long_name", "")).lower()
            if dataset.ndim >= 3 and (interpretation == "image" or "image" in long_name):
                candidates.append((key, dataset))
        if not candidates:
            raise KeyError(f"No detector image dataset found under {entry}/scan_data")
        # Prefer the largest detector if a scan happens to contain more than one.
        key, dataset = max(candidates, key=lambda item: np.prod(item[1].shape[-2:]))
        return f"{entry}/scan_data/{key}", dataset

    def load(self, image_id: int | str) -> Frame:
        source = self.path_for(image_id)
        with h5py.File(source, "r") as handle:
            entries = [key for key in handle if isinstance(handle[key], h5py.Group)]
            expected = f"scan_{int(image_id):04d}"
            entry = expected if expected in handle else entries[0]
            image_path, dataset = self._detector_dataset(handle, entry)
            raw_image = dataset[()]
            detector = _text(dataset.attrs.get("long_name", image_path))
            exposure_path, raw_exposure = _first_dataset(handle, self.exposure_paths)
            metadata: dict[str, Any] = {
                "entry": entry,
                "image_path": image_path,
                "exposure_path": exposure_path,
                "detector": detector,
            }
            distance = self._ccd_distance(handle[f"{entry}/scan_data"])
            if distance is not None:
                distance_path, distance_m = distance
                metadata["ccd_dist_m"] = distance_m
                metadata["ccd_dist_path"] = distance_path
            for name, paths in (
                ("energy_eV", self.ENERGY_PATHS),
                ("polarization_code", self.POLARIZATION_PATHS),
            ):
                try:
                    path, value = _first_dataset(handle, paths)
                    metadata[name] = np.asarray(value).squeeze().item()
                    metadata[f"{name}_path"] = path
                except (KeyError, ValueError):
                    pass

        image = np.asarray(raw_image, dtype=float).squeeze()
        exposures = np.asarray(raw_exposure, dtype=float).reshape(-1)
        if image.ndim > 2:
            if self.frame_reduction == "mean":
                image = image.mean(axis=0)
                exposure = float(exposures.mean())
            elif self.frame_reduction == "sum":
                image = image.sum(axis=0)
                exposure = float(exposures.sum())
            else:
                raise ValueError("frame_reduction must be 'mean' or 'sum'")
        else:
            exposure = _scalar(exposures, "exposure")
        if image.ndim != 2:
            raise ValueError(f"Expected a 2-D detector image, got shape {image.shape}")
        exposure = _scalar(exposure, "exposure")
        return Frame(str(image_id), image, exposure, source, metadata)

    def load_channel(self, scan_id: int | str, field: str) -> np.ndarray:
        """Load a SEXTANTS dataset by path, key, or ``long_name`` fragment."""
        source = self.path_for(scan_id)
        query = str(field).strip().lower()
        with h5py.File(source, "r") as handle:
            expected = f"scan_{int(scan_id):04d}"
            entries = [key for key in handle if isinstance(handle[key], h5py.Group)]
            entry = expected if expected in handle else entries[0]
            direct = str(field).lstrip("/")
            for candidate in (direct, f"{entry}/{direct}", f"{entry}/scan_data/{direct}"):
                if candidate in handle and isinstance(handle[candidate], h5py.Dataset):
                    return np.asarray(handle[candidate][()]).squeeze()
            matches: list[h5py.Dataset] = []

            def visitor(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                key = name.rsplit("/", 1)[-1].lower()
                long_name = _text(obj.attrs.get("long_name", "")).lower()
                if query == key or query in long_name:
                    matches.append(obj)

            handle[entry].visititems(visitor)
            if not matches:
                raise KeyError(f"No SEXTANTS channel matching {field!r} in {source}")
            if len(matches) > 1:
                names = [dataset.name for dataset in matches]
                raise KeyError(f"Ambiguous SEXTANTS channel {field!r}: {names}")
            return np.asarray(matches[0][()]).squeeze()

    def load_scan_data(self, scan_id: int | str) -> dict[str, np.ndarray]:
        """Return all datasets in the SEXTANTS ``scan_data`` group."""
        source = self.path_for(scan_id)
        with h5py.File(source, "r") as handle:
            expected = f"scan_{int(scan_id):04d}"
            entries = [key for key in handle if isinstance(handle[key], h5py.Group)]
            entry = expected if expected in handle else entries[0]
            scan_data = handle[f"{entry}/scan_data"]
            result: dict[str, np.ndarray] = {}
            for key, dataset in scan_data.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                value = np.asarray(dataset[()]).squeeze()
                result[key] = value
                long_name = _text(dataset.attrs.get("long_name", "")).strip()
                if long_name:
                    result.setdefault(long_name, value)
                    result.setdefault(long_name.rsplit("/", 1)[-1], value)
            return result

    def experimental_setup(
        self,
        image_id: int | str,
        *,
        px_size: float = 11.0e-6,
        binning: int = 1,
        oversaturation: float = 60e3,
    ) -> dict[str, Any]:
        """Build reconstruction geometry from SEXTANTS acquisition metadata."""
        frame = self.load(image_id)
        missing = [key for key in ("ccd_dist_m", "energy_eV") if key not in frame.metadata]
        if missing:
            raise KeyError(f"Missing required SEXTANTS metadata {missing} in {frame.source}")
        energy = float(frame.metadata["energy_eV"])
        wavelength = 1.239841984e-6 / energy
        return {
            "ccd_dist": float(frame.metadata["ccd_dist_m"]),
            "ccd_dist_source": frame.metadata["ccd_dist_path"],
            "ccd_dist_image_id": int(image_id),
            "energy": energy,
            "energy_source": frame.metadata.get("energy_eV_path"),
            "lambda": wavelength,
            "px_size": float(px_size),
            "binning": int(binning),
            "oversaturation": float(oversaturation),
        }


class SpeLoader:
    """Adapter for the existing SPE workflow.

    The actual SPE reader is injected to avoid forcing a particular SPE package.
    It must return a 2-D image, or ``(image, metadata)``. Exposure can be supplied
    by a callable or read from the returned metadata.
    """

    def __init__(
        self,
        raw_folder: str | Path,
        reader: Callable[[Path], Any],
        filename: str = "{id}.spe",
        exposure: Callable[[int | str], float] | None = None,
    ) -> None:
        self.raw_folder = Path(raw_folder)
        self.reader = reader
        self.filename = filename
        self.exposure_for = exposure

    def load(self, image_id: int | str) -> Frame:
        try:
            name = self.filename.format(id=int(image_id))
        except (TypeError, ValueError):
            name = self.filename.format(id=image_id)
        source = self.raw_folder / name
        loaded = self.reader(source)
        if isinstance(loaded, tuple):
            image, metadata = loaded
        else:
            image, metadata = loaded, {}
        image = np.asarray(image, dtype=float).squeeze()
        if image.ndim != 2:
            raise ValueError(f"Expected a 2-D detector image, got shape {image.shape}")
        value = self.exposure_for(image_id) if self.exposure_for else metadata.get("exposure", 1.0)
        return Frame(str(image_id), image, _scalar(value, "exposure"), source, metadata)


class LoaderRegistry:
    """Named collection of raw-data loaders used by notebooks."""

    def __init__(self) -> None:
        self._loaders: dict[str, RawLoader] = {}

    def register(self, name: str, loader: RawLoader) -> None:
        self._loaders[name.lower()] = loader

    def load(self, kind: str, image_id: int | str) -> Frame:
        try:
            loader = self._loaders[kind.lower()]
        except KeyError as exc:
            raise KeyError(f"Unknown loader {kind!r}; available: {sorted(self._loaders)}") from exc
        return loader.load(image_id)
