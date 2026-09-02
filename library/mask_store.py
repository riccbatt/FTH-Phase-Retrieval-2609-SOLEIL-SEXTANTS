"""Persistent per-image detector masks shared by notebooks."""

from pathlib import Path

import numpy as np


class MaskStore:
    """Save one binary ``mask_pixel`` array per raw image ID."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)

    def path_for(self, image_id: int | str) -> Path:
        safe_id = str(image_id).replace("/", "_").replace("\\", "_")
        return self.folder / f"mask_pixel_{safe_id}.npy"

    def save(self, image_id: int | str, mask: np.ndarray) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path_for(image_id)
        np.save(path, (np.asarray(mask) > 0).astype(np.uint8))
        return path

    def load(self, image_id: int | str, expected_shape=None) -> np.ndarray:
        path = self.path_for(image_id)
        mask = (np.load(path, allow_pickle=False) > 0).astype(np.uint8)
        if expected_shape is not None and mask.shape != tuple(expected_shape):
            raise ValueError(f"Mask {path} has shape {mask.shape}, expected {tuple(expected_shape)}")
        return mask

    def exists(self, image_id: int | str) -> bool:
        return self.path_for(image_id).is_file()
