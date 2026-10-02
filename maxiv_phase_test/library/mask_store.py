"""Persistent per-image detector masks shared by notebooks."""

from pathlib import Path

import numpy as np
from PIL import Image


RED_MIN = 200
OTHER_MAX = 80


def load_red_mask_png(path: str | Path, expected_shape=None) -> np.ndarray:
    """Load bright-red pixels from a Paint-edited PNG as a binary mask."""
    path = Path(path)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    mask = (
        (rgb[..., 0] >= RED_MIN)
        & (rgb[..., 1] <= OTHER_MAX)
        & (rgb[..., 2] <= OTHER_MAX)
    ).astype(np.uint8)
    if expected_shape is not None and mask.shape != tuple(expected_shape):
        raise ValueError(
            f"Mask {path} has shape {mask.shape}, expected {tuple(expected_shape)}. "
            "Do not resize the reference PNG while editing it."
        )
    return mask


def save_red_mask_png(path: str | Path, mask: np.ndarray) -> Path:
    """Save a binary mask as an RGB PNG with masked pixels in bright red."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.asarray(mask) > 0
    rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
    rgb[binary] = (255, 0, 0)
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


class MaskStore:
    """Save one raw-coordinate binary ``mask_pixel`` PNG per image ID.

    PNG is canonical. Loading falls back to the legacy ``.npy`` filename.
    """

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)

    @staticmethod
    def _safe_id(image_id: int | str) -> str:
        return str(image_id).replace("/", "_").replace("\\", "_")

    def path_for(self, image_id: int | str) -> Path:
        return self.folder / f"mask_pixel_{self._safe_id(image_id)}.png"

    def legacy_path_for(self, image_id: int | str) -> Path:
        return self.folder / f"mask_pixel_{self._safe_id(image_id)}.npy"

    def old_folder_path_for(self, image_id: int | str) -> Path:
        """Path used before masks moved from ``processed/masks``."""
        return self.folder.parent / "masks" / f"mask_pixel_{self._safe_id(image_id)}.npy"

    def save(self, image_id: int | str, mask: np.ndarray) -> Path:
        return save_red_mask_png(self.path_for(image_id), mask)

    def load(self, image_id: int | str, expected_shape=None) -> np.ndarray:
        path = self.path_for(image_id)
        if path.is_file():
            return load_red_mask_png(path, expected_shape)
        path = self.legacy_path_for(image_id)
        if not path.is_file():
            path = self.old_folder_path_for(image_id)
        mask = (np.load(path, allow_pickle=False) > 0).astype(np.uint8)
        if expected_shape is not None and mask.shape != tuple(expected_shape):
            raise ValueError(f"Mask {path} has shape {mask.shape}, expected {tuple(expected_shape)}")
        return mask

    def exists(self, image_id: int | str) -> bool:
        return any(
            path.is_file()
            for path in (
                self.path_for(image_id),
                self.legacy_path_for(image_id),
                self.old_folder_path_for(image_id),
            )
        )
