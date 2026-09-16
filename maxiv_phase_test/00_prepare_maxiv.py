"""Prepare two MAX IV averaged CMOS images for the current FTH/CDI notebooks."""

from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "library"))
import CCI_core as cci
import fth_phase_workflow as wf


CMOS_FOLDER = Path("/media/riccardo/EXT_16TB1/DATA/MAXIV/MaxIV2409/proc")
POS_ID, NEG_ID = 1143, 1145
POS_DARK_IDS, NEG_DARK_IDS = (1142, 1144), (1144, 1146)
DETECTOR_CENTER = (1137, 1126)  # (row, column), from the quantitative notebook
BEAMSTOP_RADIUS = 30
SATURATION = 60000


def load_average(scan_id):
    filename = CMOS_FOLDER / f"scan_{scan_id:04d}_avg.h5"
    with h5py.File(filename, "r") as handle:
        image = np.asarray(handle["image"][()]).squeeze()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image in {filename}, got {image.shape}")
    return image.astype(np.float32)


def centered_dark_corrected(scan_id, dark_ids):
    dark = np.mean([load_average(i) for i in dark_ids], axis=0)
    return wf.center_image(load_average(scan_id) - dark, DETECTOR_CENTER, cci)


def main():
    pos = centered_dark_corrected(POS_ID, POS_DARK_IDS)
    neg = centered_dark_corrected(NEG_ID, NEG_DARK_IDS)
    if pos.shape != neg.shape:
        raise ValueError(f"Image shapes differ: {pos.shape} and {neg.shape}")

    yy, xx = np.ogrid[: pos.shape[0], : pos.shape[1]]
    center = np.asarray(pos.shape) // 2
    beamstop = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= BEAMSTOP_RADIUS ** 2
    mask_pixel = (beamstop | (pos >= SATURATION) | (neg >= SATURATION)).astype(np.uint8)

    data = {
        "positive_label": "+",
        "reference_label": "-",
        "center": np.asarray(DETECTOR_CENTER),
        "mask_pixel": mask_pixel,
        "factor": 1.0,
        "offset": 0.0,
        "experimental_setup": {"ccd_dist": 0.195, "px_size": 11e-6, "binning": 1},
        "focus_fth": {"prop_dist": 0.0, "phase": 0.0},
        "holo": {
            "+": {"id": POS_ID, "dark_id": np.asarray(POS_DARK_IDS), "image_c": pos},
            "-": {"id": NEG_ID, "dark_id": np.asarray(NEG_DARK_IDS), "image_c": neg},
        },
    }
    output = ROOT / "processed" / "Logs" / f"data_recon_ImId_{POS_ID:04d}_maxiv.hdf5"
    wf.save_data_dict(data, output, overwrite=True)
    print(f"Saved {output}; shape={pos.shape}, excluded pixels={int(mask_pixel.sum())}")


if __name__ == "__main__":
    main()
