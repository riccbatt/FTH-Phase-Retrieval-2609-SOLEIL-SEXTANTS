"""Compare full- and partial-coherence reconstructions on the same MAX IV pair."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "library"))
import fth_phase_workflow as wf
import phase_retrieval_core_unified as pr


def bin_4(image, reducer=np.mean):
    rows, cols = image.shape
    return reducer(image.reshape(rows // 4, 4, cols // 4, 4), axis=(1, 3))


def recipe(include_rl):
    stages = 6 if include_rl else 3
    settings = {
        "algorithm_list": (["HAPRE", "ER", "ER"] * 2)[:stages],
        "number_iterations": [30, 10, 10, 50, 25, 25][:stages],
        "helicity": (["+", "+", "-"] * 2)[:stages],
        "beta_zero": 0.5,
        "beta_mode": (["arctan", "const", "const"] * 2)[:stages],
        "alpha_zero": 0.0,
        "alpha_mode": "const",
        "RL_its": [0, 0, 0, 5, 5, 5][:stages],
        "RL_freqs": [1e9, 1e9, 1e9, 10, 10, 10][:stages],
        "TV_freqs": 1e9,
        "plot_every": 1e9,
        "average_img": 5,
        "Fourier_last": True,
        "Startimage": [None, "+", "+", "+", "+", "+"][:stages],
        "Startgamma": [None, None, None, None, "+", "+"][:stages],
        "hologram_intensity_cutoff_vmin": 0.5,
        "hologram_offset": 0.0,
        "output": [False, True, True, False, True, True][:stages],
        "modes": [1, 2],
        "normalize_startimage_between_holograms": False,
        "subtract_startimage_fit_intercept": False,
        "return_format": "dict",
        "crop": 0,
    }
    return settings


def object_modes(fields):
    return np.fft.ifftshift(
        np.fft.fft2(np.fft.fftshift(fields, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )


def main():
    data = wf.load_data_dict(ROOT / "processed/Logs/data_recon_ImId_1143_maxiv.hdf5")
    holograms = {
        label: bin_4(np.asarray(data["holo"][label]["image_c"]))
        for label in ("+", "-")
    }
    mask_pixel = bin_4(np.asarray(data["mask_pixel"]), reducer=np.max).astype(np.uint8)
    support = zoom(np.asarray(data["supportmask"]), 0.25, order=0).astype(np.uint8)
    results = {}
    for name, include_rl in (("full", False), ("partial", True)):
        result = pr.phase_retrieval_algorithm(
            holograms, mask_pixel, support, recipe(include_rl)
        )
        collection = result["partial_coherence"] if include_rl else result["full_coherence"]
        objects = {label: object_modes(np.asarray(collection[label])) for label in ("+", "-")}
        metrics = {
            label: float(np.real([step for step in result["error"]["steps"]
                                  if step["helicity"] == label][-1]["error"][-1]))
            for label in ("+", "-")
        }
        results[name] = (objects, metrics)

    print(json.dumps({name: metrics for name, (_, metrics) in results.items()}, indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, (name, (objects, _)) in enumerate(results.items()):
        plus, minus = objects["+"][0], objects["-"][0]
        ratio_phase = np.angle(plus * np.conj(minus))
        for col, image in enumerate((np.log1p(np.abs(plus)), np.log1p(np.abs(minus)), ratio_phase)):
            axes[row, col].imshow(image, cmap="gray" if col < 2 else "twilight")
            axes[row, col].set_xlim(150, 350)
            axes[row, col].set_ylim(350, 150)
            axes[row, col].set_title(name + " " + ("+ amplitude", "- amplitude", "relative phase")[col])
    fig.tight_layout()
    output = ROOT / "processed" / "full_vs_partial_1143.png"
    fig.savefig(output, dpi=140)
    print("Saved", output)


if __name__ == "__main__":
    main()
