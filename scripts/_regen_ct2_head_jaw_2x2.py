"""Regenerate head and jaw iteration-ladder PNGs as 2x2 grids at iter 100.

Layout:
    rows: single, two-channel
    cols: ground truth, iter 100

Reads cache/ct2_{head,jaw}_recon_arc100.pkl (already contains the iter-100
snapshot). Overwrites the existing PNGs the deck points at:
    presentation/figs/ct2_head_iter_ladder_xy_arc100.png
    presentation/figs/ct2_jaw_iter_ladder_xy_arc100.png
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "presentation" / "figs"
CACHE_DIR = ROOT / "cache"

DISPLAYS = {
    "head":   {"vmin": 0.0, "vmax": 2.6},
    "jaw":    {"vmin": 0.0, "vmax": 3.0},
}

JOBS = [
    # (phantom, arc, iter, tag)
    ("head",  100, 100, "_arc100"),
    ("jaw",   100, 100, "_arc100"),
]


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _xy_slice(vol):
    nz = vol.shape[0]
    z = nz // 2
    z0, z1 = max(z - 1, 0), min(z + 2, nz)
    return vol[z0:z1].mean(axis=0)


def fig_iter_ladder_2x2(result, it, out_path, display):
    phi = result["phantom"]
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]

    slc_gt = _xy_slice(phi)
    slc_s = _xy_slice(snaps_s[it])
    slc_t = _xy_slice(snaps_t[it])
    vmin, vmax = display["vmin"], display["vmax"]

    fig, axes = plt.subplots(2, 2, figsize=(3.8, 3.9))
    for ax in axes.flat:
        _strip(ax)

    axes[0, 0].imshow(slc_gt, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1, 0].imshow(slc_gt, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0, 1].imshow(slc_s,  cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1, 1].imshow(slc_t,  cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

    axes[0, 0].set_title("ground truth", fontsize=10)
    axes[0, 1].set_title(f"iter {it}", fontsize=10)
    axes[0, 0].set_ylabel("single", fontsize=11)
    axes[1, 0].set_ylabel("two-channel", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    for name, arc, it, tag in JOBS:
        cache_path = CACHE_DIR / f"ct2_{name}_recon{tag}.pkl"
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
        cached = sorted(result["snapshots_two"].keys())
        if it not in cached:
            print(f"[{name} arc={arc}] MISSING iter {it}; have {cached}")
            continue
        print(f"[{name} arc={arc}] rendering iter {it} (cache has {cached})")
        out = FIG_DIR / f"ct2_{name}_iter_ladder_xy{tag}.png"
        fig_iter_ladder_2x2(result, it, out, DISPLAYS[name])


if __name__ == "__main__":
    main()
