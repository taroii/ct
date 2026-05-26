"""Regenerate ct-2 phantom iteration-ladder PNGs from existing recon caches.

Skips recon: loads cache/ct2_<name>_recon[_arc<N>].pkl and re-renders the
iteration-ladder figure with the supplied iter list. Used to:
  - bump the 3D analytic breast (50 deg) ladder to include iter 500,
  - render head and jaw (100 deg) ladders with three columns
    (ground truth, one intermediate, iter 500) per the talk redesign.

Cache files were produced by scripts/presentation_ct2_phantom_ladder.py;
they already contain snapshots at [10, 50, 100, 200, 300, 500].

Inlines the figure helpers so we don't import astra-toolbox.
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
    "breast": {"vmin": 0.0, "vmax": 0.6},
    "head":   {"vmin": 0.0, "vmax": 2.6},
    "jaw":    {"vmin": 0.0, "vmax": 3.0},
}

JOBS = [
    # (phantom, arc, iters, tag)
    ("breast", 50,  [50, 100, 200, 500], ""),
    # Head and jaw at 100 deg: revert to iter 300 because RMSE rises after.
    ("head",  100,  [100, 300],          "_arc100"),
    ("jaw",   100,  [100, 300],          "_arc100"),
]


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _xy_slice(vol):
    nz = vol.shape[0]
    z = nz // 2
    z0, z1 = max(z - 1, 0), min(z + 2, nz)
    return vol[z0:z1].mean(axis=0)


def fig_iter_ladder(result, iters, out_path, display):
    phi = result["phantom"]
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]

    slc_gt = _xy_slice(phi)
    vmin, vmax = display["vmin"], display["vmax"]

    n = len(iters)
    fig, axes = plt.subplots(3, n + 1, figsize=(1.9 * (n + 1), 5.6))
    for ax in axes.flat:
        _strip(ax)

    for r, label in enumerate(["ground truth", "single", "two-channel"]):
        axes[r, 0].imshow(slc_gt, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[r, 0].set_ylabel(label, fontsize=11)
    axes[0, 0].set_title("ground truth", fontsize=10)

    for i, it in enumerate(iters):
        col = i + 1
        s = _xy_slice(snaps_s[it])
        t = _xy_slice(snaps_t[it])
        axes[0, col].imshow(slc_gt, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[0, col].set_title(f"iter {it}", fontsize=10)
        axes[1, col].imshow(s, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[2, col].imshow(t, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    for name, arc, iters, tag in JOBS:
        cache_path = CACHE_DIR / f"ct2_{name}_recon{tag}.pkl"
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
        cached = sorted(result["snapshots_two"].keys())
        missing = [it for it in iters if it not in cached]
        if missing:
            print(f"[{name} arc={arc}] MISSING iters {missing}; have {cached}")
            continue
        print(f"[{name} arc={arc}] rendering iters {iters} (cache has {cached})")
        out = FIG_DIR / f"ct2_{name}_iter_ladder_xy{tag}.png"
        fig_iter_ladder(result, iters, out, DISPLAYS[name])


if __name__ == "__main__":
    main()
