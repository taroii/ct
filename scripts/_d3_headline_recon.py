"""Focused D3 slide figure: 2D paper-breast reconstructions for
single-channel, paper-config two-channel (r=1.25, already cached at
cache/iter_ladder_paper_256.pkl), and tuned two-channel (r=15). One
recon run; the rest comes from the existing cache.

Output: final_figures/H2_eps_headline_recon_256.png  (3 rows x 5 cols)
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_methods_multiresolution as cm   # noqa: E402

MFACT = 2
ITERS = [50, 100, 200, 500]
ITERMAX = 500
SNAPSHOT_ITERS = [50, 100, 200, 300, 500]
C_HI = 4.0
C_LO = 8.0
R_TUNED = 15.0

CACHE_PAPER = ROOT / "cache" / "iter_ladder_paper_256.pkl"
CACHE_TUNED = ROOT / "cache" / "iter_ladder_tuned_r15_256.pkl"
OUT_FIG     = ROOT / "final_figures" / "H2_eps_headline_recon_256.png"

DISPLAY_VMIN = 0.0
DISPLAY_VMAX = 1.6      # matches the desktop's wider window for the 2D figures


def run_tuned():
    if CACHE_TUNED.exists():
        with open(CACHE_TUNED, "rb") as f:
            return pickle.load(f)
    cm.cutoffparm = C_HI
    cm.cutoffparm_lo = C_LO
    cm.eps_hi = cm.eps
    cm.eps_lo = R_TUNED * cm.eps
    cm.RESOLUTION_PARAMS[256]["itermax"] = ITERMAX
    print(f"Running 256x256 tuned (r={R_TUNED}) recon...")
    t0 = time.time()
    result = cm.run_reconstruction_for_mfact(MFACT,
                                             snapshot_iters=SNAPSHOT_ITERS)
    print(f"  elapsed {time.time()-t0:.1f}s")
    with open(CACHE_TUNED, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached -> {CACHE_TUNED}")
    return result


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def main():
    with open(CACHE_PAPER, "rb") as f:
        paper = pickle.load(f)
    tuned = run_tuned()

    phi = paper["phimage"]
    snaps_single = paper["snapshots_single"]
    snaps_paper  = paper["snapshots_two"]
    snaps_tuned  = tuned["snapshots_two"]

    rows = [
        ("single",                 snaps_single),
        (r"two, $r{=}1.25$ (paper)", snaps_paper),
        (r"two, $r{=}15$ (tuned)",   snaps_tuned),
    ]

    n = len(ITERS)
    fig, axes = plt.subplots(
        len(rows), n + 1,
        figsize=(1.9 * (n + 1), 1.9 * len(rows) + 0.3),
        constrained_layout=True,
    )
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    for r_idx, (label, snaps) in enumerate(rows):
        axes[r_idx, 0].imshow(phi.T, cmap="gray",
                              vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX,
                              origin="lower")
        axes[r_idx, 0].set_ylabel(label, fontsize=11)
        if r_idx == 0:
            axes[r_idx, 0].set_title("ground truth", fontsize=10)
        for i, it in enumerate(ITERS):
            col = i + 1
            img = snaps[it]
            axes[r_idx, col].imshow(img.T, cmap="gray",
                                    vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX,
                                    origin="lower")
            if r_idx == 0:
                axes[r_idx, col].set_title(f"iter {it}", fontsize=10)
            axes[r_idx, col].set_xlabel(f"RMSE {rmse(img, phi):.3f}",
                                        fontsize=8)

    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
