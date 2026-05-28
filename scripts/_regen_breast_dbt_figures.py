"""Regenerate the 3D breast slide figures from the new DBT cache.

Reads cache/ct2_breast_recon.pkl (the promoted V3 cache) and produces:
    presentation/figs/ct2_breast_iter_ladder_xy.png  (mid-axial slice
        across iterations, single vs two-channel)
    presentation/figs/ct2_breast_convergence.png    (image RMSE vs iter)
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "presentation" / "figs"
CACHE   = ROOT / "cache" / "ct2_breast_recon.pkl"

LADDER_ITERS = [50, 100, 200, 500]
DISPLAY = {"vmin": 0.0, "vmax": 0.6}


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _xy_slice(vol):
    nz = vol.shape[0]
    z = nz // 2
    z0, z1 = max(z - 1, 0), min(z + 2, nz)
    return vol[z0:z1].mean(axis=0)


def fig_iter_ladder(result, iters, out_path):
    phi = result["phantom"]
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]

    slc_gt = _xy_slice(phi)
    vmin, vmax = DISPLAY["vmin"], DISPLAY["vmax"]

    n = len(iters)
    fig, axes = plt.subplots(3, n + 1, figsize=(1.9 * (n + 1), 5.6))
    for ax in axes.flat:
        _strip(ax)

    for r, label in enumerate(["ground truth", "single", "two-channel"]):
        axes[r, 0].imshow(slc_gt, cmap="gray", vmin=vmin, vmax=vmax,
                          origin="lower")
        axes[r, 0].set_ylabel(label, fontsize=11)
    axes[0, 0].set_title("ground truth", fontsize=10)

    for i, it in enumerate(iters):
        col = i + 1
        s = _xy_slice(snaps_s[it])
        t = _xy_slice(snaps_t[it])
        axes[0, col].imshow(slc_gt, cmap="gray", vmin=vmin, vmax=vmax,
                            origin="lower")
        axes[0, col].set_title(f"iter {it}", fontsize=10)
        axes[1, col].imshow(s, cmap="gray", vmin=vmin, vmax=vmax,
                            origin="lower")
        axes[2, col].imshow(t, cmap="gray", vmin=vmin, vmax=vmax,
                            origin="lower")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_convergence(result, out_path):
    s = np.asarray(result["ierrs_single"])
    t = np.asarray(result["ierrs_two"])
    iters = np.arange(1, len(s) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(iters, s, "r-", lw=1.4, label="single-channel")
    ax.semilogy(iters, t, "b-", lw=1.4, label="two-channel")
    ax.set_xlabel("iteration")
    ax.set_ylabel("image RMSE")
    ax.set_xlim(0, len(s))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    with open(CACHE, "rb") as f:
        result = pickle.load(f)
    print(f"Loaded {CACHE}")
    print(f"  ierrs_single: {len(result['ierrs_single'])} iters, "
          f"final {result['ierrs_single'][-1]:.5f}")
    print(f"  ierrs_two:    {len(result['ierrs_two'])} iters, "
          f"final {result['ierrs_two'][-1]:.5f}")
    print(f"  snapshots:    single {sorted(result['snapshots_single'].keys())}, "
          f"two {sorted(result['snapshots_two'].keys())}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_iter_ladder(result, LADDER_ITERS,
                    FIG_DIR / "ct2_breast_iter_ladder_xy.png")
    fig_convergence(result,
                    FIG_DIR / "ct2_breast_convergence.png")


if __name__ == "__main__":
    main()
