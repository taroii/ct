"""
Visual comparison of single- vs two-channel reconstruction over the early
iterations of PDHG.

Emil's framing for the talk (paper/emil_advice.md):
- Clinical practice doesn't fix an iteration count, so the early-iteration
  trajectory matters more than the asymptote.
- Show a series of images per method at increasing iterations and let the
  audience see the low-frequency blur in single-channel resolve faster in
  two-channel.

We use the matched-cutoff partition-of-unity two-channel design (c_hi=c_lo=4)
which gave the cleanest, hump-free trajectory in scripts/sweep_pou_cutoff.py.
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import compare_methods_multiresolution as cm

MFACT = 2  # 256x256
SNAPSHOT_ITERS = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
# At iters 50-200 two-channel beats single (15-17% lower RMSE);
# crossover is ~iter 500; single dominates past that.
ADVANTAGE_ITERS = [50, 100, 200, 500]
CROSSOVER_ITERS = [200, 500, 1000, 2000]
CACHE_PATH = "cache/early_iteration_snapshots_256.pkl"

# Display window chosen to emphasize soft-tissue contrast. Calcifications
# (sparse, ~1.5) saturate, but soft-tissue band [0, 1.0] reads clearly.
DISPLAY_VMIN = 0.0
DISPLAY_VMAX = 1.0

# Gaussian std for "low-frequency error" isolation, in pixels. nx=256, so
# sigma=8 picks up structures larger than ~50 px (1/5 of the image),
# matching the band the two-channel design targets.
LF_SIGMA = 8.0


def _strip_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def _rmse(img, truth):
    return float(np.sqrt(((img - truth) ** 2).mean()))


def make_recon_grid(snaps_s, snaps_t, phimage, iters, vmin, vmax, out_path,
                    title=None):
    n = len(iters)
    fig, axes = plt.subplots(2, n + 1, figsize=(1.9 * (n + 1), 4.2))

    for ax in axes.flat:
        _strip_axes(ax)

    axes[0, 0].imshow(phimage.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0, 0].set_title("ground truth", fontsize=10)
    axes[0, 0].set_ylabel("single-channel", fontsize=11)
    axes[1, 0].imshow(phimage.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1, 0].set_ylabel("two-channel (PoU)", fontsize=11)

    for i, it in enumerate(iters):
        col = i + 1
        s, t = snaps_s[it], snaps_t[it]
        axes[0, col].imshow(s.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[0, col].set_title(f"iter {it}\nRMSE {_rmse(s, phimage):.3f}", fontsize=10)
        axes[1, col].imshow(t.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[1, col].set_xlabel(f"RMSE {_rmse(t, phimage):.3f}", fontsize=10)

    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_error_grid(snaps_s, snaps_t, phimage, iters, err_range, out_path,
                    title=None):
    n = len(iters)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.0))

    for ax in axes.flat:
        _strip_axes(ax)

    axes[0, 0].set_ylabel("single − truth", fontsize=11)
    axes[1, 0].set_ylabel("two − truth", fontsize=11)

    for i, it in enumerate(iters):
        d_s = snaps_s[it] - phimage
        d_t = snaps_t[it] - phimage
        axes[0, i].imshow(d_s.T, cmap="gray", vmin=-err_range, vmax=err_range,
                          origin="lower")
        axes[0, i].set_title(f"iter {it}", fontsize=10)
        axes[1, i].imshow(d_t.T, cmap="gray", vmin=-err_range, vmax=err_range,
                          origin="lower")

    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_lf_error_grid(snaps_s, snaps_t, phimage, iters, sigma, err_range,
                       out_path, title=None):
    """Low-pass-filtered error maps: isolates the LF blur band the
    two-channel design targets. Each panel is gaussian_filter(recon - truth,
    sigma). LF RMSE printed underneath each panel.
    """
    n = len(iters)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.2))

    for ax in axes.flat:
        _strip_axes(ax)

    axes[0, 0].set_ylabel("LF(single − truth)", fontsize=11)
    axes[1, 0].set_ylabel("LF(two − truth)", fontsize=11)

    for i, it in enumerate(iters):
        lf_s = gaussian_filter(snaps_s[it] - phimage, sigma)
        lf_t = gaussian_filter(snaps_t[it] - phimage, sigma)
        axes[0, i].imshow(lf_s.T, cmap="gray", vmin=-err_range, vmax=err_range,
                          origin="lower")
        axes[0, i].set_title(f"iter {it}\nLF RMSE {float(np.sqrt((lf_s**2).mean())):.4f}",
                             fontsize=9)
        axes[1, i].imshow(lf_t.T, cmap="gray", vmin=-err_range, vmax=err_range,
                          origin="lower")
        axes[1, i].set_xlabel(f"LF RMSE {float(np.sqrt((lf_t**2).mean())):.4f}",
                              fontsize=9)

    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_convergence_plot(ierrs_s, ierrs_t, out_path, xlim=None, title=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    iters = range(1, len(ierrs_s) + 1)
    ax.semilogy(iters, ierrs_s, "r-", linewidth=1.4, label="single-channel")
    ax.semilogy(iters, ierrs_t, "b-", linewidth=1.4, label="two-channel (PoU c=4)")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel("iteration")
    ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper right")
    if title is not None:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs("cache", exist_ok=True)
    os.makedirs("final_figures", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        print(f"Loading cached run from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            result = pickle.load(f)
    else:
        cm.cutoffparm = 4.0
        cm.cutoffparm_lo = 4.0
        cm.eps_hi = cm.eps
        cm.eps_lo = 1.25 * cm.eps
        print(f"Running 256x256 PoU c=4 reconstruction with snapshots at {SNAPSHOT_ITERS}")
        result = cm.run_reconstruction_for_mfact(MFACT, snapshot_iters=SNAPSHOT_ITERS)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached to {CACHE_PATH}")

    phimage = result["phimage"]
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]
    ierrs_s = result["ierrs_single"]
    ierrs_t = result["ierrs_two"]

    make_recon_grid(
        snaps_s, snaps_t, phimage, ADVANTAGE_ITERS, DISPLAY_VMIN, DISPLAY_VMAX,
        "final_figures/early_iter_recon_256.png",
        title="Iterations where two-channel beats single (256x256, PoU c=4)",
    )

    make_recon_grid(
        snaps_s, snaps_t, phimage, CROSSOVER_ITERS, DISPLAY_VMIN, DISPLAY_VMAX,
        "final_figures/late_iter_recon_256.png",
        title="Crossover and asymptote (256x256, PoU c=4)",
    )

    err_range_early = 0.3
    err_range_late = 0.15
    make_error_grid(
        snaps_s, snaps_t, phimage, ADVANTAGE_ITERS, err_range_early,
        "final_figures/early_iter_error_256.png",
        title=f"Error maps (recon − truth), gray ±{err_range_early} (256x256, PoU c=4)",
    )
    make_error_grid(
        snaps_s, snaps_t, phimage, CROSSOVER_ITERS, err_range_late,
        "final_figures/late_iter_error_256.png",
        title=f"Error maps (recon − truth), gray ±{err_range_late} (256x256, PoU c=4)",
    )

    make_lf_error_grid(
        snaps_s, snaps_t, phimage, ADVANTAGE_ITERS, LF_SIGMA, 0.15,
        "final_figures/early_iter_lferror_256.png",
        title=f"Low-frequency error (Gaussian sigma={LF_SIGMA} px), gray ±0.15",
    )
    make_lf_error_grid(
        snaps_s, snaps_t, phimage, CROSSOVER_ITERS, LF_SIGMA, 0.10,
        "final_figures/late_iter_lferror_256.png",
        title=f"Low-frequency error (Gaussian sigma={LF_SIGMA} px), gray ±0.10",
    )

    make_convergence_plot(
        ierrs_s, ierrs_t,
        "final_figures/early_convergence_256.png",
        xlim=(1, 500),
        title="Image RMSE, first 500 iterations (256x256, PoU c=4)",
    )
    make_convergence_plot(
        ierrs_s, ierrs_t,
        "final_figures/full_convergence_256.png",
        title="Image RMSE, full 2000 iterations (256x256, PoU c=4)",
    )


if __name__ == "__main__":
    main()
