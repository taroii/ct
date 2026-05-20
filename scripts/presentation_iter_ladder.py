"""
Iteration-ladder figures for the CT-Meeting 2026 talk, using the *paper's*
two-channel configuration (c_hi=4, c_lo=8), not the matched-cutoff PoU
design the previous round of presentation figures used.

Outputs (all under presentation/figs/):
- iter_ladder_recon_256_paper.png   — 2 x 7 grid (gt + 6 iters) of recons
- iter_ladder_lferror_256_paper.png — 2 x 6 grid of LP-filtered error maps
- iter_ladder_convergence_256_paper.png — image-RMSE vs iter, linear + loglog
- iter_ladder_zoom_recon_256_paper.png  — same iter ladder but cropped to a
  homogeneous soft-tissue ROI where the LF wobble is most legible

Cache: cache/iter_ladder_paper_256.pkl
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_methods_multiresolution as cm

MFACT = 2  # 256x256
SNAPSHOT_ITERS = [1, 5, 10, 20, 50, 100, 200, 300, 500]
LADDER_ITERS = [5, 20, 50, 100, 200, 500]
MOTIVATION_ITERS = [10, 20, 50, 100, 200, 300]

CACHE_PATH = ROOT / "cache" / "iter_ladder_paper_256.pkl"
FIG_DIR = ROOT / "presentation" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_VMIN = 0.0
DISPLAY_VMAX = 1.0
LF_SIGMA = 8.0


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def regenerate():
    # Force paper config: c_hi = 4, c_lo = 8 (different from PoU c=4).
    cm.cutoffparm = 4.0
    cm.cutoffparm_lo = 8.0
    cm.eps_hi = cm.eps
    cm.eps_lo = 1.25 * cm.eps
    # Use the paper's 500-iter run, not the 2000-iter overkill.
    cm.RESOLUTION_PARAMS[256]["itermax"] = 500
    print(f"Running 256x256 paper-config recon with snapshots at {SNAPSHOT_ITERS}")
    return cm.run_reconstruction_for_mfact(MFACT, snapshot_iters=SNAPSHOT_ITERS)


def load_or_run():
    if CACHE_PATH.exists():
        print(f"Loading {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    result = regenerate()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached to {CACHE_PATH}")
    return result


def fig_error_pair(result, iters, out_path, title=None, err_range=0.3):
    """Plain (no filtering) error maps at a few iterations, single vs two,
    with a colorbar legend for the difference scale.

    Used in place of the Gaussian-LP-filtered version --- the math jargon
    is unnecessary friction for an engineering audience.
    """
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]
    phi = result["phimage"]
    n = len(iters)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6),
                             constrained_layout=True)
    for ax in axes.flat:
        _strip(ax)
    axes[0, 0].set_ylabel("single - truth", fontsize=11)
    axes[1, 0].set_ylabel("two - truth", fontsize=11)
    im = None
    for i, it in enumerate(iters):
        d_s = snaps_s[it] - phi
        d_t = snaps_t[it] - phi
        im = axes[0, i].imshow(d_s.T, cmap="gray", vmin=-err_range,
                               vmax=err_range, origin="lower")
        axes[0, i].set_title(f"iter {it}", fontsize=10)
        axes[1, i].imshow(d_t.T, cmap="gray", vmin=-err_range,
                          vmax=err_range, origin="lower")
    if title is not None:
        fig.suptitle(title, fontsize=11)
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                        fraction=0.05, pad=0.02, aspect=50,
                        ticks=[-err_range, 0.0, err_range])
    cbar.ax.set_xticklabels([f"-{err_range:g}", "0", f"+{err_range:g}"])
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(
        "reconstruction - ground truth   "
        "(black: under-estimate,  mid-gray: zero error,  "
        "white: over-estimate)", fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_single_only_ladder(result, iters, out_path, title=None):
    """Motivation slide: single-channel only iteration ladder.

    Lets the audience see the LF wobble unresolved with no comparison
    drawn yet --- two-channel is introduced later.
    """
    snaps_s = result["snapshots_single"]
    phi = result["phimage"]
    n = len(iters)
    fig, axes = plt.subplots(1, n + 1, figsize=(1.9 * (n + 1), 2.6))
    for ax in axes.flat:
        _strip(ax)
    axes[0].imshow(phi.T, cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX,
                   origin="lower")
    axes[0].set_title("ground truth", fontsize=10)
    for i, it in enumerate(iters):
        s = snaps_s[it]
        axes[i + 1].imshow(s.T, cmap="gray", vmin=DISPLAY_VMIN,
                           vmax=DISPLAY_VMAX, origin="lower")
        axes[i + 1].set_title(f"iter {it}", fontsize=10)
    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_recon_ladder(result, iters, out_path, title=None, crop=None):
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]
    phi = result["phimage"]

    def view(img):
        if crop is None:
            return img.T
        r0, r1, c0, c1 = crop
        return img[r0:r1, c0:c1].T

    n = len(iters)
    fig, axes = plt.subplots(2, n + 1, figsize=(1.9 * (n + 1), 4.4))
    for ax in axes.flat:
        _strip(ax)

    axes[0, 0].imshow(view(phi), cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX,
                      origin="lower")
    axes[0, 0].set_title("ground truth", fontsize=10)
    axes[0, 0].set_ylabel("single-channel", fontsize=11)
    axes[1, 0].imshow(view(phi), cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX,
                      origin="lower")
    axes[1, 0].set_ylabel("two-channel", fontsize=11)

    for i, it in enumerate(iters):
        col = i + 1
        s, t = snaps_s[it], snaps_t[it]
        axes[0, col].imshow(view(s), cmap="gray", vmin=DISPLAY_VMIN,
                            vmax=DISPLAY_VMAX, origin="lower")
        axes[0, col].set_title(f"iter {it}\nRMSE {_rmse(s, phi):.3f}", fontsize=10)
        axes[1, col].imshow(view(t), cmap="gray", vmin=DISPLAY_VMIN,
                            vmax=DISPLAY_VMAX, origin="lower")
        axes[1, col].set_xlabel(f"RMSE {_rmse(t, phi):.3f}", fontsize=10)

    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_lf_error_ladder(result, iters, out_path, title=None):
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]
    phi = result["phimage"]

    err_range = 0.15
    n = len(iters)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.2))
    for ax in axes.flat:
        _strip(ax)
    axes[0, 0].set_ylabel("LF(single - truth)", fontsize=11)
    axes[1, 0].set_ylabel("LF(two - truth)", fontsize=11)

    for i, it in enumerate(iters):
        lf_s = gaussian_filter(snaps_s[it] - phi, LF_SIGMA)
        lf_t = gaussian_filter(snaps_t[it] - phi, LF_SIGMA)
        axes[0, i].imshow(lf_s.T, cmap="gray", vmin=-err_range, vmax=err_range,
                          origin="lower")
        axes[0, i].set_title(
            f"iter {it}\nLF RMSE {float(np.sqrt((lf_s**2).mean())):.4f}",
            fontsize=9)
        axes[1, i].imshow(lf_t.T, cmap="gray", vmin=-err_range, vmax=err_range,
                          origin="lower")
        axes[1, i].set_xlabel(
            f"LF RMSE {float(np.sqrt((lf_t**2).mean())):.4f}", fontsize=9)

    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_convergence(result, out_path, title=None):
    ierrs_s = result["ierrs_single"]
    ierrs_t = result["ierrs_two"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    iters = np.arange(1, len(ierrs_s) + 1)
    axes[0].plot(iters, ierrs_s, "r-", linewidth=1.4, label="single-channel")
    axes[0].plot(iters, ierrs_t, "b-", linewidth=1.4, label="two-channel")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("image RMSE")
    axes[0].set_title("linear iteration axis")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=9)

    axes[1].plot(iters, ierrs_s, "r-", linewidth=1.4, label="single-channel")
    axes[1].plot(iters, ierrs_t, "b-", linewidth=1.4, label="two-channel")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("iteration (log)")
    axes[1].set_ylabel("image RMSE")
    axes[1].set_title("log-log")
    axes[1].grid(True, alpha=0.3, which="both")

    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    result = load_or_run()

    fig_single_only_ladder(
        result, MOTIVATION_ITERS,
        FIG_DIR / "single_only_ladder_256_paper.png",
        title="Single-channel reconstruction over iterations (256x256)",
    )

    fig_recon_ladder(
        result, LADDER_ITERS,
        FIG_DIR / "iter_ladder_recon_256_paper.png",
        title="Iteration ladder (256x256, paper config: c_hi=4, c_lo=8)",
    )

    fig_error_pair(
        result, [50, 100, 200, 300],
        FIG_DIR / "error_pair_256_paper.png",
        title=None,
        err_range=0.3,
    )

    fig_lf_error_ladder(
        result, LADDER_ITERS,
        FIG_DIR / "iter_ladder_lferror_256_paper.png",
        title=f"Low-frequency error map (Gaussian sigma={LF_SIGMA} px), gray +-0.15",
    )

    fig_convergence(
        result,
        FIG_DIR / "iter_ladder_convergence_256_paper.png",
        title="Image RMSE convergence, 256x256 (paper config, 500 iter)",
    )

    # ROI: a soft-tissue-only patch in the upper-left of the breast disk to
    # show LF wobble. Phantom is 256x256 in (x,y); .T-display means rows ->
    # ydisplay, cols -> xdisplay. After several looks at the recon, the patch
    # (45:155, 30:140) in (row, col) sits over predominantly adipose with
    # some glandular speckling.
    fig_recon_ladder(
        result, LADDER_ITERS,
        FIG_DIR / "iter_ladder_zoom_recon_256_paper.png",
        title="Zoomed soft-tissue ROI (256x256, paper config)",
        crop=(45, 155, 30, 140),
    )


if __name__ == "__main__":
    main()
