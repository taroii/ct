"""
Poisson-noise sweep on the 2D paper-config 256x256 pipeline.

For each photon count in NOISE_LEVELS, runs single- and two-channel PDHG
recon with c_hi=4, c_lo=8 (paper config) and saves the final image,
image-RMSE-vs-iter curve, and a 2x4 recon grid at a fixed reporting iter.

Output figure:
  presentation/figs/noise_sweep_256_paper.png  -- 2 x 4 recon grid at iter
                                                  REPORT_ITER, one column
                                                  per noise level

Cache:
  cache/noise_sweep_256_paper.pkl              -- dict {nph: result}
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# cm uses relative paths to load phantom data, so chdir to the repo root.
os.chdir(ROOT)

import compare_methods_multiresolution as cm   # noqa: E402

MFACT       = 2                  # 256x256
ITERMAX     = 300                # cap to keep sweep fast (~6 min total)
REPORT_ITER = 200                # iter to compare across noise levels
# nph below ~1e4 produces Poisson zeros on the deep-attenuation paths of
# the adipose phantom (~exp(-8) * nph < 1 expected photons), yielding nans.
# Stick with three levels: clean -> moderate -> photon-noise dominated.
NOISE_LEVELS = [None, 1e5, 1e4]
SNAPSHOT_ITERS = [50, 100, 200, 300]
CACHE_PATH = ROOT / "cache" / "noise_sweep_256_paper.pkl"
FIG_DIR    = ROOT / "presentation" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_one(nph):
    cm.cutoffparm    = 4.0
    cm.cutoffparm_lo = 8.0
    cm.eps_hi        = cm.eps
    cm.eps_lo        = 1.25 * cm.eps
    cm.RESOLUTION_PARAMS[256]["itermax"] = ITERMAX

    if nph is None:
        cm.addnoise = 0
        cm.nph      = 1.0      # unused
        label = "noiseless"
    else:
        cm.addnoise = 1
        cm.nph      = nph
        label = f"nph={nph:.0e}"

    print(f"\n========== {label} ==========")
    res = cm.run_reconstruction_for_mfact(MFACT, snapshot_iters=SNAPSHOT_ITERS)
    print(f"  single iter {ITERMAX}: RMSE = {res['ierrs_single'][-1]:.4f}")
    print(f"  two    iter {ITERMAX}: RMSE = {res['ierrs_two'][-1]:.4f}")
    return res


def load_or_run(force):
    if CACHE_PATH.exists() and not force:
        print(f"Loading cached sweep from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    out = {}
    for nph in NOISE_LEVELS:
        key = "noiseless" if nph is None else f"{nph:.0e}"
        out[key] = run_one(nph)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(out, f)
    print(f"Cached sweep -> {CACHE_PATH}")
    return out


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def fig_sweep_grid(results, report_iter, out_path, title=None):
    """2 x N grid: one column per noise level, rows single / two-channel
    at iter=report_iter. Also overlays the per-noise-level RMSE numbers."""
    keys = list(results.keys())
    n = len(keys)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    if n == 1:
        axes = axes[:, None]

    # global gray window from the noiseless single-channel recon
    ref = results[keys[0]]["snapshots_single"][report_iter]
    vmin, vmax = 0.0, float(np.percentile(ref[ref > 0], 99.0)) * 1.05

    for col, key in enumerate(keys):
        res = results[key]
        s = res["snapshots_single"][report_iter]
        t = res["snapshots_two"][report_iter]
        # RMSE at report_iter (ierrs are 1-indexed by iter number, list idx = iter-1)
        rs = res["ierrs_single"][report_iter - 1]
        rt = res["ierrs_two"][report_iter - 1]

        axes[0, col].imshow(s.T, cmap="gray", vmin=vmin, vmax=vmax,
                            origin="lower")
        axes[0, col].set_title(key, fontsize=11)
        axes[0, col].text(0.5, -0.08, f"RMSE = {rs:.3f}",
                          transform=axes[0, col].transAxes,
                          ha="center", fontsize=9)
        _strip(axes[0, col])

        axes[1, col].imshow(t.T, cmap="gray", vmin=vmin, vmax=vmax,
                            origin="lower")
        axes[1, col].text(0.5, -0.08, f"RMSE = {rt:.3f}",
                          transform=axes[1, col].transAxes,
                          ha="center", fontsize=9)
        _strip(axes[1, col])

    axes[0, 0].set_ylabel("single", fontsize=12)
    axes[1, 0].set_ylabel("two-channel", fontsize=12)

    if title is not None:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_sweep_curves(results, out_path, title=None):
    """One semilog-y plot, two curves per noise level (single solid, two dashed)
    colored by noise level."""
    keys = list(results.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, key in enumerate(keys):
        c = cmap(i / max(1, len(keys) - 1))
        s = results[key]["ierrs_single"]
        t = results[key]["ierrs_two"]
        iters = np.arange(1, len(s) + 1)
        ax.semilogy(iters, s, "-",  color=c, lw=1.4,
                    label=f"single, {key}")
        ax.semilogy(iters, t, "--", color=c, lw=1.4,
                    label=f"two,    {key}")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    if title is not None: ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    results = load_or_run(args.force)

    fig_sweep_grid(
        results, REPORT_ITER,
        FIG_DIR / "noise_sweep_256_paper.png",
        title=f"2D paper-config 256x256, iter {REPORT_ITER}, paper-config "
              f"two-channel ($c_{{\\mathrm{{hi}}}}=4, c_{{\\mathrm{{lo}}}}=8$)",
    )
    fig_sweep_curves(
        results,
        FIG_DIR / "noise_sweep_curves_256_paper.png",
        title="Image RMSE vs iteration across photon counts",
    )

    print("\nFinal RMSE summary:")
    for key, res in results.items():
        print(f"  {key:>10}  single={res['ierrs_single'][REPORT_ITER-1]:.4f}  "
              f"two={res['ierrs_two'][REPORT_ITER-1]:.4f}")


if __name__ == "__main__":
    main()
