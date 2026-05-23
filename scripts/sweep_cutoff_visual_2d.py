"""
H1 -- 2D paper-breast, non-PoU low-pass cutoff sweep, rendered as reconstruction
+ error rows across iterations (per Emil 2026-05-22).

Fixes c_hi = 4 (the paper config), varies c_lo in {6, 8, 12, 16}. The paper
config sits at c_lo = 8. Smaller c_lo widens the LF band (more of the
sinogram lands in the slow-mode channel); larger c_lo narrows it toward DC.

For each c_lo value we run the full two-channel reconstruction with
snapshots at iters [50, 100, 200, 500], then lay them out as one row per
configuration in a single recon grid + a matching error grid. A
single-channel reconstruction is included as the top reference row.

Outputs:
  final_figures/H1_cutoff_recon_256.png    recon panels (recon ladder)
  final_figures/H1_cutoff_error_256.png    |recon - truth| panels
  final_figures/H1_cutoff_lferror_256.png  Gaussian-LP error panels (sigma=8)
  final_figures/H1_cutoff_convergence_256.png  single + 4 two-channel curves
  final_figures/H1_cutoff_summary_256.txt  RMSE at each (c_lo, iter)
  cache/H1_cutoff_sweep_256.pkl            per-c_lo snapshots + curves

Run:  conda run --no-capture-output -n ct2 \
          python scripts/sweep_cutoff_visual_2d.py [--force]
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import compare_methods_multiresolution as cm   # noqa: E402

# --- Defaults --------------------------------------------------------------
MFACT = 2                            # 256 x 256
RESOLUTION = int(512 / MFACT)
C_HI_FIXED = 4.0
C_LO_DEFAULTS = [6.0, 8.0, 12.0, 16.0]
SNAPSHOT_ITERS = [50, 100, 200, 500]
ITERMAX = 500                        # only need to reach the last snapshot
EPS_LO_RATIO = 1.25                  # paper config

DISPLAY_VMIN = 0.0
DISPLAY_VMAX = 1.0                   # soft-tissue window
ERR_RANGE_RECON = 0.30
ERR_RANGE_LF = 0.15
LF_SIGMA = 8.0

CACHE_PATH = SCRIPTS_DIR.parent / "cache" / f"H1_cutoff_sweep_{RESOLUTION}.pkl"
FIG_DIR = SCRIPTS_DIR.parent / "final_figures"


# --- Helpers ---------------------------------------------------------------
def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def _lf_rmse(a, b, sigma):
    return float(np.sqrt((gaussian_filter(a - b, sigma) ** 2).mean()))


def run_sweep(c_los, force=False):
    """Returns dict keyed by c_lo (plus 'single' and 'phimage')."""
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)-2} cached two-channel runs + single from {CACHE_PATH}")
    else:
        results = {}

    # Override the 2D pipeline itermax in place (snapshot iters max == 500).
    cm.RESOLUTION_PARAMS[RESOLUTION]['itermax'] = ITERMAX
    cm.eps_hi = cm.eps
    cm.eps_lo = EPS_LO_RATIO * cm.eps

    # Single-channel reference (independent of c_lo).
    if 'single' not in results:
        # Use the paper c_hi=4, c_lo=8 setting -- only the high band is used
        # in single-channel, so cutoffparm_lo here is a no-op for the recon.
        cm.cutoffparm = C_HI_FIXED
        cm.cutoffparm_lo = 8.0
        print("\n=== single-channel reference run ===")
        t0 = time.time()
        result = cm.run_reconstruction_for_mfact(MFACT, snapshot_iters=SNAPSHOT_ITERS)
        print(f"  elapsed {time.time()-t0:.1f}s, final RMSE {result['final_rmse_single']:.4f}")
        results['single'] = {
            'snapshots': result['snapshots_single'],
            'ierrs':     result['ierrs_single'],
            'final_rmse': result['final_rmse_single'],
        }
        results['phimage'] = result['phimage']
        os.makedirs(CACHE_PATH.parent, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(results, f)

    for c_lo in c_los:
        if c_lo in results:
            print(f"[skip] c_lo={c_lo} already cached")
            continue
        cm.cutoffparm = C_HI_FIXED
        cm.cutoffparm_lo = float(c_lo)
        cm.eps_hi = cm.eps
        cm.eps_lo = EPS_LO_RATIO * cm.eps
        print(f"\n=== two-channel: c_hi={C_HI_FIXED}, c_lo={c_lo}, eps_lo/eps_hi={EPS_LO_RATIO} ===")
        t0 = time.time()
        result = cm.run_reconstruction_for_mfact(MFACT, snapshot_iters=SNAPSHOT_ITERS)
        print(f"  elapsed {time.time()-t0:.1f}s, final RMSE {result['final_rmse_two']:.4f}")
        results[c_lo] = {
            'c_hi': C_HI_FIXED,
            'c_lo': c_lo,
            'eps_lo_ratio': EPS_LO_RATIO,
            'snapshots': result['snapshots_two'],
            'ierrs':     result['ierrs_two'],
            'final_rmse': result['final_rmse_two'],
        }
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(results, f)

    return results


# --- Figures ---------------------------------------------------------------
def _build_rows(results, c_los):
    """Return (row_label, snapshots_dict) pairs in display order."""
    rows = [("single (ref)", results['single']['snapshots'])]
    for c in c_los:
        rows.append((f"two, c_lo={c:g}", results[c]['snapshots']))
    return rows


def fig_recon_grid(results, c_los, iters, out_path, title=None):
    rows = _build_rows(results, c_los)
    phi = results['phimage']
    n = len(iters)
    fig, axes = plt.subplots(len(rows), n + 1, figsize=(1.9 * (n + 1), 1.9 * len(rows) + 0.5))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        _strip(ax)

    for r, (label, snaps) in enumerate(rows):
        axes[r, 0].imshow(phi.T, cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX, origin="lower")
        axes[r, 0].set_ylabel(label, fontsize=10)
        if r == 0:
            axes[r, 0].set_title("ground truth", fontsize=9)
        for i, it in enumerate(iters):
            col = i + 1
            img = snaps[it]
            axes[r, col].imshow(img.T, cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX, origin="lower")
            if r == 0:
                axes[r, col].set_title(f"iter {it}", fontsize=9)
            axes[r, col].set_xlabel(f"RMSE {_rmse(img, phi):.3f}", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_error_grid(results, c_los, iters, out_path, title=None, lf=False):
    rows = _build_rows(results, c_los)
    phi = results['phimage']
    n = len(iters)
    err_range = ERR_RANGE_LF if lf else ERR_RANGE_RECON
    fig, axes = plt.subplots(len(rows), n, figsize=(1.9 * n, 1.9 * len(rows) + 0.5))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        _strip(ax)

    for r, (label, snaps) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=10)
        for i, it in enumerate(iters):
            img = snaps[it]
            diff = gaussian_filter(img - phi, LF_SIGMA) if lf else (img - phi)
            axes[r, i].imshow(diff.T, cmap="gray", vmin=-err_range, vmax=err_range, origin="lower")
            if r == 0:
                axes[r, i].set_title(f"iter {it}", fontsize=9)
            metric = _lf_rmse(img, phi, LF_SIGMA) if lf else _rmse(img, phi)
            axes[r, i].set_xlabel(f"{'LF ' if lf else ''}RMSE {metric:.3f}", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_convergence(results, c_los, out_path, title=None, xlim=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ierrs_s = results['single']['ierrs']
    iters = np.arange(1, len(ierrs_s) + 1)
    ax.semilogy(iters, ierrs_s, "k-", lw=1.4, label="single (ref)")
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(c_los):
        ier = results[c]['ierrs']
        color = cmap(i / max(1, len(c_los) - 1))
        ax.semilogy(np.arange(1, len(ier) + 1), ier, "-", color=color, lw=1.2,
                    label=f"two, c_lo={c:g}")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    if xlim is not None:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def write_summary(results, c_los, iters, path):
    phi = results['phimage']
    lines = [
        f"H1 cutoff sweep -- 2D paper-breast {RESOLUTION}x{RESOLUTION}",
        f"c_hi fixed at {C_HI_FIXED}, eps_lo/eps_hi = {EPS_LO_RATIO}",
        f"itermax = {ITERMAX}",
        "=" * 72, "",
        f"{'config':<22} | " + " | ".join(f"iter {it:>4}" for it in iters) + " | LF RMSE @ iter 200",
        "-" * 72,
    ]
    snaps_s = results['single']['snapshots']
    rmse_s = [_rmse(snaps_s[it], phi) for it in iters]
    lf_s = _lf_rmse(snaps_s[200], phi, LF_SIGMA) if 200 in iters else float('nan')
    lines.append(f"{'single (ref)':<22} | " + " | ".join(f"{r:.4f}    " for r in rmse_s) + f" | {lf_s:.4f}")
    for c in c_los:
        snaps_t = results[c]['snapshots']
        rmse_t = [_rmse(snaps_t[it], phi) for it in iters]
        lf_t = _lf_rmse(snaps_t[200], phi, LF_SIGMA) if 200 in iters else float('nan')
        lines.append(f"{'two c_lo='+str(c):<22} | " + " | ".join(f"{r:.4f}    " for r in rmse_t) + f" | {lf_t:.4f}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="2D non-PoU cutoff sweep (H1)")
    parser.add_argument("--c-los", type=float, nargs="+", default=C_LO_DEFAULTS,
                        help="low-pass cutoff c_lo values (c_hi stays at %g)" % C_HI_FIXED)
    parser.add_argument("--force", action="store_true",
                        help="Recompute all configurations from scratch")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = run_sweep(args.c_los, force=args.force)

    fig_recon_grid(
        results, args.c_los, SNAPSHOT_ITERS,
        FIG_DIR / f"H1_cutoff_recon_{RESOLUTION}.png",
        title=f"Non-PoU cutoff sweep (c_hi={C_HI_FIXED}, vary c_lo) -- "
              f"reconstructions, {RESOLUTION}x{RESOLUTION}",
    )
    fig_error_grid(
        results, args.c_los, SNAPSHOT_ITERS,
        FIG_DIR / f"H1_cutoff_error_{RESOLUTION}.png",
        title=f"Non-PoU cutoff sweep -- error maps (recon - truth), gray +-{ERR_RANGE_RECON}",
        lf=False,
    )
    fig_error_grid(
        results, args.c_los, SNAPSHOT_ITERS,
        FIG_DIR / f"H1_cutoff_lferror_{RESOLUTION}.png",
        title=f"Non-PoU cutoff sweep -- LF error (Gaussian sigma={LF_SIGMA} px), gray +-{ERR_RANGE_LF}",
        lf=True,
    )
    fig_convergence(
        results, args.c_los,
        FIG_DIR / f"H1_cutoff_convergence_{RESOLUTION}.png",
        title=f"Non-PoU cutoff sweep -- image RMSE vs iter, {RESOLUTION}x{RESOLUTION}",
    )
    write_summary(results, args.c_los, SNAPSHOT_ITERS,
                  FIG_DIR / f"H1_cutoff_summary_{RESOLUTION}.txt")


if __name__ == "__main__":
    main()
