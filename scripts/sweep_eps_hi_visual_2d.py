"""
H6 -- 2D paper-breast eps_hi sweep at r = eps_lo / eps_hi = 10 fixed.

H2 swept the ratio r = eps_lo / eps_hi at eps_hi = cm.eps (the paper's
high-band radius). Here we keep r fixed at the best-found value (r=10)
and vary the overall scale: eps_hi = (eps_hi_ratio) * cm.eps, with
eps_lo = r * eps_hi.

This decouples the two degrees of freedom in the constraint pair
(eps_hi, eps_lo):
  - r controls the LF/HF balance         (H2)
  - eps_hi_ratio controls the overall scale (H6, this script)

Default eps_hi_ratios = {0.5, 1.0, 2.0, 4.0}:
  - 0.5: tighten HF (push HF data fidelity harder than the paper)
  - 1.0: paper HF scale, but with r=10 the LF is much looser than paper
         (matches the H2 r=10 entry exactly)
  - 2.0: loosen HF too -- same effect as H2 r=20 (paper scale), but
         only if both scale together; here eps_lo = 20*eps so it
         coincides with H2 r=20
  - 4.0: loosen HF further; eps_lo = 40*eps

(Strictly: H2 r=20 entry at eps_hi=eps is identical to H6 eps_hi_ratio=2
with r=10 -- the data is the same point reached via two parameterizations.
Worth confirming by inspection.)

Outputs:
  final_figures/H6_eps_hi_recon_256.png
  final_figures/H6_eps_hi_error_256.png
  final_figures/H6_eps_hi_lferror_256.png
  final_figures/H6_eps_hi_convergence_256.png
  final_figures/H6_eps_hi_summary_256.txt
  cache/H6_eps_hi_sweep_256.pkl

Run:  conda run --no-capture-output -n ct2 \
          python scripts/sweep_eps_hi_visual_2d.py [--force]
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
MFACT = 2
RESOLUTION = int(512 / MFACT)
C_HI_FIXED = 4.0
C_LO_FIXED = 8.0
R_FIXED = 10.0                                       # eps_lo / eps_hi
EPS_HI_RATIO_DEFAULTS = [0.5, 1.0, 2.0, 4.0]         # eps_hi / cm.eps
SNAPSHOT_ITERS = [50, 100, 200, 500]
ITERMAX = 500

DISPLAY_VMIN = 0.0
DISPLAY_VMAX = 1.0
ERR_RANGE_RECON = 0.30
ERR_RANGE_LF = 0.15
LF_SIGMA = 8.0

CACHE_PATH = SCRIPTS_DIR.parent / "cache" / f"H6_eps_hi_sweep_{RESOLUTION}.pkl"
FIG_DIR = SCRIPTS_DIR.parent / "final_figures"


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def _lf_rmse(a, b, sigma):
    return float(np.sqrt((gaussian_filter(a - b, sigma) ** 2).mean()))


def run_sweep(eps_hi_ratios, r_fixed, force=False):
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)-2} cached runs + single from {CACHE_PATH}")
    else:
        results = {}

    cm.RESOLUTION_PARAMS[RESOLUTION]['itermax'] = ITERMAX
    cm.cutoffparm = C_HI_FIXED
    cm.cutoffparm_lo = C_LO_FIXED

    if 'single' not in results:
        cm.eps_hi = cm.eps
        cm.eps_lo = 1.25 * cm.eps   # single-channel ignores eps_lo anyway
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

    for eps_hi_ratio in eps_hi_ratios:
        if eps_hi_ratio in results:
            print(f"[skip] eps_hi/eps={eps_hi_ratio} already cached")
            continue
        cm.eps_hi = eps_hi_ratio * cm.eps
        cm.eps_lo = r_fixed * cm.eps_hi
        print(f"\n=== two-channel: eps_hi/eps={eps_hi_ratio}, r={r_fixed} "
              f"(eps_hi={cm.eps_hi:.6f}, eps_lo={cm.eps_lo:.6f}) ===")
        t0 = time.time()
        result = cm.run_reconstruction_for_mfact(MFACT, snapshot_iters=SNAPSHOT_ITERS)
        print(f"  elapsed {time.time()-t0:.1f}s, final RMSE {result['final_rmse_two']:.4f}")
        results[eps_hi_ratio] = {
            'eps_hi_ratio': eps_hi_ratio,
            'r': r_fixed,
            'snapshots': result['snapshots_two'],
            'ierrs':     result['ierrs_two'],
            'final_rmse': result['final_rmse_two'],
        }
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(results, f)

    return results


# --- Figures (parallel to H2) ---------------------------------------------
def _build_rows(results, eps_hi_ratios):
    rows = [("single (ref)", results['single']['snapshots'])]
    for eh in eps_hi_ratios:
        rows.append((f"two, eps_hi/eps={eh:g}", results[eh]['snapshots']))
    return rows


def fig_recon_grid(results, eps_hi_ratios, iters, out_path, title=None):
    rows = _build_rows(results, eps_hi_ratios)
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


def fig_error_grid(results, eps_hi_ratios, iters, out_path, title=None, lf=False):
    rows = _build_rows(results, eps_hi_ratios)
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


def fig_convergence(results, eps_hi_ratios, out_path, title=None, xlim=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ierrs_s = results['single']['ierrs']
    ax.semilogy(np.arange(1, len(ierrs_s) + 1), ierrs_s, "k-", lw=1.4, label="single (ref)")
    cmap = plt.get_cmap("viridis")
    for i, eh in enumerate(eps_hi_ratios):
        ier = results[eh]['ierrs']
        color = cmap(i / max(1, len(eps_hi_ratios) - 1))
        ax.semilogy(np.arange(1, len(ier) + 1), ier, "-", color=color, lw=1.2,
                    label=f"two, eps_hi/eps={eh:g}")
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


def write_summary(results, eps_hi_ratios, iters, path):
    phi = results['phimage']
    lines = [
        f"H6 eps_hi sweep -- 2D paper-breast {RESOLUTION}x{RESOLUTION}",
        f"c_hi={C_HI_FIXED}, c_lo={C_LO_FIXED} fixed; r=eps_lo/eps_hi={R_FIXED} fixed; itermax={ITERMAX}",
        "=" * 72, "",
        f"{'config':<28} | " + " | ".join(f"iter {it:>4}" for it in iters) + " | LF RMSE @ iter 200",
        "-" * 72,
    ]
    snaps_s = results['single']['snapshots']
    rmse_s = [_rmse(snaps_s[it], phi) for it in iters]
    lf_s = _lf_rmse(snaps_s[200], phi, LF_SIGMA) if 200 in iters else float('nan')
    lines.append(f"{'single (ref)':<28} | " + " | ".join(f"{v:.4f}    " for v in rmse_s) + f" | {lf_s:.4f}")
    for eh in eps_hi_ratios:
        snaps_t = results[eh]['snapshots']
        rmse_t = [_rmse(snaps_t[it], phi) for it in iters]
        lf_t = _lf_rmse(snaps_t[200], phi, LF_SIGMA) if 200 in iters else float('nan')
        lines.append(f"{'two eps_hi/eps='+str(eh):<28} | " + " | ".join(f"{v:.4f}    " for v in rmse_t) + f" | {lf_t:.4f}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="2D eps_hi sweep at r fixed (H6)")
    parser.add_argument("--eps-hi-ratios", type=float, nargs="+", default=EPS_HI_RATIO_DEFAULTS,
                        help="eps_hi / cm.eps values to sweep (eps_lo set to r*eps_hi)")
    parser.add_argument("--r", type=float, default=R_FIXED,
                        help="eps_lo / eps_hi ratio held fixed (default 10)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = run_sweep(args.eps_hi_ratios, args.r, force=args.force)

    fig_recon_grid(
        results, args.eps_hi_ratios, SNAPSHOT_ITERS,
        FIG_DIR / f"H6_eps_hi_recon_{RESOLUTION}.png",
        title=f"eps_hi sweep at r={args.r} fixed (c_hi={C_HI_FIXED}, c_lo={C_LO_FIXED}) -- "
              f"reconstructions, {RESOLUTION}x{RESOLUTION}",
    )
    fig_error_grid(
        results, args.eps_hi_ratios, SNAPSHOT_ITERS,
        FIG_DIR / f"H6_eps_hi_error_{RESOLUTION}.png",
        title=f"eps_hi sweep at r={args.r} fixed -- error maps (recon - truth), gray +-{ERR_RANGE_RECON}",
        lf=False,
    )
    fig_error_grid(
        results, args.eps_hi_ratios, SNAPSHOT_ITERS,
        FIG_DIR / f"H6_eps_hi_lferror_{RESOLUTION}.png",
        title=f"eps_hi sweep -- LF error (Gaussian sigma={LF_SIGMA} px), gray +-{ERR_RANGE_LF}",
        lf=True,
    )
    fig_convergence(
        results, args.eps_hi_ratios,
        FIG_DIR / f"H6_eps_hi_convergence_{RESOLUTION}.png",
        title=f"eps_hi sweep at r={args.r} fixed -- image RMSE vs iter, {RESOLUTION}x{RESOLUTION}",
    )
    write_summary(results, args.eps_hi_ratios, SNAPSHOT_ITERS,
                  FIG_DIR / f"H6_eps_hi_summary_{RESOLUTION}.txt")


if __name__ == "__main__":
    main()
