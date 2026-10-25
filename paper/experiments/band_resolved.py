"""Where the acceleration actually happens: error resolved by frequency band.

Every other figure in this study reports scalar image RMSE, which cannot
distinguish "the reconstruction got better" from "the LOW-FREQUENCY part got
better faster". The paper's claim is the latter -- the low band is the slow one,
and amplifying its dual step is what fixes it -- so that claim should be
measured directly rather than inferred from a total.

For each stored iterate f^n this records the error energy the two data channels
actually see,

    e_lo(n) = ||R_lo X (f^n - f*)|| / ||R_lo X f*||,
    e_hi(n) = ||R_hi X (f^n - f*)|| / ||R_hi X f*||,

using the solver's own filters and geometry (imported from
stability_diagnostic, not re-derived, so the two cannot drift apart). The
projection domain is the right place to measure: that is where the channels are
defined and where the tolerances act. Each band is normalised by its own share
of the clean sinogram, so the two curves are comparable on one axis despite the
ramp putting ~98% of the energy in the high band.

Single source of truth: this drives `reconstruction.py`, the same solver behind
the convergence and noise statistics, so its numbers compose with those tables.
The standalone multi-channel implementation in dyadic_compare_methods.py is NOT
used here.

    python paper/experiments/band_resolved.py --phantom breast
    python paper/experiments/band_resolved.py --phantom all --iters 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PAPER))

import stability_diagnostic as sd                      # noqa: E402  geometry + filters
import reconstruction as cm                            # noqa: E402  THE solver
from run_2d_convergence import PHANTOMS as _REG, MFACT  # noqa: E402
from run_2d_noise_stats import _breast_phantom, auto_i0  # noqa: E402

FIG = HERE / "figs"
TAB = HERE / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

HERO_S = 4.0
SOLVER_SEED = 42
NOISE_SEED = 1000


def build(name):
    f = _REG[name]()
    return _breast_phantom() if f is None else f


def snapshot_grid(itermax, n=28):
    """Log-spaced iterations: the decay is a power law, so a linear grid would
    spend most of its points where nothing changes."""
    g = np.unique(np.round(np.geomspace(1, itermax, n)).astype(int))
    return sorted(int(i) for i in g if 1 <= i <= itermax)


def band_norms(sino):
    """(low, high) filtered norms of a sinogram-shaped array."""
    return tuple(float(np.sqrt(max((sd.fft_weight(sino, W) * sino).sum(), 0.0)))
                 for W in (sd.W_lo2, sd.W_hi2))


def project(img):
    g = np.zeros((sd.nviews, sd.nbins))
    sd.project(np.ascontiguousarray(img * sd.mask), g)
    return g


def curves(phantom, snaps):
    """Relative per-band error for each stored iterate."""
    lo_ref, hi_ref = band_norms(project(phantom))
    its = sorted(snaps)
    lo, hi = [], []
    for i in its:
        l, h = band_norms(project(snaps[i] - phantom))
        lo.append(l / lo_ref if lo_ref > 0 else np.nan)
        hi.append(h / hi_ref if hi_ref > 0 else np.nan)
    return np.array(its), np.array(lo), np.array(hi)


def run_case(phantom, itermax, noisy):
    """One phantom, one noise setting; returns per-band curves for both arms."""
    snaps = snapshot_grid(itermax)
    kw = dict(phantom_override=phantom, seed=SOLVER_SEED,
              itermax_override=itermax, snapshot_iters=snaps)
    if noisy:
        kw.update(i0=auto_i0(phantom), noise_seed=NOISE_SEED,
                  eps_mode="noise_per_band", eps_factor=1.1)
    cm.sigma_lo_scale = HERO_S
    r = cm.run_reconstruction_for_mfact(MFACT, **kw)
    out = {}
    for arm in ("single", "two"):
        out[arm] = curves(phantom, r[f"snapshots_{arm}"])
    return out


def fig_bands(name, res, itermax):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, key, title in ((axes[0], "noiseless", "Noiseless data"),
                           (axes[1], "noisy", "Poisson noise")):
        if key not in res:
            ax.set_visible(False)
            continue
        d = res[key]
        for arm, colour, lab in (("single", "k", "single-channel"),
                                 ("two", "#1f77b4", f"two-channel ($s={HERO_S:g}$)")):
            it, lo, hi = d[arm]
            ax.loglog(it, lo, "-", color=colour, lw=2.0, label=f"{lab}, low band")
            ax.loglog(it, hi, "--", color=colour, lw=1.3, alpha=0.75,
                      label=f"{lab}, high band")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Relative band error")
    axes[0].legend(fontsize=7, loc="lower left")
    # Titles name the figure; the solid/dashed key and the reading of it
    # belong in the caption, where there is room to be precise.
    fig.suptitle(f"Band-resolved error vs iteration ({name.capitalize()})",
                 fontsize=10)
    fig.tight_layout()
    p = FIG / f"band_resolved_{name}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")


def write_table(name, res, itermax):
    cps = [c for c in (10, 50, 100, 200, 500, 1000) if c <= itermax]
    L = [f"# {name}: error resolved by data channel\n",
         "Relative band error "
         "$\\|R_b X (f^n - f^\\star)\\| / \\|R_b X f^\\star\\|$, measured with the "
         "solver's own filters in the projection domain, where the channels and "
         "their tolerances are defined. Each band is normalised by its own share "
         "of the clean sinogram, so the columns are comparable despite the ramp "
         "placing most energy in the high band.\n",
         f"Single source: `reconstruction.py`, $s={HERO_S:g}$, solver seed "
         f"{SOLVER_SEED}" + (f", noise seed {NOISE_SEED}" if "noisy" in res else "")
         + ".\n"]
    for key, title in (("noiseless", "Noiseless"),
                       ("noisy", "Poisson noise, per-band tolerances")):
        if key not in res:
            continue
        L += [f"## {title}\n",
              "| iter | single low | single high | two low | two high | "
              "low-band speedup |", "|---|---|---|---|---|---|"]
        d = res[key]
        for c in cps:
            row = []
            for arm in ("single", "two"):
                it, lo, hi = d[arm]
                j = int(np.argmin(np.abs(it - c)))
                row += [lo[j], hi[j]]
            sp = row[0] / row[2] if row[2] > 0 else np.nan
            L.append(f"| {c} | {row[0]:.4f} | {row[1]:.4f} | "
                     f"{row[2]:.4f} | {row[3]:.4f} | {sp:.2f}x |")
        L.append("")
    L.append("`low-band speedup` is the ratio of single- to two-channel "
             "low-band error at that iteration: >1 means the two-channel "
             "iterate has already removed more low-frequency error.")
    p = TAB / f"band_resolved_{name}.md"
    p.write_text("\n".join(L) + "\n")
    print(f"  saved {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phantom", default="breast",
                    help="phantom name, or 'all'")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--noiseless-only", action="store_true")
    args = ap.parse_args()

    names = list(_REG) if args.phantom == "all" else [args.phantom]
    for name in names:
        print(f"\n### {name}")
        phantom = build(name)
        res = {"noiseless": run_case(phantom, args.iters, noisy=False)}
        if not args.noiseless_only:
            res["noisy"] = run_case(phantom, args.iters, noisy=True)
        fig_bands(name, res, args.iters)
        write_table(name, res, args.iters)
        for key in res:
            it, lo, _ = res[key]["single"]
            _, lo2, _ = res[key]["two"]
            j = int(np.argmin(np.abs(it - min(args.iters, 500))))
            print(f"  {key:10s} low-band error @{it[j]}: "
                  f"single {lo[j]:.4f} vs two {lo2[j]:.4f} "
                  f"({lo[j]/lo2[j]:.2f}x)")


if __name__ == "__main__":
    main()
