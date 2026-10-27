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

    python src/bands.py --phantom breast
    python src/bands.py --phantom all --iters 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stability as sd                      # noqa: E402  geometry + filters
import reconstruction as cm                            # noqa: E402  THE solver
from convergence2d import PHANTOMS as _REG, MFACT  # noqa: E402
from noise2d import _breast_phantom, auto_i0  # noqa: E402
from store import RunStore                     # noqa: E402

from paths import FIG, TAB, ensure                         # noqa: E402
ensure()

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


def compute(cfg):
    """One realization, as the store calls it. Returns plain arrays.

    The band curves are what gets stored, not the iterates: the reduction
    from a 256x256 snapshot to two scalars per iteration is the whole cost
    saving, and nothing downstream wants the volumes back.
    """
    phantom = build(cfg["phantom"])
    snaps = snapshot_grid(cfg["itermax"])
    kw = dict(phantom_override=phantom, seed=cfg["solver_seed"],
              itermax_override=cfg["itermax"], snapshot_iters=snaps)
    if cfg["noisy"]:
        kw.update(i0=cfg["i0"], noise_seed=cfg["noise_seed"],
                  eps_mode=cfg["eps_mode"], eps_factor=cfg["eps_factor"])
    cm.sigma_lo_scale = cfg["s"]
    r = cm.run_reconstruction_for_mfact(MFACT, **kw)
    out = {}
    for arm in ("single", "two"):
        it, lo, hi = curves(phantom, r[f"snapshots_{arm}"])
        out[f"iters_{arm}"] = np.asarray(it, dtype=np.int32)
        out[f"lo_{arm}"] = np.asarray(lo, dtype=np.float64)
        out[f"hi_{arm}"] = np.asarray(hi, dtype=np.float64)
    return out


def run_case(name, itermax, noisy, noise_seed=NOISE_SEED, force=False):
    """One phantom, one realization; returns per-band curves for both arms.

    Goes through the run store, so a rerun of a realization already computed
    returns in milliseconds instead of ~30 s, and `audit.py` can count the
    realizations behind fig:bandresolved the same way it counts every other
    figure's.
    """
    phantom = build(name)
    cfg = dict(phantom=name, s=HERO_S, itermax=itermax, noisy=bool(noisy),
               solver_seed=SOLVER_SEED,
               noise_seed=noise_seed if noisy else None,
               i0=auto_i0(phantom) if noisy else None,
               eps_mode="noise_per_band" if noisy else None,
               eps_factor=1.1 if noisy else None,
               mfact=MFACT, solver="reconstruction.py",
               measures="per-band projection-domain relative error")
    d = RunStore("2d_band_resolved").load_or_run(
        cfg, compute, force=force,
        label=f"{name}_{'noisy' if noisy else 'clean'}_seed{noise_seed}")
    return {arm: (np.asarray(d[f"iters_{arm}"]),
                  np.asarray(d[f"lo_{arm}"]),
                  np.asarray(d[f"hi_{arm}"]))
            for arm in ("single", "two")}


def run_reps(name, itermax, noisy, reps):
    """`reps` independent realizations, stacked.

    Returns {arm: (iters, lo_stack, hi_stack)} with the stacks shaped
    (reps, n_iters), so one realization is just reps=1 and every consumer
    below reduces the same way.

    Only the noisy case is repeated: with no Poisson sampling there is no
    realization to vary, and `reps` copies of a deterministic run would be a
    spread of exactly zero reported as if it were measured.
    """
    if not noisy:
        reps = 1
    los, his, its = [], [], None
    for k in range(reps):
        if reps > 1:
            print(f"    realization {k + 1}/{reps} (seed {NOISE_SEED + k})")
        d = run_case(name, itermax, noisy, noise_seed=NOISE_SEED + k)
        for arm in ("single", "two"):
            it, lo, hi = d[arm]
            its = it
            los.append((arm, lo)); his.append((arm, hi))
    out = {}
    for arm in ("single", "two"):
        out[arm] = (its,
                    np.vstack([v for a, v in los if a == arm]),
                    np.vstack([v for a, v in his if a == arm]))
    return out


def fig_bands(name, res, itermax, reps=1):
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
            # Median and interquartile band across realizations, the same
            # summary fig_noise_grid uses; with one realization the band has
            # zero width and nothing is implied about spread.
            ax.loglog(it, np.median(lo, axis=0), "-", color=colour, lw=2.0,
                      label=f"{lab}, low band")
            ax.loglog(it, np.median(hi, axis=0), "--", color=colour, lw=1.3,
                      alpha=0.75, label=f"{lab}, high band")
            if lo.shape[0] > 1:
                for v in (lo, hi):
                    ax.fill_between(it, np.percentile(v, 25, axis=0),
                                    np.percentile(v, 75, axis=0),
                                    color=colour, alpha=0.18, lw=0)
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


def _seedline(res):
    n = res["noisy"]["single"][1].shape[0]
    return (f", noise seed {NOISE_SEED}" if n == 1 else
            f", {n} realizations over noise seeds "
            f"{NOISE_SEED}-{NOISE_SEED + n - 1}")


def write_table(name, res, itermax, reps=1):
    cps = [c for c in (10, 50, 100, 200, 500, 1000) if c <= itermax]
    L = [f"# {name}: error resolved by data channel\n",
         "Relative band error "
         "$\\|R_b X (f^n - f^\\star)\\| / \\|R_b X f^\\star\\|$, measured with the "
         "solver's own filters in the projection domain, where the channels and "
         "their tolerances are defined. Each band is normalised by its own share "
         "of the clean sinogram, so the columns are comparable despite the ramp "
         "placing most energy in the high band.\n",
         f"Single source: `reconstruction.py`, $s={HERO_S:g}$, solver seed "
         f"{SOLVER_SEED}" + (_seedline(res) if "noisy" in res else "")
         + ". Noisy entries are medians over realizations.\n"]
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
                row += [float(np.median(lo[:, j])), float(np.median(hi[:, j]))]
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
    ap.add_argument("--reps", type=int, default=1,
                    help="independent Poisson realizations for the noisy "
                         "panel; the manuscript requires at least 20. The "
                         "noiseless panel has no realization to vary.")
    ap.add_argument("--noiseless-only", action="store_true")
    args = ap.parse_args()

    names = list(_REG) if args.phantom == "all" else [args.phantom]
    for name in names:
        print(f"\n### {name}")
        res = {"noiseless": run_reps(name, args.iters, False, args.reps)}
        if not args.noiseless_only:
            res["noisy"] = run_reps(name, args.iters, True, args.reps)
        fig_bands(name, res, args.iters, args.reps)
        write_table(name, res, args.iters, args.reps)
        for key in res:
            it, lo, _ = res[key]["single"]
            _, lo2, _ = res[key]["two"]
            j = int(np.argmin(np.abs(it - min(args.iters, 500))))
            a, b_ = float(np.median(lo[:, j])), float(np.median(lo2[:, j]))
            print(f"  {key:10s} low-band error @{it[j]} "
                  f"(n={lo.shape[0]}): single {a:.4f} vs two {b_:.4f} "
                  f"({a / b_:.2f}x)")


if __name__ == "__main__":
    main()
