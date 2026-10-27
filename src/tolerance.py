"""Is the result robust to the low-band tolerance, or does it need that value?

The per-band rule fixes each channel's tolerance from the noise that channel
sees, so the low-band ball is derived rather than chosen. The question a reader
is still entitled to ask is whether the method only works at that value. This
sweeps the low-band ball over two decades around the rule and reports where the
two-channel arm still beats single-channel.

The sweep multiplier is a diagnostic handle only: every reported result in the
paper uses multiplier 1.0, i.e. the rule itself.

Single source of truth: this drives `reconstruction.py`, the solver behind the
convergence and noise statistics, so these numbers compose with those tables.
The standalone implementation in dyadic_compare_methods.py is NOT used.

    python src/tolerance.py --phantom breast
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reconstruction as cm                             # noqa: E402  THE solver
from convergence2d import PHANTOMS as _REG, MFACT  # noqa: E402
from noise2d import _breast_phantom, auto_i0  # noqa: E402
from store import RunStore                           # noqa: E402
from stats import paired_stats                    # noqa: E402

from paths import FIG, TAB, ensure                         # noqa: E402
ensure()

HERO_S = 4.0
SOLVER_SEED = 42
NOISE_SEED = 1000
MULTS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 70.0]


def build(name):
    f = _REG[name]()
    return _breast_phantom() if f is None else f


def compute(cfg):
    phantom = build(cfg["phantom"])
    cm.sigma_lo_scale = cfg["s"]
    r = cm.run_reconstruction_for_mfact(
        MFACT, phantom_override=phantom, seed=cfg["solver_seed"],
        itermax_override=cfg["itermax"], i0=cfg["i0"],
        noise_seed=cfg["noise_seed"], eps_mode="noise_per_band",
        eps_factor=1.1, eps_lo_mult=cfg["eps_lo_mult"])
    return {
        "ierrs_single": np.asarray(r["ierrs_single"], dtype=np.float64),
        "ierrs_two": np.asarray(r["ierrs_two"], dtype=np.float64),
        "eps_used": np.float64(r["noise"].get("eps", np.nan)),
        "eps_lo_used": np.float64(r["noise"].get("eps_lo", np.nan)),
    }


def run(name, itermax, force, reps=1):
    """The sweep, repeated over `reps` independent noise realizations.

    The solver seed is fixed and only the noise seed varies, the same
    convention as run_2d_noise_stats, so the spread reported across seeds is
    measurement noise rather than step-size jitter. The multiplier and the
    noise seed are both part of the run key, so adding realizations extends
    the store instead of invalidating it.
    """
    store = RunStore("2d_tolerance_sensitivity")
    i0 = auto_i0(build(name) if _REG[name]() is not None else None)
    out = []
    for m in MULTS:
        reps_out = []
        for k in range(reps):
            cfg = dict(phantom=name, s=HERO_S, itermax=itermax,
                       solver_seed=SOLVER_SEED, noise_seed=NOISE_SEED + k,
                       i0=i0,
                       eps_mode="noise_per_band", eps_factor=1.1,
                       eps_cal_draws=20, eps_cal_k=4.0, eps_lo_mult=m,
                       mfact=MFACT, solver="reconstruction.py")
            reps_out.append(store.load_or_run(
                cfg, compute, force=force,
                label=f"{name}_lomult{m:g}_rep{k}"))
        out.append((m, reps_out))
    return i0, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phantom", default="breast")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--reps", type=int, default=1,
                    help="independent Poisson realizations per multiplier; "
                         "the manuscript requires at least 20")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    i0, res = run(args.phantom, args.iters, args.force, args.reps)
    it = min(args.iters, 500) - 1
    mults = np.array([m for m, _ in res])
    lo = np.array([float(rs[0]["eps_lo_used"]) for _, rs in res])
    # The single-channel arm does not see the multiplier, so its realizations
    # are taken once, from the rule's own runs, and reused for every pairing.
    sing = np.array([float(r["ierrs_single"][it])
                     for r in res[list(mults).index(1.0)][1]])
    single = float(sing.mean())
    two = np.array([np.mean([float(r["ierrs_two"][it]) for r in rs])
                    for _, rs in res])
    # Paired per multiplier: both arms see the same realization, so the
    # difference removes the variation they share.
    stats_by_mult = [paired_stats([dict(ierrs_single=r["ierrs_single"],
                                        ierrs_two=r["ierrs_two"]) for r in rs],
                                  it + 1)
                     for _, rs in res]

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.axhline(single, color="k", ls="--", lw=1.6, label="single-channel")
    ax.semilogx(mults, two, "o-", color="#1f77b4", lw=1.8, ms=5,
                label=f"two-channel ($s={HERO_S:g}$)")
    if args.reps > 1:
        # Interquartile band across realizations, matching fig_noise_grid.
        q1 = np.array([np.percentile([float(r["ierrs_two"][it]) for r in rs], 25)
                       for _, rs in res])
        q3 = np.array([np.percentile([float(r["ierrs_two"][it]) for r in rs], 75)
                       for _, rs in res])
        ax.fill_between(mults, q1, q3, color="#1f77b4", alpha=0.18, lw=0)
    rule = two[list(mults).index(1.0)]
    ax.plot([1.0], [rule], "o", ms=12, mfc="none", mec="crimson", mew=2,
            label="per-band rule (all reported runs)")
    win = mults[two < single]
    if len(win):
        ax.axvspan(win.min(), win.max(), color="#1f77b4", alpha=0.07)
    ax.set_xlabel("Low-band tolerance, multiple of the per-band rule")
    ax.set_ylabel(f"Image RMSE at iteration {it+1}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(f"{args.phantom.capitalize()}: sensitivity to the low-band "
                 f"tolerance", fontsize=10)
    fig.tight_layout()
    p = FIG / f"tolerance_sensitivity_{args.phantom}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"saved {p}")

    seeds = (f"noise seed {NOISE_SEED}" if args.reps == 1 else
             f"{args.reps} realizations, noise seeds "
             f"{NOISE_SEED}-{NOISE_SEED + args.reps - 1}")
    L = [f"# {args.phantom}: robustness to the low-band tolerance\n",
         f"Poisson noise, $i_0={i0:.3g}$, {seeds}, "
         f"$s={HERO_S:g}$, image RMSE at iteration {it+1}. The multiplier "
         f"scales the low-band ball away from the per-band rule; **every "
         f"result reported in this paper uses multiplier 1.0**. Single-channel "
         f"({single:.5f}) does not depend on it. Columns 1-3 are the mean over "
         f"realizations; the interval is on the paired difference.\n",
         "| multiplier | eps_lo | two-channel RMSE | vs single | n | "
         "95% CI on the difference | win rate |",
         "|---|---|---|---|---|---|---|"]
    for m, tv, e, st in zip(mults, two, lo, stats_by_mult):
        mark = "  <- rule" if m == 1.0 else ""
        L.append(f"| {m:g}{mark} | {e:.5g} | {tv:.5f} | "
                 f"{100*(single-tv)/single:+.1f}% | {st['n']} | "
                 f"[{st['lo']:+.5f},{st['hi']:+.5f}] | {100*st['win']:.0f}% |")
    if len(win):
        L += ["", f"Two-channel beats single-channel across multipliers "
                  f"{win.min():g}-{win.max():g}, a factor of "
                  f"{win.max()/win.min():.0f} in the low-band ball, so the "
                  f"result does not depend on landing on the rule's value."]
    p = TAB / f"tolerance_sensitivity_{args.phantom}.md"
    p.write_text("\n".join(L) + "\n")
    print(f"saved {p}")

    # The same sweep as data, for the figure that plots it. Keeping the report
    # human-readable and the figure's input machine-readable means neither
    # constrains the other.
    q = TAB / f"tolerance_sensitivity_{args.phantom}.json"
    q.write_text(json.dumps(dict(
        phantom=args.phantom, reps=args.reps, itermax=it + 1, s=HERO_S,
        i0=float(i0), noise_seed_first=NOISE_SEED,
        single=single,
        mults=[float(m) for m in mults],
        eps_lo=[float(e) for e in lo],
        two=[float(v) for v in two],
    ), indent=1) + "\n")
    print(f"saved {q}")
    if len(win):
        print(f"single {single:.5f} | rule {rule:.5f} | n={args.reps} | "
              f"wins over multipliers {win.min():g}-{win.max():g}")


if __name__ == "__main__":
    main()
