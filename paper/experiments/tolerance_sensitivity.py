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

    python paper/experiments/tolerance_sensitivity.py --phantom breast
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

import reconstruction as cm                             # noqa: E402  THE solver
from run_2d_convergence import PHANTOMS as _REG, MFACT  # noqa: E402
from run_2d_noise_stats import _breast_phantom, auto_i0  # noqa: E402
from runstore import RunStore                           # noqa: E402

FIG = HERE / "figs"
TAB = HERE / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

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


def run(name, itermax, force):
    store = RunStore("2d_tolerance_sensitivity")
    i0 = auto_i0(build(name) if _REG[name]() is not None else None)
    out = []
    for m in MULTS:
        cfg = dict(phantom=name, s=HERO_S, itermax=itermax,
                   solver_seed=SOLVER_SEED, noise_seed=NOISE_SEED, i0=i0,
                   eps_mode="noise_per_band", eps_factor=1.1,
                   eps_cal_draws=20, eps_cal_k=4.0, eps_lo_mult=m,
                   mfact=MFACT, solver="reconstruction.py")
        out.append((m, store.load_or_run(cfg, compute, force=force,
                                         label=f"{name}_lomult{m:g}")))
    return i0, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phantom", default="breast")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    i0, res = run(args.phantom, args.iters, args.force)
    it = min(args.iters, 500) - 1
    single = float(res[0][1]["ierrs_single"][it])      # independent of the mult
    mults = np.array([m for m, _ in res])
    two = np.array([float(r["ierrs_two"][it]) for _, r in res])
    lo = np.array([float(r["eps_lo_used"]) for _, r in res])

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.axhline(single, color="k", ls="--", lw=1.6, label="single-channel")
    ax.semilogx(mults, two, "o-", color="#1f77b4", lw=1.8, ms=5,
                label=f"two-channel $s={HERO_S:g}$")
    rule = two[list(mults).index(1.0)]
    ax.plot([1.0], [rule], "o", ms=12, mfc="none", mec="crimson", mew=2,
            label="per-band rule (all reported runs)")
    win = mults[two < single]
    if len(win):
        ax.axvspan(win.min(), win.max(), color="#1f77b4", alpha=0.07)
    ax.set_xlabel("low-band tolerance, multiple of the per-band rule")
    ax.set_ylabel(f"image RMSE at iteration {it+1}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(f"{args.phantom}: robustness to the low-band tolerance",
                 fontsize=10)
    fig.tight_layout()
    p = FIG / f"tolerance_sensitivity_{args.phantom}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"saved {p}")

    L = [f"# {args.phantom}: robustness to the low-band tolerance\n",
         f"Poisson noise, $i_0={i0:.3g}$, noise seed {NOISE_SEED}, "
         f"$s={HERO_S:g}$, image RMSE at iteration {it+1}. The multiplier "
         f"scales the low-band ball away from the per-band rule; **every "
         f"result reported in this paper uses multiplier 1.0**. Single-channel "
         f"({single:.5f}) does not depend on it.\n",
         "| multiplier | eps_lo | two-channel RMSE | vs single |",
         "|---|---|---|---|"]
    for m, t, e in zip(mults, two, lo):
        mark = "  <- rule" if m == 1.0 else ""
        L.append(f"| {m:g}{mark} | {e:.5g} | {t:.5f} | "
                 f"{100*(single-t)/single:+.1f}% |")
    if len(win):
        L += ["", f"Two-channel beats single-channel across multipliers "
                  f"{win.min():g}-{win.max():g}, a factor of "
                  f"{win.max()/win.min():.0f} in the low-band ball, so the "
                  f"result does not depend on landing on the rule's value."]
    p = TAB / f"tolerance_sensitivity_{args.phantom}.md"
    p.write_text("\n".join(L) + "\n")
    print(f"saved {p}")
    print(f"single {single:.5f} | rule {rule:.5f} | "
          f"wins over multipliers {win.min():g}-{win.max():g}" if len(win) else "")


if __name__ == "__main__":
    main()
