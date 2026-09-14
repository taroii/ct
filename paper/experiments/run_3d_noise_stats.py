"""3D DBT VICTRE: paired single-vs-two statistics over Poisson realizations.

The 3D story has never had error bars. Every 3D result in the paper is a single
realization, and the 3D seed sweep varies only the power-iteration start vector,
so it measures step-size sensitivity, not measurement noise -- exactly the
distinction run_2d_noise_stats draws, and for the same reason. This is the 3D
counterpart: the solver seed is held fixed and only the noise seed varies, so
the reported spread is measurement noise.

recon3d already carries the whole noise model (poisson_sinogram, calibrate_eps,
and a noise_seed deliberately independent of seed); it was simply never exposed
by a runner. Nothing new is being modelled here -- this drives what is already
there over `--reps` realizations and pairs the arms within each one.

Both arms reconstruct the SAME realization, so the per-realization difference
removes variation the two arms share and the comparison is paired. Full
per-iteration RMSE traces are stored for both arms, so stability_metrics.py can
be pointed at this store later without recomputing anything.

Unlike the other 3D runners this one goes through runstore, so an interrupted
run resumes instead of starting over. That matters more here than anywhere else
in the pipeline: this is the longest single stage in the suite.

    python paper/experiments/run_3d_noise_stats.py --reps 10 --itermax 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
HERE = PAPER / "experiments"
sys.path.insert(0, str(HERE))

import recon3d                                        # noqa: E402
from run_3d_victre import load_phantom, VICTRE_GEOM   # noqa: E402
from runstore import RunStore                         # noqa: E402
from pairedstats import paired_stats               # noqa: E402

FIG = HERE / "figs"
TAB = HERE / "tables"

SOLVER_SEED = 42          # fixed on purpose: only the noise realization varies
HERO_S = 4.0
CHECKPOINTS = [10, 50, 100, 200, 300, 500]


def auto_i0(phantom, dx_cm, margin=50.0):
    """Pick i0 so even the most attenuated ray still collects photons.

    Same reasoning as the 2D auto_i0: counts are Poisson(i0 * exp(-p)), and
    where i0*exp(-p) falls below ~1 the draw returns zeros that must be clamped
    before the log, biasing those line integrals low. That is silent data
    corruption, not a noise model.

    p_max is estimated from axis-aligned column sums through the volume -- a
    proxy for the true cone-beam maximum, which is why `margin` is generous.
    """
    pmax = float(np.asarray(phantom).sum(axis=0).max()) * dx_cm
    return float(margin * np.exp(pmax))


def compute(cfg):
    phantom, dx_cm = load_phantom(cfg["down"])
    rc = dict(recon3d.DEFAULTS)
    rc.update(itermax=cfg["itermax"], sigma_lo_scale=cfg["s"],
              seed=cfg["solver_seed"], noise_seed=cfg["noise_seed"],
              i0=cfg["i0"], eps_mode=cfg["eps_mode"],
              eps_factor=cfg["eps_factor"], npower=cfg["npower"])
    # Same geometry as run_3d_victre, passed as a copy: reconstruct() pops
    # "orbit" from the dict it receives.
    res = recon3d.reconstruct(phantom, dx_cm, cfg=rc, geom=dict(VICTRE_GEOM))
    noise = res.get("noise", {})
    return {
        "ierrs_single": np.asarray(res["single"]["ierrs"], dtype=np.float64),
        "ierrs_two": np.asarray(res["two"]["ierrs"], dtype=np.float64),
        "eps_used": np.float64(noise.get("eps", rc["eps"])),
        "zero_frac": np.float64(noise.get("zero_count_frac", 0.0)),
    }


def run(reps, i0, itermax, down, npower, force):
    store = RunStore("3d_noise_stats")
    out = []
    for k in range(reps):
        cfg = dict(phantom="victre", down=down, s=HERO_S, itermax=itermax,
                   solver_seed=SOLVER_SEED,      # fixed on purpose
                   noise_seed=1000 + k,          # the only thing that varies
                   i0=i0, eps_mode="noise", eps_factor=1.1, npower=npower,
                   # the full geometry, not a label, so a geometry change
                   # re-keys the store instead of returning stale results
                   geom=dict(VICTRE_GEOM), solver="recon3d.py")
        out.append(store.load_or_run(cfg, compute, force=force,
                                     label=f"victre_i0{i0:.0e}_rep{k}"))
    return out


def write_table(runs, i0, itermax, down):
    cps = [c for c in CHECKPOINTS if c <= itermax]
    if itermax not in cps:
        cps.append(itermax)
    eps_used = np.array([float(r["eps_used"]) for r in runs])
    zf = np.array([float(r["zero_frac"]) for r in runs])
    n = len(runs)

    L = [f"# 3D DBT VICTRE: Poisson-noise statistics "
         f"(i0={i0:.0e}, {n} realizations, s={HERO_S:g}, down={down})\n",
         "Paired: both arms reconstruct the same realization, so the difference "
         "removes shared realization-to-realization variation. The solver seed "
         f"is fixed at {SOLVER_SEED} across all realizations and only the noise "
         "seed varies, so the spread below is measurement noise and NOT the "
         "step-size jitter that the 3D seed sweep measures. The two are "
         "different quantities and must not be reported interchangeably.\n",
         f"eps calibrated to the noise from an independent draw: "
         f"{eps_used.mean():.4g} +/- {eps_used.std():.2g}. "
         f"Zero-count rays: {zf.max():.2e} worst case.\n",
         "Positive difference = two-channel has LOWER RMSE.\n",
         "| iter | single RMSE | two RMSE | mean diff | 95% CI | median [IQR] "
         "| win rate | Wilcoxon p |",
         "|---|---|---|---|---|---|---|---|"]
    for it in cps:
        st = paired_stats(runs, it)
        L.append(
            f"| {it} | {st['single']:.5f} | {st['two']:.5f} | "
            f"{st['mean']:+.5f} ({st['rel']:+.1f}%) | "
            f"[{st['lo']:+.5f}, {st['hi']:+.5f}] | "
            f"{st['median']:+.5f} [{st['q1']:+.5f}, {st['q3']:+.5f}] | "
            f"{st['win']:.0%} | {st['p']:.2g} |")
    L += ["", "A 95% CI that excludes zero means the difference at that "
              "iteration is resolved by this many realizations; one that "
              "straddles zero means it is not, regardless of the point "
              "estimate's sign.",
          "", f"With n = {n} paired samples the smallest attainable two-sided "
              f"Wilcoxon p is {2.0 / 2.0 ** n:.1e}; a p at that floor means "
              f"only that every pair had the same sign and carries no "
              f"information beyond the win rate. GPU time caps n here well "
              f"below the 2D runs, so the interval is what to read."]
    p = TAB / "noise_stats_3d_victre.md"
    p.write_text("\n".join(L) + "\n")
    print(f"  saved {p}")


def fig_bands(runs, i0):
    S = np.vstack([r["ierrs_single"] for r in runs])
    T = np.vstack([r["ierrs_two"] for r in runs])
    it = np.arange(1, S.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for M, c, lab in ((S, "k", "single-channel"),
                      (T, "#1f77b4", f"two-channel s={HERO_S:g}")):
        ax.semilogy(it, np.median(M, axis=0), "-", color=c, lw=1.8,
                    label=f"{lab} (median)")
        # Interquartile band, not mean +/- sd: RMSE trajectories are skewed and
        # a couple of bad realizations would drag a symmetric band off the data.
        ax.fill_between(it, np.percentile(M, 25, axis=0),
                        np.percentile(M, 75, axis=0), color=c, alpha=0.2)
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.set_xscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(f"3D VICTRE: median and IQR over {len(runs)} noise "
                 f"realizations", fontsize=10)
    fig.tight_layout()
    p = FIG / "noise_stats_3d_victre.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=10,
                    help="Poisson realizations (default 10). Each is a full "
                         "single+two 3D solve, so this is the GPU cost driver.")
    ap.add_argument("--itermax", type=int, default=300)
    ap.add_argument("--down", type=int, default=4)
    # 200 matches run_3d_victre, so these statistics describe the same solver
    # the single-realization VICTRE results came from.
    ap.add_argument("--npower", type=int, default=200)
    ap.add_argument("--i0", default="auto",
                    help="'auto' scales with the phantom's peak line integral")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when runstore already has the result")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    phantom, dx_cm = load_phantom(args.down)
    i0 = auto_i0(phantom, dx_cm) if args.i0 == "auto" else float(args.i0)
    print(f"3D VICTRE noise statistics: i0={i0:.3e}, reps={args.reps}, "
          f"itermax={args.itermax}, down={args.down}")

    runs = run(args.reps, i0, args.itermax, args.down, args.npower, args.force)
    write_table(runs, i0, args.itermax, args.down)
    fig_bands(runs, i0)

    st = paired_stats(runs, min(args.itermax, CHECKPOINTS[-1]))
    print(f"  @{min(args.itermax, CHECKPOINTS[-1])}: two-channel better by "
          f"{st['mean']:+.5f} ({st['rel']:+.1f}%), win {st['win']:.0%}")


if __name__ == "__main__":
    main()
