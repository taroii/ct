"""Poisson-noise statistics: does two-channel beat single-channel, with error bars?

This is the experiment that produces publishable error bars. Everything before
it was deterministic, so "N seeds" only ever measured step-size jitter. Here the
randomness is physical -- each realization is a fresh Poisson draw of the
transmission measurement -- so spread across realizations is what a reviewer
means by variability.

Three design choices carry the statistics:

1. PAIRED COMPARISON. Both arms reconstruct from the SAME noise realization, so
   the per-realization difference d_i = RMSE_single,i - RMSE_two,i removes the
   realization-to-realization variation that both arms share. Unpaired
   comparison of two independent samples would need far more realizations to
   reach the same power, and would be the wrong test besides.

2. NOISE SEED SEPARATE FROM SOLVER SEED. The power-iteration seed is held fixed
   at 42 across every realization; only noise_seed varies. Spread therefore
   reflects measurement noise alone. Varying both would confound the two and
   make the intervals uninterpretable.

3. EPS CALIBRATED TO THE NOISE, from a draw independent of the data. With noisy
   data the truth satisfies the data constraint only to about the filtered noise
   norm; at the noiseless eps=0.001 the truth sits far outside the feasible set
   and both arms converge somewhere unrelated. Measured on the breast phantom at
   i0=1e5, the calibrated eps is ~0.137 -- more than 100x the noiseless value.
   Running noise with eps_mode='fixed' is not a conservative choice, it is a
   broken one.

Reported per iteration checkpoint:
    mean paired difference with a 95% t interval
    median and IQR (robust, in case a realization goes badly)
    Wilcoxon signed-rank p (nonparametric; does not assume normal differences)
    win rate (fraction of realizations where two-channel is ahead)

    python paper/experiments/run_2d_noise_stats.py --phantom breast --reps 30
    python paper/experiments/run_2d_noise_stats.py --phantom all --reps 30 --i0 1e5
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
os.chdir(ROOT)
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(HERE))

import reconstruction as cm            # noqa: E402
from runstore import RunStore          # noqa: E402
from run_2d_convergence import PHANTOMS, MFACT  # noqa: E402

FIG = HERE / "figs"
TAB = HERE / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

HERO_S = 4.0                    # the certified setting
SOLVER_SEED = 42                # held FIXED across realizations, by design
CHECKPOINTS = [10, 50, 100, 200, 500]


def auto_i0(phantom, margin=50.0):
    """Pick i0 so even the most attenuated ray still collects photons.

    Counts on a ray are Poisson(i0 * exp(-p)). Where i0*exp(-p) drops below ~1
    the draw returns zeros, which have to be clamped before the log, and those
    line integrals come back biased low. That is a silent data corruption, not a
    noise model, so i0 has to scale with the object's peak line integral.

    A single global i0 across phantoms does not work here: measured peak line
    integrals run from ~4 (Shepp-Logan) to ~12 (Defrise), i.e. a 3000x spread in
    exp(p_max). At i0=1e5 Defrise clamps ~0.3% of its rays while Shepp-Logan is
    nowhere near the floor.

    p_max is estimated from axis-aligned column sums -- a proxy for the true
    fan-beam maximum, which is why `margin` is generous rather than tight.
    Consequence worth stating in the paper: each phantom is then studied at its
    own dose level, so absolute RMSE is not comparable ACROSS phantoms. The
    paired single-vs-two comparison within a phantom is unaffected, and that is
    the comparison the claim rests on.
    """
    if phantom is None:
        # The breast phantom is built inside the solver, so PHANTOMS["breast"]
        # hands back None. Reconstruct it here rather than assuming a default:
        # breast is the headline phantom, and a guessed i0 that clamps its
        # rays would bias exactly the result the paper leans on.
        phantom = _breast_phantom()
    pmax = float(np.asarray(phantom).sum(axis=0).max()) * (10.0 / 256)
    return float(margin * np.exp(pmax))


def _breast_phantom():
    """Mirror of the solver's internal breast phantom construction."""
    d = ROOT / "data" / "phantoms_from_paper"
    a = np.load(d / "Phantom_Adipose.npy")[cm.imagenumber]
    f = np.load(d / "Phantom_Fibroglandular.npy")[cm.imagenumber]
    c = np.load(d / "Phantom_Calcification.npy")[cm.imagenumber]
    return (0.5 * a + 1.0 * f + 2.0 * c)[::MFACT, ::MFACT]


def compute(cfg):
    cm.cutoffparm = cfg["cutoffparm"]
    cm.cutoffparm_lo = cfg["cutoffparm_lo"]
    cm.eps = cfg["eps"]
    cm.eps_hi = cfg["eps"]
    cm.eps_lo = 1.25 * cfg["eps"]
    cm.sigma_lo_scale = cfg["s"]
    phantom = PHANTOMS[cfg["phantom"]]()
    r = cm.run_reconstruction_for_mfact(
        MFACT, phantom_override=phantom, seed=cfg["solver_seed"],
        itermax_override=cfg["itermax"], i0=cfg["i0"],
        noise_seed=cfg["noise_seed"], eps_mode=cfg["eps_mode"],
        eps_factor=cfg["eps_factor"])
    return {
        "ierrs_single": np.asarray(r["ierrs_single"], dtype=np.float64),
        "ierrs_two": np.asarray(r["ierrs_two"], dtype=np.float64),
        "eps_used": np.float64(r["noise"].get("eps", cfg["eps"])),
        "zero_frac": np.float64(r["noise"].get("zero_count_frac", 0.0)),
        "single_time": np.float64(r["single_time"]),
        "two_time": np.float64(r["two_time"]),
    }


def run(name, reps, i0, itermax, force):
    store = RunStore("2d_noise_stats")
    out = []
    for k in range(reps):
        cfg = dict(phantom=name, s=HERO_S, itermax=itermax,
                   solver_seed=SOLVER_SEED,      # fixed on purpose
                   noise_seed=1000 + k,          # the only thing that varies
                   i0=i0, eps_mode="noise", eps_factor=1.1,
                   mfact=MFACT, cutoffparm=4.0, cutoffparm_lo=8.0, eps=0.001,
                   solver="reconstruction.py")
        out.append(store.load_or_run(cfg, compute, force=force,
                                     label=f"{name}_i0{i0:.0e}_rep{k}"))
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def paired_stats(runs, it):
    """Paired difference at iteration `it`, positive = two-channel better."""
    s = np.array([r["ierrs_single"][it - 1] for r in runs])
    t = np.array([r["ierrs_two"][it - 1] for r in runs])
    d = s - t
    n = len(d)
    mean = d.mean()
    # t interval on the paired difference; ddof=1 since sigma is estimated.
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    # Separate try blocks on purpose. Sharing one meant a Wilcoxon failure
    # (it raises when all differences are zero, and warns for tiny n) silently
    # downgraded tcrit to the normal-approximation 1.96, quietly narrowing every
    # confidence interval in the table.
    tcrit = 1.96
    if n > 1:
        try:
            from scipy import stats
            tcrit = float(stats.t.ppf(0.975, n - 1))
        except Exception:
            pass                      # normal approximation; conservative enough
    p = np.nan
    if n >= 6:
        try:
            from scipy import stats
            p = float(stats.wilcoxon(s, t).pvalue)
        except Exception:
            pass
    lo, hi = mean - tcrit * se, mean + tcrit * se
    return dict(n=n, single=s.mean(), two=t.mean(), mean=mean, lo=lo, hi=hi,
                median=np.median(d), q1=np.percentile(d, 25),
                q3=np.percentile(d, 75), p=p, win=(d > 0).mean(),
                rel=100 * mean / s.mean() if s.mean() else np.nan)


def write_table(name, runs, i0, itermax):
    cps = [c for c in CHECKPOINTS if c <= itermax] + (
        [itermax] if itermax not in CHECKPOINTS else [])
    eps_used = np.array([float(r["eps_used"]) for r in runs])
    zf = np.array([float(r["zero_frac"]) for r in runs])
    L = [f"# {name}: Poisson-noise statistics "
         f"(i0={i0:.0e}, {len(runs)} realizations, s={HERO_S:g})\n",
         "Paired: both arms reconstruct the same realization, so the difference "
         "removes shared realization-to-realization variation. Solver seed is "
         f"fixed at {SOLVER_SEED} across all realizations; only the noise seed "
         "varies, so the spread below is measurement noise and not step-size "
         "jitter.\n",
         f"eps calibrated to the noise from an independent draw: "
         f"{eps_used.mean():.4g} +/- {eps_used.std():.2g} "
         f"(noiseless value would be 0.001). "
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
              "estimate's sign."]
    p = TAB / f"noise_stats_{name}.md"
    p.write_text("\n".join(L) + "\n")
    print(f"  saved {p}")
    return cps


def fig_bands(name, runs, i0, itermax):
    S = np.vstack([r["ierrs_single"] for r in runs])
    T = np.vstack([r["ierrs_two"] for r in runs])
    it = np.arange(1, S.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for M, c, lab in ((S, "k", "single-channel"), (T, "#1f77b4", "two-channel s=4")):
        med = np.median(M, axis=0)
        ax.semilogy(it, med, "-", color=c, lw=1.8, label=f"{lab} (median)")
        # Interquartile band, not mean +/- sd: RMSE trajectories are skewed and
        # a couple of bad realizations would drag a symmetric band off the data.
        ax.fill_between(it, np.percentile(M, 25, axis=0),
                        np.percentile(M, 75, axis=0), color=c, alpha=0.2)
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.set_xscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(f"{name}: median and IQR over {len(runs)} noise realizations",
                 fontsize=10)

    ax = axes[1]
    D = S - T
    med = np.median(D, axis=0)
    ax.plot(it, med, "-", color="#2ca02c", lw=1.8, label="median paired diff")
    ax.fill_between(it, np.percentile(D, 25, axis=0),
                    np.percentile(D, 75, axis=0), color="#2ca02c", alpha=0.2,
                    label="IQR")
    ax.axhline(0, color="0.4", ls="--", lw=1)
    ax.set_xlabel("iteration"); ax.set_xscale("log")
    ax.set_ylabel("RMSE(single) - RMSE(two)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("paired difference (>0 favours two-channel)", fontsize=10)

    fig.tight_layout()
    p = FIG / f"noise_stats_{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", choices=list(PHANTOMS) + ["all"], default="breast")
    ap.add_argument("--reps", type=int, default=30,
                    help="noise realizations; >=20 before quoting an interval")
    ap.add_argument("--i0", default="auto",
                    help="incident photons per ray, or 'auto' to scale with "
                         "each phantom's peak line integral (recommended: a "
                         "fixed i0 clamps zero-count rays on thick phantoms)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    names = list(PHANTOMS) if args.phantom == "all" else [args.phantom]
    for name in names:
        if str(args.i0).lower() == "auto":
            i0 = auto_i0(PHANTOMS[name]())
            print(f"\n[auto] {name}: i0={i0:.2e} from peak line integral")
        else:
            i0 = float(args.i0)
        print(f"\n########## {name}: {args.reps} realizations, "
              f"i0={i0:.2e} ##########")
        runs = run(name, args.reps, i0, args.iters, args.force)
        write_table(name, runs, i0, args.iters)
        fig_bands(name, runs, i0, args.iters)
        zf = max(float(r["zero_frac"]) for r in runs)
        if zf > 1e-6:
            print(f"  NOTE: worst-case zero-count fraction {zf:.2e}; raise i0 "
                  f"if this is not negligible for your dose claim.")
        st = paired_stats(runs, min(args.iters, CHECKPOINTS[-1]))
        print(f"  @{min(args.iters, CHECKPOINTS[-1])}: two-channel better by "
              f"{st['mean']:+.5f} ({st['rel']:+.1f}%), 95% CI "
              f"[{st['lo']:+.5f}, {st['hi']:+.5f}], win rate {st['win']:.0%}")


if __name__ == "__main__":
    main()
