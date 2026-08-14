"""2D long-run convergence study: do the two-channel curves meet?

The claim being tested is a *speed* claim, and it only holds up if the
compared runs solve the same problem. They do, within the two-channel family:
s in {1, 4, 8} share tolerances (eps_hi = eps, eps_lo = 1.25 eps), share the
objective, and differ only in step sizes. Same constraint set + same objective
=> same saddle point => the three RMSE curves MUST meet. At 500 iterations
they have not (on Defrise, s=1 sits near 0.055 while s=4/8 are near 0.040 and
still descending), so the run is simply too short to show it.

Running to 5000 turns "we have lower RMSE at iteration 500" -- which invites
"then pick a different budget" -- into "we reach the same solution N times
faster", which is the actual claim and is much harder to attack.

IMPORTANT -- single-channel is NOT part of that collapse. It uses one
constraint ||F_s r|| <= eps*sqrt(m) with F_s = sqrt(ramp), whereas two-channel
uses F_hi = sqrt(ramp*(1-han_4)) and F_lo = sqrt(ramp*han_8). Since
cutoffparm_lo=8 is a NARROWER window than cutoffparm=4, han_8 is contained in
han_4 and F_hi^2 + F_lo^2 = ramp*(1 - han_4 + han_8) != F_s^2. The feasible
sets genuinely differ, so single-channel converges somewhere else. Expect a
three-curve collapse plus a separate single-channel plateau. If single also
happened to land on the same value that would be luck, not theory.

Two things are therefore measured here:

  1. Collapse.  max spread between the s=1,4,8 curves, as a function of
     iteration. Should decay toward 0. If it plateaus, something is wrong --
     most likely s=8 (uncertified, tau*lambda_max(M) ~ 1.18) not actually
     converging, which would itself be a finding worth reporting.

  2. Speedup.  iterations for each s to come within a tolerance of the shared
     limit f*, taken as the final s=1 iterate (the most conservative choice:
     s=1 is the slowest of the three, so it is the least likely to be
     mistaken for converged). Reported as an iteration ratio AND a wall-clock
     ratio, since two-channel does two filter applications per iteration and
     a per-iteration comparison flatters it.

Distance to f* is the honest convergence measure here; RMSE against ground
truth plateaus at a nonzero value and need not be monotone.

Everything is persisted via runstore, keyed by config hash -- reruns with an
unchanged config are free, and a methodology change gets a new key rather than
silently overwriting.

    python paper/experiments/run_2d_longrun.py --phantom defrise --iters 5000
    python paper/experiments/run_2d_longrun.py --phantom all --iters 5000
"""
import argparse
import os
import sys
import time
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
from run_2d_convergence import shepp_logan_2d, PHANTOMS, MFACT  # noqa: E402

FIG = HERE / "figs"
TAB = HERE / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SCALES = [1.0, 4.0, 8.0]
SCALE_COLOR = {1.0: "#7f7f7f", 4.0: "#1f77b4", 8.0: "#d62728"}
REF_S = 4.0            # certified setting used for the external reference run
# Distance-to-limit thresholds at which "converged" is declared, as a fraction
# of the initial distance. Several, because a single threshold is easy to
# cherry-pick and reviewers will assume you did.
THRESHOLDS = [0.5, 0.2, 0.1, 0.05, 0.02]


def _snap_iters(itermax, n=40):
    """Log-spaced snapshot grid.

    Threshold crossings are read off this grid, so it has to be dense enough to
    resolve them -- a coarse decade grid would quantise the speedup ratio to
    factors of 10. 40 log-spaced points on a 256^2 image is ~10 MB per arm,
    which is nothing next to the cost of recomputing the run.
    """
    g = np.unique(np.round(np.geomspace(1, itermax, n)).astype(int))
    return sorted(set(g.tolist()) | {itermax})


def compute(cfg):
    """One (phantom, s) run. Returns arrays only -- config lives in the key."""
    cm.cutoffparm = cfg["cutoffparm"]
    cm.cutoffparm_lo = cfg["cutoffparm_lo"]
    cm.eps = cfg["eps"]
    cm.eps_hi = cfg["eps"] * cfg["eps_hi_ratio"]
    cm.eps_lo = cfg["eps"] * cfg["eps_lo_ratio"]
    cm.sigma_lo_scale = cfg["s"]

    phantom = PHANTOMS[cfg["phantom"]]()
    t0 = time.time()
    r = cm.run_reconstruction_for_mfact(
        MFACT, snapshot_iters=cfg["snapshot_iters"], phantom_override=phantom,
        seed=cfg["seed"], itermax_override=cfg["itermax"])
    wall = time.time() - t0

    out = {
        "ierrs_single": np.asarray(r["ierrs_single"], dtype=np.float64),
        "ierrs_two": np.asarray(r["ierrs_two"], dtype=np.float64),
        "derrs_single": np.asarray(r["derrs_single"], dtype=np.float64),
        "derrs_two": np.asarray(r["derrs_two"], dtype=np.float64),
        "tvs_single": np.asarray(r["tvs_single"], dtype=np.float64),
        "tvs_two": np.asarray(r["tvs_two"], dtype=np.float64),
        "recon_single": np.asarray(r["xbarim_single"], dtype=np.float32),
        "recon_two": np.asarray(r["xbarim_two"], dtype=np.float32),
        "phantom": np.asarray(r["phimage"], dtype=np.float32),
        "snaps_single": {int(k): np.asarray(v, np.float32)
                         for k, v in r["snapshots_single"].items()},
        "snaps_two": {int(k): np.asarray(v, np.float32)
                      for k, v in r["snapshots_two"].items()},
        # wall-clock, per arm, so RMSE-vs-time is available without a re-run
        "single_time": np.float64(r["single_time"]),
        "two_time": np.float64(r["two_time"]),
        "wall_total": np.float64(wall),
        # step sizes actually used -- lets a later run verify the norms match
        "totalnorm_single": np.float64(r["totalnorm_single"]),
        "totalnorm_two": np.float64(r["totalnorm_two"]),
        "sig_single": np.float64(r["sig_single"]),
        "tau_single": np.float64(r["tau_single"]),
        "sig_hi": np.float64(r["sig_hi"]),
        "sig_lo": np.float64(r["sig_lo"]),
        "tau_two": np.float64(r["tau_two"]),
    }
    return out


def _cfg(name, s, itermax, seed):
    return dict(phantom=name, s=s, itermax=itermax, seed=seed,
                mfact=MFACT, cutoffparm=4.0, cutoffparm_lo=8.0,
                eps=0.001, eps_hi_ratio=1.0, eps_lo_ratio=1.25,
                snapshot_iters=_snap_iters(itermax), solver="reconstruction.py")


def run_phantom(name, itermax, ref_iters, seed, force):
    """Three study arms plus one long reference run used to define f*.

    The reference must be external to the arms being compared. Taking f* as
    some arm's own final iterate drives that arm's distance curve to exactly
    zero at the end, which biases its threshold crossings. Running s=4 for
    ref_iters >> itermax and only measuring over the first itermax iterations
    keeps that contamination negligible for every arm.
    """
    if ref_iters <= itermax and REF_S in SCALES:
        # The reference config is then byte-identical to the s=REF_S arm, so it
        # resolves to the same stored run and f* becomes that arm's own final
        # iterate -- driving its distance curve to exactly zero at the end and
        # flattering its threshold crossings. Cheap and fine for a smoke test,
        # not something to read speedup numbers off.
        print(f"  WARNING: ref_iters={ref_iters} <= iters={itermax}, so f* is "
              f"the s={REF_S:g} arm's own endpoint. Threshold crossings for "
              f"that arm are biased optimistic. Use ref_iters >> iters "
              f"(default 4x) for numbers you intend to quote.")
    store = RunStore("2d_longrun")
    res = {}
    for s in SCALES:
        res[s] = store.load_or_run(_cfg(name, s, itermax, seed), compute,
                                   force=force, label=f"{name}_s{s:g}")
    ref = store.load_or_run(_cfg(name, REF_S, ref_iters, seed), compute,
                            force=force, label=f"{name}_ref_s{REF_S:g}_{ref_iters}")
    return res, ref


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def collapse_metrics(res):
    """Spread across the three two-channel curves vs iteration."""
    curves = np.vstack([res[s]["ierrs_two"] for s in SCALES])
    spread = curves.max(axis=0) - curves.min(axis=0)
    rel = spread / np.maximum(curves.mean(axis=0), 1e-12)
    return spread, rel


def distance_to_limit(res, ref):
    """||f^n - f*|| per arm, evaluated on the snapshot grid.

    f* is the final iterate of the long external reference run, so no arm is
    advantaged. Note single-channel is measured against the same f* only to
    show that it settles somewhere else -- its own limit is a different point,
    since it enforces a different constraint set.
    """
    fstar = ref["recon_two"].astype(np.float64)
    out = {}
    for s in SCALES:
        snaps = res[s]["snaps_two"]
        its = sorted(snaps)
        out[("two", s)] = (
            np.array(its),
            np.array([float(np.sqrt(((snaps[i].astype(np.float64) - fstar) ** 2).mean()))
                      for i in its]))
    snaps = res[SCALES[0]]["snaps_single"]
    its = sorted(snaps)
    out[("single", None)] = (
        np.array(its),
        np.array([float(np.sqrt(((snaps[i].astype(np.float64) - fstar) ** 2).mean()))
                  for i in its]))
    return out


def iters_to_threshold(its, dist, frac):
    """First snapshot iteration at which distance falls below frac * d[0]."""
    if len(dist) == 0:
        return None
    target = frac * dist[0]
    below = np.nonzero(dist <= target)[0]
    return int(its[below[0]]) if len(below) else None


# --------------------------------------------------------------------------
# Figures and tables
# --------------------------------------------------------------------------
def fig_collapse(name, res, itermax):
    spread, rel = collapse_metrics(res)
    it = np.arange(1, len(spread) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.semilogy(it, res[SCALES[0]]["ierrs_single"], "k-", lw=2.0,
                label="single-channel (different constraint set)")
    for s in SCALES:
        lab = {1.0: "two-channel  s=1", 4.0: "two-channel  s=4 (certified)",
               8.0: "two-channel  s=8 (uncertified)"}[s]
        ax.semilogy(it, res[s]["ierrs_two"], "-", color=SCALE_COLOR[s],
                    lw=1.6, label=lab)
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"{name}: RMSE to {itermax} iterations", fontsize=10)

    ax = axes[1]
    ax.loglog(it, np.maximum(rel, 1e-12), "b-", lw=1.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative spread across s=1,4,8")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("two-channel curves must collapse (same saddle point)",
                 fontsize=10)

    fig.tight_layout()
    p = FIG / f"longrun_{name}_collapse.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


def fig_distance(name, res, ref):
    d = distance_to_limit(res, ref)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for s in SCALES:
        its, dist = d[("two", s)]
        ax.loglog(its, np.maximum(dist, 1e-12), "o-", color=SCALE_COLOR[s],
                  lw=1.8, ms=4, label=f"two-channel s={s:g}")
    its, dist = d[("single", None)]
    ax.loglog(its, np.maximum(dist, 1e-12), "ks--", lw=1.5, ms=4,
              label="single-channel (other saddle point)")
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\|f^n - f^*\|$ (RMS)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"{name}: distance to shared limit", fontsize=10)

    # Wall-clock axis: two-channel pays two filter applications per iteration,
    # so the iteration axis alone overstates the gain.
    ax = axes[1]
    for s in SCALES:
        its, dist = d[("two", s)]
        n = len(res[s]["ierrs_two"])
        spi = float(res[s]["two_time"]) / max(n, 1)
        ax.loglog(its * spi, np.maximum(dist, 1e-12), "o-",
                  color=SCALE_COLOR[s], lw=1.8, ms=4, label=f"two-channel s={s:g}")
    its, dist = d[("single", None)]
    spi = float(res[SCALES[0]]["single_time"]) / max(
        len(res[SCALES[0]]["ierrs_single"]), 1)
    ax.loglog(its * spi, np.maximum(dist, 1e-12), "ks--", lw=1.5, ms=4,
              label="single-channel")
    ax.set_xlabel("wall-clock (s)"); ax.set_ylabel(r"$\|f^n - f^*\|$ (RMS)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("same, on a wall-clock axis", fontsize=10)

    fig.tight_layout()
    p = FIG / f"longrun_{name}_distance.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


def write_table(name, res, ref, itermax, ref_iters):
    spread, rel = collapse_metrics(res)
    d = distance_to_limit(res, ref)
    L = [f"# {name}: long-run convergence ({itermax} iterations, "
         f"256, 25 views / 50 deg)\n",
         "## Do the two-channel curves meet?\n",
         "s=1,4,8 share tolerances and objective, so they share a saddle point "
         "and must converge to the same RMSE. Single-channel uses a different "
         "constraint set (F_hi^2+F_lo^2 != F_s^2) and is NOT expected to join.\n",
         "| iter | s=1 | s=4 | s=8 | spread | rel. spread |", "|---|---|---|---|---|---|"]
    marks = sorted(set([i for i in (100, 500, 1000, 2000, 3000, 4000, 5000)
                        if i <= itermax] + [itermax]))
    for i in marks:
        vals = [res[s]["ierrs_two"][i - 1] for s in SCALES]
        L.append(f"| {i} | " + " | ".join(f"{v:.5f}" for v in vals)
                 + f" | {spread[i-1]:.2e} | {rel[i-1]:.2%} |")

    L += ["", "## Iterations to reach a given distance from the shared limit\n",
          f"f* = final iterate of an external reference run (s={REF_S:g}, "
          f"{ref_iters} iterations), so no compared arm defines its own target. "
          "Fractions are of each arm's initial distance; all arms start from "
          "zero, so the denominators are comparable.\n",
          "| threshold | " + " | ".join(f"two s={s:g}" for s in SCALES)
          + " | single | speedup s=4 vs s=1 |",
          "|---|" + "---|" * (len(SCALES) + 2)]
    for frac in THRESHOLDS:
        cells = []
        for s in SCALES:
            its, dist = d[("two", s)]
            cells.append(iters_to_threshold(its, dist, frac))
        its, dist = d[("single", None)]
        sing = iters_to_threshold(its, dist, frac)
        sp = (f"{cells[0]/cells[1]:.1f}x"
              if cells[0] and cells[1] else "n/a")
        L.append(f"| {frac:.0%} | "
                 + " | ".join(str(c) if c else ">max" for c in cells)
                 + f" | {sing if sing else '>max'} | {sp} |")

    L += ["", "## Cost per iteration\n",
          "| arm | wall-clock (s) | s/iter |", "|---|---|---|"]
    n = len(res[SCALES[0]]["ierrs_single"])
    L.append(f"| single | {float(res[SCALES[0]]['single_time']):.1f} | "
             f"{float(res[SCALES[0]]['single_time'])/max(n,1):.4f} |")
    for s in SCALES:
        n = len(res[s]["ierrs_two"])
        L.append(f"| two s={s:g} | {float(res[s]['two_time']):.1f} | "
                 f"{float(res[s]['two_time'])/max(n,1):.4f} |")

    p = TAB / f"longrun_{name}.md"
    p.write_text("\n".join(L) + "\n")
    print(f"  saved {p}")

    msg = f"  collapse check: relative spread at iter {itermax} = {rel[-1]:.3%}"
    if itermax >= 500:
        msg += f"  (was {rel[499]:.3%} at iter 500)"
    print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", choices=list(PHANTOMS) + ["all"],
                    default="defrise",
                    help="defrise shows the largest gap at iter 500")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--ref-iters", type=int, default=None,
                    help="length of the external reference run defining f* "
                         "(default 4x --iters)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a stored run matches the config")
    args = ap.parse_args()

    ref_iters = args.ref_iters or 4 * args.iters
    names = list(PHANTOMS) if args.phantom == "all" else [args.phantom]
    for name in names:
        print(f"\n########## {name} ({args.iters} iters, "
              f"ref {ref_iters}) ##########")
        res, ref = run_phantom(name, args.iters, ref_iters, args.seed, args.force)
        fig_collapse(name, res, args.iters)
        fig_distance(name, res, ref)
        write_table(name, res, ref, args.iters, ref_iters)


if __name__ == "__main__":
    main()
