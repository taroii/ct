"""Quantify the stability difference between single- and two-channel PDHG.

The noise-statistics runs already store the FULL 500-iteration RMSE trace for
both arms of every realization (run_2d_noise_stats.compute stores
`ierrs_single` / `ierrs_two` in full; CHECKPOINTS is applied only at reporting
time). So the stability argument can be built from data already on disk --
this script recomputes nothing and never touches a solver.

The claim this supports is NOT "two-channel converges monotonically". It does
not; both arms oscillate. The claim is about the ENVELOPE and the TURNAROUND:

  semiconvergence gap   rmse(final) - min_n rmse(n).  How much image quality
                        you lose by not knowing where to stop. Under noise the
                        single-channel arm walks away from the truth after its
                        minimum; this measures by how much. Lower is better.

  turnaround iteration  argmin_n rmse(n). Where the arm stops improving. An
                        arm whose minimum is the last iterate never turned
                        around within budget -- that is CENSORED, and the
                        reported value is a lower bound, not a measurement.
                        Later is better.

  oscillation band      median over the fit window of
                        (running max - running min) / running median, i.e. how
                        wide the oscillation is relative to the level it
                        oscillates about. Scale-free, and independent of which
                        way the trace is drifting. Lower is better.

                        This replaced a total-variation "excess path" metric
                        (TV / range), which was rejected on inspection: a
                        monotone trace scores 1.0 whether it is monotonically
                        improving OR monotonically worsening. Since the
                        single-channel failure mode here is smooth drift away
                        from the truth, that metric rewarded the very behaviour
                        it was meant to indict, and it further penalised
                        whichever arm had converged flattest by shrinking its
                        denominator. Amplitude and direction have to be
                        measured separately: this metric is the amplitude, the
                        envelope slope below is the direction.

  envelope slope        log-log slope of the running upper envelope over a
                        fitted window. Measures whether the oscillation is
                        decaying, and how fast, independent of its amplitude.
                        More negative is better.

Every metric is computed per realization on each arm, then differenced WITHIN a
realization (both arms saw the same Poisson draw), so the paired spread removes
variation the two arms share. Sign convention follows run_2d_noise_stats:
positive difference always means two-channel is better, whichever direction the
raw metric happens to run.

    python paper/experiments/stability_metrics.py
    python paper/experiments/stability_metrics.py --phantom breast
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "experiments" / "results" / "2d_noise_stats"
FIG = PAPER / "experiments" / "figs"
TAB = PAPER / "experiments" / "tables"

# Metric name -> (label, +1 if larger is better else -1, format spec).
# The sign is what lets every table column read "positive = two-channel better"
# without the reader having to remember which way each metric runs.
METRICS = {
    "semiconv_gap": ("semiconvergence gap", -1, ".5f"),
    "semiconv_rel": ("semiconvergence gap (% of min)", -1, ".2f"),
    "turnaround":   ("turnaround iteration", +1, ".0f"),
    "osc_band":     ("oscillation band (rel. to level)", -1, ".4f"),
    "env_slope":    ("envelope log-log slope", -1, ".4f"),
}


def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    """Group stored runs by phantom. Reads the runstore directly."""
    by_phantom: dict[str, list[dict]] = {}
    for jp in sorted(results_dir.glob("*.json")):
        meta = json.loads(jp.read_text())
        if meta.get("experiment") != "2d_noise_stats":
            continue
        npz = jp.with_suffix(".npz")
        if not npz.exists():
            print(f"  warning: {jp.name} has no sibling .npz -- skipped")
            continue
        cfg = meta["config"]
        with np.load(npz) as d:
            if "ierrs_single" not in d or "ierrs_two" not in d:
                print(f"  warning: {npz.name} has no traces -- skipped")
                continue
            run = {
                "single": np.asarray(d["ierrs_single"], dtype=float),
                "two": np.asarray(d["ierrs_two"], dtype=float),
                "noise_seed": cfg.get("noise_seed"),
                "s": cfg.get("s"),
                "i0": cfg.get("i0"),
                "itermax": cfg.get("itermax"),
                # eps_mode and its two constants are part of a run's identity:
                # the per-band and original runs differ ONLY in these, so a
                # config tuple without them silently pools two experiments.
                "eps_mode": cfg.get("eps_mode"),
                "eps_cal_draws": cfg.get("eps_cal_draws"),
                "eps_cal_k": cfg.get("eps_cal_k"),
                "label": meta.get("label", jp.stem),
            }
        by_phantom.setdefault(cfg["phantom"], []).append(run)
    _sort_runs(by_phantom)
    return by_phantom


def select_config(name: str, runs: list[dict], args) -> list[dict]:
    """Keep only runs from ONE configuration of a phantom.

    The store keys on the full config, so it can hold several experiments for
    the same phantom -- e.g. the default auto-i0 run AND a documented fixed-i0
    run. Pooling them would treat two doses as one sample: n inflates and every
    confidence interval comes out too narrow, silently. Runs of different
    lengths would also crash the figure code. So configurations are never
    pooled; if more than one survives the filters, stop and say which exist.
    """
    keep = [r for r in runs
            if (args.s is None or r["s"] == args.s)
            and (args.itermax is None or r["itermax"] == args.itermax)
            and (args.eps_mode is None or r["eps_mode"] == args.eps_mode)
            and (args.i0 is None or (r["i0"] is not None
                                     and abs(r["i0"] - args.i0) <= 1e-6 * abs(args.i0)))]

    def ident(r):
        return (r["i0"], r["itermax"], r["s"], r["eps_mode"],
                r["eps_cal_draws"], r["eps_cal_k"])

    configs = sorted({ident(r) for r in keep},
                     key=lambda c: tuple("" if v is None else str(v) for v in c))
    if len(configs) > 1:
        listing = "\n".join(f"    i0={c[0]:.6g}  itermax={c[1]}  s={c[2]}  "
                             f"eps_mode={c[3]} draws={c[4]} k={c[5]}"
                             f"  ({sum(ident(r) == c for r in keep)} runs)"
                             for c in configs)
        raise SystemExit(
            f"{name}: the store holds {len(configs)} different configurations, "
            f"which must not be pooled:\n{listing}\n"
            f"Pick one with --eps-mode / --i0 / --itermax / --s.")
    return keep


def _sort_runs(by_phantom):
    for runs in by_phantom.values():
        runs.sort(key=lambda r: (r["noise_seed"] is None, r["noise_seed"]))
    return by_phantom


def upper_envelope(y: np.ndarray, window: int) -> np.ndarray:
    """Running maximum over a centred window -- the top of the oscillation.

    Fitting the raw trace would measure the oscillation as much as its decay;
    fitting the envelope isolates the decay, which is the thing being claimed.
    """
    n = len(y)
    half = max(1, window // 2)
    out = np.empty(n)
    for i in range(n):
        out[i] = y[max(0, i - half):min(n, i + half + 1)].max()
    return out


def trace_metrics(y: np.ndarray, window: int, fit_lo: int, fit_hi: int,
                  burn_in: int) -> dict:
    """The four stability metrics for one arm of one realization.

    `burn_in` drops the opening iterations before the minimum is searched for.
    Both arms start from f = 0 and swing hard for the first few tens of
    iterations; without a burn-in the global argmin can land on a transient dip
    that is not a converged state and is not a stopping point anyone would
    choose. On breast the two-channel arm dips to 0.249 at iteration 6 and is
    back up at 0.391 by iteration 10 -- taking that as "the minimum" would
    understate the arm it is supposed to measure. The value is declared, not
    tuned; write_table reports the sensitivity across a range of burn-ins.
    """
    n = len(y)
    b = max(0, min(burn_in - 1, n - 1))        # index of first kept iterate
    imin = b + int(np.argmin(y[b:]))
    ymin = float(y[imin])
    gap = float(y[-1]) - ymin

    # Censored when the minimum IS the last iterate: the arm had not turned
    # around when the budget ran out, so its turnaround is only a lower bound.
    censored = imin == n - 1

    # Excess path uses the same burn-in, so every metric describes the same
    # window of the run rather than mixing transient and steady behaviour.
    env = upper_envelope(y, window)
    env_lo = -upper_envelope(-y, window)          # running MINIMUM
    mid = 0.5 * (env + env_lo)
    lo, hi = max(1, fit_lo), min(n, fit_hi)
    it = np.arange(1, n + 1, dtype=float)
    sl = slice(lo - 1, hi)
    x, e = it[sl], env[sl]
    ok = e > 0
    slope = (float(np.polyfit(np.log10(x[ok]), np.log10(e[ok]), 1)[0])
             if ok.sum() >= 2 else np.nan)

    # Width of the oscillation relative to the level it oscillates about,
    # measured over the same window the slope is fitted on. Median rather than
    # mean: a single spike should not set the number for the whole run.
    band_w, band_m = (env - env_lo)[sl], mid[sl]
    good = band_m > 0
    osc = float(np.median(band_w[good] / band_m[good])) if good.any() else np.nan

    return {
        "semiconv_gap": gap,
        "semiconv_rel": 100.0 * gap / ymin if ymin > 0 else np.nan,
        "turnaround": float(imin + 1),
        "osc_band": osc,
        "env_slope": slope,
        "censored": censored,
        "min_rmse": ymin,
        "final_rmse": float(y[-1]),
        "envelope": env,
    }


def paired_stats(a: np.ndarray, b: np.ndarray, better: int) -> dict:
    """Paired difference oriented so positive always means two-channel better.

    `a` is single, `b` is two. Mirrors run_2d_noise_stats.paired_stats: a
    t-interval on the paired difference and a Wilcoxon signed-rank p, with the
    two wrapped separately so a Wilcoxon failure cannot silently downgrade the
    t critical value to the normal approximation.
    """
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = (a - b) if better < 0 else (b - a)
    n = len(d)
    if n == 0:
        return dict(n=0)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan

    tcrit = 1.96
    if n > 1:
        try:
            from scipy import stats
            tcrit = float(stats.t.ppf(0.975, n - 1))
        except Exception:
            pass
    p = np.nan
    if n >= 6:
        try:
            from scipy import stats
            p = float(stats.wilcoxon(a, b).pvalue)
        except Exception:
            pass                      # raises when every difference is zero

    return dict(n=n, single=float(a.mean()), two=float(b.mean()), mean=mean,
                lo=mean - tcrit * se, hi=mean + tcrit * se,
                median=float(np.median(d)),
                q1=float(np.percentile(d, 25)), q3=float(np.percentile(d, 75)),
                p=p, win=float((d > 0).mean()))


def collect(runs: list[dict], window: int, fit_lo: int, fit_hi: int,
            burn_in: int) -> dict:
    per = {"single": [], "two": []}
    for r in runs:
        for arm in ("single", "two"):
            per[arm].append(
                trace_metrics(r[arm], window, fit_lo, fit_hi, burn_in))
    return per


def burn_in_sensitivity(runs: list[dict], args, burn_ins) -> list[str]:
    """Gap and turnaround at several burn-ins, so the choice is auditable.

    A stability claim that only survives one burn-in is not a stability claim.
    """
    L = ["", f"### Sensitivity to the burn-in (declared value: "
             f"{args.burn_in})", "",
         "| burn-in | single gap | two gap | mean diff | single turnaround | "
         "two turnaround | two censored |", "|---|---|---|---|---|---|---|"]
    for b in burn_ins:
        per = collect(runs, args.window, args.fit_lo, args.fit_hi, b)
        ga = np.array([m["semiconv_gap"] for m in per["single"]], dtype=float)
        gb = np.array([m["semiconv_gap"] for m in per["two"]], dtype=float)
        st = paired_stats(ga, gb, -1)
        ta = np.median([m["turnaround"] for m in per["single"]])
        tb = np.median([m["turnaround"] for m in per["two"]])
        cens = sum(m["censored"] for m in per["two"])
        mark = " <-" if b == args.burn_in else ""
        L.append(f"| {b}{mark} | {st['single']:.5f} | {st['two']:.5f} | "
                 f"{st['mean']:+.5f} | {ta:.0f} | {tb:.0f} | {cens}/{len(runs)} |")
    return L


def window_sensitivity(runs: list[dict], args, windows) -> list[str]:
    """Envelope metrics across window widths, so w is auditable like burn-in.

    The envelope is a smoothing choice, and a reader is entitled to ask whether
    the conclusion is an artefact of the width picked. It is reported rather
    than argued.
    """
    L = ["", f"### Sensitivity to the envelope window (declared value: "
             f"{args.window})", "",
         "| window | single slope | two slope | slope diff | single band | "
         "two band | band diff |", "|---|---|---|---|---|---|---|"]
    for w in windows:
        per = collect(runs, w, args.fit_lo, args.fit_hi, args.burn_in)
        out = []
        for key, better in (("env_slope", -1), ("osc_band", -1)):
            a = np.array([m[key] for m in per["single"]], dtype=float)
            b = np.array([m[key] for m in per["two"]], dtype=float)
            out.append(paired_stats(a, b, better))
        mark = " <-" if w == args.window else ""
        L.append(f"| {w}{mark} | {out[0]['single']:+.4f} | {out[0]['two']:+.4f} | "
                 f"{out[0]['mean']:+.4f} | {out[1]['single']:.4f} | "
                 f"{out[1]['two']:.4f} | {out[1]['mean']:+.4f} |")
    return L


def write_table(name: str, runs: list[dict], per: dict, args) -> Path:
    n = len(runs)
    cens = {arm: sum(m["censored"] for m in per[arm]) for arm in per}
    pfloor = 2.0 / (2.0 ** n)

    L = [f"# {name}: stability metrics ({n} noise realizations, s={runs[0]['s']:g})\n",
         "Computed from the stored 500-iteration RMSE traces of the "
         "`2d_noise_stats` runs -- no solver was re-run. Both arms of a "
         "realization see the same Poisson draw, so the difference is paired "
         "and removes variation the two arms share.\n",
         "**Both arms oscillate.** These metrics do not claim otherwise. They "
         "measure the envelope and the turnaround: how much is lost by "
         "overrunning the minimum, where the minimum falls, how much motion is "
         "spent per unit of progress, and how fast the oscillation decays.\n",
         "Positive difference = two-channel better, for every row, regardless "
         "of which direction the raw metric runs.\n",
         "| metric | single | two-channel | mean diff | 95% CI | median [IQR] "
         "| win rate | Wilcoxon p |",
         "|---|---|---|---|---|---|---|---|"]

    for key, (label, better, fmt) in METRICS.items():
        a = np.array([m[key] for m in per["single"]], dtype=float)
        b = np.array([m[key] for m in per["two"]], dtype=float)
        st = paired_stats(a, b, better)
        if not st.get("n"):
            L.append(f"| {label} | -- | -- | -- | -- | -- | -- | -- |")
            continue
        arrow = "higher better" if better > 0 else "lower better"
        dag = " †" if (key == "turnaround" and (cens["single"] or cens["two"])) else ""
        L.append(
            f"| {label} ({arrow}){dag} | {st['single']:{fmt}} | {st['two']:{fmt}} | "
            f"{st['mean']:+{fmt}} | [{st['lo']:+{fmt}}, {st['hi']:+{fmt}}] | "
            f"{st['median']:+{fmt}} [{st['q1']:+{fmt}}, {st['q3']:+{fmt}}] | "
            f"{st['win']:.0%} | {st['p']:.2g} |")

    if cens["single"] or cens["two"]:
        L += ["", "† The turnaround row is affected by censoring: a censored "
                  "arm contributes the budget (500) rather than a measured "
                  "value, so its mean is pinned and the CI understates the "
                  "true uncertainty. Read the medians and the censoring counts "
                  "below, not the mean, for that row."]
    L += ["",
          f"Censored turnarounds (minimum was the final iterate, so the value "
          f"is a lower bound, not a measurement): single {cens['single']}/{n}, "
          f"two-channel {cens['two']}/{n}."]
    if cens["two"] or cens["single"]:
        L.append("Where an arm is censored its turnaround row understates that "
                 "arm; a censored two-channel arm makes the reported "
                 "two-channel advantage a LOWER bound on the true one.")
    L += ["",
          f"With n = {n} paired samples the smallest attainable two-sided "
          f"Wilcoxon p is {pfloor:.1e}; a p at that floor means only that every "
          f"pair had the same sign, and carries no information beyond the win "
          f"rate. The confidence interval is the informative statistic.",
          "",
          f"Minimum searched from iteration {args.burn_in} onward (burn-in), "
          f"excluding the start-from-zero transient. Envelope: running maximum "
          f"over a {args.window}-iteration centred window, slope fitted log-log "
          f"over iterations {args.fit_lo}-{args.fit_hi}."]
    L += burn_in_sensitivity(runs, args, args.burn_ins)
    L += window_sensitivity(runs, args, args.windows)

    p = TAB / f"stability_metrics_{name}.md"
    p.write_text("\n".join(L) + "\n")
    print(f"  saved {p}")
    return p


def fig_stability(name: str, runs: list[dict], per: dict, args) -> Path:
    S = np.vstack([r["single"] for r in runs])
    T = np.vstack([r["two"] for r in runs])
    it = np.arange(1, S.shape[1] + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # --- median trace with its envelope: the oscillation and its decay -----
    ax = axes[0]
    for M, per_arm, c, lab in ((S, per["single"], "k", "single-channel"),
                               (T, per["two"], "#1f77b4", "two-channel s=4")):
        med = np.median(M, axis=0)
        env = np.median(np.vstack([m["envelope"] for m in per_arm]), axis=0)
        ax.loglog(it, med, "-", color=c, lw=1.4, alpha=0.75, label=f"{lab}")
        ax.loglog(it, env, "--", color=c, lw=1.8, alpha=0.95,
                  label=f"{lab} envelope")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7)
    ax.set_title(f"{name}: median trace and upper envelope", fontsize=10)

    # --- semiconvergence: distance above each arm's own best -------------
    ax = axes[1]
    for M, c, lab in ((S, "k", "single-channel"), (T, "#1f77b4", "two-channel s=4")):
        run_min = np.minimum.accumulate(M, axis=1)
        excess = M - run_min           # 0 while improving, >0 once past the min
        ax.semilogx(it, np.median(excess, axis=0), "-", color=c, lw=1.8, label=lab)
        ax.fill_between(it, np.percentile(excess, 25, axis=0),
                        np.percentile(excess, 75, axis=0), color=c, alpha=0.2)
    # The running minimum here starts at iteration 1, so the opening transient
    # is visible rather than hidden; the marker shows where the tables start
    # measuring, so figure and table cannot be read as disagreeing.
    ax.axvline(args.burn_in, color="0.5", ls=":", lw=1.2)
    ax.annotate(f"burn-in = {args.burn_in}", xy=(args.burn_in, 0.97),
                xycoords=("data", "axes fraction"), fontsize=7, color="0.35",
                rotation=90, ha="right", va="top")
    ax.set_xlabel("iteration"); ax.set_ylabel("RMSE above running minimum")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("how far past its own best each arm has drifted", fontsize=10)

    # --- paired per-realization semiconvergence gap -----------------------
    ax = axes[2]
    ga = np.array([m["semiconv_gap"] for m in per["single"]])
    gb = np.array([m["semiconv_gap"] for m in per["two"]])
    x = np.arange(len(ga))
    ax.plot(x, ga, "o", color="k", ms=4, label="single-channel")
    ax.plot(x, gb, "o", color="#1f77b4", ms=4, label="two-channel s=4")
    ax.vlines(x, np.minimum(ga, gb), np.maximum(ga, gb), color="0.7", lw=0.8,
              zorder=0)
    ax.set_xlabel("noise realization"); ax.set_ylabel("semiconvergence gap")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("paired gap, one line per realization", fontsize=10)

    fig.tight_layout()
    p = FIG / f"stability_metrics_{name}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")
    return p


def fig_envelope_demo(name: str, runs: list[dict], args) -> Path:
    """One realization, drawn so a reader can SEE what the envelope is.

    Every other figure and table assumes the definition; this one shows it, on
    a single trace, before any statistics are taken over realizations.
    """
    r = runs[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, arm, c, lab in ((axes[0], "single", "k", "single-channel"),
                            (axes[1], "two", "#1f77b4", "two-channel s=4")):
        y = r[arm]
        it = np.arange(1, len(y) + 1)
        env_hi = upper_envelope(y, args.window)
        env_lo = -upper_envelope(-y, args.window)
        ax.loglog(it, y, "-", color=c, lw=0.9, alpha=0.55, label="RMSE trace")
        ax.loglog(it, env_hi, "-", color=c, lw=2.0,
                  label=f"upper envelope E+ (w={args.window})")
        ax.loglog(it, env_lo, "--", color=c, lw=1.4, label="lower envelope E-")
        ax.fill_between(it, env_lo, env_hi, color=c, alpha=0.12)

        # the fitted power law, drawn where it was actually fitted
        lo, hi = max(1, args.fit_lo), min(len(y), args.fit_hi)
        sl = slice(lo - 1, hi)
        x, e = it[sl].astype(float), env_hi[sl]
        ok = e > 0
        if ok.sum() >= 2:
            k, b = np.polyfit(np.log10(x[ok]), np.log10(e[ok]), 1)
            ax.loglog(x, 10 ** (b + k * np.log10(x)), ":", color="crimson",
                      lw=2.0, label=f"fit to E+: slope {k:+.3f}")
        ax.axvline(args.burn_in, color="0.6", ls=":", lw=1.0)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"{name}, realization 1: {lab}", fontsize=10)
    axes[0].set_ylabel("image RMSE")
    fig.suptitle("What the envelope is: running max/min of the RMSE trace, "
                 "and the power law fitted to the upper one", fontsize=9)
    fig.tight_layout()
    p = FIG / f"stability_envelope_demo_{name}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")
    return p


def write_summary(rows: list[tuple], args) -> Path:
    L = ["# Stability metrics: all phantoms\n",
         "Paired mean differences, positive = two-channel better. Built from "
         "the stored `2d_noise_stats` traces; nothing was re-run.\n",
         "| phantom | n | semiconv gap (abs) | semiconv gap (%) | turnaround "
         "(iters later) | oscillation band | envelope slope |",
         "|---|---|---|---|---|---|---|"]
    for name, n, st in rows:
        def cell(key, fmt):
            s = st[key]
            if not s.get("n"):
                return "--"
            if not ((s["lo"] > 0) == (s["hi"] > 0)):
                return f"{s['mean']:+{fmt}} (n.s.)"
            if s["win"] < 0.9:
                return f"{s['mean']:+{fmt}} (split {s['win']:.0%})"
            return f"{s['mean']:+{fmt}}"
        L.append(f"| {name} | {n} | {cell('semiconv_gap', '.5f')} | "
                 f"{cell('semiconv_rel', '.1f')} | {cell('turnaround', '.0f')} | "
                 f"{cell('osc_band', '.4f')} | {cell('env_slope', '.4f')} |")
    L += ["", "`(split ..%)` marks a mean whose sign the individual "
              "realizations do not agree on -- a mean driven by a minority or "
              "by censored values, not a consistent effect.",
          "", "`(n.s.)` marks a 95% CI that straddles zero -- the difference is "
              "not resolved by this many realizations, whatever the point "
              "estimate's sign. Per-phantom tables carry the intervals, the "
              "censoring counts, and the metric definitions."]
    p = TAB / "stability_metrics.md"
    p.write_text("\n".join(L) + "\n")
    print(f"  saved {p}")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phantom", default="all",
                    help="phantom name, or 'all' (default)")
    ap.add_argument("--eps-mode", default=None,
                    help="use only runs from this tolerance mode "
                         "(noise | noise_per_band)")
    ap.add_argument("--i0", type=float, default=None,
                    help="use only runs at this i0 (needed if the store holds several)")
    ap.add_argument("--itermax", type=int, default=None,
                    help="use only runs of this length")
    ap.add_argument("--s", type=float, default=None,
                    help="use only runs at this sigma_lo scale")
    ap.add_argument("--results", type=Path, default=RESULTS,
                    help="runstore directory holding the 2d_noise_stats runs")
    ap.add_argument("--window", type=int, default=25,
                    help="centred window for the running-maximum envelope")
    ap.add_argument("--fit-lo", type=int, default=50,
                    help="first iteration of the envelope log-log fit")
    ap.add_argument("--fit-hi", type=int, default=500,
                    help="last iteration of the envelope log-log fit")
    ap.add_argument("--burn-in", type=int, default=25,
                    help="search the minimum from this iteration on, excluding "
                         "the start-from-zero transient (default: 25)")
    ap.add_argument("--windows", type=int, nargs="+",
                    default=[11, 25, 51, 101],
                    help="window widths for the envelope sensitivity table")
    ap.add_argument("--burn-ins", type=int, nargs="+",
                    default=[1, 10, 25, 50, 100],
                    help="burn-in values for the sensitivity table")
    args = ap.parse_args()

    if not args.results.is_dir():
        raise SystemExit(
            f"No runstore at {args.results}.\n"
            f"These runs live on the compute server; pull them with\n"
            f"  rsync -av <server>:ct/paper/experiments/results/2d_noise_stats/ "
            f"{args.results}/")

    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    by_phantom = load_runs(args.results)
    if not by_phantom:
        raise SystemExit(f"No 2d_noise_stats runs found in {args.results}")

    names = (sorted(by_phantom) if args.phantom == "all"
             else [p for p in [args.phantom] if p in by_phantom])
    if not names:
        raise SystemExit(f"phantom {args.phantom!r} not in "
                         f"{sorted(by_phantom)}")

    rows = []
    for name in names:
        runs = select_config(name, by_phantom[name], args)
        if not runs:
            print(f"\n{name}: no runs match the filters -- skipped")
            continue
        print(f"\n{name}: {len(runs)} realizations, "
              f"{len(runs[0]['single'])} iterations")
        per = collect(runs, args.window, args.fit_lo, args.fit_hi,
                      args.burn_in)
        write_table(name, runs, per, args)
        fig_stability(name, runs, per, args)
        fig_envelope_demo(name, runs, args)
        st = {k: paired_stats(
                  np.array([m[k] for m in per["single"]], dtype=float),
                  np.array([m[k] for m in per["two"]], dtype=float), better)
              for k, (_, better, _) in METRICS.items()}
        rows.append((name, len(runs), st))

        g = st["semiconv_gap"]
        if g.get("n"):
            print(f"  semiconvergence gap: single {g['single']:.5f} vs "
                  f"two {g['two']:.5f}  (diff {g['mean']:+.5f}, "
                  f"win {g['win']:.0%})")

    if len(rows) > 1:
        write_summary(rows, args)


if __name__ == "__main__":
    main()
