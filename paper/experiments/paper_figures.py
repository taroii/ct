"""Consolidated main-body figures, built from results already on disk.

The study produced roughly sixty figures for about eight claims: five
per-phantom convergence plots, five iteration ladders, five error ladders, two
sets of five noise plots, eight near-identical 3D panels. That is a browsing
archive, not a paper. This builds the small set the Experiments section
actually needs, each one carrying a claim no other figure carries:

  noise_grid   one figure, five phantoms: single-channel semiconverges under
               noise and the two-channel arm does not. Replaces five separate
               noise figures.
  mechanism    (a) the result survives a 300x range of low-band tolerance, and
               the derived rule is not the optimum -- answering the only
               question a reader can ask about a derived constant; (b) the gain
               tracks how much of a phantom's data lives in the low band, which
               is the two-level theory's prediction.
  phantoms     all five phantoms. The existing preview shows three.

Everything is read from the runstore and the generated tables; nothing is
recomputed, so these cannot disagree with the numbers they illustrate.

    python paper/experiments/paper_figures.py                 # all
    python paper/experiments/paper_figures.py --only noise_grid
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(HERE))

FIG = HERE / "figs"
TAB = HERE / "tables"
STORE2D = HERE / "results" / "2d_noise_stats"
FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["breast", "head", "defrise", "jaw", "shepp_logan"]
LABEL = {"breast": "breast", "head": "head", "defrise": "Defrise",
         "jaw": "jaw", "shepp_logan": "Shepp--Logan"}
SINGLE_C, TWO_C = "k", "#1f77b4"
HERO_S = 4.0


# --------------------------------------------------------------------------
def load_noise(eps_mode="noise_per_band"):
    """Per-phantom stacks of RMSE traces for both arms, one config only."""
    out = {}
    for jp in sorted(STORE2D.glob("*.json")):
        c = json.loads(jp.read_text())["config"]
        if c.get("eps_mode") != eps_mode:
            continue
        with np.load(jp.with_suffix(".npz")) as d:
            out.setdefault(c["phantom"], {"single": [], "two": []})
            out[c["phantom"]]["single"].append(np.asarray(d["ierrs_single"], float))
            out[c["phantom"]]["two"].append(np.asarray(d["ierrs_two"], float))
    return {k: {a: np.vstack(v) for a, v in d.items()} for k, d in out.items()}


def fig_noise_grid(eps_mode="noise_per_band"):
    data = load_noise(eps_mode)
    names = [n for n in ORDER if n in data]
    if not names:
        print("  noise_grid: no runs for this mode -- skipped")
        return
    fig, axes = plt.subplots(1, len(names), figsize=(2.45 * len(names), 2.9),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        d = data[name]
        it = np.arange(1, d["single"].shape[1] + 1)
        for arm, colour, lab in (("single", SINGLE_C, "single-channel"),
                                 ("two", TWO_C, f"two-channel $s={HERO_S:g}$")):
            med = np.median(d[arm], axis=0)
            ax.semilogx(it, med, "-", color=colour, lw=1.7, label=lab)
            # Interquartile band, not mean +/- sd: RMSE trajectories are skewed.
            ax.fill_between(it, np.percentile(d[arm], 25, axis=0),
                            np.percentile(d[arm], 75, axis=0),
                            color=colour, alpha=0.18, lw=0)
        # Mark where single-channel turns around -- the semiconvergence claim.
        med = np.median(d["single"], axis=0)
        j = int(np.argmin(med[9:])) + 9
        if j < len(med) - 1:
            ax.plot([it[j]], [med[j]], "v", color=SINGLE_C, ms=5)
        ax.set_title(LABEL[name], fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("iteration", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("image RMSE", fontsize=8)
    axes[0].legend(fontsize=6.5, loc="lower left")
    n = data[names[0]]["single"].shape[0]
    fig.suptitle(f"Poisson noise, median and interquartile range over {n} "
                 f"paired realizations; marker = single-channel turnaround",
                 fontsize=9)
    fig.tight_layout()
    p = FIG / f"noise_grid_{eps_mode}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


# --------------------------------------------------------------------------
def _gain_at(name, it=500, suffix="_noise_per_band"):
    t = (TAB / f"noise_stats_{name}{suffix}.md").read_text()
    row = [l for l in t.splitlines() if l.startswith(f"| {it} ")]
    if not row:
        return None
    return float(re.search(r"\(([-+][\d.]+)%\)", row[0].split("|")[4]).group(1))


def _band_ratio():
    t = (TAB / "band_energy.md").read_text()
    out = {}
    for l in t.splitlines():
        m = re.match(r"\|\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", l)
        if m and m.group(1) in LABEL:
            out[m.group(1)] = float(m.group(3))
    return out


def fig_mechanism(phantom="breast"):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    # (a) robustness to the low-band tolerance
    ax = axes[0]
    tp = TAB / f"tolerance_sensitivity_{phantom}.md"
    if tp.exists():
        txt = tp.read_text()
        single = float(re.search(r"Single-channel \(([\d.]+)\)", txt).group(1))
        mult, rmse = [], []
        for l in txt.splitlines():
            m = re.match(r"\|\s*([\d.]+)(?:\s*<- rule)?\s*\|\s*[\d.e+-]+\s*\|\s*([\d.]+)\s*\|", l)
            if m:
                mult.append(float(m.group(1))); rmse.append(float(m.group(2)))
        mult, rmse = np.array(mult), np.array(rmse)
        ax.axhline(single, color=SINGLE_C, ls="--", lw=1.5,
                   label="single-channel")
        ax.semilogx(mult, rmse, "o-", color=TWO_C, lw=1.7, ms=4.5,
                    label=f"two-channel $s={HERO_S:g}$")
        j = int(np.argmin(np.abs(mult - 1.0)))
        ax.plot([mult[j]], [rmse[j]], "o", ms=12, mfc="none", mec="crimson",
                mew=2, label="per-band rule (all reported runs)")
        win = mult[rmse < single]
        if len(win):
            ax.axvspan(win.min(), win.max(), color=TWO_C, alpha=0.07)
        ax.set_xlabel("low-band tolerance, multiple of the per-band rule")
        ax.set_ylabel("image RMSE at iteration 500")
        ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7)
        ax.set_title(f"(a) {LABEL.get(phantom, phantom)}: the rule is not the "
                     f"optimum", fontsize=9)
    else:
        ax.set_visible(False)
        print(f"  mechanism: {tp.name} missing -- panel (a) skipped")

    # (b) gain vs how much data lives in the low band
    ax = axes[1]
    ratio = _band_ratio()
    xs, ys, labs = [], [], []
    for n in ORDER:
        g = _gain_at(n)
        if g is not None and n in ratio:
            xs.append(ratio[n]); ys.append(g); labs.append(LABEL[n])
    if len(xs) >= 3:
        xs, ys = np.array(xs), np.array(ys)
        ax.semilogx(xs, ys, "o", color=TWO_C, ms=8)
        for x, y, l in zip(xs, ys, labs):
            ax.annotate(l, (x, y), textcoords="offset points", xytext=(7, -3),
                        fontsize=7.5)
        try:
            from scipy import stats
            r = stats.spearmanr(xs, ys)
            ax.set_title(f"(b) gain tracks low-band content "
                         f"(Spearman $\\rho={r.statistic:+.3f}$, "
                         f"$p={r.pvalue:.3f}$, $n={len(xs)}$)", fontsize=9)
        except Exception:
            ax.set_title("(b) gain tracks low-band content", fontsize=9)
        ax.set_xlabel("low/high band energy ratio of the phantom's projections")
        ax.set_ylabel("RMSE reduction at iteration 500 (%)")
        ax.grid(True, alpha=0.3, which="both")
        ax.margins(x=0.18, y=0.18)
    else:
        ax.set_visible(False)
        print("  mechanism: not enough phantoms for panel (b)")
    fig.tight_layout()
    p = FIG / "mechanism.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


# --------------------------------------------------------------------------
def fig_phantoms():
    """All five phantoms. The existing preview shows three."""
    sys.path.insert(0, str(PAPER))
    from run_2d_convergence import PHANTOMS as REG
    from run_2d_noise_stats import _breast_phantom
    fig, axes = plt.subplots(1, len(ORDER), figsize=(2.1 * len(ORDER), 2.4))
    for ax, name in zip(np.atleast_1d(axes), ORDER):
        f = REG[name]()
        f = _breast_phantom() if f is None else np.asarray(f, float)
        ax.imshow(f, cmap="gray", vmin=0.0, vmax=float(np.percentile(f, 99.5)))
        ax.set_title(LABEL[name], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p = FIG / "phantoms_all.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


def fig_band_split():
    """What the split IS, and what it costs -- the two things Table 1 asserts.

    (a) the three fidelity weights in detector frequency, so a reader can see
    that the low channel is a narrow near-DC window and the high channel its
    complement, both carrying the shared ramp; (b) the certified margin against
    the low-frequency scale, which is flat until the low band takes over the
    limiting eigenmode. Panel (b) is the measurement that replaces an
    interpolation between two decades.
    """
    import stability_diagnostic as sd
    u = np.asarray(sd.uar, float)
    um = float(np.abs(u).max())
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    ax = axes[0]
    for W, colour, lab in ((sd.W_single2, SINGLE_C, r"single  $|u|\,\mathrm{Hann}(c)$"),
                           (sd.W_hi2, "#d62728", r"high  $|u|\,\mathrm{Hann}_{\mathrm{hi}}$"),
                           (sd.W_lo2, TWO_C, r"low  $|u|\,\mathrm{Hann}(c_{\mathrm{lo}})$")):
        w = np.asarray(W, float)
        ax.plot(u / um, w, "-", color=colour, lw=1.8, label=lab)
    ax.fill_between(u / um, 0, np.asarray(sd.W_lo2, float), color=TWO_C, alpha=0.18)
    ax.set_xlabel("detector frequency $u$ (normalised)")
    ax.set_ylabel("fidelity weight $F^2$")
    # Zoomed to the near-DC region: over the full band the ramp dwarfs the low
    # channel and the split -- the entire point of the panel -- is invisible.
    # Beyond this window the low weight is identically zero and the high weight
    # coincides with the single-channel one, so nothing is hidden.
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(0, float(np.asarray(sd.W_single2, float)[
        np.abs(u) / um <= 0.35].max()) * 1.12)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="upper center")
    ax.annotate("beyond this window the low weight is 0\nand high $=$ single",
                xy=(0.02, 0.03), xycoords="axes fraction", fontsize=6.5,
                color="0.35")
    ax.set_title("(a) the two channels partition the single-channel weight",
                 fontsize=9)

    ax = axes[1]
    cache = HERE / "results" / "stability_curve" / "margin_vs_s.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        s_, m_, lo_ = (np.array(d["s"]), np.array(d["margin"]),
                       np.array(d["lo_share"]))
        ax.plot(s_, m_, "o-", color=TWO_C, lw=1.8, ms=4,
                label=r"margin $\tau\lambda_{\max}(M)$")
        ax.axhline(1.0, color=SINGLE_C, ls="--", lw=1.2,
                   label="single-channel boundary")
        ax2 = ax.twinx()
        ax2.plot(s_, lo_, "s--", color="crimson", lw=1.1, ms=3, alpha=0.8,
                 label="low-band share of limiting mode")
        ax2.set_ylabel("low-band share of limiting mode", color="crimson",
                       fontsize=8)
        ax2.tick_params(axis="y", labelcolor="crimson", labelsize=7)
        ax2.set_ylim(-0.02, 1.02)
        idx = np.where(lo_ > 0.5)[0]
        if len(idx):
            ax.axvline(s_[idx[0]], color="0.5", ls=":", lw=1.2)
            ax.annotate(f"low band takes over\nat $s={s_[idx[0]]:g}$",
                        xy=(s_[idx[0]], 1.0),
                        xytext=(s_[idx[0]] + 0.4, 1.02), fontsize=7, color="0.3")
        ax.plot([4.0], [m_[list(s_).index(4.0)]], "o", ms=12, mfc="none",
                mec="crimson", mew=2)
        ax.annotate("operating point", xy=(4.0, m_[list(s_).index(4.0)]),
                    xytext=(1.6, 1.30), fontsize=7, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=0.9))
        ax.set_xlabel(r"low-frequency step scale $s$")
        ax.set_ylabel(r"$\tau\lambda_{\max}(M)$")
        ax.grid(True, alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        ax.set_title("(b) amplifying the low channel is free until $s\\approx6$",
                     fontsize=9)
    else:
        ax.set_visible(False)
        print("  band_split: run stability_curve.py first -- panel (b) skipped")
    fig.tight_layout()
    p = FIG / "band_split.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+",
                    choices=["band_split", "noise_grid", "mechanism", "phantoms"],
                    default=["band_split", "noise_grid", "mechanism", "phantoms"])
    ap.add_argument("--phantom", default="breast",
                    help="phantom for the tolerance panel")
    args = ap.parse_args()
    if "band_split" in args.only:
        print("band_split"); fig_band_split()
    if "noise_grid" in args.only:
        print("noise_grid"); fig_noise_grid()
    if "mechanism" in args.only:
        print("mechanism"); fig_mechanism(args.phantom)
    if "phantoms" in args.only:
        print("phantoms"); fig_phantoms()


if __name__ == "__main__":
    main()
