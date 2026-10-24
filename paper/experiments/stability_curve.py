"""The certified margin as a continuous function of the low-frequency scale.

`stability_diagnostic.py` evaluates the margin at s in {1, 2, 4, 8} only. Four
points cannot say where the low band takes over the limiting eigenmode, yet the
manuscript currently asserts the margin "stays at the single-channel boundary
for s up to roughly 6" -- a number no run produces. This sweeps s finely enough
to locate the crossover and report it, instead of interpolating between two
decades.

For each s it reports the sharp margin

    tau*lambda_max(M) = lambda_max(Mtilde(s)) / lambda_max(Mtilde(1)),

normalised so the single-channel step-size boundary is exactly 1, together with
the low band's share of the limiting eigenmode -- which is what actually flips
when the crossover happens. Everything is imported from stability_diagnostic,
so the operator, filters and power iteration are the ones the paper describes.

Phantom-independent: this is a property of the geometry and the filters.

    python paper/experiments/stability_curve.py            # ~25 min, 2D CPU
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stability_diagnostic as sd                       # noqa: E402

FIG = HERE / "figs"
TAB = HERE / "tables"
OUT = HERE / "results" / "stability_curve"
for _d in (FIG, TAB, OUT):
    _d.mkdir(parents=True, exist_ok=True)

# Dense where the crossover is expected (lambda_hi/lambda_lo ~ 4.9), coarse
# outside it; s=1,2,4,8 included so the existing table's four points are
# reproduced exactly rather than approximated.
GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
        8.0, 10.0, 12.0, 16.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=float, nargs="+", default=GRID)
    ap.add_argument("--cache", default=str(OUT / "margin_vs_s.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    if cache.exists() and not args.force:
        data = json.loads(cache.read_text())
        print(f"loaded {cache} ({len(data['s'])} points)")
    else:
        print("computing block operator-norm scalings (matches the solver)...")
        nusino, nux, nuy = sd.compute_scalings()
        lam_base, _ = sd.lam_max(sd.make_Mtilde(nusino, nux, nuy, 1.0))
        print(f"lambda_max(Mtilde(1)) = {lam_base:.6e}")
        s_list, margin, lo_share = [], [], []
        for s in args.grid:
            lam_s, v_s = sd.lam_max(sd.make_Mtilde(nusino, nux, nuy, s))
            be = sd.block_energies(v_s, nusino, nux, nuy, s)
            s_list.append(float(s))
            margin.append(float(lam_s / lam_base))
            lo_share.append(float(be["lo"]))
            print(f"  s={s:5.2f}  margin={margin[-1]:.4f}  "
                  f"lo share of limiting mode={lo_share[-1]:.3f}")
        data = dict(s=s_list, margin=margin, lo_share=lo_share,
                    lam_base=float(lam_base))
        cache.write_text(json.dumps(data, indent=1))
        print(f"saved {cache}")

    s = np.array(data["s"]); m = np.array(data["margin"])
    lo = np.array(data["lo_share"])

    # Crossover: first s where the low band owns the limiting eigenmode. Read
    # from the eigenmode, not the margin -- the margin leaves 1 only after the
    # low band has already taken over, so the mode share is the earlier and
    # more honest indicator.
    idx = np.where(lo > 0.5)[0]
    s_cross = float(s[idx[0]]) if len(idx) else float("nan")
    cert = s[m < 1.0 + 1e-9]
    s_max_cert = float(cert.max()) if len(cert) else float("nan")

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(s, m, "o-", color="#1f77b4", lw=1.8, ms=4,
            label=r"margin $\tau\lambda_{\max}(M)$")
    ax.axhline(1.0, color="k", ls="--", lw=1.2,
               label="single-channel boundary")
    ax.set_xlabel(r"low-frequency step scale $s$")
    ax.set_ylabel(r"$\tau\lambda_{\max}(M)$")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(s, lo, "s--", color="crimson", lw=1.2, ms=3, alpha=0.8,
             label="low-band share of limiting mode")
    ax2.set_ylabel("low-band share of limiting eigenmode", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(-0.02, 1.02)
    if np.isfinite(s_cross):
        ax.axvline(s_cross, color="0.5", ls=":", lw=1.2)
        ax.annotate(f"low band takes over\nat $s={s_cross:g}$",
                    xy=(s_cross, 1.0), xytext=(s_cross * 1.05, 1.06),
                    fontsize=7, color="0.3")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.set_title("Certified margin vs low-frequency amplification", fontsize=10)
    fig.tight_layout()
    p = FIG / "stability_curve.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"saved {p}")

    L = ["# Certified margin vs low-frequency step scale\n",
         "Geometry and filters only; no phantom and no reconstruction. "
         "The margin is normalised so the single-channel step-size boundary "
         "is exactly 1.\n",
         "| s | margin | low-band share of limiting mode | certified (< 1)? |",
         "|---|---|---|---|"]
    for a, b, c in zip(s, m, lo):
        L.append(f"| {a:g} | {b:.4f} | {c:.3f} | {'yes' if b < 1.0 else 'no'} |")
    L += ["", f"Largest swept $s$ with margin strictly below 1: "
              f"**{s_max_cert:g}**. The low band first owns the limiting "
              f"eigenmode (share > 0.5) at **s = {s_cross:g}**.",
          "", "The margin is reported as computed. Values are power-iteration "
              "estimates, so a margin within the iteration's own accuracy of 1 "
              "should be read as 'at the boundary', not as strictly below it."]
    p = TAB / "stability_curve.md"
    p.write_text("\n".join(L) + "\n")
    print(f"saved {p}")
    print(f"\ncrossover s={s_cross:g} | largest certified swept s={s_max_cert:g}")


if __name__ == "__main__":
    main()
