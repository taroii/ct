"""Build a single clean design-space figure for the talk.

Plots iter-100 image-RMSE reduction (%) vs r = eps_lo / eps_hi for both
the 2D paper breast (H2) and the 3D analytic breast (H4). One panel,
two curves, marker on the paper config (r=1.25). Replaces the previous
side-by-side spaghetti plot that had too many lines to read.

Data is hard-coded from H2_eps_summary_256.txt and ct2_breast_H4_eps_summary.txt.

Output: presentation/figs/design_space_iter100.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "presentation" / "figs" / "design_space_iter100.png"

# 2D paper breast: single iter-100 RMSE, and two-channel iter-100 RMSE per r
SINGLE_2D = 0.2202
ROW_2D = [
    (0.25, 0.1711),
    (0.5,  0.1686),
    (1.0,  0.1639),
    (1.25, 0.1615),
    (2.0,  0.1552),
    (3.0,  0.1485),
    (4.0,  0.1429),
    (6.0,  0.1345),
    (10.0, 0.1252),
    (15.0, 0.1215),
    (20.0, 0.1218),
    (25.0, 0.1232),
    (30.0, 0.1247),
    (50.0, 0.1333),
]

# 3D analytic breast (50 deg arc): single iter-100 and per-r values
SINGLE_3D = 0.0773
ROW_3D = [
    (0.25, 0.0508),
    (0.5,  0.0508),
    (1.0,  0.0508),
    (1.25, 0.0508),
    (2.0,  0.0508),
    (5.0,  0.0509),
    (10.0, 0.0515),
    (15.0, 0.0525),
    (20.0, 0.0537),
    (30.0, 0.0557),
]


def reduction_pct(rows, single):
    r = np.array([row[0] for row in rows])
    rmse = np.array([row[1] for row in rows])
    return r, 100.0 * (single - rmse) / single   # positive = better than single


def main():
    fig, ax = plt.subplots(figsize=(7.5, 4.4))

    r2, red2 = reduction_pct(ROW_2D, SINGLE_2D)
    r3, red3 = reduction_pct(ROW_3D, SINGLE_3D)

    ax.plot(r2, red2, "o-", color="#1f77b4", lw=2.0, ms=6,
            label="2D breast phantom")
    ax.plot(r3, red3, "s-", color="#d62728", lw=2.0, ms=6,
            label="3D analytic breast")

    # Single-channel baseline (0% reduction)
    ax.axhline(0.0, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.text(0.27, 0.5, "single-channel", fontsize=8, color="k", alpha=0.7)

    # Paper config marker
    ax.axvline(1.25, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.text(1.30, 47, "paper\n$r{=}1.25$", fontsize=9, color="gray",
            ha="left", va="top")

    # 2D optimum marker
    best_idx_2d = int(np.argmax(red2))
    r_best_2d, red_best_2d = r2[best_idx_2d], red2[best_idx_2d]
    ax.annotate(f"2D best ($r{{\\approx}}15$, $-{red_best_2d:.0f}\\%$)",
                xy=(r_best_2d, red_best_2d),
                xytext=(33, red_best_2d - 4),
                fontsize=9, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))

    # 3D plateau annotation
    ax.text(0.27, 36, "3D: flat plateau across $r$;\noptimum near paper config.",
            fontsize=9, color="#d62728")

    ax.set_xscale("log")
    ax.set_xlim(0.2, 60)
    ax.set_ylim(-5, 50)
    ax.set_xlabel(r"per-band tolerance ratio  $r = \varepsilon_{\mathrm{lo}} / \varepsilon_{\mathrm{hi}}$")
    ax.set_ylabel(r"iter-100 RMSE reduction vs single-channel (\%)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower center", fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG}")


if __name__ == "__main__":
    main()
