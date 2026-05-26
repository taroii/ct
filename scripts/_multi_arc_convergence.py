"""Multi-arc convergence comparison for the analytic head and jaw
phantoms. Shows single + two-channel image RMSE vs iteration at 50, 75
and 100 deg arcs side by side, so the audience can read the gap at the
smaller arcs (where the visual ladder is harder to call) right next to
the wider arcs (where the visual win is obvious).

Output: presentation/figs/multi_arc_convergence_head_jaw.png
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT  = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT   = ROOT / "presentation" / "figs" / "multi_arc_convergence_head_jaw.png"

ARCS    = [50, 75, 100]
COLORS  = {50: "#C00000", 75: "#1F8A00", 100: "#1F4E79"}  # red / green / blue


def load(name, arc):
    fn = (CACHE / f"ct2_{name}_recon.pkl" if arc == 50
          else CACHE / f"ct2_{name}_recon_arc{arc}.pkl")
    with open(fn, "rb") as f:
        return pickle.load(f)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             constrained_layout=True)
    for ax, name in zip(axes, ("head", "jaw")):
        for arc in ARCS:
            r = load(name, arc)
            s = np.asarray(r["ierrs_single"])
            t = np.asarray(r["ierrs_two"])
            iters = np.arange(1, len(s) + 1)
            ax.semilogy(iters, s, ls="--", color=COLORS[arc], lw=1.3,
                        alpha=0.85, label=f"single @ {arc}$^\\circ$")
            ax.semilogy(iters, t, ls="-",  color=COLORS[arc], lw=1.7,
                        label=f"two @ {arc}$^\\circ$")
        ax.set_xlabel("iteration")
        ax.set_ylabel("image RMSE")
        ax.set_title(f"Analytic {name}", fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
