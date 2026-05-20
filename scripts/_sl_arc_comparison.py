"""Two-panel Shepp-Logan convergence comparison: 50 deg arc (two-channel
ties / slightly loses) vs 30 deg arc (two-channel wins). Demonstrates that
the two-channel advantage switches on as the problem becomes more
under-determined.

Inputs:
  cache/shepp_logan_recon_arc50.pkl   -- 50 deg run
  cache/shepp_logan_recon.pkl         -- 30 deg run (current SL default)
Output:
  presentation/figs/shepp_logan_arc_comparison.png
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG  = ROOT / "presentation" / "figs" / "shepp_logan_arc_comparison.png"


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    r50 = load(ROOT / "cache" / "shepp_logan_recon_arc50.pkl")
    r30 = load(ROOT / "cache" / "shepp_logan_recon.pkl")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, res, deg in zip(axes, (r50, r30), (50, 30)):
        s = np.asarray(res["ierrs_single"])
        t = np.asarray(res["ierrs_two"])
        iters = np.arange(1, len(s) + 1)
        ax.semilogy(iters, s, "r-", lw=1.5, label="single-channel")
        ax.semilogy(iters, t, "b-", lw=1.5, label="two-channel")
        ax.set_xlabel("iteration")
        ax.set_title(f"{deg}$^\\circ$ arc", fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(0, len(s))
        n = min(len(s), len(t))
        gap500 = (s[n-1] - t[n-1]) / s[n-1] * 100
        ax.text(0.97, 0.95, f"iter {n}: {gap500:+.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=10)
    axes[0].set_ylabel("image RMSE")
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Shepp-Logan: two-channel advantage switches on "
                 "as the arc narrows", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG}")


if __name__ == "__main__":
    main()
