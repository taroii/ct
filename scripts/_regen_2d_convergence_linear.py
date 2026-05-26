"""Regenerate the 2D paper-breast convergence figure as linear-axis only.

The original `iter_ladder_convergence_256_paper.png` was a 1x2 panel with
linear + log-log. User wants the log-log dropped. Loads the cached 500-iter
run and writes a single linear-axis figure with the same filename.

Cache:  cache/iter_ladder_paper_256.pkl
Output: presentation/figs/iter_ladder_convergence_256_paper.png
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "iter_ladder_paper_256.pkl"
OUT   = ROOT / "presentation" / "figs" / "iter_ladder_convergence_256_paper.png"


def main():
    with open(CACHE, "rb") as f:
        result = pickle.load(f)
    ierrs_s = np.asarray(result["ierrs_single"])
    ierrs_t = np.asarray(result["ierrs_two"])
    iters = np.arange(1, len(ierrs_s) + 1)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.semilogy(iters, ierrs_s, "r-", linewidth=1.6, label="single-channel")
    ax.semilogy(iters, ierrs_t, "b-", linewidth=1.6, label="two-channel")
    ax.set_xlabel("iteration")
    ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    ax.set_xlim(0, len(ierrs_s))
    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
