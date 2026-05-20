"""
500-iter persistence plot (single vs two-channel) across 128/256/512.

Pulls the cached `multiresolution_results.pkl` (built with itermax=500
at the time of `09afde4`-era paper config) and emits a 1x3 convergence
subplot. Falls back to truncating longer arrays at iter 500 if the cache
is regenerated with a different itermax later.

Output: presentation/figs/persistence_500iter.png
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "multiresolution_results.pkl"
OUT = ROOT / "presentation" / "figs" / "persistence_500iter.png"


def main():
    with open(CACHE, "rb") as f:
        r = pickle.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, res in zip(axes, [512, 256, 128]):
        d = r[res]
        s = np.asarray(d["ierrs_single"])[:500]
        t = np.asarray(d["ierrs_two"])[:500]
        iters = np.arange(1, len(s) + 1)
        ax.semilogy(iters, s, "r-", lw=1.4, label="single-channel")
        ax.semilogy(iters, t, "b-", lw=1.4, label="two-channel")
        ax.set_xlabel("iteration")
        ax.set_title(f"${res}\\times{res}$")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(0, 500)
        if res == 512:
            ax.set_ylabel("image RMSE")
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
