"""Regenerate head/jaw 100-deg convergence figures truncated at iter 300.

The original figures plotted RMSE through iter 500, but semi-convergence
kicks in and the RMSE rises after ~iter 100-200; the talk now presents
results at iter 300 instead. This script re-reads the existing recon
caches and saves trimmed convergence plots without re-running the
reconstruction.
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
FIGS = ROOT / "presentation" / "figs"

JOBS = [
    ("head", "ct2_head_recon_arc100.pkl", "ct2_head_convergence_arc100.png"),
    ("jaw",  "ct2_jaw_recon_arc100.pkl",  "ct2_jaw_convergence_arc100.png"),
]
XMAX = 300


def main():
    for name, cache_name, out_name in JOBS:
        with open(CACHE / cache_name, "rb") as f:
            res = pickle.load(f)
        s = np.asarray(res["ierrs_single"])
        t = np.asarray(res["ierrs_two"])
        iters = np.arange(1, len(s) + 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(iters, s, "r-", lw=1.6, label="single-channel")
        ax.semilogy(iters, t, "b-", lw=1.6, label="two-channel")
        ax.set_xlabel("iteration")
        ax.set_ylabel("image RMSE")
        ax.set_xlim(0, XMAX)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=10)
        plt.tight_layout()
        out = FIGS / out_name
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
