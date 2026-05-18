"""
Sweep eps_lo / eps_hi ratio at one resolution (256x256 by default) to test
whether tightening the low-frequency data constraint shifts the two-channel
fixed point to a lower at-convergence RMSE than single-channel.

The dyadic CT manuscript (paper/dyadic ct.tex, "Per-band tolerances") notes
that sigma_i only modifies the trajectory while eps_i modifies the solution,
so this is the principled lever for "boosting low frequencies" at convergence.
"""

import sys
import os
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import compare_methods_multiresolution as cm

CACHE_FILE = "cache/eps_lo_sweep_256.pkl"
TABLE_FILE = "final_figures/eps_lo_sweep_256.txt"
PLOT_FILE = "final_figures/eps_lo_sweep_256.png"


def main():
    parser = argparse.ArgumentParser(description="eps_lo ratio sweep at one resolution")
    parser.add_argument("--mfact", type=int, default=2,
                        help="2 -> 256x256 (default), 1 -> 512x512, 4 -> 128x128")
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[0.05, 0.1, 0.25, 0.5, 1.0, 1.25],
                        help="eps_lo / eps_hi ratios to sweep")
    parser.add_argument("--force", action="store_true",
                        help="Recompute all ratios from scratch")
    args = parser.parse_args()

    resolution = int(512 / args.mfact)

    os.makedirs("cache", exist_ok=True)
    os.makedirs("final_figures", exist_ok=True)

    cache_path = CACHE_FILE.replace("256", str(resolution))
    table_path = TABLE_FILE.replace("256", str(resolution))
    plot_path = PLOT_FILE.replace("256", str(resolution))

    if not args.force and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)} cached results from {cache_path}")
    else:
        results = {}

    eps_hi_value = cm.eps

    for ratio in args.ratios:
        if ratio in results:
            print(f"[skip] ratio={ratio} already cached")
            continue

        print(f"\n{'#'*70}\n# eps_lo / eps_hi = {ratio}  (eps_lo = {ratio * eps_hi_value:.6f})\n{'#'*70}")
        cm.eps_hi = eps_hi_value
        cm.eps_lo = ratio * eps_hi_value

        t0 = time.time()
        result = cm.run_reconstruction_for_mfact(args.mfact)
        elapsed = time.time() - t0
        print(f"[done] ratio={ratio} elapsed={elapsed:.1f}s "
              f"single_rmse={result['final_rmse_single']:.6f} "
              f"two_rmse={result['final_rmse_two']:.6f}")

        results[ratio] = {
            "ratio": ratio,
            "eps_hi": eps_hi_value,
            "eps_lo": ratio * eps_hi_value,
            "resolution": resolution,
            "ierrs_single": result["ierrs_single"],
            "ierrs_two": result["ierrs_two"],
            "final_rmse_single": result["final_rmse_single"],
            "final_rmse_two": result["final_rmse_two"],
        }

        with open(cache_path, "wb") as f:
            pickle.dump(results, f)

    sorted_ratios = sorted(results.keys())

    lines = []
    lines.append(f"eps_lo / eps_hi sweep, {resolution}x{resolution}, "
                 f"itermax={cm.RESOLUTION_PARAMS[resolution]['itermax']}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"| eps_lo/eps_hi | eps_lo    | Single RMSE | Two RMSE   | Improvement |")
    lines.append(f"|---------------|-----------|-------------|------------|-------------|")
    for r in sorted_ratios:
        d = results[r]
        improvement = (d["final_rmse_single"] - d["final_rmse_two"]) / d["final_rmse_single"] * 100
        lines.append(f"| {r:<13.3f} | {d['eps_lo']:<9.6f} | {d['final_rmse_single']:<11.6f} | "
                     f"{d['final_rmse_two']:<10.6f} | {improvement:+.2f}%      |")

    table_text = "\n".join(lines)
    print("\n" + table_text)
    with open(table_path, "w") as f:
        f.write(table_text + "\n")
    print(f"\nTable saved to {table_path}")

    plt.figure(figsize=(8, 5))
    first_single_plotted = False
    cmap = plt.get_cmap("viridis")
    for idx, r in enumerate(sorted_ratios):
        d = results[r]
        color = cmap(idx / max(1, len(sorted_ratios) - 1))
        iterations = range(1, len(d["ierrs_two"]) + 1)
        if not first_single_plotted:
            plt.semilogy(iterations, d["ierrs_single"], "k-", linewidth=1.5,
                         label="Single-Channel (baseline)")
            first_single_plotted = True
        plt.semilogy(iterations, d["ierrs_two"], "-", color=color, linewidth=1.2,
                     label=f"Two, eps_lo/eps_hi={r}")

    plt.xlabel("Iteration")
    plt.ylabel("Image RMSE")
    plt.title(f"eps_lo ratio sweep, {resolution}x{resolution}")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
