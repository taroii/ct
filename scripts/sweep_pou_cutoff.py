"""
Sweep matched-cutoff partition-of-unity (PoU) two-channel filters at one
resolution to test whether eliminating the spectral coverage gap in the
current c_hi=4 / c_lo=8 design recovers an at-convergence two-channel
advantage.

With c_hi = c_lo = c, we get
    han_hi(u) = 1 - H_c(u),   han_lo(u) = H_c(u),
    F_hi(u)^2 + F_lo(u)^2 = ramp(u)
i.e. a true partition of unity, eliminating the non-PoU artifact noted in
the dyadic CT manuscript (paper/dyadic ct.tex). The PoU construction
should pull the two-channel feasible set closer to the single-channel
feasible set (modulo eps split) and tighten ||Sigma^(1/2) K||.

Smaller c -> wider LF band; larger c -> narrower LF band concentrated near DC.
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

CACHE_TEMPLATE = "cache/pou_cutoff_sweep_RES.pkl"
TABLE_TEMPLATE = "final_figures/pou_cutoff_sweep_RES.txt"
PLOT_TEMPLATE = "final_figures/pou_cutoff_sweep_RES.png"


def main():
    parser = argparse.ArgumentParser(description="PoU matched-cutoff sweep")
    parser.add_argument("--mfact", type=int, default=2,
                        help="2 -> 256x256 (default), 1 -> 512x512, 4 -> 128x128")
    parser.add_argument("--cutoffs", type=float, nargs="+",
                        default=[2.0, 4.0, 8.0, 16.0],
                        help="matched cutoff values c (han_hi = 1 - H_c, han_lo = H_c)")
    parser.add_argument("--eps-lo-ratio", type=float, default=1.25,
                        help="eps_lo / eps_hi (default matches the paper)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute all cutoffs from scratch")
    args = parser.parse_args()

    resolution = int(512 / args.mfact)

    os.makedirs("cache", exist_ok=True)
    os.makedirs("final_figures", exist_ok=True)

    cache_path = CACHE_TEMPLATE.replace("RES", str(resolution))
    table_path = TABLE_TEMPLATE.replace("RES", str(resolution))
    plot_path = PLOT_TEMPLATE.replace("RES", str(resolution))

    if not args.force and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)} cached results from {cache_path}")
    else:
        results = {}

    eps_hi_value = cm.eps

    for c in args.cutoffs:
        if c in results:
            print(f"[skip] cutoff={c} already cached")
            continue

        print(f"\n{'#'*70}\n# PoU matched cutoff c = {c}  (han_hi = 1 - H_{c}, han_lo = H_{c})\n{'#'*70}")
        cm.cutoffparm = c
        cm.cutoffparm_lo = c
        cm.eps_hi = eps_hi_value
        cm.eps_lo = args.eps_lo_ratio * eps_hi_value

        t0 = time.time()
        result = cm.run_reconstruction_for_mfact(args.mfact)
        elapsed = time.time() - t0
        print(f"[done] c={c} elapsed={elapsed:.1f}s "
              f"single_rmse={result['final_rmse_single']:.6f} "
              f"two_rmse={result['final_rmse_two']:.6f}")

        results[c] = {
            "cutoff": c,
            "eps_lo_ratio": args.eps_lo_ratio,
            "resolution": resolution,
            "ierrs_single": result["ierrs_single"],
            "ierrs_two": result["ierrs_two"],
            "final_rmse_single": result["final_rmse_single"],
            "final_rmse_two": result["final_rmse_two"],
        }

        with open(cache_path, "wb") as f:
            pickle.dump(results, f)

    sorted_cutoffs = sorted(results.keys())

    lines = []
    lines.append(f"Matched-cutoff PoU sweep, {resolution}x{resolution}, "
                 f"itermax={cm.RESOLUTION_PARAMS[resolution]['itermax']}, "
                 f"eps_lo/eps_hi={args.eps_lo_ratio}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"| cutoff c | LF half-width fraction | Single RMSE | Two RMSE   | Improvement |")
    lines.append(f"|----------|------------------------|-------------|------------|-------------|")
    for c in sorted_cutoffs:
        d = results[c]
        improvement = (d["final_rmse_single"] - d["final_rmse_two"]) / d["final_rmse_single"] * 100
        lines.append(f"| {c:<8.2f} | {1.0/c:<22.4f} | {d['final_rmse_single']:<11.6f} | "
                     f"{d['final_rmse_two']:<10.6f} | {improvement:+.2f}%      |")

    table_text = "\n".join(lines)
    print("\n" + table_text)
    with open(table_path, "w") as f:
        f.write(table_text + "\n")
    print(f"\nTable saved to {table_path}")

    plt.figure(figsize=(8, 5))
    first_single_plotted = False
    cmap = plt.get_cmap("viridis")
    for idx, c in enumerate(sorted_cutoffs):
        d = results[c]
        color = cmap(idx / max(1, len(sorted_cutoffs) - 1))
        iterations = range(1, len(d["ierrs_two"]) + 1)
        if not first_single_plotted:
            plt.semilogy(iterations, d["ierrs_single"], "k-", linewidth=1.5,
                         label="Single-Channel (baseline)")
            first_single_plotted = True
        plt.semilogy(iterations, d["ierrs_two"], "-", color=color, linewidth=1.2,
                     label=f"Two, PoU c={c}")

    plt.xlabel("Iteration")
    plt.ylabel("Image RMSE")
    plt.title(f"PoU matched-cutoff sweep, {resolution}x{resolution}, "
              f"eps_lo/eps_hi={args.eps_lo_ratio}")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
