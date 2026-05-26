"""Plot the new VICTRE combined sweep results.

Reads cache/_victre_combined.txt produced by _sweep_victre_combined.py.
Builds a grouped bar chart: x = arc, color = (sigma, r) combo, y =
iter-400 RMSE reduction (%). Positive favours two-channel.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TXT  = ROOT / "cache" / "_victre_combined.txt"
OUT  = ROOT / "presentation" / "figs" / "victre_combined.png"


def parse():
    rows = []
    for line in TXT.read_text().splitlines():
        m = re.match(
            r"^\s*(\d+)\s+(\S+)\s*\|\s*([\+\-\d\.]+)\s*\|\s*([\+\-\d\.]+)"
            r"\s*\|\s*([\+\-\d\.]+)\s*\|\s*([\+\-\d\.]+)", line)
        if not m or "i100" in line:
            continue
        rows.append({
            "arc":   int(m.group(1)),
            "label": m.group(2),
            "i100":  float(m.group(3)),
            "i200":  float(m.group(4)),
            "i300":  float(m.group(5)),
            "i400":  float(m.group(6)),
        })
    return rows


def main():
    if not TXT.exists():
        print(f"No data at {TXT} yet.")
        return
    rows = parse()
    if not rows:
        print("No completed runs in the cache file yet.")
        return

    # Drop diverged combos.
    rows = [r for r in rows if r["i400"] > -100]
    if not rows:
        print("All runs diverged.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    iters = ["i100", "i200", "i300", "i400"]
    iters_lbl = ["100", "200", "300", "400"]
    n = len(rows)
    x = np.arange(n)
    width = 0.18
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]

    for k, (it, lab, col) in enumerate(zip(iters, iters_lbl, colors)):
        vals = [r[it] for r in rows]
        ax.bar(x + (k - 1.5) * width, vals, width=width, label=f"iter {lab}",
               color=col, edgecolor="k", lw=0.4)

    ax.axhline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
    labels = [f"{r['arc']}°\n{r['label']}" for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"RMSE reduction vs single-channel (\%)")
    ax.set_title("VICTRE: combined (arc, $\\sigma_{lo}/\\sigma_{hi}$, $r$) sweep"
                 " (full config)", fontsize=11)
    ax.legend(fontsize=9, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.30))
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
