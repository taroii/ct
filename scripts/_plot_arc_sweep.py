"""Plot the existing analytic-phantom arc sweep results.

Reads cache/_ct2_arc_sweep.txt produced by _sweep_ct2_arcs.py and turns
the table into a line chart: two-channel vs single image-RMSE reduction
(%) versus arc angle, one curve per phantom.

The y-axis is positive-favours-two-channel reduction percentage. The
sweep is at the reduced config (itermax 250, halved detector) so this is
a trend plot, not an absolute number for the talk.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TXT = ROOT / "cache" / "_ct2_arc_sweep.txt"
OUT = ROOT / "presentation" / "figs" / "analytic_arc_sweep.png"


def parse():
    rows = []
    for line in TXT.read_text().splitlines():
        m = re.match(r"^(\w+)\s+(\d+)\s*\|\s*([\+\-\d\.]+)\s*\|\s*"
                     r"([\+\-\d\.]+)\s*\|\s*([\+\-\d\.]+)\s*\|\s*([\+\-\d\.]+)",
                     line)
        if not m:
            continue
        name, arc, i100, i150, i200, i250 = m.groups()
        rows.append({
            "phantom": name,
            "arc": float(arc),
            "i100": float(i100),
            "i150": float(i150),
            "i200": float(i200),
            "i250": float(i250),
        })
    return rows


def main():
    rows = parse()
    by_phantom = {}
    for r in rows:
        by_phantom.setdefault(r["phantom"], []).append(r)
    for k in by_phantom:
        by_phantom[k].sort(key=lambda r: r["arc"])

    colors = {"breast": "#1f77b4", "head": "#d62728", "jaw": "#2ca02c"}
    markers = {"breast": "o", "head": "s", "jaw": "^"}

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for name, rs in by_phantom.items():
        arcs = [r["arc"] for r in rs]
        i200 = [r["i200"] for r in rs]
        ax.plot(arcs, i200, color=colors[name], marker=markers[name],
                lw=1.8, ms=7, label=f"analytic {name}")

    ax.axhline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.text(arcs[0] + 1, 0.7, "single-channel", fontsize=8, color="k", alpha=0.7)
    ax.axvspan(50, 100, alpha=0.06, color="gray")
    ax.text(75, 39, "[50, 100] range of interest",
            fontsize=9, color="gray", ha="center")

    ax.set_xlabel(r"source arc (degrees)")
    ax.set_ylabel("iter-200 RMSE reduction vs single-channel (\\%)")
    ax.set_title("Two-channel gain across source arcs"
                 " (reduced-config sweep)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)
    ax.set_ylim(-10, 42)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
