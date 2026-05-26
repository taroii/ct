"""Plot existing VICTRE sweep results.

Reads:
  cache/_victre_sweep.txt      reduced-config arc + sigma combinations.
  cache/_victre_cp_sweep.txt   full-config CP-parameter sweep at arc=50.

Produces a two-panel summary figure:
  Left  - reduced-config arc sweep (base + siglo8) vs arc.
  Right - full-config CP sweep at 50 deg, bar chart per combo at iter 400.

Both panels report two-channel image-RMSE reduction (%) vs single-channel.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ARC_TXT = ROOT / "cache" / "_victre_sweep.txt"
CP_TXT  = ROOT / "cache" / "_victre_cp_sweep.txt"
OUT     = ROOT / "presentation" / "figs" / "victre_existing_sweeps.png"


def parse_arc():
    rows = []
    for line in ARC_TXT.read_text().splitlines():
        m = re.match(r"^(\S+)\s+arc\s+(\d+)\s*\|\s*([\+\-\d\.]+)\s*\|\s*"
                     r"([\+\-\d\.]+)\s*\|\s*([\+\-\d\.]+)\s*\|\s*([\+\-\d\.]+)",
                     line)
        if not m:
            continue
        name, arc, *gaps = m.groups()
        rows.append({"label": name, "arc": float(arc),
                     "gaps": [float(g) for g in gaps]})
    return rows


def parse_cp():
    rows = []
    for line in CP_TXT.read_text().splitlines():
        m = re.match(r"^(\S+)\s*\|\s*([\+\-\d\.]+)\s*\|\s*([\+\-\d\.]+)"
                     r"\s*\|\s*([\+\-\d\.]+)\s*\|\s*([\+\-\d\.]+)", line)
        if not m or m.group(1) == "combo":
            continue
        rows.append({"label": m.group(1),
                     "i100": float(m.group(2)),
                     "i200": float(m.group(3)),
                     "i300": float(m.group(4)),
                     "i400": float(m.group(5))})
    return rows


def main():
    arc_rows = parse_arc()
    cp_rows  = parse_cp()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    # --- Left panel: arc sweep (reduced config) ---
    ax = axes[0]
    base = sorted([r for r in arc_rows if "siglo" not in r["label"]],
                  key=lambda r: r["arc"])
    siglo = sorted([r for r in arc_rows if "siglo" in r["label"]],
                   key=lambda r: r["arc"])

    if base:
        arcs_b = [r["arc"] for r in base]
        i250_b = [r["gaps"][3] for r in base]
        ax.plot(arcs_b, i250_b, "o-", color="#1f77b4", lw=1.8, ms=7,
                label=r"base ($\sigma_{lo}/\sigma_{hi}=4$)")
    if siglo:
        arcs_s = [r["arc"] for r in siglo]
        i250_s = [r["gaps"][3] for r in siglo]
        ax.plot(arcs_s, i250_s, "s-", color="#d62728", lw=1.8, ms=7,
                label=r"aggressive LF step ($\sigma_{lo}/\sigma_{hi}=8$)")

    ax.axhline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("source arc (degrees)")
    ax.set_ylabel(r"iter-250 RMSE reduction vs single (\%)")
    ax.set_title("VICTRE: arc vs CP-step sweep (reduced config)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")

    # --- Right panel: CP sweep at fixed 50 deg (full config) ---
    # Drop diverged combos (impossibly negative); annotate as DIVERGED.
    ax = axes[1]
    kept = [r for r in cp_rows if r["i400"] > -100]
    dropped = [r for r in cp_rows if r["i400"] <= -100]
    labels = [r["label"] for r in kept]
    vals = [r["i400"] for r in kept]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
    bars = ax.barh(range(len(labels)), vals, color=colors, alpha=0.7,
                   edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0.0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel(r"iter-400 RMSE reduction vs single (\%)")
    ax.set_title(r"VICTRE CP sweep at $50^\circ$ (full config)", fontsize=11)
    ax.grid(True, alpha=0.25, axis="x")
    for bar, v in zip(bars, vals):
        ax.text(v - 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}", va="center", ha="right", fontsize=8)
    if dropped:
        names = ", ".join(r["label"] for r in dropped)
        ax.text(0.02, 0.02, f"diverged: {names}",
                transform=ax.transAxes, fontsize=8, color="gray")
    ax.invert_yaxis()
    ax.set_xlim(min(vals) - 5, 5)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
