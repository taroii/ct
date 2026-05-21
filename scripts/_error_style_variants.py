"""Generate the analytic-phantom difference plots in three styles so the
team can choose one:

  signed     -- signed grayscale (current): black under, mid-gray 0, white over
  magnitude  -- absolute error in grayscale: black 0, white largest
  diverging  -- colourblind-safe diverging map: blue under, white 0, red over

Outputs (presentation/figs/):
  ct2_{head,jaw}_error_{signed,magnitude,diverging}.png
Reads cached recons; no reconstruction is run.
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ct-2"))

import matplotlib
matplotlib.use("Agg")

import presentation_ct2_phantom_ladder as pl   # noqa: E402

CACHE = ROOT / "cache"
FIGS  = ROOT / "presentation" / "figs"
ITERS = [50, 100, 200, 500]


def main():
    for name in ("head", "jaw"):
        cfg = pl.PHANTOM_CONFIGS[name]
        with open(CACHE / f"ct2_{name}_recon.pkl", "rb") as f:
            result = pickle.load(f)
        for style in ("signed", "magnitude", "diverging"):
            pl.fig_error_ladder(
                result, ITERS,
                FIGS / f"ct2_{name}_error_{style}.png",
                crop=cfg["roi"], style=style)


if __name__ == "__main__":
    main()
