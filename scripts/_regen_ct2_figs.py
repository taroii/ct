"""Regenerate the analytic-phantom (breast/head/jaw) intro and
iteration-ladder figures from cached recons, with the embedded figure
title removed entirely (it carried the 'ct-2' repo label, and the slide
frame title already names the phantom). No reconstruction is run.
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


def main():
    for name in ("breast", "head", "jaw"):
        cfg = pl.PHANTOM_CONFIGS[name]
        with open(CACHE / f"ct2_{name}_recon.pkl", "rb") as f:
            result = pickle.load(f)
        display = cfg.get("display")

        pl.fig_phantom_intro(
            result, FIGS / f"ct2_{name}_phantom_intro.png",
            title=None, display=display)
        pl.fig_iter_ladder(
            result, cfg["ladder_iters"],
            FIGS / f"ct2_{name}_iter_ladder_xy.png",
            title=None, display=display)
        pl.fig_iter_ladder(
            result, cfg["ladder_iters"],
            FIGS / f"ct2_{name}_iter_ladder_roi.png",
            title=None, crop=cfg["roi"], display=display)


if __name__ == "__main__":
    main()
