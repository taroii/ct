"""Inspect phantom / reconstruction value ranges and difference-image
magnitudes, to calibrate display windows for the talk figures."""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

CASES = [
    ("2D-256", "iter_ladder_paper_256.pkl", "phimage"),
    ("breast", "ct2_breast_recon.pkl",      "phantom"),
    ("head",   "ct2_head_recon.pkl",        "phantom"),
    ("jaw",    "ct2_jaw_recon.pkl",         "phantom"),
]


def main():
    for name, fn, pkey in CASES:
        with open(CACHE / fn, "rb") as f:
            r = pickle.load(f)
        phi = np.asarray(r[pkey])
        print(f"\n=== {name} ===")
        print(f"  phantom : min={phi.min():.3f} max={phi.max():.3f} "
              f"p99={np.percentile(phi,99):.3f} "
              f"p99.9={np.percentile(phi,99.9):.3f}")
        for ch in ("single", "two"):
            snaps = r[f"snapshots_{ch}"]
            for it in sorted(snaps):
                s = np.asarray(snaps[it])
                d = s - phi
                print(f"  {ch:6s} it{it:<4d}: "
                      f"recon max={s.max():.3f} p99={np.percentile(s,99):.3f}"
                      f"  |diff| p99={np.percentile(np.abs(d),99):.3f} "
                      f"max={np.abs(d).max():.3f}")


if __name__ == "__main__":
    main()
