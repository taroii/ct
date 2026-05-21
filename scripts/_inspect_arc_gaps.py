"""Report two-channel vs single image-RMSE for the analytic phantoms at
50 / 75 / 100 deg arc: gap at a few iterations, the RMSE minimum (best
transient image) and where it occurs, and the iter-500 value."""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"


def gap(s, t, it):
    a, b = s[it - 1], t[it - 1]
    return (a - b) / a * 100.0


def main():
    for name in ("breast", "head", "jaw"):
        print(f"\n=== {name} ===")
        for arc in (50, 75, 100):
            fn = (CACHE / f"ct2_{name}_recon.pkl" if arc == 50
                  else CACHE / f"ct2_{name}_recon_arc{arc}.pkl")
            with open(fn, "rb") as f:
                r = pickle.load(f)
            s = np.asarray(r["ierrs_single"])
            t = np.asarray(r["ierrs_two"])
            print(f" arc{arc:>3}:  gap i100 {gap(s,t,100):+5.1f}  "
                  f"i200 {gap(s,t,200):+5.1f}  i500 {gap(s,t,500):+5.1f}   |  "
                  f"single  min {s.min():.4f}@{s.argmin()+1:<3d} end {s[-1]:.4f}   |  "
                  f"two  min {t.min():.4f}@{t.argmin()+1:<3d} end {t[-1]:.4f}")


if __name__ == "__main__":
    main()
