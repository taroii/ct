"""Print a summary table of single vs two-channel RMSE for every phantom
cache, at key iterations. Useful for the gap-audit step of the pivot.
"""
import pickle
import os
from pathlib import Path

CACHE = Path("cache")

phantoms = [
    ("2D 512^2",          "multiresolution_results.pkl", "ierrs_single", "ierrs_two",   512),
    ("2D 256^2",          "multiresolution_results.pkl", "ierrs_single", "ierrs_two",   256),
    ("2D 128^2",          "multiresolution_results.pkl", "ierrs_single", "ierrs_two",   128),
    ("ct-2 breast",       "ct2_breast_recon.pkl",        "ierrs_single", "ierrs_two",   None),
    ("ct-2 head",         "ct2_head_recon.pkl",          "ierrs_single", "ierrs_two",   None),
    ("ct-2 jaw",          "ct2_jaw_recon.pkl",           "ierrs_single", "ierrs_two",   None),
    ("Shepp-Logan",       "shepp_logan_recon.pkl",       "ierrs_single", "ierrs_two",   None),
    ("VICTRE",            "victre_iter_ladder.pkl",      "ierrs_single", "ierrs_two",   None),
]

iters_of_interest = (50, 100, 200, 300, 500)

print(f"{'Phantom':<18} | " + " | ".join(f"iter {i:3d}" for i in iters_of_interest))
print("-" * (18 + 3 + 12 * len(iters_of_interest)))

for name, fname, key_s, key_t, res in phantoms:
    path = CACHE / fname
    if not path.exists():
        print(f"{name:<18} | (no cache)")
        continue
    with open(path, "rb") as f:
        r = pickle.load(f)
    if res is not None:
        r = r[res]
    cells = []
    for it in iters_of_interest:
        if it - 1 < len(r[key_s]):
            rs = r[key_s][it-1]
            rt = r[key_t][it-1]
            gap = (rs - rt) / rs * 100
            cells.append(f"{gap:+5.1f}%")
        else:
            cells.append("  --  ")
    print(f"{name:<18} | " + " | ".join(f"{c:>8}" for c in cells))
