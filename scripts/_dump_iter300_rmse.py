"""Print iter-100 / iter-300 / iter-500 single- and two-channel RMSE for the
head and jaw 100-deg arc caches, so the slide captions can be updated.
"""
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"


def dump(name, path):
    with open(path, "rb") as f:
        res = pickle.load(f)
    s = res["ierrs_single"]
    t = res["ierrs_two"]
    print(f"--- {name} ({path.name}) ---")
    for it in (100, 200, 300, 500):
        if it - 1 >= len(s):
            continue
        ss = s[it - 1]
        tt = t[it - 1]
        red = 100.0 * (ss - tt) / ss
        print(f"  iter {it:>3}: single={ss:.4f}  two={tt:.4f}  reduction={red:+.1f}%")


for name in ("head", "jaw"):
    dump(name, CACHE / f"ct2_{name}_recon_arc100.pkl")
