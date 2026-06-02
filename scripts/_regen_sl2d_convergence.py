"""Regenerate shepp_logan_2d_convergence.png as a single 0-500 panel, no title."""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
d = pickle.load(open(ROOT / "cache" / "shepp_logan_2d_recon_256.pkl", "rb"))
s = np.asarray(d["ierrs_single"]); t = np.asarray(d["ierrs_two"])
its = np.arange(1, len(s) + 1)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(its, s, "r-", lw=1.8, label="single")
ax.plot(its, t, "b-", lw=1.8, label="two-channel")
ax.set_xlim(0, len(s))
ax.set_xlabel("iteration")
ax.set_ylabel("image RMSE")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
out = ROOT / "presentation" / "figs" / "shepp_logan_2d_convergence.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
