"""Regenerate convergence figures for the CT-Meeting 2026 talk.

All read existing recon caches -- no reconstruction is run.

  shepp_logan_convergence.png  <- cache/shepp_logan_recon_arc50.pkl
       The talk uses the realistic 50 deg arc (negative control). The
       default cache shepp_logan_recon.pkl currently holds a 30 deg run;
       this also re-syncs it to the 50 deg data.
  ct2_{breast,head,jaw}_convergence.png  <- cache/ct2_*_recon.pkl
       Extended to a 0-500 iteration window so the full two-channel
       gap is visible; figure titles drop the "ct-2" repo label.

Figure titles drop the "ct-2" repo label.
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT  = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
FIGS  = ROOT / "presentation" / "figs"


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def convergence_fig(res, out, title, xlim=(0, 500)):
    s = np.asarray(res["ierrs_single"])
    t = np.asarray(res["ierrs_two"])
    iters = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(iters, s, "r-", lw=1.4, label="single-channel")
    ax.semilogy(iters, t, "b-", lw=1.4, label="two-channel")
    ax.set_xlabel("iteration")
    ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    # Shepp-Logan -- 50 deg arc (negative control).
    sl = load(CACHE / "shepp_logan_recon_arc50.pkl")
    convergence_fig(
        sl, FIGS / "shepp_logan_convergence.png",
        "Shepp-Logan phantom (50 deg arc): image RMSE vs iteration")
    with open(CACHE / "shepp_logan_recon.pkl", "wb") as f:
        pickle.dump(sl, f)
    print("Synced cache/shepp_logan_recon.pkl <- arc50 cache")

    # Analytic breast / head / jaw -- 0-500 iteration window.
    for name in ("breast", "head", "jaw"):
        res = load(CACHE / f"ct2_{name}_recon.pkl")
        convergence_fig(
            res, FIGS / f"ct2_{name}_convergence.png",
            f"Analytic {name} phantom: image RMSE vs iteration")


if __name__ == "__main__":
    main()
