"""2D Shepp-Logan recon using the SAME fan-beam pipeline as the
2D paper-breast result (compare_methods_multiresolution.run_reconstruction_for_mfact).

Why: gives the story
    -- our method works on 2D DBT (paper-breast)
    -- our method works on Shepp-Logan (more general, harder LF regime)
    -- our method works on 3D DBT (synthetic A/B)

Setup (matches the 2D paper-breast slide):
    -- 256 x 256 grid, 10 x 10 cm box
    -- circular fan-beam, 25 views over a 50 deg arc, 1024 detector bins,
       SOD=50 cm, SDD=100 cm
    -- alpha=1.9, beta=10.0, rho=1.75, stepbalance=100, sigma_lo_scale=4
    -- cutoffparm=4, cutoffparm_lo=8 (paper config)
    -- 500 iterations
    -- ground-truth phantom is the modified Shepp-Logan rasterised on the
       256 grid and SCALED so that its peak matches the breast peak
       (~1.62), keeping ||g|| in the same regime so eps does not need to
       be retuned. eps_hi/eps_lo are the same values used for breast.

Outputs:
    cache/shepp_logan_2d_recon_256.pkl
    presentation/figs/shepp_logan_2d_phantom_intro.png
    presentation/figs/shepp_logan_2d_iter_ladder.png
    presentation/figs/shepp_logan_2d_error_ladder.png
    presentation/figs/shepp_logan_2d_convergence.png
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import os
os.chdir(ROOT)

import compare_methods_multiresolution as cm  # noqa: E402


CACHE_PATH = ROOT / "cache" / "shepp_logan_2d_recon_256.pkl"
FIG_DIR    = ROOT / "presentation" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MFACT          = 2          # 256 x 256
SNAPSHOT_ITERS = [10, 50, 100, 200, 300, 500]
LADDER_ITERS   = [10, 50, 100, 200, 500]
ITERMAX        = 500

# Target peak matches breast phantom peak (~1.62) so ||g|| is comparable
# and the same eps_hi / eps_lo as the breast paper run can be re-used.
PEAK_SCALE = 1.6

DISPLAY_VMIN = 0.0
DISPLAY_VMAX = 1.6


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def shepp_logan_2d(n):
    """Modified Shepp-Logan in a unit disc, rasterised on an n x n grid.

    Background returns 0.0; skull rim peaks at 1.0; soft-tissue interior
    sits at 0.2 with several 0.1 / 0.2 features. The phantom is then
    scaled externally so its peak matches the paper-breast peak.
    """
    # (A, a, b, x0, y0, phi_deg)
    table = [
        ( 1.0,   .69,    .92,     0,      0,       0),
        (-0.8,   .6624,  .8740,   0,     -.0184,   0),
        (-0.2,   .11,    .31,     .22,    0,      -18),
        (-0.2,   .16,    .41,    -.22,    0,       18),
        ( 0.1,   .21,    .25,     0,      .35,     0),
        ( 0.1,   .046,   .046,    0,      .1,      0),
        ( 0.1,   .046,   .046,    0,     -.1,      0),
        ( 0.1,   .046,   .023,   -.08,   -.605,    0),
        ( 0.1,   .023,   .023,    0,     -.606,    0),
        ( 0.1,   .023,   .046,    .06,   -.605,    0),
    ]
    img = np.zeros((n, n), dtype=np.float64)
    coord = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    for (A, a, b, x0, y0, phi_deg) in table:
        ph = np.deg2rad(phi_deg)
        cos_p, sin_p = np.cos(ph), np.sin(ph)
        xt =  (X - x0) * cos_p + (Y - y0) * sin_p
        yt = -(X - x0) * sin_p + (Y - y0) * cos_p
        mask = (xt / a) ** 2 + (yt / b) ** 2 <= 1.0
        img[mask] += A
    return np.maximum(img, 0.0)


def build_phantom():
    n = int(512 / MFACT)
    sl = shepp_logan_2d(n)
    peak = float(sl.max())
    sl = sl * (PEAK_SCALE / max(peak, 1e-12))
    print(f"Shepp-Logan {n}x{n}: range [{sl.min():.4f}, {sl.max():.4f}], "
          f"mean {sl.mean():.4f}")
    return sl


def load_or_run(force=False):
    if CACHE_PATH.exists() and not force:
        print(f"Loading cache: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    sl = build_phantom()

    # Paper-breast config (identical to the 2D breast slide):
    cm.cutoffparm    = 4.0
    cm.cutoffparm_lo = 8.0
    cm.eps           = 0.001
    cm.eps_hi        = cm.eps
    cm.eps_lo        = 1.25 * cm.eps
    cm.RESOLUTION_PARAMS[256]["itermax"] = ITERMAX

    print(f"\nRunning 256x256 Shepp-Logan with paper-breast hyperparameters")
    print(f"  cutoffparm_hi = {cm.cutoffparm}, cutoffparm_lo = {cm.cutoffparm_lo}")
    print(f"  eps_hi = {cm.eps_hi}, eps_lo = {cm.eps_lo}")
    print(f"  itermax = {ITERMAX}, snapshots at {SNAPSHOT_ITERS}")

    t0 = time.time()
    result = cm.run_reconstruction_for_mfact(
        MFACT,
        snapshot_iters=SNAPSHOT_ITERS,
        phantom_override=sl,
    )
    print(f"Total time: {time.time() - t0:.1f}s")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached -> {CACHE_PATH}")
    return result


def fig_phantom_intro(result, out_path):
    phi = result["phimage"]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    im = ax.imshow(phi.T, cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX,
                    origin="lower")
    ax.set_title("Shepp-Logan ground truth (256x256, peak-scaled to breast range)",
                  fontsize=10)
    _strip(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_iter_ladder(result, iters, out_path):
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]
    phi = result["phimage"]
    n = len(iters)
    fig, axes = plt.subplots(2, n + 1, figsize=(1.9 * (n + 1), 4.4))
    for ax in axes.flat:
        _strip(ax)
    axes[0, 0].imshow(phi.T, cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX,
                      origin="lower")
    axes[0, 0].set_title("ground truth", fontsize=10)
    axes[0, 0].set_ylabel("single-channel", fontsize=11)
    axes[1, 0].imshow(phi.T, cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX,
                      origin="lower")
    axes[1, 0].set_ylabel("two-channel", fontsize=11)
    for i, it in enumerate(iters):
        col = i + 1
        s, t = snaps_s[it], snaps_t[it]
        axes[0, col].imshow(s.T, cmap="gray", vmin=DISPLAY_VMIN,
                            vmax=DISPLAY_VMAX, origin="lower")
        axes[0, col].set_title(f"iter {it}\nRMSE {_rmse(s, phi):.3f}", fontsize=10)
        axes[1, col].imshow(t.T, cmap="gray", vmin=DISPLAY_VMIN,
                            vmax=DISPLAY_VMAX, origin="lower")
        axes[1, col].set_xlabel(f"RMSE {_rmse(t, phi):.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_error_ladder(result, iters, out_path, err_range=0.5):
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]
    phi = result["phimage"]
    n = len(iters)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.4))
    for ax in axes.flat:
        _strip(ax)
    axes[0, 0].set_ylabel("|single - truth|", fontsize=11)
    axes[1, 0].set_ylabel("|two - truth|", fontsize=11)
    for i, it in enumerate(iters):
        es = np.abs(snaps_s[it] - phi).T
        et = np.abs(snaps_t[it] - phi).T
        axes[0, i].imshow(es, cmap="gray", vmin=0.0, vmax=err_range,
                           origin="lower")
        axes[0, i].set_title(f"iter {it}", fontsize=10)
        axes[1, i].imshow(et, cmap="gray", vmin=0.0, vmax=err_range,
                           origin="lower")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_convergence(result, out_path):
    ierrs_s = np.asarray(result["ierrs_single"])
    ierrs_t = np.asarray(result["ierrs_two"])
    iters = np.arange(1, len(ierrs_s) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(iters, ierrs_s, "r-", linewidth=1.4, label="single-channel")
    axes[0].plot(iters, ierrs_t, "b-", linewidth=1.4, label="two-channel")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("image RMSE")
    axes[0].set_title("linear-y, full")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    zlo, zhi = 5, 100
    axes[1].plot(iters[zlo - 1:zhi], ierrs_s[zlo - 1:zhi], "r-", linewidth=1.4,
                 label="single-channel")
    axes[1].plot(iters[zlo - 1:zhi], ierrs_t[zlo - 1:zhi], "b-", linewidth=1.4,
                 label="two-channel")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("image RMSE")
    axes[1].set_title(f"zoom: iter {zlo}-{zhi}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    fig.suptitle("Shepp-Logan 2D fan-beam (25 views / 50 deg arc, 500 iter)",
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = load_or_run(force=args.force)

    fig_phantom_intro(result, FIG_DIR / "shepp_logan_2d_phantom_intro.png")
    fig_iter_ladder(result, LADDER_ITERS,
                     FIG_DIR / "shepp_logan_2d_iter_ladder.png")
    fig_error_ladder(result, LADDER_ITERS,
                      FIG_DIR / "shepp_logan_2d_error_ladder.png")
    fig_convergence(result, FIG_DIR / "shepp_logan_2d_convergence.png")

    print("\n=== iter-by-iter ===")
    print(f"  {'iter':>5}  {'single':>9}  {'two':>9}  {'red %':>7}")
    ierrs_s = result["ierrs_single"]
    ierrs_t = result["ierrs_two"]
    for it in [10, 50, 100, 200, 300, 500]:
        a = ierrs_s[it - 1]; b = ierrs_t[it - 1]
        print(f"  {it:5d}  {a:9.5f}  {b:9.5f}  {(a-b)/a*100:+7.2f}")


if __name__ == "__main__":
    main()
