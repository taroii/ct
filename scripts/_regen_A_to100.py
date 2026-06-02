"""Regenerate two synthetic-A figures restricted to iter <=100 for the slide:

  ct2_breast_synthetic_A_powerlaw_iter_ladder.png   -> gt + iters 25,50,75,100
  ct2_breast_synthetic_A_powerlaw_convergence.png   -> linear, iter 0-100, no titles

The cached A run snapshotted iters [10,50,100,200,300,500]; 25 & 75 were not
saved, so we re-run A's recon to itermax=100 with snapshots [25,50,75,100].
The phantom builder is deterministic (seed 42) so this reproduces the same
phantom as the committed v5 (anisotropic islands) result.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import _run_synthetic_phantoms as srp  # noqa: E402
import victre_reconstruction as vr  # noqa: E402

FIG_DIR = ROOT / "presentation" / "figs"
LADDER_ITERS = [25, 50, 75, 100]
SNAPS = [25, 50, 75, 100]
ITERMAX = 100


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def main():
    astra_shape = (srp.SHAPE[2], srp.SHAPE[1], srp.SHAPE[0])  # (NZ,NY,NX)
    phantom = srp.build_phantom("A_powerlaw", astra_shape, srp.DX_CM)

    det_row, det_col, det_sp = srp.DET
    vol_geom, proj_geom, gi = vr.build_dbt_geometry(
        phantom.shape, srp.DX_CM,
        det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
        nviews=srp.NVIEWS, arc_deg=srp.ARC_DEG, sod=srp.SOD_CM, odd=srp.ODD_CM,
    )
    A, At = vr.make_projector(vol_geom, proj_geom)
    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"]
    )

    saved = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = ITERMAX
    try:
        srp.reset_cp()
        print("\n--- single (itermax=100) ---")
        t0 = time.time()
        _, is_, _, _, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPS,
        )
        print(f"  single {time.time()-t0:.0f}s")

        srp.reset_cp()
        for k, v in srp.CP_OVERRIDES.items():
            vr.CONFIG[k] = v
        R_hi, R_lo = vr.build_sinogram_filters(
            gi["det_col_count"], gi["det_spacing"],
            vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
            axis="2d", det_row_count=gi["det_row_count"],
        )
        print("\n--- two (itermax=100) ---")
        t0 = time.time()
        _, it_, _, _, snaps_t = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPS,
        )
        print(f"  two {time.time()-t0:.0f}s")
    finally:
        vr.CONFIG["itermax"] = saved
        srp.reset_cp()

    # --- iter ladder: gt + 25,50,75,100 (5 columns) ---
    slc_gt = srp._xy_slice(phantom)
    n = len(LADDER_ITERS)
    fig, axes = plt.subplots(2, n + 1, figsize=(1.9 * (n + 1), 3.8))
    for ax in axes.flat:
        _strip(ax)
    for r, lab in enumerate(["single", "two-channel"]):
        axes[r, 0].imshow(slc_gt, cmap="gray", vmin=srp.DISPLAY["vmin"],
                          vmax=srp.DISPLAY["vmax"], origin="lower")
        axes[r, 0].set_ylabel(lab, fontsize=11)
    axes[0, 0].set_title("ground truth", fontsize=10)
    for i, it in enumerate(LADDER_ITERS):
        col = i + 1
        axes[0, col].set_title(f"iter {it}", fontsize=10)
        axes[0, col].imshow(srp._xy_slice(snaps_s[it]), cmap="gray",
                            vmin=srp.DISPLAY["vmin"], vmax=srp.DISPLAY["vmax"],
                            origin="lower")
        axes[1, col].imshow(srp._xy_slice(snaps_t[it]), cmap="gray",
                            vmin=srp.DISPLAY["vmin"], vmax=srp.DISPLAY["vmax"],
                            origin="lower")
    plt.tight_layout()
    out_ladder = FIG_DIR / "ct2_breast_synthetic_A_powerlaw_iter_ladder.png"
    plt.savefig(out_ladder, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_ladder}")

    # --- convergence: linear, iter 0-100, no titles ---
    s = np.asarray(is_); t = np.asarray(it_)
    its = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(its, s, "r-", lw=1.8, label="single")
    ax.plot(its, t, "b-", lw=1.8, label="two-channel")
    ax.set_xlim(0, 100)
    ax.set_xlabel("iteration")
    ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    out_conv = FIG_DIR / "ct2_breast_synthetic_A_powerlaw_convergence.png"
    plt.savefig(out_conv, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_conv}")

    print("\n=== iter-by-iter (A, to 100) ===")
    for it in LADDER_ITERS:
        a = is_[it - 1]; b = it_[it - 1]
        print(f"  {it:4d}  single {a:.5f}  two {b:.5f}  red {(a-b)/a*100:+.2f}%")


if __name__ == "__main__":
    main()
