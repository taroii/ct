"""
3D LAR cone-beam reconstruction of the full VICTRE breast phantom with
snapshot-iter support, producing an iter-ladder slice figure for the
CT-Meeting 2026 talk.

Mirrors scripts/presentation_ct2_breast_ladder.py but with the .raw VICTRE
phantom as the volume source.

Outputs (under presentation/figs/):
  victre_iter_ladder_xy.png   xy mid-slice single vs two across iterations
  victre_iter_ladder_roi.png  same iters cropped to a glandular ROI
  victre_iter_ladder_convergence.png

Cache:  cache/victre_iter_ladder.pkl

Run:    python scripts/presentation_victre_ladder.py [--force]
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# victre_reconstruction.py resolves PHANTOM_MHD / PHANTOM_RAW relative to
# the current working directory, so make sure we run from the repo root.
os.chdir(ROOT)

import victre_reconstruction as vr   # noqa: E402

CACHE_PATH = ROOT / "cache" / "victre_iter_ladder.pkl"
FIG_DIR = ROOT / "presentation" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_ITERS = [10, 50, 100, 200, 300, 500]
ITERMAX = 500


def run_recon():
    phantom, dx_cm = vr.load_and_downsample_phantom()
    vol_geom, proj_geom, geom_info = vr.build_geometry(phantom.shape, dx_cm)
    A, At = vr.make_projector(vol_geom, proj_geom)

    R_hi, R_lo = vr.build_sinogram_filters(
        geom_info["det_col_count"], geom_info["det_spacing"],
        vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
    )

    sino_shape = (geom_info["det_row_count"],
                  geom_info["nviews"],
                  geom_info["det_col_count"])
    vr.adjoint_test(A, At, phantom.shape, sino_shape)

    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"]
    )

    saved_itermax = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = ITERMAX
    try:
        print("\n--- single-channel ---")
        rs, is_, ds_, ts_, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            geom_info["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        print("\n--- two-channel ---")
        rt, it_, dt_, tt_, snaps_t = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            geom_info["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
    finally:
        vr.CONFIG["itermax"] = saved_itermax

    return {
        "phantom": phantom,
        "recon_single": rs, "recon_two": rt,
        "ierrs_single": is_, "ierrs_two": it_,
        "derrs_single": ds_, "derrs_two": dt_,
        "tvs_single": ts_, "tvs_two": tt_,
        "snapshots_single": snaps_s,
        "snapshots_two":    snaps_t,
        "dx_cm": dx_cm,
        "geometry": geom_info,
    }


def load_or_run(force):
    if CACHE_PATH.exists() and not force:
        print(f"Loading cached recon from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    res = run_recon()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(res, f)
    print(f"Cached recon -> {CACHE_PATH}")
    return res


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _xy_slice(vol):
    nz = vol.shape[0]
    z = nz // 2
    return vol[z-1:z+2].mean(axis=0)


def fig_iter_ladder(result, iters, out_path, title=None, crop=None):
    phi = result["phantom"]
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]

    slc_gt = _xy_slice(phi)
    vmin = 0.0
    vmax = float(np.percentile(slc_gt[slc_gt > 0], 99.0)) * 1.05

    def crop_slice(s):
        if crop is None:
            return s
        r0, r1, c0, c1 = crop
        return s[r0:r1, c0:c1]

    n = len(iters)
    fig, axes = plt.subplots(3, n + 1, figsize=(1.9 * (n + 1), 5.6))
    for ax in axes.flat:
        _strip(ax)

    gt_slice = crop_slice(slc_gt)
    for r, label in enumerate(["ground truth", "single", "two-channel"]):
        axes[r, 0].imshow(gt_slice, cmap="gray", vmin=vmin, vmax=vmax,
                          origin="lower")
        axes[r, 0].set_ylabel(label, fontsize=11)
    axes[0, 0].set_title("ground truth", fontsize=10)

    for i, it in enumerate(iters):
        col = i + 1
        s = crop_slice(_xy_slice(snaps_s[it]))
        t = crop_slice(_xy_slice(snaps_t[it]))
        axes[0, col].imshow(gt_slice, cmap="gray", vmin=vmin, vmax=vmax,
                            origin="lower")
        axes[0, col].set_title(f"iter {it}", fontsize=10)
        axes[1, col].imshow(s, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[2, col].imshow(t, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_phantom_intro(result, out_path, title=None):
    """Three-pane view of the full VICTRE phantom: axial, coronal,
    sagittal mid-slices. Used as the intro slide before the iter ladder.
    """
    phi = result["phantom"]                  # (NZ, NY, NX) astra-order
    nz, ny, nx = phi.shape
    axial    = phi[nz // 2]
    coronal  = phi[:, ny // 2, :]
    sagittal = phi[:, :, nx // 2]

    nonzero = phi[phi > 1e-4]
    vmax = float(np.percentile(nonzero, 99.0)) * 1.05 if nonzero.size else 1.0
    vmin = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, img, label in zip(
        axes,
        (axial, coronal, sagittal),
        ("axial (xy)", "coronal (xz)", "sagittal (yz)"),
    ):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                  origin="lower", aspect="equal")
        ax.set_title(label, fontsize=11)
        _strip(ax)
    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_convergence(result, out_path, title=None):
    s = result["ierrs_single"]
    t = result["ierrs_two"]
    iters = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(iters, s, "r-", lw=1.4, label="single-channel")
    ax.semilogy(iters, t, "b-", lw=1.4, label="two-channel")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    if title is not None:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = load_or_run(args.force)

    fig_phantom_intro(
        result,
        FIG_DIR / "victre_phantom_intro.png",
        title="VICTRE voxelized breast phantom (0.4 mm after 4x downsampling)",
    )

    fig_iter_ladder(
        result, SNAPSHOT_ITERS,
        FIG_DIR / "victre_iter_ladder_xy.png",
        title="VICTRE phantom -- mid-axial slice across iterations "
              "(25 views / 50 deg CBCT-LAR)",
    )

    # ROI: pick a glandular patch. Slice shape depends on the .raw phantom.
    # We compute a generic centered window inside the breast tissue.
    nz, ny, nx = result["phantom"].shape
    z = nz // 2
    sl = result["phantom"][z]
    nonzero = np.argwhere(sl > 1e-4)
    if nonzero.size:
        r0, c0 = nonzero.min(axis=0)
        r1, c1 = nonzero.max(axis=0)
        # 1/3 of the breast bbox, biased lateral
        h = (r1 - r0) // 3
        w = (c1 - c0) // 3
        cy = (r0 + r1) // 2
        cx = (c0 + c1) // 2 + (c1 - c0) // 6
        crop = (max(cy - h, 0), min(cy + h, ny),
                max(cx - w, 0), min(cx + w, nx))
    else:
        crop = None

    fig_iter_ladder(
        result, SNAPSHOT_ITERS,
        FIG_DIR / "victre_iter_ladder_roi.png",
        title="VICTRE phantom -- glandular ROI across iterations",
        crop=crop,
    )

    fig_convergence(
        result,
        FIG_DIR / "victre_iter_ladder_convergence.png",
        title="VICTRE phantom: image RMSE vs iteration",
    )


if __name__ == "__main__":
    main()
