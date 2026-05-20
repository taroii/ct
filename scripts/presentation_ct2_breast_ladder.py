"""
3D LAR cone-beam reconstruction of the ct-2 analytic breast phantom, with
single- vs two-channel iteration-ladder slice figures for the CT-Meeting
2026 talk.

Reuses the projector / norm / CP machinery in scripts/victre_reconstruction.py
(now snapshot-aware) but feeds it the ct-2 breast volume instead of the
VICTRE .raw.

Volume: 144 x 144 x 32 voxels at 1.5 mm isotropic = 21.6 x 21.6 x 4.8 cm.
The breast phantom natural extent is x = [-2.5, 8.5], y = [-10, 10],
z = [-3, 3] cm; we center it so x_center sits at 3 cm (mid-breast) and
clip the z extent to +-2.4 cm.

Outputs (all under presentation/figs/):
  ct2_breast_iter_ladder_xy.png    single vs two at iters [10, 50, 100, 200, 300, 500]
                                    mid-axial (xy) slice, both methods on
                                    top of ground truth row
  ct2_breast_iter_ladder_roi.png   same iters, cropped to a soft-tissue ROI
                                    on the lateral side
  ct2_breast_convergence.png       image-RMSE vs iter, single vs two

Cache:  cache/ct2_breast_recon.pkl

Run:    python scripts/presentation_ct2_breast_ladder.py [--force]
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ct-2"))

import victre_reconstruction as vr   # noqa: E402
from phantom3d import image3D        # noqa: E402
from breast_phantom_demo import build_breast_phantom  # noqa: E402


# ---------------------------------------------------------------------------
# Speed patch: cache the (X, Y, Z) meshgrid at float32 so each gen_object
# embed_in reuses the same arrays instead of re-allocating 16 MB of float64.
# With 49 components in the breast phantom this turns ~6 min into seconds.
# ---------------------------------------------------------------------------
def _install_grid_cache():
    _orig_grids = image3D._grids
    def _cached_grids(self):
        key = (self.nx, self.ny, self.nz,
               self.x0, self.y0, self.z0,
               self.dx, self.dy, self.dz)
        cache = getattr(self, "_grid_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        x = self.x0 + (np.arange(self.nx, dtype=np.float32) + 0.5) * self.dx
        y = self.y0 + (np.arange(self.ny, dtype=np.float32) + 0.5) * self.dy
        z = self.z0 + (np.arange(self.nz, dtype=np.float32) + 0.5) * self.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        X = X.astype(np.float32, copy=False)
        Y = Y.astype(np.float32, copy=False)
        Z = Z.astype(np.float32, copy=False)
        self._grid_cache = (key, (X, Y, Z))
        return (X, Y, Z)
    image3D._grids = _cached_grids
    return _orig_grids


_install_grid_cache()

CACHE_PATH = ROOT / "cache" / "ct2_breast_recon.pkl"
FIG_DIR = ROOT / "presentation" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Volume sampling
# ---------------------------------------------------------------------------
DX_CM = 0.15          # 1.5 mm isotropic voxel
NX = 144              # 21.6 cm x extent
NY = 144              # 21.6 cm y extent
NZ = 32               # 4.8 cm z extent (clips top/bottom of natural breast)

# Snapshot ladder. SNAPSHOT_ITERS is what we recorded during the recon
# (cached in ct2_breast_recon.pkl); LADDER_ITERS is the (smaller) subset
# shown in the talk's ROI / xy ladder figures. We drop 300/500 because the
# reconstruction is in its regularization-mismatch regime past iter ~150
# (RMSE grows again -- see the convergence figure) and the late frames
# distract from the LF-wobble story.
SNAPSHOT_ITERS = [10, 50, 100, 200, 300, 500]
LADDER_ITERS   = [10, 50, 100, 200]
ITERMAX = 500


def build_breast_volume():
    """Embed the ct-2 breast phantom in an isotropic-voxel image volume.

    Centers the breast around (x=3 cm, y=0, z=0) inside the image box.
    Returns array in (Z, Y, X) order required by astra.
    """
    print("Building ct-2 breast phantom (analytic) ...")
    phantom = build_breast_phantom(breast_xc=0., breast_yc=0., breast_zc=0.)

    # image3D is (nx, ny, nz) with origin (x0, y0, z0) and side lengths.
    # Center the breast around x=3 (mid-breast) so the chest wall and
    # ductal structures all fall inside the box.
    xlen = NX * DX_CM
    ylen = NY * DX_CM
    zlen = NZ * DX_CM
    x_center = 3.0    # mid-breast x
    img = image3D(
        shape=(NX, NY, NZ),
        xlen=xlen, ylen=ylen, zlen=zlen,
        x0=x_center - xlen / 2.,
        y0=-ylen / 2.,
        z0=-zlen / 2.,
    )
    t0 = time.time()
    phantom.embed_in(img)
    print(f"  embed_in: {time.time()-t0:.1f}s")
    vol = img.mat                                          # (NX, NY, NZ)
    print(f"  volume range: [{vol.min():.4f}, {vol.max():.4f}] cm^-1")
    print(f"  non-zero fraction: {(vol > 1e-6).mean():.3f}")

    # astra expects (Z, Y, X)
    vol = np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32)
    print(f"  astra-ordered shape: {vol.shape}")
    return vol


# ---------------------------------------------------------------------------
# Recon driver
# ---------------------------------------------------------------------------
def run_recon():
    phantom = build_breast_volume()
    vol_geom, proj_geom, geom_info = vr.build_geometry(phantom.shape, DX_CM)
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

    # Override CONFIG itermax in-place (the CP loops read from the module
    # global; this is the cleanest in-process override).
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
        "dx_cm": DX_CM,
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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _xy_slice(vol):
    """Mid-axial slice (mean of central 3 z planes for a bit of averaging)."""
    nz = vol.shape[0]
    z = nz // 2
    return vol[z-1:z+2].mean(axis=0)


def fig_iter_ladder(result, iters, out_path, title=None,
                    vmin=None, vmax=None, crop=None):
    phi  = result["phantom"]
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]

    slc_gt = _xy_slice(phi)
    if vmin is None:
        vmin = float(slc_gt.min())
    if vmax is None:
        # Clip the calc/metal hot spots to keep soft-tissue contrast visible.
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

    # Column 0: ground truth in all three rows
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


def fig_convergence(result, out_path, title=None, xlim=None):
    s = result["ierrs_single"]
    t = result["ierrs_two"]
    iters = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(iters, s, "r-", lw=1.4, label="single-channel")
    ax.semilogy(iters, t, "b-", lw=1.4, label="two-channel")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    if xlim is not None:
        ax.set_xlim(xlim)
    if title is not None:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_phantom_intro(result, out_path, title=None):
    """Three-pane view of the full ct-2 breast phantom: axial, coronal,
    sagittal mid-slices. Used as the intro slide before the iter ladder.
    """
    phi = result["phantom"]                 # (NZ, NY, NX) astra-order
    nz, ny, nx = phi.shape
    axial   = phi[nz // 2]                  # xy plane
    coronal = phi[:, ny // 2, :]            # xz plane
    sagittal = phi[:, :, nx // 2]           # yz plane

    vmax = float(np.percentile(phi[phi > 0], 99.0)) * 1.05
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Discard cache and rerun reconstruction")
    args = parser.parse_args()

    result = load_or_run(args.force)

    fig_phantom_intro(
        result,
        FIG_DIR / "ct2_breast_phantom_intro.png",
        title="ct-2 analytic breast phantom (144x144x32 @ 1.5 mm)",
    )

    fig_iter_ladder(
        result, LADDER_ITERS,
        FIG_DIR / "ct2_breast_iter_ladder_xy.png",
        title="ct-2 breast phantom -- mid-axial slice across iterations "
              "(25 views / 50 deg LAR)",
    )

    # ROI: lateral soft-tissue patch including a couple of fatty masses.
    # Slice is (NY, NX) after _xy_slice; crop in (row=y, col=x) image coords.
    # The breast occupies roughly the upper-half of the y axis around the
    # center; pick a window around the masses.
    fig_iter_ladder(
        result, LADDER_ITERS,
        FIG_DIR / "ct2_breast_iter_ladder_roi.png",
        title="Soft-tissue ROI (lateral fatty region with mass inserts)",
        crop=(36, 110, 50, 124),    # 74 x 74 pixel patch
    )

    fig_convergence(
        result,
        FIG_DIR / "ct2_breast_convergence.png",
        title="ct-2 breast phantom: image RMSE vs iteration",
        xlim=(0, 300),
    )


if __name__ == "__main__":
    main()
