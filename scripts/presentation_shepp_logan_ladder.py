"""
3D Shepp-Logan iteration ladder for the CT-Meeting 2026 talk.
Reuses scripts/victre_reconstruction.py for projector / norms / PDHG;
the phantom is the classical 10-ellipsoid 3D Shepp-Logan built directly
from ct-2's ellipsoid primitive.

Run:
    python scripts/presentation_shepp_logan_ladder.py [--force]

Outputs (under presentation/figs/):
    shepp_logan_phantom_intro.png    three-view ground-truth
    shepp_logan_iter_ladder_xy.png   single vs two ladder, xy slice
    shepp_logan_iter_ladder_roi.png  same ladder cropped to central low-contrast region
    shepp_logan_convergence.png      image RMSE vs iter

Cache:  cache/shepp_logan_recon.pkl
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
from phantom3d import image3D, phantom3D, ellipsoid, tissue_map  # noqa: E402


# Classical 3D Shepp-Logan ellipsoid table (Kak & Slaney convention).
# Coordinates are in the unit cube [-1, 1]^3; we scale by RAD_CM to make
# the skull radius about RAD_CM cm.
#   a   b   c   x0   y0   z0   mu_increment   z-rot (deg)
SL_TABLE = [
    (0.6900, 0.9200, 0.900,  0.0,   0.0,    0.0,   2.00,    0),
    (0.6624, 0.8740, 0.880,  0.0,   0.0,    0.0,  -0.98,    0),
    (0.4100, 0.1600, 0.210, -0.22,  0.0,   -0.25, -0.02,  108),
    (0.3100, 0.1100, 0.220,  0.22,  0.0,   -0.25, -0.02,   72),
    (0.2100, 0.2500, 0.500,  0.0,   0.35,  -0.25,  0.01,    0),
    (0.0460, 0.0460, 0.046,  0.0,   0.1,   -0.25,  0.01,    0),
    (0.0460, 0.0230, 0.020, -0.08, -0.605, -0.25,  0.01,    0),
    (0.0460, 0.0230, 0.020,  0.06, -0.605, -0.25,  0.01,   90),
    (0.0560, 0.0400, 0.100,  0.06, -0.105,  0.625, 0.02,   90),
    (0.0560, 0.0560, 0.100,  0.0,   0.100,  0.625,-0.02,    0),
]

RAD_CM = 10.0    # skull half-extent in cm (y direction)

CACHE_PATH = ROOT / "cache" / "shepp_logan_recon.pkl"
FIG_DIR    = ROOT / "presentation" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Image grid. Skull occupies y in +-9.2 cm, x in +-6.9 cm, z in +-9.0 cm.
SHAPE  = (128, 128, 64)              # NX, NY, NZ
DX_CM  = 0.20                        # 2.0 mm isotropic -> 25.6 cm box
DET    = (512, 512, 0.05)            # 25.6 cm detector to span the skull

SNAPSHOT_ITERS = [10, 50, 100, 200, 300, 500]
LADDER_ITERS   = [10, 50, 100, 200]
ITERMAX        = 500

# LAR arc. Kept at the DBT-typical 50 deg. At this realistic arc
# Shepp-Logan is well-conditioned enough that the two-channel split does
# not help -- it serves as a negative-control phantom.
ARC_DEG = 50.0


def build_shepp_logan_phantom():
    """Build a 3D Shepp-Logan with ct-2 ellipsoid primitives. Returns a
    phantom3D object. Coordinates scaled by RAD_CM (so a 0.92 unit-cube
    ellipsoid becomes a 9.2 cm half-axis in y).
    """
    print(f"Building Shepp-Logan phantom (RAD_CM={RAD_CM}) ...")
    head = phantom3D()
    for (a, b, c, x0, y0, z0, mu, rot_deg) in SL_TABLE:
        ax = a * RAD_CM
        ay = b * RAD_CM
        az = c * RAD_CM
        # ct-2 ellipsoid only exposes alpha (Z-rot) and beta (Y-rot); for
        # in-plane Shepp-Logan rotations we use alpha.
        ell = ellipsoid(
            x0=x0 * RAD_CM, y0=y0 * RAD_CM, z0=z0 * RAD_CM,
            att=mu, ax=ax, ay=ay, az=az,
            alpha=np.deg2rad(rot_deg),
        )
        tm = tissue_map(physical_material=False)
        tm.add_component(ell)
        head.add_component(tm)
    return head


def build_volume():
    head = build_shepp_logan_phantom()
    NX, NY, NZ = SHAPE
    xlen, ylen, zlen = NX * DX_CM, NY * DX_CM, NZ * DX_CM
    img = image3D(
        shape=(NX, NY, NZ),
        xlen=xlen, ylen=ylen, zlen=zlen,
        x0=-xlen / 2., y0=-ylen / 2., z0=-zlen / 2.,
    )
    t0 = time.time()
    head.embed_in(img)
    print(f"  embed_in: {time.time()-t0:.1f}s")
    vol = img.mat
    print(f"  volume range: [{vol.min():.4f}, {vol.max():.4f}]")
    print(f"  non-zero fraction: {(vol > 1e-6).mean():.3f}")
    vol = np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32)
    print(f"  astra-ordered shape: {vol.shape}")
    return vol


def run_recon():
    phantom = build_volume()
    det_row, det_col, det_sp = DET
    vol_geom, proj_geom, geom_info = vr.build_geometry(
        phantom.shape, DX_CM,
        det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
        nviews=25, arc_deg=ARC_DEG,
    )
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

    saved = vr.CONFIG["itermax"]
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
        vr.CONFIG["itermax"] = saved

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


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _xy_slice(vol):
    nz = vol.shape[0]; z = nz // 2
    z0, z1 = max(z-1, 0), min(z+2, nz)
    return vol[z0:z1].mean(axis=0)


def fig_phantom_intro(result, out_path, title=None):
    phi = result["phantom"]
    nz, ny, nx = phi.shape
    axial    = phi[nz // 2]
    coronal  = phi[:, ny // 2, :]
    sagittal = phi[:, :, nx // 2]
    # Shepp-Logan dynamic range: background 0, brain 1.0, bone 2.0, inserts
    # at brain +-0.02. Use the canonical narrow brain-tissue window so the
    # low-contrast inserts are visible.
    vmin, vmax = 0.95, 1.05
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, img, label in zip(
        axes, (axial, coronal, sagittal),
        ("axial (xy)", "coronal (xz)", "sagittal (yz)"),
    ):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                  origin="lower", aspect="equal")
        ax.set_title(label, fontsize=11); _strip(ax)
    if title is not None: fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_iter_ladder(result, iters, out_path, title=None, crop=None,
                    vmin=0.95, vmax=1.05):
    phi = result["phantom"]
    snaps_s = result["snapshots_single"]
    snaps_t = result["snapshots_two"]
    slc_gt = _xy_slice(phi)

    def crop_slice(s):
        if crop is None: return s
        r0, r1, c0, c1 = crop
        return s[r0:r1, c0:c1]

    n = len(iters)
    fig, axes = plt.subplots(3, n + 1, figsize=(1.9 * (n + 1), 5.6))
    for ax in axes.flat: _strip(ax)
    gt = crop_slice(slc_gt)
    for r, label in enumerate(["ground truth", "single", "two-channel"]):
        axes[r, 0].imshow(gt, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[r, 0].set_ylabel(label, fontsize=11)
    axes[0, 0].set_title("ground truth", fontsize=10)
    for i, it in enumerate(iters):
        col = i + 1
        s = crop_slice(_xy_slice(snaps_s[it]))
        t = crop_slice(_xy_slice(snaps_t[it]))
        axes[0, col].imshow(gt, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[0, col].set_title(f"iter {it}", fontsize=10)
        axes[1, col].imshow(s, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[2, col].imshow(t, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    if title is not None: fig.suptitle(title, fontsize=11)
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
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9)
    if xlim is not None: ax.set_xlim(xlim)
    if title is not None: ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    result = load_or_run(args.force)

    fig_phantom_intro(
        result, FIG_DIR / "shepp_logan_phantom_intro.png",
        title="Shepp-Logan phantom (10 ellipsoids, ~10 cm half-extent)",
    )
    # ROI: central insert region around (0, 0.1) in normalized coords ->
    # roughly row 60-100, col 50-78 in the 128x128 mid-axial slice.
    # XY ladder: wide window (0 .. 2.1) so the bony skull rim is visible
    # alongside the brain interior. ROI ladder: narrow brain-tissue window
    # so the tiny low-contrast inserts read.
    fig_iter_ladder(
        result, LADDER_ITERS,
        FIG_DIR / "shepp_logan_iter_ladder_xy.png",
        title="Shepp-Logan -- mid-axial slice across iterations",
        vmin=0.0, vmax=2.1,
    )
    fig_iter_ladder(
        result, LADDER_ITERS,
        FIG_DIR / "shepp_logan_iter_ladder_roi.png",
        title="Central insert ROI (low-contrast small ellipsoids)",
        crop=(48, 96, 40, 88),
        vmin=0.95, vmax=1.05,
    )
    fig_convergence(
        result, FIG_DIR / "shepp_logan_convergence.png",
        title=f"Shepp-Logan phantom ({ARC_DEG:.0f} deg arc): "
              f"image RMSE vs iteration",
        xlim=(0, 500),
    )


if __name__ == "__main__":
    main()
