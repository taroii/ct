"""3D Shepp-Logan recon on the SAME 3D DBT pipeline as synthetic A/B.

Story parallel:
    -- 2D DBT (paper-breast)        -> 2D Shepp-Logan   (presentation_shepp_logan_2d.py)
    -- 3D DBT (synthetic A/B)       -> 3D Shepp-Logan   (this script)

Geometry / recon config: IDENTICAL to _run_synthetic_phantoms.py (phantom A/B):
    432 x 432 x 96 voxels @ 0.05 cm, DBT cone-beam, 15 deg arc, 25 views,
    240 x 240 detector @ 0.10 cm, SOD=65, ODD=5, itermax=500,
    two-channel CP overrides cutoffparm_lo=3.0, eps_lo_ratio=0.5.

Phantom: modified Kak & Slaney 3D Shepp-Logan (10 ellipsoids, full Euler
rotation), sampled isotropically so the head fills the xy petri dish; the
4.8 cm z-slab images the central axial band of the head (the ellipsoids
genuinely vary through z, so y is non-trivially 3D).

Data scaling (option 1): the SL attenuations are peak-scaled so the max
mu = 0.50 cm^-1, matching phantom A's peak (mu_calc). This keeps the
sinogram magnitude ||g|| in the same regime as A, so the SAME eps
data-fidelity tolerance is reused for BOTH single- and two-channel runs
(eps=0.001, eps_hi_ratio=1.0, eps_lo_ratio=0.5) -- no per-method tuning.

Outputs:
    cache/shepp_logan_3d_recon.pkl
    presentation/figs/shepp_logan_3d_phantom_intro.png
    presentation/figs/shepp_logan_3d_iter_ladder.png
    presentation/figs/shepp_logan_3d_error_ladder.png
    presentation/figs/shepp_logan_3d_convergence.png
"""
import argparse
import os
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
os.chdir(ROOT)

import victre_reconstruction as vr  # noqa: E402

CACHE_DIR = ROOT / "cache"
FIG_DIR   = ROOT / "presentation" / "figs"

# -- Geometry (identical to synthetic A/B) --
SHAPE   = (96, 432, 432)   # (NZ, NY, NX) astra ordering
DX_CM   = 0.05
DET     = (240, 240, 0.10)
NVIEWS  = 25
ARC_DEG = 15.0
SOD_CM  = 65.0
ODD_CM  = 5.0

ITERMAX        = 500
SNAPSHOT_ITERS = [10, 50, 100, 200, 300, 500]
LADDER_ITERS   = [10, 50, 100, 200, 500]

CP_OVERRIDES = {"cutoffparm_lo": 3.0, "eps_lo_ratio": 0.5}

PETRI_RADIUS_CM = 9.5

# SL sampling: rad_cm sets the head size. Outer ellipsoid b=0.92 in SL
# units; rad_cm=9.0 -> head half-height 8.28 cm (fits the 9.5 cm dish).
SL_RAD_CM  = 9.0
SL_PEAK_MU = 0.50     # match phantom A peak (mu_calc)

# Display windows (mu cm^-1). Shell sits at SL_PEAK_MU; brain ~0.1.
DISPLAY_FULL = {"vmin": 0.0, "vmax": 0.50}   # intro: show shell
DISPLAY_SOFT = {"vmin": 0.0, "vmax": 0.18}   # ladders: show soft-tissue wobble

DEFAULTS = {
    "sigma_lo_scale":  vr.CONFIG["sigma_lo_scale"],
    "norm_inflate_3d": vr.CONFIG["norm_inflate_3d"],
    "eps_hi_ratio":    vr.CONFIG["eps_hi_ratio"],
    "eps_lo_ratio":    vr.CONFIG["eps_lo_ratio"],
    "cutoffparm":      vr.CONFIG["cutoffparm"],
    "cutoffparm_lo":   vr.CONFIG["cutoffparm_lo"],
    "beta":            vr.CONFIG["beta"],
}


def reset_cp():
    for k, v in DEFAULTS.items():
        vr.CONFIG[k] = v


# ---------------------------------------------------------------------------
# Phantom
# ---------------------------------------------------------------------------

# Modified Shepp-Logan 3D table (Kak & Slaney / Schabel phantom3d):
#   A, a, b, c, x0, y0, z0, phi(deg), theta(deg), psi(deg)
SL3D_TABLE = [
    ( 1.0,  .6900, .920, .810,    0.0,    0.0,   0.0,   0,  0,  0),
    (-0.8,  .6624, .874, .780,    0.0, -.0184,   0.0,   0,  0,  0),
    (-0.2,  .1100, .310, .220,    .22,    0.0,   0.0, -18,  0, 10),
    (-0.2,  .1600, .410, .280,   -.22,    0.0,   0.0,  18,  0, 10),
    ( 0.1,  .2100, .250, .410,    0.0,    .35, -.150,   0,  0,  0),
    ( 0.1,  .0460, .046, .050,    0.0,    .10,  .250,   0,  0,  0),
    ( 0.1,  .0460, .046, .050,    0.0,   -.10,  .250,   0,  0,  0),
    ( 0.1,  .0460, .023, .050,   -.08,  -.605,   0.0,   0,  0,  0),
    ( 0.1,  .0230, .023, .020,    0.0,  -.606,   0.0,   0,  0,  0),
    ( 0.1,  .0230, .046, .020,    .06,  -.605,   0.0,   0,  0,  0),
]


def _euler_matrix(phi, theta, psi):
    """Schabel phantom3d Euler rotation matrix (angles in radians)."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi*cphi - cth*sphi*spsi,   cpsi*sphi + cth*cphi*spsi,  spsi*sth],
        [-spsi*cphi - cth*sphi*cpsi, -spsi*sphi + cth*cphi*cpsi,  cpsi*sth],
        [sth*sphi,                   -sth*cphi,                   cth],
    ])


def build_shepp_logan_3d(shape, dx_cm, rad_cm, peak_mu):
    """Rasterise the modified 3D Shepp-Logan on an (NZ,NY,NX) grid.

    Physical coords are centred; SL unit coords = physical / rad_cm. The
    xy plane fills the head; the thin z-slab cuts the central axial band.
    """
    NZ, NY, NX = shape
    z = (np.arange(NZ) - (NZ - 1) / 2) * dx_cm / rad_cm
    y = (np.arange(NY) - (NY - 1) / 2) * dx_cm / rad_cm
    x = (np.arange(NX) - (NX - 1) / 2) * dx_cm / rad_cm
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    vol = np.zeros(shape, dtype=np.float64)

    for (A, a, b, c, x0, y0, z0, phi_d, th_d, psi_d) in SL3D_TABLE:
        xt = X - x0
        yt = Y - y0
        zt = Z - z0
        if phi_d or th_d or psi_d:
            R = _euler_matrix(np.deg2rad(phi_d), np.deg2rad(th_d),
                              np.deg2rad(psi_d))
            xr = R[0, 0]*xt + R[0, 1]*yt + R[0, 2]*zt
            yr = R[1, 0]*xt + R[1, 1]*yt + R[1, 2]*zt
            zr = R[2, 0]*xt + R[2, 1]*yt + R[2, 2]*zt
        else:
            xr, yr, zr = xt, yt, zt
        inside = (xr / a) ** 2 + (yr / b) ** 2 + (zr / c) ** 2 <= 1.0
        vol[inside] += A

    vol = np.maximum(vol, 0.0)
    peak = float(vol.max())
    vol = vol * (peak_mu / max(peak, 1e-12))
    return vol.astype(np.float32)


def petri_dish_mask(shape, dx_cm, radius_cm):
    NZ, NY, NX = shape
    y = (np.arange(NY) - (NY - 1) / 2) * dx_cm
    x = (np.arange(NX) - (NX - 1) / 2) * dx_cm
    Y, X = np.meshgrid(y, x, indexing="ij")
    in_disk = (Y * Y + X * X) <= radius_cm * radius_cm
    mask = np.broadcast_to(in_disk[np.newaxis, :, :], shape)
    return np.ascontiguousarray(mask).astype(bool)


def build_phantom():
    print(f"Building 3D Shepp-Logan {SHAPE} @ {DX_CM} cm "
          f"(rad={SL_RAD_CM} cm, peak_mu={SL_PEAK_MU})")
    vol = build_shepp_logan_3d(SHAPE, DX_CM, SL_RAD_CM, SL_PEAK_MU)
    print(f"  raw range [{vol.min():.4f}, {vol.max():.4f}], "
          f"non-zero {100*(vol > 1e-4).mean():.1f}%")
    mask = petri_dish_mask(SHAPE, DX_CM, PETRI_RADIUS_CM)
    vol = np.where(mask, vol, 0.0).astype(np.float32)
    print(f"  after petri mask (r={PETRI_RADIUS_CM} cm): "
          f"non-zero {100*(vol > 1e-4).mean():.1f}%")
    return vol


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _xy_slice(vol):
    nz = vol.shape[0]
    z = nz // 2
    z0, z1 = max(z - 1, 0), min(z + 2, nz)
    return vol[z0:z1].mean(axis=0)


def _rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def fig_phantom_intro(phantom, out_path):
    nz, ny, nx = phantom.shape
    axial = _xy_slice(phantom)
    coronal = phantom[:, ny // 2, :]
    sagittal = phantom[:, :, nx // 2]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, img, lab, extent in zip(
        axes, (axial, coronal, sagittal),
        ("axial (xy)", "coronal (xz)", "sagittal (yz)"),
        ((0, nx*DX_CM, 0, ny*DX_CM),
         (0, nx*DX_CM, 0, nz*DX_CM),
         (0, ny*DX_CM, 0, nz*DX_CM)),
    ):
        ax.imshow(img, cmap="gray", vmin=DISPLAY_FULL["vmin"],
                  vmax=DISPLAY_FULL["vmax"], origin="lower",
                  aspect="equal", extent=extent)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("cm"); ax.set_ylabel("cm")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def fig_iter_ladder(phantom, snaps_s, snaps_t, iters, out_path):
    slc_gt = _xy_slice(phantom)
    n = len(iters)
    fig, axes = plt.subplots(2, n + 1, figsize=(1.9 * (n + 1), 3.8))
    for ax in axes.flat:
        _strip(ax)
    for r, lab in enumerate(["single", "two-channel"]):
        axes[r, 0].imshow(slc_gt, cmap="gray", vmin=DISPLAY_SOFT["vmin"],
                          vmax=DISPLAY_SOFT["vmax"], origin="lower")
        axes[r, 0].set_ylabel(lab, fontsize=11)
    axes[0, 0].set_title("ground truth", fontsize=10)
    for i, it in enumerate(iters):
        col = i + 1
        s, t = snaps_s[it], snaps_t[it]
        axes[0, col].set_title(f"iter {it}\nRMSE {_rmse(s, phantom):.3f}",
                               fontsize=9)
        axes[0, col].imshow(_xy_slice(s), cmap="gray",
                            vmin=DISPLAY_SOFT["vmin"], vmax=DISPLAY_SOFT["vmax"],
                            origin="lower")
        axes[1, col].imshow(_xy_slice(t), cmap="gray",
                            vmin=DISPLAY_SOFT["vmin"], vmax=DISPLAY_SOFT["vmax"],
                            origin="lower")
        axes[1, col].set_xlabel(f"RMSE {_rmse(t, phantom):.3f}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def fig_error_ladder(phantom, snaps_s, snaps_t, iters, out_path, err_max=0.08):
    slc_gt = _xy_slice(phantom)
    n = len(iters)
    fig, axes = plt.subplots(2, n + 1, figsize=(1.9 * (n + 1), 3.8))
    for ax in axes.flat:
        _strip(ax)
    axes[0, 0].imshow(slc_gt, cmap="gray", vmin=DISPLAY_SOFT["vmin"],
                      vmax=DISPLAY_SOFT["vmax"], origin="lower")
    axes[0, 0].set_title("ground truth", fontsize=10)
    axes[1, 0].imshow(slc_gt, cmap="gray", vmin=DISPLAY_SOFT["vmin"],
                      vmax=DISPLAY_SOFT["vmax"], origin="lower")
    axes[0, 0].set_ylabel("single error", fontsize=10)
    axes[1, 0].set_ylabel("two error", fontsize=10)
    for i, it in enumerate(iters):
        col = i + 1
        err_s = np.abs(_xy_slice(snaps_s[it]) - slc_gt)
        err_t = np.abs(_xy_slice(snaps_t[it]) - slc_gt)
        axes[0, col].set_title(f"iter {it}", fontsize=10)
        axes[0, col].imshow(err_s, cmap="hot", vmin=0, vmax=err_max,
                            origin="lower")
        axes[1, col].imshow(err_t, cmap="hot", vmin=0, vmax=err_max,
                            origin="lower")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def fig_convergence(ierrs_single, ierrs_two, out_path, title=None):
    # Single 0-500 panel, no titles (slide captions carry the details).
    s = np.asarray(ierrs_single); t = np.asarray(ierrs_two)
    its = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(its, s, "r-", lw=1.8, label="single")
    ax.plot(its, t, "b-", lw=1.8, label="two-channel")
    ax.set_xlim(0, len(s))
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Recon driver (mirrors _run_synthetic_phantoms.run_one)
# ---------------------------------------------------------------------------

def run():
    phantom = build_phantom()
    det_row, det_col, det_sp = DET
    vol_geom, proj_geom, gi = vr.build_dbt_geometry(
        phantom.shape, DX_CM,
        det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
        nviews=NVIEWS, arc_deg=ARC_DEG, sod=SOD_CM, odd=ODD_CM,
    )
    A, At = vr.make_projector(vol_geom, proj_geom)
    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"]
    )

    saved = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = ITERMAX
    try:
        reset_cp()
        print("\n--- single ---")
        t0 = time.time()
        rs, is_, _, _, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        print(f"  single {time.time()-t0:.0f}s, RMSE@iter{ITERMAX} = {is_[-1]:.5f}")

        reset_cp()
        for k, v in CP_OVERRIDES.items():
            vr.CONFIG[k] = v
        R_hi, R_lo = vr.build_sinogram_filters(
            gi["det_col_count"], gi["det_spacing"],
            vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
            axis="2d", det_row_count=gi["det_row_count"],
        )
        print("\n--- two ---")
        t0 = time.time()
        rt, it_, _, _, snaps_t = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        print(f"  two    {time.time()-t0:.0f}s, RMSE@iter{ITERMAX} = {it_[-1]:.5f}")
    finally:
        vr.CONFIG["itermax"] = saved
        reset_cp()

    print("\n=== iter-by-iter ===")
    print(f"  {'iter':>5}  {'single':>9}  {'two':>9}  {'red %':>7}")
    for it in SNAPSHOT_ITERS:
        a = is_[it - 1]; b = it_[it - 1]
        print(f"  {it:5d}  {a:9.5f}  {b:9.5f}  {(a-b)/a*100:+7.2f}")

    out_pkl = CACHE_DIR / "shepp_logan_3d_recon.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "phantom":          phantom,
            "recon_single":     rs, "recon_two": rt,
            "ierrs_single":     is_, "ierrs_two": it_,
            "snapshots_single": snaps_s, "snapshots_two": snaps_t,
            "dx_cm":            DX_CM, "geometry": gi,
            "cp_overrides":     CP_OVERRIDES,
            "sl_rad_cm":        SL_RAD_CM, "sl_peak_mu": SL_PEAK_MU,
        }, f)
    print(f"  cached {out_pkl.name}")

    fig_phantom_intro(phantom, FIG_DIR / "shepp_logan_3d_phantom_intro.png")
    fig_iter_ladder(phantom, snaps_s, snaps_t, LADDER_ITERS,
                    FIG_DIR / "shepp_logan_3d_iter_ladder.png")
    fig_error_ladder(phantom, snaps_s, snaps_t, LADDER_ITERS,
                     FIG_DIR / "shepp_logan_3d_error_ladder.png")
    fig_convergence(is_, it_, FIG_DIR / "shepp_logan_3d_convergence.png",
                    f"3D Shepp-Logan -- DBT arc=15, 25v, 432^2x96, "
                    f"cutLo=3 eps_lo=0.5 itermax={ITERMAX}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true",
                    help="Build phantom and save intro fig only (no recon)")
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.preview:
        phantom = build_phantom()
        fig_phantom_intro(phantom, FIG_DIR / "shepp_logan_3d_phantom_intro.png")
        return
    run()


if __name__ == "__main__":
    main()
