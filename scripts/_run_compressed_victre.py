"""Single vs two-channel reconstruction of the compressed +
lesion-inserted VICTRE dense breast phantom at the FDA-VICTRE reference
DBT geometry (Hologic Selenia Dimensions, per Badano 2018).

Geometry (FDA-VICTRE reference):
    - DBT orbit (build_dbt_geometry): source arcs in y-z, detector fixed
      below at z = -ODD, rotation axis parallel to x
    - SOD = 65 cm, ODD = 1 cm  (SDD = 66 cm, Hologic Selenia)
    - Detector physical 25.5 x 12.75 cm (rectangular, long axis along y
      = lateral). Native MCGPU is 3000 x 1500 @ 0.085 mm; we bin 8x to
      375 x 187 at 0.68 mm for tractable compute.
    - 25 views over 15 deg arc (+-7.5 deg, Hologic narrow-arc mode)

Phantom processing:
    - load data/compressed_legion_victre/dense_pcl_...raw.gz
    - map labels -> mu (30 keV); paddle (label 50) -> 0
    - downsample 8x -> 0.4 mm effective voxel
    - axis convention: array is (z, y, x) with z = compression,
      y = lateral (arc-sweep), x = chest-to-nipple

CP parameters: default 2D-paper config (cutoffparm_lo=8, etc) -- this
script validates the FDA-spec geometry, not the tuning.

Outputs:
    presentation/figs/victre_pcl_phantom_intro.png
    presentation/figs/victre_pcl_iter_ladder_xy.png
    presentation/figs/victre_pcl_convergence.png
    cache/victre_pcl_recon.pkl
"""
import gzip
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

import victre_reconstruction as vr  # noqa: E402

RAW_GZ   = ROOT / "data" / "compressed_legion_victre" / "dense_pcl_-321964974_crop.raw.gz"
NX, NY, NZ = 810, 1920, 745          # native VICTRE compressed crop dims
NATIVE_DX_CM = 0.005                 # 0.05 mm
DOWNSAMPLE = 8                       # 8x -> 0.4 mm voxels

FIG_DIR   = ROOT / "presentation" / "figs"
CACHE_PKL = ROOT / "cache" / "victre_pcl_recon.pkl"

ITERMAX = 30
SNAPSHOT_ITERS = [5, 10, 15, 20, 25, 30]
DISPLAY = {"vmin": 0.0, "vmax": 0.6}

# FDA-VICTRE detector + orbit (Hologic Selenia Dimensions, Badano 2018).
# Native MCGPU detector is 3000 x 1500 at 0.085 mm pitch (25.5 x 12.75 cm,
# long axis along the lateral / arc-sweep direction). 8x binning ->
# 375 x 187 at 0.68 mm, same physical extent.
#   det_col_count: pixels along u = +x (chest-to-nipple)  -> 1500/8 = 187
#   det_row_count: pixels along v = +y (lateral)          -> 3000/8 = 375
DET_ROW_COUNT = 375          # rows along y (lateral, long detector axis)
DET_COL_COUNT = 187          # cols along x (chest-to-nipple, short axis)
DET_SPACING_CM = 0.068       # 0.68 mm pitch (8x binned from 0.085 mm)
NVIEWS = 25
ARC_DEG = 15.0               # +-7.5 deg, Hologic narrow-arc tomosynthesis
SOD_CM = 65.0
ODD_CM = 1.0                 # rotation axis near detector plane


# Tissue label -> linear attenuation (cm^-1, ~30 keV).
# Paddle (50) -> 0 since it's not anatomy we want to reconstruct.
MU_TABLE = {
    0:   0.000,
    1:   0.275,  # fat
    2:   0.375,  # skin
    29:  0.368,  # glandular
    33:  0.368,  # nipple
    40:  0.368,  # muscle (placeholder, label 40 absent from this phantom)
    50:  0.000,  # compression paddle -- ignored
    88:  0.368,  # ligament
    95:  0.368,  # TDLU
    125: 0.368,  # duct
    150: 0.368,  # artery
    200: 0.450,  # cancerous mass
    225: 0.368,  # vein
    250: 4.310,  # calcification
}


def load_and_downsample():
    print(f"Loading {RAW_GZ.name}")
    t0 = time.time()
    with gzip.open(RAW_GZ, "rb") as f:
        buf = f.read()
    expected = NX * NY * NZ
    assert len(buf) == expected, (
        f"phantom size {len(buf)} != expected {expected}")
    print(f"  read {len(buf)/1e9:.2f} GB in {time.time()-t0:.1f}s")

    # Reshape: VICTRE raw is X-fastest (z, y, x) per the .mhd
    # convention.
    vol_lbl = np.frombuffer(buf, np.uint8).reshape(NZ, NY, NX)

    # Build LUT and convert labels -> mu.
    mu_lut = np.zeros(256, dtype=np.float32)
    for k, v in MU_TABLE.items():
        mu_lut[k] = v

    d = DOWNSAMPLE
    nz_use = (NZ // d) * d
    ny_use = (NY // d) * d
    nx_use = (NX // d) * d
    nz_d, ny_d, nx_d = nz_use // d, ny_use // d, nx_use // d
    print(f"  downsample {d}x: {NX}x{NY}x{NZ} -> {nx_d}x{ny_d}x{nz_d}")
    print(f"  effective dx = {NATIVE_DX_CM * d:.4f} cm")

    # Chunk along z to keep peak RAM down -- the full float32 conversion
    # at native resolution would be ~4.6 GB.
    out = np.zeros((nz_d, ny_d, nx_d), dtype=np.float32)
    chunk = 8 * d
    t1 = time.time()
    for z0 in range(0, nz_use, chunk):
        z1 = min(z0 + chunk, nz_use)
        block = mu_lut[vol_lbl[z0:z1, :ny_use, :nx_use]]  # float32 chunk
        binned = block.reshape(
            (z1 - z0) // d, d, ny_d, d, nx_d, d
        ).mean(axis=(1, 3, 5))
        out[z0 // d : z1 // d] = binned
    print(f"  downsampled in {time.time()-t1:.1f}s")
    print(f"  mu range  : [{out.min():.4f}, {out.max():.4f}]")
    print(f"  non-zero %: {100 * (out > 1e-6).mean():.2f}")
    return out, NATIVE_DX_CM * d


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _xy_slice(vol):
    nz = vol.shape[0]
    z = nz // 2
    z0, z1 = max(z - 1, 0), min(z + 2, nz)
    return vol[z0:z1].mean(axis=0)


def fig_phantom_intro(phantom, dx_cm, out_path):
    nz, ny, nx = phantom.shape
    axial    = _xy_slice(phantom)
    coronal  = phantom[:, ny // 2, :]
    sagittal = phantom[:, :, nx // 2]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    extents = [
        (0, nx*dx_cm, 0, ny*dx_cm),
        (0, nx*dx_cm, 0, nz*dx_cm),
        (0, ny*dx_cm, 0, nz*dx_cm),
    ]
    for ax, img, label, extent in zip(
        axes, (axial, coronal, sagittal),
        ("axial (xy, perpendicular to compression)", "coronal (xz)", "sagittal (yz)"),
        extents,
    ):
        ax.imshow(img, cmap="gray", vmin=DISPLAY["vmin"], vmax=DISPLAY["vmax"],
                  origin="lower", aspect="equal", extent=extent)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("cm")
        ax.set_ylabel("cm")
    fig.suptitle(
        f"compressed-cropped lesion VICTRE dense phantom @ {dx_cm*10:.2f} mm "
        f"({nx*dx_cm:.1f} x {ny*dx_cm:.1f} x {nz*dx_cm:.1f} cm)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_iter_ladder(phantom, snaps_s, snaps_t, iters, out_path):
    slc_gt = _xy_slice(phantom)
    n = len(iters)
    fig, axes = plt.subplots(3, n + 1, figsize=(1.9 * (n + 1), 5.6))
    for ax in axes.flat:
        _strip(ax)
    for r, label in enumerate(["ground truth", "single", "two-channel"]):
        axes[r, 0].imshow(slc_gt, cmap="gray", vmin=DISPLAY["vmin"],
                          vmax=DISPLAY["vmax"], origin="lower")
        axes[r, 0].set_ylabel(label, fontsize=11)
    axes[0, 0].set_title("ground truth", fontsize=10)
    for i, it in enumerate(iters):
        col = i + 1
        axes[0, col].imshow(slc_gt, cmap="gray", vmin=DISPLAY["vmin"],
                            vmax=DISPLAY["vmax"], origin="lower")
        axes[0, col].set_title(f"iter {it}", fontsize=10)
        axes[1, col].imshow(_xy_slice(snaps_s[it]), cmap="gray",
                            vmin=DISPLAY["vmin"], vmax=DISPLAY["vmax"],
                            origin="lower")
        axes[2, col].imshow(_xy_slice(snaps_t[it]), cmap="gray",
                            vmin=DISPLAY["vmin"], vmax=DISPLAY["vmax"],
                            origin="lower")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_convergence(ierrs_single, ierrs_two, out_path):
    s = np.asarray(ierrs_single); t = np.asarray(ierrs_two)
    iters = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(iters, s, "r-", lw=1.4, label="single-channel")
    ax.semilogy(iters, t, "b-", lw=1.4, label="two-channel")
    ax.set_xlabel("iteration")
    ax.set_ylabel("image RMSE")
    ax.set_xlim(0, len(s))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_title("compressed VICTRE breast: image RMSE vs iteration")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PKL.parent.mkdir(parents=True, exist_ok=True)

    # 1. phantom
    phantom, dx_cm = load_and_downsample()

    # 2. DBT geometry
    print("\nBuilding DBT geometry")
    vol_geom, proj_geom, gi = vr.build_dbt_geometry(
        phantom.shape, dx_cm,
        det_row_count=DET_ROW_COUNT, det_col_count=DET_COL_COUNT,
        det_spacing=DET_SPACING_CM,
        nviews=NVIEWS, arc_deg=ARC_DEG,
        sod=SOD_CM, odd=ODD_CM,
    )
    A, At = vr.make_projector(vol_geom, proj_geom)

    # 3. Two-channel filters (axis=2d for DBT, matching analytic breast)
    R_hi, R_lo = vr.build_sinogram_filters(
        gi["det_col_count"], gi["det_spacing"],
        vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
        axis="2d", det_row_count=gi["det_row_count"],
    )

    # 4. Operator norms
    print("\nOperator-norm power iteration")
    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"]
    )

    # 5. CP loops at default 2D-paper params
    saved_itermax = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = ITERMAX
    try:
        print("\n--- single-channel ---")
        t0 = time.time()
        rs, is_, ds_, ts_, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        single_time = time.time() - t0
        print(f"  single elapsed: {single_time:.0f}s, final RMSE {is_[-1]:.5f}")

        print("\n--- two-channel ---")
        t0 = time.time()
        rt, it_, dt_, tt_, snaps_t = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        two_time = time.time() - t0
        print(f"  two elapsed: {two_time:.0f}s, final RMSE {it_[-1]:.5f}")
    finally:
        vr.CONFIG["itermax"] = saved_itermax

    # 6. Summary
    print("\n=== iter-by-iter comparison ===")
    print(f"  {'iter':>5}  {'single':>9}  {'two':>9}  {'red %':>7}")
    for it in SNAPSHOT_ITERS:
        a = is_[it - 1]; b = it_[it - 1]
        red = (a - b) / a * 100 if a > 0 else float("nan")
        print(f"  {it:5d}  {a:9.5f}  {b:9.5f}  {red:+7.2f}")

    # 7. Cache
    with open(CACHE_PKL, "wb") as f:
        pickle.dump({
            "phantom": phantom,
            "recon_single": rs, "recon_two": rt,
            "ierrs_single": is_, "ierrs_two": it_,
            "derrs_single": ds_, "derrs_two": dt_,
            "tvs_single": ts_, "tvs_two": tt_,
            "snapshots_single": snaps_s, "snapshots_two": snaps_t,
            "dx_cm": dx_cm,
            "geometry": gi,
            "config": dict(vr.CONFIG),
        }, f)
    print(f"\nCached {CACHE_PKL}")

    # 8. Figures
    fig_phantom_intro(phantom, dx_cm, FIG_DIR / "victre_pcl_phantom_intro.png")
    fig_iter_ladder(phantom, snaps_s, snaps_t, SNAPSHOT_ITERS,
                    FIG_DIR / "victre_pcl_iter_ladder_xy.png")
    fig_convergence(is_, it_,
                    FIG_DIR / "victre_pcl_convergence.png")


if __name__ == "__main__":
    main()
