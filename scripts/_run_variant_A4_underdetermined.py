"""A4_underdetermined: same extruded 2D paper-breast phantom as A4, but
at a much finer in-plane voxel grid (1024x1024) so the effective DOF
ratio (rays/effective-unknowns) drops to ~0.5 -- matching the 2D paper's
under-determined regime.

Compared to standard A4 (432^2x96 at 0.05 cm, 25 views, ratio 7.7) this
config has:
    1024x1024 x 96 voxels at dx = 0.021 cm (21.5 x 21.5 x 2.0 cm)
    9 views x 240^2 detector = 519k rays
    effective unknowns (z-uniform): 1024^2 = 1.049M
    ratio = 0.49  (under-determined, like the 2D paper at 0.39)

Compute: ~25-30 min per recon at 432^3 -> ~50-60 min total.

CP and other params identical to A4 to keep the comparison clean.
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ct-2"))
os.chdir(ROOT)

import victre_reconstruction as vr  # noqa: E402

CACHE_DIR = ROOT / "cache"
PHANTOM_DIR = ROOT / "data" / "phantoms_from_paper"
PHANTOM_IDX = 0

SHAPE = (1024, 1024, 96)
DX_CM = 0.021       # 21.5 cm in-plane, 2.0 cm in z
DET = (240, 240, 0.10)
NVIEWS = 9
ARC_DEG = 15.0
SOD_CM = 65.0
ODD_CM = 5.0

ITERMAX = 50
SNAPSHOT_ITERS = [5, 10, 15, 20, 30, 40, 50]

CP_OVERRIDES = {"cutoffparm_lo": 1.5}
MU_SCALE = 0.6 / 3.0

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


def build_extruded_phantom():
    print(f"Loading 2D paper-breast phantoms (slice {PHANTOM_IDX})")
    adipose = np.load(PHANTOM_DIR / "Phantom_Adipose.npy")[PHANTOM_IDX]
    fibro   = np.load(PHANTOM_DIR / "Phantom_Fibroglandular.npy")[PHANTOM_IDX]
    calc    = np.load(PHANTOM_DIR / "Phantom_Calcification.npy")[PHANTOM_IDX]
    test2d = (0.5 * adipose + 1.0 * fibro + 2.0 * calc).astype(np.float32)
    print(f"  composed testimage range "
          f"[{test2d.min():.3f}, {test2d.max():.3f}]")
    from scipy.ndimage import zoom
    NX, NY, NZ = SHAPE
    if test2d.shape != (NY, NX):
        zy, zx = NY / test2d.shape[0], NX / test2d.shape[1]
        test2d = zoom(test2d, (zy, zx), order=1).astype(np.float32)
        print(f"  zoomed to {test2d.shape}")
    test2d *= MU_SCALE
    print(f"  scaled to mu range "
          f"[{test2d.min():.4f}, {test2d.max():.4f}] cm^-1")
    phantom = np.broadcast_to(test2d[np.newaxis, :, :], (NZ, NY, NX))
    phantom = np.ascontiguousarray(phantom).astype(np.float32)
    print(f"  extruded to {phantom.shape}, "
          f"non-zero {100*(phantom > 1e-4).mean():.1f}%")
    print(f"  memory: phantom {phantom.nbytes/1e6:.0f} MB")
    return phantom


def main():
    print(f"\n=== A4_underdetermined: 1024^2 x 96 voxels, 9 views ===")
    print(f"Effective DOF (z-uniform): {SHAPE[0]*SHAPE[1]:,d}")
    print(f"Rays: {NVIEWS * DET[0] * DET[1]:,d}")
    print(f"Ratio: {NVIEWS * DET[0] * DET[1] / (SHAPE[0]*SHAPE[1]):.2f}")
    print(f"Geometry: {SHAPE} @ {DX_CM} cm, arc={ARC_DEG}, n={NVIEWS}, "
          f"det={DET}, itermax={ITERMAX}")
    print(f"CP override: {CP_OVERRIDES}\n")

    phantom = build_extruded_phantom()
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
        print(f"  single {time.time()-t0:.0f}s, "
              f"RMSE@iter50 = {is_[-1]:.5f}")

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
        print(f"  two    {time.time()-t0:.0f}s, "
              f"RMSE@iter50 = {it_[-1]:.5f}")
    finally:
        vr.CONFIG["itermax"] = saved
        reset_cp()

    print("\n=== iter-by-iter ===")
    print(f"  {'iter':>5}  {'single':>9}  {'two':>9}  {'red %':>7}")
    for it in SNAPSHOT_ITERS:
        a = is_[it - 1]; b = it_[it - 1]
        print(f"  {it:5d}  {a:9.5f}  {b:9.5f}  {(a-b)/a*100:+7.2f}")

    out_pkl = CACHE_DIR / "ct2_breast_variant_A4_underdetermined.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "phantom":          phantom,
            "recon_single":     rs, "recon_two": rt,
            "ierrs_single":     is_, "ierrs_two": it_,
            "snapshots_single": snaps_s, "snapshots_two": snaps_t,
            "dx_cm":            DX_CM, "geometry": gi,
            "variant":          "A4_underdetermined",
            "cp_overrides":     CP_OVERRIDES,
            "phantom_source":   "2D paper-breast extruded at 1024^2",
            "underdetermined_ratio": NVIEWS * DET[0] * DET[1] / (SHAPE[0]*SHAPE[1]),
        }, f)
    print(f"\nCached {out_pkl.name}")


if __name__ == "__main__":
    main()
