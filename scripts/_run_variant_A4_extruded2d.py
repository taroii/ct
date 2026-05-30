"""A4: take the 2D paper-breast phantom (the same testimage used in the
2D fan-beam slide) and extrude it along z to make a 3D phantom for DBT
reconstruction. Every z-slice is identical -- this is the literal
extrusion the user asked for. The phantom is the same 2D content that
produces the 2D-paper wobble; extruding it lets us test whether 3D DBT
exhibits the same single-channel semi-convergence on identical structure.

2D testimage recipe (from compare_methods_multiresolution.py:92-98):
    testimage = 0.5*Adipose + 1.0*Fibroglandular + 2.0*Calcification
This puts the testimage in range [0, ~3]. We rescale to a physical
attenuation coefficient by multiplying by 0.6/3 so the max maps to
0.6 cm^-1 (consistent with our breast display window).

3D geometry to match the existing deck setup: 432^2 xy * 96 z voxels @
0.05 cm. The 2D phantom is at 512^2; we downsample by zoom(0.844)
to 432.

Recon config: same as the A1-A3 variants:
    DBT arc=15 deg, 25 views, FDA-spec detector, itermax=50
    cutoffparm_lo=1.5, default inflate=2

Output: cache/ct2_breast_variant_A4_extruded2d_432.pkl
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
PHANTOM_IDX = 0   # which of the 10 paper-breast realisations to use

SHAPE = (432, 432, 96)
DX_CM = 0.05
DET = (240, 240, 0.10)
NVIEWS = 25
ARC_DEG = 15.0
SOD_CM = 65.0
ODD_CM = 5.0

ITERMAX = 50
SNAPSHOT_ITERS = [5, 10, 15, 20, 30, 40, 50]

CP_OVERRIDES = {"cutoffparm_lo": 1.5}

# Map testimage (range ~[0, 3]) to physical mu range [0, 0.6 cm^-1]
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
    """Compose the 2D testimage + extrude along z to a 3D phantom of shape
    (NZ, NY, NX) where NX,NY come from SHAPE and NZ is the z thickness."""
    print(f"Loading 2D paper-breast phantoms (slice {PHANTOM_IDX})")
    adipose = np.load(PHANTOM_DIR / "Phantom_Adipose.npy")[PHANTOM_IDX]
    fibro   = np.load(PHANTOM_DIR / "Phantom_Fibroglandular.npy")[PHANTOM_IDX]
    calc    = np.load(PHANTOM_DIR / "Phantom_Calcification.npy")[PHANTOM_IDX]
    print(f"  adipose {adipose.shape} {adipose.dtype}")

    test2d = (0.5 * adipose + 1.0 * fibro + 2.0 * calc).astype(np.float32)
    print(f"  composed testimage range [{test2d.min():.3f}, {test2d.max():.3f}]")

    # Resize to (NY, NX) using bilinear interpolation
    from scipy.ndimage import zoom
    NX, NY, NZ = SHAPE
    if test2d.shape != (NY, NX):
        zy, zx = NY / test2d.shape[0], NX / test2d.shape[1]
        test2d = zoom(test2d, (zy, zx), order=1).astype(np.float32)
        print(f"  zoomed to {test2d.shape}")
    test2d *= MU_SCALE
    print(f"  scaled to mu range [{test2d.min():.4f}, {test2d.max():.4f}] cm^-1")

    # Extrude along z: replicate the 2D slice NZ times
    phantom = np.broadcast_to(test2d[np.newaxis, :, :], (NZ, NY, NX))
    phantom = np.ascontiguousarray(phantom).astype(np.float32)
    print(f"  extruded to {phantom.shape}, "
          f"non-zero {100*(phantom > 1e-4).mean():.1f}%")
    return phantom


def main():
    print(f"\n=== A4: extruded 2D paper-breast ===")
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
        print(f"  single {time.time()-t0:.0f}s, RMSE@iter50 = {is_[-1]:.5f}")

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
        print(f"  two    {time.time()-t0:.0f}s, RMSE@iter50 = {it_[-1]:.5f}")
    finally:
        vr.CONFIG["itermax"] = saved
        reset_cp()

    print("\n=== iter-by-iter ===")
    print(f"  {'iter':>5}  {'single':>9}  {'two':>9}  {'red %':>7}")
    for it in SNAPSHOT_ITERS:
        a = is_[it - 1]; b = it_[it - 1]
        print(f"  {it:5d}  {a:9.5f}  {b:9.5f}  {(a-b)/a*100:+7.2f}")

    out_pkl = CACHE_DIR / "ct2_breast_variant_A4_extruded2d_432.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "phantom":          phantom,
            "recon_single":     rs, "recon_two": rt,
            "ierrs_single":     is_, "ierrs_two": it_,
            "snapshots_single": snaps_s, "snapshots_two": snaps_t,
            "dx_cm":            DX_CM, "geometry": gi,
            "variant":          "A4_extruded2d",
            "cp_overrides":     CP_OVERRIDES,
            "phantom_source":   "2D paper-breast (0.5*adipose + 1.0*fibro "
                                "+ 2.0*calc) extruded along z",
        }, f)
    print(f"\nCached {out_pkl.name}")


if __name__ == "__main__":
    main()
