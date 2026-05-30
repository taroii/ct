"""Phantom-modification experiments to make the ct-2 analytic breast
behave more like the 2D paper-breast (with LF content / texture that
single-channel struggles to recover while two-channel handles cleanly).

Anchor recon config (held constant across variants):
    288^3 x 64 voxels @ 0.075 cm
    DBT arc=15 deg, 25 views, FDA-spec detector
    cutoffparm_lo=1.5 + default inflate=2 (other CP at 2D-paper defaults)
    itermax=50, snapshots at [5, 10, 15, 20, 30, 40, 50]

Variants (run one per script invocation via --variant):
    baseline   - unmodified analytic phantom (reference)
    A1_speckle - add Gaussian-smoothed multiplicative texture
                 (sigma ~ 3 mm correlation, +-20% modulation,
                  applied only to non-air voxels)
    A2_gradient- add a smooth chest-wall-to-nipple density gradient
                 (+15% at chest wall, -10% at nipple)
    A3_both    - speckle + gradient combined
    A4_2D_extruded - replace the breast bulk with the 2D paper-breast
                     phantom extruded along z (TBD if needed)

Outputs:
    cache/ct2_breast_variant_<label>.pkl  per-variant recon snapshots
"""
import argparse
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

from presentation_ct2_phantom_ladder import _install_grid_cache  # noqa: E402
import victre_reconstruction as vr  # noqa: E402
from phantom3d import image3D  # noqa: E402
import importlib  # noqa: E402

_install_grid_cache()

CACHE_DIR = ROOT / "cache"

SHAPE = (288, 288, 64)
DX_CM = 0.075
CENTER = (3.0, 0.0, 0.0)
DET = (240, 240, 0.10)
NVIEWS = 25
ARC_DEG = 15.0
SOD_CM = 65.0
ODD_CM = 5.0

ITERMAX = 50
SNAPSHOT_ITERS = [5, 10, 15, 20, 30, 40, 50]

CP_OVERRIDES = {"cutoffparm_lo": 1.5}
SPECKLE_SIGMA_VOXELS = 4.0    # ~3 mm correlation at 0.075 cm voxels
SPECKLE_AMPLITUDE   = 0.20    # +-20% multiplicative
GRADIENT_AMPL_CHEST = 0.15    # +15% near chest wall
GRADIENT_AMPL_NIP   = -0.10   # -10% near nipple
SPECKLE_SEED        = 1234

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


def build_phantom_base():
    """Build the base (uniform-tissue) analytic breast phantom and embed
    in the voxel grid. Returns the (nz, ny, nx) float32 array."""
    mod = importlib.import_module("breast_phantom_demo")
    builder = getattr(mod, "build_breast_phantom")
    phantom = builder(breast_xc=0., breast_yc=0., breast_zc=0.)
    NX, NY, NZ = SHAPE
    xc, yc, zc = CENTER
    xlen, ylen, zlen = NX * DX_CM, NY * DX_CM, NZ * DX_CM
    img = image3D(shape=(NX, NY, NZ),
                  xlen=xlen, ylen=ylen, zlen=zlen,
                  x0=xc - xlen/2., y0=yc - ylen/2., z0=zc - zlen/2.)
    t0 = time.time()
    phantom.embed_in(img)
    print(f"  embed_in {time.time()-t0:.1f}s")
    vol = img.mat
    print(f"  base mu range [{vol.min():.4f}, {vol.max():.4f}], "
          f"non-zero frac {(vol > 1e-6).mean()*100:.2f}%")
    return np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32)


def add_speckle(vol, amplitude=SPECKLE_AMPLITUDE,
                sigma_voxels=SPECKLE_SIGMA_VOXELS, seed=SPECKLE_SEED):
    """Add Gaussian-smoothed multiplicative texture to non-air voxels."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(vol.shape).astype(np.float32)
    smoothed = gaussian_filter(noise, sigma=sigma_voxels)
    smoothed /= max(float(smoothed.std()), 1e-12)
    mask = vol > 1e-4
    out = vol.copy()
    out[mask] = vol[mask] * (1.0 + amplitude * smoothed[mask])
    out = np.maximum(out, 0.0)
    print(f"  speckle: sigma={sigma_voxels} vox (~{sigma_voxels*DX_CM*10:.1f} mm), "
          f"amplitude +-{amplitude*100:.0f}%, "
          f"mu range [{out.min():.4f}, {out.max():.4f}]")
    return out


def add_gradient(vol, ampl_chest=GRADIENT_AMPL_CHEST,
                 ampl_nip=GRADIENT_AMPL_NIP):
    """Add smooth chest-wall-to-nipple linear density gradient. vol is
    shape (nz, ny, nx); the chest-to-nipple direction is +x (last axis).

    Volume x-extent: image-grid offset puts ASTRA world x in roughly
    [-7.8, +13.8] cm with phantom built around x=0. Chest wall lives at
    x_voxel ~ 38, nipple tip at x_voxel ~ 92. We just linearly ramp
    from chest_wall end of the voxel grid to nipple end based on x_index.
    """
    nz, ny, nx = vol.shape
    # Linear ramp along x: ampl_chest at x=0 (start), ampl_nip at x=nx-1.
    ramp = np.linspace(ampl_chest, ampl_nip, nx, dtype=np.float32)
    factor = 1.0 + ramp                            # shape (nx,)
    mask = vol > 1e-4
    out = vol.copy()
    # Broadcast factor against last axis
    out_view = out.reshape(-1, nx)
    mask_view = mask.reshape(-1, nx)
    out_view[mask_view] = (vol.reshape(-1, nx)[mask_view] *
                           np.broadcast_to(factor, vol.reshape(-1, nx).shape)[mask_view])
    print(f"  gradient: +{ampl_chest*100:.0f}% chest-wall to {ampl_nip*100:+.0f}% nipple, "
          f"mu range [{out.min():.4f}, {out.max():.4f}]")
    return out


def apply_variant(base, variant):
    if variant == "baseline":
        return base
    if variant == "A1_speckle":
        return add_speckle(base)
    if variant == "A2_gradient":
        return add_gradient(base)
    if variant == "A3_both":
        return add_gradient(add_speckle(base))
    raise ValueError(f"unknown variant: {variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["baseline", "A1_speckle", "A2_gradient", "A3_both"])
    args = ap.parse_args()

    print(f"\n=== variant: {args.variant} ===")
    print(f"geometry: {SHAPE} @ {DX_CM} cm, arc={ARC_DEG}, n={NVIEWS}, "
          f"det={DET}, itermax={ITERMAX}")
    print(f"CP override: {CP_OVERRIDES}\n")

    base = build_phantom_base()
    phantom = apply_variant(base, args.variant)

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

    saved_itermax = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = ITERMAX
    try:
        print("\n--- single ---")
        reset_cp()
        t0 = time.time()
        rs, is_, ds_, ts_, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        single_time = time.time() - t0
        print(f"  single {single_time:.0f}s, RMSE @ iter "
              f"{SNAPSHOT_ITERS[-1]} = {is_[-1]:.5f}")

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
        rt, it_, dt_, tt_, snaps_t = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        two_time = time.time() - t0
        print(f"  two    {two_time:.0f}s, RMSE @ iter "
              f"{SNAPSHOT_ITERS[-1]} = {it_[-1]:.5f}")
    finally:
        vr.CONFIG["itermax"] = saved_itermax
        reset_cp()

    print("\n=== iter-by-iter ===")
    print(f"  {'iter':>5}  {'single':>9}  {'two':>9}  {'red %':>7}")
    for it in SNAPSHOT_ITERS:
        a = is_[it - 1]; b = it_[it - 1]
        red = (a - b) / a * 100
        print(f"  {it:5d}  {a:9.5f}  {b:9.5f}  {red:+7.2f}")

    out_pkl = CACHE_DIR / f"ct2_breast_variant_{args.variant}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "phantom":          phantom,
            "recon_single":     rs,
            "recon_two":        rt,
            "ierrs_single":     is_,
            "ierrs_two":        it_,
            "snapshots_single": snaps_s,
            "snapshots_two":    snaps_t,
            "dx_cm":            DX_CM,
            "geometry":         gi,
            "variant":          args.variant,
            "cp_overrides":     CP_OVERRIDES,
        }, f)
    print(f"\nCached {out_pkl.name}")


if __name__ == "__main__":
    main()
