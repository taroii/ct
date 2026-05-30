"""Sequential 432^3 runs of phantom variants A1 (speckle), A2 (gradient),
A3 (both) at the deck resolution. Same anchor recon config as the 288^3
proof-of-concept: arc=15 deg, 25 views, cutoffparm_lo=1.5, itermax=50.

Single channel is shared across all three runs (only the phantom changes,
single-channel CP is the same so the resulting RMSE trajectory will be
the same up to numerical noise from the operator-norm power iteration --
but we re-run single per variant to keep snapshots aligned with each
variant's own GT for visualization).

Outputs:
    cache/ct2_breast_variant_<label>_432.pkl   per-variant snapshots
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

from presentation_ct2_phantom_ladder import _install_grid_cache  # noqa: E402
import victre_reconstruction as vr  # noqa: E402
from phantom3d import image3D  # noqa: E402
import importlib  # noqa: E402

_install_grid_cache()

CACHE_DIR = ROOT / "cache"

SHAPE = (432, 432, 96)
DX_CM = 0.05
CENTER = (3.0, 0.0, 0.0)
DET = (240, 240, 0.10)
NVIEWS = 25
ARC_DEG = 15.0
SOD_CM = 65.0
ODD_CM = 5.0

ITERMAX = 50
SNAPSHOT_ITERS = [5, 10, 15, 20, 30, 40, 50]

CP_OVERRIDES = {"cutoffparm_lo": 1.5}

# Scale speckle sigma to keep correlation length ~3 mm regardless of voxel size.
# At 288^3 we used sigma=4 voxels at 0.075 cm = 3 mm. At 432^3 with 0.05 cm
# voxels we want sigma = 3 mm / 0.05 cm = 6 voxels.
SPECKLE_SIGMA_VOXELS = 6.0
SPECKLE_AMPLITUDE   = 0.20
GRADIENT_AMPL_CHEST = 0.15
GRADIENT_AMPL_NIP   = -0.10
SPECKLE_SEED        = 1234

VARIANTS = ["A1_speckle", "A2_gradient", "A3_both"]

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
    return np.ascontiguousarray(img.mat.transpose(2, 1, 0)).astype(np.float32)


def add_speckle(vol):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(SPECKLE_SEED)
    noise = rng.standard_normal(vol.shape).astype(np.float32)
    smoothed = gaussian_filter(noise, sigma=SPECKLE_SIGMA_VOXELS)
    smoothed /= max(float(smoothed.std()), 1e-12)
    mask = vol > 1e-4
    out = vol.copy()
    out[mask] = vol[mask] * (1.0 + SPECKLE_AMPLITUDE * smoothed[mask])
    out = np.maximum(out, 0.0)
    print(f"  speckle: sigma={SPECKLE_SIGMA_VOXELS} vox "
          f"(~{SPECKLE_SIGMA_VOXELS*DX_CM*10:.1f} mm), "
          f"+-{SPECKLE_AMPLITUDE*100:.0f}%, "
          f"mu [{out.min():.4f}, {out.max():.4f}]")
    return out


def add_gradient(vol):
    nz, ny, nx = vol.shape
    ramp = np.linspace(GRADIENT_AMPL_CHEST, GRADIENT_AMPL_NIP,
                       nx, dtype=np.float32)
    factor = 1.0 + ramp
    mask = vol > 1e-4
    out = vol.copy()
    flat_out = out.reshape(-1, nx)
    flat_in  = vol.reshape(-1, nx)
    flat_mask = mask.reshape(-1, nx)
    broadcast_factor = np.broadcast_to(factor, flat_in.shape)
    flat_out[flat_mask] = flat_in[flat_mask] * broadcast_factor[flat_mask]
    print(f"  gradient: +{GRADIENT_AMPL_CHEST*100:.0f}% chest -> "
          f"{GRADIENT_AMPL_NIP*100:+.0f}% nipple, "
          f"mu [{out.min():.4f}, {out.max():.4f}]")
    return out


def apply_variant(base, variant):
    if variant == "A1_speckle":  return add_speckle(base)
    if variant == "A2_gradient": return add_gradient(base)
    if variant == "A3_both":     return add_gradient(add_speckle(base))
    raise ValueError(variant)


def run_variant(variant, base, A, At, gi, op_norms):
    print(f"\n##### {variant} #####")
    phantom = apply_variant(base, variant)
    nusino, nuxgrad, nuygrad, nuzgrad = op_norms

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

    out_pkl = CACHE_DIR / f"ct2_breast_variant_{variant}_432.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "phantom":          phantom,
            "recon_single":     rs, "recon_two": rt,
            "ierrs_single":     is_, "ierrs_two": it_,
            "snapshots_single": snaps_s, "snapshots_two": snaps_t,
            "dx_cm":            DX_CM, "geometry": gi,
            "variant":          variant,
            "cp_overrides":     CP_OVERRIDES,
        }, f)
    print(f"  cached {out_pkl.name}")


def main():
    print(f"Geometry: {SHAPE} @ {DX_CM} cm, arc={ARC_DEG}, n={NVIEWS}, "
          f"det={DET}, itermax={ITERMAX}")
    print(f"CP override: {CP_OVERRIDES}\n")

    base = build_phantom_base()
    print(f"  base phantom shape {base.shape}\n")

    det_row, det_col, det_sp = DET
    vol_geom, proj_geom, gi = vr.build_dbt_geometry(
        base.shape, DX_CM,
        det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
        nviews=NVIEWS, arc_deg=ARC_DEG, sod=SOD_CM, odd=ODD_CM,
    )
    A, At = vr.make_projector(vol_geom, proj_geom)
    op_norms = vr.operator_norms(
        base.shape, A, At, vr.CONFIG["npower"]
    )

    for v in VARIANTS:
        run_variant(v, base, A, At, gi, op_norms)

    print("\nAll variants done.")


if __name__ == "__main__":
    main()
