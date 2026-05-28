"""Two follow-up experiments for the DBT breast two-channel question:

A) 288x288x64 voxels at 0.075 cm, itermax 500 (does the gap close past
   iter 200?). Same reduced detector as the highres sweep.
B) 432x432x96 voxels at 0.05 cm, itermax 200 (does even more
   under-determination push two-channel over?).

Both at det 240x240 (0.1 cm), 25 views, 50 deg arc, 2D Hanning filter,
inflate=1.5 (best variant from the previous sweep), plus a baseline
(default inflate=sqrt(4)=2) trial for comparison.

Output: cache/_breast_dbt_extended_sweep.txt
"""
import os
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

DET = (240, 240, 0.10)
ARC_DEG = 50.0
NVIEWS = 25
FILTER_AXIS = "2d"
CENTER = (3.0, 0.0, 0.0)

# (label, shape_xyz, dx_cm, itermax, snapshot_iters)
EXPERIMENTS = [
    ("A_288x64_iter500",
     (288, 288, 64), 0.075, 500, [50, 100, 200, 300, 400, 500]),
    ("B_432x96_iter200",
     (432, 432, 96), 0.05,  200, [50, 100, 150, 200]),
]

# (label, sigma_lo_scale, eps_lo_ratio, norm_inflate_3d)
TRIALS = [
    ("baseline", 4.0, 1.25, None),
    ("inf1p5",   4.0, 1.25, 1.5),
]

OUT_TXT = ROOT / "cache" / "_breast_dbt_extended_sweep.txt"


def reduction(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


def build_phantom(shape_xyz, dx_cm):
    mod = importlib.import_module("breast_phantom_demo")
    builder = getattr(mod, "build_breast_phantom")
    phantom = builder(breast_xc=0., breast_yc=0., breast_zc=0.)

    NX, NY, NZ = shape_xyz
    DX = dx_cm
    xc, yc, zc = CENTER
    xlen, ylen, zlen = NX * DX, NY * DX, NZ * DX
    img = image3D(
        shape=(NX, NY, NZ),
        xlen=xlen, ylen=ylen, zlen=zlen,
        x0=xc - xlen/2., y0=yc - ylen/2., z0=zc - zlen/2.,
    )
    t0 = time.time()
    phantom.embed_in(img)
    print(f"  embed_in: {time.time()-t0:.1f}s")
    vol = img.mat
    nz = int((vol > 1e-6).sum())
    print(f"  range [{vol.min():.4f}, {vol.max():.4f}]  "
          f"non-zero {nz}/{vol.size} ({100*nz/vol.size:.1f}%)")
    vol = np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32)
    print(f"  astra-ordered shape: {vol.shape}")
    return vol


def fmt_iters(arr, iters):
    return " ".join(f"{arr[it-1]:.5f}" for it in iters)


def fmt_reds(s, t, iters):
    return " ".join(
        "  --  " if reduction(s, t, it) is None
        else f"{reduction(s, t, it):+6.1f}"
        for it in iters
    )


def main():
    saved = {k: vr.CONFIG[k] for k in
             ("sigma_lo_scale", "eps_lo_ratio", "norm_inflate_3d", "itermax")}

    OUT_TXT.write_text(
        f"DBT breast extended sweep (det {DET}, axis={FILTER_AXIS}, "
        f"nviews={NVIEWS}, arc={ARC_DEG} deg)\n"
        f"Per experiment: shape/itermax then per trial:\n"
        "  | single@<iters> | two@<iters> | red@<iters>\n\n"
    )

    for exp_tag, shape, dx_cm, itermax, snap in EXPERIMENTS:
        print(f"\n##############  {exp_tag}: shape={shape}, dx={dx_cm}, "
              f"itermax={itermax}  ##############")
        with open(OUT_TXT, "a") as f:
            f.write(f"---- {exp_tag}: shape={shape}, dx={dx_cm} cm, "
                    f"itermax={itermax}, iters {snap} ----\n")

        vr.CONFIG["itermax"] = itermax
        phantom = build_phantom(shape, dx_cm)
        det_row, det_col, det_sp = DET
        vol_geom, proj_geom, geom_info = vr.build_dbt_geometry(
            phantom.shape, dx_cm,
            det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
            nviews=NVIEWS, arc_deg=ARC_DEG, sod=65.0, odd=5.0,
        )
        A, At = vr.make_projector(vol_geom, proj_geom)
        nrays = geom_info["nrays"]
        nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
            phantom.shape, A, At, vr.CONFIG["npower"],
        )

        t0 = time.time()
        rs, isng, ds_, ts_, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            nrays, snapshot_iters=snap,
        )
        single_time = time.time() - t0
        isng_np = np.asarray(isng)
        print(f"  [single {exp_tag}] {single_time:.0f}s "
              f"final RMSE {isng_np[-1]:.5f}")
        with open(OUT_TXT, "a") as f:
            f.write(f"  single  | {fmt_iters(isng_np, snap)} | "
                    f"({single_time:.0f}s)\n")

        R_hi, R_lo = vr.build_sinogram_filters(
            geom_info["det_col_count"], geom_info["det_spacing"],
            vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
            axis=FILTER_AXIS, det_row_count=geom_info["det_row_count"],
        )

        for tlabel, sig, r, inflate in TRIALS:
            vr.CONFIG["sigma_lo_scale"]    = sig
            vr.CONFIG["eps_lo_ratio"]      = r
            vr.CONFIG["norm_inflate_3d"]   = inflate
            t0 = time.time()
            rt, itwo, dt_, tt_, snaps_t = vr.run_two_channel(
                phantom, A, At, R_hi, R_lo,
                nusino, nuxgrad, nuygrad, nuzgrad,
                nrays, snapshot_iters=snap,
            )
            two_time = time.time() - t0
            itwo_np = np.asarray(itwo)
            line = (f"  {tlabel:<9} | {fmt_iters(isng_np, snap)} | "
                    f"{fmt_iters(itwo_np, snap)} | "
                    f"{fmt_reds(isng_np, itwo_np, snap)}  ({two_time:.0f}s)")
            print(line)
            with open(OUT_TXT, "a") as f:
                f.write(line + "\n")

        with open(OUT_TXT, "a") as f:
            f.write("\n")

    for k, v in saved.items():
        vr.CONFIG[k] = v
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
