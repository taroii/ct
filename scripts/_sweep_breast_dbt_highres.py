"""High-resolution DBT breast sweep: 288x288x64 voxel grid (0.075 cm).

Tests whether bumping the volume resolution from the 144x144x32 grid
(46x over-determined effective) to 288x288x64 (~6x over-determined
effective) lets two-channel beat single-channel. Detector stays at
240x240 (1 mm) for reduced-config compute.

Inner trials are the two best variants from the standard-res sweep:
    baseline      (defaults: r=1.25, sigma_lo=4, inflate=2)
    inf1p5        (r=1.25, sigma_lo=4, inflate=1.5  -- best so far)

Outer sweep: nviews in {25, 15, 9}, all 2D Hanning filter.
Output: cache/_breast_dbt_highres_sweep.txt
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

from presentation_ct2_phantom_ladder import PHANTOM_CONFIGS, _install_grid_cache  # noqa: E402
import victre_reconstruction as vr  # noqa: E402
from phantom3d import image3D  # noqa: E402

import importlib  # noqa: E402

_install_grid_cache()

ITERMAX = 200
SNAPSHOT_ITERS = [50, 100, 200]
REPORT_ITERS = [50, 100, 200]

# High-resolution volume: 288x288x64 at 0.075 cm = 21.6 x 21.6 x 4.8 cm
# (same physical extent as the standard breast config).
HIGHRES_SHAPE = (288, 288, 64)
HIGHRES_DX_CM = 0.075
CENTER = (3.0, 0.0, 0.0)

DET = (240, 240, 0.10)
ARC_DEG = 50.0
FILTER_AXIS = "2d"

TRIALS = [
    # (label, sigma_lo_scale, eps_lo_ratio, norm_inflate_3d)
    ("baseline", 4.0, 1.25, None),   # default inflate = sqrt(4) = 2
    ("inf1p5",   4.0, 1.25, 1.5),    # best variant from prior sweep
]

OUTER = [
    ("n25", 25),
    ("n15", 15),
    ("n9",   9),
]

OUT_TXT = ROOT / "cache" / "_breast_dbt_highres_sweep.txt"


def reduction(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


def build_highres_phantom():
    mod = importlib.import_module("breast_phantom_demo")
    builder = getattr(mod, "build_breast_phantom")
    phantom = builder(breast_xc=0., breast_yc=0., breast_zc=0.)

    NX, NY, NZ = HIGHRES_SHAPE
    DX = HIGHRES_DX_CM
    xc, yc, zc = CENTER
    xlen, ylen, zlen = NX * DX, NY * DX, NZ * DX

    img = image3D(
        shape=(NX, NY, NZ),
        xlen=xlen, ylen=ylen, zlen=zlen,
        x0=xc - xlen / 2., y0=yc - ylen / 2., z0=zc - zlen / 2.,
    )
    t0 = time.time()
    phantom.embed_in(img)
    print(f"  embed_in: {time.time()-t0:.1f}s")
    vol = img.mat
    print(f"  volume range: [{vol.min():.4f}, {vol.max():.4f}]")
    print(f"  non-zero fraction: {(vol > 1e-6).mean():.3f}")
    vol = np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32)
    print(f"  astra-ordered shape: {vol.shape}")
    return vol


def main():
    phantom = build_highres_phantom()

    saved = {k: vr.CONFIG[k] for k in
             ("sigma_lo_scale", "eps_lo_ratio", "norm_inflate_3d", "itermax")}
    vr.CONFIG["itermax"] = ITERMAX

    n_voxels = int(np.prod(HIGHRES_SHAPE))
    n_nonzero = int((phantom > 1e-6).sum())
    print(f"\nVolume: {HIGHRES_SHAPE}, {n_voxels} voxels "
          f"({n_nonzero} non-zero, {100*n_nonzero/n_voxels:.1f}%)")

    OUT_TXT.write_text(
        f"High-res DBT breast sweep (volume {HIGHRES_SHAPE} @ {HIGHRES_DX_CM} cm, "
        f"det {DET}, axis={FILTER_AXIS}, itermax {ITERMAX}).\n"
        f"Non-zero voxel count: {n_nonzero}  total: {n_voxels}\n"
        "Per row: outer  trial  | single@50 single@100 single@200  "
        "two@50 two@100 two@200 | red50 red100 red200\n\n"
    )

    for outer_tag, nviews in OUTER:
        print(f"\n=========  {outer_tag}: nviews={nviews}  =========")
        det_row, det_col, det_sp = DET
        vol_geom, proj_geom, geom_info = vr.build_dbt_geometry(
            phantom.shape, HIGHRES_DX_CM,
            det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
            nviews=nviews, arc_deg=ARC_DEG, sod=65.0, odd=5.0,
        )
        A, At = vr.make_projector(vol_geom, proj_geom)
        nrays = geom_info["nrays"]
        nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
            phantom.shape, A, At, vr.CONFIG["npower"],
        )

        t0 = time.time()
        rs, isng, ds_, ts_, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            nrays, snapshot_iters=SNAPSHOT_ITERS,
        )
        print(f"  [single {outer_tag}] {time.time()-t0:.0f}s "
              f"final RMSE {isng[-1]:.5f}")

        R_hi, R_lo = vr.build_sinogram_filters(
            geom_info["det_col_count"], geom_info["det_spacing"],
            vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
            axis=FILTER_AXIS, det_row_count=geom_info["det_row_count"],
        )

        for label, sig, r, inflate in TRIALS:
            vr.CONFIG["sigma_lo_scale"]    = sig
            vr.CONFIG["eps_lo_ratio"]      = r
            vr.CONFIG["norm_inflate_3d"]   = inflate
            t0 = time.time()
            rt, itwo, dt_, tt_, snaps_t = vr.run_two_channel(
                phantom, A, At, R_hi, R_lo,
                nusino, nuxgrad, nuygrad, nuzgrad,
                nrays, snapshot_iters=SNAPSHOT_ITERS,
            )
            isng_np = np.asarray(isng); itwo_np = np.asarray(itwo)
            reds = [reduction(isng_np, itwo_np, it) for it in REPORT_ITERS]
            sing = [isng_np[it-1] for it in REPORT_ITERS]
            two  = [itwo_np[it-1] for it in REPORT_ITERS]
            cells_s = " ".join(f"{v:.5f}" for v in sing)
            cells_t = " ".join(f"{v:.5f}" for v in two)
            cells_r = " ".join(
                "   --  " if r is None else f"{r:+6.1f}"
                for r in reds
            )
            line = (f"{outer_tag} {label:<10} | {cells_s} | {cells_t} | "
                    f"{cells_r}  ({time.time()-t0:.0f}s)")
            print(line)
            with open(OUT_TXT, "a") as f:
                f.write(line + "\n")

    for k, v in saved.items():
        vr.CONFIG[k] = v
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
