"""Validate the overnight sweep's best DBT breast configurations at
itermax=500. These are the candidates for the deck.

Configs:
  V1: 432x432x96 voxels, 50 deg arc, 25 views, default CP, iter 500
      (matches P4_default which won +4.7% at iter 200)
  V2: 432x432x96 voxels, 15 deg arc, 25 views, default CP, iter 500
      (combine high-res + narrow arc -- two biggest wins compounded)
  V3: 432x432x96 voxels, 50 deg arc, 25 views, cutoffparm_lo=4, iter 500
      (combine high-res with the best P1 single-knob change)

Reduced detector 240x240 (0.1 cm) at all configs since that is where
the wins live.
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
import pickle

_install_grid_cache()

OUT_TXT = ROOT / "cache" / "_breast_dbt_validate.txt"
CACHE = ROOT / "cache"
SHAPE = (432, 432, 96)
DX_CM = 0.05
DET = (240, 240, 0.10)
CENTER = (3.0, 0.0, 0.0)
NVIEWS = 25
ITERMAX = 500
SNAPSHOT_ITERS = [50, 100, 200, 300, 400, 500]
REPORT_ITERS   = SNAPSHOT_ITERS

ANCHOR_CP = {
    "sigma_lo_scale":  4.0,
    "norm_inflate_3d": 1.5,
    "eps_hi_ratio":    1.0,
    "eps_lo_ratio":    1.25,
    "cutoffparm":      4.0,
    "cutoffparm_lo":   8.0,
    "beta":            5.0,
}

CONFIGS = [
    # (label, arc_deg, axis, CP-overrides)
    ("V1_arc50_default",  50.0, "2d", {}),
    ("V2_arc15_default",  15.0, "2d", {}),
    ("V3_arc50_cutLo4",   50.0, "2d", {"cutoffparm_lo": 4.0}),
]


def reduction(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


def fmt_iters(arr, iters):
    return " ".join(f"{arr[it-1]:.5f}" for it in iters)


def fmt_reds(s, t, iters):
    return " ".join(
        "   --  " if reduction(s, t, it) is None
        else f"{reduction(s, t, it):+6.1f}"
        for it in iters
    )


def log(line):
    print(line)
    with open(OUT_TXT, "a") as f:
        f.write(line + "\n")


def build_phantom():
    mod = importlib.import_module("breast_phantom_demo")
    builder = getattr(mod, "build_breast_phantom")
    phantom = builder(breast_xc=0., breast_yc=0., breast_zc=0.)
    NX, NY, NZ = SHAPE
    xc, yc, zc = CENTER
    xlen, ylen, zlen = NX * DX_CM, NY * DX_CM, NZ * DX_CM
    img = image3D(shape=(NX, NY, NZ),
                  xlen=xlen, ylen=ylen, zlen=zlen,
                  x0=xc - xlen/2., y0=yc - ylen/2., z0=zc - zlen/2.)
    phantom.embed_in(img)
    vol = img.mat
    vol = np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32)
    return vol


def reset_cp():
    for k, v in ANCHOR_CP.items():
        vr.CONFIG[k] = v


def main():
    if OUT_TXT.exists():
        OUT_TXT.unlink()
    vr.CONFIG["itermax"] = ITERMAX
    log(f"DBT breast validation: shape={SHAPE}, dx={DX_CM} cm, "
        f"det={DET}, nviews={NVIEWS}, itermax={ITERMAX}, snapshots={SNAPSHOT_ITERS}")
    log(f"anchor CP: {ANCHOR_CP}\n")

    phantom = build_phantom()
    log(f"Phantom: range [{phantom.min():.4f}, {phantom.max():.4f}], "
        f"non-zero {(phantom > 1e-6).mean()*100:.1f}%\n")

    # Cache single per (arc, axis) since they only depend on geometry.
    seen_single = {}

    for label, arc, axis, cp_over in CONFIGS:
        log(f"##### {label}: arc={arc}, axis={axis}, "
            f"overrides={cp_over} #####")
        reset_cp()
        det_row, det_col, det_sp = DET
        vol_geom, proj_geom, gi = vr.build_dbt_geometry(
            phantom.shape, DX_CM,
            det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
            nviews=NVIEWS, arc_deg=arc, sod=65.0, odd=5.0,
        )
        A, At = vr.make_projector(vol_geom, proj_geom)
        nrays = gi["nrays"]
        nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
            phantom.shape, A, At, vr.CONFIG["npower"],
        )

        single_key = (arc,)
        if single_key in seen_single:
            isng_np, recon_single, snaps_single = seen_single[single_key]
            log(f"  reusing single from arc={arc}")
        else:
            reset_cp()
            t0 = time.time()
            rs, isng, ds_, ts_, snaps_single = vr.run_single_channel(
                phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
                nrays, snapshot_iters=SNAPSHOT_ITERS,
            )
            isng_np = np.asarray(isng)
            recon_single = rs
            log(f"  single | {fmt_iters(isng_np, REPORT_ITERS)}  "
                f"({time.time()-t0:.0f}s)")
            seen_single[single_key] = (isng_np, recon_single, snaps_single)

        # apply CP overrides for this trial
        reset_cp()
        for k, v in cp_over.items():
            vr.CONFIG[k] = v
        R_hi, R_lo = vr.build_sinogram_filters(
            gi["det_col_count"], gi["det_spacing"],
            vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
            axis=axis, det_row_count=gi["det_row_count"],
        )
        t0 = time.time()
        rt, itwo, dt_, tt_, snaps_two = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            nrays, snapshot_iters=SNAPSHOT_ITERS,
        )
        itwo_np = np.asarray(itwo)
        dt = time.time() - t0
        log(f"  two    | {fmt_iters(itwo_np, REPORT_ITERS)}  ({dt:.0f}s)")
        log(f"  red    | {fmt_reds(isng_np, itwo_np, REPORT_ITERS)}")

        # cache the run so we can build figures later
        out_pkl = CACHE / f"ct2_breast_recon_dbt_{label}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump({
                "phantom":          phantom,
                "recon_single":     recon_single,
                "recon_two":        rt,
                "ierrs_single":     isng_np.tolist(),
                "ierrs_two":        itwo_np.tolist(),
                "snapshots_single": snaps_single,
                "snapshots_two":    snaps_two,
                "dx_cm":            DX_CM,
                "geometry":         gi,
                "arc_deg":          arc,
                "config_label":     label,
                "cp_overrides":     cp_over,
            }, f)
        log(f"  cached -> {out_pkl}\n")

    log("Done.")


if __name__ == "__main__":
    main()
