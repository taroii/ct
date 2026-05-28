"""Reduced-config multi-parameter sweep for two-channel DBT breast recon.

Faster (240x240 detector, itermax 200) than the full config so we can
explore (filter axis, sigma_lo_scale, eps_lo_ratio, norm_inflate_3d, nviews)
in one pass. Each two-channel run is ~90 s on the current GPU.

The single-channel baseline is computed once per (axis, nviews) pair and
shared across the two-channel trials at that pair.

Output: cache/_breast_dbt_reduced_sweep.txt
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

import victre_reconstruction as vr  # noqa: E402
from presentation_ct2_phantom_ladder import (
    build_phantom_volume, PHANTOM_CONFIGS,
)  # noqa: E402

ITERMAX = 200
SNAPSHOT_ITERS = [50, 100, 200]
REPORT_ITERS = [50, 100, 200]
DET = (240, 240, 0.10)   # 24 cm at 1 mm spacing, half-resolution sweep config
NVIEWS_DEFAULT = 25
ARC_DEG = 50.0

# Inner sweep: per (axis, nviews), what two-channel knobs to try.
TWO_CHANNEL_TRIALS = [
    # (label, sigma_lo_scale, eps_lo_ratio, norm_inflate_3d)
    ("baseline",   4.0, 1.25, None),  # None -> sqrt(sigma_lo)
    ("inf1",       4.0, 1.25, 1.0),
    ("inf1p5",     4.0, 1.25, 1.5),
    ("siglo2",     2.0, 1.25, None),
    ("siglo2_inf1",2.0, 1.25, 1.0),
    ("r0p5",       4.0, 0.5,  None),
]

# Outer sweep: which (filter axis, nviews) combinations to try.
OUTER = [
    ("axu_n25",  "u",  25),
    ("axv_n25",  "v",  25),
    ("ax2d_n25", "2d", 25),
    ("ax2d_n15", "2d", 15),   # fewer views: more LAR-like
    ("ax2d_n9",  "2d",  9),   # Hologic-typical view count
]

OUT_TXT = ROOT / "cache" / "_breast_dbt_reduced_sweep.txt"


def reduction(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


def build(phantom_shape, dx_cm, nviews):
    det_row, det_col, det_sp = DET
    return vr.build_dbt_geometry(
        phantom_shape, dx_cm,
        det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
        nviews=nviews, arc_deg=ARC_DEG, sod=65.0, odd=5.0,
    )


def main():
    cfg = PHANTOM_CONFIGS["breast"]
    phantom = build_phantom_volume(cfg)

    saved = {k: vr.CONFIG[k] for k in
             ("sigma_lo_scale", "eps_lo_ratio", "norm_inflate_3d", "itermax")}
    vr.CONFIG["itermax"] = ITERMAX

    OUT_TXT.write_text(
        f"Reduced-config sweep (DBT breast, det {DET}, itermax {ITERMAX}).\n"
        "Per row: outer_tag  trial  | single@50 single@100 single@200  "
        "two@50 two@100 two@200 | red50 red100 red200\n\n"
    )

    for outer_tag, axis, nviews in OUTER:
        print(f"\n=========  {outer_tag}: axis={axis}, nviews={nviews}  =========")
        vol_geom, proj_geom, geom_info = build(phantom.shape, cfg["dx_cm"],
                                               nviews)
        A, At = vr.make_projector(vol_geom, proj_geom)
        nrays = geom_info["nrays"]
        nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
            phantom.shape, A, At, vr.CONFIG["npower"],
        )

        # Single channel once per outer setting.
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
            axis=axis, det_row_count=geom_info["det_row_count"],
        )

        for label, sig, r, inflate in TWO_CHANNEL_TRIALS:
            vr.CONFIG["sigma_lo_scale"]    = sig
            vr.CONFIG["eps_lo_ratio"]      = r
            vr.CONFIG["norm_inflate_3d"]   = inflate
            t0 = time.time()
            try:
                rt, itwo, dt_, tt_, snaps_t = vr.run_two_channel(
                    phantom, A, At, R_hi, R_lo,
                    nusino, nuxgrad, nuygrad, nuzgrad,
                    nrays, snapshot_iters=SNAPSHOT_ITERS,
                )
            except Exception as exc:
                line = f"{outer_tag} {label:<14}: ERRORED ({exc})"
                print(line)
                with open(OUT_TXT, "a") as f:
                    f.write(line + "\n")
                continue
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
            line = (f"{outer_tag} {label:<14} | {cells_s} | {cells_t} | "
                    f"{cells_r}  ({time.time()-t0:.0f}s)")
            print(line)
            with open(OUT_TXT, "a") as f:
                f.write(line + "\n")

    for k, v in saved.items():
        vr.CONFIG[k] = v
    print(f"\nWrote {OUT_TXT}")


if __name__ == "__main__":
    main()
