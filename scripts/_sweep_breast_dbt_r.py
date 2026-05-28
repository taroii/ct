"""Sweep eps_lo_ratio (and optionally norm_inflate_3d) for the analytic
breast at the full DBT config to find a regime where two-channel beats
single-channel.

Reads cache/ct2_breast_recon.pkl for the baseline single-channel result
(produced by presentation_ct2_phantom_ladder.py --phantom breast).
Skips single, sweeps two-channel only, writes per-run pickles plus a
summary text file.

Run:
    conda run --no-capture-output -n ct \
        python scripts/_sweep_breast_dbt_r.py
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
from presentation_ct2_phantom_ladder import (
    build_phantom_volume, PHANTOM_CONFIGS,
)  # noqa: E402

# (label, eps_lo_ratio, norm_inflate_3d).
# inflate=None means default = sqrt(sigma_lo_scale) = 2.
TRIALS = [
    # 2D-Hanning filter (joint u,v LF lobe). Default r=1.25 first to
    # confirm the fix, then a small sweep around it.
    ("uv_r1p25",  1.25,  None),
    ("uv_r2p5",   2.5,   None),
    ("uv_r5",     5.0,   None),
    ("uv_r10",    10.0,  None),
]

SNAPSHOT_ITERS = [10, 50, 100, 200, 300, 500]
REPORT_ITERS = [50, 100, 200, 300, 500]

CACHE = ROOT / "cache"
OUT_TXT = CACHE / "_breast_dbt_r_sweep.txt"


def run_two_channel_only(phantom, A, At, R_hi, R_lo,
                         nusino, nuxgrad, nuygrad, nuzgrad, nrays):
    return vr.run_two_channel(
        phantom, A, At, R_hi, R_lo,
        nusino, nuxgrad, nuygrad, nuzgrad,
        nrays, snapshot_iters=SNAPSHOT_ITERS,
    )


def reduction(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


def main():
    cfg = PHANTOM_CONFIGS["breast"]

    # Reuse the existing single-channel baseline.
    baseline_pkl = CACHE / "ct2_breast_recon.pkl"
    print(f"Loading single-channel baseline from {baseline_pkl}")
    with open(baseline_pkl, "rb") as f:
        baseline = pickle.load(f)
    ierrs_single = np.array(baseline["ierrs_single"])
    print(f"  single-channel iter-500 RMSE: {ierrs_single[-1]:.6f}")
    for it in REPORT_ITERS:
        print(f"  single iter {it:3d}: {ierrs_single[it-1]:.6f}")

    # Rebuild geometry + projector once (matches presentation script).
    phantom = build_phantom_volume(cfg)
    det_row, det_col, det_sp = cfg["det"]
    vol_geom, proj_geom, geom_info = vr.build_dbt_geometry(
        phantom.shape, cfg["dx_cm"],
        det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
        arc_deg=50.0, sod=cfg["sod"], odd=cfg["odd"],
    )
    A, At = vr.make_projector(vol_geom, proj_geom)
    R_hi, R_lo = vr.build_sinogram_filters(
        geom_info["det_col_count"], geom_info["det_spacing"],
        vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
        axis="2d", det_row_count=geom_info["det_row_count"],
    )
    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"],
    )
    nrays = geom_info["nrays"]

    # Save defaults so we can restore after each trial.
    saved = {
        "eps_lo_ratio":     vr.CONFIG["eps_lo_ratio"],
        "norm_inflate_3d":  vr.CONFIG["norm_inflate_3d"],
        "itermax":          vr.CONFIG["itermax"],
    }
    vr.CONFIG["itermax"] = cfg["itermax"]

    OUT_TXT.write_text(
        "Analytic breast (DBT geometry) -- two-channel sweep over r and inflate.\n"
        "Single-channel image RMSE: " +
        " ".join(f"i{it}={ierrs_single[it-1]:.5f}" for it in REPORT_ITERS) +
        "\n\n"
        "label        | iter100  iter200  iter300  iter500 "
        "|  red100  red200  red300  red500\n"
    )

    summary_rows = []
    for label, r, inflate in TRIALS:
        vr.CONFIG["eps_lo_ratio"]    = r
        vr.CONFIG["norm_inflate_3d"] = inflate
        print(f"\n=== {label}: r={r}, inflate={inflate} ===")

        t0 = time.time()
        try:
            rt, it_, dt_, tt_, snaps_t = run_two_channel_only(
                phantom, A, At, R_hi, R_lo,
                nusino, nuxgrad, nuygrad, nuzgrad, nrays,
            )
        except Exception as exc:
            line = f"{label:<12}: ERRORED ({exc})"
            print(line)
            with open(OUT_TXT, "a") as f:
                f.write(line + "\n")
            continue
        ierrs_two = np.array(it_)
        dt = time.time() - t0

        # Save full cache for the best ones.
        out_pkl = CACHE / f"ct2_breast_recon_dbt_{label}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump({
                "phantom":         baseline["phantom"],
                "recon_single":    baseline["recon_single"],
                "recon_two":       rt,
                "ierrs_single":    baseline["ierrs_single"],
                "ierrs_two":       it_,
                "derrs_single":    baseline["derrs_single"],
                "derrs_two":       dt_,
                "tvs_single":      baseline["tvs_single"],
                "tvs_two":         tt_,
                "snapshots_single": baseline["snapshots_single"],
                "snapshots_two":    snaps_t,
                "dx_cm":           cfg["dx_cm"],
                "geometry":        geom_info,
                "r":               r,
                "inflate":         inflate,
            }, f)
        print(f"  cached -> {out_pkl}  ({dt:.0f}s)")

        cells = " ".join(f"{ierrs_two[it-1]:.5f}" for it in REPORT_ITERS[1:])
        reds = " ".join(
            "  --  " if reduction(ierrs_single, ierrs_two, it) is None
            else f"{reduction(ierrs_single, ierrs_two, it):+6.1f}"
            for it in REPORT_ITERS[1:]
        )
        line = f"{label:<12} | {cells} | {reds}"
        print(line)
        with open(OUT_TXT, "a") as f:
            f.write(line + "\n")
        summary_rows.append((label, r, inflate, ierrs_two))

    # Restore defaults.
    for k, v in saved.items():
        vr.CONFIG[k] = v

    print(f"\nWrote summary -> {OUT_TXT}")

    if summary_rows:
        print("\n--- best two-channel iter-100 reduction ---")
        best = max(summary_rows,
                   key=lambda r: reduction(ierrs_single, r[3], 100) or -1e9)
        print(f"  {best[0]}: iter100 = {best[3][99]:.5f}  "
              f"(reduction {reduction(ierrs_single, best[3], 100):+.1f}%)")


if __name__ == "__main__":
    main()
