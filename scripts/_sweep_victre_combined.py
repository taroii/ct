"""VICTRE combined sweep over (arc, sigma_lo_scale, eps_lo_ratio).

The fixed-50-deg `scripts/_sweep_victre_cp.py` showed two-channel cannot
win at the DBT-typical 50 deg arc by step-size tuning alone. Reduced-config
arc-sweep results in `cache/_victre_sweep.txt` show clear two-channel wins
once the arc drops to ~30 deg, especially with `sigma_lo_scale=8`. This
script runs the full-resolution config across a small grid of arcs and
CP knobs to verify those reduced-config trends and look for the best
sustained two-channel point.

Per-arc, single-channel is computed once and reused across all two-channel
combos at that arc. Results are appended to cache/_victre_combined.txt
after every run so a kill never loses completed work.

Run:  conda run --no-capture-output -n ct \
          python scripts/_sweep_victre_combined.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import victre_reconstruction as vr   # noqa: E402

DET       = (384, 384, 0.05)         # real talk detector
ITERMAX   = 400
SNAPSHOTS = [50, 100, 200, 300, 400]
REPORT    = [100, 200, 300, 400]
OUT       = ROOT / "cache" / "_victre_combined.txt"

# Per-arc, the two-channel combos to try.
# tuple = (label, sigma_lo_scale, eps_lo_ratio).
# norm_inflate_3d stays at default (None -> sqrt(sigma_lo_scale)).
ARC_COMBOS = {
    30.0: [
        ("base",          4.0,  1.25),
        ("siglo8",        8.0,  1.25),
        ("r10",           4.0, 10.00),
        ("siglo8-r10",    8.0, 10.00),
    ],
    40.0: [
        ("siglo8",        8.0,  1.25),
    ],
    50.0: [
        ("r10",           4.0, 10.00),    # does loose r alone help at 50?
    ],
}

DEFAULTS = {
    "sigma_lo_scale":   4.0,
    "norm_inflate_3d":  None,
    "eps_lo_ratio":     1.25,
    "eps_hi_ratio":     1.0,
    "cutoffparm":       4.0,
    "cutoffparm_lo":    8.0,
}


def gap(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


def main():
    vr.DOWNSAMPLE = 4
    phantom, dx_cm = vr.load_and_downsample_phantom()
    det_row, det_col, det_sp = DET
    vr.CONFIG["itermax"] = ITERMAX

    header = ("arc  combo            | " +
              " | ".join(f"i{i}" for i in REPORT) +
              "   | single@400 two@400")
    OUT.write_text(
        f"VICTRE combined sweep over (arc, sigma_lo_scale, eps_lo_ratio).\n"
        f"det={DET}, downsample=4, itermax={ITERMAX}.\n"
        "two-channel vs single image-RMSE gap (%); positive favours "
        "two-channel\n" + header + "\n")
    print(header)

    for arc, combos in ARC_COMBOS.items():
        print(f"\n=== arc {arc:.0f} deg ===")
        vg, pg, gi = vr.build_geometry(
            phantom.shape, dx_cm,
            det_row_count=det_row, det_col_count=det_col,
            det_spacing=det_sp, nviews=25, arc_deg=arc)
        A, At = vr.make_projector(vg, pg)
        nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
            phantom.shape, A, At, vr.CONFIG["npower"])
        nrays = gi["nrays"]

        # Single-channel once per arc.
        for k, v in DEFAULTS.items():
            vr.CONFIG[k] = v
        t0 = time.time()
        _, single, _, _, _ = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            nrays, snapshot_iters=SNAPSHOTS)
        print(f"  [single arc={arc:.0f}: {time.time()-t0:.0f}s, "
              f"final RMSE {single[-1]:.4f}]")

        for label, sig, r in combos:
            for k, v in DEFAULTS.items():
                vr.CONFIG[k] = v
            vr.CONFIG["sigma_lo_scale"] = sig
            vr.CONFIG["eps_lo_ratio"]   = r

            R_hi, R_lo = vr.build_sinogram_filters(
                gi["det_col_count"], gi["det_spacing"],
                vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"])

            t0 = time.time()
            try:
                _, it_, _, _, _ = vr.run_two_channel(
                    phantom, A, At, R_hi, R_lo,
                    nusino, nuxgrad, nuygrad, nuzgrad,
                    nrays, snapshot_iters=SNAPSHOTS)
            except Exception as exc:
                line = (f"{arc:>3.0f}  {label:<16} | ERRORED: {exc}")
                print(line)
                with open(OUT, "a") as f:
                    f.write(line + "\n")
                continue

            gaps = [gap(single, it_, it) for it in REPORT]
            cells = " | ".join("  --  " if g is None else f"{g:+6.1f}"
                               for g in gaps)
            diverged = (not np.isfinite(it_[-1])) or it_[-1] > 10 * single[0]
            flag = "  DIVERGED" if diverged else ""
            line = (f"{arc:>3.0f}  {label:<16} | {cells}"
                    f"   | {single[-1]:.4f}   {it_[-1]:.4f}"
                    f"  ({time.time()-t0:.0f}s){flag}")
            print(line)
            with open(OUT, "a") as f:
                f.write(line + "\n")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
