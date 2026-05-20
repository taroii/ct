"""VICTRE breast-phantom CP-parameter sweep at a FIXED 50 deg LAR arc.

Mirror of scripts/_sweep_shepp_logan_cp.py, applied to the realistic
voxelized VICTRE breast phantom. Per team guidance the LAR arc stays at
the DBT-typical 50 deg; we look for a two-channel win by tuning the
Chambolle-Pock step-size parameters, not by narrowing the arc.

Runs at the real talk config -- 4x phantom downsampling, 384x384 @ 0.05
cm detector, itermax 400 -- so results are directly trustworthy (the
reduced 8x/256 sweep config flips the baseline sign). The arc is fixed,
so the geometry, operator norms, and single-channel recon are computed
ONCE and reused across every two-channel combo.

Levers (pure two-channel -- single-channel is invariant to them):
  norm_inflate_3d  -- 3D step-size handicap; tau_two = tau_single/inflate.
  sigma_lo_scale   -- LF dual step multiplier (default 4.0).

The VICTRE 50 deg baseline was the deepest deficit seen (~-11% at i200,
~-26% at i500), so this is the hardest case to flip.

Results are appended to cache/_victre_cp_sweep.txt after every combo so a
kill never loses completed runs.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)                       # victre loader resolves the .raw from cwd

import victre_reconstruction as vr   # noqa: E402

DET       = (384, 384, 0.05)     # real talk detector
ITERMAX   = 400
ARC       = 50.0                 # FIXED -- arc is not a sweep lever here
SNAPSHOTS = [50, 100, 200, 300, 400]
REPORT    = [100, 200, 300, 400]
OUT       = ROOT / "cache" / "_victre_cp_sweep.txt"

# name, CONFIG overrides. norm_inflate_3d=None -> sqrt(sigma_lo_scale).
COMBOS = [
    ("base",            {}),
    ("inflate1.7",      {"norm_inflate_3d": 1.70}),
    ("inflate1.5",      {"norm_inflate_3d": 1.50}),
    ("inflate1.3",      {"norm_inflate_3d": 1.30}),
    ("siglo8",          {"sigma_lo_scale": 8.0}),   # auto inflate -> sqrt(8)
    ("siglo8-inf1.7",   {"sigma_lo_scale": 8.0, "norm_inflate_3d": 1.70}),
]

DEFAULTS = {
    "sigma_lo_scale": 4.0, "norm_inflate_3d": None,
    "eps_lo_ratio": 1.25, "eps_hi_ratio": 1.0,
    "cutoffparm": 4.0, "cutoffparm_lo": 8.0,
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

    # Geometry + norms: arc is fixed, so this is computed exactly once.
    vg, pg, gi = vr.build_geometry(
        phantom.shape, dx_cm,
        det_row_count=det_row, det_col_count=det_col,
        det_spacing=det_sp, nviews=25, arc_deg=ARC)
    A, At = vr.make_projector(vg, pg)
    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"])
    nrays = gi["nrays"]

    # Single-channel: geometry-only, computed once and reused.
    for k, v in DEFAULTS.items():
        vr.CONFIG[k] = v
    t0 = time.time()
    _, single, _, _, _ = vr.run_single_channel(
        phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
        nrays, snapshot_iters=SNAPSHOTS)
    print(f"[single @ arc {ARC:.0f}: {time.time()-t0:.0f}s]\n")

    header = ("combo            | " +
              " | ".join(f"i{i}" for i in REPORT) +
              "   | single@400 two@400")
    OUT.write_text(f"VICTRE CP sweep, arc={ARC:.0f} deg, "
                   f"det={DET}, downsample=4, itermax={ITERMAX}\n" +
                   header + "\n")
    print(header)

    lines = []
    for name, ov in COMBOS:
        for k, v in DEFAULTS.items():
            vr.CONFIG[k] = v
        for k, v in ov.items():
            vr.CONFIG[k] = v

        R_hi, R_lo = vr.build_sinogram_filters(
            gi["det_col_count"], gi["det_spacing"],
            vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"])

        t0 = time.time()
        try:
            _, it_, _, _, _ = vr.run_two_channel(
                phantom, A, At, R_hi, R_lo,
                nusino, nuxgrad, nuygrad, nuzgrad,
                nrays, snapshot_iters=SNAPSHOTS)
        except Exception as exc:               # keep the sweep going
            line = f"{name:<16} | ERRORED: {exc}"
            print(line)
            lines.append(line)
            with open(OUT, "a") as f:
                f.write(line + "\n")
            continue

        gaps = [gap(single, it_, it) for it in REPORT]
        cells = " | ".join("  --  " if g is None else f"{g:+6.1f}"
                           for g in gaps)
        diverged = (not np.isfinite(it_[-1])) or it_[-1] > 10 * single[0]
        flag = "  DIVERGED" if diverged else ""
        line = (f"{name:<16} | {cells}"
                f"   | {single[-1]:.4f}   {it_[-1]:.4f}"
                f"  ({time.time()-t0:.0f}s){flag}")
        print(line)
        lines.append(line)
        with open(OUT, "a") as f:                # incremental save
            f.write(line + "\n")

    print("\n" + "=" * 70)
    print(header)
    for ln in lines:
        print(ln)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
