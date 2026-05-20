"""Shepp-Logan CP-parameter sweep at a FIXED 50 deg LAR arc.

Per team guidance, the LAR arc stays at the DBT-typical 50 deg (already
very narrow in practice) -- we look for a two-channel win by tuning the
Chambolle-Pock step-size parameters instead, not by narrowing the arc.

Runs at the real talk detector (512 @ 0.05 cm), itermax 400, so results
are directly trustworthy (the reduced 256-detector sweep flips the
baseline sign and is not a faithful proxy). The arc is fixed, so the
geometry, operator norms, and single-channel recon are computed ONCE and
reused across every two-channel combo.

Levers (all pure two-channel -- single-channel is invariant to them):
  norm_inflate_3d  -- 3D step-size handicap. tau_two = tau_single/inflate.
                      Default sqrt(sigma_lo_scale)=2.0 halves the primal
                      step. Smaller -> less handicap, more divergence risk.
  sigma_lo_scale   -- LF dual step multiplier (default 4.0).
  eps_lo_ratio     -- LF data-tolerance ratio (default 1.25; smaller =
                      tighter LF fit).

Results are appended to cache/_sl_cp_sweep.txt after every combo so a
kill never loses completed runs.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ct-2"))

import victre_reconstruction as vr            # noqa: E402
import presentation_shepp_logan_ladder as sl  # noqa: E402

DET       = (512, 512, 0.05)     # real talk detector
ITERMAX   = 400
ARC       = 50.0                 # FIXED -- arc is not a sweep lever here
SNAPSHOTS = [50, 100, 200, 300, 400]
REPORT    = [100, 200, 300, 400]
OUT       = ROOT / "cache" / "_sl_cp_sweep.txt"

# name, CONFIG overrides. norm_inflate_3d=None -> sqrt(sigma_lo_scale).
COMBOS = [
    ("base",                {}),
    ("inflate1.7",          {"norm_inflate_3d": 1.70}),
    ("inflate1.5",          {"norm_inflate_3d": 1.50}),
    ("inflate1.3",          {"norm_inflate_3d": 1.30}),
    ("inflate1.15",         {"norm_inflate_3d": 1.15}),
    ("siglo8-inf1.5",       {"sigma_lo_scale": 8.0, "norm_inflate_3d": 1.50}),
    ("siglo8-eps0.6-inf1.5", {"sigma_lo_scale": 8.0, "eps_lo_ratio": 0.60,
                              "norm_inflate_3d": 1.50}),
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
    phantom = sl.build_volume()
    det_row, det_col, det_sp = DET
    vr.CONFIG["itermax"] = ITERMAX

    # Geometry + norms: arc is fixed, so this is computed exactly once.
    vg, pg, gi = vr.build_geometry(
        phantom.shape, sl.DX_CM,
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

    header = ("combo                   | " +
              " | ".join(f"i{i}" for i in REPORT) +
              "   | single@400 two@400")
    OUT.write_text(f"Shepp-Logan CP sweep, arc={ARC:.0f} deg, "
                   f"det={DET}, itermax={ITERMAX}\n" + header + "\n")
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
            line = f"{name:<23} | ERRORED: {exc}"
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
        line = (f"{name:<23} | {cells}"
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
