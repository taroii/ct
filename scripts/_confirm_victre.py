"""Confirm the VICTRE sweep findings at the real talk config (4x phantom
downsampling, 384 detector), not the reduced 8x/256 sweep config.

The fast sweep showed every combo positive (arc050 +8%), but the real
VICTRE run gave -11% at iter 200 -- so the reduced config flips the
baseline sign. This checks whether the narrow-arc win survives.

Tests, at 4x downsample, 384x384 @ 0.05 cm, itermax 400, arc 30 deg:
  arc030        -- narrowed LAR arc (matches the Shepp-Logan choice)
  arc030-siglo8 -- narrow arc + bigger LF dual step

Single-channel is geometry-only here, computed once and reused.
Writes a summary to cache/_victre_confirm.txt.
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

DET       = (384, 384, 0.05)     # real talk detector
ITERMAX   = 400
ARC       = 30.0
SNAPSHOTS = [50, 100, 200, 300, 400]
REPORT    = [100, 200, 300, 400]
OUT       = ROOT / "cache" / "_victre_confirm.txt"

COMBOS = [
    ("arc030",        {}),
    ("arc030-siglo8", {"sigma_lo_scale": 8.0}),
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

    vg, pg, gi = vr.build_geometry(
        phantom.shape, dx_cm,
        det_row_count=det_row, det_col_count=det_col,
        det_spacing=det_sp, nviews=25, arc_deg=ARC)
    A, At = vr.make_projector(vg, pg)
    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"])
    nrays = gi["nrays"]

    for k, v in DEFAULTS.items():
        vr.CONFIG[k] = v
    t0 = time.time()
    _, single, _, _, _ = vr.run_single_channel(
        phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
        nrays, snapshot_iters=SNAPSHOTS)
    print(f"[single @ arc {ARC:.0f}: {time.time()-t0:.0f}s]")

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
        _, it_, _, _, _ = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            nrays, snapshot_iters=SNAPSHOTS)
        gaps = [gap(single, it_, it) for it in REPORT]
        cells = " | ".join("  --  " if g is None else f"{g:+6.1f}" for g in gaps)
        line = (f"{name:<16} arc{ARC:>4.0f} | {cells}"
                f"   single@400={single[-1]:.4f} two@400={it_[-1]:.4f}"
                f"  ({time.time()-t0:.0f}s)")
        print(line)
        lines.append(line)

    header = "combo            arc  | " + " | ".join(f"i{i}" for i in REPORT)
    OUT.write_text(header + "\n" + "\n".join(lines) + "\n")
    print("\n" + header)
    for ln in lines:
        print(ln)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
