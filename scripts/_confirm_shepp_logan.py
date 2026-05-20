"""Confirm the Shepp-Logan sweep findings at the *real* talk detector
(512 @ 0.05 cm), not the reduced sweep detector. The fast 256-detector
sweep showed every combo flipping positive; this checks whether that
holds at full detector resolution and longer iteration counts.

Tests, at 512x512 @ 0.05 cm, itermax 400:
  arc050-base   -- original config (full SL run gave -7.6% at iter 500)
  arc030        -- narrowed LAR arc, the sweep's standout
  arc030-siglo8 -- narrow arc + the best CP lever from the sweep

Writes a summary to cache/_sl_confirm.txt so nothing is lost to buffering.
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
SNAPSHOTS = [50, 100, 200, 300, 400]
REPORT    = [100, 200, 300, 400]
OUT       = ROOT / "cache" / "_sl_confirm.txt"

COMBOS = [
    ("arc050-base",   50.0, {}),
    ("arc030",        30.0, {}),
    ("arc030-siglo8", 30.0, {"sigma_lo_scale": 8.0}),
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

    geom_cache, single_cache = {}, {}
    lines = []

    for name, arc, ov in COMBOS:
        for k, v in DEFAULTS.items():
            vr.CONFIG[k] = v
        for k, v in ov.items():
            vr.CONFIG[k] = v

        if arc not in geom_cache:
            vg, pg, gi = vr.build_geometry(
                phantom.shape, sl.DX_CM,
                det_row_count=det_row, det_col_count=det_col,
                det_spacing=det_sp, nviews=25, arc_deg=arc)
            A, At = vr.make_projector(vg, pg)
            norms = vr.operator_norms(phantom.shape, A, At, vr.CONFIG["npower"])
            geom_cache[arc] = (A, At, gi, norms)

        A, At, gi, (nusino, nuxgrad, nuygrad, nuzgrad) = geom_cache[arc]
        nrays = gi["nrays"]

        if arc not in single_cache:
            t0 = time.time()
            _, is_, _, _, _ = vr.run_single_channel(
                phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
                nrays, snapshot_iters=SNAPSHOTS)
            single_cache[arc] = is_
            print(f"[single @ arc {arc:.0f}: {time.time()-t0:.0f}s]")
        single = single_cache[arc]

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
        line = (f"{name:<16} arc{arc:>4.0f} | " + cells +
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
