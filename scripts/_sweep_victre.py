"""Parameter sweep on the VICTRE breast phantom, applying the Shepp-Logan
finding (narrowing the LAR arc flips two-channel from a loss to a win).

For speed the sweep uses 8x phantom downsampling (vs 4x for the talk
figure) and a reduced detector + itermax; this is a *relative* comparison.
Re-run the winning combo at full resolution to confirm.

Single-channel depends only on geometry + {stepbalance, eps, beta, rho,
tv_cap}; computed once per arc and reused across CP-only combos.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import os
os.chdir(ROOT)   # victre_reconstruction resolves the .raw path from cwd

import victre_reconstruction as vr   # noqa: E402

# --- sweep settings -------------------------------------------------------
SWEEP_DOWNSAMPLE = 8            # 0.8 mm voxels (talk figure uses 4 = 0.4 mm)
DET        = (256, 256, 0.05)   # 12.8 cm coverage, enough for the ~9 cm breast
ITERMAX    = 250
SNAPSHOTS  = [50, 100, 150, 200, 250]
REPORT     = [100, 150, 200, 250]

COMBOS = [
    # arc sweep, default CP
    ("arc020",        20.0, {}),
    ("arc030",        30.0, {}),
    ("arc050-base",   50.0, {}),
    ("arc090",        90.0, {}),
    # narrow arc + bigger LF dual step
    ("arc030-siglo8", 30.0, {"sigma_lo_scale": 8.0}),
    ("arc020-siglo8", 20.0, {"sigma_lo_scale": 8.0}),
]

DEFAULTS = {
    "sigma_lo_scale": 4.0, "norm_inflate_3d": None,
    "eps_lo_ratio": 1.25, "eps_hi_ratio": 1.0,
    "cutoffparm": 4.0, "cutoffparm_lo": 8.0,
}
OUT = ROOT / "cache" / "_victre_sweep.txt"


def gap(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


def main():
    vr.DOWNSAMPLE = SWEEP_DOWNSAMPLE
    phantom, dx_cm = vr.load_and_downsample_phantom()
    det_row, det_col, det_sp = DET
    vr.CONFIG["itermax"] = ITERMAX

    geom_cache, single_cache = {}, {}
    lines = []
    print("=" * 78)

    for name, arc, ov in COMBOS:
        for k, v in DEFAULTS.items():
            vr.CONFIG[k] = v
        for k, v in ov.items():
            vr.CONFIG[k] = v

        if arc not in geom_cache:
            vg, pg, gi = vr.build_geometry(
                phantom.shape, dx_cm,
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
            print(f"  [single @ arc {arc:.0f}: {time.time()-t0:.0f}s]")
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
        diverged = (not np.isfinite(it_[-1])) or it_[-1] > 10 * single[0]
        flag = "  DIVERGED" if diverged else ""
        line = f"{name:<16} arc{arc:>4.0f} | {cells}   ({time.time()-t0:.0f}s){flag}"
        print(line)
        lines.append(line)

    header = "combo            arc  | " + " | ".join(f"i{i}" for i in REPORT)
    OUT.write_text(header + "\n" + "\n".join(lines) + "\n")
    print("=" * 78)
    print(header)
    for ln in lines:
        print(ln)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
