"""Arc sweep on the analytic phantoms (breast / head / jaw).

Question: does the two-channel advantage grow as the LAR arc narrows --
the same behaviour seen on Shepp-Logan and VICTRE -- or do the analytic
phantoms behave differently?

Reduced config for speed (halved detector -> ~1/4 the rays, itermax 250,
npower 100). This is a RELATIVE trend comparison; the reduced detector
can shift the absolute baseline (as it did for SL/VICTRE), so a winning
arc should be confirmed at the full talk config. Single-channel depends
only on geometry, so it is computed once per (phantom, arc).

Results are appended to cache/_ct2_arc_sweep.txt after every arc.
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

import victre_reconstruction as vr            # noqa: E402
import presentation_ct2_phantom_ladder as pl  # noqa: E402

import argparse

ARCS_DEFAULT = [30.0, 50.0, 75.0, 100.0, 120.0]
ITERMAX   = 250
NPOWER    = 100
SNAPSHOTS = None
REPORT    = [100, 150, 200, 250]
OUT       = ROOT / "cache" / "_ct2_arc_sweep.txt"

# Reduced detector per phantom: half the count, double the spacing --
# same physical coverage, ~1/4 the rays.
RED_DET = {
    "breast": (192, 192, 0.10),
    "head":   (272, 272, 0.10),
    "jaw":    (216, 216, 0.10),
}


def gap(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcs", type=float, nargs="+", default=ARCS_DEFAULT,
                        help="Arcs (deg) to sweep, e.g. 60 70 80 90")
    parser.add_argument("--phantoms", type=str, nargs="+",
                        default=["breast", "head", "jaw"],
                        help="Which phantoms to run")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing OUT file instead of overwriting")
    args = parser.parse_args()
    arcs = args.arcs

    vr.CONFIG["itermax"] = ITERMAX
    vr.CONFIG["npower"] = NPOWER

    header = ("phantom  arc  | " + " | ".join(f"i{i}" for i in REPORT))
    if not args.append:
        OUT.write_text(
            f"Analytic-phantom arc sweep (reduced config: itermax={ITERMAX}, "
            f"npower={NPOWER}, halved detector)\n"
            "two-channel vs single image-RMSE gap (%); positive favours "
            "two-channel\n" + header + "\n")
    print(header)

    for name in args.phantoms:
        cfg = pl.PHANTOM_CONFIGS[name]
        phantom = pl.build_phantom_volume(cfg)
        det_row, det_col, det_sp = RED_DET[name]

        for arc in arcs:
            vg, pg, gi = vr.build_geometry(
                phantom.shape, cfg["dx_cm"],
                det_row_count=det_row, det_col_count=det_col,
                det_spacing=det_sp, nviews=25, arc_deg=arc)
            A, At = vr.make_projector(vg, pg)
            nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
                phantom.shape, A, At, vr.CONFIG["npower"])
            nrays = gi["nrays"]

            t0 = time.time()
            try:
                _, single, _, _, _ = vr.run_single_channel(
                    phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
                    nrays, snapshot_iters=SNAPSHOTS)
                R_hi, R_lo = vr.build_sinogram_filters(
                    gi["det_col_count"], gi["det_spacing"],
                    vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"])
                _, two, _, _, _ = vr.run_two_channel(
                    phantom, A, At, R_hi, R_lo,
                    nusino, nuxgrad, nuygrad, nuzgrad,
                    nrays, snapshot_iters=SNAPSHOTS)
            except Exception as exc:                # keep the sweep going
                line = f"{name:<8} {arc:>4.0f} | ERRORED: {exc}"
                print(line)
                with open(OUT, "a") as f:
                    f.write(line + "\n")
                continue

            gaps = [gap(single, two, it) for it in REPORT]
            cells = " | ".join("  --  " if g is None else f"{g:+6.1f}"
                               for g in gaps)
            diverged = (not np.isfinite(two[-1])) or two[-1] > 10 * single[0]
            flag = "  DIVERGED" if diverged else ""
            line = (f"{name:<8} {arc:>4.0f} | {cells}"
                    f"   ({time.time()-t0:.0f}s){flag}")
            print(line)
            with open(OUT, "a") as f:
                f.write(line + "\n")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
