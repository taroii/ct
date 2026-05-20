"""Parameter sweep on the 3D Shepp-Logan phantom, looking for a regime
where two-channel beats single-channel.

Levers tested:
  * arc_deg          -- LAR angular range (25 views, fixed)
  * norm_inflate_3d  -- the 3D stability factor; smaller => two-channel
                        gets a larger tau (less handicap vs single)
  * eps_lo_ratio     -- LF data-tolerance; smaller => tighter LF fit
  * sigma_lo_scale   -- LF dual step multiplier

Runs at a reduced detector (256 @ 0.1 cm) and itermax to keep each combo
fast; this is a *relative* comparison. Re-run the winning combo at full
resolution to confirm.

Single-channel depends only on geometry + {stepbalance, eps, beta, rho,
tv_cap}, so it is computed once per arc and reused across CP-only combos.
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

# --- sweep settings -------------------------------------------------------
DET        = (256, 256, 0.10)   # 25.6 cm coverage, 4x fewer rays than talk fig
ITERMAX    = 250
SNAPSHOTS  = [50, 100, 150, 200, 250]
REPORT_ITERS = [100, 150, 200, 250]

# Each combo: name, arc_deg, CONFIG overrides (two-channel-relevant only
# unless noted). arc change forces a single-channel recompute.
COMBOS = [
    # Phase A -- arc sweep, default CP params
    ("arc030",        30.0,  {}),
    ("arc050-base",   50.0,  {}),
    ("arc090",        90.0,  {}),
    ("arc120",       120.0,  {}),
    # Phase B -- reduce the 3D tau handicap (default inflate = sqrt(4) = 2.0)
    ("inflate1.30",   50.0,  {"norm_inflate_3d": 1.30}),
    ("inflate1.50",   50.0,  {"norm_inflate_3d": 1.50}),
    ("inflate1.75",   50.0,  {"norm_inflate_3d": 1.75}),
    # Phase C -- tighten the LF data tolerance
    ("epslo0.25",     50.0,  {"eps_lo_ratio": 0.25}),
    ("epslo0.50",     50.0,  {"eps_lo_ratio": 0.50}),
    ("epslo0.75",     50.0,  {"eps_lo_ratio": 0.75}),
    # Phase D -- LF dual step multiplier
    ("siglo2",        50.0,  {"sigma_lo_scale": 2.0}),
    ("siglo8",        50.0,  {"sigma_lo_scale": 8.0}),
    # Phase E -- combine the two most promising levers (filled by trends)
    ("inflate1.5+epslo0.5", 50.0, {"norm_inflate_3d": 1.50, "eps_lo_ratio": 0.50}),
]

DEFAULTS = {
    "sigma_lo_scale": 4.0,
    "norm_inflate_3d": None,
    "eps_lo_ratio": 1.25,
    "eps_hi_ratio": 1.0,
    "cutoffparm": 4.0,
    "cutoffparm_lo": 8.0,
}


def gap(single, two, it):
    """Relative RMSE improvement of two-channel over single at iter `it`."""
    if it - 1 >= len(single) or it - 1 >= len(two):
        return None
    s, t = single[it - 1], two[it - 1]
    if not np.isfinite(s) or not np.isfinite(t) or s <= 0:
        return None
    return (s - t) / s * 100.0


def main():
    phantom = sl.build_volume()
    det_row, det_col, det_sp = DET

    vr.CONFIG["itermax"] = ITERMAX

    # cache: arc_deg -> (single ierrs, geometry tuple)
    single_cache = {}
    geom_cache = {}

    results = []
    print("\n" + "=" * 78)
    print(f"{'combo':<22} {'arc':>5} | " +
          " | ".join(f"i{it}" for it in REPORT_ITERS))
    print("=" * 78)

    for name, arc, ov in COMBOS:
        # reset CONFIG to defaults, then apply overrides
        for k, v in DEFAULTS.items():
            vr.CONFIG[k] = v
        for k, v in ov.items():
            vr.CONFIG[k] = v

        if arc not in geom_cache:
            vol_geom, proj_geom, geom_info = vr.build_geometry(
                phantom.shape, sl.DX_CM,
                det_row_count=det_row, det_col_count=det_col,
                det_spacing=det_sp, nviews=25, arc_deg=arc,
            )
            A, At = vr.make_projector(vol_geom, proj_geom)
            nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
                phantom.shape, A, At, vr.CONFIG["npower"])
            geom_cache[arc] = (A, At, geom_info,
                               nusino, nuxgrad, nuygrad, nuzgrad)

        A, At, geom_info, nusino, nuxgrad, nuygrad, nuzgrad = geom_cache[arc]
        nrays = geom_info["nrays"]

        if arc not in single_cache:
            t0 = time.time()
            _, is_, _, _, _ = vr.run_single_channel(
                phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
                nrays, snapshot_iters=SNAPSHOTS)
            single_cache[arc] = is_
            print(f"  [single @ arc {arc:.0f} done in {time.time()-t0:.0f}s]")
        single = single_cache[arc]

        R_hi, R_lo = vr.build_sinogram_filters(
            geom_info["det_col_count"], geom_info["det_spacing"],
            vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"])

        t0 = time.time()
        _, it_, _, _, _ = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            nrays, snapshot_iters=SNAPSHOTS)
        dt = time.time() - t0

        gaps = [gap(single, it_, it) for it in REPORT_ITERS]
        diverged = (not np.isfinite(it_[-1])) or it_[-1] > 10 * single[0]
        results.append((name, arc, gaps, diverged))

        cells = []
        for g in gaps:
            cells.append("  --  " if g is None else f"{g:+6.1f}")
        flag = "  DIVERGED" if diverged else ""
        print(f"{name:<22} {arc:>5.0f} | " + " | ".join(cells) +
              f"   ({dt:.0f}s){flag}")

    print("=" * 78)
    print("\nSummary (positive = two-channel beats single):")
    best = None
    for name, arc, gaps, diverged in results:
        if diverged:
            continue
        last = gaps[-1]
        if last is not None and (best is None or last > best[1]):
            best = (name, last)
    if best:
        print(f"  best combo: {best[0]} ({best[1]:+.1f}% at iter {REPORT_ITERS[-1]})")
    else:
        print("  no combo produced a positive gap")


if __name__ == "__main__":
    main()
