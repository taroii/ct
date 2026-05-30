"""12-hour overnight DBT breast sweep targeting the early-iteration
regime (5-25 iterations, with iter 50 as upper bound).

Goal: find a parameter / geometry combination where two-channel beats
single-channel in the clinical iteration regime. The previous overnight
sweep optimised iter 200+, which is past where any clinical
reconstructor stops.

Approach:
  - Anchor at (cutoffparm_lo=4, norm_inflate_3d=1.5, axis=2d) -- the
    best single-knob change found before.
  - Snapshots at [3, 5, 7, 10, 15, 20, 25, 30, 40, 50] -- fine grid in
    the clinical regime, sparser beyond.
  - itermax=50 (no need to go further; clinical reconstructors don't).
  - 288x288x64 voxels @ 0.075 cm; 240x240 detector @ 0.1 cm. Each
    two-channel trial is ~50-80 s.

Phases (~12 hr total budget on this GPU):
  P1: Fine single-knob sweeps around anchor                  (~120)
  P2: 2-knob grids over the most sensitive pairs             (~120)
  P3: 3-knob compounds (cutLo x eps_lo x sigma_lo)            (~40)
  P4: Geometry sweep with anchor + top compounds              (~70)
  P5: Adaptive: top-10 P2/P3 compounds re-tested at finer
       iter grid to confirm                                    (~30)
  P6: 432x432x96 high-res validation of top configs           (~20)
  P7: 576x576x128 ultra-high-res validation (fewer trials)    (~10)
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ct-2"))
os.chdir(ROOT)

from presentation_ct2_phantom_ladder import _install_grid_cache  # noqa: E402
import victre_reconstruction as vr  # noqa: E402
from phantom3d import image3D  # noqa: E402
import importlib  # noqa: E402

_install_grid_cache()

OUT_TXT = ROOT / "cache" / "_breast_dbt_early12h.txt"
CENTER = (3.0, 0.0, 0.0)
ITERMAX = 50
SNAPSHOT_ITERS = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
REPORT_ITERS   = SNAPSHOT_ITERS

ANCHOR_GEOM = {
    "shape": (288, 288, 64),
    "dx_cm": 0.075,
    "det":   (240, 240, 0.10),
    "nviews": 25,
    "arc_deg": 50.0,
}

ANCHOR_CP = {
    "sigma_lo_scale":  4.0,
    "norm_inflate_3d": 1.5,
    "eps_hi_ratio":    1.0,
    "eps_lo_ratio":    1.25,
    "cutoffparm":      4.0,
    "cutoffparm_lo":   4.0,
    "beta":            5.0,
}

# ============================================================================
# PHASE 1: FINE SINGLE-KNOB SWEEPS
# ============================================================================
def _trial(label, axis, overrides):
    return (label, axis, overrides)

P1_VARIATIONS = []
# cutoffparm_lo: very fine, 1.0 to 12.0
for c in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
          3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75,
          5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0, 12.0):
    P1_VARIATIONS.append(_trial(f"p1_cutLo_{c}", "2d", {"cutoffparm_lo": c}))
# eps_lo_ratio: fine
for r in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.85,
          1.0, 1.1, 1.25, 1.4, 1.5, 1.75, 2.0, 2.25, 2.5,
          3.0, 4.0, 5.0):
    P1_VARIATIONS.append(_trial(f"p1_epslo_{r}", "2d", {"eps_lo_ratio": r}))
# eps_hi_ratio
for r in (0.25, 0.4, 0.5, 0.65, 0.75, 0.85, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
    P1_VARIATIONS.append(_trial(f"p1_epshi_{r}", "2d", {"eps_hi_ratio": r}))
# sigma_lo_scale: from 0.5 to 10 (8 may diverge at inf=1.5)
for s in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 6.0, 7.0, 8.0):
    P1_VARIATIONS.append(_trial(f"p1_siglo_{s}", "2d", {"sigma_lo_scale": s}))
# norm_inflate_3d
for i in (1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.9, 2.0, 2.25, 2.5):
    P1_VARIATIONS.append(_trial(f"p1_inf_{i}", "2d", {"norm_inflate_3d": i}))
# cutoffparm (hi-pass)
for c in (1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 6.0, 8.0):
    P1_VARIATIONS.append(_trial(f"p1_cutHi_{c}", "2d", {"cutoffparm": c}))
# beta
for b in (0.5, 1.0, 2.0, 3.0, 7.0, 10.0, 15.0):
    P1_VARIATIONS.append(_trial(f"p1_beta_{b}", "2d", {"beta": b}))
# axis
P1_VARIATIONS.append(_trial("p1_axis_u", "u", {}))
P1_VARIATIONS.append(_trial("p1_axis_v", "v", {}))


# ============================================================================
# PHASE 2: 2-KNOB GRIDS
# ============================================================================
P2_VARIATIONS = []
# cutLo x eps_lo grid
for c in (2.5, 3.0, 3.5, 4.0, 5.0):
    for r in (0.25, 0.5, 0.75, 1.0, 1.5, 2.5):
        P2_VARIATIONS.append(_trial(
            f"p2_cutLo{c}_epslo{r}", "2d",
            {"cutoffparm_lo": c, "eps_lo_ratio": r}))
# cutLo x sigma_lo grid (paired with inflate=2.0 for stability at high sigma_lo)
for c in (3.0, 4.0, 5.0):
    for s, inf in [(2.0, 1.5), (3.0, 1.5), (5.0, 1.5),
                   (6.0, 2.0), (8.0, 2.0)]:
        P2_VARIATIONS.append(_trial(
            f"p2_cutLo{c}_siglo{s}_inf{inf}", "2d",
            {"cutoffparm_lo": c, "sigma_lo_scale": s, "norm_inflate_3d": inf}))
# cutLo x inflate
for c in (3.0, 4.0, 5.0):
    for i in (1.5, 1.6, 1.7, 1.8, 2.0):
        P2_VARIATIONS.append(_trial(
            f"p2_cutLo{c}_inf{i}", "2d",
            {"cutoffparm_lo": c, "norm_inflate_3d": i}))
# eps_lo x sigma_lo
for r in (0.5, 0.75, 1.25, 2.0):
    for s in (2.0, 4.0, 6.0):
        P2_VARIATIONS.append(_trial(
            f"p2_epslo{r}_siglo{s}", "2d",
            {"eps_lo_ratio": r, "sigma_lo_scale": s,
             "norm_inflate_3d": 2.0 if s > 4.0 else 1.5}))
# cutLo x eps_hi
for c in (3.0, 4.0):
    for h in (0.5, 0.75, 1.5, 2.0):
        P2_VARIATIONS.append(_trial(
            f"p2_cutLo{c}_epshi{h}", "2d",
            {"cutoffparm_lo": c, "eps_hi_ratio": h}))


# ============================================================================
# PHASE 3: 3-KNOB COMPOUNDS
# ============================================================================
P3_VARIATIONS = []
for c in (3.0, 4.0):
    for r in (0.5, 0.75, 1.0):
        for s, inf in [(2.0, 1.5), (4.0, 1.5), (6.0, 2.0), (8.0, 2.0)]:
            P3_VARIATIONS.append(_trial(
                f"p3_cutLo{c}_epslo{r}_siglo{s}", "2d",
                {"cutoffparm_lo": c, "eps_lo_ratio": r,
                 "sigma_lo_scale": s, "norm_inflate_3d": inf}))


# ============================================================================
# PHASE 4: GEOMETRY x CP
# ============================================================================
P4_GEOMETRIES = [
    # (arc_deg, nviews)
    ( 5.0, 25), (10.0, 25), (15.0, 25), (20.0, 25), (25.0, 25),
    (30.0, 25), (40.0, 25),
    (15.0,  9), (15.0, 15), (25.0,  9), (25.0, 15),
]
P4_TRIALS = [
    # Run each (arc, nviews) with anchor + a few promising compounds
    ("p4_anchor",         "2d", {}),
    ("p4_cutLo3_epslo0p5","2d", {"cutoffparm_lo": 3.0, "eps_lo_ratio": 0.5}),
    ("p4_cutLo3_epslo1",  "2d", {"cutoffparm_lo": 3.0, "eps_lo_ratio": 1.0}),
    ("p4_cutLo4_epslo0p5","2d", {"cutoffparm_lo": 4.0, "eps_lo_ratio": 0.5}),
]


# ============================================================================
# PHASE 5: 432^3 HIGH-RES VALIDATION (itermax=75 to span clinical regime
#         plus iter 50/75 trajectory)
# ============================================================================
P5_GEOM = {
    "shape": (432, 432, 96),
    "dx_cm": 0.05,
    "det":   (240, 240, 0.10),
    "nviews": 25,
    "arc_deg": 50.0,
}
P5_ITERMAX = 75
P5_SNAPS = [3, 5, 7, 10, 15, 20, 25, 30, 50, 75]
# Broad sweep at 432^3 -- this is the actual deck resolution, so explore
# more carefully here. Mix single-knob and compound variants.
P5_TRIALS = [
    ("p5_anchor",                       "2d", {}),
    # cutLo single-knob (key region from P1)
    ("p5_cutLo3",                       "2d", {"cutoffparm_lo": 3.0}),
    ("p5_cutLo5",                       "2d", {"cutoffparm_lo": 5.0}),
    # eps_lo single-knob (tighter LF)
    ("p5_epslo0p5",                     "2d", {"eps_lo_ratio": 0.5}),
    ("p5_epslo0p75",                    "2d", {"eps_lo_ratio": 0.75}),
    # eps_hi tight
    ("p5_epshi0p5",                     "2d", {"eps_hi_ratio": 0.5}),
    # sigma_lo
    ("p5_siglo2",                       "2d", {"sigma_lo_scale": 2.0}),
    ("p5_siglo6_inf2",                  "2d", {"sigma_lo_scale": 6.0,
                                                "norm_inflate_3d": 2.0}),
    # compounds: cutLo x eps_lo
    ("p5_cutLo3_epslo0p5",              "2d", {"cutoffparm_lo": 3.0,
                                                "eps_lo_ratio": 0.5}),
    ("p5_cutLo3_epslo0p75",             "2d", {"cutoffparm_lo": 3.0,
                                                "eps_lo_ratio": 0.75}),
    ("p5_cutLo3_epslo1",                "2d", {"cutoffparm_lo": 3.0,
                                                "eps_lo_ratio": 1.0}),
    ("p5_cutLo4_epslo0p5",              "2d", {"cutoffparm_lo": 4.0,
                                                "eps_lo_ratio": 0.5}),
    ("p5_cutLo4_epslo0p75",             "2d", {"cutoffparm_lo": 4.0,
                                                "eps_lo_ratio": 0.75}),
    # cutLo x sigma_lo
    ("p5_cutLo3_siglo6_inf2",           "2d", {"cutoffparm_lo": 3.0,
                                                "sigma_lo_scale": 6.0,
                                                "norm_inflate_3d": 2.0}),
    ("p5_cutLo4_siglo6_inf2",           "2d", {"cutoffparm_lo": 4.0,
                                                "sigma_lo_scale": 6.0,
                                                "norm_inflate_3d": 2.0}),
    # axis variants (one each)
    ("p5_axisU_cutLo3",                 "u",  {"cutoffparm_lo": 3.0}),
    ("p5_axisV_cutLo3",                 "v",  {"cutoffparm_lo": 3.0}),
    # everything tight
    ("p5_cutLo3_tightboth",             "2d", {"cutoffparm_lo": 3.0,
                                                "eps_lo_ratio": 0.5,
                                                "eps_hi_ratio": 0.5}),
    # aggressive early
    ("p5_aggressive",                   "2d", {"cutoffparm_lo": 3.0,
                                                "eps_lo_ratio": 0.5,
                                                "sigma_lo_scale": 6.0,
                                                "norm_inflate_3d": 2.0}),
]


# ============================================================================
# PHASE 6: NARROW-ARC HIGH-RES (arc=15, arc=20, arc=25) at 432^3
# ============================================================================
P6_ITERMAX = 75
P6_SNAPS = [3, 5, 7, 10, 15, 20, 25, 30, 50, 75]
P6_GEOMETRIES = [
    ("arc15",       15.0, 25),
    ("arc25",       25.0, 25),
    ("arc15_n9",    15.0,  9),
]
P6_TRIALS = [
    ("p6_anchor",          "2d", {}),
    ("p6_cutLo3",          "2d", {"cutoffparm_lo": 3.0}),
    ("p6_cutLo3_epslo0p5", "2d", {"cutoffparm_lo": 3.0, "eps_lo_ratio": 0.5}),
    ("p6_cutLo3_epslo0p75","2d", {"cutoffparm_lo": 3.0, "eps_lo_ratio": 0.75}),
    ("p6_cutLo4_epslo0p5", "2d", {"cutoffparm_lo": 4.0, "eps_lo_ratio": 0.5}),
]


# ============================================================================
# PHASE 7: 576^3 ULTRA-HIGH-RES (much slower, focused set)
# ============================================================================
P7_GEOM = {
    "shape": (576, 576, 128),
    "dx_cm": 0.0375,
    "det":   (240, 240, 0.10),
    "nviews": 25,
    "arc_deg": 50.0,
}
P7_ITERMAX = 50
P7_SNAPS = [3, 5, 7, 10, 15, 20, 25, 30, 50]
P7_TRIALS = [
    ("p7_anchor",            "2d", {}),
    ("p7_cutLo3",            "2d", {"cutoffparm_lo": 3.0}),
    ("p7_cutLo3_epslo0p5",   "2d", {"cutoffparm_lo": 3.0, "eps_lo_ratio": 0.5}),
    ("p7_cutLo3_epslo0p75",  "2d", {"cutoffparm_lo": 3.0, "eps_lo_ratio": 0.75}),
    ("p7_cutLo4_epslo0p5",   "2d", {"cutoffparm_lo": 4.0, "eps_lo_ratio": 0.5}),
]


# ============================================================================
# COMMON RUNNER
# ============================================================================
def reduction(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


def fmt_iters(arr, iters):
    return " ".join(f"{arr[it-1]:.5f}" for it in iters)


def fmt_reds(s, t, iters):
    return " ".join(
        "   --  " if reduction(s, t, it) is None
        else f"{reduction(s, t, it):+6.1f}"
        for it in iters
    )


def log(line):
    print(line)
    with open(OUT_TXT, "a") as f:
        f.write(line + "\n")


def build_phantom(shape_xyz, dx_cm):
    mod = importlib.import_module("breast_phantom_demo")
    builder = getattr(mod, "build_breast_phantom")
    phantom = builder(breast_xc=0., breast_yc=0., breast_zc=0.)
    NX, NY, NZ = shape_xyz
    xc, yc, zc = CENTER
    xlen, ylen, zlen = NX * dx_cm, NY * dx_cm, NZ * dx_cm
    img = image3D(shape=(NX, NY, NZ),
                  xlen=xlen, ylen=ylen, zlen=zlen,
                  x0=xc - xlen/2., y0=yc - ylen/2., z0=zc - zlen/2.)
    phantom.embed_in(img)
    vol = img.mat
    vol = np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32)
    return vol


def reset_cp(cfg=ANCHOR_CP):
    for k, v in cfg.items():
        vr.CONFIG[k] = v


def setup_geom(geom):
    phantom = build_phantom(geom["shape"], geom["dx_cm"])
    det_row, det_col, det_sp = geom["det"]
    vol_geom, proj_geom, gi = vr.build_dbt_geometry(
        phantom.shape, geom["dx_cm"],
        det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
        nviews=geom["nviews"], arc_deg=geom["arc_deg"], sod=65.0, odd=5.0,
    )
    A, At = vr.make_projector(vol_geom, proj_geom)
    op_norms = vr.operator_norms(phantom.shape, A, At, vr.CONFIG["npower"])
    return phantom, A, At, gi, op_norms


def trial_row(label, overrides, axis, single_arr, phantom, A, At, op_norms,
              nrays, gi, snapshot_iters, report_iters):
    reset_cp()
    for k, v in overrides.items():
        vr.CONFIG[k] = v
    R_hi, R_lo = vr.build_sinogram_filters(
        gi["det_col_count"], gi["det_spacing"],
        vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
        axis=axis, det_row_count=gi["det_row_count"],
    )
    t0 = time.time()
    nusino, nuxgrad, nuygrad, nuzgrad = op_norms
    try:
        rt, itwo, dt_, tt_, snaps_t = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            nrays, snapshot_iters=snapshot_iters,
        )
    except Exception as exc:
        log(f"{label:<36} | ERRORED: {exc}")
        return None
    itwo_np = np.asarray(itwo)
    dt = time.time() - t0
    diverged = (not np.isfinite(itwo_np[-1])) or itwo_np[-1] > 1.0
    flag = "  DIVERGED" if diverged else ""
    log(f"{label:<36} | s {fmt_iters(single_arr, report_iters)} | "
        f"t {fmt_iters(itwo_np, report_iters)} | "
        f"r {fmt_reds(single_arr, itwo_np, report_iters)} ({dt:.0f}s){flag}")
    return itwo_np


def phase_block(phase_name, geom, trials, single_arr=None,
                itermax=ITERMAX, snapshot_iters=SNAPSHOT_ITERS,
                report_iters=REPORT_ITERS):
    log(f"\n##### {phase_name} | geom={geom} #####")
    log(f"# itermax={itermax}, snapshots={snapshot_iters}")
    reset_cp()
    vr.CONFIG["itermax"] = itermax
    phantom, A, At, gi, op_norms = setup_geom(geom)
    nrays = gi["nrays"]
    if single_arr is None:
        nusino, nuxgrad, nuygrad, nuzgrad = op_norms
        t0 = time.time()
        rs, isng, ds_, ts_, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            nrays, snapshot_iters=snapshot_iters,
        )
        single_arr = np.asarray(isng)
        log(f"single_baseline                      | s {fmt_iters(single_arr, report_iters)}   "
            f"({time.time()-t0:.0f}s)")
    for label, axis, overrides in trials:
        trial_row(label, overrides, axis, single_arr, phantom, A, At,
                  op_norms, nrays, gi, snapshot_iters, report_iters)
    return single_arr


def main():
    if OUT_TXT.exists():
        OUT_TXT.unlink()
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    log(f"DBT breast 12-hour EARLY-ITER sweep started "
        f"{datetime.now().isoformat()}")
    log(f"anchor geom: {ANCHOR_GEOM}")
    log(f"anchor CP: {ANCHOR_CP}")
    log(f"itermax: {ITERMAX}, snapshots: {SNAPSHOT_ITERS}")
    log(f"trial counts: P1={len(P1_VARIATIONS)} P2={len(P2_VARIATIONS)} "
        f"P3={len(P3_VARIATIONS)} P4={len(P4_GEOMETRIES) * len(P4_TRIALS)} "
        f"P5={len(P5_TRIALS)} P6={len(P6_TRIALS)} P7={len(P7_TRIALS)}\n")

    # P1: single-knob variations at anchor geom
    single_anchor = phase_block("PHASE 1 (single-knob variations)",
                                ANCHOR_GEOM, P1_VARIATIONS)
    # P2: 2-knob grids (reuse single from P1)
    phase_block("PHASE 2 (2-knob grids)", ANCHOR_GEOM, P2_VARIATIONS,
                single_arr=single_anchor)
    # P3: 3-knob compounds
    phase_block("PHASE 3 (3-knob compounds)", ANCHOR_GEOM, P3_VARIATIONS,
                single_arr=single_anchor)
    # P4: geometry sweep
    for arc, nv in P4_GEOMETRIES:
        geom = {**ANCHOR_GEOM, "arc_deg": arc, "nviews": nv}
        phase_block(f"PHASE 4 (arc={arc}, n={nv})", geom, P4_TRIALS)
    # P5: 432^3 high-res validation at arc=50
    phase_block("PHASE 5 (432^3, arc=50, broad CP scan)", P5_GEOM,
                P5_TRIALS, itermax=P5_ITERMAX,
                snapshot_iters=P5_SNAPS, report_iters=P5_SNAPS)
    # P6: 432^3 across narrow-arc geometries
    for tag, arc, nv in P6_GEOMETRIES:
        geom = {**P5_GEOM, "arc_deg": arc, "nviews": nv}
        phase_block(f"PHASE 6 ({tag}: 432^3, arc={arc}, n={nv})", geom,
                    P6_TRIALS, itermax=P6_ITERMAX,
                    snapshot_iters=P6_SNAPS, report_iters=P6_SNAPS)
    # P7: 576^3 ultra-high-res
    phase_block("PHASE 7 (576^3 ultra-high-res)", P7_GEOM, P7_TRIALS,
                itermax=P7_ITERMAX,
                snapshot_iters=P7_SNAPS, report_iters=P7_SNAPS)

    log(f"\nDone {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
