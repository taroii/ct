"""Comprehensive overnight sweep: find any DBT-geometry config where
two-channel beats single-channel for the analytic breast.

Phases:
  P1: Anchor at 288x288x64 voxels, det 240x240 (0.1 cm), nviews=25, arc=50,
      2D Hanning filter, inflate=1.5 (best from previous sweep). Vary ONE
      CP knob at a time (eps_lo_ratio, eps_hi_ratio, sigma_lo_scale,
      norm_inflate_3d, cutoffparm, cutoffparm_lo, beta, filter axis).
      Single-channel computed once and shared across all P1 trials.
  P2: Hand-picked compound parameter combinations.
  P3: Geometry sweep (arc x nviews) at default + inf1p5 + best-P1-guess.
  P4: Higher-resolution validation (432x432x96, ~18M voxels) at the
      most promising config families.

Output: cache/_breast_dbt_overnight.txt is appended after every trial so
nothing is lost on a kill.
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

OUT_TXT = ROOT / "cache" / "_breast_dbt_overnight.txt"
CENTER = (3.0, 0.0, 0.0)
SNAPSHOT_ITERS = [50, 100, 200]
REPORT_ITERS   = [50, 100, 200]
ITERMAX = 200

# Anchor (P1, P2)
ANCHOR_GEOM = {
    "shape": (288, 288, 64),
    "dx_cm": 0.075,
    "det":   (240, 240, 0.10),
    "nviews": 25,
    "arc_deg": 50.0,
}

# CP-knob defaults (the anchor variant: inf1p5)
ANCHOR_CP = {
    "sigma_lo_scale":  4.0,
    "norm_inflate_3d": 1.5,
    "eps_hi_ratio":    1.0,
    "eps_lo_ratio":    1.25,
    "cutoffparm":      4.0,
    "cutoffparm_lo":   8.0,
    "beta":            5.0,
}

# ----- Phase 1: vary one CP knob at a time, axis=2d --------------------
P1_VARIATIONS = [
    # eps_lo_ratio sweep
    *[(f"p1_epslo_{r}", "2d", {"eps_lo_ratio": r}) for r in
      (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)],
    # eps_hi_ratio sweep
    *[(f"p1_epshi_{r}", "2d", {"eps_hi_ratio": r}) for r in
      (0.5, 1.5, 2.0, 3.0)],
    # sigma_lo_scale sweep
    *[(f"p1_siglo_{s}", "2d", {"sigma_lo_scale": float(s)}) for s in
      (1, 2, 8, 16)],
    # norm_inflate_3d sweep
    *[(f"p1_inf_{i}", "2d", {"norm_inflate_3d": i}) for i in
      (1.6, 1.8, 2.0, 2.5, 3.0)],
    # cutoffparm sweep (hi-pass cutoff)
    *[(f"p1_cutHi_{c}", "2d", {"cutoffparm": float(c)}) for c in
      (1, 2, 6, 8, 16)],
    # cutoffparm_lo sweep
    *[(f"p1_cutLo_{c}", "2d", {"cutoffparm_lo": float(c)}) for c in
      (2, 4, 6, 12, 16, 32)],
    # beta sweep (L1 cap)
    *[(f"p1_beta_{b}", "2d", {"beta": float(b)}) for b in
      (0.5, 1, 2, 10, 20)],
    # filter axis variations
    ("p1_axis_u", "u", {}),
    ("p1_axis_v", "v", {}),
]

# ----- Phase 2: hand-picked compound combos ---------------------------
P2_COMBOS = [
    ("p2_loose_both",   "2d", {"eps_hi_ratio": 2.0, "eps_lo_ratio": 5.0}),
    ("p2_tight_both",   "2d", {"eps_hi_ratio": 0.5, "eps_lo_ratio": 0.5}),
    ("p2_loose_lf",     "2d", {"eps_lo_ratio": 5.0, "sigma_lo_scale": 2.0}),
    ("p2_tight_lf_aggr","2d", {"eps_lo_ratio": 0.5, "sigma_lo_scale": 8.0}),
    ("p2_wide_lf",      "2d", {"cutoffparm_lo": 16.0, "eps_lo_ratio": 5.0}),
    ("p2_narrow_lf",    "2d", {"cutoffparm_lo": 4.0, "eps_lo_ratio": 2.5}),
    ("p2_u_loose",      "u",  {"eps_lo_ratio": 5.0, "sigma_lo_scale": 2.0}),
    ("p2_v_loose",      "v",  {"eps_lo_ratio": 5.0, "sigma_lo_scale": 2.0}),
    ("p2_low_beta",     "2d", {"beta": 1.0, "eps_lo_ratio": 5.0}),
    ("p2_low_beta_tight","2d", {"beta": 1.0, "eps_lo_ratio": 0.5}),
    ("p2_super_loose",  "2d", {"eps_hi_ratio": 2.0, "eps_lo_ratio": 10.0,
                                "sigma_lo_scale": 8.0, "norm_inflate_3d": 2.0}),
    ("p2_extreme_loose","2d", {"eps_hi_ratio": 3.0, "eps_lo_ratio": 20.0,
                                "sigma_lo_scale": 1.0, "norm_inflate_3d": 2.5}),
    ("p2_no_lf_push",   "2d", {"sigma_lo_scale": 1.0, "eps_lo_ratio": 1.0}),
    ("p2_zsplit_loose", "2d", {"cutoffparm": 8.0, "cutoffparm_lo": 16.0,
                                "eps_lo_ratio": 5.0}),
]

# ----- Phase 3: geometry sweep ----------------------------------------
# (arc_deg, nviews) outer; for each runs single + the trials below.
P3_GEOMETRIES = [
    # arc, nviews
    (15.0,  9), (15.0, 25),
    (25.0,  9), (25.0, 25),
    (40.0, 25),
    (60.0, 25),
    (90.0, 25),
]
P3_TRIALS = [
    ("p3_default",  "2d", {}),
    ("p3_loose_lf", "2d", {"eps_lo_ratio": 5.0, "sigma_lo_scale": 2.0}),
]

# ----- Phase 4: ultra-high-res validation --------------------------
P4_GEOM = {
    "shape": (432, 432, 96),
    "dx_cm": 0.05,
    "det":   (240, 240, 0.10),
    "nviews": 25,
    "arc_deg": 50.0,
}
P4_TRIALS = [
    ("p4_default",  "2d", {}),
    ("p4_loose_lf", "2d", {"eps_lo_ratio": 5.0, "sigma_lo_scale": 2.0}),
    ("p4_loose_hi", "2d", {"eps_hi_ratio": 2.0, "eps_lo_ratio": 5.0}),
]


def reduction(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


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


def log(line):
    print(line)
    with open(OUT_TXT, "a") as f:
        f.write(line + "\n")


def fmt_iters(arr, iters):
    return " ".join(f"{arr[it-1]:.5f}" for it in iters)


def fmt_reds(s, t, iters):
    return " ".join(
        "   --  " if reduction(s, t, it) is None
        else f"{reduction(s, t, it):+6.1f}"
        for it in iters
    )


def setup_geom(geom):
    """Build phantom, projector, sino filters base. Returns (phantom, A, At, geom_info, op_norms)."""
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


def run_single(phantom, A, At, op_norms, nrays):
    t0 = time.time()
    nusino, nuxgrad, nuygrad, nuzgrad = op_norms
    rs, isng, ds_, ts_, snaps_s = vr.run_single_channel(
        phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
        nrays, snapshot_iters=SNAPSHOT_ITERS,
    )
    return np.asarray(isng), time.time() - t0


def run_two(phantom, A, At, op_norms, nrays, R_hi, R_lo):
    t0 = time.time()
    nusino, nuxgrad, nuygrad, nuzgrad = op_norms
    try:
        rt, itwo, dt_, tt_, snaps_t = vr.run_two_channel(
            phantom, A, At, R_hi, R_lo,
            nusino, nuxgrad, nuygrad, nuzgrad,
            nrays, snapshot_iters=SNAPSHOT_ITERS,
        )
        return np.asarray(itwo), time.time() - t0, None
    except Exception as exc:
        return None, time.time() - t0, str(exc)


def apply_trial(overrides):
    reset_cp()
    for k, v in overrides.items():
        vr.CONFIG[k] = v


def trial_row(label, overrides, axis, single_arr, phantom, A, At, op_norms,
              nrays, gi):
    """Run a single two-channel trial and write a summary row."""
    apply_trial(overrides)
    R_hi, R_lo = vr.build_sinogram_filters(
        gi["det_col_count"], gi["det_spacing"],
        vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
        axis=axis, det_row_count=gi["det_row_count"],
    )
    two, dt, err = run_two(phantom, A, At, op_norms, nrays, R_hi, R_lo)
    if err is not None:
        log(f"{label:<28} | ERRORED: {err}")
        return None
    diverged = (not np.isfinite(two[-1])) or two[-1] > 1.0
    flag = "  DIVERGED" if diverged else ""
    log(f"{label:<28} | s {fmt_iters(single_arr, REPORT_ITERS)} | "
        f"t {fmt_iters(two, REPORT_ITERS)} | "
        f"r {fmt_reds(single_arr, two, REPORT_ITERS)} ({dt:.0f}s){flag}")
    return two


def phase_block(phase_name, geom, trials, single_arr=None):
    """Common phase runner: rebuild geom if needed, compute single, run trials."""
    log(f"\n##### {phase_name} | geom={geom} #####")
    log(f"# anchor CP: {ANCHOR_CP}")
    reset_cp()
    vr.CONFIG["itermax"] = ITERMAX
    phantom, A, At, gi, op_norms = setup_geom(geom)
    nrays = gi["nrays"]
    if single_arr is None:
        t0 = time.time()
        single_arr, dt = run_single(phantom, A, At, op_norms, nrays)
        log(f"single_baseline             | s {fmt_iters(single_arr, REPORT_ITERS)} "
            f"  ({dt:.0f}s)")
    for label, axis, overrides in trials:
        trial_row(label, overrides, axis, single_arr, phantom, A, At, op_norms,
                  nrays, gi)
    return single_arr


def main():
    if OUT_TXT.exists():
        OUT_TXT.unlink()
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    log(f"DBT breast overnight sweep, started {datetime.now().isoformat()}")
    log(f"anchor geom: {ANCHOR_GEOM}, anchor CP: {ANCHOR_CP}")
    log(f"itermax: {ITERMAX}, snapshots: {SNAPSHOT_ITERS}\n")

    # -- Phase 1: anchor geometry, vary one CP knob at a time ----------
    single_anchor = phase_block("PHASE 1 (CP knobs at anchor geom)",
                                ANCHOR_GEOM, P1_VARIATIONS)

    # -- Phase 2: hand-picked compound CP combos -----------------------
    phase_block("PHASE 2 (compound CP)", ANCHOR_GEOM, P2_COMBOS,
                single_arr=single_anchor)

    # -- Phase 3: geometry sweep (arc x nviews) ------------------------
    for arc, nv in P3_GEOMETRIES:
        geom = {**ANCHOR_GEOM, "arc_deg": arc, "nviews": nv}
        phase_block(f"PHASE 3 (arc={arc}, n={nv})", geom, P3_TRIALS)

    # -- Phase 4: ultra-high-res ---------------------------------------
    phase_block("PHASE 4 (432x432x96)", P4_GEOM, P4_TRIALS)

    log(f"\nDone {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
