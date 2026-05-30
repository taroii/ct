"""CP-parameter sweep on compressed + lesion-inserted VICTRE dense
phantom at the REAL Hologic Selenia narrow-arc DBT geometry:
    - SOD = 65 cm, ODD = 1 cm  (SDD = 66 cm)
    - Detector 375 x 187 @ 0.68 mm  (25.5 x 12.75 cm, 8x binned native)
    - 9 views over 15 deg arc  (Hologic Selenia "low-dose" mode)
    - itermax = 100

This is significantly more LAR-ill-conditioned than the previous
50 deg sweep -- single-channel should struggle for many more iterations,
which gives two-channel a long window to stay ahead and produces a
larger absolute gap.

Trials all use axis="2d". Anchored on the T7 winner from the previous
sweep (cutoffparm_lo=1.5, norm_inflate_3d=1.5), with neighborhood
variants.
"""
import gzip
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import victre_reconstruction as vr  # noqa: E402

RAW_GZ   = ROOT / "data" / "compressed_legion_victre" / "dense_pcl_-321964974_crop.raw.gz"
NX, NY, NZ = 810, 1920, 745
NATIVE_DX_CM = 0.005
DOWNSAMPLE = 8

CACHE_DIR  = ROOT / "cache"
SUMMARY_TXT = CACHE_DIR / "victre_pcl_narrow_sweep.txt"

ITERMAX = 100
SNAPSHOT_ITERS = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100]
REPORT_ITERS   = SNAPSHOT_ITERS

# Hologic Selenia narrow-arc DBT geometry
DET_ROW_COUNT  = 375
DET_COL_COUNT  = 187
DET_SPACING_CM = 0.068
NVIEWS  = 9
ARC_DEG = 15.0
SOD_CM = 65.0
ODD_CM = 1.0

MU_TABLE = {
    0: 0.000, 1: 0.275, 2: 0.375, 29: 0.368, 33: 0.368,
    40: 0.368, 50: 0.000, 88: 0.368, 95: 0.368,
    125: 0.368, 150: 0.368, 200: 0.450, 225: 0.368, 250: 4.310,
}

# Broad sweep. Hologic narrow-arc (15 deg, 9 views) is much more LAR-ill-
# conditioned than the prior 50 deg setup, so we should NOT assume the
# T7 (cutLo=1.5, inflate=1.5) winner carries over. Cover the whole
# parameter space, then compound around whatever winners emerge.
TRIALS = [
    # --- baselines ---
    ("N00_default",                "2d", {}),
    ("N01_T7",                     "2d", {"cutoffparm_lo": 1.5,
                                            "norm_inflate_3d": 1.5}),

    # --- cutoffparm_lo single-knob (wide) ---
    ("N10_cutLo0p5",               "2d", {"cutoffparm_lo": 0.5}),
    ("N11_cutLo1",                 "2d", {"cutoffparm_lo": 1.0}),
    ("N12_cutLo1p5",               "2d", {"cutoffparm_lo": 1.5}),
    ("N13_cutLo2",                 "2d", {"cutoffparm_lo": 2.0}),
    ("N14_cutLo3",                 "2d", {"cutoffparm_lo": 3.0}),
    ("N15_cutLo6",                 "2d", {"cutoffparm_lo": 6.0}),

    # --- norm_inflate_3d single-knob (wide) ---
    ("N20_inf1p3",                 "2d", {"norm_inflate_3d": 1.3}),
    ("N21_inf1p5",                 "2d", {"norm_inflate_3d": 1.5}),
    ("N22_inf1p7",                 "2d", {"norm_inflate_3d": 1.7}),
    ("N23_inf2p5",                 "2d", {"norm_inflate_3d": 2.5}),

    # --- sigma_lo_scale single-knob ---
    ("N30_siglo1",                 "2d", {"sigma_lo_scale": 1.0}),
    ("N31_siglo2",                 "2d", {"sigma_lo_scale": 2.0}),
    ("N32_siglo6_inf2",            "2d", {"sigma_lo_scale": 6.0,
                                            "norm_inflate_3d": 2.0}),
    ("N33_siglo8_inf2p5",          "2d", {"sigma_lo_scale": 8.0,
                                            "norm_inflate_3d": 2.5}),

    # --- eps_lo single-knob ---
    ("N40_epslo0p25",              "2d", {"eps_lo_ratio": 0.25}),
    ("N41_epslo0p5",               "2d", {"eps_lo_ratio": 0.5}),
    ("N42_epslo2p5",               "2d", {"eps_lo_ratio": 2.5}),
    ("N43_epslo5",                 "2d", {"eps_lo_ratio": 5.0}),
    ("N44_epslo10",                "2d", {"eps_lo_ratio": 10.0}),

    # --- eps_hi single-knob (tightening HF too) ---
    ("N50_epshi0p5",               "2d", {"eps_hi_ratio": 0.5}),
    ("N51_epshi2",                 "2d", {"eps_hi_ratio": 2.0}),

    # --- cutoffparm (HF cutoff) single-knob ---
    ("N60_cutHi2",                 "2d", {"cutoffparm": 2.0}),
    ("N61_cutHi8",                 "2d", {"cutoffparm": 8.0}),

    # --- beta (L1 weight) single-knob ---
    ("N70_beta1",                  "2d", {"beta": 1.0}),
    ("N71_beta10",                 "2d", {"beta": 10.0}),

    # --- filter axis variants (1D u, 1D v -- previously underexplored
    #     in 9-view setup since both 1D axes might behave very
    #     differently than at 25 views) ---
    ("N80_axisU",                  "u",  {}),
    ("N81_axisV",                  "v",  {}),
    ("N82_axisU_cutLo1p5_inf1p5",  "u",  {"cutoffparm_lo": 1.5,
                                            "norm_inflate_3d": 1.5}),
    ("N83_axisV_cutLo1p5_inf1p5",  "v",  {"cutoffparm_lo": 1.5,
                                            "norm_inflate_3d": 1.5}),

    # --- compounds: aggressive LF push + various inflates ---
    ("N90_cutLo1_siglo6_inf2",     "2d", {"cutoffparm_lo": 1.0,
                                            "sigma_lo_scale": 6.0,
                                            "norm_inflate_3d": 2.0}),
    ("N91_cutLo0p5_siglo6_inf2",   "2d", {"cutoffparm_lo": 0.5,
                                            "sigma_lo_scale": 6.0,
                                            "norm_inflate_3d": 2.0}),
    ("N92_cutLo1_epslo5",          "2d", {"cutoffparm_lo": 1.0,
                                            "eps_lo_ratio": 5.0}),
    ("N93_cutLo1_epslo0p5",        "2d", {"cutoffparm_lo": 1.0,
                                            "eps_lo_ratio": 0.5}),
    ("N94_aggressive",             "2d", {"cutoffparm_lo": 1.0,
                                            "sigma_lo_scale": 6.0,
                                            "eps_lo_ratio": 0.5,
                                            "norm_inflate_3d": 2.0}),
    ("N95_loose_loose",            "2d", {"eps_hi_ratio": 2.0,
                                            "eps_lo_ratio": 10.0,
                                            "sigma_lo_scale": 4.0,
                                            "norm_inflate_3d": 2.0}),
    ("N96_tight_tight",            "2d", {"eps_hi_ratio": 0.5,
                                            "eps_lo_ratio": 0.25,
                                            "cutoffparm_lo": 1.5}),
    ("N97_cutLo3_siglo2",          "2d", {"cutoffparm_lo": 3.0,
                                            "sigma_lo_scale": 2.0}),
    ("N98_cutLo6_inf1p5",          "2d", {"cutoffparm_lo": 6.0,
                                            "norm_inflate_3d": 1.5}),
]

DEFAULTS = {
    "sigma_lo_scale":  vr.CONFIG["sigma_lo_scale"],
    "norm_inflate_3d": vr.CONFIG["norm_inflate_3d"],
    "eps_hi_ratio":    vr.CONFIG["eps_hi_ratio"],
    "eps_lo_ratio":    vr.CONFIG["eps_lo_ratio"],
    "cutoffparm":      vr.CONFIG["cutoffparm"],
    "cutoffparm_lo":   vr.CONFIG["cutoffparm_lo"],
    "beta":            vr.CONFIG["beta"],
}


def reset_cp():
    for k, v in DEFAULTS.items():
        vr.CONFIG[k] = v


def reduction(s, t, it):
    if it - 1 >= len(s) or it - 1 >= len(t):
        return None
    a, b = s[it - 1], t[it - 1]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return (a - b) / a * 100.0


def fmt_iters(arr, iters):
    return " ".join(f"{arr[it-1]:.4f}" for it in iters)


def fmt_reds(s, t, iters):
    return " ".join(
        "  --  " if reduction(s, t, it) is None
        else f"{reduction(s, t, it):+6.2f}"
        for it in iters
    )


def log(line):
    print(line)
    with open(SUMMARY_TXT, "a") as f:
        f.write(line + "\n")


def load_and_downsample():
    print(f"Loading {RAW_GZ.name}")
    t0 = time.time()
    with gzip.open(RAW_GZ, "rb") as f:
        buf = f.read()
    print(f"  read {len(buf)/1e9:.2f} GB in {time.time()-t0:.1f}s")
    vol_lbl = np.frombuffer(buf, np.uint8).reshape(NZ, NY, NX)
    mu_lut = np.zeros(256, dtype=np.float32)
    for k, v in MU_TABLE.items():
        mu_lut[k] = v
    d = DOWNSAMPLE
    nz_use = (NZ // d) * d; ny_use = (NY // d) * d; nx_use = (NX // d) * d
    nz_d, ny_d, nx_d = nz_use // d, ny_use // d, nx_use // d
    out = np.zeros((nz_d, ny_d, nx_d), dtype=np.float32)
    t1 = time.time()
    chunk = 8 * d
    for z0 in range(0, nz_use, chunk):
        z1 = min(z0 + chunk, nz_use)
        block = mu_lut[vol_lbl[z0:z1, :ny_use, :nx_use]]
        binned = block.reshape(
            (z1 - z0) // d, d, ny_d, d, nx_d, d
        ).mean(axis=(1, 3, 5))
        out[z0 // d : z1 // d] = binned
    print(f"  downsampled in {time.time()-t1:.1f}s   shape {out.shape}")
    return out, NATIVE_DX_CM * d


def main():
    if SUMMARY_TXT.exists():
        SUMMARY_TXT.unlink()
    log("VICTRE compressed+lesion phantom -- narrow-arc Hologic Selenia DBT sweep")
    log(f"  geom: SOD={SOD_CM} ODD={ODD_CM} det {DET_COL_COUNT}x{DET_ROW_COUNT}"
        f" @ {DET_SPACING_CM*10:.2f} mm, {NVIEWS} views over {ARC_DEG} deg arc")
    log(f"  itermax {ITERMAX}, snapshots {SNAPSHOT_ITERS}\n")

    phantom, dx_cm = load_and_downsample()

    vol_geom, proj_geom, gi = vr.build_dbt_geometry(
        phantom.shape, dx_cm,
        det_row_count=DET_ROW_COUNT, det_col_count=DET_COL_COUNT,
        det_spacing=DET_SPACING_CM,
        nviews=NVIEWS, arc_deg=ARC_DEG,
        sod=SOD_CM, odd=ODD_CM,
    )
    A, At = vr.make_projector(vol_geom, proj_geom)
    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"]
    )

    saved_itermax = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = ITERMAX

    try:
        print("\n--- single-channel ---")
        reset_cp()
        t0 = time.time()
        rs, isng, ds_, ts_, snaps_s = vr.run_single_channel(
            phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
            gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
        )
        isng_np = np.asarray(isng)
        log(f"\nsingle_baseline                    | s {fmt_iters(isng_np, REPORT_ITERS)}   "
            f"({time.time()-t0:.0f}s)")

        for label, axis, overrides in TRIALS:
            reset_cp()
            for k, v in overrides.items():
                vr.CONFIG[k] = v
            R_hi, R_lo = vr.build_sinogram_filters(
                gi["det_col_count"], gi["det_spacing"],
                vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
                axis=axis, det_row_count=gi["det_row_count"],
            )
            t0 = time.time()
            try:
                rt, itwo, dt_, tt_, snaps_t = vr.run_two_channel(
                    phantom, A, At, R_hi, R_lo,
                    nusino, nuxgrad, nuygrad, nuzgrad,
                    gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
                )
            except Exception as exc:
                log(f"{label:<28} | ERRORED {exc}")
                continue
            itwo_np = np.asarray(itwo)
            dt = time.time() - t0
            log(f"{label:<28} | s {fmt_iters(isng_np, REPORT_ITERS)} | "
                f"t {fmt_iters(itwo_np, REPORT_ITERS)} | "
                f"r {fmt_reds(isng_np, itwo_np, REPORT_ITERS)} ({dt:.0f}s)")
            out_pkl = CACHE_DIR / f"victre_pcl_narrow_{label}.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump({
                    "phantom":          phantom,
                    "recon_single":     rs,
                    "recon_two":        rt,
                    "ierrs_single":     isng_np.tolist(),
                    "ierrs_two":        itwo_np.tolist(),
                    "snapshots_single": snaps_s,
                    "snapshots_two":    snaps_t,
                    "dx_cm":            dx_cm,
                    "geometry":         gi,
                    "config_label":     label,
                    "cp_overrides":     overrides,
                    "filter_axis":      axis,
                }, f)
    finally:
        vr.CONFIG["itermax"] = saved_itermax
        reset_cp()

    log("\nDone.")


if __name__ == "__main__":
    main()
