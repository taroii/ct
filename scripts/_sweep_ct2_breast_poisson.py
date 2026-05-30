"""Structured-noise sweep on the ct-2 analytic breast: Poisson photon
noise plus a low-frequency-correlated Gaussian component mimicking
residual scatter after standard scatter correction.

Pure Poisson on its own produces pixel-independent noise that looks
gaussian and doesn't reproduce the LF "wobble" the 2D paper sees. Real
DBT residuals are dominated by post-correction scatter, which is
spatially smooth (LF-correlated). Adding a Gaussian-smoothed random
field to the line-integral sinogram captures that character without
needing to run MCGPU.

Geometry: 288^3 x 64 voxels @ 0.075 cm, DBT arc=15 deg, 25 views.
CP: cutoffparm_lo=1.5 (from the acceleration sweep), others at the
2D-paper defaults (inflate=2, eps_lo=1.25, sigma_lo=4, beta=5).

Scenarios (4):
    noiseless                 - clean reference
    LF_only                   - LF-correlated noise alone (no Poisson)
    Poisson_only              - Poisson at I0=1e4 (no LF)
    Poisson_plus_LF           - Both stacked

LF noise model: per-view, Gaussian-smooth a white-noise field by sigma
(in detector pixels), then scale so its std equals lf_amplitude_pct of
the per-view mean line integral.

Outputs:
    cache/ct2_breast_poisson_<label>.pkl     per-(scenario, method)
    cache/ct2_breast_poisson_sweep.txt       summary
"""
import os
import pickle
import sys
import time
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

CACHE_DIR = ROOT / "cache"
SUMMARY = CACHE_DIR / "ct2_breast_poisson_sweep.txt"

# Reduced config for fast dose-level exploration (288^3 instead of 432^3
# and itermax 20). Once a promising dose region is found we can re-run at
# 432^3 for the deck-quality figure.
SHAPE = (288, 288, 64)
DX_CM = 0.075
CENTER = (3.0, 0.0, 0.0)
DET = (240, 240, 0.10)
NVIEWS = 25
ARC_DEG = 15.0
SOD_CM = 65.0
ODD_CM = 5.0

ITERMAX = 20
SNAPSHOT_ITERS = [3, 5, 7, 10, 15, 20]
REPORT_ITERS = SNAPSHOT_ITERS

CP_OVERRIDES = {"cutoffparm_lo": 1.5}   # winner from acceleration sweep

# (label, I0, lf_pct)  -- I0 = incident photons per pixel (None = no
# Poisson); lf_pct = LF noise std as % of per-view mean line integral
# (None = no LF noise)
SCENARIOS = [
    ("noiseless",        None,    None),
    ("LF_only",          None,    3.0),    # 3% scatter-like LF noise alone
    ("Poisson_only",     1.0e4,   None),   # standard Poisson alone
    ("Poisson_plus_LF",  1.0e4,   3.0),    # both stacked
]

POISSON_SEED = 12345
LF_SEED = 54321
LF_SIGMA_PIXELS = 8.0  # Gaussian smoothing sigma (about 5.4 mm at 0.68 mm
                        # pitch -- scatter-correlation-length-ish)


def reset_cp():
    # Restore CP defaults captured at import time below
    for k, v in DEFAULTS.items():
        vr.CONFIG[k] = v


DEFAULTS = {
    "sigma_lo_scale":  vr.CONFIG["sigma_lo_scale"],
    "norm_inflate_3d": vr.CONFIG["norm_inflate_3d"],
    "eps_hi_ratio":    vr.CONFIG["eps_hi_ratio"],
    "eps_lo_ratio":    vr.CONFIG["eps_lo_ratio"],
    "cutoffparm":      vr.CONFIG["cutoffparm"],
    "cutoffparm_lo":   vr.CONFIG["cutoffparm_lo"],
    "beta":            vr.CONFIG["beta"],
}


def reduction(s, t, it):
    a = s[it - 1]; b = t[it - 1]
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
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def build_phantom():
    mod = importlib.import_module("breast_phantom_demo")
    builder = getattr(mod, "build_breast_phantom")
    phantom = builder(breast_xc=0., breast_yc=0., breast_zc=0.)
    NX, NY, NZ = SHAPE
    xc, yc, zc = CENTER
    xlen, ylen, zlen = NX * DX_CM, NY * DX_CM, NZ * DX_CM
    img = image3D(shape=(NX, NY, NZ),
                  xlen=xlen, ylen=ylen, zlen=zlen,
                  x0=xc - xlen/2., y0=yc - ylen/2., z0=zc - zlen/2.)
    t0 = time.time()
    phantom.embed_in(img)
    print(f"  embed_in {time.time()-t0:.1f}s")
    vol = np.ascontiguousarray(img.mat.transpose(2, 1, 0)).astype(np.float32)
    return vol


def add_poisson_noise(clean_sino, I0, rng):
    """Simulate Poisson photon counts at I0 incident, return the
    negative-log noisy line integral. Zero-count pixels clamped to 1."""
    N_mean = I0 * np.exp(-clean_sino)
    N = rng.poisson(N_mean).astype(np.float32)
    N = np.maximum(N, 1.0)
    return np.log(I0 / N).astype(np.float32)


def add_lf_noise(clean_sino, lf_amplitude_pct, lf_sigma_pixels, rng):
    """Generate Gaussian-smoothed white noise per view, scale to
    lf_amplitude_pct of the per-view mean line integral, and return the
    LF-noise field (caller adds to the clean or Poisson-noisy sinogram).

    Sino shape is (det_row, nviews, det_col). The smoothing is applied
    per-view (axis 1 untouched) since scatter is spatially correlated
    within a single projection, not across views.
    """
    from scipy.ndimage import gaussian_filter
    out = np.zeros_like(clean_sino)
    for v in range(clean_sino.shape[1]):
        white = rng.standard_normal(
            (clean_sino.shape[0], clean_sino.shape[2])).astype(np.float32)
        smoothed = gaussian_filter(white, sigma=lf_sigma_pixels)
        smoothed /= max(float(smoothed.std()), 1e-12)
        view_mean = float(clean_sino[:, v, :].mean())
        out[:, v, :] = smoothed * (lf_amplitude_pct / 100.0) * view_mean
    return out


def build_noisy_sino(clean_sino, I0, lf_pct):
    """Compose Poisson and LF noise per scenario. Returns None if both
    are None (the noise-free baseline)."""
    if I0 is None and lf_pct is None:
        return None
    if I0 is not None:
        rng_p = np.random.default_rng(POISSON_SEED)
        noisy = add_poisson_noise(clean_sino, I0, rng_p)
    else:
        noisy = clean_sino.copy()
    if lf_pct is not None:
        rng_lf = np.random.default_rng(LF_SEED)
        noisy = noisy + add_lf_noise(clean_sino, lf_pct, LF_SIGMA_PIXELS, rng_lf)
    return noisy.astype(np.float32)


def main():
    if SUMMARY.exists():
        SUMMARY.unlink()
    log(f"ct-2 breast structured-noise sweep at {SHAPE} + arc={ARC_DEG} + "
        f"{NVIEWS}v, itermax={ITERMAX}")
    log(f"  CP override: {CP_OVERRIDES}; other CP at 2D defaults "
        f"(inflate={DEFAULTS['norm_inflate_3d']}, "
        f"eps_lo={DEFAULTS['eps_lo_ratio']}, sigma_lo={DEFAULTS['sigma_lo_scale']})")
    log(f"  scenarios: {SCENARIOS}")
    log(f"  LF sigma (pixels): {LF_SIGMA_PIXELS}")
    log(f"  snapshots {SNAPSHOT_ITERS}\n")

    phantom = build_phantom()
    print(f"  phantom shape {phantom.shape}\n")

    det_row, det_col, det_sp = DET
    vol_geom, proj_geom, gi = vr.build_dbt_geometry(
        phantom.shape, DX_CM,
        det_row_count=det_row, det_col_count=det_col, det_spacing=det_sp,
        nviews=NVIEWS, arc_deg=ARC_DEG, sod=SOD_CM, odd=ODD_CM,
    )
    A, At = vr.make_projector(vol_geom, proj_geom)
    nusino, nuxgrad, nuygrad, nuzgrad = vr.operator_norms(
        phantom.shape, A, At, vr.CONFIG["npower"]
    )

    # Forward-project once -- this is the noise-free truth sinogram.
    print("\nForward-projecting clean truth sinogram...")
    t0 = time.time()
    clean_sino = A(phantom).astype(np.float32)
    print(f"  done in {time.time()-t0:.0f}s; "
          f"range [{clean_sino.min():.4f}, {clean_sino.max():.4f}], "
          f"mean {clean_sino.mean():.4f}")

    saved_itermax = vr.CONFIG["itermax"]
    vr.CONFIG["itermax"] = ITERMAX

    try:
        for label, I0, lf_pct in SCENARIOS:
            print(f"\n##### scenario: {label}  (I0={I0}, lf={lf_pct}%) #####")

            noisy_sino = build_noisy_sino(clean_sino, I0, lf_pct)
            if noisy_sino is not None:
                diff = noisy_sino - clean_sino
                print(f"  noisy_sino range "
                      f"[{noisy_sino.min():.4f}, {noisy_sino.max():.4f}], "
                      f"noise RMS = {float(np.sqrt((diff*diff).mean())):.5f}")
                # Also break down LF vs total RMS so we can see how much
                # came from each component.
                if lf_pct is not None:
                    lf_only = build_noisy_sino(clean_sino, None, lf_pct) - clean_sino
                    print(f"  LF component RMS = {float(np.sqrt((lf_only*lf_only).mean())):.5f}")

            # single
            reset_cp()
            print("\n  --- single ---")
            t0 = time.time()
            rs, is_, ds_, ts_, snaps_s = vr.run_single_channel(
                phantom, A, At, nusino, nuxgrad, nuygrad, nuzgrad,
                gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
                presimulated_sino=noisy_sino,
            )
            isng_np = np.asarray(is_)
            single_time = time.time() - t0
            print(f"  single {single_time:.0f}s, RMSE@iter30 = {isng_np[-1]:.5f}")

            # two-channel
            reset_cp()
            for k, v in CP_OVERRIDES.items():
                vr.CONFIG[k] = v
            R_hi, R_lo = vr.build_sinogram_filters(
                gi["det_col_count"], gi["det_spacing"],
                vr.CONFIG["cutoffparm"], vr.CONFIG["cutoffparm_lo"],
                axis="2d", det_row_count=gi["det_row_count"],
            )
            print("\n  --- two-channel ---")
            t0 = time.time()
            rt, it_, dt_, tt_, snaps_t = vr.run_two_channel(
                phantom, A, At, R_hi, R_lo,
                nusino, nuxgrad, nuygrad, nuzgrad,
                gi["nrays"], snapshot_iters=SNAPSHOT_ITERS,
                presimulated_sino=noisy_sino,
            )
            itwo_np = np.asarray(it_)
            two_time = time.time() - t0
            print(f"  two    {two_time:.0f}s, RMSE@iter30 = {itwo_np[-1]:.5f}")

            log(f"{label:<16}  | single {fmt_iters(isng_np, REPORT_ITERS)} "
                f"| two {fmt_iters(itwo_np, REPORT_ITERS)} "
                f"| r {fmt_reds(isng_np, itwo_np, REPORT_ITERS)}")

            out_pkl = CACHE_DIR / f"ct2_breast_poisson_{label}.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump({
                    "phantom":          phantom,
                    "recon_single":     rs,
                    "recon_two":        rt,
                    "ierrs_single":     isng_np.tolist(),
                    "ierrs_two":        itwo_np.tolist(),
                    "snapshots_single": snaps_s,
                    "snapshots_two":    snaps_t,
                    "dx_cm":            DX_CM,
                    "geometry":         gi,
                    "label":            label,
                    "I0":               I0,
                    "lf_pct":           lf_pct,
                    "lf_sigma_pixels":  LF_SIGMA_PIXELS,
                    "cp_overrides":     CP_OVERRIDES,
                    "clean_sino_stats": {
                        "min": float(clean_sino.min()),
                        "max": float(clean_sino.max()),
                        "mean": float(clean_sino.mean()),
                    },
                }, f)
            print(f"  cached {out_pkl.name}")
    finally:
        vr.CONFIG["itermax"] = saved_itermax
        reset_cp()

    log("\nDone.")


if __name__ == "__main__":
    main()
