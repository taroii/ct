"""Pre-registered diagnostic for the 3D noise sign flip.

THE RESULT BEING EXPLAINED
    run_3d_noise_stats (VICTRE, Poisson noise, s=4, 300 iters): two-channel
    loses to single-channel on 10/10 realizations (0.194 vs 0.160 at iter 300),
    while the same method wins noiselessly in 3D and under noise in 2D.

THE HYPOTHESIS UNDER TEST (H)
    Under noise both arms share one eps, calibrated on the single-channel
    filtered noise, and two-channel sets eps_hi = 1.0*eps, eps_lo = 1.25*eps
    (recon3d.py solve_two). Ramp-filtered noise is almost entirely high-band --
    under the separable 2D DBT window the low band holds ~0.03% of it -- so the
    low-channel ball is enormous: comparable to the low-band SIGNAL itself. The
    two-channel problem is then a different, far looser problem whose solution
    discards most large-scale contrast. The sign flip is a property of the
    constraint set, not of the amplified dual step and not of a bug.

    A 2D scratch experiment already showed the mechanism (breast, one seed):
    eps_lo 1.25 -> two wins +33.8%; 2.6 -> loses -19%; 0.1 -> wins +63%. This
    script is the 3D test.

PREDICTIONS -- written before the run, evaluated automatically at the end
    P0  harness   The control (s=4, eps_lo=1.25, and single) reproduces the
                  stored run_3d_noise_stats trace for this realization, max
                  |diff| <= 1e-4 over 300 iterations. If P0 fails, NOTHING
                  below may be interpreted: the test is not measuring the
                  system that produced the result.
    P1  dose-response   With eps_hi fixed at 1.0, two-channel RMSE@300 is
                  non-decreasing in eps_lo over {0.02, 0.1, 0.3, 0.6, 1.25,
                  2.6} (tolerance 1e-3).
    P2  sign      Two-channel beats single-channel at iter 300 when the low
                  ball is tight: both at eps_lo=0.1 and under per-band
                  calibration (each band's eps = 1.1 x that band's own
                  calibration-draw noise, i.e. the proposed fix).
    P3  mechanism At eps_lo=1.25 the final low-band residual sits on its ball
                  (activity >= 0.95) AND is >= 5x the low-band noise -- the
                  reconstruction is discarding real low-band data, not noise.
                  Under per-band calibration it is <= 2x the low-band noise.
    P4  problem, not dynamics   s only changes the path, not the saddle point.
                  So s=1 at eps_lo=1.25 must ALSO lose to single at iter 600,
                  and end within 10% of s=4 at iter 600. If s=1 wins, the
                  amplified step -- not the constraint -- is to blame.
    P5  where the error is (supporting)   At eps_lo=1.25 at least half of
                  two-channel's excess squared error over single is in the
                  low-pass (Gaussian, sigma = --lp-sigma voxels) component.
    P6  dose (supporting)   At 10x i0 every ball shrinks relative to the
                  signal, so two-channel's relative deficit at iter 300 is
                  smaller than at 1x.

    H is SUPPORTED iff P0 passes and P1, P2, P3 all pass. P4 separates the
    constraint from the dynamics; P5 and P6 are corroborating, not decisive.
    Any failure of P1-P4 is reported as evidence AGAINST H, not explained away.

REPRODUCING THE STORED RUN
    reconstruct() builds the geometry, filters, norms, noisy data and eps
    internally and does not return the data, so this script rebuilds them
    with recon3d's own functions and the same seed formulas. P0 is what
    guarantees that rebuild is exact rather than assumed.

Every case is saved as soon as it finishes and skipped on re-run (use
--force to redo), so an interruption loses at most the case in flight.

    python paper/experiments/diagnose_3d_noise.py            # everything, ~2 h GPU
    python paper/experiments/diagnose_3d_noise.py --only ctrl_single ctrl_two
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy import sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The solver stack imports ASTRA, which only exists on the GPU server. It is
# loaded on first use so that --report-only can re-evaluate saved cases on a
# machine without it.
r3 = load_phantom = VICTRE_GEOM = auto_i0 = SOLVER_SEED = None


def _solver_imports():
    global r3, load_phantom, VICTRE_GEOM, auto_i0, SOLVER_SEED
    if r3 is None:
        import recon3d as _r3
        from run_3d_victre import load_phantom as _lp, VICTRE_GEOM as _vg
        from run_3d_noise_stats import auto_i0 as _ai, SOLVER_SEED as _ss
        r3, load_phantom, VICTRE_GEOM, auto_i0, SOLVER_SEED = _r3, _lp, _vg, _ai, _ss



OUT = HERE / "results" / "diagnose_3d_noise"
FIG = HERE / "figs"
TAB = HERE / "tables"
STORE = HERE / "results" / "3d_noise_stats"

S_HERO = 4.0
LO_SWEEP = [0.02, 0.1, 0.3, 0.6, 1.25, 2.6]


def cases():
    """Every run in the test. 'hi'/'lo' of None mean per-band calibrated."""
    C = [dict(name="ctrl_single", arm="single", s=S_HERO, hi=1.0, lo=1.25,
              itermax=600, dose=1),
         dict(name="ctrl_two", arm="two", s=S_HERO, hi=1.0, lo=1.25,
              itermax=600, dose=1)]
    for lo in LO_SWEEP:
        if lo == 1.25:
            continue                      # the control IS this point
        C.append(dict(name=f"two_lo{lo:g}", arm="two", s=S_HERO, hi=1.0,
                      lo=lo, itermax=300, dose=1))
    C += [dict(name="two_perband", arm="two", s=S_HERO, hi=None, lo=None,
               itermax=300, dose=1),
          dict(name="two_s1_lo1.25", arm="two", s=1.0, hi=1.0, lo=1.25,
               itermax=600, dose=1),
          dict(name="dose10_single", arm="single", s=S_HERO, hi=1.0, lo=1.25,
               itermax=300, dose=10),
          dict(name="dose10_two", arm="two", s=S_HERO, hi=1.0, lo=1.25,
               itermax=300, dose=10)]
    return C


# --------------------------------------------------------------------------
# Rebuilding exactly what reconstruct() built
# --------------------------------------------------------------------------
class System:
    """Geometry, projector, filters and norms: data-independent, built once."""

    def __init__(self, down, npower):
        _solver_imports()
        self.phantom, self.dx_cm = load_phantom(down)
        self.base = {**r3.DEFAULTS, **dict(
            itermax=300, sigma_lo_scale=S_HERO, seed=SOLVER_SEED,
            eps_mode="noise", eps_factor=1.1, npower=npower)}
        geom = dict(VICTRE_GEOM)
        orbit = geom.pop("orbit")
        assert orbit == "dbt", "VICTRE runs are DBT; the filter axis depends on it"
        vol_geom, proj_geom, self.info = r3.dbt_geometry(
            self.phantom.shape, self.dx_cm, **geom)
        self.base["_nrays"] = self.info["nrays"]
        self.A, self.At = r3.make_projector(vol_geom, proj_geom)
        # reconstruct(): filter_axis = cfg.get("filter_axis", "2d" if dbt else "u")
        axis = self.base.get("filter_axis", "2d")
        (self.R_single, self.R_hi, self.R_lo, self.R_single_sq,
         self.R_data_sq) = r3.band_filters(
            self.info["det_cols"], self.info["det_spacing"],
            self.base["cutoffparm"], self.base["cutoffparm_lo"], axis=axis,
            det_rows=self.info["det_rows"], ramp=self.base.get("ramp", True))
        self.norms = r3.block_norms(self.phantom.shape, self.A, self.At,
                                    self.R_single_sq, self.base)
        self.clean = self.A(self.phantom)
        self._data = {}

    def nrm(self, R, s):
        v = R(s)
        return float(sqrt((v * v).sum()))

    def data(self, noise_seed, dose):
        """Noisy sinogram + eps for one realization, as reconstruct() makes them."""
        key = (noise_seed, dose)
        if key in self._data:
            return self._data[key]
        i0 = auto_i0(self.phantom, self.dx_cm) * dose
        nrng = np.random.default_rng(
            (int(noise_seed) * 2_654_435_761 + 12345) % (2 ** 63))
        sino, n_zero = r3.poisson_sinogram(self.clean, i0, nrng)
        crng = np.random.default_rng(
            (int(noise_seed) * 40_503 + 777) % (2 ** 63))
        nrays = self.info["nrays"]
        eps = r3.calibrate_eps(self.clean, i0, self.R_single, nrays, crng,
                               factor=self.base["eps_factor"])

        # Per-band calibration from the SAME independent calibration draw that
        # set eps, so the per-band case differs from the control only in how
        # that draw's noise is split between the bands.
        crng2 = np.random.default_rng(
            (int(noise_seed) * 40_503 + 777) % (2 ** 63))
        cal, _ = r3.poisson_sinogram(self.clean, i0, crng2)
        cn = cal - self.clean
        nS, nH, nL = (self.nrm(self.R_single, cn), self.nrm(self.R_hi, cn),
                      self.nrm(self.R_lo, cn))
        eps_check = self.base["eps_factor"] * nS / sqrt(nrays)
        assert abs(eps_check - eps) <= 1e-9 * max(1.0, abs(eps)), \
            "per-band calibration is not using the draw that set eps"

        n = sino - self.clean
        d = dict(i0=i0, sino=sino, eps=eps, zero_frac=n_zero / sino.size,
                 pb_hi=nH / nS, pb_lo=nL / nS,
                 noise=dict(single=self.nrm(self.R_single, n),
                            hi=self.nrm(self.R_hi, n), lo=self.nrm(self.R_lo, n)),
                 signal=dict(single=self.nrm(self.R_single, self.clean),
                             hi=self.nrm(self.R_hi, self.clean),
                             lo=self.nrm(self.R_lo, self.clean)))
        print(f"data: seed={noise_seed} dose x{dose} i0={i0:.6g} eps={eps:.6g} "
              f"per-band ratios hi={d['pb_hi']:.4f} lo={d['pb_lo']:.5f} "
              f"zero-count {d['zero_frac']:.1e}")
        self._data[key] = d
        return d


def run_case(S: System, c: dict, noise_seed: int, lp_sigma: float) -> dict:
    D = S.data(noise_seed, c["dose"])
    hi = D["pb_hi"] if c["hi"] is None else c["hi"]
    lo = D["pb_lo"] if c["lo"] is None else c["lo"]
    cfg = dict(S.base)
    cfg.update(itermax=c["itermax"], sigma_lo_scale=c["s"], noise_seed=noise_seed,
               i0=D["i0"], eps=D["eps"], eps_hi_ratio=hi, eps_lo_ratio=lo)
    if c["arm"] == "single":
        res = r3.solve_single(S.phantom, S.A, S.At, S.R_single, S.norms, cfg,
                              sinodata=D["sino"])
    else:
        res = r3.solve_two(S.phantom, S.A, S.At, S.R_hi, S.R_lo, S.R_data_sq,
                           S.norms, cfg, sinodata=D["sino"])

    x = res["recon"]
    r = S.A(x) - D["sino"]
    root = sqrt(S.info["nrays"])
    resid = dict(single=S.nrm(S.R_single, r), hi=S.nrm(S.R_hi, r),
                 lo=S.nrm(S.R_lo, r))
    # The ball each band is actually held to (solve_single / solve_two).
    if c["arm"] == "single":
        balls = dict(single=D["eps"] * root)
    else:
        balls = dict(hi=hi * D["eps"] * root, lo=lo * D["eps"] * root)

    e = x - S.phantom
    e_lp = gaussian_filter(e, lp_sigma)
    nv = e.size
    return dict(
        name=c["name"], arm=c["arm"], s=c["s"], dose=c["dose"],
        itermax=c["itermax"], hi_ratio=float(hi), lo_ratio=float(lo),
        i0=D["i0"], eps=D["eps"], ierrs=np.asarray(res["ierrs"], float),
        resid=resid, balls=balls, noise=D["noise"], signal=D["signal"],
        # truth's own residual vs each ball: > 1 means the truth is infeasible
        truth_over_ball={b: D["noise"][b] / v for b, v in balls.items()},
        rmse_lp=float(sqrt((e_lp ** 2).sum() / nv)),
        rmse_hp=float(sqrt(((e - e_lp) ** 2).sum() / nv)),
        rmse=float(sqrt((e ** 2).sum() / nv)))


def _files(stem: Path):
    """(json, npz) for a case. NOT Path.with_suffix: case names carry decimals
    ("two_lo0.02"), which with_suffix reads as an extension -- every eps_lo
    case would collapse onto one file and the resume check would skip the
    rest of the sweep as "already saved"."""
    return (stem.parent / f"{stem.name}.json", stem.parent / f"{stem.name}.npz")


def save(res, stem):
    jp, npz = _files(stem)
    meta = {k: v for k, v in res.items() if k != "ierrs"}
    with open(npz, "wb") as fh:           # a handle, so savez adds no suffix
        np.savez(fh, ierrs=res["ierrs"])
    jp.write_text(json.dumps(meta, indent=1, default=float))


def load(stem):
    jp, npz = _files(stem)
    meta = json.loads(jp.read_text())
    with np.load(npz) as d:
        meta["ierrs"] = np.asarray(d["ierrs"], float)
    return meta


def saved(stem):
    jp, npz = _files(stem)
    return jp.exists() and npz.exists()


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def stored_trace(noise_seed, i0):
    """The production run_3d_noise_stats trace for this realization, if present."""
    for jp in sorted(STORE.glob("*.json")):
        c = json.loads(jp.read_text())["config"]
        if (c.get("noise_seed") == noise_seed and c.get("itermax") == 300
                and c.get("s") == S_HERO
                and abs(c.get("i0", 0) - i0) <= 1e-9 * i0):
            with np.load(jp.with_suffix(".npz")) as d:
                return (np.asarray(d["ierrs_single"], float),
                        np.asarray(d["ierrs_two"], float))
    return None


def evaluate(R: dict, noise_seed: int) -> tuple[list, str]:
    """Each prediction -> (id, verdict, detail). Verdicts: PASS FAIL MISSING."""
    at = lambda n, it: float(R[n]["ierrs"][it - 1])      # noqa: E731
    have = lambda *ns: all(n in R for n in ns)            # noqa: E731
    out = []

    # P0
    if have("ctrl_single", "ctrl_two"):
        st = stored_trace(noise_seed, R["ctrl_single"]["i0"])
        if st is None:
            out.append(("P0", "MISSING", "no stored run_3d_noise_stats trace "
                        f"for noise_seed={noise_seed} to compare against"))
        else:
            ds = float(np.abs(R["ctrl_single"]["ierrs"][:300] - st[0]).max())
            dt = float(np.abs(R["ctrl_two"]["ierrs"][:300] - st[1]).max())
            out.append(("P0", "PASS" if max(ds, dt) <= 1e-4 else "FAIL",
                        f"max |diff| vs stored: single {ds:.2e}, two {dt:.2e}"))
    else:
        out.append(("P0", "MISSING", "control not run"))

    # P1
    names = [f"two_lo{lo:g}" if lo != 1.25 else "ctrl_two" for lo in LO_SWEEP]
    if have(*names):
        v = [at(n, 300) for n in names]
        mono = all(b >= a - 1e-3 for a, b in zip(v, v[1:]))
        out.append(("P1", "PASS" if mono else "FAIL",
                    "RMSE@300 by eps_lo " + ", ".join(
                        f"{lo:g}:{x:.4f}" for lo, x in zip(LO_SWEEP, v))))
    else:
        out.append(("P1", "MISSING", "sweep incomplete"))

    # P2
    if have("ctrl_single", "two_lo0.1", "two_perband"):
        sg = at("ctrl_single", 300)
        a, b = at("two_lo0.1", 300), at("two_perband", 300)
        out.append(("P2", "PASS" if (a < sg and b < sg) else "FAIL",
                    f"single {sg:.4f} | two eps_lo=0.1 {a:.4f} "
                    f"({100*(sg-a)/sg:+.1f}%) | two per-band {b:.4f} "
                    f"({100*(sg-b)/sg:+.1f}%)"))
    else:
        out.append(("P2", "MISSING", "needs ctrl_single, two_lo0.1, two_perband"))

    # P3
    if have("ctrl_two", "two_perband"):
        c, p = R["ctrl_two"], R["two_perband"]
        act = c["resid"]["lo"] / c["balls"]["lo"]
        over = c["resid"]["lo"] / c["noise"]["lo"]
        pover = p["resid"]["lo"] / p["noise"]["lo"]
        ok = act >= 0.95 and over >= 5.0 and pover <= 2.0
        out.append(("P3", "PASS" if ok else "FAIL",
                    f"eps_lo=1.25: lo residual / lo ball {act:.3f}, / lo noise "
                    f"{over:.1f}x, / lo signal {c['resid']['lo']/c['signal']['lo']:.3f} | "
                    f"per-band: lo residual / lo noise {pover:.2f}x"))
    else:
        out.append(("P3", "MISSING", "needs ctrl_two and two_perband"))

    # P4
    if have("ctrl_single", "ctrl_two", "two_s1_lo1.25"):
        sg, s4, s1 = (at("ctrl_single", 600), at("ctrl_two", 600),
                      at("two_s1_lo1.25", 600))
        ok = s1 > sg and abs(s1 - s4) / s4 <= 0.10
        out.append(("P4", "PASS" if ok else "FAIL",
                    f"@600: single {sg:.4f}, two s=4 {s4:.4f}, two s=1 {s1:.4f} "
                    f"(s=1 vs s=4 {100*(s1-s4)/s4:+.1f}%)"))
    else:
        out.append(("P4", "MISSING", "needs the 600-iteration runs"))

    # P5
    if have("ctrl_single", "ctrl_two"):
        a, b = R["ctrl_single"], R["ctrl_two"]
        tot = b["rmse"] ** 2 - a["rmse"] ** 2
        lp = b["rmse_lp"] ** 2 - a["rmse_lp"] ** 2
        frac = lp / tot if tot > 0 else float("nan")
        out.append(("P5", "PASS" if tot > 0 and frac >= 0.5 else "FAIL",
                    f"share of excess squared error that is low-pass: {frac:.2f} "
                    f"(at final iterate, iter 600)"))
    else:
        out.append(("P5", "MISSING", "control not run"))

    # P6
    if have("ctrl_single", "ctrl_two", "dose10_single", "dose10_two"):
        d1 = (at("ctrl_two", 300) - at("ctrl_single", 300)) / at("ctrl_single", 300)
        d10 = (at("dose10_two", 300) - at("dose10_single", 300)) / at("dose10_single", 300)
        out.append(("P6", "PASS" if d10 < d1 else "FAIL",
                    f"two-channel deficit @300: dose x1 {100*d1:+.1f}%, "
                    f"dose x10 {100*d10:+.1f}%"))
    else:
        out.append(("P6", "MISSING", "dose runs not done"))

    v = {pid: verdict for pid, verdict, _ in out}
    if v["P0"] == "FAIL":
        overall = ("INVALID: the control does not reproduce the stored run, so "
                   "no other prediction can be interpreted.")
    elif v["P0"] != "PASS" or any(v[p] == "MISSING" for p in ("P1", "P2", "P3")):
        overall = "INCOMPLETE: P0-P3 have not all been evaluated."
    elif all(v[p] == "PASS" for p in ("P1", "P2", "P3")):
        overall = "H SUPPORTED: P0-P3 all pass."
    else:
        failed = [p for p in ("P1", "P2", "P3") if v[p] == "FAIL"]
        overall = f"H NOT SUPPORTED: {', '.join(failed)} failed."
    return out, overall


def write_report(R, verdicts, overall, noise_seed, lp_sigma):
    L = ["# 3D noise sign flip: pre-registered diagnostic\n",
         f"VICTRE, noise realization {noise_seed}. Predictions are defined in "
         f"the docstring of `diagnose_3d_noise.py` and were fixed before the "
         f"run; verdicts below are computed, not judged.\n",
         f"**Overall: {overall}**\n",
         "| prediction | verdict | evidence |", "|---|---|---|"]
    L += [f"| {p} | {v} | {d} |" for p, v, d in verdicts]
    L += ["", "## All cases\n",
          "| case | arm | s | dose | eps_hi | eps_lo | RMSE@300 | final RMSE | "
          "lo resid / lo ball | lo resid / lo noise | lo resid / lo signal | "
          "truth / lo ball |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in R.items():
        lo_ball = r["balls"].get("lo")
        row = [n, r["arm"], f"{r['s']:g}", f"x{r['dose']}",
               f"{r['hi_ratio']:.4g}", f"{r['lo_ratio']:.4g}",
               f"{r['ierrs'][299]:.4f}", f"{r['ierrs'][-1]:.4f} (@{len(r['ierrs'])})"]
        if lo_ball:
            row += [f"{r['resid']['lo']/lo_ball:.3f}",
                    f"{r['resid']['lo']/r['noise']['lo']:.1f}",
                    f"{r['resid']['lo']/r['signal']['lo']:.3f}",
                    f"{r['truth_over_ball']['lo']:.3f}"]
        else:
            row += ["-", f"{r['resid']['lo']/r['noise']['lo']:.1f}",
                    f"{r['resid']['lo']/r['signal']['lo']:.3f}", "-"]
        L.append("| " + " | ".join(row) + " |")
    L += ["", "`truth / lo ball` > 1 means the ground truth itself violates the "
              "low-band constraint, i.e. that ball is too tight to be fair.",
          f"Low-pass split for P5: Gaussian, sigma = {lp_sigma} voxels."]
    p = TAB / "diagnose_3d_noise.md"
    p.write_text("\n".join(L) + "\n")
    print(f"saved {p}")


def write_fig(R):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    if "ctrl_single" in R:
        ax.semilogx(np.arange(1, len(R["ctrl_single"]["ierrs"]) + 1),
                    R["ctrl_single"]["ierrs"], "k-", lw=2.2, label="single")
    cmap = plt.get_cmap("viridis")
    names = [f"two_lo{lo:g}" if lo != 1.25 else "ctrl_two" for lo in LO_SWEEP]
    for i, (lo, n) in enumerate(zip(LO_SWEEP, names)):
        if n in R:
            y = R[n]["ierrs"]
            ax.semilogx(np.arange(1, len(y) + 1), y, "-",
                        color=cmap(i / (len(LO_SWEEP) - 1)), lw=1.5,
                        label=f"two, eps_lo={lo:g}")
    if "two_perband" in R:
        y = R["two_perband"]["ierrs"]
        ax.semilogx(np.arange(1, len(y) + 1), y, "--", color="crimson", lw=2,
                    label=f"two, per-band (eps_lo={R['two_perband']['lo_ratio']:.3g})")
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7)
    ax.set_title("low-band tolerance sweep (s=4, dose x1)", fontsize=10)

    ax = axes[1]
    for n, st, lab in (("ctrl_single", "k-", "single"),
                       ("ctrl_two", "b-", "two s=4, eps_lo=1.25"),
                       ("two_s1_lo1.25", "b:", "two s=1, eps_lo=1.25"),
                       ("dose10_single", "k--", "single, dose x10"),
                       ("dose10_two", "b--", "two s=4, dose x10")):
        if n in R:
            y = R[n]["ierrs"]
            ax.semilogx(np.arange(1, len(y) + 1), y, st, lw=1.8, label=lab)
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7)
    ax.set_title("dynamics (s) and dose controls", fontsize=10)
    fig.tight_layout()
    p = FIG / "diagnose_3d_noise.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise-seed", type=int, default=1000,
                    help="realization; 1000 is rep 0 of run_3d_noise_stats, "
                         "which P0 checks against")
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--npower", type=int, default=200)
    ap.add_argument("--lp-sigma", type=float, default=4.0,
                    help="Gaussian sigma (voxels) for the P5 low-pass split")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these cases (see cases())")
    ap.add_argument("--force", action="store_true", help="recompute saved cases")
    ap.add_argument("--report-only", action="store_true",
                    help="evaluate saved cases without touching the GPU")
    ap.add_argument("--smoke", action="store_true",
                    help="run single, two and per-band for 3 iterations through "
                         "the full code path, print their metrics, save nothing")
    args = ap.parse_args()

    if args.smoke:
        # Exercises every solver call and every metric before committing hours
        # of GPU time; results are printed only, so the resume cache stays clean.
        S = System(args.down, args.npower)
        for c in (dict(name="smoke_single", arm="single", s=S_HERO, hi=1.0,
                       lo=1.25, itermax=3, dose=1),
                  dict(name="smoke_two", arm="two", s=S_HERO, hi=1.0, lo=1.25,
                       itermax=3, dose=1),
                  dict(name="smoke_perband", arm="two", s=S_HERO, hi=None,
                       lo=None, itermax=3, dose=1)):
            r = run_case(S, c, args.noise_seed, args.lp_sigma)
            print(f"{c['name']}: ierrs {np.round(r['ierrs'], 5).tolist()} "
                  f"resid {r['resid']} balls {r['balls']}")
        st = stored_trace(args.noise_seed, S.data(args.noise_seed, 1)["i0"])
        print("stored trace for P0:", "found" if st is not None else "NOT FOUND")
        if st is not None:
            print(f"  stored first 3 iters: single {np.round(st[0][:3], 5).tolist()} "
                  f"two {np.round(st[1][:3], 5).tolist()}  <- smoke should match")
        return

    for d in (OUT, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases() if args.only is None or c["name"] in args.only]
    unknown = set(args.only or []) - {c["name"] for c in cases()}
    if unknown:
        raise SystemExit(f"unknown case(s) {sorted(unknown)}; "
                         f"known: {[c['name'] for c in cases()]}")

    tag = f"seed{args.noise_seed}"
    if not args.report_only:
        S = None
        for c in todo:
            path = OUT / f"{tag}_{c['name']}"
            if saved(path) and not args.force:
                print(f"[skip] {c['name']} already saved")
                continue
            if S is None:
                S = System(args.down, args.npower)
            print(f"\n===== {c['name']} =====")
            save(run_case(S, c, args.noise_seed, args.lp_sigma), path)

    R = {}
    for c in cases():
        path = OUT / f"{tag}_{c['name']}"
        if saved(path):
            R[c["name"]] = load(path)
    verdicts, overall = evaluate(R, args.noise_seed)
    print("\n" + "\n".join(f"{p}  {v:8s} {d}" for p, v, d in verdicts))
    print(f"\n{overall}")
    write_report(R, verdicts, overall, args.noise_seed, args.lp_sigma)
    write_fig(R)


if __name__ == "__main__":
    main()
