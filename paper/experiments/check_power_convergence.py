"""How many power iterations does the certified step-size bound actually need?

The step sizes are derived from lambda_max(M), estimated by power iteration
from a random start vector. That start vector is the only stochastic element in
an otherwise deterministic 3D pipeline, so the SPREAD OF THE ESTIMATE ACROSS
SEEDS is a direct, assumption-free measure of how converged the estimate is.

This matters more than it looks. The certificate is tau*lambda_max(M) < 1,
enforced with a fixed safety slack (cvg_slack, default 1e-3). Power iteration's
Rayleigh quotient converges from BELOW, so an unconverged estimate understates
lambda_max and tau is set too large -- the run is then uncertified without
saying so. A relative error d in lambda eats d of the slack directly.

Do NOT use the per-iteration change as the convergence measure: with a small
spectral gap the increment can reach 1e-5 while the estimate is still 0.3% off.
The seed spread is the honest measure, which is what this script reports.

    python paper/experiments/check_power_convergence.py
    python paper/experiments/check_power_convergence.py --geom victre --down 4

Read the output as: pick the smallest npower whose two-channel spread is
comfortably under cvg_slack, then set that npower in the runner.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recon3d  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
NPOWERS = [10, 20, 40, 60, 100, 150, 200, 300]
HEADROOM = 4.0        # required safety factor between lambda spread and slack


def synthetic(nz=48, ny=96, nx=96):
    z, y, x = np.ogrid[:nz, :ny, :nx]
    v = np.zeros((nz, ny, nx), np.float32)
    v[((z-nz/2)/(nz/3))**2 + ((y-ny/2)/(ny/3))**2 + ((x-nx/2)/(nx/3))**2 < 1] = 1.0
    for k in range(4, nz, 8):                    # slabs, for low-frequency content
        v[k:k+3] *= 0.4
    return np.ascontiguousarray(v), 10.0 / nx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", choices=["synthetic", "victre"], default="synthetic")
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--slo", type=float, default=4.0)
    ap.add_argument("--slack", type=float, default=1e-3)
    args = ap.parse_args()

    if args.geom == "victre":
        from run_3d_victre import load_phantom
        ph, dx = load_phantom(args.down)
        geom = dict(det_rows=480, det_cols=480, det_spacing=0.04,
                    nviews=25, arc_deg=50.0, sod=65.0, odd=5.0)
    else:
        ph, dx = synthetic()
        geom = dict(det_rows=128, det_cols=128, det_spacing=0.06,
                    nviews=25, arc_deg=50.0, sod=65.0, odd=5.0)

    vg, pg, info = recon3d.dbt_geometry(ph.shape, dx, **geom)
    A, At = recon3d.make_projector(vg, pg)
    Rs, Rhi, Rlo, Rs2, Rd2 = recon3d.band_filters(
        info["det_cols"], info["det_spacing"], 4.0, 8.0, axis="2d",
        det_rows=info["det_rows"], ramp=True)

    print(f"\ngeometry: {args.geom}  volume {ph.shape}  "
          f"detector {info['det_rows']}x{info['det_cols']}  s={args.slo}")
    print(f"seeds: {SEEDS}   safety slack: {args.slack:.1e}\n")
    print(f"{'npower':>7} | {'||K|| single':>27} | {'sqrt(lam_max(M)) two':>27} | ok?")
    print(f"{'':>7} | {'mean':>13} {'seed spread':>13} | "
          f"{'mean':>13} {'seed spread':>13} |")
    print("-" * 82)

    best = None
    for npw in NPOWERS:
        sn, tn = [], []
        for sd in SEEDS:
            cfg = {**recon3d.DEFAULTS, "npower": npw, "seed": sd,
                   "npower_tol": 0.0,              # force the full npw iterations
                   "sigma_lo_scale": args.slo, "_nrays": info["nrays"]}
            nus, nux, nuy, nuz = recon3d.block_norms(ph.shape, A, At, Rs2, cfg)
            sn.append(recon3d.single_norm(ph.shape, A, At, Rs,
                                          nus, nux, nuy, nuz, cfg))
            tn.append(recon3d.certified_norm_two(ph.shape, A, At, Rd2,
                                                 nus, nux, nuy, nuz, cfg))
        sn, tn = np.array(sn), np.array(tn)
        # lambda = norm^2, so a relative spread d in the norm is 2d in lambda,
        # and it is lambda that the certificate consumes slack against.
        lam_spread = 2 * np.ptp(tn) / tn.mean()
        # Require real headroom, not bare satisfaction. Passing at
        # lam_spread = 0.78 * slack (which npower=60 does here) means 78% of the
        # safety margin is already spent on norm-estimation error before the
        # certificate does any work. HEADROOM=4 asks for a factor of four.
        ok = lam_spread < args.slack / HEADROOM
        if ok and best is None:
            best = npw
        print(f"{npw:>7} | {sn.mean():>13.6f} {np.ptp(sn)/sn.mean():>13.2e} | "
              f"{tn.mean():>13.6f} {np.ptp(tn)/tn.mean():>13.2e} | "
              f"{'OK' if ok else 'UNSAFE'}")

    print(f"\nlambda spread = 2 x norm spread; required < cvg_slack/{HEADROOM:.0f} "
          f"= {args.slack/HEADROOM:.1e}  (cvg_slack={args.slack:.1e})")
    if best:
        print(f"RECOMMENDATION: npower >= {best} for this geometry.")
    else:
        print("RECOMMENDATION: none of the tested npower values is safe -- "
              "raise NPOWERS or cvg_slack.")
    print("Reminder: seed spread is the honest measure here. The per-iteration "
          "change inside the solver is NOT the error and can look converged "
          "while the estimate is still well off.")


if __name__ == "__main__":
    main()
