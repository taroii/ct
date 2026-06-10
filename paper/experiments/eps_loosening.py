"""Per-band epsilon loosening for the certified multi-channel method.

The matched-tolerance dyadic run gave only transient gains. The two-channel
headline result loosened the low band (eps_lo = 1.25 eps), which moves the
fixed point and gave a sustained final-iterate benefit. Here we test the
multi-channel analogue: keep the certified norm-balanced sigma schedule and
loosen the low-frequency bands' tolerances.

eps changes the constraint set (the solution), not the step-size condition, so
all runs remain certified regardless of the eps schedule.

Breast phantom, 256. Compares:
  single
  two (s=4, eps_lo=1.25)        -- the 2-band headline config
  dyadic k=4 (matched eps)      -- norm-balanced sigma, eps_i = 1
  dyadic k=4 (eps ramp ->1.25)  -- norm-balanced sigma, eps loosened toward LF
Writes a plot and an RMSE table. Delete when done.
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import dyadic_compare_methods as dc  # noqa: E402

RES = 256
K = 4
REPORT_ITERS = (10, 50, 100, 200, 300, 500)
FIG = PAPER / "experiments" / "figs"
TAB = PAPER / "experiments" / "tables"


def band_lambdas(g, project, backproject, dyf, mask, nusino, n_iter=120):
    nus2 = nusino ** 2
    out = []
    for W in dyf.band_weights_sq:
        filt = dc.fft_weight_factory(W, g.nbins)
        v = np.random.default_rng(0).standard_normal(mask.shape) * mask
        v /= np.sqrt((v ** 2).sum())
        lam = 0.0
        for _ in range(n_iter):
            sino = np.zeros((g.nviews, g.nbins))
            project(np.ascontiguousarray(v), sino)
            bp = np.zeros((g.nx, g.ny))
            backproject(filt(sino), bp)
            w = nus2 * (bp * mask)
            lam = float((v * w).sum())
            nw = np.sqrt((w ** 2).sum())
            if nw < 1e-300:
                break
            v = w / nw
        out.append(lam)
    return out


def main():
    cfg = dc.Config()
    g = dc.make_geometry(RES, cfg.larc)
    mask = dc.make_image_mask(g)
    project, backproject = dc.make_projectors(g)
    filters = dc.make_filters(g, cfg.cutoff, cfg.cutoff_lo)
    res_params = dc.RESOLUTION_PARAMS[RES]

    phimage = dc.load_phantom(cfg.image_number, RES)
    sinodata = dc.make_sinogram(phimage, project, g, cfg,
                                np.random.default_rng(dc.SEED))
    nusino, nuxgrad, nuygrad = dc.compute_normalization_constants(
        project, backproject, filters.full, mask, g, cfg)
    epssc = cfg.eps * np.sqrt(g.nviews * g.nbins)

    dyf = dc.make_dyadic_filters(g, K)
    lam = band_lambdas(g, project, backproject, dyf, mask, nusino)
    nb = tuple(lam[0] / lam[i] for i in range(K))
    eps_ramp = tuple(1.0 + 0.25 * (i / (K - 1)) for i in range(K))  # 1.0 -> 1.25
    print(f"norm-balanced sigma = {[f'{s:.2f}' for s in nb]}")
    print(f"eps ramp            = {[f'{e:.3f}' for e in eps_ramp]}")

    results = {}
    # single-channel references bracketing the dyadic per-band eps range:
    # eps=1.0 (smallest band tolerance) and eps=1.25 (largest).
    results["single (eps=1.0)"] = dc.solve_single_channel(
        sinodata, phimage, project, backproject, filters.full, mask,
        nusino, nuxgrad, nuygrad, g, res_params, cfg, epssc).ierrs
    results["single (eps=1.25)"] = dc.solve_single_channel(
        sinodata, phimage, project, backproject, filters.full, mask,
        nusino, nuxgrad, nuygrad, g, res_params, cfg, 1.25 * epssc).ierrs

    cfg_two = replace(cfg, sigma_lo_scale=4.0, eps_lo_scale=1.25)
    results["two (s=4, eps_lo=1.25)"] = dc.solve_two_channel(
        sinodata, phimage, project, backproject, filters.hi, filters.lo, mask,
        nusino, nuxgrad, nuygrad, g, res_params, cfg_two).ierrs

    cfg_m = replace(cfg, sigma_scales=nb)
    results[f"dyadic k={K} (matched eps)"] = dc.solve_multi_channel(
        sinodata, phimage, project, backproject, dyf, mask,
        nusino, nuxgrad, nuygrad, g, res_params, cfg_m).ierrs

    cfg_e = replace(cfg, sigma_scales=nb, eps_scales=eps_ramp)
    results[f"dyadic k={K} (eps ramp)"] = dc.solve_multi_channel(
        sinodata, phimage, project, backproject, dyf, mask,
        nusino, nuxgrad, nuygrad, g, res_params, cfg_e).ierrs

    s = results["single (eps=1.0)"]
    lines = [f"# Per-band eps loosening: image RMSE vs iteration (breast, {RES})\n",
             "Reductions are vs single (eps=1.0).\n",
             "| iter | " + " | ".join(results) + " |",
             "|" + "---|" * (len(results) + 1)]
    for it in REPORT_ITERS:
        row = [f"{it}"]
        for name, ierrs in results.items():
            v = ierrs[it - 1]
            row.append(f"{v:.5f}" if name == "single (eps=1.0)"
                       else f"{v:.5f} ({(s[it-1]-v)/s[it-1]*100:+.1f}%)")
        lines.append("| " + " | ".join(row) + " |")
    (TAB / "eps_loosening.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {TAB / 'eps_loosening.md'}")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    it = np.arange(1, len(s) + 1)
    styles = {"single (eps=1.0)": ("k", "-", 2.0),
              "single (eps=1.25)": ("k", "--", 1.6),
              "two (s=4, eps_lo=1.25)": ("#1f77b4", "-", 1.8),
              f"dyadic k={K} (matched eps)": ("#ff7f0e", "-", 1.6),
              f"dyadic k={K} (eps ramp)": ("#d62728", "-", 1.8)}
    for name, ierrs in results.items():
        c, ls, lw = styles[name]
        ax.semilogy(it, ierrs, ls, color=c, lw=lw, label=name)
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.set_xlim(0, len(s)); ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "eps_loosening_breast.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {FIG / 'eps_loosening_breast.png'}")


if __name__ == "__main__":
    main()
