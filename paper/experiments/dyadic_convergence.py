"""Convergence of FEW-band certified multi-channel reconstruction.

Answers: if we use a few dyadic bands, each amplified above baseline but kept
near the certified boundary (norm-balanced schedule sigma_i = lam0/lam_i), how
does convergence compare to single-channel and to the certified two-channel
(narrow near-DC band, s=4)?

Runs on the breast phantom at 256, reusing the dyadic_compare_methods solver.
Writes a convergence plot and an RMSE table. Delete when done.
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
KS = (2, 3, 4)
REPORT_ITERS = (10, 50, 100, 200, 300, 500)
FIG = PAPER / "experiments" / "figs"
TAB = PAPER / "experiments" / "tables"


def band_lambdas(g, project, backproject, dyf, mask, nusino, n_iter=120):
    """lambda_max(nusino^2 X^T F_i^2 X) per band, via power iteration."""
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
    rng = np.random.default_rng(dc.SEED)
    sinodata = dc.make_sinogram(phimage, project, g, cfg, rng)
    nusino, nuxgrad, nuygrad = dc.compute_normalization_constants(
        project, backproject, filters.full, mask, g, cfg)
    epssc = cfg.eps * np.sqrt(g.nviews * g.nbins)

    results = {}
    single = dc.solve_single_channel(
        sinodata, phimage, project, backproject, filters.full, mask,
        nusino, nuxgrad, nuygrad, g, res_params, cfg, epssc)
    results["single"] = single.ierrs

    two = dc.solve_two_channel(
        sinodata, phimage, project, backproject, filters.hi, filters.lo, mask,
        nusino, nuxgrad, nuygrad, g, res_params, cfg)
    results["two (s=4)"] = two.ierrs

    for k in KS:
        dyf = dc.make_dyadic_filters(g, k)
        lam = band_lambdas(g, project, backproject, dyf, mask, nusino)
        nb = tuple(lam[0] / lam[i] for i in range(k))
        print(f"\nk={k} norm-balanced sigma_scales = "
              f"{[f'{s:.2f}' for s in nb]}")
        cfg_k = replace(cfg, sigma_scales=nb)
        multi = dc.solve_multi_channel(
            sinodata, phimage, project, backproject, dyf, mask,
            nusino, nuxgrad, nuygrad, g, res_params, cfg_k)
        results[f"dyadic k={k}"] = multi.ierrs

    # --- table ---
    s = results["single"]
    lines = ["# Few-band certified multi-channel: image RMSE vs iteration "
             "(breast, 256)\n",
             "| iter | " + " | ".join(results) + " |",
             "|" + "---|" * (len(results) + 1)]
    for it in REPORT_ITERS:
        row = [f"{it}"]
        for name, ierrs in results.items():
            v = ierrs[it - 1]
            if name == "single":
                row.append(f"{v:.5f}")
            else:
                row.append(f"{v:.5f} ({(s[it-1]-v)/s[it-1]*100:+.1f}%)")
        lines.append("| " + " | ".join(row) + " |")
    (TAB / "dyadic_convergence.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {TAB / 'dyadic_convergence.md'}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    it = np.arange(1, len(s) + 1)
    styles = {"single": ("k", 2.0), "two (s=4)": ("#1f77b4", 1.8),
              "dyadic k=2": ("#2ca02c", 1.5), "dyadic k=3": ("#ff7f0e", 1.5),
              "dyadic k=4": ("#d62728", 1.5)}
    for name, ierrs in results.items():
        c, lw = styles.get(name, ("gray", 1.2))
        ax.semilogy(it, ierrs, "-", color=c, lw=lw, label=name)
    ax.set_xlabel("iteration"); ax.set_ylabel("image RMSE")
    ax.set_xlim(0, len(s)); ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "dyadic_convergence_breast.png", dpi=200,
                bbox_inches="tight")
    print(f"Wrote {FIG / 'dyadic_convergence_breast.png'}")


if __name__ == "__main__":
    main()
