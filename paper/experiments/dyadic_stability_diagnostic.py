"""Stability diagnostic for the DYADIC multi-channel bank.

Generalizes experiments/stability_diagnostic.py (two-channel) to k dyadic
bands, reusing the exact operators and filter bank of dyadic_compare_methods.

Question (from the manuscript / our two-channel result): the dyadic schedule
sigma_i = 2^i sigma_0 amplifies the low-frequency bands. Does that amplification
stay inside the certified region -- i.e., does it leave lambda_max(M) at the
single-channel boundary so tau is not throttled -- or does the low-frequency
tail govern the limiting eigenvalue and force tau down?

With all blocks pre-scaled as in the solver (nusino, nuxgrad, nuygrad, l1), the
dyadic data normal operator with UNIFORM weights equals the single-channel one
(partition of unity: sum_i F_i^2 = ramp). So the certified margin is

    tau*lambda_max(M) = lambda_max(Mtilde(r)) / lambda_max(Mtilde(1)),

    Mtilde(r) = nusino^2 X^T (sum_i r_i F_i^2) X
                + nuxgrad^2 grad_x^T grad_x + nuygrad^2 grad_y^T grad_y + l1^2 I,

with r_i = sigma_i/sigma_0. r_i = 1 (uniform) is the single-channel boundary
(=1); r_i = 2^i is the dyadic schedule. A value near 1 means the schedule is
amplified "for free"; a value near k means tau must shrink by ~k.

Prints a report and writes tables/dyadic_stability.md. Delete when done.
"""
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))

from dyadic_compare_methods import (   # noqa: E402
    Config, make_geometry, make_image_mask, make_projectors,
    make_dyadic_filters, compute_normalization_constants,
    gradx, grady, mdivx, mdivy, fft_weight_factory, auto_n_channels,
)

RESOLUTION = 256
NPOWER_EIG = 250
KSWEEP = (2, 3, 4, 5, 6, 7)


def lam_max(applyop, mask, n_iter=NPOWER_EIG, seed=0):
    """Largest eigenvalue (Rayleigh) and eigenvector of a symmetric PSD op."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(mask.shape) * mask
    v /= np.sqrt((v ** 2).sum())
    lam = 0.0
    for _ in range(n_iter):
        w = applyop(v)
        lam = float((v * w).sum())
        nw = np.sqrt((w ** 2).sum())
        if nw < 1e-300:
            break
        v = w / nw
    return lam, v


def main():
    cfg = Config()
    g = make_geometry(RESOLUTION, cfg.larc)
    mask = make_image_mask(g)
    project, backproject = make_projectors(g)

    k_auto = auto_n_channels(g)
    kmax = max(KSWEEP + (k_auto,))
    dyadic = make_dyadic_filters(g, kmax)
    nusino, nuxgrad, nuygrad = compute_normalization_constants(
        project, backproject, dyadic.full, mask, g, cfg)
    nus2, nux2, nuy2 = nusino ** 2, nuxgrad ** 2, nuygrad ** 2
    l1sq = cfg.l1_weight ** 2

    # per-band F_i^2 detector multipliers (already = ramp * window_i)
    Wbands = list(dyadic.band_weights_sq)            # length kmax
    band_filt = [fft_weight_factory(W, g.nbins) for W in Wbands]

    print("=" * 72)
    print(f"Dyadic multi-channel stability diagnostic  "
          f"({RESOLUTION}x{RESOLUTION}, {g.nviews} views / 50 deg)")
    print(f"analytic depth auto_n_channels = {k_auto}; "
          f"diagnosing bands up to k = {kmax}")
    print("=" * 72)

    def data_normal(f, W):
        sino = np.zeros((g.nviews, g.nbins))
        project(np.ascontiguousarray(f), sino)
        out = np.zeros((g.nx, g.ny))
        backproject(fft_weight_factory(W, g.nbins)(sino), out)
        return nus2 * (out * mask)

    def make_M(weights):
        """weights: detector multiplier for the data term (sum_i r_i F_i^2)."""
        filt = fft_weight_factory(weights, g.nbins)

        def apply(f):
            f = f * mask
            sino = np.zeros((g.nviews, g.nbins))
            project(np.ascontiguousarray(f), sino)
            bp = np.zeros((g.nx, g.ny))
            backproject(filt(sino), bp)
            out = nus2 * (bp * mask)
            out += nux2 * (mdivx(gradx(f)) * mask)
            out += nuy2 * (mdivy(grady(f)) * mask)
            out += l1sq * f
            return out
        return apply

    # --- per-band spectra and the sigma_i = 2^i schedule (for k = kmax) ---
    print(f"\nPer-band data normal operator (k = {kmax} dyadic bands):")
    print("  i | band energy % | lambda_max(K_i^T K_i) | sigma_i=2^i | "
          "sigma_i * lambda_max")
    print("  " + "-" * 70)
    band_energy = [float(W.sum()) for W in Wbands]
    tot_energy = sum(band_energy)
    band_lam = []
    for i in range(kmax):
        lam_i, _ = lam_max(lambda f, W=Wbands[i]: data_normal(f, W), mask,
                           n_iter=120)
        band_lam.append(lam_i)
        sig_i = 2.0 ** i
        print(f"  {i} | {100*band_energy[i]/tot_energy:12.3f} | "
              f"{lam_i:21.4e} | {sig_i:11.0f} | {sig_i*lam_i:18.4e}")

    # --- certified margin: dyadic schedule vs uniform, for each k ---
    ramp = Wbands[0] * 0.0
    for W in Wbands:
        ramp = ramp + W                                   # = full ramp
    lam_single, _ = lam_max(make_M(ramp), mask)           # uniform = single-ch
    print(f"\nlambda_max(Mtilde(uniform)) = ||K_single||^2 = {lam_single:.6e}")
    print("\n  k | tau*lambda_max(M) [dyadic 2^i] | certified (<=1)? | "
          "tau shrinks x")
    print("  " + "-" * 64)
    rows = []
    for k in KSWEEP:
        # dyadic schedule on the first k bands of a k-channel bank
        dk = make_dyadic_filters(g, k)
        Wk = [w for w in dk.band_weights_sq]
        Wdy = Wk[0] * 0.0
        for i in range(k):
            Wdy = Wdy + (2.0 ** i) * Wk[i]
        lam_dy, v_dy = lam_max(make_M(Wdy), mask)
        # normalize to the single-channel boundary (uniform schedule, same k)
        Wun = Wk[0] * 0.0
        for i in range(k):
            Wun = Wun + Wk[i]
        lam_un, _ = lam_max(make_M(Wun), mask)
        margin = lam_dy / lam_un
        cert = "yes" if margin <= 1.005 else "no"
        rows.append((k, margin, cert))
        print(f"  {k} | {margin:30.3f} | {cert:>15} | {margin:13.2f}")

    # --- per-band energy share of the limiting eigenmode (k = k_auto) ---
    dk = make_dyadic_filters(g, k_auto)
    Wk = list(dk.band_weights_sq)
    Wdy = Wk[0] * 0.0
    for i in range(k_auto):
        Wdy = Wdy + (2.0 ** i) * Wk[i]
    _, v = lam_max(make_M(Wdy), mask)
    sino_v = np.zeros((g.nviews, g.nbins))
    project(np.ascontiguousarray(v * mask), sino_v)
    e_band = []
    for i in range(k_auto):
        fw = fft_weight_factory(Wk[i], g.nbins)(sino_v)
        e_band.append((2.0 ** i) * nus2 * float((sino_v * fw).sum()))
    e_tvx = nux2 * float((gradx(v * mask) ** 2).sum())
    e_tvy = nuy2 * float((grady(v * mask) ** 2).sum())
    e_l1 = l1sq * float(((v * mask) ** 2).sum())
    tot = sum(e_band) + e_tvx + e_tvy + e_l1
    print(f"\nLimiting-eigenmode energy share (k = {k_auto}, dyadic 2^i):")
    print("  band i:  " + "  ".join(f"{i}:{e_band[i]/tot:5.3f}"
                                    for i in range(k_auto)))
    print(f"  tv_x {e_tvx/tot:.3f}  tv_y {e_tvy/tot:.3f}  l1 {e_l1/tot:.3f}")

    # --- norm-balanced schedule: sigma_i = lambda_max(band_0)/lambda_max(band_i)
    #     (compensates the OPERATOR NORM rather than the band energy, so each
    #     band contributes equally to lambda_max if the bands were disjoint) ---
    r_nb = [band_lam[0] / band_lam[i] for i in range(kmax)]
    print(f"\nNorm-balanced schedule sigma_i (k = {kmax}):")
    print("  " + "  ".join(f"{i}:{r_nb[i]:.2f}" for i in range(kmax)))
    print("  (vs dyadic 2^i: " + ", ".join(f"{2**i}" for i in range(kmax)) + ")")
    nb_rows = []
    for k in KSWEEP:
        dk = make_dyadic_filters(g, k)
        Wk = list(dk.band_weights_sq)
        lam_band_k = [lam_max(lambda f, W=Wk[i]: data_normal(f, W), mask,
                              n_iter=120)[0] for i in range(k)]
        Wnb = Wk[0] * 0.0
        Wun = Wk[0] * 0.0
        for i in range(k):
            Wnb = Wnb + (lam_band_k[0] / lam_band_k[i]) * Wk[i]
            Wun = Wun + Wk[i]
        lam_nb, _ = lam_max(make_M(Wnb), mask)
        lam_un, _ = lam_max(make_M(Wun), mask)
        nb_rows.append((k, lam_nb / lam_un))
    print("\n  k | tau*lambda_max(M) [norm-balanced] | certified (<=1)?")
    print("  " + "-" * 52)
    for k, m in nb_rows:
        print(f"  {k} | {m:32.3f} | {'yes' if m <= 1.005 else 'no':>15}")

    # --- write markdown ---
    out = PAPER / "experiments" / "tables" / "dyadic_stability.md"
    lines = ["# Dyadic multi-channel stability diagnostic\n",
             f"Geometry: {RESOLUTION}x{RESOLUTION}, {g.nviews} views / 50 deg, "
             f"{g.nbins} bins. Analytic depth auto_n_channels = {k_auto}.\n",
             "## Certified margin vs dyadic depth\n",
             "tau*lambda_max(M) for the schedule sigma_i = 2^i, normalized so "
             "the uniform schedule (= single-channel) is 1.\n",
             "| k | tau*lambda_max(M) | certified? |", "|---|---|---|"]
    for k, margin, cert in rows:
        lines.append(f"| {k} | {margin:.3f} | {cert} |")
    lines += ["", f"## Per-band spectrum (k = {kmax})\n",
              "| band i | energy % | lambda_max(K_i^T K_i) | sigma_i=2^i | "
              "sigma_i*lambda_max |", "|---|---|---|---|---|"]
    for i in range(kmax):
        lines.append(f"| {i} | {100*band_energy[i]/tot_energy:.2f} | "
                     f"{band_lam[i]:.3e} | {2**i} | {(2**i)*band_lam[i]:.3e} |")
    lines += ["", f"## Limiting-eigenmode energy share (k = {k_auto})\n",
              "| " + " | ".join(f"band {i}" for i in range(k_auto))
              + " | tv_x | tv_y | l1 |",
              "|" + "---|" * (k_auto + 3)]
    lines.append("| " + " | ".join(f"{e_band[i]/tot:.3f}" for i in range(k_auto))
                 + f" | {e_tvx/tot:.3f} | {e_tvy/tot:.3f} | {e_l1/tot:.3f} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
