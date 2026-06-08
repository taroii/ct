"""Diagnostic: is the two-channel acceleration certified by the sharp PDHG
condition, or does it break it?

Reuses the exact 256x256 fan-beam geometry and filters of
compare_methods_multiresolution.run_reconstruction_for_mfact(mfact=2) (the
config behind the 2D breast / Shepp-Logan slides).

The two-channel solver sets tau*sig = 1/||K_two||^2 (boundary at
sigma_lo_scale = 1) and then scales ONLY the low-frequency dual step by
sigma_lo_scale. The sharp preconditioned condition tau*lambda_max(M) < 1
with M = sum_j sigma_j K_j^T K_j therefore reduces to the ratio

    stability(s) = lambda_max( Mtilde(s) ) / lambda_max( Mtilde(1) ),

    Mtilde(s) = K_hi^T K_hi + s K_lo^T K_lo + K_tv^T K_tv + K_l1^T K_l1.

stability(1) = 1 exactly (the boundary). stability(s) > 1 means the chosen
sigma_lo_scale violates the certified condition; how far above 1 measures by
how much. If the low band were spectrally disjoint from the dominant
subspace, stability(s) would stay ~1 even for large s.

It also reports:
  - lambda_max of the hi and lo data normal operators, and the overlap
    |<v_hi, v_lo>| of their dominant image-space eigenvectors, and
  - the block-energy decomposition of Mtilde(s)'s dominant eigenvector,
    i.e. where the limiting mode actually lives.

Prints a report; writes nothing. Delete this file when done.
"""
import sys
from pathlib import Path

import numpy as np
from numpy import (arange, arcsin, cos, floor, ones, pi, sin, sqrt, tan,
                   zeros, abs as nabs)
from numpy.fft import fft, ifft, fftshift
from numpy.random import default_rng
from numba import njit

# ---------------------------------------------------------------------------
# Geometry / config -- copied verbatim from compare_methods_multiresolution
# for mfact = 2 (256x256), incl. the Shepp-Logan-slide filter settings.
# ---------------------------------------------------------------------------
MFACT = 2
nx = ny = int(512 / MFACT)            # 256
ximageside = yimageside = 10.0
dx = ximageside / nx
dy = yimageside / ny

radius = 50.0
source_to_detector = 100.0
larc = 1.0
nviews = 25
nbins = 1024
slen = (50.0 / 180.0) * pi
slen0 = -slen / 2.0
fanangle2 = arcsin((ximageside / 2.0) / radius)
detectorlength = 2.0 * tan(fanangle2) * source_to_detector

# regularizer block scalings (cm defaults)
nuxfact = 0.5
nuyfact = 0.5
l1f = 1.0

# filter cutoffs -- Shepp-Logan slide config (cm breast uses lo-scale 4)
cutoffparm = 4.0       # high channel
cutoffparm_lo = 8.0    # low channel
SIGMA_LO_SCALES = [1.0, 2.0, 4.0, 8.0]

NPOWER_NORM = 200
NPOWER_EIG = 250

# mask
_xar = arange(-ximageside / 2.0 + dx / 2, ximageside / 2.0, dx)[:, None] * ones([ny])
_yar = ones([nx, ny]) * arange(-yimageside / 2.0 + dy / 2, yimageside / 2.0, dy)
_rar = sqrt(_xar ** 2 + _yar ** 2)
mask = zeros([nx, ny])
mask[_rar <= ximageside / 2.0] = 1.0


# ---------------------------------------------------------------------------
# Fan-beam projector / backprojector -- copied from cm (globals frozen by njit)
# ---------------------------------------------------------------------------
@njit(cache=True)
def project(image, sinogram):
    x0 = -ximageside / 2.0
    y0 = -yimageside / 2.0
    u0 = -detectorlength / 2.0
    du = detectorlength / nbins
    ds = slen / (nviews - larc)
    for sindex in range(nviews):
        s = sindex * ds + slen0
        xsource = radius * cos(s)
        ysource = radius * sin(s)
        xDetCenter = (radius - source_to_detector) * cos(s)
        yDetCenter = (radius - source_to_detector) * sin(s)
        eux = -sin(s)
        euy = cos(s)
        for uindex in range(nbins):
            u = u0 + (uindex + 0.5) * du
            xbin = xDetCenter + eux * u
            ybin = yDetCenter + euy * u
            xl = x0
            yl = y0
            xdiff = xbin - xsource
            ydiff = ybin - ysource
            xad = nabs(xdiff) * dy
            yad = nabs(ydiff) * dx
            if xad > yad:
                slope = ydiff / xdiff
                travPixlen = dx * sqrt(1.0 + slope * slope)
                yIntOld = ysource + slope * (xl - xsource)
                iyOld = int(floor((yIntOld - y0) / dy))
                raysum = 0.0
                for ix in range(nx):
                    x = xl + dx * (ix + 1.0)
                    yIntercept = ysource + slope * (x - xsource)
                    iy = int(floor((yIntercept - y0) / dy))
                    if iy == iyOld:
                        if (iy >= 0) and (iy < ny):
                            raysum += travPixlen * image[ix, iy]
                    else:
                        yMid = dy * (iy if iy > iyOld else iyOld) + yl
                        ydist1 = nabs(yMid - yIntOld)
                        ydist2 = nabs(yIntercept - yMid)
                        frac1 = ydist1 / (ydist1 + ydist2)
                        frac2 = 1.0 - frac1
                        if (iyOld >= 0) and (iyOld < ny):
                            raysum += frac1 * travPixlen * image[ix, iyOld]
                        if (iy >= 0) and (iy < ny):
                            raysum += frac2 * travPixlen * image[ix, iy]
                    iyOld = iy
                    yIntOld = yIntercept
            else:
                slopeinv = xdiff / ydiff
                travPixlen = dy * sqrt(1.0 + slopeinv * slopeinv)
                xIntOld = xsource + slopeinv * (yl - ysource)
                ixOld = int(floor((xIntOld - x0) / dx))
                raysum = 0.0
                for iy in range(ny):
                    y = yl + dy * (iy + 1.0)
                    xIntercept = xsource + slopeinv * (y - ysource)
                    ix = int(floor((xIntercept - x0) / dx))
                    if ix == ixOld:
                        if (ix >= 0) and (ix < nx):
                            raysum += travPixlen * image[ix, iy]
                    else:
                        xMid = dx * (ix if ix > ixOld else ixOld) + xl
                        xdist1 = nabs(xMid - xIntOld)
                        xdist2 = nabs(xIntercept - xMid)
                        frac1 = xdist1 / (xdist1 + xdist2)
                        frac2 = 1.0 - frac1
                        if (ixOld >= 0) and (ixOld < nx):
                            raysum += frac1 * travPixlen * image[ixOld, iy]
                        if (ix >= 0) and (ix < nx):
                            raysum += frac2 * travPixlen * image[ix, iy]
                    ixOld = ix
                    xIntOld = xIntercept
            sinogram[sindex, uindex] = raysum


@njit(cache=True)
def backproject(sinogram, image):
    image.fill(0.0)
    x0 = -ximageside / 2.0
    y0 = -yimageside / 2.0
    u0 = -detectorlength / 2.0
    du = detectorlength / nbins
    ds = slen / (nviews - larc)
    for sindex in range(nviews):
        s = sindex * ds + slen0
        xsource = radius * cos(s)
        ysource = radius * sin(s)
        xDetCenter = (radius - source_to_detector) * cos(s)
        yDetCenter = (radius - source_to_detector) * sin(s)
        eux = -sin(s)
        euy = cos(s)
        for uindex in range(nbins):
            sinoval = sinogram[sindex, uindex]
            u = u0 + (uindex + 0.5) * du
            xbin = xDetCenter + eux * u
            ybin = yDetCenter + euy * u
            xl = x0
            yl = y0
            xdiff = xbin - xsource
            ydiff = ybin - ysource
            xad = nabs(xdiff) * dy
            yad = nabs(ydiff) * dx
            if xad > yad:
                slope = ydiff / xdiff
                travPixlen = dx * sqrt(1.0 + slope * slope)
                yIntOld = ysource + slope * (xl - xsource)
                iyOld = int(floor((yIntOld - y0) / dy))
                for ix in range(nx):
                    x = xl + dx * (ix + 1.0)
                    yIntercept = ysource + slope * (x - xsource)
                    iy = int(floor((yIntercept - y0) / dy))
                    if iy == iyOld:
                        if (iy >= 0) and (iy < ny):
                            image[ix, iy] += sinoval * travPixlen
                    else:
                        yMid = dy * (iy if iy > iyOld else iyOld) + yl
                        ydist1 = nabs(yMid - yIntOld)
                        ydist2 = nabs(yIntercept - yMid)
                        frac1 = ydist1 / (ydist1 + ydist2)
                        frac2 = 1.0 - frac1
                        if (iyOld >= 0) and (iyOld < ny):
                            image[ix, iyOld] += frac1 * sinoval * travPixlen
                        if (iy >= 0) and (iy < ny):
                            image[ix, iy] += frac2 * sinoval * travPixlen
                    iyOld = iy
                    yIntOld = yIntercept
            else:
                slopeinv = xdiff / ydiff
                travPixlen = dy * sqrt(1.0 + slopeinv * slopeinv)
                xIntOld = xsource + slopeinv * (yl - ysource)
                ixOld = int(floor((xIntOld - x0) / dx))
                for iy in range(ny):
                    y = yl + dy * (iy + 1.0)
                    xIntercept = xsource + slopeinv * (y - ysource)
                    ix = int(floor((xIntercept - x0) / dx))
                    if ix == ixOld:
                        if (ix >= 0) and (ix < nx):
                            image[ix, iy] += sinoval * travPixlen
                    else:
                        xMid = dx * (ix if ix > ixOld else ixOld) + xl
                        xdist1 = nabs(xMid - xIntOld)
                        xdist2 = nabs(xIntercept - xMid)
                        frac1 = xdist1 / (xdist1 + xdist2)
                        frac2 = 1.0 - frac1
                        if (ixOld >= 0) and (ixOld < nx):
                            image[ixOld, iy] += frac1 * sinoval * travPixlen
                        if (ix >= 0) and (ix < nx):
                            image[ix, iy] += frac2 * sinoval * travPixlen
                    ixOld = ix
                    xIntOld = xIntercept


# ---------------------------------------------------------------------------
# Filters (detector-frequency multipliers), gradients
# ---------------------------------------------------------------------------
b00 = -detectorlength / 2.0
db = detectorlength / nbins
uar = arange(b00 + db / 2.0, b00 + detectorlength, db) * 1.0


def hanning_window(uar, c):
    uhanp = abs(b00) / c
    han = 0.5 * (1.0 + cos(pi * uar / uhanp))
    han[abs(uar) > uhanp] = 0.0
    return han


ramp = nabs(uar)
han_lo = np.clip(hanning_window(uar, cutoffparm_lo), 0.0, 1.0)
han_hi = np.clip(1.0 - hanning_window(uar, cutoffparm), 0.0, 1.0)
# F^2 multipliers (the normal-operator weights): F_hi^2 = ramp*han_hi, etc.
W_hi2 = ramp * han_hi
W_lo2 = ramp * han_lo
W_single2 = ramp


def fft_weight(sino, W):
    imft = fft(sino, axis=1)
    pimft = (ones([nbins]) * fftshift(W)) * imft
    return ifft(pimft, axis=1).real


# gradient matrices (forward differences, as in cm)
gmatx = zeros([nx, nx])
for i in range(nx):
    gmatx[i, i] = -1.0
for i in range(nx - 1):
    gmatx[i, i + 1] = 1.0
gmaty = zeros([ny, ny])
for i in range(ny):
    gmaty[i, i] = -1.0
for i in range(ny - 1):
    gmaty[i, i + 1] = 1.0


def gradx(im):
    return gmatx @ im


def grady(im):
    return np.ascontiguousarray((gmaty @ im.T).T)


def mdivx(im):
    return gmatx.T @ im


def mdivy(im):
    return np.ascontiguousarray((gmaty.T @ im.T).T)


# ---------------------------------------------------------------------------
# Operator-norm scalings (replicates cm exactly)
# ---------------------------------------------------------------------------
def data_normal(f, W):
    """X^T diag(W) X f, masked.  W is an F^2 detector multiplier."""
    sino = zeros([nviews, nbins])
    project(np.ascontiguousarray(f), sino)
    sino = fft_weight(sino, W)
    out = zeros([nx, ny])
    backproject(np.ascontiguousarray(sino), out)
    return out * mask


def lam_max(applyop, n_iter=NPOWER_EIG, seed=0):
    """Largest eigenvalue (Rayleigh) and eigenvector of a symmetric PSD op."""
    rng = default_rng(seed)
    v = rng.standard_normal((nx, ny)) * mask
    v /= sqrt((v ** 2).sum())
    lam = 0.0
    for _ in range(n_iter):
        w = applyop(v)
        lam = float((v * w).sum())          # Rayleigh quotient
        nw = sqrt((w ** 2).sum())
        if nw < 1e-300:
            break
        v = w / nw
    return lam, v


def compute_scalings():
    # nusino = 1/||R_single|| with R_single^T R_single = X^T ramp X
    lam_s, _ = lam_max(lambda f: data_normal(f, W_single2), n_iter=NPOWER_NORM)
    nusino = 1.0 / sqrt(lam_s)
    # grad norms
    lam_gx, _ = lam_max(lambda f: mdivx(gradx(f)) * mask, n_iter=NPOWER_NORM)
    lam_gy, _ = lam_max(lambda f: mdivy(grady(f)) * mask, n_iter=NPOWER_NORM)
    nuxgrad = nuxfact / sqrt(lam_gx)
    nuygrad = nuyfact / sqrt(lam_gy)
    return nusino, nuxgrad, nuygrad


# ---------------------------------------------------------------------------
# Mtilde(s) and block-energy decomposition
# ---------------------------------------------------------------------------
def make_Mtilde(nusino, nuxgrad, nuygrad, s):
    nus2 = nusino ** 2
    nux2 = nuxgrad ** 2
    nuy2 = nuygrad ** 2

    def apply(f):
        f = f * mask
        # data hi + s*lo, combined into one weighted backprojection
        sino = zeros([nviews, nbins])
        project(np.ascontiguousarray(f), sino)
        sino = fft_weight(sino, W_hi2 + s * W_lo2)
        bp = zeros([nx, ny])
        backproject(np.ascontiguousarray(sino), bp)
        out = nus2 * (bp * mask)
        out = out + nux2 * (mdivx(gradx(f)) * mask)
        out = out + nuy2 * (mdivy(grady(f)) * mask)
        out = out + (l1f ** 2) * f
        return out

    return apply


def block_energies(f, nusino, nuxgrad, nuygrad, s):
    """Fractional contribution of each block to f^T Mtilde(s) f."""
    f = f * mask
    sino = zeros([nviews, nbins])
    project(np.ascontiguousarray(f), sino)
    e_hi = (nusino ** 2) * (fft_weight(sino, W_hi2) * sino).sum()
    e_lo = (nusino ** 2) * s * (fft_weight(sino, W_lo2) * sino).sum()
    e_tx = (nuxgrad ** 2) * (gradx(f) ** 2).sum()
    e_ty = (nuygrad ** 2) * (grady(f) ** 2).sum()
    e_l1 = (l1f ** 2) * (f ** 2).sum()
    tot = e_hi + e_lo + e_tx + e_ty + e_l1
    return {k: v / tot for k, v in
            dict(hi=e_hi, lo=e_lo, tv_x=e_tx, tv_y=e_ty, l1=e_l1).items()}


def main():
    print("=" * 70)
    print("PDHG two-channel stability diagnostic  (256x256, 25 views / 50 deg)")
    print(f"cutoff_hi={cutoffparm}, cutoff_lo={cutoffparm_lo}")
    print("=" * 70)

    print("\nComputing block operator-norm scalings (matches solver)...")
    nusino, nuxgrad, nuygrad = compute_scalings()
    print(f"  nusino = {nusino:.6e}, nuxgrad = {nuxgrad:.6e}, "
          f"nuygrad = {nuygrad:.6e}")

    # --- band spectral separation, independent of relative scalings ---
    print("\nData-band normal operators X^T F^2 X (unscaled):")
    lam_hi, v_hi = lam_max(lambda f: data_normal(f, W_hi2))
    lam_lo, v_lo = lam_max(lambda f: data_normal(f, W_lo2))
    overlap = abs(float((v_hi * v_lo).sum()))
    print(f"  lambda_max(hi) = {lam_hi:.4e}")
    print(f"  lambda_max(lo) = {lam_lo:.4e}   (ratio lo/hi = {lam_lo/lam_hi:.3f})")
    print(f"  |<v_hi, v_lo>| of dominant eigvecs = {overlap:.4f}   "
          f"(0 = disjoint dominant subspaces, 1 = aligned)")

    # --- the certified-stability ratio vs sigma_lo_scale ---
    lam_base, _ = lam_max(make_Mtilde(nusino, nuxgrad, nuygrad, 1.0))
    print(f"\nlambda_max(Mtilde(1)) = ||K_two||^2 = {lam_base:.6e}")
    print("\n  s = sigma_lo_scale | tau*lambda_max(M) | certified? | "
          "tau must shrink x")
    print("  " + "-" * 64)
    results = {}
    for s in SIGMA_LO_SCALES:
        ap = make_Mtilde(nusino, nuxgrad, nuygrad, s)
        lam_s, v_s = lam_max(ap)
        ratio = lam_s / lam_base                 # = tau*lambda_max(M)
        # within 0.5% of the single-channel boundary (=1) counts as "at budget"
        certified = "YES" if ratio < 1.005 else "NO"
        results[s] = (ratio, v_s)
        print(f"  {s:>17.1f} | {ratio:>17.4f} | {certified:>10} | "
              f"{max(ratio,1.0):>13.3f}")

    print("\nBlock-energy share of the limiting mode (dominant eigvec of "
          "Mtilde(s)):")
    print("  s     |    hi      lo     tv_x    tv_y     l1")
    print("  " + "-" * 52)
    for s in SIGMA_LO_SCALES:
        _, v_s = results[s]
        be = block_energies(v_s, nusino, nuxgrad, nuygrad, s)
        print(f"  {s:>4.1f}  | {be['hi']:>6.3f}  {be['lo']:>6.3f}  "
              f"{be['tv_x']:>6.3f}  {be['tv_y']:>6.3f}  {be['l1']:>6.3f}")

    print("\nReading:")
    print("  * tau*lambda_max(M) > 1  => the certified condition is violated;")
    print("    the solver was running unstable PDHG (benefit is a transient).")
    print("  * If it stays ~1 even at large s, the low band is spectrally")
    print("    disjoint and the acceleration is in fact certified.")
    print("  * 'lo' energy share shows whether the low band drives the")
    print("    limiting eigenmode (it does <=> raising sigma_lo costs tau).")

    # --- write markdown table for the paper ---
    out = Path(__file__).resolve().parent / "tables" / "stability.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# PDHG two-channel stability diagnostic\n")
    lines.append(f"Geometry: {nx}x{ny} image, {nviews} views / 50 deg arc, "
                 f"{nbins} detector bins. Filters: Hann^(1/2) cutoff_hi="
                 f"{cutoffparm}, cutoff_lo={cutoffparm_lo}.\n")
    lines.append("## Band separation (operator-level, phantom-independent)\n")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| lambda_max(X^T F_hi^2 X) | {lam_hi:.4f} |")
    lines.append(f"| lambda_max(X^T F_lo^2 X) | {lam_lo:.4f} |")
    lines.append(f"| ratio lo/hi | {lam_lo/lam_hi:.3f} |")
    lines.append(f"| dominant-eigvec overlap \\|<v_hi,v_lo>\\| | {overlap:.4f} |\n")
    lines.append("## Sharp condition vs sigma_lo_scale\n")
    lines.append("tau*lambda_max(M) = lambda_max(Mtilde(s)) / lambda_max(Mtilde(1)); "
                 "s=1 is the single-channel boundary (=1).\n")
    lines.append("| sigma_lo_scale | tau*lambda_max(M) | certified (<1)? | "
                 "lo energy share of limiting mode |")
    lines.append("|---|---|---|---|")
    for s in SIGMA_LO_SCALES:
        ratio, v_s = results[s]
        be = block_energies(v_s, nusino, nuxgrad, nuygrad, s)
        cert = "yes" if ratio < 1.005 else "no"
        lines.append(f"| {s:.0f} | {ratio:.4f} | {cert} | {be['lo']:.3f} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
