"""Multi-resolution DTV reconstruction: single-channel vs. two-channel.

The two-channel method splits the data fidelity into a low- and a high-
frequency residual via Hanning-windowed ramp filters, then solves the
resulting product-space saddle-point with the Chambolle-Pock primal-dual
algorithm. The single-channel method uses the same ramp filter
(unsplit) for comparison.

The two channels keep independent dual step sizes (``sigma_hi``,
``sigma_lo``); the primal step ``tau`` is then forced to satisfy

    tau * || diag(sqrt(sigma_i)) * K ||_2^2 < 1

which is the product-space CP convergence condition with diagonal dual
preconditioning. This is implemented as written -- the two-channel run
is not collapsed to a scalar-sigma single-channel form, so the extra
degree of freedom in the dual step is preserved.

Reference: Sidky, Jorgensen, Pan 2012, Phys. Med. Biol. 57, 3065
(Chambolle-Pock primal-dual for CT).
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.fft import fft, fftshift, ifft

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
ROOT = Path(".")
DATA_DIR = ROOT / "data" / "phantoms_from_paper"
RESULTS_DIR = ROOT / "final_figures"
CACHE_DIR = ROOT / "cache"
CACHE_FILE = CACHE_DIR / "multiresolution_results.pkl"
ISTOPS: frozenset[int] = frozenset({1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500})


@dataclass(frozen=True, slots=True)
class ResolutionParams:
    """Per-resolution DTV directional weights and iteration count.

    `alpha` controls the y-gradient ball radius (and 2-alpha the x-gradient),
    `beta` weights the L1 prior. These are inherited from the original
    multiresolution sweep and are not changed here.
    """

    alpha: float
    beta: float
    itermax: int = 500


@dataclass(frozen=True, slots=True)
class Config:
    image_number: int = 3
    add_noise: bool = False
    photon_count: float = 1e6
    nux_factor: float = 0.5
    nuy_factor: float = 0.5
    l1_weight: float = 1.0
    eps: float = 0.001
    larc: float = 1.0
    rho: float = 1.75

    # Filter cutoffs for the band split. cutoff_lo > cutoff means the high
    # band starts attenuating before the low band ends -- there's overlap
    # by construction (sqrt(han_lo) + sqrt(1-han_lo) > 1 in general).
    cutoff: float = 4.0
    cutoff_lo: float = 8.0

    # tau/sigma split. Both solvers use sigma = step_balance / ||K||
    # and tau s.t. tau * sigma * ||K||^2 = 1/(1+slack) < 1 (CP condition).
    # The two-channel run gets its own knob so we can sweep it independently.
    step_balance_single: float = 100.0
    step_balance_two: float = 100.0
    cvg_slack: float = 1e-3

    # Two-channel preconditioner: sigma_lo = sigma_lo_scale * sigma_hi, etc.
    # These multiply the *base* sigma_hi to give per-block dual steps; the
    # CP condition then constrains tau given the weighted operator norm.
    # NB: sigma_*_scale > 1 inflates ||Sigma^(1/2) K|| and *forces tau down*,
    # which slows per-iteration convergence. To genuinely change the answer
    # (favor LF accuracy), use eps_lo_scale instead -- that tightens the
    # band's data-fidelity tolerance, which moves the fixed point.
    sigma_lo_scale: float = 4.0
    sigma_tv_scale: float = 1.0
    sigma_l1_scale: float = 1.0

    # Per-band data-fidelity tolerances. The CP iteration projects the
    # band's dual residual onto an L2 ball of radius eps_band * sqrt(nrays);
    # smaller radius = tighter fit on that band.
    # Defaults are 1.0 / 1.0 (matched), preserving the "data-fit at level eps"
    # interpretation. Original script had eps_hi_scale=1.0, eps_lo_scale=1.25.
    eps_hi_scale: float = 1.0
    eps_lo_scale: float = 1.0

    # Resolutions (image side length in pixels) to run. Native phantom is
    # 512x512; smaller resolutions decimate, larger ones upsample by
    # Kronecker. Each must appear in RESOLUTION_PARAMS.
    resolutions: tuple[int, ...] = (512, 256, 128)

    # ---- Multi-channel (dyadic) data fidelity ----
    # Number of dyadic bands. If 0, choose analytically from detector
    # geometry via auto_n_channels(); if >= 2, use that many channels.
    # Set to 1 to skip the multichannel run entirely.
    n_channels: int = 0
    # Per-channel sigma scaling. By default sigma_i = 2^i * sigma_0,
    # giving stronger dual emphasis to lower-frequency bands. Override
    # with an explicit tuple of length n_channels for custom scaling.
    sigma_scales: tuple[float, ...] = ()  # () => default 2^i
    sigma_dyadic_base: float = 2.0  # base for the default 2^i schedule
    # Per-channel eps scaling. () => all 1.0 (matched tolerances).
    eps_scales: tuple[float, ...] = ()
    step_balance_multi: float = 100.0


RESOLUTION_PARAMS: dict[int, ResolutionParams] = {
    128: ResolutionParams(alpha=1.95, beta=10.0),
    256: ResolutionParams(alpha=1.9, beta=10.0),
    512: ResolutionParams(alpha=1.7, beta=5.0),
    # 1024: phantom is upsampled from 512 by Kronecker x2; alpha extrapolated
    # from the 256->512 trend (decreasing). beta likewise. Tune as needed.
    1024: ResolutionParams(alpha=1.5, beta=2.5),
}
PLOT_ORDER: tuple[int, ...] = (1024, 512, 256, 128)


# ---------------------------------------------------------------------------
# Geometry & operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FanbeamGeometry:
    nx: int
    ny: int
    nviews: int
    nbins: int
    ximageside: float
    yimageside: float
    radius: float
    source_to_detector: float
    detectorlength: float
    slen: float
    slen0: float
    larc: float

    @property
    def dx(self) -> float:
        return self.ximageside / self.nx

    @property
    def dy(self) -> float:
        return self.yimageside / self.ny


def make_geometry(resolution: int, larc: float) -> FanbeamGeometry:
    ximageside = yimageside = 10.0
    radius = 50.0
    source_to_detector = 100.0
    nviews = 25
    nbins = 1024
    slen = (50.0 / 180.0) * np.pi
    detectorlength = (
        2.0 * np.tan(np.arcsin((ximageside / 2.0) / radius)) * source_to_detector
    )
    return FanbeamGeometry(
        nx=resolution,
        ny=resolution,
        nviews=nviews,
        nbins=nbins,
        ximageside=ximageside,
        yimageside=yimageside,
        radius=radius,
        source_to_detector=source_to_detector,
        detectorlength=detectorlength,
        slen=slen,
        slen0=-slen / 2.0,
        larc=larc,
    )


def make_image_mask(g: FanbeamGeometry) -> np.ndarray:
    x = np.arange(-g.ximageside / 2.0 + g.dx / 2.0, g.ximageside / 2.0, g.dx)
    y = np.arange(-g.yimageside / 2.0 + g.dy / 2.0, g.yimageside / 2.0, g.dy)
    rar = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    return (rar <= g.ximageside / 2.0).astype(np.float64)


# ---------------------------------------------------------------------------
# Numba projectors. Module-level so they JIT-compile once across resolutions.
# Keeping the original ray-trace code path unchanged.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _project(
    image: np.ndarray,
    sinogram: np.ndarray,
    nx: int,
    ny: int,
    nviews: int,
    nbins: int,
    ximageside: float,
    yimageside: float,
    radius: float,
    source_to_detector: float,
    detectorlength: float,
    slen: float,
    slen0: float,
    larc: float,
) -> None:
    dx = ximageside / nx
    dy = yimageside / ny
    x0 = -ximageside / 2.0
    y0 = -yimageside / 2.0
    u0 = -detectorlength / 2.0
    du = detectorlength / nbins
    ds = slen / (nviews - larc)

    for sindex in range(nviews):
        s = sindex * ds + slen0
        xsource = radius * np.cos(s)
        ysource = radius * np.sin(s)
        xDetCenter = (radius - source_to_detector) * np.cos(s)
        yDetCenter = (radius - source_to_detector) * np.sin(s)
        eux = -np.sin(s)
        euy = np.cos(s)

        for uindex in range(nbins):
            u = u0 + (uindex + 0.5) * du
            xbin = xDetCenter + eux * u
            ybin = yDetCenter + euy * u
            xl, yl = x0, y0
            xdiff = xbin - xsource
            ydiff = ybin - ysource
            xad = abs(xdiff) * dy
            yad = abs(ydiff) * dx

            if xad > yad:
                slope = ydiff / xdiff
                travPixlen = dx * np.sqrt(1.0 + slope * slope)
                yIntOld = ysource + slope * (xl - xsource)
                iyOld = int(np.floor((yIntOld - y0) / dy))
                raysum = 0.0
                for ix in range(nx):
                    xcurr = xl + dx * (ix + 1.0)
                    yIntercept = ysource + slope * (xcurr - xsource)
                    iy = int(np.floor((yIntercept - y0) / dy))
                    if iy == iyOld:
                        if 0 <= iy < ny:
                            raysum += travPixlen * image[ix, iy]
                    else:
                        yMid = dy * (iy if iy > iyOld else iyOld) + yl
                        ydist1 = abs(yMid - yIntOld)
                        ydist2 = abs(yIntercept - yMid)
                        frac1 = ydist1 / (ydist1 + ydist2)
                        frac2 = 1.0 - frac1
                        if 0 <= iyOld < ny:
                            raysum += frac1 * travPixlen * image[ix, iyOld]
                        if 0 <= iy < ny:
                            raysum += frac2 * travPixlen * image[ix, iy]
                    iyOld = iy
                    yIntOld = yIntercept
            else:
                slopeinv = xdiff / ydiff
                travPixlen = dy * np.sqrt(1.0 + slopeinv * slopeinv)
                xIntOld = xsource + slopeinv * (yl - ysource)
                ixOld = int(np.floor((xIntOld - x0) / dx))
                raysum = 0.0
                for iy in range(ny):
                    ycurr = yl + dy * (iy + 1.0)
                    xIntercept = xsource + slopeinv * (ycurr - ysource)
                    ix = int(np.floor((xIntercept - x0) / dx))
                    if ix == ixOld:
                        if 0 <= ix < nx:
                            raysum += travPixlen * image[ix, iy]
                    else:
                        xMid = dx * (ix if ix > ixOld else ixOld) + xl
                        xdist1 = abs(xMid - xIntOld)
                        xdist2 = abs(xIntercept - xMid)
                        frac1 = xdist1 / (xdist1 + xdist2)
                        frac2 = 1.0 - frac1
                        if 0 <= ixOld < nx:
                            raysum += frac1 * travPixlen * image[ixOld, iy]
                        if 0 <= ix < nx:
                            raysum += frac2 * travPixlen * image[ix, iy]
                    ixOld = ix
                    xIntOld = xIntercept
            sinogram[sindex, uindex] = raysum


@njit(cache=True)
def _backproject(
    sinogram: np.ndarray,
    image: np.ndarray,
    nx: int,
    ny: int,
    nviews: int,
    nbins: int,
    ximageside: float,
    yimageside: float,
    radius: float,
    source_to_detector: float,
    detectorlength: float,
    slen: float,
    slen0: float,
    larc: float,
) -> None:
    dx = ximageside / nx
    dy = yimageside / ny
    image.fill(0.0)
    x0 = -ximageside / 2.0
    y0 = -yimageside / 2.0
    u0 = -detectorlength / 2.0
    du = detectorlength / nbins
    ds = slen / (nviews - larc)

    for sindex in range(nviews):
        s = sindex * ds + slen0
        xsource = radius * np.cos(s)
        ysource = radius * np.sin(s)
        xDetCenter = (radius - source_to_detector) * np.cos(s)
        yDetCenter = (radius - source_to_detector) * np.sin(s)
        eux = -np.sin(s)
        euy = np.cos(s)

        for uindex in range(nbins):
            sinoval = sinogram[sindex, uindex]
            u = u0 + (uindex + 0.5) * du
            xbin = xDetCenter + eux * u
            ybin = yDetCenter + euy * u
            xl, yl = x0, y0
            xdiff = xbin - xsource
            ydiff = ybin - ysource
            xad = abs(xdiff) * dy
            yad = abs(ydiff) * dx

            if xad > yad:
                slope = ydiff / xdiff
                travPixlen = dx * np.sqrt(1.0 + slope * slope)
                yIntOld = ysource + slope * (xl - xsource)
                iyOld = int(np.floor((yIntOld - y0) / dy))
                for ix in range(nx):
                    xcurr = xl + dx * (ix + 1.0)
                    yIntercept = ysource + slope * (xcurr - xsource)
                    iy = int(np.floor((yIntercept - y0) / dy))
                    if iy == iyOld:
                        if 0 <= iy < ny:
                            image[ix, iy] += sinoval * travPixlen
                    else:
                        yMid = dy * (iy if iy > iyOld else iyOld) + yl
                        ydist1 = abs(yMid - yIntOld)
                        ydist2 = abs(yIntercept - yMid)
                        frac1 = ydist1 / (ydist1 + ydist2)
                        frac2 = 1.0 - frac1
                        if 0 <= iyOld < ny:
                            image[ix, iyOld] += frac1 * sinoval * travPixlen
                        if 0 <= iy < ny:
                            image[ix, iy] += frac2 * sinoval * travPixlen
                    iyOld = iy
                    yIntOld = yIntercept
            else:
                slopeinv = xdiff / ydiff
                travPixlen = dy * np.sqrt(1.0 + slopeinv * slopeinv)
                xIntOld = xsource + slopeinv * (yl - ysource)
                ixOld = int(np.floor((xIntOld - x0) / dx))
                for iy in range(ny):
                    ycurr = yl + dy * (iy + 1.0)
                    xIntercept = xsource + slopeinv * (ycurr - ysource)
                    ix = int(np.floor((xIntercept - x0) / dx))
                    if ix == ixOld:
                        if 0 <= ix < nx:
                            image[ix, iy] += sinoval * travPixlen
                    else:
                        xMid = dx * (ix if ix > ixOld else ixOld) + xl
                        xdist1 = abs(xMid - xIntOld)
                        xdist2 = abs(xIntercept - xMid)
                        frac1 = xdist1 / (xdist1 + xdist2)
                        frac2 = 1.0 - frac1
                        if 0 <= ixOld < nx:
                            image[ixOld, iy] += frac1 * sinoval * travPixlen
                        if 0 <= ix < nx:
                            image[ix, iy] += frac2 * sinoval * travPixlen
                    ixOld = ix
                    xIntOld = xIntercept


def make_projectors(
    g: FanbeamGeometry,
) -> tuple[Callable[[np.ndarray, np.ndarray], None], Callable[[np.ndarray, np.ndarray], None]]:
    """Bind the geometry into closures for ergonomic projection calls."""

    def project(image: np.ndarray, sino: np.ndarray) -> None:
        _project(
            image, sino, g.nx, g.ny, g.nviews, g.nbins,
            g.ximageside, g.yimageside, g.radius, g.source_to_detector,
            g.detectorlength, g.slen, g.slen0, g.larc,
        )

    def backproject(sino: np.ndarray, image: np.ndarray) -> None:
        _backproject(
            sino, image, g.nx, g.ny, g.nviews, g.nbins,
            g.ximageside, g.yimageside, g.radius, g.source_to_detector,
            g.detectorlength, g.slen, g.slen0, g.larc,
        )

    return project, backproject


# ---------------------------------------------------------------------------
# Finite-difference gradients (no NxN matrices)
# ---------------------------------------------------------------------------


def gradx(image: np.ndarray) -> np.ndarray:
    """Forward-difference x-gradient with -I outside the last row.

    Equivalent to (gmatx @ image) where gmatx = -I + shift(+1) along axis 0,
    but without allocating an NxN dense matrix.
    """
    out = np.empty_like(image)
    out[:-1, :] = image[1:, :] - image[:-1, :]
    out[-1, :] = -image[-1, :]
    return out


def grady(image: np.ndarray) -> np.ndarray:
    out = np.empty_like(image)
    out[:, :-1] = image[:, 1:] - image[:, :-1]
    out[:, -1] = -image[:, -1]
    return out


def mdivx(image: np.ndarray) -> np.ndarray:
    """Adjoint of gradx, equal to gmatx.T @ image."""
    out = np.empty_like(image)
    out[0, :] = -image[0, :]
    out[1:, :] = image[:-1, :] - image[1:, :]
    return out


def mdivy(image: np.ndarray) -> np.ndarray:
    out = np.empty_like(image)
    out[:, 0] = -image[:, 0]
    out[:, 1:] = image[:, :-1] - image[:, 1:]
    return out


def gradim(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same as (gradx, grady) -- separate function preserved for TV reporting."""
    return gradx(image), grady(image)


# ---------------------------------------------------------------------------
# Filtering: ramp + Hanning band split
# ---------------------------------------------------------------------------


def hanning_window(u_coords: np.ndarray, start: float, cutoff: float) -> np.ndarray:
    uhanp = abs(start) / cutoff
    out = 0.5 * (1.0 + np.cos(np.pi * u_coords / uhanp))
    out[np.abs(u_coords) > uhanp] = 0.0
    return out


def fft_weight_factory(weights: np.ndarray, nbins: int) -> Callable[[np.ndarray], np.ndarray]:
    shifted = fftshift(weights)

    def apply(sino: np.ndarray) -> np.ndarray:
        return ifft(shifted * fft(sino, axis=1), axis=1).real

    return apply


class BandFilters(NamedTuple):
    full: Callable[[np.ndarray], np.ndarray]
    hi: Callable[[np.ndarray], np.ndarray]
    lo: Callable[[np.ndarray], np.ndarray]


def make_filters(g: FanbeamGeometry, cutoff: float, cutoff_lo: float) -> BandFilters:
    db = g.detectorlength / g.nbins
    b00 = -g.detectorlength / 2.0
    uar = np.arange(b00 + db / 2.0, b00 + g.detectorlength, db)
    w_sqrt_ramp = np.sqrt(np.abs(uar) + 1e-12)
    han_lo = np.clip(hanning_window(uar, b00, cutoff_lo), 0.0, 1.0)
    han_hi = np.clip(1.0 - hanning_window(uar, b00, cutoff), 0.0, 1.0)
    return BandFilters(
        full=fft_weight_factory(w_sqrt_ramp, g.nbins),
        hi=fft_weight_factory(w_sqrt_ramp * np.sqrt(han_hi), g.nbins),
        lo=fft_weight_factory(w_sqrt_ramp * np.sqrt(han_lo), g.nbins),
    )


# ---------------------------------------------------------------------------
# Dyadic multi-channel filter bank
#
# Channel 0 covers the top half of the detector spectrum |u| in [|b00|/2, |b00|];
# channel i (0 < i < k-1) covers the dyadic shell [|b00|/2^(i+1), |b00|/2^i];
# channel k-1 is the closed LF tail [0, |b00|/2^(k-1)].
#
# Each filter is sqrt(|u|) * sqrt(H_i - H_{i+1}) where H_i = hanning(cutoff=2^i)
# is the Hanning window with half-width |b00|/2^i. The differences telescope so
# sum_i |f_i|^2 = |u| = |f_single|^2 to machine precision (partition of unity).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DyadicFilters:
    bands: tuple[Callable[[np.ndarray], np.ndarray], ...]
    full: Callable[[np.ndarray], np.ndarray]
    band_weights_sq: tuple[np.ndarray, ...]
    u_coords: np.ndarray

    def __len__(self) -> int:
        return len(self.bands)


def auto_n_channels(g: FanbeamGeometry, min_bins_per_band: int = 8) -> int:
    """Pick the dyadic depth analytically from the detector geometry.

    Each successive dyadic band halves in bandwidth. The lowest band has
    half-width |b00|/2^(k-1), which corresponds to roughly
    nbins / 2^k Fourier bins on the detector grid. To keep the Hanning
    band-pass numerically well-resolved we want this >= min_bins_per_band:

        nbins / 2^k >= min_bins_per_band
        => k <= log2(nbins / min_bins_per_band)

    Going deeper than this puts the LF tail inside the discretization noise
    floor -- the band carries only a handful of Fourier bins and the
    Hanning shape is no longer faithfully represented. With nbins=1024 and
    the default min_bins_per_band=8, this gives k=7.

    This is purely a geometry property, independent of image resolution.
    The image grid does set an *upper bound* on what is "useful" LF detail
    (one cycle across the image), but in fan-beam CT with oversampled
    detectors the detector bound is hit first.
    """
    k = int(np.floor(np.log2(g.nbins / min_bins_per_band)))
    return max(2, k)


def make_dyadic_filters(g: FanbeamGeometry, n_channels: int) -> DyadicFilters:
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")
    db = g.detectorlength / g.nbins
    b00 = -g.detectorlength / 2.0
    uar = np.arange(b00 + db / 2.0, b00 + g.detectorlength, db)
    w_sqrt_ramp = np.sqrt(np.abs(uar) + 1e-12)

    # Hannings H_i = hanning(cutoff=2^i), half-widths |b00|/2^i, i=0..k-1.
    hannings: list[np.ndarray] = []
    for i in range(n_channels):
        cutoff_i = float(2 ** i)
        H_i = np.clip(hanning_window(uar, b00, cutoff_i), 0.0, 1.0)
        hannings.append(H_i)

    # Band-pass windows: telescoping differences for partition of unity.
    band_weights_sq_list: list[np.ndarray] = []
    band_filters: list[Callable[[np.ndarray], np.ndarray]] = []
    for i in range(n_channels):
        if i == 0:
            # Top band: 1 - H_1 (the part of the spectrum H_1 does not cover)
            pass_window = (np.ones_like(uar) if n_channels == 1
                           else np.clip(1.0 - hannings[1], 0.0, 1.0))
        elif i == n_channels - 1:
            # LF tail: closed at zero, no further subtraction
            pass_window = hannings[i]
        else:
            pass_window = np.clip(hannings[i] - hannings[i + 1], 0.0, 1.0)
        weight = w_sqrt_ramp * np.sqrt(pass_window)
        band_weights_sq_list.append(weight ** 2)
        band_filters.append(fft_weight_factory(weight, g.nbins))

    return DyadicFilters(
        bands=tuple(band_filters),
        full=fft_weight_factory(w_sqrt_ramp, g.nbins),
        band_weights_sq=tuple(band_weights_sq_list),
        u_coords=uar,
    )


# ---------------------------------------------------------------------------
# Power iteration helpers (operator norm estimates)
# ---------------------------------------------------------------------------


def _norm_pair(mag1: float, mag2: float) -> float:
    return 0.5 * (mag1 + mag2)


def operator_norm_AtA(
    apply_AAt: Callable[[np.ndarray], np.ndarray],
    nx: int,
    ny: int,
    n_iter: int = 50,
    rng: np.random.Generator | None = None,
    mask: np.ndarray | None = None,
) -> float:
    """Power iteration for sqrt of largest eigenvalue of A^T A applied to images.

    `apply_AAt` should compute A^T A v in image space (and apply mask if needed).
    """
    rng = rng or np.random.default_rng(SEED)
    v = rng.standard_normal((nx, ny))
    if mask is not None:
        v *= mask
    last = 0.0
    for _ in range(n_iter):
        v = apply_AAt(v)
        last = float(np.sqrt((v ** 2).sum()))
        if last > 0:
            v /= last + 1e-12
    return 1.0 / np.sqrt(last + 1e-12)


# ---------------------------------------------------------------------------
# Solver helpers
# ---------------------------------------------------------------------------


def prox_l2_ball(y: np.ndarray, radius: float) -> np.ndarray:
    """Project y onto the L2 ball of given radius."""
    if radius <= 0.0:
        return np.zeros_like(y)
    norm = np.sqrt((y ** 2).sum())
    if norm <= radius or norm == 0.0:
        return y
    return y * (radius / norm)


def prox_residual_ball(
    y: np.ndarray, sigma_eps_factor: float
) -> np.ndarray:
    """Shrink y toward zero by `sigma_eps_factor` (the original code's ε-ball
    prox on dual residuals)."""
    ymag = np.sqrt((y ** 2).sum())
    if ymag <= 0.0:
        return y
    return y * max(ymag - sigma_eps_factor, 0.0) / ymag


def prox_dtv_dual(
    p_til: np.ndarray, weight: float
) -> np.ndarray:
    """DTV dual prox: clamp |p| to `weight` componentwise. Returns
    weight * p_til / max(|p_til|, weight)."""
    return weight * p_til / np.maximum(np.abs(p_til), weight)


# ---------------------------------------------------------------------------
# Single-channel CP solver
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StepSizes:
    sigma: float
    tau: float
    norm: float

    def cp_lhs(self) -> float:
        return self.tau * self.sigma * self.norm ** 2


@dataclass(slots=True)
class SolverResult:
    image: np.ndarray
    ierrs: list[float]
    derrs: list[float]
    tvs: list[float]
    elapsed: float
    steps: dict
    # Two-channel only: per-band data residuals. None for single-channel.
    derrs_hi: list[float] | None = None
    derrs_lo: list[float] | None = None
    # Multichannel only: list of length n_channels of per-band residuals.
    derrs_bands: list[list[float]] | None = None


def _operator_norm_single(
    project: Callable, backproject: Callable,
    filt_full: Callable, mask: np.ndarray, nusino: float,
    nuxgrad: float, nuygrad: float, l1_weight: float,
    g: FanbeamGeometry, n_iter: int = 200,
) -> float:
    rng = np.random.default_rng(SEED)
    xim = rng.standard_normal((g.nx, g.ny)) * mask
    worksino = np.zeros((g.nviews, g.nbins))
    xim1 = np.zeros_like(xim)
    mag1 = mag2 = 0.0

    for _ in range(n_iter):
        project(xim, worksino)
        worksino_f = filt_full(worksino) * nusino
        xg = gradx(xim) * nuxgrad
        yg = grady(xim) * nuygrad
        yim_l1 = l1_weight * xim
        mag1 = float(np.sqrt(
            (yim_l1 ** 2).sum() + (yg ** 2).sum() + (xg ** 2).sum() + (worksino_f ** 2).sum()
        ))
        if mag1 > 0:
            yim_l1 /= mag1
            yg /= mag1
            xg /= mag1
            worksino_f /= mag1

        xim1.fill(0.0)
        backproject(filt_full(worksino_f), xim1)
        xim = (
            xim1 * (nusino * mask)
            + mdivx(xg) * (nuxgrad * mask)
            + mdivy(yg) * (nuygrad * mask)
            + l1_weight * yim_l1
        )
        mag2 = float(np.sqrt((xim ** 2).sum()))
        if mag2 > 0:
            xim /= mag2

    return _norm_pair(mag1, mag2)


def solve_single_channel(
    sinodata: np.ndarray,
    phimage: np.ndarray,
    project: Callable,
    backproject: Callable,
    filt_full: Callable,
    mask: np.ndarray,
    nusino: float,
    nuxgrad: float,
    nuygrad: float,
    g: FanbeamGeometry,
    res_params: ResolutionParams,
    cfg: Config,
    epssc: float,
) -> SolverResult:
    print("\nRunning single-channel...")
    sinodatasc = nusino * filt_full(sinodata)

    norm = _operator_norm_single(
        project, backproject, filt_full, mask, nusino,
        nuxgrad, nuygrad, cfg.l1_weight, g,
    )
    # tau * sigma * norm^2 = 1 / (1+slack) < 1   (CP condition)
    sigma = cfg.step_balance_single / norm
    tau = 1.0 / (norm * cfg.step_balance_single * (1.0 + cfg.cvg_slack))
    steps = StepSizes(sigma=sigma, tau=tau, norm=norm)
    print(
        f"  ||K||={norm:.4f}  sigma={sigma:.6f}  tau={tau:.6f}  "
        f"tau*sigma*||K||^2={steps.cp_lhs():.6f} (< 1 required)"
    )
    assert steps.cp_lhs() < 1.0, "Single-channel CP condition violated"

    nx, ny = g.nx, g.ny
    xim = np.zeros((nx, ny))
    yim_l1 = np.zeros_like(xim)
    xbarim = np.zeros_like(xim)
    ysino = np.zeros((g.nviews, g.nbins))
    ygradx = np.zeros_like(xim)
    ygrady = np.zeros_like(xim)

    ierrs: list[float] = []
    derrs: list[float] = []
    tvs: list[float] = []
    alpha, beta, itermax = res_params.alpha, res_params.beta, res_params.itermax
    start_time = time.time()

    for itr in range(1, itermax + 1):
        ysino_old = ysino.copy()
        ygradx_old = ygradx.copy()
        ygrady_old = ygrady.copy()
        yim_old = yim_l1.copy()

        wimp = np.zeros_like(xim)
        backproject(filt_full(ysino), wimp)
        wimp *= nusino * mask
        wimqx = mdivx(ygradx) * (nuxgrad * mask)
        wimqy = mdivy(ygrady) * (nuygrad * mask)
        wiml1 = cfg.l1_weight * yim_l1

        ximold = xim.copy()
        xim = xim - tau * (wimp + wimqx + wimqy + wiml1)
        np.maximum(xim, 0.0, out=xim)
        xbarim = 2.0 * xim - ximold

        worksino = np.zeros((g.nviews, g.nbins))
        project(xbarim, worksino)
        resid = nusino * filt_full(worksino) - sinodatasc
        derrs.append(float(np.sqrt((resid ** 2).sum()) / nusino / np.sqrt(g.nviews * g.nbins)))

        ysino = prox_residual_ball(ysino + sigma * resid, sigma * nusino * epssc)

        ptilx = ygradx + sigma * (gradx(xbarim) * nuxgrad)
        ygradx = prox_dtv_dual(ptilx, 2.0 - alpha)

        ptily = ygrady + sigma * (grady(xbarim) * nuygrad)
        ygrady = prox_dtv_dual(ptily, alpha)

        ptil1 = yim_l1 + sigma * (cfg.l1_weight * xbarim)
        yim_l1 = prox_dtv_dual(ptil1, beta)

        tgx, tgy = gradim(xbarim)
        tvs.append(float(np.sqrt(tgx ** 2 + tgy ** 2).sum()))

        # Over-relaxation step
        ygradx = ygradx_old - cfg.rho * (ygradx_old - ygradx)
        ygrady = ygrady_old - cfg.rho * (ygrady_old - ygrady)
        ysino = ysino_old - cfg.rho * (ysino_old - ysino)
        yim_l1 = yim_old - cfg.rho * (yim_old - yim_l1)
        xim = ximold - cfg.rho * (ximold - xim)

        ierrs.append(float(np.sqrt(((xbarim - phimage) ** 2).sum() / (nx * ny))))
        if itr in ISTOPS:
            print(
                f"  Iter {itr}: data_err={derrs[-1]:.6f}, "
                f"img_err={ierrs[-1]:.6f}, TV={tvs[-1]:.2f}"
            )

    return SolverResult(
        image=xbarim.copy(), ierrs=ierrs, derrs=derrs, tvs=tvs,
        elapsed=time.time() - start_time, steps=dict(sigma=sigma, tau=tau, norm=norm),
    )


# ---------------------------------------------------------------------------
# Two-channel CP solver
# ---------------------------------------------------------------------------


def _operator_norm_two(
    project: Callable, backproject: Callable,
    filt_hi: Callable, filt_lo: Callable, mask: np.ndarray, nusino: float,
    nuxgrad: float, nuygrad: float, l1_weight: float,
    sqrt_lo: float, sqrt_tv: float, sqrt_l1: float,
    g: FanbeamGeometry, n_iter: int = 200,
) -> float:
    """Returns ||Sigma^(1/2) K|| / sqrt(sig_hi) = sqrt(lambda_max(Mtilde)),
    where Mtilde = K_hi^T K_hi + r_lo K_lo^T K_lo + r_tv (gx^T gx + gy^T gy)
    + r_l1 I is the sig_hi-factored weighted normal operator (r_* = sig_*/sig_hi).

    Computed as a symmetric power iteration on Mtilde. The data term needs the
    squared band filter F^2 = X^T R^2 X, so each filter is applied TWICE
    (forward and adjoint) -- consistent with _operator_norm_single. Applying it
    once (as a previous version did) computes X^T R X and overestimates the
    norm, throttling the step sizes.
    """
    r_lo, r_tv, r_l1 = sqrt_lo ** 2, sqrt_tv ** 2, sqrt_l1 ** 2
    nus2, nux2, nuy2, l1sq = nusino ** 2, nuxgrad ** 2, nuygrad ** 2, l1_weight ** 2
    rng = np.random.default_rng(SEED)
    v = rng.standard_normal((g.nx, g.ny)) * mask
    v /= np.sqrt((v ** 2).sum()) + 1e-12
    worksino = np.zeros((g.nviews, g.nbins))
    bp = np.zeros_like(v)
    lam = 0.0
    for _ in range(n_iter):
        project(np.ascontiguousarray(v), worksino)
        data_sino = filt_hi(filt_hi(worksino)) + r_lo * filt_lo(filt_lo(worksino))
        bp.fill(0.0)
        backproject(data_sino, bp)
        Mv = nus2 * (bp * mask)
        Mv += (nux2 * r_tv) * (mdivx(gradx(v)) * mask)
        Mv += (nuy2 * r_tv) * (mdivy(grady(v)) * mask)
        Mv += (l1sq * r_l1) * v
        lam = float((v * Mv).sum())          # Rayleigh quotient
        nv = np.sqrt((Mv ** 2).sum())
        if nv < 1e-300:
            break
        v = Mv / nv
    return float(np.sqrt(max(lam, 0.0)))


def solve_two_channel(
    sinodata: np.ndarray,
    phimage: np.ndarray,
    project: Callable,
    backproject: Callable,
    filt_hi: Callable,
    filt_lo: Callable,
    mask: np.ndarray,
    nusino: float,
    nuxgrad: float,
    nuygrad: float,
    g: FanbeamGeometry,
    res_params: ResolutionParams,
    cfg: Config,
) -> SolverResult:
    print("Running two-channel...")
    sinodata_hi_sc = nusino * filt_hi(sinodata)
    sinodata_lo_sc = nusino * filt_lo(sinodata)

    sqrt_lo = float(np.sqrt(cfg.sigma_lo_scale))
    sqrt_tv = float(np.sqrt(cfg.sigma_tv_scale))
    sqrt_l1 = float(np.sqrt(cfg.sigma_l1_scale))

    # ||Sigma^(1/2) K|| / sqrt(sig_hi). All sigma_*_scale weights are
    # already absorbed; multiplying by sqrt(sig_hi) gives the true norm.
    norm_unscaled = _operator_norm_two(
        project, backproject, filt_hi, filt_lo, mask, nusino,
        nuxgrad, nuygrad, cfg.l1_weight, sqrt_lo, sqrt_tv, sqrt_l1, g,
    )

    # CP condition: tau * sig_hi * norm_unscaled^2 < 1.
    # Pick sig_hi to give the requested step balance, then derive tau.
    # NB: no spurious 0.5 safety factor -- the (1 + cvg_slack) is the only
    # margin, matching what the single-channel solver uses.
    sig_hi = cfg.step_balance_two / norm_unscaled
    sig_lo = cfg.sigma_lo_scale * sig_hi
    sig_tv = cfg.sigma_tv_scale * sig_hi
    sig_l1 = cfg.sigma_l1_scale * sig_hi
    tau = 1.0 / (sig_hi * norm_unscaled ** 2 * (1.0 + cfg.cvg_slack))
    cp_lhs = tau * sig_hi * norm_unscaled ** 2
    print(
        f"  ||Sigma^(1/2) K||/sqrt(sig_hi)={norm_unscaled:.4f}  "
        f"sig_hi={sig_hi:.6f}  sig_lo={sig_lo:.6f}  tau={tau:.6f}  "
        f"tau*sig_hi*||K||^2={cp_lhs:.6f} (< 1 required)"
    )
    assert cp_lhs < 1.0, "Two-channel CP condition violated"

    nrays = g.nviews * g.nbins
    epssc_hi = cfg.eps_hi_scale * cfg.eps * np.sqrt(nrays)
    epssc_lo = cfg.eps_lo_scale * cfg.eps * np.sqrt(nrays)
    print(
        f"  data tolerances: epssc_hi={epssc_hi:.4f} (scale={cfg.eps_hi_scale}), "
        f"epssc_lo={epssc_lo:.4f} (scale={cfg.eps_lo_scale})"
    )

    nx, ny = g.nx, g.ny
    xim = np.zeros((nx, ny))
    yim_l1 = np.zeros_like(xim)
    xbarim = np.zeros_like(xim)
    ysino_hi = np.zeros((g.nviews, g.nbins))
    ysino_lo = np.zeros((g.nviews, g.nbins))
    ygradx = np.zeros_like(xim)
    ygrady = np.zeros_like(xim)

    ierrs: list[float] = []
    derrs: list[float] = []
    derrs_hi: list[float] = []
    derrs_lo: list[float] = []
    tvs: list[float] = []
    alpha, beta, itermax = res_params.alpha, res_params.beta, res_params.itermax
    start_time = time.time()

    for itr in range(1, itermax + 1):
        ysinoold_hi = ysino_hi.copy()
        ysinoold_lo = ysino_lo.copy()
        ygradx_old = ygradx.copy()
        ygrady_old = ygrady.copy()
        yim_old = yim_l1.copy()

        # Adjoint step: aggregate dual variables back to image space
        wimp = np.zeros_like(xim)
        imtmp = np.zeros_like(xim)
        backproject(filt_hi(ysino_hi), imtmp)
        wimp += imtmp
        imtmp.fill(0.0)
        backproject(filt_lo(ysino_lo), imtmp)
        wimp += imtmp
        wimp *= nusino * mask
        wimqx = mdivx(ygradx) * (nuxgrad * mask)
        wimqy = mdivy(ygrady) * (nuygrad * mask)
        wiml1 = cfg.l1_weight * yim_l1

        ximold = xim.copy()
        xim = xim - tau * (wimp + wimqx + wimqy + wiml1)
        np.maximum(xim, 0.0, out=xim)
        xbarim = 2.0 * xim - ximold

        # Forward step + per-channel residuals
        worksino = np.zeros((g.nviews, g.nbins))
        project(xbarim, worksino)
        resid_hi = nusino * filt_hi(worksino) - sinodata_hi_sc
        resid_lo = nusino * filt_lo(worksino) - sinodata_lo_sc
        derr_hi = float(np.sqrt(((resid_hi / nusino) ** 2).sum()) / np.sqrt(nrays))
        derr_lo = float(np.sqrt(((resid_lo / nusino) ** 2).sum()) / np.sqrt(nrays))
        derrs_hi.append(derr_hi)
        derrs_lo.append(derr_lo)
        derrs.append(float(np.sqrt(derr_hi ** 2 + derr_lo ** 2)))

        # Independent dual updates per channel -- the whole point of two-channel
        ysino_hi = prox_residual_ball(ysino_hi + sig_hi * resid_hi, sig_hi * nusino * epssc_hi)
        ysino_lo = prox_residual_ball(ysino_lo + sig_lo * resid_lo, sig_lo * nusino * epssc_lo)

        ptilx = ygradx + sig_tv * (gradx(xbarim) * nuxgrad)
        ygradx = prox_dtv_dual(ptilx, 2.0 - alpha)

        ptily = ygrady + sig_tv * (grady(xbarim) * nuygrad)
        ygrady = prox_dtv_dual(ptily, alpha)

        ptil1 = yim_l1 + sig_l1 * (cfg.l1_weight * xbarim)
        yim_l1 = prox_dtv_dual(ptil1, beta)

        tgx, tgy = gradim(xbarim)
        tvs.append(float(np.sqrt(tgx ** 2 + tgy ** 2).sum()))

        # Over-relaxation
        ygradx = ygradx_old - cfg.rho * (ygradx_old - ygradx)
        ygrady = ygrady_old - cfg.rho * (ygrady_old - ygrady)
        ysino_hi = ysinoold_hi - cfg.rho * (ysinoold_hi - ysino_hi)
        ysino_lo = ysinoold_lo - cfg.rho * (ysinoold_lo - ysino_lo)
        yim_l1 = yim_old - cfg.rho * (yim_old - yim_l1)
        xim = ximold - cfg.rho * (ximold - xim)

        ierrs.append(float(np.sqrt(((xbarim - phimage) ** 2).sum() / (nx * ny))))
        if itr in ISTOPS:
            print(
                f"  Iter {itr}: data_err={derrs[-1]:.6f}, "
                f"img_err={ierrs[-1]:.6f}, TV={tvs[-1]:.2f}"
            )

    return SolverResult(
        image=xbarim.copy(), ierrs=ierrs, derrs=derrs, tvs=tvs,
        elapsed=time.time() - start_time,
        steps=dict(sig_hi=sig_hi, sig_lo=sig_lo, sig_tv=sig_tv, sig_l1=sig_l1, tau=tau, norm=norm_unscaled),
        derrs_hi=derrs_hi, derrs_lo=derrs_lo,
    )


# ---------------------------------------------------------------------------
# Multi-channel (dyadic) CP solver
# ---------------------------------------------------------------------------


def resolve_sigma_scales(
    n_channels: int, explicit: tuple[float, ...], base: float
) -> tuple[float, ...]:
    """Return the per-channel sigma multipliers.

    If `explicit` is given (length matches n_channels), use it. Otherwise
    use the dyadic schedule sigma_i = base^i * sigma_0. With base=2 and
    the dyadic band structure, this exactly compensates the energy halving
    per band: ||R_i X||^2 ~ 2^(-i) ||R_0 X||^2 in the ramp, so
    sigma_i ||R_i X||^2 ~ ||R_0 X||^2 is roughly constant across i,
    and the inflation of ||Sigma^(1/2) K|| is ~ k (linear in depth)
    rather than 2^k (exponential).
    """
    if explicit:
        if len(explicit) != n_channels:
            raise ValueError(
                f"sigma_scales length {len(explicit)} != n_channels {n_channels}"
            )
        return tuple(float(s) for s in explicit)
    return tuple(float(base ** i) for i in range(n_channels))


def _operator_norm_multi(
    project: Callable, backproject: Callable,
    filt_bands: tuple[Callable, ...], mask: np.ndarray, nusino: float,
    nuxgrad: float, nuygrad: float, l1_weight: float,
    sqrt_sigma_ratios: tuple[float, ...],
    sqrt_tv: float, sqrt_l1: float,
    g: FanbeamGeometry, n_iter: int = 200,
) -> float:
    """Returns ||Sigma^(1/2) K|| / sqrt(sigma_0) = sqrt(lambda_max(Mtilde)),
    where Mtilde = sum_i r_i K_i^T K_i + r_tv (gx^T gx + gy^T gy) + r_l1 I is the
    sigma_0-factored weighted normal operator (r_i = sigma_i/sigma_0 =
    sqrt_sigma_ratios[i]^2).

    Symmetric power iteration on Mtilde; each band filter is applied TWICE so the
    data term is the true X^T R_i^2 X (see _operator_norm_two).
    """
    r_data = [s ** 2 for s in sqrt_sigma_ratios]
    r_tv, r_l1 = sqrt_tv ** 2, sqrt_l1 ** 2
    nus2, nux2, nuy2, l1sq = nusino ** 2, nuxgrad ** 2, nuygrad ** 2, l1_weight ** 2
    k = len(filt_bands)
    rng = np.random.default_rng(SEED)
    v = rng.standard_normal((g.nx, g.ny)) * mask
    v /= np.sqrt((v ** 2).sum()) + 1e-12
    worksino = np.zeros((g.nviews, g.nbins))
    bp = np.zeros_like(v)
    lam = 0.0
    for _ in range(n_iter):
        project(np.ascontiguousarray(v), worksino)
        data_sino = np.zeros_like(worksino)
        for i in range(k):
            data_sino += r_data[i] * filt_bands[i](filt_bands[i](worksino))
        bp.fill(0.0)
        backproject(data_sino, bp)
        Mv = nus2 * (bp * mask)
        Mv += (nux2 * r_tv) * (mdivx(gradx(v)) * mask)
        Mv += (nuy2 * r_tv) * (mdivy(grady(v)) * mask)
        Mv += (l1sq * r_l1) * v
        lam = float((v * Mv).sum())          # Rayleigh quotient
        nv = np.sqrt((Mv ** 2).sum())
        if nv < 1e-300:
            break
        v = Mv / nv
    return float(np.sqrt(max(lam, 0.0)))


def solve_multi_channel(
    sinodata: np.ndarray,
    phimage: np.ndarray,
    project: Callable,
    backproject: Callable,
    filters: DyadicFilters,
    mask: np.ndarray,
    nusino: float,
    nuxgrad: float,
    nuygrad: float,
    g: FanbeamGeometry,
    res_params: ResolutionParams,
    cfg: Config,
) -> SolverResult:
    """k-channel CP solver with dyadic frequency partition.

    Convergence condition (product-space CP with diagonal dual preconditioning):

        tau * || diag(sqrt(sigma_0) I, ..., sqrt(sigma_{k-1}) I) * K ||_2^2 < 1

    where K = [R_0 X; ...; R_{k-1} X] stacks the k band-pass projectors.
    With the dyadic partition (partition of unity) and sigma_i = 2^i sigma_0,
    the stacked operator norm grows only linearly in k rather than
    exponentially, because the ramp puts most energy in the highest band.
    """
    k = len(filters)
    sigma_scales = resolve_sigma_scales(k, cfg.sigma_scales, cfg.sigma_dyadic_base)
    if sigma_scales[0] <= 0:
        raise ValueError("sigma_scales[0] must be positive (it sets the base)")
    sigma_ratios = tuple(s / sigma_scales[0] for s in sigma_scales)
    sqrt_sigma_ratios = tuple(float(np.sqrt(r)) for r in sigma_ratios)

    eps_scales = cfg.eps_scales if cfg.eps_scales else tuple([1.0] * k)
    if len(eps_scales) != k:
        raise ValueError(f"eps_scales length {len(eps_scales)} != n_channels {k}")

    sqrt_tv = float(np.sqrt(cfg.sigma_tv_scale))
    sqrt_l1 = float(np.sqrt(cfg.sigma_l1_scale))

    print(f"Running {k}-channel (dyadic)...")
    print(f"  sigma_ratios = {[f'{r:.2f}' for r in sigma_ratios]}")
    print(f"  eps_scales   = {[f'{r:.2f}' for r in eps_scales]}")

    sinodata_bands_sc = [nusino * filters.bands[i](sinodata) for i in range(k)]

    norm_unscaled = _operator_norm_multi(
        project, backproject, filters.bands, mask, nusino,
        nuxgrad, nuygrad, cfg.l1_weight,
        sqrt_sigma_ratios, sqrt_tv, sqrt_l1, g,
    )

    sigma_0 = cfg.step_balance_multi / norm_unscaled
    sigmas = tuple(sigma_0 * r for r in sigma_ratios)
    sig_tv = cfg.sigma_tv_scale * sigma_0
    sig_l1 = cfg.sigma_l1_scale * sigma_0
    tau = 1.0 / (sigma_0 * norm_unscaled ** 2 * (1.0 + cfg.cvg_slack))
    cp_lhs = tau * sigma_0 * norm_unscaled ** 2
    print(
        f"  ||Sigma^(1/2) K||/sqrt(sig_0)={norm_unscaled:.4f}  "
        f"sig_0={sigma_0:.6f}  tau={tau:.6f}  "
        f"tau*sig_0*||K||^2={cp_lhs:.6f} (< 1 required)"
    )
    assert cp_lhs < 1.0, "Multichannel CP condition violated"

    nrays = g.nviews * g.nbins
    epsscs = tuple(eps_scales[i] * cfg.eps * np.sqrt(nrays) for i in range(k))
    print(f"  data tolerances epssc_i = {[f'{e:.4f}' for e in epsscs]}")

    nx, ny = g.nx, g.ny
    xim = np.zeros((nx, ny))
    yim_l1 = np.zeros_like(xim)
    xbarim = np.zeros_like(xim)
    ysino_bands = [np.zeros((g.nviews, g.nbins)) for _ in range(k)]
    ygradx = np.zeros_like(xim)
    ygrady = np.zeros_like(xim)

    ierrs: list[float] = []
    derrs: list[float] = []
    derrs_bands: list[list[float]] = [[] for _ in range(k)]
    tvs: list[float] = []
    alpha, beta, itermax = res_params.alpha, res_params.beta, res_params.itermax
    start_time = time.time()

    for itr in range(1, itermax + 1):
        ysino_olds = [y.copy() for y in ysino_bands]
        ygradx_old = ygradx.copy()
        ygrady_old = ygrady.copy()
        yim_old = yim_l1.copy()

        wimp = np.zeros_like(xim)
        imtmp = np.zeros_like(xim)
        for i in range(k):
            imtmp.fill(0.0)
            backproject(filters.bands[i](ysino_bands[i]), imtmp)
            wimp += imtmp
        wimp *= nusino * mask
        wimqx = mdivx(ygradx) * (nuxgrad * mask)
        wimqy = mdivy(ygrady) * (nuygrad * mask)
        wiml1 = cfg.l1_weight * yim_l1

        ximold = xim.copy()
        xim = xim - tau * (wimp + wimqx + wimqy + wiml1)
        np.maximum(xim, 0.0, out=xim)
        xbarim = 2.0 * xim - ximold

        worksino = np.zeros((g.nviews, g.nbins))
        project(xbarim, worksino)
        resids = [nusino * filters.bands[i](worksino) - sinodata_bands_sc[i]
                  for i in range(k)]
        derr_acc_sq = 0.0
        for i, r in enumerate(resids):
            d_i = float(np.sqrt(((r / nusino) ** 2).sum()) / np.sqrt(nrays))
            derrs_bands[i].append(d_i)
            derr_acc_sq += d_i ** 2
        derrs.append(float(np.sqrt(derr_acc_sq)))

        for i in range(k):
            ysino_bands[i] = prox_residual_ball(
                ysino_bands[i] + sigmas[i] * resids[i],
                sigmas[i] * nusino * epsscs[i],
            )

        ptilx = ygradx + sig_tv * (gradx(xbarim) * nuxgrad)
        ygradx = prox_dtv_dual(ptilx, 2.0 - alpha)

        ptily = ygrady + sig_tv * (grady(xbarim) * nuygrad)
        ygrady = prox_dtv_dual(ptily, alpha)

        ptil1 = yim_l1 + sig_l1 * (cfg.l1_weight * xbarim)
        yim_l1 = prox_dtv_dual(ptil1, beta)

        tgx, tgy = gradim(xbarim)
        tvs.append(float(np.sqrt(tgx ** 2 + tgy ** 2).sum()))

        ygradx = ygradx_old - cfg.rho * (ygradx_old - ygradx)
        ygrady = ygrady_old - cfg.rho * (ygrady_old - ygrady)
        for i in range(k):
            ysino_bands[i] = ysino_olds[i] - cfg.rho * (ysino_olds[i] - ysino_bands[i])
        yim_l1 = yim_old - cfg.rho * (yim_old - yim_l1)
        xim = ximold - cfg.rho * (ximold - xim)

        ierrs.append(float(np.sqrt(((xbarim - phimage) ** 2).sum() / (nx * ny))))
        if itr in ISTOPS:
            print(
                f"  Iter {itr}: data_err={derrs[-1]:.6f}, "
                f"img_err={ierrs[-1]:.6f}, TV={tvs[-1]:.2f}"
            )

    return SolverResult(
        image=xbarim.copy(), ierrs=ierrs, derrs=derrs, tvs=tvs,
        elapsed=time.time() - start_time,
        steps=dict(
            sigma_0=sigma_0, sigmas=sigmas, sig_tv=sig_tv, sig_l1=sig_l1,
            tau=tau, norm=norm_unscaled, n_channels=k,
            sigma_ratios=sigma_ratios, eps_scales=eps_scales,
        ),
        derrs_bands=derrs_bands,
    )


# ---------------------------------------------------------------------------
# Driver: single resolution
# ---------------------------------------------------------------------------


def load_phantom(image_number: int, resolution: int) -> np.ndarray:
    """Load and resample the phantom to the requested resolution.

    Native phantom data is 512x512. For resolution < 512, decimate by an
    integer factor. For resolution > 512, upsample by integer Kronecker
    repetition (nearest-neighbor); this preserves sharp edges but adds no
    information beyond the native 512x512 -- the higher-res run is testing
    the *reconstruction* at finer pixels, not finer ground truth.
    """
    p1 = np.load(DATA_DIR / "Phantom_Adipose.npy")[image_number]
    p2 = np.load(DATA_DIR / "Phantom_Fibroglandular.npy")[image_number]
    p3 = np.load(DATA_DIR / "Phantom_Calcification.npy")[image_number]
    base = (0.5 * p1 + 1.0 * p2 + 2.0 * p3).astype(np.float64)
    native = base.shape[0]  # expect 512

    if resolution == native:
        return base.copy()
    if resolution < native:
        if native % resolution != 0:
            raise ValueError(f"resolution {resolution} must divide native {native}")
        mfact = native // resolution
        return base[::mfact, ::mfact].copy()
    # resolution > native: upsample
    if resolution % native != 0:
        raise ValueError(f"resolution {resolution} must be an integer multiple of native {native}")
    upfact = resolution // native
    print(f"  (upsampling phantom from {native} -> {resolution} by Kronecker x{upfact})")
    return np.kron(base, np.ones((upfact, upfact))).copy()


def make_sinogram(
    phimage: np.ndarray,
    project: Callable,
    g: FanbeamGeometry,
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    truesino = np.zeros((g.nviews, g.nbins))
    project(phimage, truesino)
    if cfg.add_noise:
        return -np.log(rng.poisson(cfg.photon_count * np.exp(-truesino)) / cfg.photon_count)
    return truesino.copy()


def compute_normalization_constants(
    project: Callable,
    backproject: Callable,
    filt_full: Callable,
    mask: np.ndarray,
    g: FanbeamGeometry,
    cfg: Config,
) -> tuple[float, float, float]:
    """Operator norms for X^T X and the gradient operators -- gives nusino,
    nuxgrad, nuygrad scaling factors."""
    rng = np.random.default_rng(SEED)

    # ||X^T X|| via filtered projection-backprojection
    xim = rng.standard_normal((g.nx, g.ny)) * mask
    worksino = np.zeros((g.nviews, g.nbins))
    xnorm2 = 0.0
    for _ in range(50):
        project(xim, worksino)
        worksino_f = filt_full(filt_full(worksino))
        xim.fill(0.0)
        backproject(worksino_f, xim)
        xim *= mask
        xnorm2 = float(np.sqrt((xim ** 2).sum()))
        xim /= xnorm2 + 1e-12
    nusino = 1.0 / np.sqrt(xnorm2 + 1e-12)

    # ||grad_x^T grad_x||
    xim = rng.standard_normal((g.nx, g.ny)) * mask
    for _ in range(50):
        xim = mdivx(gradx(xim)) * mask
        xnorm2 = float(np.sqrt((xim ** 2).sum()))
        xim /= xnorm2 + 1e-12
    nuxgrad = cfg.nux_factor / np.sqrt(xnorm2 + 1e-12)

    # ||grad_y^T grad_y||
    xim = rng.standard_normal((g.nx, g.ny)) * mask
    for _ in range(50):
        xim = mdivy(grady(xim)) * mask
        xnorm2 = float(np.sqrt((xim ** 2).sum()))
        xim /= xnorm2 + 1e-12
    nuygrad = cfg.nuy_factor / np.sqrt(xnorm2 + 1e-12)

    return nusino, nuxgrad, nuygrad


def run_for_resolution(resolution: int, cfg: Config) -> dict:
    res_params = RESOLUTION_PARAMS[resolution]

    print(f"\n{'=' * 60}")
    print(f"RUNNING RECONSTRUCTION FOR {resolution}x{resolution}")
    print(f"alpha={res_params.alpha}, beta={res_params.beta}, itermax={res_params.itermax}")
    print("=" * 60)

    print("Loading phantom data...")
    phimage = load_phantom(cfg.image_number, resolution)

    g = make_geometry(resolution, cfg.larc)
    mask = make_image_mask(g)
    project, backproject = make_projectors(g)
    filters = make_filters(g, cfg.cutoff, cfg.cutoff_lo)

    rng = np.random.default_rng(SEED)
    print("Generating sinogram data...")
    sinodata = make_sinogram(phimage, project, g, cfg, rng)

    tgx, tgy = gradim(phimage)
    truetv = float(np.sqrt(tgx ** 2 + tgy ** 2).sum())
    print(f"Ground truth TV: {truetv:.2f}")

    print("Computing operator norms...")
    nusino, nuxgrad, nuygrad = compute_normalization_constants(
        project, backproject, filters.full, mask, g, cfg,
    )

    nrays = g.nviews * g.nbins
    epssc = cfg.eps * np.sqrt(nrays)

    single = solve_single_channel(
        sinodata, phimage, project, backproject, filters.full,
        mask, nusino, nuxgrad, nuygrad, g, res_params, cfg, epssc,
    )
    two = solve_two_channel(
        sinodata, phimage, project, backproject, filters.hi, filters.lo,
        mask, nusino, nuxgrad, nuygrad, g, res_params, cfg,
    )

    # Multichannel run (skip if cfg.n_channels == 1)
    multi: SolverResult | None = None
    multi_k: int | None = None
    if cfg.n_channels != 1:
        multi_k = cfg.n_channels if cfg.n_channels >= 2 else auto_n_channels(g)
        dyadic = make_dyadic_filters(g, multi_k)
        band_energies = [w.sum() for w in dyadic.band_weights_sq]
        total_e = sum(band_energies)
        print(f"  Dyadic n_channels={multi_k} "
              f"(auto: {auto_n_channels(g)}); "
              f"band energies (%) = "
              f"{[f'{100*e/total_e:.2f}' for e in band_energies]}")
        multi = solve_multi_channel(
            sinodata, phimage, project, backproject, dyadic,
            mask, nusino, nuxgrad, nuygrad, g, res_params, cfg,
        )

    print(f"  Single-channel: {single.elapsed:.1f}s, final RMSE={single.ierrs[-1]:.6f}")
    print(f"  Two-channel:    {two.elapsed:.1f}s, final RMSE={two.ierrs[-1]:.6f}")
    if multi is not None:
        print(f"  {multi_k}-channel:     {multi.elapsed:.1f}s, final RMSE={multi.ierrs[-1]:.6f}")

    # Step-size comparison: if two-channel has tau << single-channel tau,
    # that's why it's slower per iteration -- the band split inflated ||K||.
    s_tau = single.steps["tau"]
    s_sig = single.steps["sigma"]
    s_norm = single.steps["norm"]
    t_tau = two.steps["tau"]
    t_sig_hi = two.steps["sig_hi"]
    t_norm = two.steps["norm"]
    print(f"  -- step diagnostics --")
    print(f"  Single: tau={s_tau:.6f}  sigma={s_sig:.4f}  ||K||={s_norm:.4f}")
    print(f"  Two   : tau={t_tau:.6f}  sig_hi={t_sig_hi:.4f}  ||K||/sqrt(sig_hi)={t_norm:.4f}")
    print(f"  tau ratio (two/single) = {t_tau/s_tau:.4f}  "
          f"(<1 means two-channel takes smaller primal steps)")
    print(f"  ||K_two effective|| / ||K_single|| = "
          f"{(t_norm * np.sqrt(t_sig_hi)) / s_norm:.4f}")
    if multi is not None:
        m_tau = multi.steps["tau"]
        m_sig0 = multi.steps["sigma_0"]
        m_norm = multi.steps["norm"]
        print(f"  Multi : tau={m_tau:.6f}  sig_0={m_sig0:.4f}  ||K||/sqrt(sig_0)={m_norm:.4f}")
        print(f"  tau ratio (multi/single) = {m_tau/s_tau:.4f}")
        print(f"  ||K_multi effective|| / ||K_single|| = "
              f"{(m_norm * np.sqrt(m_sig0)) / s_norm:.4f}")

    result = {
        "resolution": resolution,
        "ierrs_single": single.ierrs,
        "ierrs_two": two.ierrs,
        "derrs_single": single.derrs,
        "derrs_two": two.derrs,
        "derrs_hi": two.derrs_hi,
        "derrs_lo": two.derrs_lo,
        "final_rmse_single": single.ierrs[-1],
        "final_rmse_two": two.ierrs[-1],
        "single_time": single.elapsed,
        "two_time": two.elapsed,
        "xbarim_single": single.image,
        "xbarim_two": two.image,
        "phimage": phimage,
        "truetv": truetv,
        "single_steps": single.steps,
        "two_steps": two.steps,
    }
    if multi is not None:
        result.update({
            "ierrs_multi": multi.ierrs,
            "derrs_multi": multi.derrs,
            "derrs_multi_bands": multi.derrs_bands,
            "final_rmse_multi": multi.ierrs[-1],
            "multi_time": multi.elapsed,
            "xbarim_multi": multi.image,
            "multi_steps": multi.steps,
            "n_channels_multi": multi_k,
        })
    return result


# ---------------------------------------------------------------------------
# Output: tables, plots
# ---------------------------------------------------------------------------


def write_rmse_table(all_results: dict[int, dict]) -> None:
    print("\n" + "=" * 78)
    print("RMSE COMPARISON TABLE")
    print("=" * 78)
    has_multi = any("final_rmse_multi" in r for r in all_results.values())
    if has_multi:
        header = ("| Resolution | Single  | Two-Ch  | Multi-Ch | Two Δ% | Multi Δ% |")
        sep = "|" + "|".join(["-" * 12, "-" * 9, "-" * 9, "-" * 10, "-" * 8, "-" * 10]) + "|"
    else:
        header = "| Resolution | Single-Channel RMSE | Two-Channel RMSE | Improvement |"
        sep = "|------------|---------------------|------------------|-------------|"
    print("\n" + header)
    print(sep)

    title = ("RMSE Comparison: Single vs Two-Channel vs Multi-Channel DTV" if has_multi
             else "RMSE Comparison: Single-Channel vs Two-Channel DTV")
    lines = [title, "=" * 70, "", header, sep]
    for res in PLOT_ORDER:
        if res not in all_results:
            continue
        r = all_results[res]
        imp_two = (r["final_rmse_single"] - r["final_rmse_two"]) / r["final_rmse_single"] * 100
        if has_multi and "final_rmse_multi" in r:
            imp_multi = (r["final_rmse_single"] - r["final_rmse_multi"]) / r["final_rmse_single"] * 100
            line = (
                f"| {res}x{res} | {r['final_rmse_single']:.5f} | "
                f"{r['final_rmse_two']:.5f} | {r['final_rmse_multi']:.5f}  | "
                f"{imp_two:+6.2f}% | {imp_multi:+8.2f}% |"
            )
        elif has_multi:
            line = (
                f"| {res}x{res} | {r['final_rmse_single']:.5f} | "
                f"{r['final_rmse_two']:.5f} | {'—':>8}  | "
                f"{imp_two:+6.2f}% | {'—':>8} |"
            )
        else:
            line = (
                f"| {res}x{res}  | {r['final_rmse_single']:.6f}            | "
                f"{r['final_rmse_two']:.6f}         | {imp_two:+.2f}%      |"
            )
        print(line)
        lines.append(line)

    out_path = RESULTS_DIR / "rmse_table.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nTable saved to '{out_path}'")


def plot_convergence_subplots(all_results: dict[int, dict]) -> None:
    print("\nGenerating per-resolution subplot figure...")
    available = [r for r in PLOT_ORDER if r in all_results]
    if not available:
        return
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n + 0.5, 3.5), squeeze=False)
    for idx, res in enumerate(available):
        ax = axes[0, idx]
        r = all_results[res]
        iterations = range(1, len(r["ierrs_single"]) + 1)
        ax.semilogy(iterations, r["ierrs_single"], "r-", linewidth=1.5, label="Single")
        ax.semilogy(iterations, r["ierrs_two"], "b-", linewidth=1.5, label="Two-Channel")
        if "ierrs_multi" in r:
            k = r.get("n_channels_multi", "k")
            ax.semilogy(iterations, r["ierrs_multi"], "g-", linewidth=1.5,
                        label=f"{k}-Ch (dyadic)")
        ax.set_xlabel("Iteration", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Image RMSE", fontsize=10)
        ax.set_title(f"{res}x{res}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim([1e-2, 1.0])
    plt.tight_layout()
    out_path = RESULTS_DIR / "convergence_subplots.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_band_residuals(all_results: dict[int, dict]) -> None:
    """For each resolution, plot the high- and low-band data residuals over
    iterations alongside their respective epssc_* tolerance.

    A band whose residual sits above the tolerance is being held back by
    the prior; a band whose residual converges below the tolerance has
    "spent" its data-fit budget.
    """
    print("Generating per-band data-residual plot...")
    available = [r for r in PLOT_ORDER if r in all_results and all_results[r].get("derrs_hi")]
    if not available:
        print("  (no per-band residuals to plot)")
        return
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
    for idx, res in enumerate(available):
        ax = axes[0, idx]
        r = all_results[res]
        iters = range(1, len(r["derrs_hi"]) + 1)
        ax.semilogy(iters, r["derrs_hi"], "b-", linewidth=1.5, label="high-band")
        ax.semilogy(iters, r["derrs_lo"], "r-", linewidth=1.5, label="low-band")
        if "derrs_single" in r:
            ax.semilogy(iters, r["derrs_single"], "k--", linewidth=1.0, alpha=0.6, label="single (full)")
        ax.set_xlabel("Iteration")
        if idx == 0:
            ax.set_ylabel("Data residual (RMS)")
        ax.set_title(f"{res}x{res} band residuals")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = RESULTS_DIR / "band_residuals.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_dyadic_filter_spectra(g: FanbeamGeometry, n_channels: int) -> None:
    """Plot the k dyadic band-pass filters and verify partition of unity."""
    print("Generating dyadic filter spectrum plot...")
    if n_channels < 2:
        print(f"  (n_channels={n_channels} < 2, skipping)")
        return
    df = make_dyadic_filters(g, n_channels)
    sum_sq = sum(df.band_weights_sq)
    f_single_sq = np.abs(df.u_coords) + 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    cmap = plt.get_cmap("viridis")

    axes[0].plot(df.u_coords, np.sqrt(f_single_sq), "k-", linewidth=1.5,
                 label="$f_\\mathrm{single}=\\sqrt{|u|}$")
    for i, w_sq in enumerate(df.band_weights_sq):
        color = cmap(i / max(n_channels - 1, 1))
        if i == 0:
            label = "$f_0$ (HF, top half)"
        elif i == n_channels - 1:
            label = f"$f_{{{i}}}$ (LF tail)"
        else:
            label = f"$f_{{{i}}}$"
        axes[0].plot(df.u_coords, np.sqrt(w_sq), color=color, linewidth=1.3,
                     label=label)
    axes[0].set_xlabel("u (detector coord)")
    axes[0].set_ylabel("Filter magnitude")
    axes[0].set_title(f"Dyadic filter bank (k={n_channels})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=1 if n_channels <= 4 else 2)

    axes[1].plot(df.u_coords, f_single_sq, "k-", linewidth=2.0,
                 label="$|f_\\mathrm{single}|^2$")
    axes[1].plot(df.u_coords, sum_sq, "g--", linewidth=2.0,
                 label="$\\sum_i |f_i|^2$")
    err = float(np.abs(sum_sq - f_single_sq).max())
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("Squared magnitude")
    axes[1].set_title(f"Partition of unity check (max err = {err:.1e})")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    energies = [w.sum() for w in df.band_weights_sq]
    total = sum(energies)
    pcts = [100 * e / total for e in energies]
    print(f"  Band energies (%): {[f'{p:.2f}' for p in pcts]}")
    print(f"  Partition-of-unity error: {err:.2e}")

    plt.tight_layout()
    out_path = RESULTS_DIR / "dyadic_filter_spectra.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_multi_band_residuals(all_results: dict[int, dict]) -> None:
    """Per-band data residuals over iterations for the multichannel solver."""
    print("Generating multichannel band-residuals plot...")
    available = [r for r in PLOT_ORDER if r in all_results and all_results[r].get("derrs_multi_bands")]
    if not available:
        print("  (no multichannel residuals to plot)")
        return
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for idx, res in enumerate(available):
        ax = axes[0, idx]
        r = all_results[res]
        bands = r["derrs_multi_bands"]
        k = len(bands)
        for i, band in enumerate(bands):
            color = cmap(i / max(k - 1, 1))
            label = ("band 0 (HF)" if i == 0
                     else f"band {i} (LF tail)" if i == k - 1
                     else f"band {i}")
            ax.semilogy(range(1, len(band) + 1), band, color=color, linewidth=1.4,
                        label=label)
        if "derrs_single" in r:
            iters_s = range(1, len(r["derrs_single"]) + 1)
            ax.semilogy(iters_s, r["derrs_single"], "k--", linewidth=0.9, alpha=0.5,
                        label="single (full)")
        ax.set_xlabel("Iteration")
        if idx == 0:
            ax.set_ylabel("Data residual (RMS)")
        ax.set_title(f"{res}x{res} multichannel ({k} bands)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7)
    plt.tight_layout()
    out_path = RESULTS_DIR / "multi_band_residuals.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_filter_spectra(cfg: Config) -> None:
    """Plot |f_single|^2 vs |f_hi|^2 + |f_lo|^2 to verify the band split.

    A *partition of unity* split would have these two curves identical:
    every Fourier component the single-channel weights also gets weighted
    by the two-channel split, no more, no less. If sum-of-squares > single,
    the bands overlap and the two-channel solver is paying extra norm cost
    for redundant data; if < single, there's a coverage gap.
    """
    print("\nGenerating filter spectrum diagnostic plot...")
    g = make_geometry(512, cfg.larc)
    db = g.detectorlength / g.nbins
    b00 = -g.detectorlength / 2.0
    uar = np.arange(b00 + db / 2.0, b00 + g.detectorlength, db)
    w_sqrt_ramp = np.sqrt(np.abs(uar) + 1e-12)
    han_lo = np.clip(hanning_window(uar, b00, cfg.cutoff_lo), 0.0, 1.0)
    han_hi = np.clip(1.0 - hanning_window(uar, b00, cfg.cutoff), 0.0, 1.0)

    f_single = w_sqrt_ramp
    f_hi = w_sqrt_ramp * np.sqrt(han_hi)
    f_lo = w_sqrt_ramp * np.sqrt(han_lo)

    # Squared magnitude per Fourier bin -- this is what shows up in ||K||^2
    s_single = f_single ** 2
    s_split = f_hi ** 2 + f_lo ** 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].plot(uar, f_single, "k-", label="$f_{\\mathrm{single}} = \\sqrt{|u|}$", linewidth=1.5)
    axes[0].plot(uar, f_hi, "b-", label="$f_{\\mathrm{hi}}$", linewidth=1.5)
    axes[0].plot(uar, f_lo, "r-", label="$f_{\\mathrm{lo}}$", linewidth=1.5)
    axes[0].set_xlabel("u (detector coord, doubles as frequency bin)")
    axes[0].set_ylabel("Filter magnitude")
    axes[0].set_title("Filter shapes")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(uar, s_single, "k-", label="$|f_{\\mathrm{single}}|^2$", linewidth=2.0)
    axes[1].plot(uar, s_split, "g--", label="$|f_{\\mathrm{hi}}|^2 + |f_{\\mathrm{lo}}|^2$", linewidth=2.0)
    axes[1].plot(uar, s_split - s_single, "m:", label="overlap (split - single)", linewidth=1.5)
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("Squared magnitude")
    axes[1].set_title(f"Spectral coverage   (cutoff={cfg.cutoff}, cutoff_lo={cfg.cutoff_lo})")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].axhline(0.0, color="0.5", linewidth=0.5)

    overlap_int = float(np.trapezoid(s_split - s_single, uar))
    coverage_ratio = float(s_split.sum() / s_single.sum())
    hi_fraction = float((f_hi ** 2).sum() / s_split.sum())
    lo_fraction = float((f_lo ** 2).sum() / s_split.sum())
    # LF half-width in u: |b00| / cutoff_lo
    lf_halfwidth = abs(b00) / cfg.cutoff_lo
    hf_onset = abs(b00) / cfg.cutoff
    gap_present = cfg.cutoff > cfg.cutoff_lo  # HF starts after LF ends => gap
    print(f"  Spectral coverage ratio (split/single) = {coverage_ratio:.4f}")
    print(f"  (>1 means bands overlap and inflate ||K||; <1 means coverage gap)")
    print(f"  Band energy split: HF={hi_fraction*100:.1f}%, LF={lo_fraction*100:.1f}%")
    print(f"  LF support: |u| < {lf_halfwidth:.3f}   HF onset: |u| > {hf_onset:.3f}")
    if gap_present:
        print(f"  WARNING: cutoff ({cfg.cutoff}) > cutoff_lo ({cfg.cutoff_lo}) "
              f"creates a coverage gap in {lf_halfwidth:.3f} < |u| < {hf_onset:.3f}")
    elif cfg.cutoff < cfg.cutoff_lo:
        print(f"  Note: cutoff ({cfg.cutoff}) < cutoff_lo ({cfg.cutoff_lo}) "
              f"creates band overlap in {hf_onset:.3f} < |u| < {lf_halfwidth:.3f}")
    if lo_fraction < 0.10:
        print(f"  WARNING: LF band carries only {lo_fraction*100:.1f}% of energy. "
              f"Two-channel will look almost like single-channel. Try smaller cutoff_lo.")

    plt.tight_layout()
    out_path = RESULTS_DIR / "filter_spectra.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_image_diff(all_results: dict[int, dict]) -> None:
    """Plot xbarim_two - xbarim_single side-by-side per resolution.

    LF differences look smooth, HF differences look like edges.
    """
    print("Generating image-difference diagnostic plot...")
    available = [r for r in PLOT_ORDER if r in all_results]
    if not available:
        print("  (no results to plot)")
        return
    n = len(available)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for row, res in enumerate(available):
        r = all_results[res]
        diff = r["xbarim_two"] - r["xbarim_single"]
        vmax_img = max(r["xbarim_single"].max(), r["xbarim_two"].max())
        vmax_diff = float(np.abs(diff).max()) or 1e-12

        axes[row, 0].imshow(r["xbarim_single"], cmap="gray", vmin=0, vmax=vmax_img)
        axes[row, 0].set_title(f"{res}x{res} Single")
        axes[row, 1].imshow(r["xbarim_two"], cmap="gray", vmin=0, vmax=vmax_img)
        axes[row, 1].set_title(f"{res}x{res} Two-Ch")
        im = axes[row, 2].imshow(diff, cmap="seismic", vmin=-vmax_diff, vmax=vmax_diff)
        axes[row, 2].set_title(f"Diff (max |x|={vmax_diff:.4f})")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.colorbar(im, ax=axes[row, 2], fraction=0.045)

    plt.tight_layout()
    out_path = RESULTS_DIR / "image_diffs.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_convergence_combined(all_results: dict[int, dict]) -> None:
    print("Generating combined single-plot figure...")
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    colors = {1024: "#9467bd", 512: "#1f77b4", 256: "#ff7f0e", 128: "#2ca02c"}
    for res in PLOT_ORDER:
        if res not in all_results:
            continue
        r = all_results[res]
        iterations = range(1, len(r["ierrs_single"]) + 1)
        ax.semilogy(iterations, r["ierrs_single"], color=colors[res], linestyle="-",
                    linewidth=1.5, label=f"{res}x{res} Single")
        ax.semilogy(iterations, r["ierrs_two"], color=colors[res], linestyle="--",
                    linewidth=1.5, label=f"{res}x{res} Two-Ch")
        if "ierrs_multi" in r:
            k = r.get("n_channels_multi", "k")
            ax.semilogy(iterations, r["ierrs_multi"], color=colors[res], linestyle=":",
                        linewidth=1.8, label=f"{res}x{res} {k}-Ch")
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Image RMSE", fontsize=10)
    ax.set_title("Convergence Comparison", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_ylim([1e-2, 1.0])
    plt.tight_layout()
    out_path = RESULTS_DIR / "convergence_combined.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def maybe_load_cache(force: bool) -> tuple[bool, dict]:
    if force:
        print("\n--force specified, running full reconstruction...")
        return True, {}
    if not CACHE_FILE.exists():
        return True, {}
    print(f"\nFound cached results in '{CACHE_FILE}'")
    print("Loading cached results (use --force to recompute)...\n")
    try:
        with CACHE_FILE.open("rb") as f:
            return False, pickle.load(f)
    except Exception as exc:
        print(f"Error loading cache: {exc}. Running full reconstruction...")
        return True, {}


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-resolution DTV comparison")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    parser.add_argument(
        "--step-balance-two", type=float, default=None,
        help=f"Override step_balance_two (default {Config().step_balance_two}). Useful for tau/sigma sweeps.",
    )
    parser.add_argument(
        "--sigma-lo-scale", type=float, default=None,
        help=f"Override sigma_lo_scale (default {Config().sigma_lo_scale}). "
        "NB: > 1 inflates ||K|| and forces tau down. To favor LF accuracy, "
        "use --eps-lo-scale instead.",
    )
    parser.add_argument(
        "--eps-lo-scale", type=float, default=None,
        help=f"Override eps_lo_scale (default {Config().eps_lo_scale}). "
        "Smaller = tighter LF data fit. This changes the fixed point, not "
        "just the optimization trajectory.",
    )
    parser.add_argument(
        "--eps-hi-scale", type=float, default=None,
        help=f"Override eps_hi_scale (default {Config().eps_hi_scale}).",
    )
    parser.add_argument(
        "--noise", action="store_true",
        help="Enable Poisson noise on the sinogram. Without this, the data is "
        "noise-free and band-split denoising can't help.",
    )
    parser.add_argument(
        "--no-l1", action="store_true",
        help="Disable the L1 prior (sets l1_weight=0). Useful to isolate the "
        "data-fidelity contribution of the band split.",
    )
    parser.add_argument(
        "--cutoff", type=float, default=None,
        help=f"Override cutoff for HF band roll-on (default {Config().cutoff}). "
        "Hanning half-width = |b00|/cutoff. Bigger cutoff = HF starts later "
        "(narrower transition near DC).",
    )
    parser.add_argument(
        "--cutoff-lo", type=float, default=None,
        help=f"Override cutoff_lo for LF band roll-off (default {Config().cutoff_lo}). "
        "LF half-width = |b00|/cutoff_lo. Bigger cutoff_lo = NARROWER LF band. "
        "Set cutoff = cutoff_lo for partition-of-unity coverage.",
    )
    parser.add_argument(
        "--itermax", type=int, default=None,
        help="Override iteration count for all resolutions (default per-resolution: 500).",
    )
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=None,
        help=f"Resolutions to run (default {list(Config().resolutions)}). "
        "Each must appear in RESOLUTION_PARAMS. Native phantom is 512x512; "
        "values < 512 decimate, values > 512 upsample by Kronecker.",
    )
    parser.add_argument(
        "--n-channels", type=int, default=None,
        help="Number of dyadic bands for multichannel solver. 0 = analytic auto "
        "(default), 1 = skip multichannel, 2+ = use that many. Auto chooses "
        "max k such that the LF tail has at least 8 detector bins.",
    )
    parser.add_argument(
        "--sigma-scales", type=float, nargs="+", default=None,
        help="Per-channel sigma multipliers (length must match n_channels). "
        "Default schedule is sigma_i = 2^i * sigma_0 (stronger LF emphasis).",
    )
    parser.add_argument(
        "--sigma-base", type=float, default=None,
        help=f"Base for the default sigma schedule sigma_i = base^i * sigma_0. "
        f"Default {Config().sigma_dyadic_base}. Set to 1.0 for flat (all equal). "
        "Ignored if --sigma-scales is explicit.",
    )
    parser.add_argument(
        "--eps-scales", type=float, nargs="+", default=None,
        help="Per-channel eps multipliers (length must match n_channels). "
        "Default all 1.0. Smaller = tighter data fit on that band.",
    )
    return parser.parse_args()


def main() -> None:
    global RESOLUTION_PARAMS
    args = parse_args()
    np.random.seed(SEED)
    ensure_output_dirs()

    cfg_overrides: dict = {}
    if args.step_balance_two is not None:
        cfg_overrides["step_balance_two"] = args.step_balance_two
    if args.sigma_lo_scale is not None:
        cfg_overrides["sigma_lo_scale"] = args.sigma_lo_scale
    if args.eps_lo_scale is not None:
        cfg_overrides["eps_lo_scale"] = args.eps_lo_scale
    if args.eps_hi_scale is not None:
        cfg_overrides["eps_hi_scale"] = args.eps_hi_scale
    if args.cutoff is not None:
        cfg_overrides["cutoff"] = args.cutoff
    if args.cutoff_lo is not None:
        cfg_overrides["cutoff_lo"] = args.cutoff_lo
    if args.noise:
        cfg_overrides["add_noise"] = True
    if args.no_l1:
        cfg_overrides["l1_weight"] = 0.0
    if args.resolutions is not None:
        # Validate against RESOLUTION_PARAMS
        for r in args.resolutions:
            if r not in RESOLUTION_PARAMS:
                raise ValueError(f"Resolution {r} not in RESOLUTION_PARAMS; "
                                 f"add an entry or pick from {sorted(RESOLUTION_PARAMS)}.")
        cfg_overrides["resolutions"] = tuple(args.resolutions)
    if args.n_channels is not None:
        cfg_overrides["n_channels"] = args.n_channels
    if args.sigma_scales is not None:
        cfg_overrides["sigma_scales"] = tuple(args.sigma_scales)
    if args.sigma_base is not None:
        cfg_overrides["sigma_dyadic_base"] = args.sigma_base
    if args.eps_scales is not None:
        cfg_overrides["eps_scales"] = tuple(args.eps_scales)
    cfg = Config(**cfg_overrides) if cfg_overrides else Config()
    if cfg_overrides:
        print(f"Config overrides: {cfg_overrides}")

    # Optional iteration override applies to all resolutions.
    if args.itermax is not None:
        RESOLUTION_PARAMS = {
            res: ResolutionParams(alpha=p.alpha, beta=p.beta, itermax=args.itermax)
            for res, p in RESOLUTION_PARAMS.items()
        }
        print(f"Iteration count overridden: itermax={args.itermax}")

    run_reconstruction, all_results = maybe_load_cache(args.force)

    if run_reconstruction:
        for resolution in cfg.resolutions:
            result = run_for_resolution(resolution, cfg)
            all_results[result["resolution"]] = result

        print(f"\nSaving results to '{CACHE_FILE}'...")
        with CACHE_FILE.open("wb") as f:
            pickle.dump(all_results, f)
        print("Results saved!")

    write_rmse_table(all_results)
    plot_convergence_subplots(all_results)
    plot_convergence_combined(all_results)
    plot_filter_spectra(cfg)
    plot_image_diff(all_results)
    plot_band_residuals(all_results)
    if any("derrs_multi_bands" in r for r in all_results.values()):
        # Use the first available resolution's geometry for the spectrum plot
        first_res = next(iter(cfg.resolutions))
        first_g = make_geometry(first_res, cfg.larc)
        first_k = next(iter(all_results.values())).get("n_channels_multi",
                                                       cfg.n_channels or auto_n_channels(first_g))
        plot_dyadic_filter_spectra(first_g, first_k)
        plot_multi_band_residuals(all_results)


if __name__ == "__main__":
    main()
