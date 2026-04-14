"""
Multi-Resolution Comparison of Single-Channel vs Two-Channel DTV Reconstruction

Runs both methods on 512x512, 256x256, and 128x128 image sizes and generates:
1. RMSE comparison table
2. Convergence plots (separate subplots and combined)

Results are cached per resolution - use --force to recompute all.
"""

import numpy as np
from numpy import *
from numpy.random import randn, poisson
import matplotlib.pyplot as plt
from numba import njit
import time
import os
import pickle
import argparse

# Set seed for reproducibility
np.random.seed(42)

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Multi-resolution DTV comparison')
parser.add_argument('--force', action='store_true', help='Force recomputation')
args = parser.parse_args()

RESULTS_DIR = 'final_figures/'
CACHE_FILE = 'cache/multiresolution_results.pkl'

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

imagenumber = 3
addnoise = 0
nph = 1.e6
nuxfact = 0.5
nuyfact = 0.5
l1f = 1.0
eps = 0.001
larc = 1.0
rho = 1.75
stepbalance = 100.0
cutoffparm = 4.0

# Two-channel parameters
cutoffparm_lo = 8.0
eps_hi = eps
eps_lo = 1.25*eps

# Diagonal dual-preconditioning ratios for the two-channel CP form.
# sig_hi is the pivot; every other dual block's sigma is sig_hi * r_<block>.
# The power iteration in the two-channel branch uses these ratios to
# estimate ||Sigma^(1/2) K||_2 directly, so that the theorem
#     tau * || Sigma^(1/2) K ||_2^2  <  1
# is enforced by construction at step-size selection.
sigma_lo_scale = 4.0   # r_lo = sig_lo / sig_hi
sigma_tv_scale = 1.0   # r_tv = sig_tv / sig_hi  (used for ygradx, ygrady)
sigma_l1_scale = 1.0   # r_l1 = sig_l1 / sig_hi  (used for yim)

# Resolution-specific alpha/beta (from DTVminHan.py comments)
# 128: alpha=1.95, beta=10
# 256: alpha=1.9, beta=10
# 512: alpha=1.7, beta=5
RESOLUTION_PARAMS = {
    128: {'alpha': 1.95, 'beta': 10.0, 'itermax': 500},
    256: {'alpha': 1.9, 'beta': 10.0, 'itermax': 500},
    512: {'alpha': 1.7, 'beta': 5.0, 'itermax': 500},
}

# mfact values: 1->512, 2->256, 4->128
MFACT_VALUES = [1, 2, 4]

# ============================================================================
# CHECK FOR CACHED RESULTS
# ============================================================================

if not args.force and os.path.exists(CACHE_FILE):
    print(f"\nFound cached results in '{CACHE_FILE}'")
    print("Loading cached results (use --force to recompute)...\n")
    try:
        with open(CACHE_FILE, 'rb') as f:
            all_results = pickle.load(f)
        RUN_RECONSTRUCTION = False
        print("Cached results loaded!")
    except Exception as e:
        print(f"Error loading cache: {e}. Running full reconstruction...")
        RUN_RECONSTRUCTION = True
else:
    if args.force:
        print("\n--force specified, running full reconstruction...")
    RUN_RECONSTRUCTION = True
    all_results = {}


def run_reconstruction_for_mfact(mfact):
    """Run both single and two-channel reconstruction for a given mfact."""

    resolution = int(512/mfact)
    params = RESOLUTION_PARAMS[resolution]
    alpha = params['alpha']
    beta = params['beta']
    itermax = params['itermax']

    print(f"\n{'='*60}")
    print(f"RUNNING RECONSTRUCTION FOR {resolution}x{resolution} (mfact={mfact})")
    print(f"alpha={alpha}, beta={beta}, itermax={itermax}")
    print('='*60)

    # Load phantom data
    print("Loading phantom data...")
    phantom1 = load("data/phantoms_from_paper/Phantom_Adipose.npy")[imagenumber]
    phantom2 = load("data/phantoms_from_paper/Phantom_Fibroglandular.npy")[imagenumber]
    phantom3 = load("data/phantoms_from_paper/Phantom_Calcification.npy")[imagenumber]

    testimage = (0.5*phantom1 + 1.0*phantom2 + 2.0*phantom3).astype("float64")
    testimage = testimage[::mfact, ::mfact]*1.
    phimage = testimage*1.

    # Image parameters
    ximageside = 10.0
    yimageside = 10.0
    nx = int(512/mfact)
    ny = int(512/mfact)
    npix = nx*ny
    dx = ximageside/nx
    dy = yimageside/ny

    xar = arange(-ximageside/2. + dx/2, ximageside/2., dx)[:, newaxis]*ones([ny])
    yar = ones([nx, ny])*arange(-yimageside/2. + dy/2, yimageside/2., dy)
    rar = sqrt(xar**2 + yar**2)
    mask = phimage*0.
    mask[rar <= ximageside/2.] = 1.

    # Sinogram parameters
    radius = 50.0
    source_to_detector = 100.0
    srad = radius
    sd = source_to_detector
    slen = (50./180.)*pi
    slen0 = -slen/2.0
    ns0 = 25
    nu0 = 1024
    nviews = ns0
    nbins = nu0
    nrays = nbins*nviews
    epssc = eps*sqrt(nrays)

    fanangle2 = arcsin((ximageside/2.)/radius)
    detectorlength = 2.*tan(fanangle2)*source_to_detector

    # Store larc in local scope for njit functions
    larc_local = larc

    # ========================================================================
    # PROJECTION/BACKPROJECTION OPERATORS
    # ========================================================================

    @njit
    def circularFanbeamProjection(image, sinogram):
        dx = ximageside/nx
        dy = yimageside/ny
        x0 = -ximageside/2.
        y0 = -yimageside/2.
        u0 = -detectorlength/2.
        du = detectorlength/nbins
        ds = slen/(nviews-larc_local)

        for sindex in range(nviews):
            s = sindex*ds + slen0
            xsource = radius*cos(s)
            ysource = radius*sin(s)
            xDetCenter = (radius - source_to_detector)*cos(s)
            yDetCenter = (radius - source_to_detector)*sin(s)
            eux = -sin(s)
            euy = cos(s)

            for uindex in range(nbins):
                u = u0 + (uindex + 0.5)*du
                xbin = xDetCenter + eux*u
                ybin = yDetCenter + euy*u
                xl = x0
                yl = y0
                xdiff = xbin - xsource
                ydiff = ybin - ysource
                xad = abs(xdiff)*dy
                yad = abs(ydiff)*dx

                if (xad > yad):
                    slope = ydiff/xdiff
                    travPixlen = dx*sqrt(1.0 + slope*slope)
                    yIntOld = ysource + slope*(xl - xsource)
                    iyOld = int(floor((yIntOld - y0)/dy))
                    raysum = 0.
                    for ix in range(nx):
                        x = xl + dx*(ix + 1.0)
                        yIntercept = ysource + slope*(x - xsource)
                        iy = int(floor((yIntercept - y0)/dy))
                        if iy == iyOld:
                            if ((iy >= 0) and (iy < ny)):
                                raysum = raysum + travPixlen*image[ix, iy]
                        else:
                            yMid = dy*(iy if iy > iyOld else iyOld) + yl
                            ydist1 = abs(yMid - yIntOld)
                            ydist2 = abs(yIntercept - yMid)
                            frac1 = ydist1/(ydist1 + ydist2)
                            frac2 = 1.0 - frac1
                            if ((iyOld >= 0) and (iyOld < ny)):
                                raysum = raysum + frac1*travPixlen*image[ix, iyOld]
                            if ((iy >= 0) and (iy < ny)):
                                raysum = raysum + frac2*travPixlen*image[ix, iy]
                        iyOld = iy
                        yIntOld = yIntercept
                else:
                    slopeinv = xdiff/ydiff
                    travPixlen = dy*sqrt(1.0 + slopeinv*slopeinv)
                    xIntOld = xsource + slopeinv*(yl - ysource)
                    ixOld = int(floor((xIntOld - x0)/dx))
                    raysum = 0.
                    for iy in range(ny):
                        y = yl + dy*(iy + 1.0)
                        xIntercept = xsource + slopeinv*(y - ysource)
                        ix = int(floor((xIntercept - x0)/dx))
                        if (ix == ixOld):
                            if ((ix >= 0) and (ix < nx)):
                                raysum = raysum + travPixlen*image[ix, iy]
                        else:
                            xMid = dx*(ix if ix > ixOld else ixOld) + xl
                            xdist1 = abs(xMid - xIntOld)
                            xdist2 = abs(xIntercept - xMid)
                            frac1 = xdist1/(xdist1 + xdist2)
                            frac2 = 1.0 - frac1
                            if ((ixOld >= 0) and (ixOld < nx)):
                                raysum = raysum + frac1*travPixlen*image[ixOld, iy]
                            if ((ix >= 0) and (ix < nx)):
                                raysum = raysum + frac2*travPixlen*image[ix, iy]
                        ixOld = ix
                        xIntOld = xIntercept
                sinogram[sindex, uindex] = raysum


    @njit
    def circularFanbeamBackProjection(sinogram, image):
        image.fill(0.)
        dx = ximageside/nx
        dy = yimageside/ny
        x0 = -ximageside/2.
        y0 = -yimageside/2.
        u0 = -detectorlength/2.
        du = detectorlength/nbins
        ds = slen/(nviews - larc_local)

        for sindex in range(nviews):
            s = sindex*ds + slen0
            xsource = radius*cos(s)
            ysource = radius*sin(s)
            xDetCenter = (radius - source_to_detector)*cos(s)
            yDetCenter = (radius - source_to_detector)*sin(s)
            eux = -sin(s)
            euy = cos(s)

            for uindex in range(nbins):
                sinoval = sinogram[sindex, uindex]
                u = u0 + (uindex + 0.5)*du
                xbin = xDetCenter + eux*u
                ybin = yDetCenter + euy*u
                xl = x0
                yl = y0
                xdiff = xbin - xsource
                ydiff = ybin - ysource
                xad = abs(xdiff)*dy
                yad = abs(ydiff)*dx

                if (xad > yad):
                    slope = ydiff/xdiff
                    travPixlen = dx*sqrt(1.0 + slope*slope)
                    yIntOld = ysource + slope*(xl - xsource)
                    iyOld = int(floor((yIntOld - y0)/dy))
                    for ix in range(nx):
                        x = xl + dx*(ix + 1.0)
                        yIntercept = ysource + slope*(x - xsource)
                        iy = int(floor((yIntercept - y0)/dy))
                        if iy == iyOld:
                            if ((iy >= 0) and (iy < ny)):
                                image[ix, iy] = image[ix, iy] + sinoval*travPixlen
                        else:
                            yMid = dy*(iy if iy > iyOld else iyOld) + yl
                            ydist1 = abs(yMid - yIntOld)
                            ydist2 = abs(yIntercept - yMid)
                            frac1 = ydist1/(ydist1 + ydist2)
                            frac2 = 1.0 - frac1
                            if ((iyOld >= 0) and (iyOld < ny)):
                                image[ix, iyOld] = image[ix, iyOld] + frac1*sinoval*travPixlen
                            if ((iy >= 0) and (iy < ny)):
                                image[ix, iy] = image[ix, iy] + frac2*sinoval*travPixlen
                        iyOld = iy
                        yIntOld = yIntercept
                else:
                    slopeinv = xdiff/ydiff
                    travPixlen = dy*sqrt(1.0 + slopeinv*slopeinv)
                    xIntOld = xsource + slopeinv*(yl - ysource)
                    ixOld = int(floor((xIntOld - x0)/dx))
                    for iy in range(ny):
                        y = yl + dy*(iy + 1.0)
                        xIntercept = xsource + slopeinv*(y - ysource)
                        ix = int(floor((xIntercept - x0)/dx))
                        if (ix == ixOld):
                            if ((ix >= 0) and (ix < nx)):
                                image[ix, iy] = image[ix, iy] + sinoval*travPixlen
                        else:
                            xMid = dx*(ix if ix > ixOld else ixOld) + xl
                            xdist1 = abs(xMid - xIntOld)
                            xdist2 = abs(xIntercept - xMid)
                            frac1 = xdist1/(xdist1 + xdist2)
                            frac2 = 1.0 - frac1
                            if ((ixOld >= 0) and (ixOld < nx)):
                                image[ixOld, iy] = image[ixOld, iy] + frac1*sinoval*travPixlen
                            if ((ix >= 0) and (ix < nx)):
                                image[ix, iy] = image[ix, iy] + frac2*sinoval*travPixlen
                        ixOld = ix
                        xIntOld = xIntercept

    # ========================================================================
    # GRADIENT OPERATORS
    # ========================================================================

    gmatx = zeros([nx, nx])
    for i in range(nx):
        gmatx[i, i] = -1.0
    for i in range(nx-1):
        gmatx[i, i+1] = 1.0

    gmaty = zeros([ny, ny])
    for i in range(ny):
        gmaty[i, i] = -1.0
    for i in range(ny-1):
        gmaty[i, i+1] = 1.0

    def gradx(image):
        return dot(gmatx, image)

    def grady(image):
        return array(dot(gmaty, image.T).T, order="C")

    def mdivx(image):
        return dot(gmatx.T, image)

    def mdivy(image):
        return array(dot(gmaty.T, image.T).T, order="C")

    def gradim(image):
        xgrad = image.copy()
        ygrad = image.copy()
        temp = image
        xgrad[:-1, :] = temp[1:, :] - temp[:-1, :]
        ygrad[:, :-1] = temp[:, 1:] - temp[:, :-1]
        xgrad[-1, :] = -1.0*temp[-1, :]
        ygrad[:, -1] = -1.0*temp[:, -1]
        return xgrad, ygrad

    # ========================================================================
    # FILTERING OPERATORS
    # ========================================================================

    nb0 = nbins
    blen0 = detectorlength
    db = blen0/nb0
    b00 = -blen0/2.
    uar = arange(b00 + db/2., b00 + blen0, db)*1.

    def hanning_window(uar, cutoffparm):
        uhanp = abs(b00)/cutoffparm
        han = 0.5*(1.0 + cos(pi*uar/uhanp))
        han[abs(uar) > uhanp] = 0.0
        return han

    ramp = abs(uar)
    W_sqrt_ramp = sqrt(ramp + 1e-12)

    F_single = W_sqrt_ramp

    han_lo = hanning_window(uar, cutoffparm_lo)
    han_hi = 1.0 - hanning_window(uar, cutoffparm)
    han_lo = clip(han_lo, 0.0, 1.0)
    han_hi = clip(han_hi, 0.0, 1.0)
    F_lo = W_sqrt_ramp * sqrt(han_lo)
    F_hi = W_sqrt_ramp * sqrt(han_hi)

    def R_fft_weight(sino, W):
        imft = fft.fft(sino, axis=1)
        pimft = (ones([nbins])*fft.fftshift(W))*imft
        res = (1.0*fft.ifft(pimft, axis=1).real)
        return res

    def R_lo(sino):
        return R_fft_weight(sino, F_lo)

    def R_hi(sino):
        return R_fft_weight(sino, F_hi)

    def fo(sinom):
        return R_fft_weight(sinom, F_single)

    # ========================================================================
    # DATA GENERATION
    # ========================================================================

    print("Generating sinogram data...")
    sinogram = zeros([nviews, nbins])
    truesino = sinogram*0.
    circularFanbeamProjection(phimage, truesino)

    if addnoise:
        sinodata = -log(poisson(nph*exp(-truesino))/nph)
    else:
        sinodata = truesino*1.

    xgrad_t, ygrad_t = gradim(phimage)
    gim = sqrt(xgrad_t**2 + ygrad_t**2)
    truetv = gim.sum()
    print(f"Ground truth TV: {truetv:.2f}")

    # ========================================================================
    # COMPUTE OPERATOR NORMS
    # ========================================================================

    print("Computing operator norms...")

    xim = randn(nx, ny)
    xim *= mask
    npower = 50
    worksino = truesino*0.

    for i in range(npower):
        circularFanbeamProjection(xim, worksino)
        worksino_f = fo(worksino)
        xim.fill(0.)
        worksino_f = fo(worksino_f)
        circularFanbeamBackProjection(worksino_f, xim)
        xim *= mask
        xnorm2 = sqrt((xim**2.).sum())
        xim /= (xnorm2 + 1e-12)

    snorm = sqrt(xnorm2 + 1e-12)
    nusino = 1./snorm

    xim = randn(nx, ny)
    xim *= mask
    for i in range(npower):
        xg = gradx(xim)
        xim.fill(0.)
        xim = mdivx(xg)
        xim *= mask
        xnorm2 = sqrt((xim**2.).sum())
        xim /= (xnorm2 + 1e-12)
    gnorm = sqrt(xnorm2 + 1e-12)
    nuxgrad = nuxfact/gnorm

    xim = randn(nx, ny)
    xim *= mask
    for i in range(npower):
        yg = grady(xim)
        xim.fill(0.)
        xim = mdivy(yg)
        xim *= mask
        xnorm2 = sqrt((xim**2.).sum())
        xim /= (xnorm2 + 1e-12)
    gnorm = sqrt(xnorm2 + 1e-12)
    nuygrad = nuyfact/gnorm

    # ========================================================================
    # SINGLE-CHANNEL RECONSTRUCTION
    # ========================================================================

    print("\nRunning single-channel...")

    sinodata_single = fo(sinodata)
    sinodatasc_single = nusino*sinodata_single

    xim = randn(nx, ny)
    xim *= mask
    xim1 = xim*0.
    npower = 200
    worksino = truesino*0.

    for i in range(npower):
        circularFanbeamProjection(xim, worksino)
        worksino_f = fo(worksino)
        worksino_f *= nusino
        xg = gradx(xim)
        xg *= nuxgrad
        yg = grady(xim)
        yg *= nuygrad
        yim = l1f*xim
        mag1 = sqrt((yim**2).sum() + (yg**2).sum() + (xg**2).sum() + (worksino_f**2).sum())

        if mag1 > 0:
            yim /= mag1
            yg /= mag1
            xg /= mag1
            worksino_f /= mag1

        xim1.fill(0.)
        worksino_f = fo(worksino_f)
        circularFanbeamBackProjection(worksino_f, xim1)
        xim1 *= (nusino*mask)
        xim2 = mdivx(xg)
        xim2 *= (nuxgrad*mask)
        xim3 = mdivy(yg)
        xim3 *= (nuygrad*mask)
        xim = xim1 + xim2 + xim3 + l1f*yim
        mag2 = sqrt((xim**2.).sum())
        if mag2 > 0:
            xim /= mag2

    totalnorm_single = (mag1 + mag2)*0.5
    sig_single = stepbalance/totalnorm_single
    tau_single = 1./(totalnorm_single*stepbalance)
    print(f"  Total norm: {totalnorm_single:.4f}, sig: {sig_single:.6f}, tau: {tau_single:.6f}")

    # Initialize
    xim = zeros([nx, ny])
    yim = xim*0.
    xbarim = xim*0.
    wimp = xim*0.
    ysino_single = zeros([nviews, nbins])
    ygradx = zeros([nx, ny])
    ygrady = zeros([nx, ny])

    ierrs_single = []
    derrs_single = []
    tvs_single = []
    istops = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]
    theta = 1.0
    start_time = time.time()

    for itr in range(1, itermax + 1):
        ysinoold = ysino_single*1.
        ygradxold = ygradx*1.
        ygradyold = ygrady*1.
        yimold = yim*1.

        wimp.fill(0.)
        worksino = fo(ysino_single)
        circularFanbeamBackProjection(worksino, wimp)
        wimp *= nusino
        wimp *= mask
        wimqx = mdivx(ygradx)
        wimqx *= nuxgrad
        wimqx *= mask
        wimqy = mdivy(ygrady)
        wimqy *= nuygrad
        wimqy *= mask
        wiml1 = l1f*yim

        ximold = xim*1.
        xim = xim - tau_single*(wimp + wimqx + wimqy + wiml1)
        xim[xim < 0.0] = 0.
        xbarim = xim + theta*(xim - ximold)

        worksino = zeros([nviews, nbins])
        circularFanbeamProjection(xbarim, worksino)
        worksino = fo(worksino)
        worksino *= nusino
        resid = worksino - sinodatasc_single
        wdist = sqrt((resid**2).sum())
        wdistn = (wdist/nusino)/sqrt(1.*nviews*nbins)
        derrs_single.append(wdistn)

        ysino_single = ysino_single + sig_single*resid
        ymag = sqrt((ysino_single**2).sum())
        if ymag - sig_single*nusino*epssc > 0:
            ysino_single *= (ymag - sig_single*nusino*epssc)/ymag
        else:
            ysino_single *= 0.

        tgx = gradx(xbarim)
        tgx *= nuxgrad
        ptilx = ygradx + sig_single*tgx
        ptilmag = maximum(sqrt(ptilx**2), (2.-alpha))
        ygradx = (2.-alpha)*ptilx/ptilmag

        tgy = grady(xbarim)
        tgy *= nuygrad
        ptily = ygrady + sig_single*tgy
        ptilmag = maximum(sqrt(ptily**2), (alpha))
        ygrady = alpha*ptily/ptilmag

        tl1 = l1f*xbarim
        ptil1 = yim + sig_single*tl1
        ptilmag = maximum(sqrt(ptil1**2), (beta))
        yim = beta*ptil1/maximum(ptilmag, 1.e-10)

        tgx_log, tgy_log = gradim(xbarim)
        tvc = sqrt((tgx_log**2 + tgy_log**2)).sum()
        tvs_single.append(tvc)

        ygradx = ygradxold - rho*(ygradxold - ygradx)
        ygrady = ygradyold - rho*(ygradyold - ygrady)
        ysino_single = ysinoold - rho*(ysinoold - ysino_single)
        yim = yimold - rho*(yimold - yim)
        xim = ximold - rho*(ximold - xim)

        idist = sqrt(((xbarim - phimage)**2).sum()/(nx*ny))
        ierrs_single.append(idist)

        if itr in istops:
            print(f"  Iter {itr}: data_err={derrs_single[-1]:.6f}, img_err={ierrs_single[-1]:.6f}, TV={tvs_single[-1]:.2f}")

    single_time = time.time() - start_time
    xbarim_single = xbarim.copy()

    # ========================================================================
    # TWO-CHANNEL RECONSTRUCTION
    # ========================================================================

    print("Running two-channel...")

    sinodata_lo = R_lo(sinodata)
    sinodata_hi = R_hi(sinodata)
    sinodata_lo_sc = nusino*sinodata_lo
    sinodata_hi_sc = nusino*sinodata_hi

    # ------------------------------------------------------------------
    # Power iteration on the *weighted* operator  K_w = Sigma_ratio^(1/2) K
    # with K = [ R_hi A ; R_lo A ; grad_x ; grad_y ; l1f ] and
    # Sigma_ratio = diag(1, r_lo, r_tv, r_tv, r_l1) (sig_hi factored out).
    # The resulting norm estimate is  ||Sigma^(1/2) K||_2 / sqrt(sig_hi),
    # which is exactly what the convergence condition below needs.
    # ------------------------------------------------------------------
    r_lo = sigma_lo_scale
    r_tv = sigma_tv_scale
    r_l1 = sigma_l1_scale
    sqrt_r_lo = sqrt(r_lo)
    sqrt_r_tv = sqrt(r_tv)
    sqrt_r_l1 = sqrt(r_l1)

    xim = randn(nx, ny)
    xim *= mask
    xim1 = xim*0.
    npower = 200
    worksino = truesino*0.

    for i in range(npower):
        # Forward  K_w x: apply K, then scale each block by sqrt(r_block).
        circularFanbeamProjection(xim, worksino)
        s_hi = R_hi(worksino)
        s_lo = R_lo(worksino)
        s_hi *= nusino                 # block weight sqrt(r_hi) = 1
        s_lo *= (nusino*sqrt_r_lo)     # block weight sqrt(r_lo)

        xg = gradx(xim)
        xg *= (nuxgrad*sqrt_r_tv)      # block weight sqrt(r_tv)
        yg = grady(xim)
        yg *= (nuygrad*sqrt_r_tv)      # block weight sqrt(r_tv)
        yim_loc = (l1f*sqrt_r_l1)*xim  # block weight sqrt(r_l1)

        mag1 = sqrt((yim_loc**2).sum() + (yg**2).sum() + (xg**2).sum() + (s_hi**2).sum() + (s_lo**2).sum())

        if mag1 > 0:
            yim_loc /= mag1
            yg /= mag1
            xg /= mag1
            s_hi /= mag1
            s_lo /= mag1

        # Adjoint  K_w^T y: scale each block of y by sqrt(r_block) again
        # (adjoint of the diagonal weighting), then apply K^T.
        xim1.fill(0.)
        imtmp = xim1*0.
        circularFanbeamBackProjection(s_hi, imtmp)
        xim1 += imtmp
        imtmp.fill(0.0)
        circularFanbeamBackProjection(s_lo*sqrt_r_lo, imtmp)
        xim1 += imtmp
        xim1 *= (nusino*mask)

        xim2 = mdivx(xg*sqrt_r_tv)
        xim2 *= (nuxgrad*mask)
        xim3 = mdivy(yg*sqrt_r_tv)
        xim3 *= (nuygrad*mask)
        xim = xim1 + xim2 + xim3 + (l1f*sqrt_r_l1)*yim_loc
        mag2 = sqrt((xim**2.).sum())
        if mag2 > 0:
            xim /= mag2

    # norm_weighted estimates  ||Sigma_ratio^(1/2) K||_2  (sig_hi factored out).
    norm_weighted = (mag1 + mag2)*0.5

    # ------------------------------------------------------------------
    # Convergence condition (product-space CP with diagonal dual preconditioning):
    #     tau * || Sigma^(1/2) K ||_2^2  <  1
    # with  Sigma = diag(sig_hi I, sig_lo I, sig_tv I, sig_tv I, sig_l1 I)
    # and   K     = [ R_hi A ; R_lo A ; grad_x ; grad_y ; l1f ].
    # The weighted power iteration above estimates  ||Sigma^(1/2) K||_2
    # directly (sig_hi factored out as norm_weighted). Without slack the
    # assignments below would put the product at 1; cvg_slack > 0 shrinks
    # tau just enough that  tau * sig_hi * norm_weighted^2 == 1/(1+cvg_slack),
    # making the theorem hold strictly. stepbalance trades tau against sig_hi.
    # ------------------------------------------------------------------
    cvg_slack = 1.e-3
    sig_hi  = stepbalance/norm_weighted
    sig_lo  = r_lo*sig_hi
    sig_tv  = r_tv*sig_hi           # used for ygradx, ygrady dual updates
    sig_l1  = r_l1*sig_hi           # used for yim dual update
    tau_two = 1./(norm_weighted*stepbalance*(1.0 + cvg_slack))

    lhs_cvg = tau_two*sig_hi*norm_weighted**2
    assert lhs_cvg < 1.0, f"CP convergence condition violated: {lhs_cvg:.6f} >= 1"

    epssc_hi = eps_hi*sqrt(nrays)
    epssc_lo = eps_lo*sqrt(nrays)
    print(f"  ||Sigma^(1/2) K||={norm_weighted:.4f}  sig_hi={sig_hi:.6f}  "
          f"sig_lo={sig_lo:.6f}  tau={tau_two:.6f}  "
          f"tau*sig_hi*||.||^2={lhs_cvg:.6f}  (< 1 required)")

    # Initialize
    xim = zeros([nx, ny])
    yim = xim*0.
    xbarim = xim*0.
    wimp = xim*0.
    ysino_hi = zeros([nviews, nbins])
    ysino_lo = zeros([nviews, nbins])
    ygradx = zeros([nx, ny])
    ygrady = zeros([nx, ny])

    ierrs_two = []
    derrs_two = []
    tvs_two = []
    theta = 1.0
    start_time = time.time()

    for itr in range(1, itermax + 1):
        ysinoold_hi = ysino_hi*1.
        ysinoold_lo = ysino_lo*1.
        ygradxold = ygradx*1.
        ygradyold = ygrady*1.
        yimold = yim*1.

        wimp.fill(0.)
        imtmp = zeros_like(xim)
        circularFanbeamBackProjection(R_hi(ysino_hi), imtmp)
        wimp += imtmp
        imtmp.fill(0.)
        circularFanbeamBackProjection(R_lo(ysino_lo), imtmp)
        wimp += imtmp
        wimp *= nusino
        wimp *= mask

        wimqx = mdivx(ygradx)
        wimqx *= nuxgrad
        wimqx *= mask
        wimqy = mdivy(ygrady)
        wimqy *= nuygrad
        wimqy *= mask
        wiml1 = l1f*yim

        ximold = xim*1.
        xim = xim - tau_two*(wimp + wimqx + wimqy + wiml1)
        xim[xim < 0.0] = 0.
        xbarim = xim + theta*(xim - ximold)

        worksino = zeros([nviews, nbins])
        circularFanbeamProjection(xbarim, worksino)
        Ax_hi = R_hi(worksino)
        Ax_lo = R_lo(worksino)
        Ax_hi *= nusino
        Ax_lo *= nusino

        resid_hi = Ax_hi - sinodata_hi_sc
        resid_lo = Ax_lo - sinodata_lo_sc
        derr = sqrt(((resid_hi/nusino)**2).sum() + ((resid_lo/nusino)**2).sum())/sqrt(1.*nviews*nbins)
        derrs_two.append(derr)

        ysino_hi = ysino_hi + sig_hi*resid_hi
        ymag_hi = sqrt((ysino_hi**2).sum())
        if ymag_hi - sig_hi*nusino*epssc_hi > 0:
            ysino_hi *= (ymag_hi - sig_hi*nusino*epssc_hi)/ymag_hi
        else:
            ysino_hi *= 0.0

        ysino_lo = ysino_lo + sig_lo*resid_lo
        ymag_lo = sqrt((ysino_lo**2).sum())
        if ymag_lo - sig_lo*nusino*epssc_lo > 0:
            ysino_lo *= (ymag_lo - sig_lo*nusino*epssc_lo)/ymag_lo
        else:
            ysino_lo *= 0.0

        # grad_x dual: block 3 of Sigma, weight sig_tv = r_tv * sig_hi.
        tgx = gradx(xbarim)
        tgx *= nuxgrad
        ptilx = ygradx + sig_tv*tgx
        ptilmag = maximum(sqrt(ptilx**2), (2.-alpha))
        ygradx = (2.-alpha)*ptilx/ptilmag

        # grad_y dual: block 4 of Sigma, weight sig_tv = r_tv * sig_hi.
        tgy = grady(xbarim)
        tgy *= nuygrad
        ptily = ygrady + sig_tv*tgy
        ptilmag = maximum(sqrt(ptily**2), (alpha))
        ygrady = alpha*ptily/ptilmag

        # L1 dual: block 5 of Sigma, weight sig_l1 = r_l1 * sig_hi.
        tl1 = l1f*xbarim
        ptil1 = yim + sig_l1*tl1
        ptilmag = maximum(sqrt(ptil1**2), (beta))
        yim = beta*ptil1/maximum(ptilmag, 1.e-10)

        tgx_log, tgy_log = gradim(xbarim)
        tvc = sqrt((tgx_log**2 + tgy_log**2)).sum()
        tvs_two.append(tvc)

        ygradx = ygradxold - rho*(ygradxold - ygradx)
        ygrady = ygradyold - rho*(ygradyold - ygrady)
        ysino_hi = ysinoold_hi - rho*(ysinoold_hi - ysino_hi)
        ysino_lo = ysinoold_lo - rho*(ysinoold_lo - ysino_lo)
        yim = yimold - rho*(yimold - yim)
        xim = ximold - rho*(ximold - xim)

        idist = sqrt(((xbarim - phimage)**2).sum()/(nx*ny))
        ierrs_two.append(idist)

        if itr in istops:
            print(f"  Iter {itr}: data_err={derrs_two[-1]:.6f}, img_err={ierrs_two[-1]:.6f}, TV={tvs_two[-1]:.2f}")

    two_time = time.time() - start_time
    xbarim_two = xbarim.copy()

    print(f"  Single-channel: {single_time:.1f}s, final RMSE={ierrs_single[-1]:.6f}")
    print(f"  Two-channel: {two_time:.1f}s, final RMSE={ierrs_two[-1]:.6f}")

    return {
        'resolution': resolution,
        'ierrs_single': ierrs_single,
        'ierrs_two': ierrs_two,
        'final_rmse_single': ierrs_single[-1],
        'final_rmse_two': ierrs_two[-1],
        'single_time': single_time,
        'two_time': two_time,
        'xbarim_single': xbarim_single,
        'xbarim_two': xbarim_two,
        'phimage': phimage,
        'truetv': truetv,
    }


# ============================================================================
# RUN RECONSTRUCTIONS
# ============================================================================

if RUN_RECONSTRUCTION:
    for mfact in MFACT_VALUES:
        resolution = int(512/mfact)
        result = run_reconstruction_for_mfact(mfact)
        all_results[resolution] = result

    print(f"\nSaving results to '{CACHE_FILE}'...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(all_results, f)
    print("Results saved!")


# ============================================================================
# GENERATE TABLE
# ============================================================================

print("\n" + "="*60)
print("RMSE COMPARISON TABLE")
print("="*60)

print("\n| Resolution | Single-Channel RMSE | Two-Channel RMSE | Improvement |")
print("|------------|---------------------|------------------|-------------|")

for res in [512, 256, 128]:
    r = all_results[res]
    improvement = (r['final_rmse_single'] - r['final_rmse_two']) / r['final_rmse_single'] * 100
    print(f"| {res}x{res}    | {r['final_rmse_single']:.6f}            | {r['final_rmse_two']:.6f}         | {improvement:+.2f}%      |")

# Save table as text file
with open(RESULTS_DIR + 'rmse_table.txt', 'w') as f:
    f.write("RMSE Comparison: Single-Channel vs Two-Channel DTV\n")
    f.write("="*55 + "\n\n")
    f.write("| Resolution | Single-Channel RMSE | Two-Channel RMSE | Improvement |\n")
    f.write("|------------|---------------------|------------------|-------------|\n")
    for res in [512, 256, 128]:
        r = all_results[res]
        improvement = (r['final_rmse_single'] - r['final_rmse_two']) / r['final_rmse_single'] * 100
        f.write(f"| {res}x{res}    | {r['final_rmse_single']:.6f}            | {r['final_rmse_two']:.6f}         | {improvement:+.2f}%      |\n")

print(f"\nTable saved to '{RESULTS_DIR}rmse_table.txt'")


# ============================================================================
# FIGURE 1: THREE SUBPLOTS (one per resolution)
# ============================================================================

print("\nGenerating three-subplot figure...")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

for idx, res in enumerate([512, 256, 128]):
    ax = axes[idx]
    r = all_results[res]

    iterations = range(1, len(r['ierrs_single']) + 1)

    ax.semilogy(iterations, r['ierrs_single'], 'r-', linewidth=1.5, label='Single-Channel')
    ax.semilogy(iterations, r['ierrs_two'], 'b-', linewidth=1.5, label='Two-Channel')

    ax.set_xlabel('Iteration', fontsize=10)
    if idx == 0:
        ax.set_ylabel('Image RMSE', fontsize=10)
    ax.set_title(f'{res}x{res}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8, loc='upper right')

    # Set consistent y-axis range
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(RESULTS_DIR + 'convergence_subplots.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR}convergence_subplots.png")


# ============================================================================
# FIGURE 2: COMBINED SINGLE PLOT
# ============================================================================

print("Generating combined single-plot figure...")

fig, ax = plt.subplots(figsize=(5, 4))

colors = {'512': '#1f77b4', '256': '#ff7f0e', '128': '#2ca02c'}
linestyles_single = {'512': '-', '256': '-', '128': '-'}
linestyles_two = {'512': '--', '256': '--', '128': '--'}

for res in [512, 256, 128]:
    r = all_results[res]
    iterations = range(1, len(r['ierrs_single']) + 1)

    ax.semilogy(iterations, r['ierrs_single'],
                color=colors[str(res)], linestyle='-', linewidth=1.5,
                label=f'{res}x{res} Single')
    ax.semilogy(iterations, r['ierrs_two'],
                color=colors[str(res)], linestyle='--', linewidth=1.5,
                label=f'{res}x{res} Two-Ch')

ax.set_xlabel('Iteration', fontsize=10)
ax.set_ylabel('Image RMSE', fontsize=10)
ax.set_title('Convergence Comparison', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=8, loc='upper right', ncol=2)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(RESULTS_DIR + 'convergence_combined.png', dpi=300, bbox_inches='tight')
