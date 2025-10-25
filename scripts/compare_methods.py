"""
Comparison of Single-Channel vs Two-Channel DTV Reconstruction

This script runs both the original single-channel approach and the two-channel
frequency-split approach on the same phantom data and compares their convergence.

Results are cached - rerun to load from cache, use --force to recompute.
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

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Compare DTV reconstruction methods')
parser.add_argument('--force', action='store_true', help='Force recomputation')
args = parser.parse_args()

RESULTS_FILE = 'comparison_results.pkl'

# ============================================================================
# CONFIGURATION
# ============================================================================

# Phantom parameters
mfact = 2  # Image size = 512/mfact
imagenumber = 3  # Which phantom to use (0-9)

# Reconstruction parameters
addnoise = 0
nph = 1.e6
nuxfact = 0.5
nuyfact = 0.5
l1f = 1.0
eps = 0.001
larc = 1.0  # 1.0 for limited angle, 0.0 for full 360
alpha = 1.75
beta = 5.0
rho = 1.75
stepbalance = 100.0
cutoffparm = 4.0

# Two-channel parameters
cutoffparm_lo = 8.0
eps_hi = eps
eps_lo = 1.25*eps
sigma_lo_scale = 4.0

# Iteration settings
itermax = 500  # Reduced for comparison
istops = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]

# ============================================================================
# CHECK FOR CACHED RESULTS
# ============================================================================

if not args.force and os.path.exists(RESULTS_FILE):
    print(f"\nFound cached results in '{RESULTS_FILE}'")
    print("Loading cached results (use --force to recompute)...\n")
    try:
        with open(RESULTS_FILE, 'rb') as f:
            cached = pickle.load(f)

        # Extract all results
        derrs_single = cached['derrs_single']
        ierrs_single = cached['ierrs_single']
        tvs_single = cached['tvs_single']
        xbarim_single = cached['xbarim_single']
        single_time = cached['single_time']
        derrs_two = cached['derrs_two']
        ierrs_two = cached['ierrs_two']
        tvs_two = cached['tvs_two']
        xbarim_two = cached['xbarim_two']
        two_time = cached['two_time']
        phimage = cached['phimage']
        truetv = cached['truetv']

        print("Cached results loaded! Skipping to visualization...")
        RUN_RECONSTRUCTION = False
    except Exception as e:
        print(f"Error loading cache: {e}. Running full reconstruction...")
        RUN_RECONSTRUCTION = True
else:
    if args.force:
        print("\n--force specified, running full reconstruction...")
    RUN_RECONSTRUCTION = True

# ============================================================================
# LOAD PHANTOM DATA
# ============================================================================

if RUN_RECONSTRUCTION:
    print("\nLoading phantom data...")
    phantom1 = load("Phantom_Adipose.npy")[imagenumber]
    phantom2 = load("Phantom_Fibroglandular.npy")[imagenumber]
    phantom3 = load("Phantom_Calcification.npy")[imagenumber]
    
    testimage = (0.5*phantom1 + 1.0*phantom2 + 2.0*phantom3).astype("float64")
    testimage = testimage[::mfact, ::mfact]*1.
    
    phimage = testimage*1.
    
    # Image parameters
    ximageside = 10.0  # cm
    yimageside = 10.0  # cm
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
    radius = 50.0  # cm
    source_to_detector = 100.0  # cm
    srad = radius
    sd = source_to_detector
    slen = (50./180.)*pi  # 50 degree angular range
    slen0 = -slen/2.0
    ns0 = 25
    nu0 = 1024
    nviews = ns0
    nbins = nu0
    nrays = nbins*nviews
    epssc = eps*sqrt(nrays)
    
    fanangle2 = arcsin((ximageside/2.)/radius)
    detectorlength = 2.*tan(fanangle2)*source_to_detector
    
    # ============================================================================
    # PROJECTION/BACKPROJECTION OPERATORS
    # ============================================================================
    
    @njit
    def circularFanbeamProjection(image, sinogram,
                                  nx=nx, ny=ny, ximageside=ximageside, yimageside=yimageside,
                                  radius=srad, source_to_detector=sd, detectorlength=detectorlength,
                                  nviews=ns0, slen=slen, slen0=slen0, nbins=nu0):
        dx = ximageside/nx
        dy = yimageside/ny
        x0 = -ximageside/2.
        y0 = -yimageside/2.
        u0 = -detectorlength/2.
        du = detectorlength/nbins
        ds = slen/(nviews-larc)
    
        for sindex in range(nviews):
            s = sindex*ds + slen0
            xsource = radius*cos(s)
            ysource = radius*sin(s)
            xDetCenter = (radius - source_to_detector)*cos(s)
            yDetCenter = (radius - source_to_detector)*sin(s)
            eux = -sin(s)
            euy = cos(s)
            ewx = cos(s)
            ewy = sin(s)
    
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
    
    
    @njit(cache=True)
    def circularFanbeamBackProjection(sinogram, image,
                                      nx=nx, ny=ny, ximageside=ximageside, yimageside=yimageside,
                                      radius=srad, source_to_detector=sd, detectorlength=detectorlength,
                                      nviews=ns0, slen=slen, slen0=slen0, nbins=nu0):
        image.fill(0.)
        dx = ximageside/nx
        dy = yimageside/ny
        x0 = -ximageside/2.
        y0 = -yimageside/2.
        u0 = -detectorlength/2.
        du = detectorlength/nbins
        ds = slen/(nviews - larc)
    
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
    
    
    # ============================================================================
    # GRADIENT OPERATORS
    # ============================================================================
    
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
    
    
    # ============================================================================
    # FILTERING OPERATORS
    # ============================================================================
    
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
    
    # Single-channel filter
    F_single = W_sqrt_ramp
    
    # Two-channel filters
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
    
    
    # ============================================================================
    # DATA GENERATION
    # ============================================================================
    
    print("Generating sinogram data...")
    sinogram = zeros([nviews, nbins])
    truesino = sinogram*0.
    circularFanbeamProjection(phimage, truesino)
    
    if addnoise:
        sinodata = -log(poisson(nph*exp(-truesino))/nph)
    else:
        sinodata = truesino*1.
    
    # Ground truth TV
    xgrad_t = gradx(phimage)
    gim = sqrt(xgrad_t**2)
    truetvx = gim.sum()
    
    ygrad_t = grady(phimage)
    gim = sqrt(ygrad_t**2)
    truetvy = gim.sum()
    
    xgrad_t, ygrad_t = gradim(phimage)
    gim = sqrt(xgrad_t**2 + ygrad_t**2)
    truetv = gim.sum()
    
    print(f"Ground truth TV: {truetv:.2f}")
    
    
    # ============================================================================
    # COMPUTE OPERATOR NORMS (shared by both methods)
    # ============================================================================
    
    print("Computing operator norms...")
    
    # Sino norm (using single channel for both)
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
    
    # Grad norms
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
    
    
    # ============================================================================
    # SINGLE-CHANNEL RECONSTRUCTION
    # ============================================================================
    
    print("\n" + "="*60)
    print("RUNNING SINGLE-CHANNEL RECONSTRUCTION")
    print("="*60)
    
    sinodata_single = fo(sinodata)
    sinodatasc_single = nusino*sinodata_single
    
    # Compute total norm for single channel
    xim = randn(nx, ny)
    xim *= mask
    xim1 = xim*0.
    xim2 = xim*0.
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
    
    print(f"Total norm: {totalnorm_single:.4f}")
    print(f"sig: {sig_single:.6f}, tau: {tau_single:.6f}")
    
    # Initialize
    xim = zeros([nx, ny])
    yim = xim*0.
    xbarim = xim*0.
    wimp = xim*0.
    ysino_single = zeros([nviews, nbins])
    ygradx = zeros([nx, ny])
    ygrady = zeros([nx, ny])
    
    derrs_single = []
    ierrs_single = []
    tvxs_single = []
    tvys_single = []
    tvs_single = []
    
    theta = 1.0
    start_time = time.time()
    
    for itr in range(1, itermax + 1):
        ysinoold = ysino_single*1.
        ygradxold = ygradx*1.
        ygradyold = ygrady*1.
        yimold = yim*1.
    
        # Primal update
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
    
        # Dual updates
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
        tvc = sqrt((tgx**2)).sum()/nuxgrad
        tvxs_single.append(tvc)
        ptilx = ygradx + sig_single*tgx
        ptilmag = maximum(sqrt(ptilx**2), (2.-alpha))
        ygradx = (2.-alpha)*ptilx/ptilmag
    
        tgy = grady(xbarim)
        tgy *= nuygrad
        tvc = sqrt((tgy**2)).sum()/nuygrad
        tvys_single.append(tvc)
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
    
        # Predictor-corrector
        ygradx = ygradxold - rho*(ygradxold - ygradx)
        ygrady = ygradyold - rho*(ygradyold - ygrady)
        ysino_single = ysinoold - rho*(ysinoold - ysino_single)
        yim = yimold - rho*(yimold - yim)
        xim = ximold - rho*(ximold - xim)
    
        idist = sqrt(((xbarim - phimage)**2).sum()/(nx*ny))
        ierrs_single.append(idist)
    
        if itr in istops:
            print(f"Iter {itr}: data_err={derrs_single[-1]:.6f}, img_err={ierrs_single[-1]:.6f}, TV={tvs_single[-1]:.2f}")
    
    single_time = time.time() - start_time
    xbarim_single = xbarim.copy()
    print(f"Single-channel completed in {single_time:.2f} seconds")
    
    
    # ============================================================================
    # TWO-CHANNEL RECONSTRUCTION
    # ============================================================================
    
    print("\n" + "="*60)
    print("RUNNING TWO-CHANNEL RECONSTRUCTION")
    print("="*60)
    
    sinodata_lo = R_lo(sinodata)
    sinodata_hi = R_hi(sinodata)
    sinodata_lo_sc = nusino*sinodata_lo
    sinodata_hi_sc = nusino*sinodata_hi
    
    # Compute total norm for two channels
    xim = randn(nx, ny)
    xim *= mask
    xim1 = xim*0.
    xim2 = xim*0.
    npower = 200
    worksino = truesino*0.
    
    for i in range(npower):
        circularFanbeamProjection(xim, worksino)
        s_hi = R_hi(worksino)
        s_lo = R_lo(worksino)
        s_hi *= nusino
        s_lo *= nusino
    
        xg = gradx(xim)
        xg *= nuxgrad
        yg = grady(xim)
        yg *= nuygrad
        yim_loc = l1f*xim
    
        mag1 = sqrt((yim_loc**2).sum() + (yg**2).sum() + (xg**2).sum() + (s_hi**2).sum() + (s_lo**2).sum())
    
        if mag1 > 0:
            yim_loc /= mag1
            yg /= mag1
            xg /= mag1
            s_hi /= mag1
            s_lo /= mag1
    
        xim1.fill(0.)
        imtmp = xim1*0.
        circularFanbeamBackProjection(s_hi, imtmp)
        xim1 += imtmp
        imtmp.fill(0.0)
        circularFanbeamBackProjection(s_lo, imtmp)
        xim1 += imtmp
        xim1 *= (nusino*mask)
    
        xim2 = mdivx(xg)
        xim2 *= (nuxgrad*mask)
        xim3 = mdivy(yg)
        xim3 *= (nuygrad*mask)
        xim = xim1 + xim2 + xim3 + l1f*yim_loc
        mag2 = sqrt((xim**2.).sum())
        if mag2 > 0:
            xim /= mag2
    
    totalnorm_two = (mag1 + mag2)*0.5
    sig_two = stepbalance/totalnorm_two
    tau_two = 1./(totalnorm_two*stepbalance)
    sig_hi = sig_two
    sig_lo = sigma_lo_scale*sig_two
    
    epssc_hi = eps_hi*sqrt(nrays)
    epssc_lo = eps_lo*sqrt(nrays)
    
    print(f"Total norm: {totalnorm_two:.4f}")
    print(f"sig_hi: {sig_hi:.6f}, sig_lo: {sig_lo:.6f}, tau: {tau_two:.6f}")
    
    # Initialize
    xim = zeros([nx, ny])
    yim = xim*0.
    xbarim = xim*0.
    wimp = xim*0.
    ysino_hi = zeros([nviews, nbins])
    ysino_lo = zeros([nviews, nbins])
    ygradx = zeros([nx, ny])
    ygrady = zeros([nx, ny])
    
    derrs_two = []
    ierrs_two = []
    tvxs_two = []
    tvys_two = []
    tvs_two = []
    
    theta = 1.0
    start_time = time.time()
    
    for itr in range(1, itermax + 1):
        ysinoold_hi = ysino_hi*1.
        ysinoold_lo = ysino_lo*1.
        ygradxold = ygradx*1.
        ygradyold = ygrady*1.
        yimold = yim*1.
    
        # Primal update
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
    
        # Dual updates
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
    
        tgx = gradx(xbarim)
        tgx *= nuxgrad
        tvc = sqrt((tgx**2)).sum()/nuxgrad
        tvxs_two.append(tvc)
        ptilx = ygradx + sig_two*tgx
        ptilmag = maximum(sqrt(ptilx**2), (2.-alpha))
        ygradx = (2.-alpha)*ptilx/ptilmag
    
        tgy = grady(xbarim)
        tgy *= nuygrad
        tvc = sqrt((tgy**2)).sum()/nuygrad
        tvys_two.append(tvc)
        ptily = ygrady + sig_two*tgy
        ptilmag = maximum(sqrt(ptily**2), (alpha))
        ygrady = alpha*ptily/ptilmag
    
        tl1 = l1f*xbarim
        ptil1 = yim + sig_two*tl1
        ptilmag = maximum(sqrt(ptil1**2), (beta))
        yim = beta*ptil1/maximum(ptilmag, 1.e-10)
    
        tgx_log, tgy_log = gradim(xbarim)
        tvc = sqrt((tgx_log**2 + tgy_log**2)).sum()
        tvs_two.append(tvc)
    
        # Predictor-corrector
        ygradx = ygradxold - rho*(ygradxold - ygradx)
        ygrady = ygradyold - rho*(ygradyold - ygrady)
        ysino_hi = ysinoold_hi - rho*(ysinoold_hi - ysino_hi)
        ysino_lo = ysinoold_lo - rho*(ysinoold_lo - ysino_lo)
        yim = yimold - rho*(yimold - yim)
        xim = ximold - rho*(ximold - xim)
    
        idist = sqrt(((xbarim - phimage)**2).sum()/(nx*ny))
        ierrs_two.append(idist)
    
        if itr in istops:
            print(f"Iter {itr}: data_err={derrs_two[-1]:.6f}, img_err={ierrs_two[-1]:.6f}, TV={tvs_two[-1]:.2f}")
    
    two_time = time.time() - start_time
    xbarim_two = xbarim.copy()
    print(f"Two-channel completed in {two_time:.2f} seconds")

    # Save results
    print(f"\nSaving results to '{RESULTS_FILE}'...")
    results = {
        'derrs_single': derrs_single,
        'ierrs_single': ierrs_single,
        'tvs_single': tvs_single,
        'xbarim_single': xbarim_single,
        'single_time': single_time,
        'derrs_two': derrs_two,
        'ierrs_two': ierrs_two,
        'tvs_two': tvs_two,
        'xbarim_two': xbarim_two,
        'two_time': two_time,
        'phimage': phimage,
        'truetv': truetv,
    }
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved! Next run will load from cache.\n")


# ============================================================================
# COMPARISON PLOTS
# ============================================================================

print("\n" + "="*60)
print("GENERATING COMPARISON PLOTS")
print("="*60)

fig = plt.figure(figsize=(16, 10))

# Convergence plots
ax1 = plt.subplot(2, 3, 1)
plt.plot(derrs_single, 'b-', label='Single-channel', linewidth=2)
plt.plot(derrs_two, 'r-', label='Two-channel', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Data RMSE', fontsize=12)
plt.yscale('log')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.title('Data Fidelity Convergence', fontsize=13, fontweight='bold')

ax2 = plt.subplot(2, 3, 2)
plt.plot(ierrs_single, 'b-', label='Single-channel', linewidth=2)
plt.plot(ierrs_two, 'r-', label='Two-channel', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Image RMSE', fontsize=12)
plt.yscale('log')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.title('Image Error Convergence', fontsize=13, fontweight='bold')

# TV plots
ax3 = plt.subplot(2, 3, 3)
plt.plot(tvs_single, 'b-', label='Single-channel', linewidth=2)
plt.plot(tvs_two, 'r-', label='Two-channel', linewidth=2)
plt.axhline(y=truetv, color='g', linestyle='--', label='Ground truth', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Total TV', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.title('Total Variation', fontsize=13, fontweight='bold')

# Reconstructed images
ax4 = plt.subplot(2, 3, 4)
im1 = plt.imshow(xbarim_single.T, cmap='gray', origin='lower', vmin=0, vmax=2)
plt.colorbar(im1, ax=ax4)
plt.title('Single-Channel Reconstruction', fontsize=13, fontweight='bold')
plt.axis('off')

ax5 = plt.subplot(2, 3, 5)
im2 = plt.imshow(xbarim_two.T, cmap='gray', origin='lower', vmin=0, vmax=2)
plt.colorbar(im2, ax=ax5)
plt.title('Two-Channel Reconstruction', fontsize=13, fontweight='bold')
plt.axis('off')

ax6 = plt.subplot(2, 3, 6)
im3 = plt.imshow(phimage.T, cmap='gray', origin='lower', vmin=0, vmax=2)
plt.colorbar(im3, ax=ax6)
plt.title('Ground Truth Phantom', fontsize=13, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
print("Saved comparison plot to 'method_comparison.png'")
plt.show()

# Summary statistics
print("\n" + "="*60)
print("FINAL COMPARISON SUMMARY")
print("="*60)
print(f"\nSingle-channel:")
print(f"  Final data RMSE: {derrs_single[-1]:.6f}")
print(f"  Final image RMSE: {ierrs_single[-1]:.6f}")
print(f"  Final TV: {tvs_single[-1]:.2f} (true: {truetv:.2f})")
print(f"  Runtime: {single_time:.2f} seconds")

print(f"\nTwo-channel:")
print(f"  Final data RMSE: {derrs_two[-1]:.6f}")
print(f"  Final image RMSE: {ierrs_two[-1]:.6f}")
print(f"  Final TV: {tvs_two[-1]:.2f} (true: {truetv:.2f})")
print(f"  Runtime: {two_time:.2f} seconds")

print(f"\nImprovement:")
print(f"  Data RMSE: {(derrs_single[-1] - derrs_two[-1])/derrs_single[-1]*100:.2f}%")
print(f"  Image RMSE: {(ierrs_single[-1] - ierrs_two[-1])/ierrs_single[-1]*100:.2f}%")
print(f"  TV error: {abs(tvs_single[-1]-truetv):.2f} vs {abs(tvs_two[-1]-truetv):.2f}")
