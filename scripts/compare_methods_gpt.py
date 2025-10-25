# compare_methods.py
"""
Comparison of Single-Channel vs Two-Channel DTV Reconstruction

Runs the original single-channel approach and the two-channel
(low+high) fidelity split on the same DBT phantom and compares convergence.

Saves:
  - method_comparison.png
  - comparison_results.pkl (cache)

Usage:
  python compare_methods.py         # load from cache if present
  python compare_methods.py --force # recompute
"""
import numpy as np
from numpy import *
from numpy.random import randn, poisson
import matplotlib.pyplot as plt
import time, os, pickle, argparse
from numba import njit

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Compare DTV reconstruction methods')
parser.add_argument('--force', action='store_true', help='Force recomputation (ignore cache)')
args = parser.parse_args()
RESULTS_FILE = 'comparison_results.pkl'

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
# Phantom
mfact = 2
imagenumber = 3

# Recon (shared)
addnoise = 0
nph = 1.e6
nuxfact = 0.5
nuyfact = 0.5
l1f = 1.0
eps = 0.001
larc = 1.0          # 1.0 => limited angle
alpha = 1.75
beta = 5.0
rho = 1.75
stepbalance = 100.0
cutoffparm = 4.0

# Two-channel fidelity
cutoffparm_lo = 8.0
eps_hi = eps
eps_lo = 1.25*eps
sigma_lo_scale = 4.0

# Iterations / checkpoints
itermax = 500
istops = [1,2,5,10,20,50,100,200,300,400,500]

# -----------------------------------------------------------------------------
# Load or run
# -----------------------------------------------------------------------------
if (not args.force) and os.path.exists(RESULTS_FILE):
    print(f"\nFound cache '{RESULTS_FILE}', loading (use --force to recompute)...\n")
    with open(RESULTS_FILE, 'rb') as f:
        cached = pickle.load(f)
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
else:
    # -------------------------------------------------------------------------
    # PHANTOM + GEOMETRY
    # -------------------------------------------------------------------------
    print("\nLoading phantom...")
    phantom1 = load("Phantom_Adipose.npy")[imagenumber]
    phantom2 = load("Phantom_Fibroglandular.npy")[imagenumber]
    phantom3 = load("Phantom_Calcification.npy")[imagenumber]
    testimage = (0.5*phantom1 + 1.0*phantom2 + 2.0*phantom3).astype("float64")
    phimage = testimage[::mfact, ::mfact]*1.

    ximageside = 10.0
    yimageside = 10.0
    nx = int(512/mfact); ny = int(512/mfact)
    dx = ximageside/nx;   dy = yimageside/ny

    xar = arange(-ximageside/2.+dx/2, ximageside/2., dx)[:, newaxis]*ones([ny])
    yar = ones([nx, ny])*arange(-yimageside/2.+dy/2, yimageside/2., dy)
    rar = sqrt(xar**2 + yar**2)
    mask = phimage*0.; mask[rar <= ximageside/2.] = 1.

    # Fan-beam setup
    radius = 50.0
    source_to_detector = 100.0
    srad = radius; sd = source_to_detector
    slen = (50./180.)*pi
    slen0 = -slen/2.0
    ns0 = 25; nu0 = 1024
    nviews = ns0; nbins = nu0
    nrays = nviews*nbins
    epssc = eps*sqrt(nrays)
    fanangle2 = arcsin((ximageside/2.)/radius)
    detectorlength = 2.*tan(fanangle2)*source_to_detector

    # -------------------------------------------------------------------------
    # PROJECTOR / BACKPROJECTOR
    # -------------------------------------------------------------------------
    @njit
    def circularFanbeamProjection(image, sinogram,
                                  nx=nx, ny=ny, ximageside=ximageside, yimageside=yimageside,
                                  radius=srad, source_to_detector=sd, detectorlength=detectorlength,
                                  nviews=ns0, slen=slen, slen0=slen0, nbins=nu0):
        dx = ximageside/nx; dy = yimageside/ny
        x0 = -ximageside/2.; y0 = -yimageside/2.
        u0 = -detectorlength/2.; du = detectorlength/nbins
        ds = slen/(nviews - larc)
        for sindex in range(nviews):
            s = sindex*ds + slen0
            xsource = radius*cos(s); ysource = radius*sin(s)
            xDetCenter = (radius - source_to_detector)*cos(s)
            yDetCenter = (radius - source_to_detector)*sin(s)
            eux = -sin(s); euy = cos(s); ewx = cos(s); ewy = sin(s)
            for uindex in range(nbins):
                u = u0 + (uindex + 0.5)*du
                xbin = xDetCenter + eux*u; ybin = yDetCenter + euy*u
                xl = -ximageside/2.; yl = -yimageside/2.
                xdiff = xbin - xsource; ydiff = ybin - ysource
                xad = abs(xdiff)*dy;    yad = abs(ydiff)*dx
                raysum = 0.0
                if xad > yad:
                    slope = ydiff/xdiff
                    trav = dx*sqrt(1.0+slope*slope)
                    yIntOld = ysource + slope*(xl - xsource)
                    iyOld = int(floor((yIntOld - y0)/dy))
                    for ix in range(nx):
                        x = xl + dx*(ix+1.0)
                        yIntercept = ysource + slope*(x - xsource)
                        iy = int(floor((yIntercept - y0)/dy))
                        if iy == iyOld:
                            if 0 <= iy < ny: raysum += trav*image[ix, iy]
                        else:
                            yMid = dy*(iy if iy>iyOld else iyOld) + yl
                            ydist1 = abs(yMid - yIntOld); ydist2 = abs(yIntercept - yMid)
                            frac1 = ydist1/(ydist1+ydist2); frac2 = 1.0 - frac1
                            if 0 <= iyOld < ny: raysum += frac1*trav*image[ix, iyOld]
                            if 0 <= iy    < ny: raysum += frac2*trav*image[ix, iy]
                        iyOld = iy; yIntOld = yIntercept
                else:
                    slopeinv = xdiff/ydiff
                    trav = dy*sqrt(1.0+slopeinv*slopeinv)
                    xIntOld = xsource + slopeinv*(yl - ysource)
                    ixOld = int(floor((xIntOld - x0)/dx))
                    for iy in range(ny):
                        y = yl + dy*(iy+1.0)
                        xIntercept = xsource + slopeinv*(y - ysource)
                        ix = int(floor((xIntercept - x0)/dx))
                        if ix == ixOld:
                            if 0 <= ix < nx: raysum += trav*image[ix, iy]
                        else:
                            xMid = dx*(ix if ix>ixOld else ixOld) + xl
                            xdist1 = abs(xMid - xIntOld); xdist2 = abs(xIntercept - xMid)
                            frac1 = xdist1/(xdist1+xdist2); frac2 = 1.0 - frac1
                            if 0 <= ixOld < nx: raysum += frac1*trav*image[ixOld, iy]
                            if 0 <= ix    < nx: raysum += frac2*trav*image[ix, iy]
                        ixOld = ix; xIntOld = xIntercept
                sinogram[sindex, uindex] = raysum

    @njit(cache=True)
    def circularFanbeamBackProjection(sinogram, image,
                                      nx=nx, ny=ny, ximageside=ximageside, yimageside=yimageside,
                                      radius=srad, source_to_detector=sd, detectorlength=detectorlength,
                                      nviews=ns0, slen=slen, slen0=slen0, nbins=nu0):
        image.fill(0.)
        dx = ximageside/nx; dy = yimageside/ny
        x0 = -ximageside/2.; y0 = -yimageside/2.
        u0 = -detectorlength/2.; du = detectorlength/nbins
        ds = slen/(nviews - larc)
        for sindex in range(nviews):
            s = sindex*ds + slen0
            xsource = radius*cos(s); ysource = radius*sin(s)
            xDetCenter = (radius - source_to_detector)*cos(s)
            yDetCenter = (radius - source_to_detector)*sin(s)
            eux = -sin(s); euy = cos(s)
            for uindex in range(nbins):
                val = sinogram[sindex, uindex]
                u = u0 + (uindex + 0.5)*du
                xbin = xDetCenter + eux*u; ybin = yDetCenter + euy*u
                xl = -ximageside/2.; yl = -yimageside/2.
                xdiff = xbin - xsource; ydiff = ybin - ysource
                xad = abs(xdiff)*dy;    yad = abs(ydiff)*dx
                if xad > yad:
                    slope = ydiff/xdiff
                    trav = dx*sqrt(1.0+slope*slope)
                    yIntOld = ysource + slope*(xl - xsource)
                    iyOld = int(floor((yIntOld - y0)/dy))
                    for ix in range(nx):
                        x = xl + dx*(ix+1.0)
                        yIntercept = ysource + slope*(x - xsource)
                        iy = int(floor((yIntercept - y0)/dy))
                        if iy == iyOld:
                            if 0 <= iy < ny: image[ix, iy] += val*trav
                        else:
                            yMid = dy*(iy if iy>iyOld else iyOld) + yl
                            ydist1 = abs(yMid - yIntOld); ydist2 = abs(yIntercept - yMid)
                            frac1 = ydist1/(ydist1+ydist2); frac2 = 1.0 - frac1
                            if 0 <= iyOld < ny: image[ix, iyOld] += frac1*val*trav
                            if 0 <= iy    < ny: image[ix, iy]    += frac2*val*trav
                        iyOld = iy; yIntOld = yIntercept
                else:
                    slopeinv = xdiff/ydiff
                    trav = dy*sqrt(1.0+slopeinv*slopeinv)
                    xIntOld = xsource + slopeinv*(yl - ysource)
                    ixOld = int(floor((xIntOld - x0)/dx))
                    for iy in range(ny):
                        y = yl + dy*(iy+1.0)
                        xIntercept = xsource + slopeinv*(y - ysource)
                        ix = int(floor((xIntercept - x0)/dx))
                        if ix == ixOld:
                            if 0 <= ix < nx: image[ix, iy] += val*trav
                        else:
                            xMid = dx*(ix if ix>ixOld else ixOld) + xl
                            xdist1 = abs(xMid - xIntOld); xdist2 = abs(xIntercept - xMid)
                            frac1 = xdist1/(xdist1+xdist2); frac2 = 1.0 - frac1
                            if 0 <= ixOld < nx: image[ixOld, iy] += frac1*val*trav
                            if 0 <= ix    < nx: image[ix, iy]    += frac2*val*trav
                        ixOld = ix; xIntOld = xIntercept

    # -------------------------------------------------------------------------
    # GRAD / DIV
    # -------------------------------------------------------------------------
    gmatx = zeros([nx, nx]); gmatx[range(nx), range(nx)] = -1.0; gmatx[range(nx-1), range(1,nx)] = 1.0
    gmaty = zeros([ny, ny]); gmaty[range(ny), range(ny)] = -1.0; gmaty[range(ny-1), range(1,ny)] = 1.0
    def gradx(im): return dot(gmatx, im)
    def grady(im): return array(dot(gmaty, im.T).T, order="C")
    def mdivx(im): return dot(gmatx.T, im)
    def mdivy(im): return array(dot(gmaty.T, im.T).T, order="C")
    def gradim(im):
        xg = im.copy(); yg = im.copy(); t = im
        xg[:-1,:] = t[1:,:] - t[:-1,:]; xg[-1,:] = -t[-1,:]
        yg[:,:-1] = t[:,1:] - t[:,:-1]; yg[:,-1] = -t[:,-1]
        return xg, yg

    # -------------------------------------------------------------------------
    # FILTERS (FFT along detector channels)
    # -------------------------------------------------------------------------
    nb0 = nbins; blen0 = detectorlength; db = blen0/nb0; b00 = -blen0/2.
    uar = arange(b00+db/2., b00+blen0, db)*1.
    def hanning_window(uar, c):
        uhanp = abs(b00)/c
        han = 0.5*(1.0 + cos(pi*uar/uhanp))
        han[abs(uar) > uhanp] = 0.0
        return han
    ramp = abs(uar); W_sqrt_ramp = sqrt(ramp + 1e-12)
    F_single = W_sqrt_ramp
    han_lo = clip(hanning_window(uar, cutoffparm_lo), 0.0, 1.0)
    han_hi = clip(1.0 - hanning_window(uar, cutoffparm), 0.0, 1.0)
    F_lo = W_sqrt_ramp*sqrt(han_lo)
    F_hi = W_sqrt_ramp*sqrt(han_hi)
    def R_fft_weight(sino, W):
        imft = np.fft.fft(sino, axis=1)
        pimft = (ones([nbins])*np.fft.fftshift(W))*imft
        return np.fft.ifft(pimft, axis=1).real
    def R_lo(s): return R_fft_weight(s, F_lo)
    def R_hi(s): return R_fft_weight(s, F_hi)
    def fo(s):   return R_fft_weight(s, F_single)

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------
    print("Generating forward data...")
    truesino = zeros([nviews, nbins]); circularFanbeamProjection(phimage, truesino)
    sinodata = -log(poisson(nph*exp(-truesino))/nph) if addnoise else truesino*1.

    # TV truth
    xg = gradx(phimage); truetvx = sqrt(xg**2).sum()
    yg = grady(phimage); truetvy = sqrt(yg**2).sum()
    xg, yg = gradim(phimage); truetv = sqrt(xg**2 + yg**2).sum()
    print(f"Ground truth TV: {truetv:.2f}")

    # -------------------------------------------------------------------------
    # NORMS (nusino, nuxgrad, nuygrad)
    # -------------------------------------------------------------------------
    print("Estimating operator norms...")
    xim = randn(nx, ny)*mask; worksino = truesino*0.; npower = 50
    for _ in range(npower):
        circularFanbeamProjection(xim, worksino)
        worksino_f = fo(fo(worksino))   # keep same scaling as single
        xim *= 0.; circularFanbeamBackProjection(worksino_f, xim); xim *= mask
        xnorm2 = sqrt((xim**2.).sum()); xim /= (xnorm2 + 1e-12)
    snorm = sqrt(xnorm2 + 1e-12); nusino = 1./snorm

    xim = randn(nx, ny)*mask
    for _ in range(npower):
        xg = gradx(xim); xim *= 0.; xim = mdivx(xg); xim *= mask
        xnorm2 = sqrt((xim**2.).sum()); xim /= (xnorm2 + 1e-12)
    gnorm = sqrt(xnorm2 + 1e-12); nuxgrad = nuxfact/gnorm

    xim = randn(nx, ny)*mask
    for _ in range(npower):
        yg = grady(xim); xim *= 0.; xim = mdivy(yg); xim *= mask
        xnorm2 = sqrt((xim**2.).sum()); xim /= (xnorm2 + 1e-12)
    gnorm = sqrt(xnorm2 + 1e-12); nuygrad = nuyfact/gnorm

    # --------------------------- SINGLE-CHANNEL -------------------------------
    print("\n" + "="*60); print("RUNNING SINGLE-CHANNEL"); print("="*60)
    sinodata_single = fo(sinodata); sinodatasc_single = nusino*sinodata_single

    # total norm (power iter)
    xim = randn(nx, ny)*mask; xim1 = xim*0.; xim2 = xim*0.; worksino = truesino*0.
    for _ in range(200):
        circularFanbeamProjection(xim, worksino)
        w = fo(worksino); w *= nusino
        xg = gradx(xim)*nuxgrad; yg = grady(xim)*nuygrad; yimloc = l1f*xim
        mag1 = sqrt((yimloc**2).sum() + (yg**2).sum() + (xg**2).sum() + (w**2).sum())
        if mag1>0: yimloc/=mag1; yg/=mag1; xg/=mag1; w/=mag1
        xim1 *= 0.; circularFanbeamBackProjection(fo(w), xim1); xim1 *= (nusino*mask)
        xim2 = mdivx(xg)*(nuxgrad*mask); xim3 = mdivy(yg)*(nuygrad*mask)
        xim = xim1 + xim2 + xim3 + l1f*yimloc
        mag2 = sqrt((xim**2.).sum()); 
        if mag2>0: xim /= mag2
    totalnorm_single = (mag1 + mag2)*0.5
    sig_single = stepbalance/totalnorm_single
    tau_single = 1./(totalnorm_single*stepbalance)
    print(f"Total norm(single)={totalnorm_single:.4f}  sig={sig_single:.6f}  tau={tau_single:.6f}")

    # init
    xim = zeros([nx,ny]); yim = xim*0.; xbarim = xim*0.; wimp = xim*0.
    ysino_single = zeros([nviews, nbins]); ygradx = zeros([nx,ny]); ygrady = zeros([nx,ny])
    derrs_single=[]; ierrs_single=[]; tvs_single=[]

    t0 = time.time()
    for itr in range(1, itermax+1):
        ysinoold = ysino_single.copy(); ygradxold=ygradx.copy(); ygradyold=ygrady.copy(); yimold=yim.copy()
        # primal
        wimp *= 0.; circularFanbeamBackProjection(fo(ysino_single), wimp); wimp *= nusino; wimp *= mask
        wimqx = mdivx(ygradx)*nuxgrad*mask; wimqy = mdivy(ygrady)*nuygrad*mask; wiml1 = l1f*yim
        ximold = xim.copy()
        xim = xim - tau_single*(wimp + wimqx + wimqy + wiml1)
        xim[xim<0] = 0.; xbarim = xim + (xim - ximold)
        # dual
        worksino = zeros([nviews, nbins]); circularFanbeamProjection(xbarim, worksino)
        w = fo(worksino); w *= nusino
        resid = w - sinodatasc_single
        wdist = sqrt((resid**2).sum()); derrs_single.append((wdist/nusino)/sqrt(1.*nviews*nbins))
        ysino_single = ysino_single + sig_single*resid
        ymag = sqrt((ysino_single**2).sum())
        ysino_single *= (np.maximum(0.0, ymag - sig_single*nusino*epssc)/(ymag+1e-12))
        tgx = gradx(xbarim)*nuxgrad; ptilx = ygradx + sig_single*tgx
        ygradx = (2.-alpha)*ptilx/maximum(abs(ptilx), (2.-alpha))
        tgy = grady(xbarim)*nuygrad; ptily = ygrady + sig_single*tgy
        ygrady = alpha*ptily/maximum(abs(ptily), alpha)
        ptil1 = yim + sig_single*(l1f*xbarim); yim = beta*ptil1/maximum(sqrt(ptil1**2), 1e-10)
        tvs_single.append(sqrt((gradim(xbarim)[0]**2 + gradim(xbarim)[1]**2)).sum())
        # predictor-corrector
        ygradx = ygradxold - rho*(ygradxold - ygradx)
        ygrady = ygradyold - rho*(ygradyold - ygrady)
        ysino_single = ysinoold - rho*(ysinoold - ysino_single)
        yim = yimold - rho*(yimold - yim)
        xim = ximold - rho*(ximold - xim)
        ierrs_single.append(sqrt(((xbarim - phimage)**2).sum()/(nx*ny)))
        if itr in istops:
            print(f"[single] it {itr:4d}  data={derrs_single[-1]:.6e}  img={ierrs_single[-1]:.6e}  TV={tvs_single[-1]:.2f}")
    single_time = time.time()-t0
    xbarim_single = xbarim.copy()
    print(f"Single-channel done in {single_time:.2f}s")

    # ----------------------------- TWO-CHANNEL --------------------------------
    print("\n" + "="*60); print("RUNNING TWO-CHANNEL"); print("="*60)
    sinodata_lo = R_lo(sinodata); sinodata_hi = R_hi(sinodata)
    sinodata_lo_sc = nusino*sinodata_lo; sinodata_hi_sc = nusino*sinodata_hi

    # total norm for stacked operator
    xim = randn(nx, ny)*mask; xim1 = xim*0.; xim2 = xim*0.; worksino = truesino*0.
    for _ in range(200):
        circularFanbeamProjection(xim, worksino)
        s_hi = R_hi(worksino)*nusino; s_lo = R_lo(worksino)*nusino
        xg = gradx(xim)*nuxgrad; yg = grady(xim)*nuygrad; yimloc = l1f*xim
        mag1 = sqrt((yimloc**2).sum() + (yg**2).sum() + (xg**2).sum() + (s_hi**2).sum() + (s_lo**2).sum())
        if mag1>0: yimloc/=mag1; yg/=mag1; xg/=mag1; s_hi/=mag1; s_lo/=mag1
        xim1 *= 0.; imtmp = xim1*0.; circularFanbeamBackProjection(s_hi, imtmp); xim1 += imtmp
        imtmp *= 0.; circularFanbeamBackProjection(s_lo, imtmp); xim1 += imtmp
        xim1 *= (nusino*mask)
        xim2 = mdivx(xg)*(nuxgrad*mask); xim3 = mdivy(yg)*(nuygrad*mask)
        xim = xim1 + xim2 + xim3 + l1f*yimloc
        mag2 = sqrt((xim**2.).sum()); 
        if mag2>0: xim /= mag2
    totalnorm_two = (mag1 + mag2)*0.5
    sig_two = stepbalance/totalnorm_two; tau_two = 1./(totalnorm_two*stepbalance)
    sig_hi = sig_two; sig_lo = sigma_lo_scale*sig_two
    epssc_hi = eps_hi*sqrt(nrays); epssc_lo = eps_lo*sqrt(nrays)
    print(f"Total norm(two)={totalnorm_two:.4f}  sig_hi={sig_hi:.6f}  sig_lo={sig_lo:.6f}  tau={tau_two:.6f}")

    # init
    xim = zeros([nx,ny]); yim = xim*0.; xbarim = xim*0.; wimp = xim*0.
    ysino_hi = zeros([nviews, nbins]); ysino_lo = zeros([nviews, nbins])
    ygradx = zeros([nx,ny]); ygrady = zeros([nx,ny])
    derrs_two=[]; ierrs_two=[]; tvs_two=[]

    t0 = time.time()
    for itr in range(1, itermax+1):
        yhi_old=ysino_hi.copy(); ylo_old=ysino_lo.copy(); ygradxold=ygradx.copy(); ygradyold=ygrady.copy(); yimold=yim.copy()
        # primal
        wimp *= 0.; imtmp = zeros_like(xim)
        circularFanbeamBackProjection(R_hi(ysino_hi), imtmp); wimp += imtmp
        imtmp *= 0.; circularFanbeamBackProjection(R_lo(ysino_lo), imtmp); wimp += imtmp
        wimp *= nusino; wimp *= mask
        wimqx = mdivx(ygradx)*nuxgrad*mask; wimqy = mdivy(ygrady)*nuygrad*mask; wiml1 = l1f*yim
        ximold = xim.copy()
        xim = xim - tau_two*(wimp + wimqx + wimqy + wiml1)
        xim[xim<0] = 0.; xbarim = xim + (xim - ximold)

        # duals
        worksino = zeros([nviews, nbins]); circularFanbeamProjection(xbarim, worksino)
        Ax_hi = R_hi(worksino)*nusino; Ax_lo = R_lo(worksino)*nusino
        resid_hi = Ax_hi - sinodata_hi_sc; resid_lo = Ax_lo - sinodata_lo_sc
        derrs_two.append( sqrt(((resid_hi/nusino)**2).sum() + ((resid_lo/nusino)**2).sum())/sqrt(1.*nviews*nbins) )
        ysino_hi = ysino_hi + sig_hi*resid_hi
        ymag_hi = sqrt((ysino_hi**2).sum())
        ysino_hi *= (maximum(0.0, ymag_hi - sig_hi*nusino*epssc_hi)/(ymag_hi+1e-12))
        ysino_lo = ysino_lo + sig_lo*resid_lo
        ymag_lo = sqrt((ysino_lo**2).sum())
        ysino_lo *= (maximum(0.0, ymag_lo - sig_lo*nusino*epssc_lo)/(ymag_lo+1e-12))

        # DTV blocks (same as single)
        tgx = gradx(xbarim)*nuxgrad; ptilx = ygradx + sig_two*tgx
        ygradx = (2.-alpha)*ptilx/maximum(abs(ptilx), (2.-alpha))
        tgy = grady(xbarim)*nuygrad; ptily = ygrady + sig_two*tgy
        ygrady = alpha*ptily/maximum(abs(ptily), alpha)
        ptil1 = yim + sig_two*(l1f*xbarim); yim = beta*ptil1/maximum(sqrt(ptil1**2), 1e-10)
        tvs_two.append(sqrt((gradim(xbarim)[0]**2 + gradim(xbarim)[1]**2)).sum())

        # predictor–corrector
        ygradx = ygradxold - rho*(ygradxold - ygradx)
        ygrady = ygradyold - rho*(ygradyold - ygrady)
        ysino_hi = yhi_old - rho*(yhi_old - ysino_hi)
        ysino_lo = ylo_old - rho*(ylo_old - ysino_lo)
        yim = yimold - rho*(yimold - yim)
        xim = ximold - rho*(ximold - xim)

        ierrs_two.append(sqrt(((xbarim - phimage)**2).sum()/(nx*ny)))
        if itr in istops:
            print(f"[two]    it {itr:4d}  data={derrs_two[-1]:.6e}  img={ierrs_two[-1]:.6e}  TV={tvs_two[-1]:.2f}")
    two_time = time.time()-t0
    xbarim_two = xbarim.copy()
    print(f"Two-channel done in {two_time:.2f}s")

    # cache
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump({
            'derrs_single': derrs_single, 'ierrs_single': ierrs_single, 'tvs_single': tvs_single,
            'xbarim_single': xbarim_single, 'single_time': single_time,
            'derrs_two': derrs_two, 'ierrs_two': ierrs_two, 'tvs_two': tvs_two,
            'xbarim_two': xbarim_two, 'two_time': two_time,
            'phimage': phimage, 'truetv': truetv
        }, f)
    print(f"Saved cache -> {RESULTS_FILE}")

# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("GENERATING COMPARISON PLOTS")
print("="*60)

fig = plt.figure(figsize=(16, 10))

# Data RMSE
ax1 = plt.subplot(2,3,1)
plt.plot(derrs_single, label='Single-channel', linewidth=2)
plt.plot(derrs_two, label='Two-channel', linewidth=2)
plt.xlabel('Iteration'); plt.ylabel('Data RMSE'); plt.yscale('log'); plt.grid(True, alpha=0.3)
plt.title('Data Fidelity Convergence'); plt.legend()

# Image RMSE
ax2 = plt.subplot(2,3,2)
plt.plot(ierrs_single, label='Single-channel', linewidth=2)
plt.plot(ierrs_two, label='Two-channel', linewidth=2)
plt.xlabel('Iteration'); plt.ylabel('Image RMSE'); plt.yscale('log'); plt.grid(True, alpha=0.3)
plt.title('Image Error Convergence'); plt.legend()

# TV
ax3 = plt.subplot(2,3,3)
plt.plot(tvs_single, label='Single-channel', linewidth=2)
plt.plot(tvs_two, label='Two-channel', linewidth=2)
plt.axhline(y=truetv, linestyle='--', label='Ground truth', linewidth=2)
plt.xlabel('Iteration'); plt.ylabel('Total TV'); plt.grid(True, alpha=0.3)
plt.title('Total Variation'); plt.legend()

# Recon images
ax4 = plt.subplot(2,3,4)
im1 = plt.imshow(xbarim_single.T, cmap='gray', origin='lower'); plt.colorbar(im1, ax=ax4)
plt.title('Single-Channel Reconstruction'); plt.axis('off')

ax5 = plt.subplot(2,3,5)
im2 = plt.imshow(xbarim_two.T, cmap='gray', origin='lower'); plt.colorbar(im2, ax=ax5)
plt.title('Two-Channel Reconstruction'); plt.axis('off')

ax6 = plt.subplot(2,3,6)
im3 = plt.imshow(phimage.T, cmap='gray', origin='lower'); plt.colorbar(im3, ax=ax6)
plt.title('Ground Truth Phantom'); plt.axis('off')

plt.tight_layout()
plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
print("Saved comparison plot to 'method_comparison.png'")
plt.show()

# -----------------------------------------------------------------------------
# SUMMARY BLOCK (as requested)
# -----------------------------------------------------------------------------
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
print(f"  Data RMSE: {(derrs_single[-1]-derrs_two[-1])/derrs_single[-1]*100:.2f}%")
print(f"  Image RMSE: {(ierrs_single[-1]-ierrs_two[-1])/ierrs_single[-1]*100:.2f}%")
print(f"  TV error: {abs(tvs_single[-1]-truetv):.2f} vs {abs(tvs_two[-1]-truetv):.2f}")
