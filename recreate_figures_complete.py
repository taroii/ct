"""
Complete implementation to recreate Figure 6 and Figure 7 from the paper:
"Accurate volume image reconstruction for digital breast tomosynthesis
with directional-gradient and pixel sparsity regularization"

THREE-WAY COMPARISON:
1. FBP (baseline)
2. Single-channel L1-DTV (paper method)
3. Two-channel L1-DTV (frequency-split method from compare_methods_gpt.py)
"""

import numpy as np
from numpy import *
from numpy.random import randn, poisson
import matplotlib.pyplot as plt
from numba import njit
import time
import pickle
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

mfact = 2  # Image size = 512/mfact
addnoise = 0
nph = 1.e6
nuxfact = 0.5
nuyfact = 0.5
l1f = 1.0
eps = 0.001  # data discrepancy RMSE
larc = 1.0
alpha = 1.75  # DTV parameter (paper uses 1.7-1.9 depending on size)
beta = 5.0
rho = 1.75
stepbalance = 100.0
cutoffparm = 4.0

# Two-channel parameters
cutoffparm_lo = 8.0
eps_hi = eps
eps_lo = 1.25*eps
sigma_lo_scale = 4.0

# Iterations
itermax = 500
istops = [1,2,5,10,20,50,100,200,300,400,500]

RESULTS_FILE = 'fig6_fig7_results.pkl'
FORCE_RECOMPUTE = True

print("="*70)
print("FIGURE 6 & 7 RECREATION - L1-DTV COMPARISON")
print("="*70)
print("Comparing two L1-DTV methods:")
print("  1. Single-channel L1-DTV (paper method)")
print("  2. Two-channel L1-DTV (frequency-split method)")
print("="*70)

# ============================================================================
# ANALYTIC PHANTOM (3D with geometric shapes)
# ============================================================================

def create_3d_analytic_phantom(nx, ny, nz):
    """Create 3D analytic phantom with spheres at different depths"""
    phantom = zeros((nx, ny, nz))

    dx = dy = 10.0 / nx
    dz = 5.0 / nz  # 5 cm in z (depth)

    x = arange(nx) * dx - 5.0 + dx/2.0
    y = arange(ny) * dy - 5.0 + dy/2.0
    z = arange(nz) * dz + dz/2.0  # z starts at 0
    X, Y, Z = meshgrid(x, y, z, indexing='ij')

    # Circular background in x-y, extends through all z
    mask_xy = (X**2 + Y**2) <= 4.8**2
    phantom[mask_xy] = 0.2

    # Glandular tissue slab (mid-depth)
    slab1 = mask_xy & (Z > 1.5) & (Z < 3.0) & (X > -2.5) & (X < 1.5)
    phantom[slab1] = 0.35

    # Spheres at different depths (key for demonstrating depth resolution!)
    spheres = [
        # Overlapping pair in DEPTH (x, y, z, radius, value)
        (0.0, 2.0, 2.0, 0.5, 0.45),   # First sphere
        (0.0, 2.0, 3.2, 0.5, 0.48),   # Overlapping in depth!

        # Additional spheres at various depths
        (-2.0, -2.0, 1.5, 0.4, 0.42),
        (2.0, -2.0, 3.5, 0.4, 0.43),
        (-1.5, 1.5, 2.5, 0.35, 0.40),
        (1.5, 0.0, 2.8, 0.45, 0.46),
        (0.0, -1.5, 1.8, 0.35, 0.41),
    ]

    for cx, cy, cz, r, val in spheres:
        sphere_mask = ((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) <= r**2
        phantom[sphere_mask] = val

    return phantom

def create_2d_analytic_phantom(nx, ny):
    """Create single 2D slice for faster testing"""
    # For 2D, just take a middle slice of the 3D phantom
    nz_temp = 40
    phantom_3d = create_3d_analytic_phantom(nx, ny, nz_temp)
    return phantom_3d[:, :, nz_temp//2]

# ============================================================================
# SETUP - 3D PHANTOM
# ============================================================================

nx = int(512/mfact)
ny = int(512/mfact)
nz = 10  # Number of z-slices to reconstruct

phantom_3d = create_3d_analytic_phantom(nx, ny, nz)

print(f"\nPhantom: {nx}x{ny}x{nz} voxels")
print(f"Pixel size: {10.0/nx*10:.2f} mm")
print(f"Slice thickness: {5.0/nz*10:.2f} mm")

ximageside = 10.0
yimageside = 10.0
dx = ximageside/nx
dy = yimageside/ny

xar = arange(-ximageside/2. + dx/2, ximageside/2., dx)[:, newaxis]*ones([ny])
yar = ones([nx, ny])*arange(-yimageside/2. + dy/2, yimageside/2., dy)
rar = sqrt(xar**2 + yar**2)
mask = zeros((nx, ny))
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

# ============================================================================
# PROJECTION/BACKPROJECTION
# ============================================================================

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
            xad = abs(xdiff)*dy; yad = abs(ydiff)*dx
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
                        if 0 <= iy < ny: raysum += frac2*trav*image[ix, iy]
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
                        if 0 <= ix < nx: raysum += frac2*trav*image[ix, iy]
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
            xad = abs(xdiff)*dy; yad = abs(ydiff)*dx
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
                        if 0 <= iy < ny: image[ix, iy] += frac2*val*trav
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
                        if 0 <= ix < nx: image[ix, iy] += frac2*val*trav
                    ixOld = ix; xIntOld = xIntercept

# ============================================================================
# GRAD / DIV OPERATORS
# ============================================================================

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

# ============================================================================
# FILTERS
# ============================================================================

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
    imft = fft.fft(sino, axis=1)
    pimft = (ones([nbins])*fft.fftshift(W))*imft
    return fft.ifft(pimft, axis=1).real

def R_lo(s): return R_fft_weight(s, F_lo)
def R_hi(s): return R_fft_weight(s, F_hi)
def fo(s): return R_fft_weight(s, F_single)

# ============================================================================
# DATA GENERATION - 3D (sinograms for all slices)
# ============================================================================

print("\nGenerating sinogram data for all slices...")
truesino_3d = zeros([nz, nviews, nbins])
for iz in range(nz):
    phimage_slice = phantom_3d[:, :, iz]
    circularFanbeamProjection(phimage_slice, truesino_3d[iz])
    if iz % 3 == 0:
        print(f"  Generated sinogram for slice {iz+1}/{nz}")

sinodata_3d = truesino_3d * 1.
print(f"Sinogram data shape: {sinodata_3d.shape}")

# Ground truth TV (for middle slice as reference)
phimage_mid = phantom_3d[:, :, nz//2]
xg = gradx(phimage_mid); truetvx = sqrt(xg**2).sum()
yg = grady(phimage_mid); truetvy = sqrt(yg**2).sum()
xg, yg = gradim(phimage_mid); truetv = sqrt(xg**2 + yg**2).sum()
print(f"Ground truth TV (middle slice): {truetv:.2f}")

# ============================================================================
# OPERATOR NORMS
# ============================================================================

print("Estimating operator norms...")
xim = randn(nx, ny)*mask; worksino = zeros([nviews, nbins]); npower = 50

for _ in range(npower):
    circularFanbeamProjection(xim, worksino)
    worksino_f = fo(fo(worksino))
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

print(f"nusino={nusino:.6f}, nuxgrad={nuxgrad:.6f}, nuygrad={nuygrad:.6f}")

# ============================================================================
# METHOD 1: FBP (COMMENTED OUT - FOCUSING ON L1-DTV COMPARISON)
# ============================================================================

# print("\n" + "="*70)
# print("METHOD 1: FBP (Filtered Back-Projection)")
# print("="*70)
#
# sinodata_fbp = fo(sinodata)
# xim_fbp = zeros([nx, ny])
# circularFanbeamBackProjection(sinodata_fbp, xim_fbp)
# xim_fbp *= mask
#
# fbp_err = sqrt(((xim_fbp - phimage)**2).sum()/(nx*ny))
# print(f"FBP image RMSE: {fbp_err:.6f}")

# ============================================================================
# METHOD 2: SINGLE-CHANNEL L1-DTV - 3D (SLICE-BY-SLICE)
# ============================================================================

print("\n" + "="*70)
print("METHOD 2: SINGLE-CHANNEL L1-DTV (3D reconstruction)")
print("="*70)

# Compute total norm (using first slice)
sinodata_first = sinodata_3d[0]
sinodata_single_temp = fo(sinodata_first)
worksino = zeros([nviews, nbins])

xim = randn(nx, ny)*mask; xim1 = xim*0.; xim2 = xim*0.
for _ in range(200):
    circularFanbeamProjection(xim, worksino)
    w = fo(worksino); w *= nusino
    xg = gradx(xim)*nuxgrad; yg = grady(xim)*nuygrad; yimloc = l1f*xim
    mag1 = sqrt((yimloc**2).sum() + (yg**2).sum() + (xg**2).sum() + (w**2).sum())
    if mag1>0: yimloc/=mag1; yg/=mag1; xg/=mag1; w/=mag1
    xim1 *= 0.; circularFanbeamBackProjection(fo(w), xim1); xim1 *= (nusino*mask)
    xim2 = mdivx(xg)*(nuxgrad*mask); xim3 = mdivy(yg)*(nuygrad*mask)
    xim = xim1 + xim2 + xim3 + l1f*yimloc
    mag2 = sqrt((xim**2.).sum())
    if mag2>0: xim /= mag2

totalnorm_single = (mag1 + mag2)*0.5
sig_single = stepbalance/totalnorm_single
tau_single = 1./(totalnorm_single*stepbalance)
print(f"Total norm={totalnorm_single:.4f}, sig={sig_single:.6f}, tau={tau_single:.6f}")

# Storage for 3D reconstruction
recon_single_3d = zeros([nx, ny, nz])
ierrs_single_all = []  # Store final errors for all slices
ierrs_single_mid = []  # Store iteration errors for middle slice (for convergence plot)

t0_total = time.time()

# LOOP OVER Z-SLICES
for iz in range(nz):
    print(f"\n--- Reconstructing slice {iz+1}/{nz} ---")

    sinodata = sinodata_3d[iz]
    phimage = phantom_3d[:, :, iz]

    sinodata_single = fo(sinodata)
    sinodatasc_single = nusino*sinodata_single

    # Initialize
    xim = zeros([nx,ny]); yim = xim*0.; xbarim = xim*0.; wimp = xim*0.
    ysino_single = zeros([nviews, nbins]); ygradx = zeros([nx,ny]); ygrady = zeros([nx,ny])
    ierrs_single = []

    t0 = time.time()
    for itr in range(1, itermax+1):
        ysinoold = ysino_single.copy(); ygradxold=ygradx.copy(); ygradyold=ygrady.copy(); yimold=yim.copy()

        # Primal
        wimp *= 0.; circularFanbeamBackProjection(fo(ysino_single), wimp); wimp *= nusino; wimp *= mask
        wimqx = mdivx(ygradx)*nuxgrad*mask; wimqy = mdivy(ygrady)*nuygrad*mask; wiml1 = l1f*yim
        ximold = xim.copy()
        xim = xim - tau_single*(wimp + wimqx + wimqy + wiml1)
        xim[xim<0] = 0.; xbarim = xim + (xim - ximold)

        # Dual
        worksino = zeros([nviews, nbins]); circularFanbeamProjection(xbarim, worksino)
        w = fo(worksino); w *= nusino
        resid = w - sinodatasc_single
        ysino_single = ysino_single + sig_single*resid
        ymag = sqrt((ysino_single**2).sum())
        ysino_single *= (maximum(0.0, ymag - sig_single*nusino*epssc)/(ymag+1e-12))

        tgx = gradx(xbarim)*nuxgrad; ptilx = ygradx + sig_single*tgx
        ygradx = (2.-alpha)*ptilx/maximum(abs(ptilx), (2.-alpha))
        tgy = grady(xbarim)*nuygrad; ptily = ygrady + sig_single*tgy
        ygrady = alpha*ptily/maximum(abs(ptily), alpha)
        ptil1 = yim + sig_single*(l1f*xbarim)
        yim = beta*ptil1/maximum(sqrt(ptil1**2), beta)

        # Predictor-corrector
        ygradx = ygradxold - rho*(ygradxold - ygradx)
        ygrady = ygradyold - rho*(ygradyold - ygrady)
        ysino_single = ysinoold - rho*(ysinoold - ysino_single)
        yim = yimold - rho*(yimold - yim)
        xim = ximold - rho*(ximold - xim)

        ierrs_single.append(sqrt(((xbarim - phimage)**2).sum()/(nx*ny)))
        if itr in istops:
            print(f"[single] it {itr:4d}  img_err={ierrs_single[-1]:.6e}")

    slice_time = time.time()-t0
    recon_single_3d[:, :, iz] = xbarim.copy()
    ierrs_single_all.append(ierrs_single[-1])

    # Save convergence history for middle slice
    if iz == nz // 2:
        ierrs_single_mid = ierrs_single.copy()

    print(f"Slice {iz+1} done in {slice_time:.2f}s, final RMSE={ierrs_single[-1]:.6f}")

single_time = time.time()-t0_total
avg_rmse_single = mean(ierrs_single_all)
print(f"\nSingle-channel 3D done in {single_time:.2f}s")
print(f"Average RMSE across all slices: {avg_rmse_single:.6f}")

# ============================================================================
# METHOD 3: TWO-CHANNEL L1-DTV - 3D (SLICE-BY-SLICE)
# ============================================================================

print("\n" + "="*70)
print("METHOD 3: TWO-CHANNEL L1-DTV (3D reconstruction)")
print("="*70)

# Compute total norm (using first slice)
sinodata_first = sinodata_3d[0]
worksino = zeros([nviews, nbins])

xim = randn(nx, ny)*mask; xim1 = xim*0.; xim2 = xim*0.
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
    mag2 = sqrt((xim**2.).sum())
    if mag2>0: xim /= mag2

totalnorm_two = (mag1 + mag2)*0.5
sig_two = stepbalance/totalnorm_two; tau_two = 1./(totalnorm_two*stepbalance)
sig_hi = sig_two; sig_lo = sigma_lo_scale*sig_two
epssc_hi = eps_hi*sqrt(nrays); epssc_lo = eps_lo*sqrt(nrays)
print(f"Total norm={totalnorm_two:.4f}, sig_hi={sig_hi:.6f}, sig_lo={sig_lo:.6f}, tau={tau_two:.6f}")

# Storage for 3D reconstruction
recon_two_3d = zeros([nx, ny, nz])
ierrs_two_all = []  # Store final errors for all slices
ierrs_two_mid = []  # Store iteration errors for middle slice (for convergence plot)

t0_total = time.time()

# LOOP OVER Z-SLICES
for iz in range(nz):
    print(f"\n--- Reconstructing slice {iz+1}/{nz} ---")

    sinodata = sinodata_3d[iz]
    phimage = phantom_3d[:, :, iz]

    sinodata_lo = R_lo(sinodata); sinodata_hi = R_hi(sinodata)
    sinodata_lo_sc = nusino*sinodata_lo; sinodata_hi_sc = nusino*sinodata_hi

    # Initialize
    xim = zeros([nx,ny]); yim = xim*0.; xbarim = xim*0.; wimp = xim*0.
    ysino_hi = zeros([nviews, nbins]); ysino_lo = zeros([nviews, nbins])
    ygradx = zeros([nx,ny]); ygrady = zeros([nx,ny])
    ierrs_two = []

    t0 = time.time()
    for itr in range(1, itermax+1):
        yhi_old=ysino_hi.copy(); ylo_old=ysino_lo.copy(); ygradxold=ygradx.copy(); ygradyold=ygrady.copy(); yimold=yim.copy()

        # Primal
        wimp *= 0.; imtmp = zeros_like(xim)
        circularFanbeamBackProjection(R_hi(ysino_hi), imtmp); wimp += imtmp
        imtmp *= 0.; circularFanbeamBackProjection(R_lo(ysino_lo), imtmp); wimp += imtmp
        wimp *= nusino; wimp *= mask
        wimqx = mdivx(ygradx)*nuxgrad*mask; wimqy = mdivy(ygrady)*nuygrad*mask; wiml1 = l1f*yim
        ximold = xim.copy()
        xim = xim - tau_two*(wimp + wimqx + wimqy + wiml1)
        xim[xim<0] = 0.; xbarim = xim + (xim - ximold)

        # Dual
        worksino = zeros([nviews, nbins]); circularFanbeamProjection(xbarim, worksino)
        Ax_hi = R_hi(worksino)*nusino; Ax_lo = R_lo(worksino)*nusino
        resid_hi = Ax_hi - sinodata_hi_sc; resid_lo = Ax_lo - sinodata_lo_sc

        ysino_hi = ysino_hi + sig_hi*resid_hi
        ymag_hi = sqrt((ysino_hi**2).sum())
        ysino_hi *= (maximum(0.0, ymag_hi - sig_hi*nusino*epssc_hi)/(ymag_hi+1e-12))

        ysino_lo = ysino_lo + sig_lo*resid_lo
        ymag_lo = sqrt((ysino_lo**2).sum())
        ysino_lo *= (maximum(0.0, ymag_lo - sig_lo*nusino*epssc_lo)/(ymag_lo+1e-12))

        tgx = gradx(xbarim)*nuxgrad; ptilx = ygradx + sig_two*tgx
        ygradx = (2.-alpha)*ptilx/maximum(abs(ptilx), (2.-alpha))
        tgy = grady(xbarim)*nuygrad; ptily = ygrady + sig_two*tgy
        ygrady = alpha*ptily/maximum(abs(ptily), alpha)
        ptil1 = yim + sig_two*(l1f*xbarim)
        yim = beta*ptil1/maximum(sqrt(ptil1**2), beta)

        # Predictor-corrector
        ygradx = ygradxold - rho*(ygradxold - ygradx)
        ygrady = ygradyold - rho*(ygradyold - ygrady)
        ysino_hi = yhi_old - rho*(yhi_old - ysino_hi)
        ysino_lo = ylo_old - rho*(ylo_old - ysino_lo)
        yim = yimold - rho*(yimold - yim)
        xim = ximold - rho*(ximold - xim)

        ierrs_two.append(sqrt(((xbarim - phimage)**2).sum()/(nx*ny)))
        if itr in istops:
            print(f"[two]    it {itr:4d}  img_err={ierrs_two[-1]:.6e}")

    slice_time = time.time()-t0
    recon_two_3d[:, :, iz] = xbarim.copy()
    ierrs_two_all.append(ierrs_two[-1])

    # Save convergence history for middle slice
    if iz == nz // 2:
        ierrs_two_mid = ierrs_two.copy()

    print(f"Slice {iz+1} done in {slice_time:.2f}s, final RMSE={ierrs_two[-1]:.6f}")

two_time = time.time()-t0_total
avg_rmse_two = mean(ierrs_two_all)
print(f"\nTwo-channel 3D done in {two_time:.2f}s")
print(f"Average RMSE across all slices: {avg_rmse_two:.6f}")

# ============================================================================
# FIGURE 6: COMPARISON IMAGES (Paper format: x-y top, x-z bottom)
# ============================================================================

print("\n" + "="*70)
print("CREATING FIGURE 6 - PAPER FORMAT (TRUE 3D)")
print("="*70)

fig = plt.figure(figsize=(15, 10))

# Define grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)

vmin, vmax = 0, 0.5

# ===== TOP ROW: x-y plane (in-plane) - Use middle slice =====
z_mid = nz // 2
y_mid = ny // 2  # For x-z plane extraction

ax00 = fig.add_subplot(gs[0, 0])
ax00.imshow(phantom_3d[:, :, z_mid].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax00.set_title('Phantom\n(x-y plane)', fontsize=13, fontweight='bold')
ax00.set_xlabel('x'); ax00.set_ylabel('y')
ax00.set_xticks([]); ax00.set_yticks([])

ax01 = fig.add_subplot(gs[0, 1])
ax01.imshow(recon_single_3d[:, :, z_mid].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax01.set_title('Single-channel L1-DTV\n(x-y plane)', fontsize=13, fontweight='bold')
ax01.set_xlabel('x'); ax01.set_ylabel('y')
ax01.set_xticks([]); ax01.set_yticks([])

ax02 = fig.add_subplot(gs[0, 2])
ax02.imshow(recon_two_3d[:, :, z_mid].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax02.set_title('Two-channel L1-DTV\n(x-y plane)', fontsize=13, fontweight='bold')
ax02.set_xlabel('x'); ax02.set_ylabel('y')
ax02.set_xticks([]); ax02.set_yticks([])

# ===== BOTTOM ROW: x-z plane (DEPTH PLANE - Shows overlapping spheres!) =====
# This is the key figure showing depth resolution!
# Extract x-z slice by taking all x, all z, at fixed y

phantom_xz = phantom_3d[:, y_mid, :]  # Shape: [nx, nz]
single_xz = recon_single_3d[:, y_mid, :]  # Shape: [nx, nz]
two_xz = recon_two_3d[:, y_mid, :]  # Shape: [nx, nz]

ax10 = fig.add_subplot(gs[1, 0])
ax10.imshow(phantom_xz.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
ax10.set_title('Phantom\n(x-z plane)', fontsize=13, fontweight='bold')
ax10.set_xlabel('x'); ax10.set_ylabel('z (depth)')
ax10.set_xticks([]); ax10.set_yticks([])

ax11 = fig.add_subplot(gs[1, 1])
ax11.imshow(single_xz.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
ax11.set_title(f'Single-channel L1-DTV\n(RMSE={avg_rmse_single:.6f})', fontsize=13, fontweight='bold')
ax11.set_xlabel('x'); ax11.set_ylabel('z (depth)')
ax11.set_xticks([]); ax11.set_yticks([])

ax12 = fig.add_subplot(gs[1, 2])
ax12.imshow(two_xz.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
ax12.set_title(f'Two-channel L1-DTV\n(RMSE={avg_rmse_two:.6f})', fontsize=13, fontweight='bold')
ax12.set_xlabel('x'); ax12.set_ylabel('z (depth)')
ax12.set_xticks([]); ax12.set_yticks([])

plt.savefig('figure_6_comparison.png', dpi=200, bbox_inches='tight')
print("Saved: figure_6_comparison.png")
print("Top row: x-y plane (in-plane imaging)")
print("Bottom row: x-z plane (depth resolution - shows overlapping spheres!)")

# ============================================================================
# FIGURE 7: DEPTH PROFILES (cross-sections through overlapping spheres)
# ============================================================================

print("\n" + "="*70)
print("CREATING FIGURE 7 - DEPTH PROFILE COMPARISON")
print("="*70)

# Depth profile (z-axis) at x=center, y=center where spheres overlap
# This demonstrates depth resolution capability
x_profile_idx = nx // 2  # center (x=0)
y_profile_idx = int(ny * 0.65)  # y=2.0 cm (where overlapping spheres are located)

z_range = arange(nz)
dz = 5.0 / nz
z_coords_plot = z_range * dz + dz/2.0  # z coordinates in cm

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Extract depth profiles
phantom_depth = phantom_3d[x_profile_idx, y_profile_idx, :]
single_depth = recon_single_3d[x_profile_idx, y_profile_idx, :]
two_depth = recon_two_3d[x_profile_idx, y_profile_idx, :]

ax.plot(z_coords_plot, phantom_depth, 'k-', linewidth=3, label='Phantom (Ground Truth)', marker='o', markersize=6)
ax.plot(z_coords_plot, single_depth, 'r-', linewidth=2.5, label='Single-channel L1-DTV', alpha=0.8, marker='s', markersize=5)
ax.plot(z_coords_plot, two_depth, 'b-', linewidth=2.5, label='Two-channel L1-DTV', alpha=0.8, marker='^', markersize=5)

ax.set_xlabel('z position (depth, cm)', fontsize=14)
ax.set_ylabel('Attenuation (cm⁻¹)', fontsize=14)
ax.set_title('Depth Profile through Overlapping Spheres (x=0, y=2.0)', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_7_profile.png', dpi=200, bbox_inches='tight')
print("Saved: figure_7_profile.png")
print("This shows depth resolution: can the methods distinguish spheres at different depths?")

# ============================================================================
# CONVERGENCE PLOT (for middle slice)
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.plot(ierrs_single_mid, 'r-', linewidth=2.5, label='Single-channel L1-DTV')
ax.plot(ierrs_two_mid, 'b-', linewidth=2.5, label='Two-channel L1-DTV')
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Image RMSE', fontsize=14)
ax.set_yscale('log')
ax.set_title(f'Convergence Comparison: Single vs Two-Channel L1-DTV (Slice {nz//2+1})', fontsize=15, fontweight='bold')
ax.legend(fontsize=13, loc='best')
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('convergence_comparison.png', dpi=200, bbox_inches='tight')
print("Saved: convergence_comparison.png")
print("Convergence plot shows middle slice (used in Figure 6 top row)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("FINAL COMPARISON SUMMARY - 3D RECONSTRUCTION")
print("="*70)

print(f"\nSingle-channel L1-DTV (Paper method):")
print(f"  Average RMSE across {nz} slices: {avg_rmse_single:.6f}")
print(f"  Runtime: {single_time:.2f}s ({single_time/nz:.2f}s per slice)")

print(f"\nTwo-channel L1-DTV (Frequency-split method):")
print(f"  Average RMSE across {nz} slices: {avg_rmse_two:.6f}")
print(f"  Runtime: {two_time:.2f}s ({two_time/nz:.2f}s per slice)")

# Compute relative improvement
if avg_rmse_two < avg_rmse_single:
    improvement = (avg_rmse_single - avg_rmse_two) / avg_rmse_single * 100
    print(f"\n✓ Two-channel is BETTER by {improvement:.2f}%")
elif avg_rmse_single < avg_rmse_two:
    improvement = (avg_rmse_two - avg_rmse_single) / avg_rmse_two * 100
    print(f"\n✓ Single-channel is BETTER by {improvement:.2f}%")
else:
    print(f"\n≈ Both methods have equivalent performance")

print(f"\nAbsolute difference: {abs(avg_rmse_single - avg_rmse_two):.8f}")

print("\n" + "="*70)
print("3D RECONSTRUCTION COMPLETE!")
print("="*70)
print(f"Reconstructed {nz} slices with {itermax} iterations each")
print(f"Total iterations: {nz * itermax}")
print("\nFigures saved:")
print("  - figure_6_comparison.png (x-y and x-z planes)")
print("  - figure_7_profile.png (depth profile)")
print("  - convergence_comparison.png (iteration convergence)")
print("="*70)
plt.show()
