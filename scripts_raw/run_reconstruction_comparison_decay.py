"""
Modular reconstruction comparison script for RAW VICTRE phantom.
Supports plug-and-play algorithms for easy comparison.

This version uses the RAW VICTRE phantom data (not synthetic ICA distribution).

KEY FEATURE: UNIFIED DECAY for two-channel method
Both sigma_lo and eps_lo decay from elevated initial values toward the
high-frequency channel values using the same gamma:
    sig_lo(t) = sig_hi + (sig_lo_initial - sig_hi) * gamma^t
    eps_lo(t) = eps_hi + (eps_lo_initial - eps_hi) * gamma^t

This allows larger steps early (faster convergence) while taking smaller
steps near the solution (better accuracy).

This script:
1. Loads the VICTRE phantom ROI with raw tissue values
2. Normalizes to appropriate attenuation values
3. Runs multiple reconstruction algorithms
4. Generates comparison plots and saves results
"""

import numpy as np
from numpy import *
from numpy.random import randn, poisson
import matplotlib.pyplot as plt
from numba import njit
import time
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import sys

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Algorithm selection - add/remove algorithms here
ALGORITHMS = {
    'single_channel': 'Single-channel L1-DTV',
    'two_channel': 'Two-channel L1-DTV'
}

# Main configuration dictionary
# Default values match original compare_methods.py parameters
CONFIG = {
    # Geometry
    'mfact': 2,  # Image size = 512/mfact

    # Algorithm parameters (from original compare_methods.py)
    'alpha': 1.75,  # DTV parameter (paper uses 1.7-1.9 depending on size)
    'beta': 5.0,    # L1 sparsity penalty (original value from compare_methods.py)
    'rho': 1.75,
    'eps': 0.001,   # data discrepancy RMSE
    'nuxfact': 0.5,
    'nuyfact': 0.5,
    'l1f': 1.0,
    'larc': 1.0,
    'stepbalance': 100.0,
    'cutoffparm': 4.0,

    # Two-channel parameters (from original compare_methods.py)
    'cutoffparm_lo': 8.0,      # Low-frequency cutoff
    'sigma_lo_scale': 4.0,     # Initial sig_lo = sigma_lo_scale * sig_hi (decays to 1x)
    'eps_lo_ratio': 1.25,      # Initial eps_lo = eps_lo_ratio * eps_hi (decays to 1x)

    # UNIFIED DECAY: both sig_lo and eps_lo decay with same gamma
    'sigma_decay_gamma': 0.998,  # Decay factor: param(t) = final + (initial - final) * gamma^t

    # Simulation
    'addnoise': 0,
    'nph': 1.e6,

    # Iterations
    'itermax': 500,
    'istops': [1,2,5,10,20,50,100,200,300,400,500],
    'verbose': False
}

# Results
RESULTS_FILE = 'reconstruction_results.pkl'
FORCE_RECOMPUTE = True

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Raw VICTRE phantom reconstruction comparison (with sigma decay)')
    parser.add_argument('--quick', action='store_true', help='Quick test with 200 iterations and 1 slice')
    parser.add_argument('--algorithms', nargs='+', default=['single_channel', 'two_channel'],
                       help='Algorithms to run')
    args = parser.parse_args()

    if args.quick:
        CONFIG['itermax'] = 200
        print("Quick mode: Using 200 iterations and 1 slice only")
else:
    # Default when imported or run without args
    class MockArgs:
        quick = False
        algorithms = ['single_channel', 'two_channel']
    args = MockArgs()

print("="*70)
print("RAW VICTRE PHANTOM RECONSTRUCTION")
print("="*70)
print("Reconstructing RAW VICTRE phantom (not ICA distribution) with:")
if 'single_channel' in args.algorithms:
    print("  1. Single-channel L1-DTV (paper method)")
if 'two_channel' in args.algorithms:
    print("  2. Two-channel L1-DTV (frequency-split method)")
print("="*70)

# ============================================================================
# SETUP OUTPUT DIRECTORY
# ============================================================================

# Create output directory for raw VICTRE results
output_dir = Path('../results/raw_victre')
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\nOutput directory: {output_dir.resolve()}")

# ============================================================================
# PARAMETERS (using defaults from original compare_methods.py)
# ============================================================================

print("Using DEFAULT parameters (from original compare_methods.py)")
print(f"  beta={CONFIG['beta']}, cutoffparm_lo={CONFIG['cutoffparm_lo']}, "
      f"sigma_lo_scale={CONFIG['sigma_lo_scale']}, eps_lo_ratio={CONFIG['eps_lo_ratio']}")
print(f"  sigma_decay_gamma={CONFIG['sigma_decay_gamma']}")

# ============================================================================
# LOAD RAW VICTRE PHANTOM
# ============================================================================

data_path = Path('../data/generated_roi')

print("\nLoading RAW VICTRE phantom ROI...")
phantom_victre = np.load(data_path / 'victre_phantom_roi.npy')
lesions_roi = np.load(data_path / 'victre_lesions_roi.npy')

nx, ny, nz = phantom_victre.shape
print(f"VICTRE ROI: {nx}x{ny}x{nz} voxels")
print(f"Lesions: {len(lesions_roi)}")
print(f"Raw value range: [{phantom_victre.min()}, {phantom_victre.max()}]")

# Find slice with highest tissue variance (for display slice selection)
slice_variances = np.array([np.var(phantom_victre[:, :, iz].astype(float)) for iz in range(nz)])
high_variance_slice = np.argmax(slice_variances)
print(f"High variance slice: {high_variance_slice} (variance={slice_variances[high_variance_slice]:.2f})")

# ============================================================================
# NORMALIZE RAW PHANTOM TO ATTENUATION VALUES
# ============================================================================
# VICTRE phantom values: 0=air, ~29=adipose, ~100+=glandular tissue, 200+=calcifications
# Convert to realistic X-ray attenuation coefficients (cm^-1)
# Typical breast tissue attenuation at ~20 keV: 0.4-0.8 cm^-1

print("\nNormalizing raw phantom to attenuation values...")

# Normalize to range [0, 1] for attenuation-like values
# This preserves the full tissue structure
phantom_3d = phantom_victre.astype(np.float64) / 255.0

# Scale to realistic attenuation range (0 to ~0.8 cm^-1)
# Air (0) stays 0, max tissue gets ~0.8
ATTENUATION_SCALE = 0.8
phantom_3d = phantom_3d * ATTENUATION_SCALE

print(f"Normalized phantom range: [{phantom_3d.min():.4f}, {phantom_3d.max():.4f}] cm^-1")
print(f"Physical size: {nx*0.1:.1f} x {ny*0.1:.1f} x {nz*0.1:.1f} mm")  # 0.1mm voxels from VICTRE

ximageside = 10.0
yimageside = 10.0
dx = ximageside/nx
dy = yimageside/ny

xar = arange(-ximageside/2. + dx/2, ximageside/2., dx)[:, newaxis]*ones([ny])
yar = ones([nx, ny])*arange(-yimageside/2. + dy/2, yimageside/2., dy)
rar = sqrt(xar**2 + yar**2)
mask = zeros((nx, ny))
mask[rar <= ximageside/2.] = 1.

# Apply circular FOV mask to 3D phantom - corners become zero (air)
# This matches the circular fan-beam geometry used in the original compare_methods.py
for iz in range(nz):
    phantom_3d[:, :, iz] = phantom_3d[:, :, iz] * mask
print(f"Applied circular FOV mask (radius = {ximageside/2:.1f})")
n_valid_pixels = mask.sum()  # Number of valid pixels for RMSE calculation

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
larc = CONFIG['larc']  # Limited angular range parameter
epssc = CONFIG['eps']*sqrt(nrays)

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
han_lo = clip(hanning_window(uar, CONFIG['cutoffparm_lo']), 0.0, 1.0)
han_hi = clip(1.0 - hanning_window(uar, CONFIG['cutoffparm']), 0.0, 1.0)
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

# Determine number of slices to process
nz_loop = 1 if args.quick else nz
if args.quick:
    print(f"\nGenerating sinogram data for {nz_loop} slice (quick mode)...")
else:
    print(f"\nGenerating sinogram data for all {nz} slices...")
    
truesino_3d = zeros([nz, nviews, nbins])
if args.quick:
    # Use high variance slice for quick mode
    iz = high_variance_slice
    phimage_slice = phantom_3d[:, :, iz]
    circularFanbeamProjection(phimage_slice, truesino_3d[iz])
    print(f"  Generated sinogram for slice {iz+1}/{nz} (high variance slice)")
else:
    for iz in range(nz_loop):
        phimage_slice = phantom_3d[:, :, iz]
        circularFanbeamProjection(phimage_slice, truesino_3d[iz])
        if iz % 3 == 0:
            print(f"  Generated sinogram for slice {iz+1}/{nz_loop}")

sinodata_3d = truesino_3d * 1.
print(f"Sinogram data shape: {sinodata_3d.shape}")

# Ground truth TV (for high variance slice as reference)
phimage_ref = phantom_3d[:, :, high_variance_slice]
xg = gradx(phimage_ref); truetvx = sqrt(xg**2).sum()
yg = grady(phimage_ref); truetvy = sqrt(yg**2).sum()
xg, yg = gradim(phimage_ref); truetv = sqrt(xg**2 + yg**2).sum()
print(f"Ground truth TV (high variance slice): {truetv:.2f}")

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
gnorm = sqrt(xnorm2 + 1e-12); nuxgrad = CONFIG['nuxfact']/gnorm

xim = randn(nx, ny)*mask
for _ in range(npower):
    yg = grady(xim); xim *= 0.; xim = mdivy(yg); xim *= mask
    xnorm2 = sqrt((xim**2.).sum()); xim /= (xnorm2 + 1e-12)
gnorm = sqrt(xnorm2 + 1e-12); nuygrad = CONFIG['nuyfact']/gnorm

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
# METHOD 1: SINGLE-CHANNEL L1-DTV - 3D (SLICE-BY-SLICE)
# ============================================================================

if 'single_channel' in args.algorithms:
    print("\n" + "="*70)
    print("METHOD 1: SINGLE-CHANNEL L1-DTV (3D reconstruction)")
    print("="*70)

# Compute total norm (using first slice)
sinodata_first = sinodata_3d[0]
sinodata_single_temp = fo(sinodata_first)
worksino = zeros([nviews, nbins])

xim = randn(nx, ny)*mask; xim1 = xim*0.; xim2 = xim*0.
for _ in range(200):
    circularFanbeamProjection(xim, worksino)
    w = fo(worksino); w *= nusino
    xg = gradx(xim)*nuxgrad; yg = grady(xim)*nuygrad; yimloc = CONFIG['l1f']*xim
    mag1 = sqrt((yimloc**2).sum() + (yg**2).sum() + (xg**2).sum() + (w**2).sum())
    if mag1>0: yimloc/=mag1; yg/=mag1; xg/=mag1; w/=mag1
    xim1 *= 0.; circularFanbeamBackProjection(fo(w), xim1); xim1 *= (nusino*mask)
    xim2 = mdivx(xg)*(nuxgrad*mask); xim3 = mdivy(yg)*(nuygrad*mask)
    xim = xim1 + xim2 + xim3 + CONFIG['l1f']*yimloc
    mag2 = sqrt((xim**2.).sum())
    if mag2>0: xim /= mag2

totalnorm_single = (mag1 + mag2)*0.5
sig_single = CONFIG['stepbalance']/totalnorm_single
tau_single = 1./(totalnorm_single*CONFIG['stepbalance'])
print(f"Total norm={totalnorm_single:.4f}, sig={sig_single:.6f}, tau={tau_single:.6f}")

# Storage for 3D reconstruction
recon_single_3d = zeros([nx, ny, nz])
ierrs_single_all = []  # Store final errors for all slices
# Store full convergence history for ALL slices (keyed by slice index)
single_convergence_history = {}  # {slice_idx: {'ierrs': [], 'derrs': [], 'tvs': []}}

t0_total = time.time()

# LOOP OVER Z-SLICES
if args.quick:
    iz = high_variance_slice
    print(f"\n--- Reconstructing slice {iz+1}/{nz} ---")
    sinodata = sinodata_3d[iz]
    phimage = phantom_3d[:, :, iz]
    
    # Single iteration for quick mode
    iz_loop = [iz]
else:
    iz_loop = list(range(nz_loop))

for iz in iz_loop:
    if not args.quick:
        print(f"\n--- Reconstructing slice {iz+1}/{nz_loop} ---")
        sinodata = sinodata_3d[iz]
        phimage = phantom_3d[:, :, iz]

    sinodata_single = fo(sinodata)
    sinodatasc_single = nusino*sinodata_single

    # Initialize
    xim = zeros([nx,ny]); yim = xim*0.; xbarim = xim*0.; wimp = xim*0.
    ysino_single = zeros([nviews, nbins]); ygradx = zeros([nx,ny]); ygrady = zeros([nx,ny])
    ierrs_single = []  # Image RMSE
    derrs_single = []  # Data RMSE
    tvs_single = []    # Total Variation

    t0 = time.time()
    for itr in range(1, CONFIG['itermax']+1):
        ysinoold = ysino_single.copy(); ygradxold=ygradx.copy(); ygradyold=ygrady.copy(); yimold=yim.copy()

        # Primal
        wimp *= 0.; circularFanbeamBackProjection(fo(ysino_single), wimp); wimp *= nusino; wimp *= mask
        wimqx = mdivx(ygradx)*nuxgrad*mask; wimqy = mdivy(ygrady)*nuygrad*mask; wiml1 = CONFIG['l1f']*yim
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
        ygradx = (2.-CONFIG['alpha'])*ptilx/maximum(abs(ptilx), (2.-CONFIG['alpha']))
        tgy = grady(xbarim)*nuygrad; ptily = ygrady + sig_single*tgy
        ygrady = CONFIG['alpha']*ptily/maximum(abs(ptily), CONFIG['alpha'])
        ptil1 = yim + sig_single*(CONFIG['l1f']*xbarim)
        yim = CONFIG['beta']*ptil1/maximum(sqrt(ptil1**2), CONFIG['beta'])

        # Predictor-corrector
        ygradx = ygradxold - CONFIG['rho']*(ygradxold - ygradx)
        ygrady = ygradyold - CONFIG['rho']*(ygradyold - ygrady)
        ysino_single = ysinoold - CONFIG['rho']*(ysinoold - ysino_single)
        yim = yimold - CONFIG['rho']*(yimold - yim)
        xim = ximold - CONFIG['rho']*(ximold - xim)

        # Track metrics (using mask for valid FOV pixels only)
        ierrs_single.append(sqrt(((xbarim - phimage)**2 * mask).sum() / n_valid_pixels))  # Image RMSE
        derrs_single.append(sqrt((resid**2).sum())/(nusino*sqrt(nviews*nbins)))  # Data RMSE
        tgx_tv, tgy_tv = gradim(xbarim)
        tvs_single.append(sqrt((tgx_tv**2 + tgy_tv**2)).sum())  # Total Variation

        if itr in CONFIG['istops']:
            print(f"[single] it {itr:4d}  img_err={ierrs_single[-1]:.6e}  data_err={derrs_single[-1]:.6e}  TV={tvs_single[-1]:.2f}")

    slice_time = time.time()-t0
    recon_single_3d[:, :, iz] = xbarim.copy()
    ierrs_single_all.append(ierrs_single[-1])

    # Save convergence history for this slice (all three metrics)
    single_convergence_history[iz] = {
        'ierrs': ierrs_single.copy(),
        'derrs': derrs_single.copy(),
        'tvs': tvs_single.copy()
    }

    print(f"Slice {iz+1} done in {slice_time:.2f}s, final RMSE={ierrs_single[-1]:.6f}")

single_time = time.time()-t0_total
avg_rmse_single = mean(ierrs_single_all)
print(f"\nSingle-channel 3D done in {single_time:.2f}s")
print(f"Average RMSE across all slices: {avg_rmse_single:.6f}")

# ============================================================================
# METHOD 2: TWO-CHANNEL L1-DTV - 3D (SLICE-BY-SLICE)
# ============================================================================

if 'two_channel' in args.algorithms:
    print("\n" + "="*70)
    print("METHOD 2: TWO-CHANNEL L1-DTV (3D reconstruction)")
    print("="*70)

# Compute total norm (using first slice)
sinodata_first = sinodata_3d[0]
worksino = zeros([nviews, nbins])

xim = randn(nx, ny)*mask; xim1 = xim*0.; xim2 = xim*0.
for _ in range(200):
    circularFanbeamProjection(xim, worksino)
    s_hi = R_hi(worksino)*nusino; s_lo = R_lo(worksino)*nusino
    xg = gradx(xim)*nuxgrad; yg = grady(xim)*nuygrad; yimloc = CONFIG['l1f']*xim
    mag1 = sqrt((yimloc**2).sum() + (yg**2).sum() + (xg**2).sum() + (s_hi**2).sum() + (s_lo**2).sum())
    if mag1>0: yimloc/=mag1; yg/=mag1; xg/=mag1; s_hi/=mag1; s_lo/=mag1
    xim1 *= 0.; imtmp = xim1*0.; circularFanbeamBackProjection(s_hi, imtmp); xim1 += imtmp
    imtmp *= 0.; circularFanbeamBackProjection(s_lo, imtmp); xim1 += imtmp
    xim1 *= (nusino*mask)
    xim2 = mdivx(xg)*(nuxgrad*mask); xim3 = mdivy(yg)*(nuygrad*mask)
    xim = xim1 + xim2 + xim3 + CONFIG['l1f']*yimloc
    mag2 = sqrt((xim**2.).sum())
    if mag2>0: xim /= mag2

totalnorm_two = (mag1 + mag2)*0.5
sig_two = CONFIG['stepbalance']/totalnorm_two; tau_two = 1./(totalnorm_two*CONFIG['stepbalance'])

# Unified decay factor
gamma = CONFIG['sigma_decay_gamma']

# Sigma decay parameters
sig_hi = sig_two
sig_lo_initial = CONFIG['sigma_lo_scale'] * sig_two  # Start at 4x sig_hi
sig_lo_final = sig_hi                                 # Decay to 1x sig_hi

# Epsilon decay parameters (UNIFIED with sigma decay)
eps_hi = CONFIG['eps']
eps_lo_initial = CONFIG['eps_lo_ratio'] * CONFIG['eps']  # Start at 1.25x eps_hi
eps_lo_final = eps_hi                                     # Decay to 1x eps_hi
epssc_hi = eps_hi * sqrt(nrays)

print(f"Total norm={totalnorm_two:.4f}, tau={tau_two:.6f}")
print(f"UNIFIED DECAY with gamma={gamma}:")
print(f"  sig_lo: {sig_lo_initial:.6f} -> {sig_lo_final:.6f} (ratio: {CONFIG['sigma_lo_scale']}x -> 1x)")
print(f"  eps_lo: {eps_lo_initial:.6f} -> {eps_lo_final:.6f} (ratio: {CONFIG['eps_lo_ratio']}x -> 1x)")

# Storage for 3D reconstruction
recon_two_3d = zeros([nx, ny, nz])
ierrs_two_all = []  # Store final errors for all slices
# Store full convergence history for ALL slices (keyed by slice index)
two_convergence_history = {}  # {slice_idx: {'ierrs': [], 'derrs': [], 'tvs': []}}

t0_total = time.time()

# LOOP OVER Z-SLICES
if args.quick:
    iz = high_variance_slice
    print(f"\n--- Reconstructing slice {iz+1}/{nz} ---")
    sinodata = sinodata_3d[iz]
    phimage = phantom_3d[:, :, iz]
    
    # Single iteration for quick mode
    iz_loop = [iz]
else:
    iz_loop = list(range(nz_loop))

for iz in iz_loop:
    if not args.quick:
        print(f"\n--- Reconstructing slice {iz+1}/{nz_loop} ---")
        sinodata = sinodata_3d[iz]
        phimage = phantom_3d[:, :, iz]

    sinodata_lo = R_lo(sinodata); sinodata_hi = R_hi(sinodata)
    sinodata_lo_sc = nusino*sinodata_lo; sinodata_hi_sc = nusino*sinodata_hi

    # Initialize
    xim = zeros([nx,ny]); yim = xim*0.; xbarim = xim*0.; wimp = xim*0.
    ysino_hi = zeros([nviews, nbins]); ysino_lo = zeros([nviews, nbins])
    ygradx = zeros([nx,ny]); ygrady = zeros([nx,ny])
    ierrs_two = []   # Image RMSE
    derrs_two = []   # Data RMSE
    tvs_two = []     # Total Variation

    t0 = time.time()
    for itr in range(1, CONFIG['itermax']+1):
        # UNIFIED DECAY: both sig_lo and eps_lo decay with same gamma
        # param(t) = param_final + (param_initial - param_final) * gamma^t
        decay_factor = gamma ** itr
        sig_lo = sig_lo_final + (sig_lo_initial - sig_lo_final) * decay_factor
        eps_lo = eps_lo_final + (eps_lo_initial - eps_lo_final) * decay_factor
        epssc_lo = eps_lo * sqrt(nrays)

        yhi_old=ysino_hi.copy(); ylo_old=ysino_lo.copy(); ygradxold=ygradx.copy(); ygradyold=ygrady.copy(); yimold=yim.copy()

        # Primal
        wimp *= 0.; imtmp = zeros_like(xim)
        circularFanbeamBackProjection(R_hi(ysino_hi), imtmp); wimp += imtmp
        imtmp *= 0.; circularFanbeamBackProjection(R_lo(ysino_lo), imtmp); wimp += imtmp
        wimp *= nusino; wimp *= mask
        wimqx = mdivx(ygradx)*nuxgrad*mask; wimqy = mdivy(ygrady)*nuygrad*mask; wiml1 = CONFIG['l1f']*yim
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
        ygradx = (2.-CONFIG['alpha'])*ptilx/maximum(abs(ptilx), (2.-CONFIG['alpha']))
        tgy = grady(xbarim)*nuygrad; ptily = ygrady + sig_two*tgy
        ygrady = CONFIG['alpha']*ptily/maximum(abs(ptily), CONFIG['alpha'])
        ptil1 = yim + sig_two*(CONFIG['l1f']*xbarim)
        yim = CONFIG['beta']*ptil1/maximum(sqrt(ptil1**2), CONFIG['beta'])

        # Predictor-corrector
        ygradx = ygradxold - CONFIG['rho']*(ygradxold - ygradx)
        ygrady = ygradyold - CONFIG['rho']*(ygradyold - ygrady)
        ysino_hi = yhi_old - CONFIG['rho']*(yhi_old - ysino_hi)
        ysino_lo = ylo_old - CONFIG['rho']*(ylo_old - ysino_lo)
        yim = yimold - CONFIG['rho']*(yimold - yim)
        xim = ximold - CONFIG['rho']*(ximold - xim)

        # Track metrics (using mask for valid FOV pixels only)
        ierrs_two.append(sqrt(((xbarim - phimage)**2 * mask).sum() / n_valid_pixels))  # Image RMSE
        derr_two = sqrt(((resid_hi/nusino)**2).sum() + ((resid_lo/nusino)**2).sum())/sqrt(nviews*nbins)
        derrs_two.append(derr_two)  # Data RMSE (combined hi+lo)
        tgx_tv, tgy_tv = gradim(xbarim)
        tvs_two.append(sqrt((tgx_tv**2 + tgy_tv**2)).sum())  # Total Variation

        if itr in CONFIG['istops']:
            print(f"[two]    it {itr:4d}  img_err={ierrs_two[-1]:.6e}  data_err={derrs_two[-1]:.6e}  TV={tvs_two[-1]:.2f}  sig_lo={sig_lo:.6f}")

    slice_time = time.time()-t0
    recon_two_3d[:, :, iz] = xbarim.copy()
    ierrs_two_all.append(ierrs_two[-1])

    # Save convergence history for this slice (all three metrics)
    two_convergence_history[iz] = {
        'ierrs': ierrs_two.copy(),
        'derrs': derrs_two.copy(),
        'tvs': tvs_two.copy()
    }

    print(f"Slice {iz+1} done in {slice_time:.2f}s, final RMSE={ierrs_two[-1]:.6f}")

two_time = time.time()-t0_total
avg_rmse_two = mean(ierrs_two_all)
print(f"\nTwo-channel 3D done in {two_time:.2f}s")
print(f"Average RMSE across all slices: {avg_rmse_two:.6f}")

# ============================================================================
# FIGURE: RAW VICTRE PHANTOM RECONSTRUCTION (Paper format: x-y top, x-z bottom)
# ============================================================================

print("\n" + "="*70)
print("CREATING FIGURE - RAW VICTRE PHANTOM RECONSTRUCTION (TRUE 3D)")
print("="*70)

fig = plt.figure(figsize=(15, 10))

# Define grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)

# Use full attenuation range for visualization
vmin, vmax = 0, ATTENUATION_SCALE

# ===== TOP ROW: x-y plane (in-plane) - Use high variance slice =====
z_mid = high_variance_slice
y_mid = ny // 2  # For x-z plane extraction

ax00 = fig.add_subplot(gs[0, 0])
ax00.imshow(phantom_3d[:, :, z_mid].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax00.set_title('Raw Phantom\n(x-y plane)', fontsize=13, fontweight='bold')
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

# ===== BOTTOM ROW: x-z plane (DEPTH PLANE - Shows tissue structure in depth) =====
# This is the key figure showing depth resolution!
# Extract x-z slice by taking all x, all z, at fixed y

phantom_xz = phantom_3d[:, y_mid, :]  # Shape: [nx, nz]
single_xz = recon_single_3d[:, y_mid, :]  # Shape: [nx, nz]
two_xz = recon_two_3d[:, y_mid, :]  # Shape: [nx, nz]

ax10 = fig.add_subplot(gs[1, 0])
ax10.imshow(phantom_xz.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
ax10.set_title('Raw Phantom\n(x-z plane)', fontsize=13, fontweight='bold')
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

plt.savefig('../results/raw_victre/figure_raw_victre.png', dpi=200, bbox_inches='tight')
print("Saved: ../results/raw_victre/figure_raw_victre.png")
print("Top row: x-y plane (in-plane imaging)")
print("Bottom row: x-z plane (depth resolution - shows tissue structure)")

# ============================================================================
# DEPTH PROFILES (cross-sections through tissue)
# ============================================================================

print("\n" + "="*70)
print("CREATING DEPTH PROFILE - RAW VICTRE PHANTOM")
print("="*70)

# Depth profile (z-axis) at x=center, y=center
# This demonstrates depth resolution capability
x_profile_idx = nx // 2  # center (x=0)
y_profile_idx = ny // 2  # center (y=0)

z_range = arange(nz)
dz = 5.0 / nz
z_coords_plot = z_range * dz + dz/2.0  # z coordinates in cm

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Extract depth profiles
phantom_depth = phantom_3d[x_profile_idx, y_profile_idx, :]
single_depth = recon_single_3d[x_profile_idx, y_profile_idx, :]
two_depth = recon_two_3d[x_profile_idx, y_profile_idx, :]

ax.plot(z_coords_plot, phantom_depth, 'k-', linewidth=3, label='Raw Phantom (Ground Truth)', marker='o', markersize=6)
ax.plot(z_coords_plot, single_depth, 'r-', linewidth=2.5, label='Single-channel L1-DTV', alpha=0.8, marker='s', markersize=5)
ax.plot(z_coords_plot, two_depth, 'b-', linewidth=2.5, label='Two-channel L1-DTV', alpha=0.8, marker='^', markersize=5)

ax.set_xlabel('z position (depth, cm)', fontsize=14)
ax.set_ylabel('Attenuation (cm⁻¹)', fontsize=14)
ax.set_title('Depth Profile through Raw VICTRE Phantom (x=center, y=center)', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/raw_victre/raw_victre_profile.png', dpi=200, bbox_inches='tight')
print("Saved: ../results/raw_victre/raw_victre_profile.png")
print("This shows depth resolution on raw tissue structure")

# ============================================================================
# CONVERGENCE PLOT (for high variance slice)
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.plot(single_convergence_history[high_variance_slice]['ierrs'], 'r-', linewidth=2.5, label='Single-channel L1-DTV')
ax.plot(two_convergence_history[high_variance_slice]['ierrs'], 'b-', linewidth=2.5, label='Two-channel L1-DTV')
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Image RMSE', fontsize=14)
ax.set_yscale('log')
ax.set_title(f'Convergence: Raw VICTRE Phantom (Slice {high_variance_slice+1})', fontsize=15, fontweight='bold')
ax.legend(fontsize=13, loc='best')
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('../results/raw_victre/raw_victre_convergence.png', dpi=200, bbox_inches='tight')
print("Saved: ../results/raw_victre/raw_victre_convergence.png")
print("Convergence plot shows high variance slice")

# ============================================================================
# PERFORMANCE SUMMARY PLOT (all slices)
# ============================================================================

print("\n" + "="*70)
print("CREATING PERFORMANCE SUMMARY - ALL SLICES")
print("="*70)

fig = plt.figure(figsize=(15, 10))

# Compute differences for plotting
differences = np.array(ierrs_two_all) - np.array(ierrs_single_all)
slice_numbers = np.arange(1, nz_loop+1)

# Plot 1: RMSE for all slices
ax1 = plt.subplot(3, 1, 1)
ax1.plot(slice_numbers, ierrs_single_all, 'r-', linewidth=2, label='Single-channel', alpha=0.8, marker='o', markersize=4)
ax1.plot(slice_numbers, ierrs_two_all, 'b-', linewidth=2, label='Two-channel', alpha=0.8, marker='s', markersize=4)
ax1.set_xlabel('Slice Number', fontsize=12)
ax1.set_ylabel('RMSE', fontsize=12)
ax1.set_title('Reconstruction Error Across All Slices', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Difference (positive = single better, negative = two better)
ax2 = plt.subplot(3, 1, 2)
colors = ['red' if d > 0 else 'blue' for d in differences]
ax2.bar(slice_numbers, differences, color=colors, alpha=0.7)
ax2.set_xlabel('Slice Number', fontsize=12)
ax2.set_ylabel('RMSE Difference\n(Two - Single)', fontsize=12)
ax2.set_title('Performance Difference per Slice (Red: Single Better, Blue: Two Better)', fontsize=14, fontweight='bold')
ax2.axhline(0, color='black', linewidth=1)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Histogram of differences
ax3 = plt.subplot(3, 1, 3)
# Use builtins to avoid numpy min/max which expect arrays
import builtins
n_bins = builtins.max(5, builtins.min(20, nz_loop//2))  # At least 5 bins, max 20 bins
ax3.hist(differences, bins=n_bins, edgecolor='black', alpha=0.7, color='gray')
ax3.set_xlabel('RMSE Difference (Two - Single)', fontsize=12)
ax3.set_ylabel('Number of Slices', fontsize=12)
ax3.set_title('Distribution of Performance Differences', fontsize=14, fontweight='bold')
ax3.axvline(0, color='black', linewidth=2, linestyle='--', label='Equal performance')
ax3.axvline(np.mean(differences), color='green', linewidth=2, label=f'Mean = {np.mean(differences):.6f}')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../results/raw_victre/raw_victre_performance_summary.png', dpi=150, bbox_inches='tight')
print("Saved: ../results/raw_victre/raw_victre_performance_summary.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("FINAL COMPARISON SUMMARY - 3D RECONSTRUCTION")
print("="*70)

# Compute slice-by-slice wins
differences = np.array(ierrs_two_all) - np.array(ierrs_single_all)
single_wins = np.sum(differences > 0)  # Two-channel had higher error
two_wins = np.sum(differences < 0)     # Two-channel had lower error
ties = np.sum(differences == 0)

print(f"\nSingle-channel L1-DTV (Paper method):")
print(f"  Wins: {single_wins}/{nz_loop} slices ({single_wins/nz_loop*100:.1f}%)")
print(f"  Average RMSE: {avg_rmse_single:.6f}")
print(f"  Runtime: {single_time:.2f}s ({single_time/nz_loop:.2f}s per slice)")

print(f"\nTwo-channel L1-DTV (Frequency-split method):")
print(f"  Wins: {two_wins}/{nz_loop} slices ({two_wins/nz_loop*100:.1f}%)")
print(f"  Average RMSE: {avg_rmse_two:.6f}")
print(f"  Runtime: {two_time:.2f}s ({two_time/nz_loop:.2f}s per slice)")

if ties > 0:
    print(f"\nTies: {ties}/{nz_loop} slices")

# Overall winner
if single_wins > two_wins:
    print(f"\nSingle-channel WINS on more slices ({single_wins} vs {two_wins})")
elif two_wins > single_wins:
    print(f"\nTwo-channel WINS on more slices ({two_wins} vs {single_wins})")
else:
    print(f"\n≈ Both methods win equal number of slices ({single_wins} each)")

print(f"\nAverage RMSE difference: {np.mean(differences):.8f}")
print(f"Median RMSE difference: {np.median(differences):.8f}")

print("\n" + "="*70)
print("RAW VICTRE PHANTOM RECONSTRUCTION COMPLETE!")
print("="*70)
print(f"Reconstructed {nz_loop} slices with {CONFIG['itermax']} iterations each")
print(f"Total iterations: {nz_loop * CONFIG['itermax']}")
print("\nFigures saved:")
print("  - ../results/raw_victre/figure_raw_victre.png (x-y and x-z planes)")
print("  - ../results/raw_victre/raw_victre_profile.png (depth profile)")
print("  - ../results/raw_victre/raw_victre_convergence.png (iteration convergence)")
print("  - ../results/raw_victre/raw_victre_performance_summary.png (slice-by-slice comparison)")

# ============================================================================
# SAVE RESULTS FOR POST-PROCESSING
# ============================================================================

print("\nSaving results for post-processing scripts...")
results_data = {
    'single_channel': recon_single_3d,
    'two_channel': recon_two_3d,
    'single_errors': ierrs_single_all,
    'two_errors': ierrs_two_all,
    'phantom': phantom_3d,
    'phantom_raw': phantom_victre,  # Also save original raw phantom
    'attenuation_scale': ATTENUATION_SCALE,
    # Full convergence history for ALL slices (keyed by slice index)
    # Each entry: {slice_idx: {'ierrs': [], 'derrs': [], 'tvs': []}}
    'single_convergence': single_convergence_history,
    'two_convergence': two_convergence_history,
    'high_variance_slice': high_variance_slice
}
with open('raw_victre_results.pkl', 'wb') as f:
    pickle.dump(results_data, f)
print("Saved: raw_victre_results.pkl")
print("  (Use this with compare_best_worst_slices.py for detailed slice analysis)")
print("  (Includes convergence histories for all slices)")
print("  (Includes raw phantom data for reference)")

print("="*70)
plt.show()
