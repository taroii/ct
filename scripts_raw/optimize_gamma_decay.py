"""
Optimize decay gamma for Two-Channel L1-DTV with unified decay.

Both sigma_lo and eps_lo decay from elevated initial values toward the
high-frequency channel values using the SAME gamma:

  sig_lo(t) = sig_hi + (sig_lo_initial - sig_hi) * gamma^t
  eps_lo(t) = eps_hi + (eps_lo_initial - eps_hi) * gamma^t

As iterations progress, both channels converge to identical behavior.
Only ONE parameter (gamma) needs to be optimized.

Runs on a small set of representative slices (default: 5) selected by variance.
"""

import numpy as np
from numpy import *
from numpy.random import randn
import matplotlib.pyplot as plt
from pathlib import Path
import time
from numba import njit
import argparse

# Import the ECP optimizer
from ecp_optimizer import ECPOptimizer

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Optimize unified decay gamma for two-channel reconstruction')
parser.add_argument('--n-slices', type=int, default=5,
                    help='Number of slices to use for optimization (default: 5)')
parser.add_argument('--slice-selection', type=str, default='variance',
                    choices=['variance', 'even', 'random'],
                    help='How to select slices: variance (by tissue variance), even (evenly spaced), random')
parser.add_argument('--iter-opt', type=int, default=500,
                    help='Iterations per reconstruction during optimization (default: 300)')
parser.add_argument('--max-evals', type=int, default=25,
                    help='Maximum number of gamma values to evaluate (default: 25)')
parser.add_argument('--gamma-min', type=float, default=0.8,
                    help='Minimum gamma value to search (default: 0.90)')
parser.add_argument('--gamma-max', type=float, default=0.999,
                    help='Maximum gamma value to search (default: 0.999)')
args = parser.parse_args()

# ============================================================================
# CONFIGURATION (from run_reconstruction_comparison_decay.py)
# ============================================================================

# Fixed parameters (not being optimized)
CONFIG = {
    'alpha': 1.75,
    'beta': 5.0,
    'rho': 1.75,
    'eps': 0.001,
    'nuxfact': 0.5,
    'nuyfact': 0.5,
    'l1f': 1.0,
    'larc': 1.0,
    'stepbalance': 100.0,
    'cutoffparm': 4.0,
    'cutoffparm_lo': 8.0,
    'sigma_lo_scale': 4.0,
    'eps_lo_ratio': 1.25,
}

# Optimization settings
N_SLICES = args.n_slices
SLICE_SELECTION = args.slice_selection
ITER_OPT = args.iter_opt  # Iterations per reconstruction during optimization
MAX_EVALS = args.max_evals
GAMMA_BOUNDS = (args.gamma_min, args.gamma_max)

print("="*70)
print("OPTIMIZE UNIFIED DECAY GAMMA")
print("="*70)
print(f"Parameter to optimize: gamma in [{GAMMA_BOUNDS[0]:.3f}, {GAMMA_BOUNDS[1]:.3f}]")
print(f"\nUnified decay model:")
print(f"  sig_lo(t) = sig_hi + (sig_lo_init - sig_hi) * gamma^t")
print(f"  eps_lo(t) = eps_hi + (eps_lo_init - eps_hi) * gamma^t")
print(f"\nInitial ratios (from CONFIG):")
print(f"  sigma_lo_scale = {CONFIG['sigma_lo_scale']} (sig_lo_init = {CONFIG['sigma_lo_scale']}x sig_hi)")
print(f"  eps_lo_ratio   = {CONFIG['eps_lo_ratio']} (eps_lo_init = {CONFIG['eps_lo_ratio']}x eps_hi)")
print(f"\nSlice selection: {SLICE_SELECTION} ({N_SLICES} slices)")
print(f"Iterations per reconstruction: {ITER_OPT}")
print(f"Maximum evaluations: {MAX_EVALS}")
print("="*70)

# ============================================================================
# LOAD RAW VICTRE PHANTOM
# ============================================================================

data_path = Path('../data/generated_roi')
output_dir = Path('../results/raw_victre')
output_dir.mkdir(parents=True, exist_ok=True)

print("\nLoading RAW VICTRE phantom ROI...")
phantom_victre = np.load(data_path / 'victre_phantom_roi.npy')
lesions_roi = np.load(data_path / 'victre_lesions_roi.npy')

nx, ny, nz = phantom_victre.shape
print(f"VICTRE ROI: {nx}x{ny}x{nz} voxels")

# Normalize raw phantom to attenuation values
phantom_3d = phantom_victre.astype(np.float64) / 255.0
ATTENUATION_SCALE = 0.8
phantom_3d = phantom_3d * ATTENUATION_SCALE
print(f"Normalized phantom range: [{phantom_3d.min():.4f}, {phantom_3d.max():.4f}]")

# ============================================================================
# GEOMETRY SETUP
# ============================================================================

ximageside = 10.0
yimageside = 10.0
dx = ximageside/nx
dy = yimageside/ny

xar = arange(-ximageside/2. + dx/2, ximageside/2., dx)[:, newaxis]*ones([ny])
yar = ones([nx, ny])*arange(-yimageside/2. + dy/2, yimageside/2., dy)
rar = sqrt(xar**2 + yar**2)
mask = zeros((nx, ny))
mask[rar <= ximageside/2.] = 1.

# Apply circular FOV mask to 3D phantom
for iz in range(nz):
    phantom_3d[:, :, iz] = phantom_3d[:, :, iz] * mask
print(f"Applied circular FOV mask (radius = {ximageside/2:.1f})")
n_valid_pixels = mask.sum()

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
larc = CONFIG['larc']
epssc = CONFIG['eps']*sqrt(nrays)

fanangle2 = arcsin((ximageside/2.)/radius)
detectorlength = 2.*tan(fanangle2)*source_to_detector

# ============================================================================
# PROJECTION/BACKPROJECTION (from run_reconstruction_comparison_decay.py)
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
# GRADIENT OPERATORS
# ============================================================================

gmatx = zeros([nx, nx]); gmatx[range(nx), range(nx)] = -1.0; gmatx[range(nx-1), range(1,nx)] = 1.0
gmaty = zeros([ny, ny]); gmaty[range(ny), range(ny)] = -1.0; gmaty[range(ny-1), range(1,ny)] = 1.0

def gradx(im): return dot(gmatx, im)
def grady(im): return array(dot(gmaty, im.T).T, order="C")
def mdivx(im): return dot(gmatx.T, im)
def mdivy(im): return array(dot(gmaty.T, im.T).T, order="C")

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
# SELECT SLICES FOR OPTIMIZATION
# ============================================================================

# Compute variance for each slice (for variance-based selection)
slice_variances = np.array([np.var(phantom_3d[:, :, iz]) for iz in range(nz)])

if SLICE_SELECTION == 'variance':
    # Select slices with diverse variance levels (mix of high, medium, low variance)
    # Sort by variance and pick evenly from the sorted list
    sorted_indices = np.argsort(slice_variances)
    # Pick N_SLICES evenly from the variance-sorted list to get diverse slices
    pick_indices = np.linspace(0, nz-1, N_SLICES, dtype=int)
    slice_indices = sorted_indices[pick_indices]
    print(f"\nSlice selection: VARIANCE-BASED (diverse variance levels)")
    print(f"  Variance range of selected slices: [{slice_variances[slice_indices].min():.2f}, {slice_variances[slice_indices].max():.2f}]")

elif SLICE_SELECTION == 'even':
    # Evenly spaced across volume
    slice_indices = np.linspace(0, nz-1, N_SLICES, dtype=int)
    print(f"\nSlice selection: EVENLY SPACED")

elif SLICE_SELECTION == 'random':
    # Random selection
    np.random.seed(42)  # Reproducible
    slice_indices = np.random.choice(nz, size=N_SLICES, replace=False)
    slice_indices = np.sort(slice_indices)
    print(f"\nSlice selection: RANDOM (seed=42)")

print(f"Selected {N_SLICES} slices: {slice_indices.tolist()}")
print(f"Slice variances: {[f'{slice_variances[i]:.2f}' for i in slice_indices]}")

# ============================================================================
# GENERATE SINOGRAMS FOR SELECTED SLICES
# ============================================================================

print("\nGenerating sinograms for selected slices...")
sinodata_dict = {}
phimage_dict = {}

for i, iz in enumerate(slice_indices):
    phimage = phantom_3d[:, :, iz]
    sino = zeros([nviews, nbins])
    circularFanbeamProjection(phimage, sino)
    sinodata_dict[iz] = sino
    phimage_dict[iz] = phimage
    if (i+1) % 5 == 0:
        print(f"  Generated {i+1}/{N_SLICES} sinograms")

print(f"Generated sinograms for {N_SLICES} slices")

# ============================================================================
# ESTIMATE OPERATOR NORMS (once, for all reconstructions)
# ============================================================================

print("\nEstimating operator norms...")
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
# COMPUTE TOTAL NORM FOR TWO-CHANNEL (once)
# ============================================================================

sinodata_first = sinodata_dict[slice_indices[0]]
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
sig_two = CONFIG['stepbalance']/totalnorm_two
tau_two = 1./(totalnorm_two*CONFIG['stepbalance'])

# Sigma decay parameters
sig_hi = sig_two
sig_lo_initial = CONFIG['sigma_lo_scale'] * sig_two  # Start at 4x
sig_lo_final = sig_hi                                 # Decay to 1x

# Epsilon decay parameters (UNIFIED with sigma decay)
eps_hi = CONFIG['eps']
eps_lo_initial = CONFIG['eps_lo_ratio'] * CONFIG['eps']  # Start at 1.25x
eps_lo_final = eps_hi                                     # Decay to 1x
epssc_hi = eps_hi * sqrt(nrays)

print(f"Total norm={totalnorm_two:.4f}")
print(f"Sigma:   sig_hi={sig_hi:.6f}, sig_lo: {sig_lo_initial:.6f} -> {sig_lo_final:.6f}")
print(f"Epsilon: eps_hi={eps_hi:.6f}, eps_lo: {eps_lo_initial:.6f} -> {eps_lo_final:.6f}")

# ============================================================================
# TWO-CHANNEL DECAY RECONSTRUCTION FUNCTION
# ============================================================================

def reconstruct_two_channel_decay(gamma, sinodata, phimage, itermax=ITER_OPT, verbose=False):
    """
    Run two-channel L1-DTV reconstruction with UNIFIED decay for sigma_lo and eps_lo.

    Both parameters decay from elevated initial values toward high-freq values:
        sig_lo(t) = sig_hi + (sig_lo_init - sig_hi) * gamma^t
        eps_lo(t) = eps_hi + (eps_lo_init - eps_hi) * gamma^t

    Args:
        gamma: Decay factor (0 < gamma < 1), same for both sigma and epsilon
        sinodata: Sinogram data for this slice
        phimage: Ground truth phantom for this slice
        itermax: Number of iterations
        verbose: Print progress

    Returns:
        Final RMSE
    """
    sinodata_lo = R_lo(sinodata)
    sinodata_hi = R_hi(sinodata)
    sinodata_lo_sc = nusino*sinodata_lo
    sinodata_hi_sc = nusino*sinodata_hi

    # Initialize
    xim = zeros([nx,ny]); yim = xim*0.; xbarim = xim*0.; wimp = xim*0.
    ysino_hi = zeros([nviews, nbins]); ysino_lo = zeros([nviews, nbins])
    ygradx = zeros([nx,ny]); ygrady = zeros([nx,ny])

    for itr in range(1, itermax+1):
        # Unified decay: both sig_lo and eps_lo decay with same gamma
        decay_factor = gamma ** itr
        sig_lo = sig_lo_final + (sig_lo_initial - sig_lo_final) * decay_factor
        eps_lo = eps_lo_final + (eps_lo_initial - eps_lo_final) * decay_factor
        epssc_lo = eps_lo * sqrt(nrays)

        yhi_old = ysino_hi.copy(); ylo_old = ysino_lo.copy()
        ygradxold = ygradx.copy(); ygradyold = ygrady.copy(); yimold = yim.copy()

        # Primal
        wimp *= 0.; imtmp = zeros_like(xim)
        circularFanbeamBackProjection(R_hi(ysino_hi), imtmp); wimp += imtmp
        imtmp *= 0.; circularFanbeamBackProjection(R_lo(ysino_lo), imtmp); wimp += imtmp
        wimp *= nusino; wimp *= mask
        wimqx = mdivx(ygradx)*nuxgrad*mask; wimqy = mdivy(ygrady)*nuygrad*mask
        wiml1 = CONFIG['l1f']*yim
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

    # Calculate RMSE (using mask for valid FOV pixels only)
    rmse = sqrt(((xbarim - phimage)**2 * mask).sum() / n_valid_pixels)
    return rmse

# ============================================================================
# OBJECTIVE FUNCTION: Average RMSE over all selected slices
# ============================================================================

eval_count = [0]  # Mutable counter for tracking

def objective_function(gamma_array):
    """
    Objective function: average RMSE across all selected slices.

    Args:
        gamma_array: Array containing single gamma value

    Returns:
        Average RMSE across all slices
    """
    gamma = gamma_array[0]
    eval_count[0] += 1

    t_start = time.time()
    rmse_values = []

    for iz in slice_indices:
        sinodata = sinodata_dict[iz]
        phimage = phimage_dict[iz]
        rmse = reconstruct_two_channel_decay(gamma, sinodata, phimage)
        rmse_values.append(rmse)

    avg_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    t_elapsed = time.time() - t_start

    print(f"  Eval {eval_count[0]:3d}: gamma={gamma:.6f} -> "
          f"avg_RMSE={avg_rmse:.6f} (std={std_rmse:.6f}) [{t_elapsed:.1f}s]")

    return avg_rmse

# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

print("\n" + "="*70)
print("STARTING GAMMA OPTIMIZATION")
print("="*70)

# Initialize optimizer for 1D search
optimizer = ECPOptimizer(
    bounds=[GAMMA_BOUNDS],
    epsilon_init=0.001,
    tau=1.1,
    C=30,
    max_iters=MAX_EVALS,
    verbose=False  # We do our own verbose output
)

t_opt_start = time.time()
results = optimizer.optimize(objective_function)
t_opt_total = time.time() - t_opt_start

# ============================================================================
# RESULTS
# ============================================================================

best_gamma = results['best_params'][0]
best_rmse = results['best_value']

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70)
print(f"Best gamma: {best_gamma:.6f}")
print(f"Best average RMSE: {best_rmse:.6f}")
print(f"Total evaluations: {results['n_evaluations']}")
print(f"Total time: {t_opt_total:.1f}s ({t_opt_total/60:.1f} min)")

# ============================================================================
# COMPARE WITH DEFAULT GAMMA
# ============================================================================

print("\n" + "-"*50)
print("Comparing with default gamma=0.95...")

default_gamma = 0.95
eval_count[0] = 0  # Reset for display purposes
default_rmse = objective_function(np.array([default_gamma]))

improvement = (default_rmse - best_rmse) / default_rmse * 100
print(f"\nDefault gamma=0.95 RMSE: {default_rmse:.6f}")
print(f"Optimized gamma={best_gamma:.6f} RMSE: {best_rmse:.6f}")
print(f"Improvement: {improvement:.2f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Extract history
history_gamma = np.array([h[0][0] for h in results['history']])
history_rmse = np.array([h[1] for h in results['history']])

np.savez(output_dir / 'gamma_optimization_results.npz',
         best_gamma=best_gamma,
         best_rmse=best_rmse,
         default_gamma=default_gamma,
         default_rmse=default_rmse,
         history_gamma=history_gamma,
         history_rmse=history_rmse,
         n_slices=N_SLICES,
         slice_indices=slice_indices,
         iter_opt=ITER_OPT,
         gamma_bounds=GAMMA_BOUNDS)

print(f"\nResults saved to: {output_dir / 'gamma_optimization_results.npz'}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Optimization convergence
ax1 = axes[0]
ax1.plot(range(1, len(history_rmse)+1), history_rmse, 'b.-', linewidth=2, markersize=8)
ax1.axhline(y=default_rmse, color='r', linestyle='--', linewidth=2, label=f'Default gamma=0.95')
ax1.axhline(y=best_rmse, color='g', linestyle='--', linewidth=2, label=f'Best gamma={best_gamma:.4f}')
ax1.set_xlabel('Function Evaluation', fontsize=12)
ax1.set_ylabel('Average RMSE', fontsize=12)
ax1.set_title('Gamma Optimization Convergence', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Gamma vs RMSE scatter
ax2 = axes[1]
ax2.scatter(history_gamma, history_rmse, c=range(len(history_gamma)),
            cmap='viridis', s=100, edgecolors='black', linewidth=1)
ax2.scatter([best_gamma], [best_rmse], c='red', s=200, marker='*',
            edgecolors='black', linewidth=2, label=f'Best: gamma={best_gamma:.4f}', zorder=5)
ax2.scatter([default_gamma], [default_rmse], c='orange', s=150, marker='s',
            edgecolors='black', linewidth=2, label=f'Default: gamma=0.95', zorder=5)
ax2.set_xlabel('Gamma', fontsize=12)
ax2.set_ylabel('Average RMSE', fontsize=12)
ax2.set_title(f'Gamma vs RMSE ({N_SLICES} slices)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'gamma_optimization.png', dpi=150, bbox_inches='tight')
print(f"Saved plot: {output_dir / 'gamma_optimization.png'}")

plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY - UNIFIED DECAY MODEL")
print("="*70)
print(f"Optimal gamma: {best_gamma:.6f}")
print(f"\nThis gamma controls BOTH decays:")
print(f"  sig_lo(t) = {sig_lo_final:.6f} + ({sig_lo_initial:.6f} - {sig_lo_final:.6f}) * {best_gamma:.4f}^t")
print(f"  eps_lo(t) = {eps_lo_final:.6f} + ({eps_lo_initial:.6f} - {eps_lo_final:.6f}) * {best_gamma:.4f}^t")
print(f"\nTo use in run_reconstruction_comparison_decay.py, update CONFIG:")
print(f"  CONFIG['sigma_decay_gamma'] = {best_gamma:.6f}")
print(f"\nAnd ensure unified decay is implemented for eps_lo as well.")
print("="*70)
