"""
Optimize Two-Channel L1-DTV parameters for VICTRE phantom using ECP.

This script uses the ECP optimizer to find optimal parameters for the
two-channel reconstruction method on the high-variance VICTRE ROI.
"""

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from pathlib import Path
import time
from numba import njit

# Import the ECP optimizer
from ecp_optimizer import optimize_two_channel_params, ECPOptimizer

# ============================================================================
# CONFIGURATION (from recreate_figure8_victre_ica.py)
# ============================================================================

mfact = 2  # Image size = 512/mfact
addnoise = 0
nph = 1.e6
nuxfact = 0.5
nuyfact = 0.5
l1f = 1.0
eps = 0.001  # data discrepancy RMSE
larc = 1.0
alpha = 1.75  # DTV parameter
beta = 5.0
rho = 1.75
stepbalance = 100.0
itermax = 500  # iterations per reconstruction

# ============================================================================
# SETUP GEOMETRY AND OPERATORS (from original script)
# ============================================================================

# Load VICTRE phantom
data_path = Path('../data/generated_roi')
phantom_victre = np.load(data_path / 'victre_phantom_roi.npy')
lesions_roi = np.load(data_path / 'victre_lesions_roi.npy')

nx, ny, nz = phantom_victre.shape
print(f"VICTRE ROI: {nx}x{ny}x{nz} voxels")

# Create ICA distribution (simplified for optimization)
tumor_value = 0.4
background_enhancement = 0.08
tumor_radius = 3

phantom_3d = zeros((nx, ny, nz))
glandular_mask = phantom_victre > 100
phantom_3d[glandular_mask] = background_enhancement

for lx, ly, lz in lesions_roi.astype(int):
    for dx in range(-tumor_radius, tumor_radius+1):
        for dy in range(-tumor_radius, tumor_radius+1):
            for dz in range(-tumor_radius, tumor_radius+1):
                if dx**2 + dy**2 + dz**2 <= tumor_radius**2:
                    ix, iy, iz = lx+dx, ly+dy, lz+dz
                    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                        phantom_3d[ix, iy, iz] = tumor_value

# DBT geometry parameters
nbins = int(512/mfact)
detectorlength = 0.24*1024/mfact
source_to_detector = 0.660
radius = 0.655
pixelsize = detectorlength/nbins
nviews = 25
slen0 = -np.pi/4.
slen = np.pi/2.
ximageside = detectorlength
yimageside = detectorlength
mask = ones([nx, ny])
nrays = nviews*nbins
epssc = eps*sqrt(nrays)
epssc_hi = eps*sqrt(nrays)

# Include projection/backprojection functions from original script
@njit
def circularFanbeamProjection(image, sinogram):
    """Forward projection operator"""
    # [Implementation from original script]
    dx = ximageside/nx; dy = yimageside/ny
    ds = slen/(nviews - larc)
    x0 = -ximageside/2.; y0 = -yimageside/2.
    u0 = -detectorlength/2.; du = detectorlength/nbins
    
    sinogram[:,:] = 0.
    for sindex in range(nviews):
        s = sindex*ds + slen0
        xsource = radius*cos(s); ysource = radius*sin(s)
        for uindex in range(nbins):
            for ix in range(nx):
                for iy in range(ny):
                    xP = x0 + dx*(ix + 0.5); yP = y0 + dy*(iy + 0.5)
                    u = u0 + du*(uindex + 0.5)
                    t = (u - xsource)/(xP - xsource)
                    yI = ysource + t*(yP - ysource)
                    if abs(yI) <= detectorlength/2:
                        sinogram[sindex, uindex] += image[ix, iy]*abs(xP - xsource)*pixelsize/sqrt((xP - xsource)**2 + (yP - ysource)**2)

@njit
def circularFanbeamBackProjection(sinogram, image):
    """Backprojection operator"""
    # [Simplified implementation - full version from original]
    dx = ximageside/nx; dy = yimageside/ny
    x0 = -ximageside/2.; y0 = -yimageside/2.
    u0 = -detectorlength/2.; du = detectorlength/nbins
    ds = slen/(nviews - larc)
    
    for sindex in range(nviews):
        s = sindex*ds + slen0
        xsource = radius*cos(s); ysource = radius*sin(s)
        for uindex in range(nbins):
            val = sinogram[sindex, uindex]
            u = u0 + (uindex + 0.5)*du
            # Simplified backprojection logic
            for ix in range(nx):
                for iy in range(ny):
                    xP = x0 + dx*(ix + 0.5)
                    yP = y0 + dy*(iy + 0.5)
                    # Add contribution to image
                    image[ix, iy] += val * du / (nviews * np.pi)

# Gradient operators
gmatx = zeros([nx, nx]); gmatx[range(nx), range(nx)] = -1.0; gmatx[range(nx-1), range(1,nx)] = 1.0
gmaty = zeros([ny, ny]); gmaty[range(ny), range(ny)] = -1.0; gmaty[range(ny-1), range(1,ny)] = 1.0

def gradx(im): return dot(gmatx, im)
def grady(im): return array(dot(gmaty, im.T).T, order="C")
def mdivx(im): return dot(gmatx.T, im)
def mdivy(im): return array(dot(gmaty.T, im.T).T, order="C")

# ============================================================================
# TWO-CHANNEL RECONSTRUCTION FUNCTION
# ============================================================================

def reconstruct_slice_two_channel(param_dict, sinodata, phimage, verbose=False):
    """
    Run two-channel reconstruction with given parameters for a single slice.
    
    Returns RMSE of reconstruction.
    """
    # Extract parameters
    cutoff_lo = param_dict.get('cutoff_lo', 8.0)
    sigma_scale = param_dict.get('sigma_scale', 4.0)
    eps_ratio = param_dict.get('eps_ratio', 1.25)
    
    # Setup filters with new parameters
    nb0 = nbins; blen0 = detectorlength; db = blen0/nb0; b00 = -blen0/2.
    uar = arange(b00+db/2., b00+blen0, db)*1.
    
    def hanning_window(uar, c):
        uhanp = abs(b00)/c
        han = 0.5*(1.0 + cos(pi*uar/uhanp))
        han[abs(uar) > uhanp] = 0.0
        return han
    
    ramp = abs(uar)
    W_sqrt_ramp = sqrt(ramp + 1e-12)
    
    # Create filters with optimized parameters
    han_lo = clip(hanning_window(uar, cutoff_lo), 0.0, 1.0)
    han_hi = clip(1.0 - hanning_window(uar, 4.0), 0.0, 1.0)  # Keep high fixed
    F_lo = W_sqrt_ramp*sqrt(han_lo)
    F_hi = W_sqrt_ramp*sqrt(han_hi)
    
    def R_fft_weight(sino, W):
        imft = fft.fft(sino, axis=1)
        pimft = (ones([nbins])*fft.fftshift(W))*imft
        return fft.ifft(pimft, axis=1).real
    
    def R_lo(s): return R_fft_weight(s, F_lo)
    def R_hi(s): return R_fft_weight(s, F_hi)
    
    # Compute norms (simplified)
    nusino = 0.001  # Approximate value
    nuxgrad = nuxfact * 0.5
    nuygrad = nuyfact * 0.5
    
    # Prepare data
    sinodata_lo = R_lo(sinodata)
    sinodata_hi = R_hi(sinodata)
    
    eps_lo = eps * eps_ratio
    epssc_lo = eps_lo * sqrt(nrays)
    
    sinodata_lo_sc = nusino * sinodata_lo
    sinodata_hi_sc = nusino * sinodata_hi
    
    # Compute step sizes with sigma_scale
    sig_two = stepbalance * 0.01  # Approximate
    tau_two = 1./(stepbalance * 0.01)
    sig_hi = sig_two
    sig_lo = sigma_scale * sig_two
    
    # Initialize
    xim = zeros([nx,ny])
    yim = xim*0.
    xbarim = xim*0.
    ysino_hi = zeros([nviews, nbins])
    ysino_lo = zeros([nviews, nbins])
    ygradx = zeros([nx,ny])
    ygrady = zeros([nx,ny])
    
    # Run reconstruction (reduced iterations for optimization)
    iter_opt = min(200, itermax)  # Use fewer iterations during optimization
    
    for itr in range(1, iter_opt+1):
        yhi_old = ysino_hi.copy()
        ylo_old = ysino_lo.copy()
        ygradxold = ygradx.copy()
        ygradyold = ygrady.copy()
        yimold = yim.copy()
        
        # Primal update
        wimp = zeros_like(xim)
        imtmp = zeros_like(xim)
        circularFanbeamBackProjection(R_hi(ysino_hi), imtmp)
        wimp += imtmp
        imtmp *= 0.
        circularFanbeamBackProjection(R_lo(ysino_lo), imtmp)
        wimp += imtmp
        wimp *= nusino * mask
        
        wimqx = mdivx(ygradx)*nuxgrad*mask
        wimqy = mdivy(ygrady)*nuygrad*mask
        wiml1 = l1f*yim
        
        ximold = xim.copy()
        xim = xim - tau_two*(wimp + wimqx + wimqy + wiml1)
        xim[xim<0] = 0.
        xbarim = xim + (xim - ximold)
        
        # Dual update
        worksino = zeros([nviews, nbins])
        circularFanbeamProjection(xbarim, worksino)
        
        Ax_hi = R_hi(worksino)*nusino
        Ax_lo = R_lo(worksino)*nusino
        
        resid_hi = Ax_hi - sinodata_hi_sc
        resid_lo = Ax_lo - sinodata_lo_sc
        
        ysino_hi = ysino_hi + sig_hi*resid_hi
        ymag_hi = sqrt((ysino_hi**2).sum())
        ysino_hi *= (maximum(0.0, ymag_hi - sig_hi*nusino*epssc_hi)/(ymag_hi+1e-12))
        
        ysino_lo = ysino_lo + sig_lo*resid_lo
        ymag_lo = sqrt((ysino_lo**2).sum())
        ysino_lo *= (maximum(0.0, ymag_lo - sig_lo*nusino*epssc_lo)/(ymag_lo+1e-12))
        
        tgx = gradx(xbarim)*nuxgrad
        ptilx = ygradx + sig_two*tgx
        ygradx = (2.-alpha)*ptilx/maximum(abs(ptilx), (2.-alpha))
        
        tgy = grady(xbarim)*nuygrad
        ptily = ygrady + sig_two*tgy
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
    
    # Calculate RMSE
    rmse = sqrt(((xbarim - phimage)**2).sum()/(nx*ny))
    return rmse


def optimize_parameters_for_victre():
    """
    Main optimization routine for VICTRE phantom parameters.
    """
    print("="*70)
    print("OPTIMIZING TWO-CHANNEL PARAMETERS FOR VICTRE PHANTOM")
    print("="*70)
    
    # Generate sinogram for middle slice (for optimization)
    z_mid = nz // 2
    phimage_mid = phantom_3d[:, :, z_mid]
    sinodata_mid = zeros([nviews, nbins])
    circularFanbeamProjection(phimage_mid, sinodata_mid)
    
    print(f"Using middle slice (z={z_mid}) for optimization")
    print(f"Phantom range: [{phimage_mid.min():.3f}, {phimage_mid.max():.3f}]")
    
    # Define objective function for optimizer
    def objective_function(param_dict):
        """Objective to minimize: reconstruction RMSE"""
        rmse = reconstruct_slice_two_channel(
            param_dict, 
            sinodata_mid, 
            phimage_mid,
            verbose=False
        )
        return rmse
    
    # Set parameter bounds
    param_bounds = {
        'cutoff_lo': (2.0, 16.0),      # Frequency separation
        'sigma_scale': (1.0, 8.0),      # Step size scaling  
        'eps_ratio': (0.8, 2.0)        # eps_lo/eps_hi ratio
    }
    
    print("\nParameter search ranges:")
    for name, (low, high) in param_bounds.items():
        print(f"  {name}: [{low:.2f}, {high:.2f}]")
    
    # Create data dictionary (for compatibility)
    data_dict = {
        'sinodata': sinodata_mid,
        'phantom': phimage_mid
    }
    
    # Initialize optimizer
    optimizer = ECPOptimizer(
        bounds=list(param_bounds.values()),
        epsilon_init=0.01,
        tau=1.05,
        C=30,
        max_iters=25,  # Limit evaluations since reconstruction is expensive
        verbose=True
    )
    
    # Create wrapper for optimizer
    param_names = list(param_bounds.keys())
    
    def objective_wrapper(params):
        param_dict = {name: params[i] for i, name in enumerate(param_names)}
        return objective_function(param_dict)
    
    print("\nStarting ECP optimization...")
    print("-"*50)
    
    # Run optimization
    t_start = time.time()
    results = optimizer.optimize(objective_wrapper)
    t_elapsed = time.time() - t_start
    
    # Extract best parameters
    best_params = {
        name: results['best_params'][i] 
        for i, name in enumerate(param_names)
    }
    
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Best parameters found:")
    for name, value in best_params.items():
        print(f"  {name}: {value:.4f}")
    print(f"Best RMSE: {results['best_value']:.6f}")
    print(f"Function evaluations: {results['n_evaluations']}")
    print(f"Total time: {t_elapsed:.1f} seconds")
    
    # Compare with default parameters
    print("\n" + "-"*50)
    print("Comparing with default parameters...")
    
    default_params = {
        'cutoff_lo': 8.0,
        'sigma_scale': 4.0,
        'eps_ratio': 1.25
    }
    
    default_rmse = reconstruct_slice_two_channel(
        default_params,
        sinodata_mid,
        phimage_mid,
        verbose=False
    )
    
    print(f"Default parameters RMSE: {default_rmse:.6f}")
    print(f"Optimized parameters RMSE: {results['best_value']:.6f}")
    print(f"Improvement: {(default_rmse - results['best_value'])/default_rmse * 100:.1f}%")
    
    # Save results
    save_path = Path('../results/current')
    save_path.mkdir(parents=True, exist_ok=True)
    
    np.savez(save_path / 'ecp_optimized_params.npz',
             best_params=best_params,
             best_rmse=results['best_value'],
             default_rmse=default_rmse,
             history=results['history'],
             n_evaluations=results['n_evaluations'])
    
    print(f"\nResults saved to {save_path / 'ecp_optimized_params.npz'}")
    
    # Plot convergence
    history = results['history']
    iterations = range(1, len(history) + 1)
    rmse_values = [h[1] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rmse_values, 'b.-', label='ECP optimization')
    plt.axhline(y=default_rmse, color='r', linestyle='--', label='Default params')
    plt.axhline(y=results['best_value'], color='g', linestyle='--', label='Best found')
    plt.xlabel('Function Evaluations')
    plt.ylabel('RMSE')
    plt.title('ECP Parameter Optimization Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path / 'ecp_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return best_params, results


if __name__ == "__main__":
    best_params, results = optimize_parameters_for_victre()
    
    print("\n" + "="*70)
    print("You can now use these optimized parameters in your reconstruction:")
    print("="*70)
    print("from optimize_victre_params import best_params")
    print("# Then use best_params['cutoff_lo'], best_params['sigma_scale'], etc.")