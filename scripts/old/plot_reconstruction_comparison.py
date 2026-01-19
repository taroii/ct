"""
Generate 2x2 reconstruction comparison figure for 256x256 case.

Top row: Reconstructions
Bottom row: Difference from ground truth

Loads cached results from multiresolution_results.pkl
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

CACHE_FILE = 'multiresolution_results.pkl'
RESULTS_DIR = 'final_figures/'

# Load cached results
if not os.path.exists(CACHE_FILE):
    print(f"Error: Cache file '{CACHE_FILE}' not found.")
    print("Run compare_methods_multiresolution.py --force first.")
    exit(1)

print(f"Loading results from '{CACHE_FILE}'...")
with open(CACHE_FILE, 'rb') as f:
    all_results = pickle.load(f)

# Check if images are in the cache
if 'xbarim_single' not in all_results[256]:
    print("Error: Reconstructed images not found in cache.")
    print("Run compare_methods_multiresolution.py --force to regenerate with images.")
    exit(1)

# Extract 256x256 results
r = all_results[256]
xbarim_single = r['xbarim_single']
xbarim_two = r['xbarim_two']
phimage = r['phimage']
rmse_single = r['final_rmse_single']
rmse_two = r['final_rmse_two']

print(f"Loaded 256x256 results:")
print(f"  Single-channel RMSE: {rmse_single:.6f}")
print(f"  Two-channel RMSE: {rmse_two:.6f}")

# Compute differences
diff_single = xbarim_single - phimage
diff_two = xbarim_two - phimage

# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(7, 7))

# Determine vmin/vmax for reconstructions (same scale for both)
vmin_recon = 0
vmax_recon = np.max(phimage) * 1.1

# Top left: Single-channel reconstruction
ax = axes[0, 0]
im = ax.imshow(xbarim_single.T, cmap='gray', origin='lower', vmin=vmin_recon, vmax=vmax_recon)
ax.set_title('Single-Channel', fontsize=11)
ax.set_ylabel('reconstruction', fontsize=11)
ax.set_xticks([])
ax.set_yticks([])

# Top right: Two-channel reconstruction
ax = axes[0, 1]
im = ax.imshow(xbarim_two.T, cmap='gray', origin='lower', vmin=vmin_recon, vmax=vmax_recon)
ax.set_title('Two-Channel', fontsize=11)
ax.axis('off')

# Bottom left: Difference single-channel
ax = axes[1, 0]
im_diff1 = ax.imshow(diff_single.T, cmap='gray', origin='lower', vmin=-0.75, vmax=0.75)
ax.set_ylabel('difference', fontsize=11)
ax.set_xticks([])
ax.set_yticks([])

# Bottom right: Difference two-channel
ax = axes[1, 1]
im_diff2 = ax.imshow(diff_two.T, cmap='gray', origin='lower', vmin=-0.75, vmax=0.75)
ax.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)

# Save
plt.savefig(RESULTS_DIR + 'reconstruction_2x2_256.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {RESULTS_DIR}reconstruction_2x2_256.png")

# ============================================================================
# CONVERGENCE PLOT (256x256)
# ============================================================================

ierrs_single = r['ierrs_single']
ierrs_two = r['ierrs_two']
iterations = range(1, len(ierrs_single) + 1)

fig2, ax = plt.subplots(figsize=(5, 4))

ax.semilogy(iterations, ierrs_single, 'r-', linewidth=1.5, label='Single-Channel')
ax.semilogy(iterations, ierrs_two, 'b-', linewidth=1.5, label='Two-Channel')

ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Image RMSE', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(RESULTS_DIR + 'convergence_256.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR}convergence_256.png")

# ============================================================================
# CONVERGENCE PLOT LOG-LOG (256x256)
# ============================================================================

fig3, ax = plt.subplots(figsize=(5, 4))

ax.loglog(iterations, ierrs_single, 'r-', linewidth=1.5, label='Single-Channel')
ax.loglog(iterations, ierrs_two, 'b-', linewidth=1.5, label='Two-Channel')

ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Image RMSE', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(RESULTS_DIR + 'convergence_256_loglog.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR}convergence_256_loglog.png")

# ============================================================================
# CONVERGENCE PLOT SIDE BY SIDE (256x256)
# ============================================================================

fig4, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))

# Left: semilogy
ax_left.semilogy(iterations, ierrs_single, 'r-', linewidth=1.5, label='Single-Channel')
ax_left.semilogy(iterations, ierrs_two, 'b-', linewidth=1.5, label='Two-Channel')
ax_left.legend(fontsize=10)
ax_left.grid(True, alpha=0.3, which='both')

# Right: loglog
ax_right.loglog(iterations, ierrs_single, 'r-', linewidth=1.5, label='Single-Channel')
ax_right.loglog(iterations, ierrs_two, 'b-', linewidth=1.5, label='Two-Channel')
ax_right.legend(fontsize=10)
ax_right.grid(True, alpha=0.3, which='both')

# Shared axis labels
fig4.supxlabel('Iteration', fontsize=11)
fig4.supylabel('Image RMSE', fontsize=11)

plt.tight_layout()
plt.savefig(RESULTS_DIR + 'convergence_256_sidebyside.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR}convergence_256_sidebyside.png")

plt.show()
