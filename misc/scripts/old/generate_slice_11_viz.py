"""
Generate and visualize slice 11 reconstructions
Standalone script that runs the reconstruction directly
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os
import subprocess
import sys

# Check if results already exist
RESULTS_FILE = 'fig8_victre_results.pkl'

if os.path.exists(RESULTS_FILE):
    print(f"Loading existing results from {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'rb') as f:
        results = pickle.load(f)
    recon_single_3d = results['single_channel']
    recon_two_3d = results['two_channel'] 
    ierrs_single_all = results['single_errors']
    ierrs_two_all = results['two_errors']
    phantom_3d = results['phantom']
    nx, ny, nz = phantom_3d.shape
else:
    print("Need to run the reconstruction script first...")
    print("Please run: python recreate_figure8_victre_ica.py")
    print("This will generate the reconstruction data.")
    print("\nThen run this script again to visualize slice 11.")
    sys.exit(1)

# Now visualize slice 11
slice_idx = 10  # Slice 11 (0-indexed)
print(f"\n{'='*60}")
print(f"ANALYZING SLICE 11 (index {slice_idx})")
print(f"{'='*60}")
print(f"Single-channel RMSE: {ierrs_single_all[slice_idx]:.6f}")
print(f"Two-channel RMSE: {ierrs_two_all[slice_idx]:.6f}")

if ierrs_single_all[slice_idx] < ierrs_two_all[slice_idx]:
    improvement = (ierrs_two_all[slice_idx] - ierrs_single_all[slice_idx])/ierrs_two_all[slice_idx]*100
    print(f"✓ Single-channel is BETTER by {improvement:.2f}%")
else:
    improvement = (ierrs_single_all[slice_idx] - ierrs_two_all[slice_idx])/ierrs_single_all[slice_idx]*100
    print(f"✓ Two-channel is BETTER by {improvement:.2f}%")

# Extract slice 11 data
phantom_slice = phantom_3d[:, :, slice_idx]
single_slice = recon_single_3d[:, :, slice_idx]
two_slice = recon_two_3d[:, :, slice_idx]

# Compute difference maps
diff_single = np.abs(single_slice - phantom_slice)
diff_two = np.abs(two_slice - phantom_slice)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 14))

# Color settings
vmin, vmax = 0, 0.5
diff_vmax = 0.1

# === TOP ROW: Main reconstructions ===
ax1 = plt.subplot(4, 3, 1)
im1 = ax1.imshow(phantom_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax1.set_title('Ground Truth (Phantom)\nSlice 11', fontsize=11, fontweight='bold')
ax1.set_xlabel('x'); ax1.set_ylabel('y')
ax1.axis('off')

ax2 = plt.subplot(4, 3, 2)
im2 = ax2.imshow(single_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax2.set_title(f'Single-channel L1-DTV\nRMSE={ierrs_single_all[slice_idx]:.5f}', fontsize=11, fontweight='bold', color='red')
ax2.set_xlabel('x')
ax2.axis('off')

ax3 = plt.subplot(4, 3, 3)
im3 = ax3.imshow(two_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax3.set_title(f'Two-channel L1-DTV\nRMSE={ierrs_two_all[slice_idx]:.5f}', fontsize=11, fontweight='bold', color='blue')
ax3.set_xlabel('x')
ax3.axis('off')

# === SECOND ROW: Error maps ===
ax4 = plt.subplot(4, 3, 4)
ax4.axis('off')
ax4.text(0.5, 0.5, 'Absolute\nError Maps', ha='center', va='center', fontsize=12, fontweight='bold')

ax5 = plt.subplot(4, 3, 5)
im5 = ax5.imshow(diff_single.T, cmap='hot', origin='lower', vmin=0, vmax=diff_vmax)
ax5.set_title('Single-channel Error', fontsize=11, fontweight='bold', color='red')
ax5.axis('off')
cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
cbar5.ax.tick_params(labelsize=9)

ax6 = plt.subplot(4, 3, 6)
im6 = ax6.imshow(diff_two.T, cmap='hot', origin='lower', vmin=0, vmax=diff_vmax)
ax6.set_title('Two-channel Error', fontsize=11, fontweight='bold', color='blue')
ax6.axis('off')
cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
cbar6.ax.tick_params(labelsize=9)

# === THIRD ROW: Difference and zoomed regions ===
ax7 = plt.subplot(4, 3, 7)
im7 = ax7.imshow((two_slice - single_slice).T, cmap='RdBu_r', origin='lower', vmin=-0.05, vmax=0.05)
ax7.set_title('Reconstruction Difference\n(Two-ch minus Single-ch)', fontsize=11, fontweight='bold')
ax7.axis('off')
cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
cbar7.ax.tick_params(labelsize=9)

# Find region with lesions for zoom
lesion_region = np.where(phantom_slice > 0.3)
if len(lesion_region[0]) > 0:
    x_les = lesion_region[0][0]
    y_les = lesion_region[1][0]
    zoom_size = 15
    x1, x2 = max(0, x_les-zoom_size), min(nx, x_les+zoom_size)
    y1, y2 = max(0, y_les-zoom_size), min(ny, y_les+zoom_size)
else:
    x1, x2 = 45, 75
    y1, y2 = 45, 75

ax8 = plt.subplot(4, 3, 8)
im8 = ax8.imshow(single_slice[x1:x2, y1:y2].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax8.set_title(f'Zoomed Region - Single\n({x1}:{x2}, {y1}:{y2})', fontsize=11, color='red')
ax8.axis('off')

ax9 = plt.subplot(4, 3, 9)
im9 = ax9.imshow(two_slice[x1:x2, y1:y2].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax9.set_title(f'Zoomed Region - Two\n({x1}:{x2}, {y1}:{y2})', fontsize=11, color='blue')
ax9.axis('off')

# === BOTTOM ROW: Line profiles ===
ax10 = plt.subplot(4, 1, 4)
y_center = ny // 2
x_center = nx // 2
x_coords = np.arange(nx)

# Left subplot - horizontal profile
ax10a = plt.subplot(4, 2, 7)
ax10a.plot(x_coords, phantom_slice[:, y_center], 'k-', linewidth=2.5, label='Ground Truth')
ax10a.plot(x_coords, single_slice[:, y_center], 'r-', linewidth=2, label='Single-channel', alpha=0.8)
ax10a.plot(x_coords, two_slice[:, y_center], 'b-', linewidth=2, label='Two-channel', alpha=0.8)
ax10a.set_xlabel('x position (pixels)', fontsize=10)
ax10a.set_ylabel('Attenuation', fontsize=10)
ax10a.set_title(f'Horizontal Profile through Center (y={y_center})', fontsize=11, fontweight='bold')
ax10a.legend(fontsize=9, loc='upper right')
ax10a.grid(True, alpha=0.3)
ax10a.set_xlim([0, nx-1])

# Right subplot - vertical profile  
ax10b = plt.subplot(4, 2, 8)
y_coords = np.arange(ny)
ax10b.plot(y_coords, phantom_slice[x_center, :], 'k-', linewidth=2.5, label='Ground Truth')
ax10b.plot(y_coords, single_slice[x_center, :], 'r-', linewidth=2, label='Single-channel', alpha=0.8)
ax10b.plot(y_coords, two_slice[x_center, :], 'b-', linewidth=2, label='Two-channel', alpha=0.8)
ax10b.set_xlabel('y position (pixels)', fontsize=10)
ax10b.set_ylabel('Attenuation', fontsize=10)
ax10b.set_title(f'Vertical Profile through Center (x={x_center})', fontsize=11, fontweight='bold')
ax10b.legend(fontsize=9, loc='upper right')
ax10b.grid(True, alpha=0.3)
ax10b.set_xlim([0, ny-1])

# Add main title with performance summary
if ierrs_single_all[slice_idx] < ierrs_two_all[slice_idx]:
    title_color = 'darkred'
    title_text = f'Slice 11 Analysis: Single-channel OUTPERFORMS Two-channel by {improvement:.1f}%'
else:
    title_color = 'darkblue'
    title_text = f'Slice 11 Analysis: Two-channel OUTPERFORMS Single-channel by {improvement:.1f}%'

plt.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98, color=title_color)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../results/current/slice_11_detailed_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved detailed visualization to: ../results/current/slice_11_detailed_comparison.png")

# Additional analysis - compute metrics
print(f"\n{'='*60}")
print("DETAILED METRICS FOR SLICE 11")
print(f"{'='*60}")

# MSE and other metrics
mse_single = np.mean((single_slice - phantom_slice)**2)
mse_two = np.mean((two_slice - phantom_slice)**2)
mae_single = np.mean(np.abs(single_slice - phantom_slice))
mae_two = np.mean(np.abs(two_slice - phantom_slice))

print(f"\nMean Squared Error (MSE):")
print(f"  Single-channel: {mse_single:.8f}")
print(f"  Two-channel: {mse_two:.8f}")
print(f"  Ratio: {mse_two/mse_single:.3f}")

print(f"\nMean Absolute Error (MAE):")
print(f"  Single-channel: {mae_single:.6f}")
print(f"  Two-channel: {mae_two:.6f}")
print(f"  Ratio: {mae_two/mae_single:.3f}")

# Check performance across all slices
print(f"\n{'='*60}")
print("PERFORMANCE ACROSS ALL SLICES")
print(f"{'='*60}")

single_better = 0
two_better = 0
for i in range(nz):
    if ierrs_single_all[i] < ierrs_two_all[i]:
        single_better += 1
    else:
        two_better += 1

print(f"Single-channel performs better: {single_better}/{nz} slices ({single_better/nz*100:.1f}%)")
print(f"Two-channel performs better: {two_better}/{nz} slices ({two_better/nz*100:.1f}%)")

# Find slices with largest performance differences
differences = np.array(ierrs_two_all) - np.array(ierrs_single_all)
best_single_idx = np.argmax(differences)  # Most positive = single much better
best_two_idx = np.argmin(differences)     # Most negative = two much better

print(f"\nLargest single-channel advantage: Slice {best_single_idx+1}")
print(f"  Single RMSE: {ierrs_single_all[best_single_idx]:.6f}")
print(f"  Two RMSE: {ierrs_two_all[best_single_idx]:.6f}")
print(f"  Improvement: {differences[best_single_idx]:.6f}")

print(f"\nLargest two-channel advantage: Slice {best_two_idx+1}")
print(f"  Single RMSE: {ierrs_single_all[best_two_idx]:.6f}")
print(f"  Two RMSE: {ierrs_two_all[best_two_idx]:.6f}")
print(f"  Improvement: {-differences[best_two_idx]:.6f}")

plt.show()