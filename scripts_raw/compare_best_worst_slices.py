"""
Compare slices where each method performs best
For RAW VICTRE phantom reconstruction.

Shows side-by-side comparison of:
1. A slice where single-channel outperforms two-channel
2. A slice where two-channel outperforms single-channel
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os
import sys

# Load results
RESULTS_FILE = 'raw_victre_results.pkl'

if not os.path.exists(RESULTS_FILE):
    print(f"Results file {RESULTS_FILE} not found!")
    print("Please run: python run_reconstruction_comparison.py first")
    sys.exit(1)

print(f"Loading results from {RESULTS_FILE}...")
with open(RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

recon_single_3d = results['single_channel']
recon_two_3d = results['two_channel']
ierrs_single_all = results['single_errors']
ierrs_two_all = results['two_errors']
phantom_3d = results['phantom']
nx, ny, nz = phantom_3d.shape

# Get convergence histories (keyed by slice index)
single_convergence = results.get('single_convergence', {})
two_convergence = results.get('two_convergence', {})

# Get attenuation scale for visualization (default to 0.8 if not present)
ATTENUATION_SCALE = results.get('attenuation_scale', 0.8)

# Find slices where each method performs best
differences = np.array(ierrs_two_all) - np.array(ierrs_single_all)

# Positive difference means single-channel is better (two_err > single_err)
# Negative difference means two-channel is better (two_err < single_err)

best_single_idx = np.argmax(differences)  # Most positive = single much better
best_two_idx = np.argmin(differences)     # Most negative = two much better

# Also find moderately different slices (not just extremes)
sorted_diffs = np.argsort(differences)
moderate_two_idx = sorted_diffs[nz//4]    # 25th percentile - two-channel better
moderate_single_idx = sorted_diffs[3*nz//4]  # 75th percentile - single-channel better

print(f"\n{'='*70}")
print("PERFORMANCE ANALYSIS ACROSS ALL {nz} SLICES")
print(f"{'='*70}")

single_better_count = np.sum(differences > 0)
two_better_count = np.sum(differences < 0)
print(f"Single-channel performs better: {single_better_count}/{nz} slices ({single_better_count/nz*100:.1f}%)")
print(f"Two-channel performs better: {two_better_count}/{nz} slices ({two_better_count/nz*100:.1f}%)")

print(f"\n{'='*70}")
print("EXTREME CASES")
print(f"{'='*70}")
print(f"\nBEST SINGLE-CHANNEL PERFORMANCE: Slice {best_single_idx+1}")
print(f"  Single RMSE: {ierrs_single_all[best_single_idx]:.6f}")
print(f"  Two RMSE: {ierrs_two_all[best_single_idx]:.6f}")
print(f"  Advantage: {differences[best_single_idx]:.6f} ({differences[best_single_idx]/ierrs_two_all[best_single_idx]*100:.1f}% better)")

print(f"\nBEST TWO-CHANNEL PERFORMANCE: Slice {best_two_idx+1}")
print(f"  Single RMSE: {ierrs_single_all[best_two_idx]:.6f}")
print(f"  Two RMSE: {ierrs_two_all[best_two_idx]:.6f}")
print(f"  Advantage: {-differences[best_two_idx]:.6f} ({-differences[best_two_idx]/ierrs_single_all[best_two_idx]*100:.1f}% better)")

print(f"\n{'='*70}")
print("CREATING COMPARISON VISUALIZATIONS")
print(f"{'='*70}")

def create_slice_visualization(slice_idx, fig_num=1):
    """Create detailed visualization for a single slice"""
    
    phantom_slice = phantom_3d[:, :, slice_idx]
    single_slice = recon_single_3d[:, :, slice_idx]
    two_slice = recon_two_3d[:, :, slice_idx]
    
    # Compute difference maps
    diff_single = np.abs(single_slice - phantom_slice)
    diff_two = np.abs(two_slice - phantom_slice)
    
    # Determine which method is better
    if ierrs_single_all[slice_idx] < ierrs_two_all[slice_idx]:
        winner = "Single-channel"
        winner_color = 'darkred'
        improvement = (ierrs_two_all[slice_idx] - ierrs_single_all[slice_idx])/ierrs_two_all[slice_idx]*100
    else:
        winner = "Two-channel"
        winner_color = 'darkblue'
        improvement = (ierrs_single_all[slice_idx] - ierrs_two_all[slice_idx])/ierrs_single_all[slice_idx]*100
    
    # Create figure
    fig = plt.figure(fig_num, figsize=(18, 10))
    fig.clf()

    # Color settings - use full attenuation range for raw phantom
    vmin, vmax = 0, ATTENUATION_SCALE
    diff_vmax = 0.2  # Increased for raw phantom with more variation
    
    # Row 1: Reconstructions
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(phantom_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Ground Truth\nSlice {slice_idx+1}', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(single_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Single-channel L1-DTV\nRMSE={ierrs_single_all[slice_idx]:.5f}', 
                  fontsize=11, fontweight='bold', 
                  color='red' if winner == "Single-channel" else 'black')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(two_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax3.set_title(f'Two-channel L1-DTV\nRMSE={ierrs_two_all[slice_idx]:.5f}', 
                  fontsize=11, fontweight='bold',
                  color='blue' if winner == "Two-channel" else 'black')
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 4, 4)
    im4 = ax4.imshow(two_slice.T - single_slice.T, cmap='RdBu_r', origin='lower', vmin=-0.1, vmax=0.1)
    ax4.set_title('Difference\n(Two - Single)', fontsize=11, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Row 2: Error maps
    ax5 = plt.subplot(3, 4, 5)
    ax5.axis('off')
    ax5.text(0.5, 0.5, f'{winner}\nWINS by {improvement:.1f}%', 
             ha='center', va='center', fontsize=12, fontweight='bold', color=winner_color)
    
    ax6 = plt.subplot(3, 4, 6)
    im6 = ax6.imshow(diff_single.T, cmap='hot', origin='lower', vmin=0, vmax=diff_vmax)
    ax6.set_title('Single-channel Error', fontsize=11, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    ax7 = plt.subplot(3, 4, 7)
    im7 = ax7.imshow(diff_two.T, cmap='hot', origin='lower', vmin=0, vmax=diff_vmax)
    ax7.set_title('Two-channel Error', fontsize=11, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    ax8 = plt.subplot(3, 4, 8)
    im8 = ax8.imshow((diff_two - diff_single).T, cmap='RdBu_r', origin='lower', vmin=-0.1, vmax=0.1)
    ax8.set_title('Error Difference\n(Two - Single)', fontsize=11, fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    # Row 3: Line profiles
    y_center = ny // 2
    x_center = nx // 2
    
    ax9 = plt.subplot(3, 2, 5)
    x_coords = np.arange(nx)
    ax9.plot(x_coords, phantom_slice[:, y_center], 'k-', linewidth=2, label='Ground Truth')
    ax9.plot(x_coords, single_slice[:, y_center], 'r-', linewidth=1.5, label='Single-channel', alpha=0.8)
    ax9.plot(x_coords, two_slice[:, y_center], 'b-', linewidth=1.5, label='Two-channel', alpha=0.8)
    ax9.set_xlabel('x position (pixels)', fontsize=10)
    ax9.set_ylabel('Attenuation', fontsize=10)
    ax9.set_title(f'Horizontal Profile (y={y_center})', fontsize=11, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    
    ax10 = plt.subplot(3, 2, 6)
    y_coords = np.arange(ny)
    ax10.plot(y_coords, phantom_slice[x_center, :], 'k-', linewidth=2, label='Ground Truth')
    ax10.plot(y_coords, single_slice[x_center, :], 'r-', linewidth=1.5, label='Single-channel', alpha=0.8)
    ax10.plot(y_coords, two_slice[x_center, :], 'b-', linewidth=1.5, label='Two-channel', alpha=0.8)
    ax10.set_xlabel('y position (pixels)', fontsize=10)
    ax10.set_ylabel('Attenuation', fontsize=10)
    ax10.set_title(f'Vertical Profile (x={x_center})', fontsize=11, fontweight='bold')
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)
    
    title_text = f'Slice {slice_idx+1}: {winner} outperforms by {improvement:.1f}%'
    plt.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98, color=winner_color)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def create_convergence_visualization(slice_idx, fig_num=1):
    """
    Create 6-subplot visualization with convergence metrics and reconstructions.

    Layout:
    - Top left: Data RMSE vs Iteration
    - Top middle: Image RMSE vs Iteration
    - Top right: Total Variation vs Iteration
    - Bottom left: Single-channel reconstruction
    - Bottom middle: Two-channel reconstruction
    - Bottom right: Ground truth phantom
    """

    # Check if convergence data exists for this slice
    if slice_idx not in single_convergence or slice_idx not in two_convergence:
        print(f"Warning: No convergence data for slice {slice_idx+1}")
        return None

    # Get convergence data for this slice
    single_conv = single_convergence[slice_idx]
    two_conv = two_convergence[slice_idx]

    derrs_single = np.array(single_conv['derrs'])
    derrs_two = np.array(two_conv['derrs'])
    ierrs_single = np.array(single_conv['ierrs'])
    ierrs_two = np.array(two_conv['ierrs'])
    tvs_single = np.array(single_conv['tvs'])
    tvs_two = np.array(two_conv['tvs'])

    # Get reconstruction slices
    phantom_slice = phantom_3d[:, :, slice_idx]
    single_slice = recon_single_3d[:, :, slice_idx]
    two_slice = recon_two_3d[:, :, slice_idx]

    # Compute ground truth TV
    def compute_tv(img):
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:-1, :] = img[1:, :] - img[:-1, :]
        gy[:, :-1] = img[:, 1:] - img[:, :-1]
        gx[-1, :] = -img[-1, :]
        gy[:, -1] = -img[:, -1]
        return np.sqrt(gx**2 + gy**2).sum()

    true_tv = compute_tv(phantom_slice)

    # Create figure
    fig = plt.figure(fig_num, figsize=(16, 10))
    fig.clf()

    # Color settings
    vmin, vmax = 0, ATTENUATION_SCALE

    # ===== Top Row: Convergence Plots =====

    # Plot 1: Data RMSE (Data Fidelity)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(derrs_single, 'r-', label='Single-channel', linewidth=2)
    ax1.plot(derrs_two, 'b-', label='Two-channel', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Data RMSE', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Data Fidelity Convergence', fontsize=13, fontweight='bold')

    # Plot 2: Image RMSE (Image Error)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(ierrs_single, 'r-', label='Single-channel', linewidth=2)
    ax2.plot(ierrs_two, 'b-', label='Two-channel', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Image RMSE', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Image Error Convergence', fontsize=13, fontweight='bold')

    # Plot 3: Total Variation
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(tvs_single, 'r-', label='Single-channel', linewidth=2)
    ax3.plot(tvs_two, 'b-', label='Two-channel', linewidth=2)
    ax3.axhline(y=true_tv, color='g', linestyle='--', label=f'Ground truth ({true_tv:.0f})', linewidth=2)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Total TV', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Total Variation', fontsize=13, fontweight='bold')

    # ===== Bottom Row: Reconstructed Images =====

    # Plot 4: Single-channel reconstruction
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(single_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_title(f'Single-Channel\nRMSE={ierrs_single_all[slice_idx]:.5f}', fontsize=13, fontweight='bold')
    ax4.axis('off')

    # Plot 5: Two-channel reconstruction
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(two_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.set_title(f'Two-Channel\nRMSE={ierrs_two_all[slice_idx]:.5f}', fontsize=13, fontweight='bold')
    ax5.axis('off')

    # Plot 6: Ground truth phantom
    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(phantom_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    ax6.set_title('Ground Truth Phantom', fontsize=13, fontweight='bold')
    ax6.axis('off')

    plt.suptitle(f'Slice {slice_idx+1}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


# Create visualizations for best cases
print("\nGenerating visualization for best single-channel slice...")
fig1 = create_slice_visualization(best_single_idx, fig_num=1)
fig1.savefig('../results/raw_victre/best_single_channel_slice.png', dpi=150, bbox_inches='tight')
print(f"Saved: ../results/raw_victre/best_single_channel_slice.png")

print("\nGenerating visualization for best two-channel slice...")
fig2 = create_slice_visualization(best_two_idx, fig_num=2)
fig2.savefig('../results/raw_victre/best_two_channel_slice.png', dpi=150, bbox_inches='tight')
print(f"Saved: ../results/raw_victre/best_two_channel_slice.png")

# Create NEW convergence visualizations for best slices
print("\nGenerating convergence visualization for best single-channel slice...")
fig_conv_single = create_convergence_visualization(best_single_idx, fig_num=10)
if fig_conv_single:
    fig_conv_single.savefig('../results/raw_victre/convergence_best_single_channel.png', dpi=150, bbox_inches='tight')
    print(f"Saved: ../results/raw_victre/convergence_best_single_channel.png")

print("\nGenerating convergence visualization for best two-channel slice...")
fig_conv_two = create_convergence_visualization(best_two_idx, fig_num=11)
if fig_conv_two:
    fig_conv_two.savefig('../results/raw_victre/convergence_best_two_channel.png', dpi=150, bbox_inches='tight')
    print(f"Saved: ../results/raw_victre/convergence_best_two_channel.png")

# # Also create for moderate cases
# print("\nGenerating visualization for moderate single-channel advantage...")
# fig3 = create_slice_visualization(moderate_single_idx, fig_num=3)
# fig3.savefig('../results/raw_victre/moderate_single_channel_slice.png', dpi=150, bbox_inches='tight')
# print(f"Saved: ../results/raw_victre/moderate_single_channel_slice.png")

# print("\nGenerating visualization for moderate two-channel advantage...")
# fig4 = create_slice_visualization(moderate_two_idx, fig_num=4)
# fig4.savefig('../results/raw_victre/moderate_two_channel_slice.png', dpi=150, bbox_inches='tight')
# print(f"Saved: ../results/raw_victre/moderate_two_channel_slice.png")

# Create a summary comparison plot
print("\nGenerating performance summary across all slices...")
fig5 = plt.figure(5, figsize=(15, 10))
fig5.clf()

# Plot 1: RMSE for all slices
ax1 = plt.subplot(3, 1, 1)
slice_numbers = np.arange(1, nz+1)
ax1.plot(slice_numbers, ierrs_single_all, 'r-', linewidth=2, label='Single-channel', alpha=0.8)
ax1.plot(slice_numbers, ierrs_two_all, 'b-', linewidth=2, label='Two-channel', alpha=0.8)
ax1.set_xlabel('Slice Number', fontsize=12)
ax1.set_ylabel('RMSE', fontsize=12)
ax1.set_title('Reconstruction Error Across All Slices', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axvline(best_single_idx+1, color='red', linestyle='--', alpha=0.5, label='Best single')
ax1.axvline(best_two_idx+1, color='blue', linestyle='--', alpha=0.5, label='Best two')

# Plot 2: Difference (positive = single better, negative = two better)
ax2 = plt.subplot(3, 1, 2)
colors = ['red' if d > 0 else 'blue' for d in differences]
ax2.bar(slice_numbers, differences, color=colors, alpha=0.7)
ax2.set_xlabel('Slice Number', fontsize=12)
ax2.set_ylabel('RMSE Difference\n(Two - Single)', fontsize=12)
ax2.set_title('Performance Difference (Red: Single Better, Blue: Two Better)', fontsize=14, fontweight='bold')
ax2.axhline(0, color='black', linewidth=1)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Histogram of differences
ax3 = plt.subplot(3, 1, 3)
ax3.hist(differences, bins=20, edgecolor='black', alpha=0.7)
ax3.set_xlabel('RMSE Difference (Two - Single)', fontsize=12)
ax3.set_ylabel('Number of Slices', fontsize=12)
ax3.set_title('Distribution of Performance Differences', fontsize=14, fontweight='bold')
ax3.axvline(0, color='black', linewidth=2, linestyle='--', label='Equal performance')
ax3.axvline(np.mean(differences), color='green', linewidth=2, label=f'Mean = {np.mean(differences):.6f}')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig5.savefig('../results/raw_victre/performance_summary_all_slices.png', dpi=150, bbox_inches='tight')
print(f"Saved: ../results/raw_victre/performance_summary_all_slices.png")

print(f"\n{'='*70}")
print("VISUALIZATION COMPLETE!")
print(f"{'='*70}")
print("\nGenerated files (in ../results/raw_victre/):")
print("  1. best_single_channel_slice.png - Slice where single-channel performs best")
print("  2. best_two_channel_slice.png - Slice where two-channel performs best")
print("  3. convergence_best_single_channel.png - Convergence plots for best single-channel slice")
print("  4. convergence_best_two_channel.png - Convergence plots for best two-channel slice")
print("  5. performance_summary_all_slices.png - Overview of all slices")
print("\nSpecific slices analyzed:")
print(f"  - Slice {best_single_idx+1}: Best for single-channel")
print(f"  - Slice {best_two_idx+1}: Best for two-channel")
print(f"  - Slice {moderate_single_idx+1}: Moderate single-channel advantage")
print(f"  - Slice {moderate_two_idx+1}: Moderate two-channel advantage")

plt.show()