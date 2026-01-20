"""
Compare two-channel L1-DTV performance before and after adding gamma decay.

This script compares:
1. Two-channel WITHOUT gamma (fixed sig_lo and eps_lo)
2. Two-channel WITH gamma (decaying sig_lo and eps_lo)

Similar to compare_best_worst_slices.py but comparing the effect of gamma decay.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os
import sys

# Load results
RESULTS_WITH_GAMMA = 'raw_victre_results.pkl'
RESULTS_NO_GAMMA = 'raw_victre_results_no_gamma.pkl'

if not os.path.exists(RESULTS_WITH_GAMMA):
    print(f"Results file {RESULTS_WITH_GAMMA} not found!")
    print("Please run: python run_reconstruction_comparison_decay.py first")
    sys.exit(1)

if not os.path.exists(RESULTS_NO_GAMMA):
    print(f"Results file {RESULTS_NO_GAMMA} not found!")
    print("This should contain results from before gamma decay was added.")
    sys.exit(1)

print(f"Loading results WITH gamma from {RESULTS_WITH_GAMMA}...")
with open(RESULTS_WITH_GAMMA, 'rb') as f:
    results_gamma = pickle.load(f)

print(f"Loading results WITHOUT gamma from {RESULTS_NO_GAMMA}...")
with open(RESULTS_NO_GAMMA, 'rb') as f:
    results_no_gamma = pickle.load(f)

# Extract data
recon_two_gamma = results_gamma['two_channel']
recon_two_no_gamma = results_no_gamma['two_channel']
ierrs_two_gamma = results_gamma['two_errors']
ierrs_two_no_gamma = results_no_gamma['two_errors']
phantom_3d = results_gamma['phantom']
nx, ny, nz = phantom_3d.shape

# Get convergence histories (keyed by slice index)
two_conv_gamma = results_gamma.get('two_convergence', {})
two_conv_no_gamma = results_no_gamma.get('two_convergence', {})

# Get attenuation scale for visualization
ATTENUATION_SCALE = results_gamma.get('attenuation_scale', 0.8)

# Find slices where each method performs best
differences = np.array(ierrs_two_gamma) - np.array(ierrs_two_no_gamma)

# Positive difference means no-gamma is better (gamma has higher error)
# Negative difference means gamma is better (gamma has lower error)

best_no_gamma_idx = np.argmax(differences)  # Most positive = no-gamma much better
best_gamma_idx = np.argmin(differences)     # Most negative = gamma much better

print(f"\n{'='*70}")
print(f"GAMMA DECAY EFFECT ANALYSIS ACROSS ALL {nz} SLICES")
print(f"{'='*70}")

no_gamma_better_count = np.sum(differences > 0)
gamma_better_count = np.sum(differences < 0)
ties = np.sum(differences == 0)

print(f"Two-channel WITHOUT gamma better: {no_gamma_better_count}/{nz} slices ({no_gamma_better_count/nz*100:.1f}%)")
print(f"Two-channel WITH gamma better: {gamma_better_count}/{nz} slices ({gamma_better_count/nz*100:.1f}%)")
if ties > 0:
    print(f"Ties: {ties}/{nz} slices")

avg_rmse_no_gamma = np.mean(ierrs_two_no_gamma)
avg_rmse_gamma = np.mean(ierrs_two_gamma)
print(f"\nAverage RMSE without gamma: {avg_rmse_no_gamma:.6f}")
print(f"Average RMSE with gamma: {avg_rmse_gamma:.6f}")
print(f"Average improvement with gamma: {(avg_rmse_no_gamma - avg_rmse_gamma):.6f} ({(avg_rmse_no_gamma - avg_rmse_gamma)/avg_rmse_no_gamma*100:.2f}%)")

print(f"\n{'='*70}")
print("EXTREME CASES")
print(f"{'='*70}")
print(f"\nBEST WITHOUT GAMMA: Slice {best_no_gamma_idx+1}")
print(f"  No-gamma RMSE: {ierrs_two_no_gamma[best_no_gamma_idx]:.6f}")
print(f"  With-gamma RMSE: {ierrs_two_gamma[best_no_gamma_idx]:.6f}")
print(f"  Advantage: {differences[best_no_gamma_idx]:.6f}")

print(f"\nBEST WITH GAMMA: Slice {best_gamma_idx+1}")
print(f"  No-gamma RMSE: {ierrs_two_no_gamma[best_gamma_idx]:.6f}")
print(f"  With-gamma RMSE: {ierrs_two_gamma[best_gamma_idx]:.6f}")
print(f"  Advantage: {-differences[best_gamma_idx]:.6f}")

print(f"\n{'='*70}")
print("CREATING COMPARISON VISUALIZATIONS")
print(f"{'='*70}")


def create_slice_visualization(slice_idx, fig_num=1):
    """Create detailed visualization for a single slice comparing gamma effect"""

    phantom_slice = phantom_3d[:, :, slice_idx]
    no_gamma_slice = recon_two_no_gamma[:, :, slice_idx]
    gamma_slice = recon_two_gamma[:, :, slice_idx]

    # Compute difference maps
    diff_no_gamma = np.abs(no_gamma_slice - phantom_slice)
    diff_gamma = np.abs(gamma_slice - phantom_slice)

    # Determine which method is better
    if ierrs_two_no_gamma[slice_idx] < ierrs_two_gamma[slice_idx]:
        winner = "No-gamma"
        winner_color = 'darkred'
        improvement = (ierrs_two_gamma[slice_idx] - ierrs_two_no_gamma[slice_idx])/ierrs_two_gamma[slice_idx]*100
    else:
        winner = "With-gamma"
        winner_color = 'darkgreen'
        improvement = (ierrs_two_no_gamma[slice_idx] - ierrs_two_gamma[slice_idx])/ierrs_two_no_gamma[slice_idx]*100

    # Create figure
    fig = plt.figure(fig_num, figsize=(18, 10))
    fig.clf()

    # Color settings
    vmin, vmax = 0, ATTENUATION_SCALE
    diff_vmax = 0.2

    # Row 1: Reconstructions
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(phantom_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Ground Truth\nSlice {slice_idx+1}', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(no_gamma_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Two-channel (No Gamma)\nRMSE={ierrs_two_no_gamma[slice_idx]:.5f}',
                  fontsize=11, fontweight='bold',
                  color='red' if winner == "No-gamma" else 'black')
    ax2.axis('off')

    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(gamma_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax3.set_title(f'Two-channel (With Gamma)\nRMSE={ierrs_two_gamma[slice_idx]:.5f}',
                  fontsize=11, fontweight='bold',
                  color='green' if winner == "With-gamma" else 'black')
    ax3.axis('off')

    ax4 = plt.subplot(3, 4, 4)
    im4 = ax4.imshow(gamma_slice.T - no_gamma_slice.T, cmap='RdBu_r', origin='lower', vmin=-0.1, vmax=0.1)
    ax4.set_title('Difference\n(Gamma - No Gamma)', fontsize=11, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Row 2: Error maps
    ax5 = plt.subplot(3, 4, 5)
    ax5.axis('off')
    ax5.text(0.5, 0.5, f'{winner}\nWINS by {improvement:.1f}%',
             ha='center', va='center', fontsize=12, fontweight='bold', color=winner_color)

    ax6 = plt.subplot(3, 4, 6)
    im6 = ax6.imshow(diff_no_gamma.T, cmap='hot', origin='lower', vmin=0, vmax=diff_vmax)
    ax6.set_title('No-gamma Error', fontsize=11, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    ax7 = plt.subplot(3, 4, 7)
    im7 = ax7.imshow(diff_gamma.T, cmap='hot', origin='lower', vmin=0, vmax=diff_vmax)
    ax7.set_title('With-gamma Error', fontsize=11, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    ax8 = plt.subplot(3, 4, 8)
    im8 = ax8.imshow((diff_gamma - diff_no_gamma).T, cmap='RdBu_r', origin='lower', vmin=-0.1, vmax=0.1)
    ax8.set_title('Error Difference\n(Gamma - No Gamma)', fontsize=11, fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    # Row 3: Line profiles
    y_center = ny // 2
    x_center = nx // 2

    ax9 = plt.subplot(3, 2, 5)
    x_coords = np.arange(nx)
    ax9.plot(x_coords, phantom_slice[:, y_center], 'k-', linewidth=2, label='Ground Truth')
    ax9.plot(x_coords, no_gamma_slice[:, y_center], 'r-', linewidth=1.5, label='No Gamma', alpha=0.8)
    ax9.plot(x_coords, gamma_slice[:, y_center], 'g-', linewidth=1.5, label='With Gamma', alpha=0.8)
    ax9.set_xlabel('x position (pixels)', fontsize=10)
    ax9.set_ylabel('Attenuation', fontsize=10)
    ax9.set_title(f'Horizontal Profile (y={y_center})', fontsize=11, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)

    ax10 = plt.subplot(3, 2, 6)
    y_coords = np.arange(ny)
    ax10.plot(y_coords, phantom_slice[x_center, :], 'k-', linewidth=2, label='Ground Truth')
    ax10.plot(y_coords, no_gamma_slice[x_center, :], 'r-', linewidth=1.5, label='No Gamma', alpha=0.8)
    ax10.plot(y_coords, gamma_slice[x_center, :], 'g-', linewidth=1.5, label='With Gamma', alpha=0.8)
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
    Create 6-subplot visualization comparing convergence with and without gamma.
    """

    # Check if convergence data exists for this slice
    if slice_idx not in two_conv_gamma or slice_idx not in two_conv_no_gamma:
        print(f"Warning: No convergence data for slice {slice_idx+1}")
        return None

    # Get convergence data for this slice
    conv_gamma = two_conv_gamma[slice_idx]
    conv_no_gamma = two_conv_no_gamma[slice_idx]

    # Handle different data formats (old format is just a list, new format is dict)
    if isinstance(conv_gamma, dict):
        ierrs_gamma = np.array(conv_gamma['ierrs'])
        derrs_gamma = np.array(conv_gamma.get('derrs', []))
        tvs_gamma = np.array(conv_gamma.get('tvs', []))
    else:
        # Old format - just a list of image errors
        ierrs_gamma = np.array(conv_gamma)
        derrs_gamma = np.array([])
        tvs_gamma = np.array([])

    if isinstance(conv_no_gamma, dict):
        ierrs_no_gamma = np.array(conv_no_gamma['ierrs'])
        derrs_no_gamma = np.array(conv_no_gamma.get('derrs', []))
        tvs_no_gamma = np.array(conv_no_gamma.get('tvs', []))
    else:
        # Old format - just a list of image errors
        ierrs_no_gamma = np.array(conv_no_gamma)
        derrs_no_gamma = np.array([])
        tvs_no_gamma = np.array([])

    # Get reconstruction slices
    phantom_slice = phantom_3d[:, :, slice_idx]
    no_gamma_slice = recon_two_no_gamma[:, :, slice_idx]
    gamma_slice = recon_two_gamma[:, :, slice_idx]

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
    if len(derrs_no_gamma) > 0 and len(derrs_gamma) > 0:
        ax1.plot(derrs_no_gamma, 'r-', label='No Gamma', linewidth=2)
        ax1.plot(derrs_gamma, 'g-', label='With Gamma', linewidth=2)
        ax1.set_yscale('log')
        ax1.legend(fontsize=11)
    else:
        ax1.text(0.5, 0.5, 'Data RMSE not available\nin old format',
                 ha='center', va='center', fontsize=11, transform=ax1.transAxes)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Data RMSE', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Data Fidelity Convergence', fontsize=13, fontweight='bold')

    # Plot 2: Image RMSE (Image Error)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(ierrs_no_gamma, 'r-', label='No Gamma', linewidth=2)
    ax2.plot(ierrs_gamma, 'g-', label='With Gamma', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Image RMSE', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Image Error Convergence', fontsize=13, fontweight='bold')

    # Plot 3: Total Variation
    ax3 = plt.subplot(2, 3, 3)
    if len(tvs_no_gamma) > 0 and len(tvs_gamma) > 0:
        ax3.plot(tvs_no_gamma, 'r-', label='No Gamma', linewidth=2)
        ax3.plot(tvs_gamma, 'g-', label='With Gamma', linewidth=2)
        ax3.axhline(y=true_tv, color='k', linestyle='--', label=f'Ground truth ({true_tv:.0f})', linewidth=2)
        ax3.legend(fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'TV data not available\nin old format',
                 ha='center', va='center', fontsize=11, transform=ax3.transAxes)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Total TV', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Total Variation', fontsize=13, fontweight='bold')

    # ===== Bottom Row: Reconstructed Images =====

    # Plot 4: No-gamma reconstruction
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(no_gamma_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_title(f'Two-Channel (No Gamma)\nRMSE={ierrs_two_no_gamma[slice_idx]:.5f}', fontsize=13, fontweight='bold')
    ax4.axis('off')

    # Plot 5: With-gamma reconstruction
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(gamma_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.set_title(f'Two-Channel (With Gamma)\nRMSE={ierrs_two_gamma[slice_idx]:.5f}', fontsize=13, fontweight='bold')
    ax5.axis('off')

    # Plot 6: Ground truth phantom
    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(phantom_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    ax6.set_title('Ground Truth Phantom', fontsize=13, fontweight='bold')
    ax6.axis('off')

    plt.suptitle(f'Gamma Decay Effect - Slice {slice_idx+1}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


# Create output directory
output_dir = Path('../results/raw_victre')
output_dir.mkdir(parents=True, exist_ok=True)

# Create visualizations for best cases
print("\nGenerating visualization for best no-gamma slice...")
fig1 = create_slice_visualization(best_no_gamma_idx, fig_num=1)
fig1.savefig(output_dir / 'gamma_comparison_best_no_gamma.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'gamma_comparison_best_no_gamma.png'}")

print("\nGenerating visualization for best with-gamma slice...")
fig2 = create_slice_visualization(best_gamma_idx, fig_num=2)
fig2.savefig(output_dir / 'gamma_comparison_best_gamma.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'gamma_comparison_best_gamma.png'}")

# Create convergence visualizations
print("\nGenerating convergence visualization for best no-gamma slice...")
fig_conv_no_gamma = create_convergence_visualization(best_no_gamma_idx, fig_num=10)
if fig_conv_no_gamma:
    fig_conv_no_gamma.savefig(output_dir / 'gamma_convergence_best_no_gamma.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gamma_convergence_best_no_gamma.png'}")

print("\nGenerating convergence visualization for best with-gamma slice...")
fig_conv_gamma = create_convergence_visualization(best_gamma_idx, fig_num=11)
if fig_conv_gamma:
    fig_conv_gamma.savefig(output_dir / 'gamma_convergence_best_gamma.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gamma_convergence_best_gamma.png'}")

# Create performance summary plot
print("\nGenerating performance summary across all slices...")
fig5 = plt.figure(5, figsize=(15, 10))
fig5.clf()

# Plot 1: RMSE for all slices
ax1 = plt.subplot(3, 1, 1)
slice_numbers = np.arange(1, nz+1)
ax1.plot(slice_numbers, ierrs_two_no_gamma, 'r-', linewidth=2, label='Two-channel (No Gamma)', alpha=0.8)
ax1.plot(slice_numbers, ierrs_two_gamma, 'g-', linewidth=2, label='Two-channel (With Gamma)', alpha=0.8)
ax1.set_xlabel('Slice Number', fontsize=12)
ax1.set_ylabel('RMSE', fontsize=12)
ax1.set_title('Reconstruction Error Across All Slices', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axvline(best_no_gamma_idx+1, color='red', linestyle='--', alpha=0.5)
ax1.axvline(best_gamma_idx+1, color='green', linestyle='--', alpha=0.5)

# Plot 2: Difference (positive = no-gamma better, negative = gamma better)
ax2 = plt.subplot(3, 1, 2)
colors = ['red' if d > 0 else 'green' for d in differences]
ax2.bar(slice_numbers, differences, color=colors, alpha=0.7)
ax2.set_xlabel('Slice Number', fontsize=12)
ax2.set_ylabel('RMSE Difference\n(Gamma - No Gamma)', fontsize=12)
ax2.set_title('Performance Difference (Red: No-gamma Better, Green: Gamma Better)', fontsize=14, fontweight='bold')
ax2.axhline(0, color='black', linewidth=1)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Histogram of differences
ax3 = plt.subplot(3, 1, 3)
import builtins
n_bins = builtins.max(5, builtins.min(20, nz//2))
ax3.hist(differences, bins=n_bins, edgecolor='black', alpha=0.7, color='gray')
ax3.set_xlabel('RMSE Difference (Gamma - No Gamma)', fontsize=12)
ax3.set_ylabel('Number of Slices', fontsize=12)
ax3.set_title('Distribution of Performance Differences', fontsize=14, fontweight='bold')
ax3.axvline(0, color='black', linewidth=2, linestyle='--', label='Equal performance')
ax3.axvline(np.mean(differences), color='purple', linewidth=2, label=f'Mean = {np.mean(differences):.6f}')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig5.savefig(output_dir / 'gamma_comparison_summary.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'gamma_comparison_summary.png'}")

print(f"\n{'='*70}")
print("GAMMA COMPARISON COMPLETE!")
print(f"{'='*70}")
print("\nGenerated files (in ../results/raw_victre/):")
print("  1. gamma_comparison_best_no_gamma.png - Slice where no-gamma performs best")
print("  2. gamma_comparison_best_gamma.png - Slice where gamma performs best")
print("  3. gamma_convergence_best_no_gamma.png - Convergence for best no-gamma slice")
print("  4. gamma_convergence_best_gamma.png - Convergence for best gamma slice")
print("  5. gamma_comparison_summary.png - Overview of all slices")
print(f"\nSpecific slices analyzed:")
print(f"  - Slice {best_no_gamma_idx+1}: Best for no-gamma")
print(f"  - Slice {best_gamma_idx+1}: Best for with-gamma")

plt.show()
