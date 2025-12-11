"""
Analyze convergence behavior of single-channel vs two-channel methods.
Plots Data RMSE, Image RMSE, and TV to understand semi-convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Try to load results from raw_victre
results_file = Path('raw_victre_results.pkl')

if not results_file.exists():
    print(f"Results file {results_file} not found!")
    print("Please run: python run_reconstruction_comparison.py first")
    exit(1)

print(f"Loading results from {results_file}...")
with open(results_file, 'rb') as f:
    results = pickle.load(f)

# Extract convergence histories (keyed by slice index, each with ierrs/derrs/tvs)
single_conv_all = results.get('single_convergence', {})
two_conv_all = results.get('two_convergence', {})
high_var_slice = results.get('high_variance_slice', 0)

if not single_conv_all or high_var_slice not in single_conv_all:
    print("\nNo full convergence history found!")
    print("Please re-run run_reconstruction_comparison.py to generate new results with all metrics.")
    exit(1)

# Get convergence for high variance slice
single_conv = single_conv_all[high_var_slice]
two_conv = two_conv_all[high_var_slice]

if 'ierrs' not in single_conv:
    print("\nConvergence data missing 'ierrs' key!")
    print("Please re-run run_reconstruction_comparison.py to generate new results.")
    exit(1)

print(f"Analyzing convergence for high variance slice {high_var_slice + 1}")

# Extract all three metrics
ierrs_single = np.array(single_conv['ierrs'])
derrs_single = np.array(single_conv['derrs'])
tvs_single = np.array(single_conv['tvs'])

ierrs_two = np.array(two_conv['ierrs'])
derrs_two = np.array(two_conv['derrs'])
tvs_two = np.array(two_conv['tvs'])

iterations = np.arange(1, len(ierrs_single) + 1)

# Compute ground truth TV from phantom
phantom = results['phantom']
def compute_tv(img):
    """Compute Total Variation of an image"""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:-1, :] = img[1:, :] - img[:-1, :]
    gy[:, :-1] = img[:, 1:] - img[:, :-1]
    gx[-1, :] = -img[-1, :]
    gy[:, -1] = -img[:, -1]
    return np.sqrt(gx**2 + gy**2).sum()

true_tv = compute_tv(phantom[:, :, high_var_slice])
print(f"Ground truth TV: {true_tv:.2f}")

# Create comprehensive figure
fig = plt.figure(figsize=(16, 12))

# ===== Row 1: All three metrics =====

# Plot 1: Image RMSE
ax1 = plt.subplot(2, 3, 1)
ax1.plot(iterations, ierrs_single, 'r-', label='Single-channel', linewidth=2, alpha=0.8)
ax1.plot(iterations, ierrs_two, 'b-', label='Two-channel', linewidth=2, alpha=0.8)
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Image RMSE', fontsize=11)
ax1.set_title('Image RMSE (Reconstruction Accuracy)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Mark minimum points
min_single = np.argmin(ierrs_single)
min_two = np.argmin(ierrs_two)
ax1.scatter([min_single + 1], [ierrs_single[min_single]], color='red', s=100, zorder=5, marker='*')
ax1.scatter([min_two + 1], [ierrs_two[min_two]], color='blue', s=100, zorder=5, marker='*')
ax1.axvline(min_single + 1, color='red', linestyle='--', alpha=0.3)
ax1.axvline(min_two + 1, color='blue', linestyle='--', alpha=0.3)

# Plot 2: Data RMSE
ax2 = plt.subplot(2, 3, 2)
ax2.plot(iterations, derrs_single, 'r-', label='Single-channel', linewidth=2, alpha=0.8)
ax2.plot(iterations, derrs_two, 'b-', label='Two-channel', linewidth=2, alpha=0.8)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Data RMSE', fontsize=11)
ax2.set_yscale('log')
ax2.set_title('Data RMSE (Data Fidelity)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Total Variation
ax3 = plt.subplot(2, 3, 3)
ax3.plot(iterations, tvs_single, 'r-', label='Single-channel', linewidth=2, alpha=0.8)
ax3.plot(iterations, tvs_two, 'b-', label='Two-channel', linewidth=2, alpha=0.8)
ax3.axhline(true_tv, color='green', linestyle='--', linewidth=2, label=f'Ground Truth ({true_tv:.0f})')
ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_ylabel('Total Variation', fontsize=11)
ax3.set_title('Total Variation (Edge Sparsity)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# ===== Row 2: Detailed analysis =====

# Plot 4: Image RMSE (log scale, zoomed)
ax4 = plt.subplot(2, 3, 4)
ax4.plot(iterations, ierrs_single, 'r-', label='Single-channel', linewidth=2, alpha=0.8)
ax4.plot(iterations, ierrs_two, 'b-', label='Two-channel', linewidth=2, alpha=0.8)
ax4.set_xlabel('Iteration', fontsize=11)
ax4.set_ylabel('Image RMSE (log)', fontsize=11)
ax4.set_yscale('log')
ax4.set_title('Image RMSE (Log Scale)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, which='both')

# Plot 5: Image RMSE vs Data RMSE (parametric plot)
ax5 = plt.subplot(2, 3, 5)
ax5.plot(derrs_single, ierrs_single, 'r-', label='Single-channel', linewidth=2, alpha=0.8)
ax5.plot(derrs_two, ierrs_two, 'b-', label='Two-channel', linewidth=2, alpha=0.8)
# Mark start and end points
ax5.scatter([derrs_single[0]], [ierrs_single[0]], color='red', s=80, marker='o', zorder=5)
ax5.scatter([derrs_single[-1]], [ierrs_single[-1]], color='red', s=80, marker='x', zorder=5)
ax5.scatter([derrs_two[0]], [ierrs_two[0]], color='blue', s=80, marker='o', zorder=5)
ax5.scatter([derrs_two[-1]], [ierrs_two[-1]], color='blue', s=80, marker='x', zorder=5)
# Mark minimum image RMSE
ax5.scatter([derrs_single[min_single]], [ierrs_single[min_single]], color='red', s=150, marker='*', zorder=6)
ax5.scatter([derrs_two[min_two]], [ierrs_two[min_two]], color='blue', s=150, marker='*', zorder=6)
ax5.set_xlabel('Data RMSE', fontsize=11)
ax5.set_ylabel('Image RMSE', fontsize=11)
ax5.set_xscale('log')
ax5.set_title('Image vs Data Error Trade-off\n(o=start, x=end, *=min img err)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, which='both')

# Plot 6: TV vs Image RMSE (parametric plot)
ax6 = plt.subplot(2, 3, 6)
ax6.plot(tvs_single, ierrs_single, 'r-', label='Single-channel', linewidth=2, alpha=0.8)
ax6.plot(tvs_two, ierrs_two, 'b-', label='Two-channel', linewidth=2, alpha=0.8)
ax6.axvline(true_tv, color='green', linestyle='--', linewidth=2, alpha=0.5, label='True TV')
# Mark minimum image RMSE
ax6.scatter([tvs_single[min_single]], [ierrs_single[min_single]], color='red', s=150, marker='*', zorder=6)
ax6.scatter([tvs_two[min_two]], [ierrs_two[min_two]], color='blue', s=150, marker='*', zorder=6)
ax6.set_xlabel('Total Variation', fontsize=11)
ax6.set_ylabel('Image RMSE', fontsize=11)
ax6.set_title('Image Error vs TV\n(*=min img err)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.suptitle(f'Convergence Analysis - Slice {high_var_slice+1}', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

save_path = Path('../results/raw_victre')
save_path.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path / 'convergence_all_metrics.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path / 'convergence_all_metrics.png'}")

# ===== Print Summary Statistics =====
print("\n" + "="*70)
print("CONVERGENCE ANALYSIS SUMMARY")
print("="*70)

print(f"\n--- Image RMSE (lower = better reconstruction) ---")
print(f"Single-channel:")
print(f"  Minimum: {ierrs_single[min_single]:.6f} at iteration {min_single + 1}")
print(f"  Final:   {ierrs_single[-1]:.6f} at iteration {len(ierrs_single)}")
print(f"  Increase after min: {(ierrs_single[-1] - ierrs_single[min_single]) / ierrs_single[min_single] * 100:.1f}%")
print(f"Two-channel:")
print(f"  Minimum: {ierrs_two[min_two]:.6f} at iteration {min_two + 1}")
print(f"  Final:   {ierrs_two[-1]:.6f} at iteration {len(ierrs_two)}")
print(f"  Increase after min: {(ierrs_two[-1] - ierrs_two[min_two]) / ierrs_two[min_two] * 100:.1f}%")

print(f"\n--- Data RMSE (lower = better data fit) ---")
print(f"Single-channel: {derrs_single[0]:.6f} -> {derrs_single[-1]:.6f}")
print(f"Two-channel:    {derrs_two[0]:.6f} -> {derrs_two[-1]:.6f}")

print(f"\n--- Total Variation (should approach {true_tv:.0f}) ---")
print(f"Single-channel: {tvs_single[0]:.0f} -> {tvs_single[-1]:.0f} (final error: {abs(tvs_single[-1] - true_tv):.0f})")
print(f"Two-channel:    {tvs_two[0]:.0f} -> {tvs_two[-1]:.0f} (final error: {abs(tvs_two[-1] - true_tv):.0f})")

print(f"\n--- Optimal Stopping (at minimum Image RMSE) ---")
print(f"Single-channel: stop at iteration {min_single + 1}")
print(f"Two-channel:    stop at iteration {min_two + 1}")

# Compare at optimal stopping
print(f"\n--- Comparison at Optimal Stopping ---")
if ierrs_two[min_two] < ierrs_single[min_single]:
    improvement = (ierrs_single[min_single] - ierrs_two[min_two]) / ierrs_single[min_single] * 100
    print(f"Two-channel is {improvement:.2f}% better at optimal stopping")
else:
    degradation = (ierrs_two[min_two] - ierrs_single[min_single]) / ierrs_single[min_single] * 100
    print(f"Single-channel is {degradation:.2f}% better at optimal stopping")

# Compare at final iteration
print(f"\n--- Comparison at Final Iteration ---")
if ierrs_two[-1] < ierrs_single[-1]:
    improvement = (ierrs_single[-1] - ierrs_two[-1]) / ierrs_single[-1] * 100
    print(f"Two-channel is {improvement:.2f}% better at final iteration")
else:
    degradation = (ierrs_two[-1] - ierrs_single[-1]) / ierrs_single[-1] * 100
    print(f"Single-channel is {degradation:.2f}% better at final iteration")

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print("""
1. SEMI-CONVERGENCE: Image RMSE decreases initially, then increases.
   - This is because the algorithm starts fitting noise/artifacts
   - The optimal iteration is NOT the final iteration!

2. DATA RMSE keeps decreasing (algorithm always improves data fit).
   - But better data fit doesn't mean better reconstruction
   - Limited-angle geometry has non-unique solutions

3. TV typically increases then stabilizes.
   - The algorithm adds more edges over time
   - Ground truth TV is a reference, not necessarily optimal
""")

plt.show()
