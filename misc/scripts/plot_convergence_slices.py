"""
Plot convergence curves for specific slices (best/worst for each method)

Usage:
    python plot_convergence_slices.py                    # Plot best/worst slices automatically
    python plot_convergence_slices.py --slices 5 10 15   # Plot specific slices
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse

# Load results
RESULTS_FILE = 'fig8_victre_results.pkl'

if not os.path.exists(RESULTS_FILE):
    print(f"Results file {RESULTS_FILE} not found!")
    print("Please run: python run_reconstruction_comparison.py first")
    sys.exit(1)

print(f"Loading results from {RESULTS_FILE}...")
with open(RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

# Extract data
ierrs_single_all = results['single_errors']
ierrs_two_all = results['two_errors']
single_conv = results['single_convergence']
two_conv = results['two_convergence']
nz = len(ierrs_single_all)

# Parse arguments
parser = argparse.ArgumentParser(description='Plot convergence for specific slices')
parser.add_argument('--slices', nargs='+', type=int, help='Specific slice indices to plot (0-indexed)')
parser.add_argument('--save-dir', default='../results/current', help='Directory to save figures')
args = parser.parse_args()

# Compute differences
differences = np.array(ierrs_two_all) - np.array(ierrs_single_all)

if args.slices:
    # User specified slices
    slices_to_plot = args.slices
    slice_labels = [f"Slice {s+1}" for s in slices_to_plot]
else:
    # Auto-detect best/worst slices
    best_single_idx = np.argmax(differences)  # Most positive = single much better
    best_two_idx = np.argmin(differences)     # Most negative = two much better

    # Also get median performance slices
    sorted_diffs = np.argsort(differences)
    median_idx = sorted_diffs[nz//2]

    slices_to_plot = [best_single_idx, best_two_idx, median_idx]
    slice_labels = [
        f"Best Single-Channel (Slice {best_single_idx+1})",
        f"Best Two-Channel (Slice {best_two_idx+1})",
        f"Median Performance (Slice {median_idx+1})"
    ]

    print(f"\n{'='*70}")
    print("AUTO-SELECTED SLICES FOR CONVERGENCE PLOTS")
    print(f"{'='*70}")
    print(f"Best single-channel: Slice {best_single_idx+1}")
    print(f"  Single RMSE: {ierrs_single_all[best_single_idx]:.6f}")
    print(f"  Two RMSE: {ierrs_two_all[best_single_idx]:.6f}")
    print(f"  Difference: {differences[best_single_idx]:.6f}")

    print(f"\nBest two-channel: Slice {best_two_idx+1}")
    print(f"  Single RMSE: {ierrs_single_all[best_two_idx]:.6f}")
    print(f"  Two RMSE: {ierrs_two_all[best_two_idx]:.6f}")
    print(f"  Difference: {differences[best_two_idx]:.6f}")

    print(f"\nMedian performance: Slice {median_idx+1}")
    print(f"  Single RMSE: {ierrs_single_all[median_idx]:.6f}")
    print(f"  Two RMSE: {ierrs_two_all[median_idx]:.6f}")
    print(f"  Difference: {differences[median_idx]:.6f}")

# Create convergence plots
print(f"\n{'='*70}")
print("CREATING CONVERGENCE PLOTS")
print(f"{'='*70}")

for slice_idx, label in zip(slices_to_plot, slice_labels):
    if slice_idx not in single_conv or slice_idx not in two_conv:
        print(f"Warning: Convergence data not available for slice {slice_idx+1}, skipping...")
        continue

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot convergence curves
    ax.plot(single_conv[slice_idx], 'r-', linewidth=2.5, label='Single-channel L1-DTV', alpha=0.8)
    ax.plot(two_conv[slice_idx], 'b-', linewidth=2.5, label='Two-channel L1-DTV', alpha=0.8)

    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Image RMSE', fontsize=14)
    ax.set_yscale('log')
    ax.set_title(f'Convergence Comparison: {label}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=13, loc='best')
    ax.grid(True, alpha=0.3, which='both')

    # Add text box with final errors
    textstr = f'Final RMSE:\nSingle: {ierrs_single_all[slice_idx]:.6f}\nTwo: {ierrs_two_all[slice_idx]:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # Save figure
    filename = f'convergence_slice_{slice_idx+1}.png'
    filepath = os.path.join(args.save_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"Saved: {filepath}")

# Create combined plot with all slices
if len(slices_to_plot) > 1:
    fig, axes = plt.subplots(len(slices_to_plot), 1, figsize=(12, 6*len(slices_to_plot)))
    if len(slices_to_plot) == 1:
        axes = [axes]

    for i, (slice_idx, label) in enumerate(zip(slices_to_plot, slice_labels)):
        if slice_idx not in single_conv or slice_idx not in two_conv:
            continue

        ax = axes[i]
        ax.plot(single_conv[slice_idx], 'r-', linewidth=2.5, label='Single-channel L1-DTV', alpha=0.8)
        ax.plot(two_conv[slice_idx], 'b-', linewidth=2.5, label='Two-channel L1-DTV', alpha=0.8)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Image RMSE', fontsize=12)
        ax.set_yscale('log')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, which='both')

        # Add final RMSE values
        textstr = f'Final: S={ierrs_single_all[slice_idx]:.6f}, T={ierrs_two_all[slice_idx]:.6f}'
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    filepath = os.path.join(args.save_dir, 'convergence_comparison_combined.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"Saved: {filepath}")

print(f"\n{'='*70}")
print("CONVERGENCE PLOTS COMPLETE!")
print(f"{'='*70}")

plt.show()
