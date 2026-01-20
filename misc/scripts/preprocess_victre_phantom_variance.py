"""
Find and extract VICTRE phantom ROI with highest tissue variance
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Read .mhd header
mhd_file = Path('../data/victre_phantom/p_-72912147.mhd')
raw_file = Path('../data/victre_phantom/p_-72912147.raw')
loc_file = Path('../data/victre_phantom/p_-72912147.loc')

print("="*70)
print("FINDING ROI WITH HIGHEST TISSUE VARIANCE")
print("="*70)

# Read lesion locations
lesion_locs = np.loadtxt(loc_file, delimiter=',')
print(f"Total number of lesions: {lesion_locs.shape[0]}")

# Load phantom
dims = (748, 896, 897)  # From MHD: DimSize
spacing = (0.1, 0.1, 0.1)  # ElementSpacing in mm
offset = (-9.45, -53.6013, -45.1684)  # Offset in mm

print(f"\nLoading phantom data...")
print(f"Dimensions: {dims[0]} x {dims[1]} x {dims[2]} voxels")

# Read raw data
phantom = np.fromfile(raw_file, dtype=np.uint8)
phantom = phantom.reshape(dims, order='F')  # Fortran order for medical images
print(f"Value range: [{phantom.min()}, {phantom.max()}]")

# Convert lesion coordinates from mm to voxel indices
lesion_voxels = np.zeros_like(lesion_locs, dtype=int)
for i in range(3):
    lesion_voxels[:, i] = ((lesion_locs[:, i] - offset[i]) / spacing[i]).astype(int)

# ROI size
roi_size_xy = 256
roi_size_z = 20

# Search with stride to make it tractable (check every 20 voxels)
stride = 20

print(f"\n{'-'*70}")
print(f"Searching for best ROI...")
print(f"ROI size: {roi_size_xy}x{roi_size_xy}x{roi_size_z}")
print(f"Search stride: {stride} voxels")
print(f"{'-'*70}")

best_variance = -1
best_position = None
best_lesion_count = 0

# Search grid
x_positions = range(0, dims[0] - roi_size_xy, stride)
y_positions = range(0, dims[1] - roi_size_xy, stride)
z_positions = range(0, dims[2] - roi_size_z, stride)

total_positions = len(x_positions) * len(y_positions) * len(z_positions)
print(f"Total positions to evaluate: {total_positions}")

count = 0
for x_start in x_positions:
    for y_start in y_positions:
        for z_start in z_positions:
            count += 1
            if count % 100 == 0:
                print(f"  Evaluated {count}/{total_positions} positions... (best variance so far: {best_variance:.2f})")

            # Extract ROI
            roi = phantom[x_start:x_start+roi_size_xy,
                         y_start:y_start+roi_size_xy,
                         z_start:z_start+roi_size_z]

            # Compute variance
            variance = np.var(roi.astype(float))

            # Count lesions in this ROI
            lesion_count = 0
            for lx, ly, lz in lesion_voxels:
                if (x_start <= lx < x_start+roi_size_xy and
                    y_start <= ly < y_start+roi_size_xy and
                    z_start <= lz < z_start+roi_size_z):
                    lesion_count += 1

            # Update best (prioritize variance, but require at least 1 lesion)
            if lesion_count >= 1 and variance > best_variance:
                best_variance = variance
                best_position = (x_start, y_start, z_start)
                best_lesion_count = lesion_count

print(f"\n{'-'*70}")
print(f"SEARCH COMPLETE!")
print(f"{'-'*70}")
print(f"Best ROI position: {best_position}")
print(f"Variance: {best_variance:.2f}")
print(f"Lesions in ROI: {best_lesion_count}")

# Extract best ROI
x_start, y_start, z_start = best_position
roi_x_end = x_start + roi_size_xy
roi_y_end = y_start + roi_size_xy
roi_z_end = z_start + roi_size_z

phantom_roi = phantom[x_start:roi_x_end, y_start:roi_y_end, z_start:roi_z_end]

# Find lesions within ROI
lesions_in_roi = []
for i, (lx, ly, lz) in enumerate(lesion_voxels):
    if (x_start <= lx < roi_x_end and
        y_start <= ly < roi_y_end and
        z_start <= lz < roi_z_end):
        # Convert to ROI-relative coordinates
        lesions_in_roi.append([lx - x_start, ly - y_start, lz - z_start])

lesions_in_roi = np.array(lesions_in_roi)

print(f"\nROI bounds:")
print(f"  X: [{x_start}, {roi_x_end}] ({roi_size_xy} voxels)")
print(f"  Y: [{y_start}, {roi_y_end}] ({roi_size_xy} voxels)")
print(f"  Z: [{z_start}, {roi_z_end}] ({roi_size_z} voxels)")
print(f"\nLesion coordinates (ROI-relative):")
print(lesions_in_roi)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Show 3 orthogonal slices through center
z_mid = phantom_roi.shape[2] // 2
y_mid = phantom_roi.shape[1] // 2
x_mid = phantom_roi.shape[0] // 2

# X-Y plane (axial)
axes[0, 0].imshow(phantom_roi[:, :, z_mid].T, cmap='gray', origin='lower')
axes[0, 0].set_title(f'X-Y plane (z={z_mid})')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
# Mark lesions
for lx, ly, lz in lesions_in_roi:
    if abs(lz - z_mid) < 2:  # Within 2 slices
        axes[0, 0].plot(lx, ly, 'r+', markersize=10, markeredgewidth=2)

# X-Z plane (coronal)
axes[0, 1].imshow(phantom_roi[:, y_mid, :].T, cmap='gray', origin='lower', aspect='auto')
axes[0, 1].set_title(f'X-Z plane (y={y_mid})')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('z (depth)')
for lx, ly, lz in lesions_in_roi:
    if abs(ly - y_mid) < 2:
        axes[0, 1].plot(lx, lz, 'r+', markersize=10, markeredgewidth=2)

# Y-Z plane (sagittal)
axes[0, 2].imshow(phantom_roi[x_mid, :, :].T, cmap='gray', origin='lower', aspect='auto')
axes[0, 2].set_title(f'Y-Z plane (x={x_mid})')
axes[0, 2].set_xlabel('y')
axes[0, 2].set_ylabel('z (depth)')
for lx, ly, lz in lesions_in_roi:
    if abs(lx - x_mid) < 2:
        axes[0, 2].plot(ly, lz, 'r+', markersize=10, markeredgewidth=2)

# Show slice with most lesions
if len(lesions_in_roi) > 0:
    # Find z-slice with most lesions
    z_counts = np.bincount(lesions_in_roi[:, 2].astype(int), minlength=phantom_roi.shape[2])
    z_best = np.argmax(z_counts)

    axes[1, 0].imshow(phantom_roi[:, :, z_best].T, cmap='gray', origin='lower')
    axes[1, 0].set_title(f'X-Y plane with most lesions (z={z_best}, {z_counts[z_best]} lesions)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    for lx, ly, lz in lesions_in_roi:
        if lz == z_best:
            axes[1, 0].plot(lx, ly, 'r+', markersize=10, markeredgewidth=2)
            axes[1, 0].text(lx+2, ly+2, f'({lx},{ly})', color='red', fontsize=8)

# Histogram
axes[1, 1].hist(phantom_roi.flatten(), bins=50, edgecolor='black')
axes[1, 1].set_title(f'Intensity histogram (var={best_variance:.2f})')
axes[1, 1].set_xlabel('Intensity')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_yscale('log')

# Lesion depth distribution
if len(lesions_in_roi) > 0:
    axes[1, 2].hist(lesions_in_roi[:, 2], bins=phantom_roi.shape[2], edgecolor='black')
    axes[1, 2].set_title('Lesion distribution in depth')
    axes[1, 2].set_xlabel('z-slice')
    axes[1, 2].set_ylabel('Number of lesions')
    axes[1, 2].axvline(z_mid, color='r', linestyle='--', label='Mid-plane')
    axes[1, 2].legend()

plt.suptitle(f'High-Variance ROI (var={best_variance:.2f}, {best_lesion_count} lesions)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/current/victre_phantom_roi_variance.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization: ../results/current/victre_phantom_roi_variance.png")

# Save ROI and lesion info
np.save('../data/phantoms/victre_phantom_roi.npy', phantom_roi)
np.save('../data/phantoms/victre_lesions_roi.npy', lesions_in_roi)
print(f"\nSaved ROI data:")
print(f"  - ../data/phantoms/victre_phantom_roi.npy: {phantom_roi.shape} array")
print(f"  - ../data/phantoms/victre_lesions_roi.npy: {lesions_in_roi.shape} lesion coordinates")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Extracted ROI: {phantom_roi.shape[0]} x {phantom_roi.shape[1]} x {phantom_roi.shape[2]}")
print(f"Physical size: {phantom_roi.shape[0]*spacing[0]:.1f} x {phantom_roi.shape[1]*spacing[1]:.1f} x {phantom_roi.shape[2]*spacing[2]:.1f} mm")
print(f"Tissue variance: {best_variance:.2f}")
print(f"Lesions in ROI: {len(lesions_in_roi)}")
print(f"\nThis high-variance ROI is ready for DBT reconstruction!")
print("="*70)

plt.show()
