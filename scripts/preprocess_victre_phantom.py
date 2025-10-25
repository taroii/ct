"""
Load VICTRE phantom and extract ROI containing lesions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Read .mhd header
mhd_file = Path('victre_phantom/p_-72912147.mhd')
raw_file = Path('victre_phantom/p_-72912147.raw')
loc_file = Path('victre_phantom/p_-72912147.loc')

# Parse MHD header
print("="*70)
print("VICTRE PHANTOM INFO")
print("="*70)

with open(mhd_file, 'r') as f:
    for line in f:
        print(line.strip())

# Read lesion locations
print("\n" + "="*70)
print("LESION LOCATIONS (first 10)")
print("="*70)

lesion_locs = np.loadtxt(loc_file, delimiter=',')
print(f"Total number of lesions: {lesion_locs.shape[0]}")
print(f"\nFirst 10 lesions (x, y, z in mm):")
print(lesion_locs[:10])

# Load phantom
dims = (748, 896, 897)  # From MHD: DimSize
spacing = (0.1, 0.1, 0.1)  # ElementSpacing in mm
offset = (-9.45, -53.6013, -45.1684)  # Offset in mm

print(f"\n" + "="*70)
print("LOADING PHANTOM DATA")
print("="*70)
print(f"Dimensions: {dims[0]} x {dims[1]} x {dims[2]} voxels")
print(f"Voxel spacing: {spacing[0]} mm isotropic")
print(f"Physical size: {dims[0]*spacing[0]:.1f} x {dims[1]*spacing[1]:.1f} x {dims[2]*spacing[2]:.1f} mm")

# Read raw data
phantom = np.fromfile(raw_file, dtype=np.uint8)
phantom = phantom.reshape(dims, order='F')  # Fortran order for medical images

print(f"Phantom data type: {phantom.dtype}")
print(f"Value range: [{phantom.min()}, {phantom.max()}]")

# Convert lesion coordinates from mm to voxel indices
lesion_voxels = np.zeros_like(lesion_locs, dtype=int)
for i in range(3):
    lesion_voxels[:, i] = ((lesion_locs[:, i] - offset[i]) / spacing[i]).astype(int)

print(f"\n" + "="*70)
print("LESION VOXEL COORDINATES")
print("="*70)
print(f"First 10 lesions (ix, iy, iz in voxels):")
print(lesion_voxels[:10])

# Find bounding box containing all lesions
x_min, y_min, z_min = lesion_voxels.min(axis=0)
x_max, y_max, z_max = lesion_voxels.max(axis=0)

print(f"\nLesion bounding box:")
print(f"  X: [{x_min}, {x_max}] (range: {x_max-x_min} voxels)")
print(f"  Y: [{y_min}, {y_max}] (range: {y_max-y_min} voxels)")
print(f"  Z: [{z_min}, {z_max}] (range: {z_max-z_min} voxels)")

# Extract a manageable ROI
# Target: ~256x256x20 centered on lesions
roi_size_xy = 256
roi_size_z = 20

# Center on lesions
x_center = (x_min + x_max) // 2
y_center = (y_min + y_max) // 2
z_center = (z_min + z_max) // 2

# Define ROI bounds
roi_x_start = max(0, x_center - roi_size_xy//2)
roi_x_end = min(dims[0], roi_x_start + roi_size_xy)
roi_y_start = max(0, y_center - roi_size_xy//2)
roi_y_end = min(dims[1], roi_y_start + roi_size_xy)
roi_z_start = max(0, z_center - roi_size_z//2)
roi_z_end = min(dims[2], roi_z_start + roi_size_z)

print(f"\n" + "="*70)
print("EXTRACTED ROI")
print("="*70)
print(f"ROI bounds:")
print(f"  X: [{roi_x_start}, {roi_x_end}] ({roi_x_end-roi_x_start} voxels)")
print(f"  Y: [{roi_y_start}, {roi_y_end}] ({roi_y_end-roi_y_start} voxels)")
print(f"  Z: [{roi_z_start}, {roi_z_end}] ({roi_z_end-roi_z_start} voxels)")

# Extract ROI
phantom_roi = phantom[roi_x_start:roi_x_end, roi_y_start:roi_y_end, roi_z_start:roi_z_end]

# Find lesions within ROI
lesions_in_roi = []
for i, (lx, ly, lz) in enumerate(lesion_voxels):
    if (roi_x_start <= lx < roi_x_end and
        roi_y_start <= ly < roi_y_end and
        roi_z_start <= lz < roi_z_end):
        # Convert to ROI-relative coordinates
        lesions_in_roi.append([lx - roi_x_start, ly - roi_y_start, lz - roi_z_start])

lesions_in_roi = np.array(lesions_in_roi)
print(f"\nNumber of lesions in ROI: {len(lesions_in_roi)}")
print(f"Lesion coordinates (ROI-relative):")
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
axes[1, 1].set_title('Intensity histogram')
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

plt.tight_layout()
plt.savefig('victre_phantom_roi.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization: victre_phantom_roi.png")

# Save ROI and lesion info
np.save('victre_phantom_roi.npy', phantom_roi)
np.save('victre_lesions_roi.npy', lesions_in_roi)
print(f"\nSaved ROI data:")
print(f"  - victre_phantom_roi.npy: {phantom_roi.shape} array")
print(f"  - victre_lesions_roi.npy: {lesions_in_roi.shape} lesion coordinates")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Extracted ROI: {phantom_roi.shape[0]} x {phantom_roi.shape[1]} x {phantom_roi.shape[2]}")
print(f"Physical size: {phantom_roi.shape[0]*spacing[0]:.1f} x {phantom_roi.shape[1]*spacing[1]:.1f} x {phantom_roi.shape[2]*spacing[2]:.1f} mm")
print(f"Lesions in ROI: {len(lesions_in_roi)}")
print(f"\nThis ROI is ready for DBT reconstruction!")
print("="*70)

plt.show()
