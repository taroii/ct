"""
Visualize the full VICTRE phantom slice and highlight the ROI that was extracted
This addresses Dr. Sidky's concern about the dark background in the slice images
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

print("="*70)
print("VISUALIZING ROI LOCATION IN FULL VICTRE PHANTOM")
print("="*70)

# Load the full VICTRE phantom
mhd_file = Path('../data/victre_phantom/p_-72912147.mhd')
raw_file = Path('../data/victre_phantom/p_-72912147.raw')
loc_file = Path('../data/victre_phantom/p_-72912147.loc')

# Phantom dimensions from MHD file
dims = (748, 896, 897)  # (x, y, z)
spacing = (0.1, 0.1, 0.1)  # mm
offset = (-9.45, -53.6013, -45.1684)  # mm

print(f"\nLoading full VICTRE phantom...")
print(f"Dimensions: {dims[0]} x {dims[1]} x {dims[2]} voxels")

# Read full phantom
phantom_full = np.fromfile(raw_file, dtype=np.uint8)
phantom_full = phantom_full.reshape(dims, order='F')  # Fortran order
print(f"Full phantom value range: [{phantom_full.min()}, {phantom_full.max()}]")

# Load the extracted ROI for comparison
roi_path = Path('../data/generated_roi/victre_phantom_roi.npy')
phantom_roi = np.load(roi_path)
print(f"\nExtracted ROI dimensions: {phantom_roi.shape}")
print(f"ROI value range: [{phantom_roi.min()}, {phantom_roi.max()}]")

# From the preprocessing script, we know the ROI position
# These are the values from the best position found
x_start, y_start, z_start = 220, 320, 440  # From the search result
roi_size_xy = 256
roi_size_z = 20

print(f"\nROI location in full phantom:")
print(f"  X: [{x_start}, {x_start + roi_size_xy}]")
print(f"  Y: [{y_start}, {y_start + roi_size_xy}]") 
print(f"  Z: [{z_start}, {z_start + roi_size_z}]")

# Create visualization
fig = plt.figure(figsize=(20, 14))

# We'll show multiple z-slices to give context
z_slices_to_show = [
    z_start - 50,  # Before ROI
    z_start,  # Start of ROI
    z_start + roi_size_z // 2,  # Middle of ROI
    z_start + roi_size_z - 1,  # End of ROI
    z_start + roi_size_z + 50  # After ROI
]

for idx, z_slice in enumerate(z_slices_to_show):
    if z_slice < 0 or z_slice >= dims[2]:
        continue
        
    ax = plt.subplot(2, 3, idx + 1)
    
    # Show full phantom slice
    im = ax.imshow(phantom_full[:, :, z_slice].T, cmap='gray', origin='lower', 
                   vmin=0, vmax=255)
    
    # Determine if this slice is in the ROI
    if z_start <= z_slice < z_start + roi_size_z:
        # This slice is IN the ROI - draw red box
        rect = patches.Rectangle((x_start, y_start), roi_size_xy, roi_size_xy, 
                                linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        title_color = 'red'
        roi_z_idx = z_slice - z_start
        title = f'Z={z_slice} (ROI slice {roi_z_idx})'
    else:
        # This slice is OUTSIDE the ROI - draw blue dashed box for reference
        rect = patches.Rectangle((x_start, y_start), roi_size_xy, roi_size_xy, 
                                linewidth=2, edgecolor='blue', facecolor='none',
                                linestyle='--', alpha=0.5)
        ax.add_patch(rect)
        title_color = 'black'
        if z_slice < z_start:
            title = f'Z={z_slice} ({z_start - z_slice} slices before ROI)'
        else:
            title = f'Z={z_slice} ({z_slice - (z_start + roi_size_z - 1)} slices after ROI)'
    
    ax.set_title(title, fontsize=11, color=title_color, fontweight='bold')
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    
    # Add grid for reference
    ax.grid(True, alpha=0.2, linestyle=':')
    
    # Add colorbar for the first subplot
    if idx == 0:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Add a side-by-side comparison in the last subplot
ax6 = plt.subplot(2, 3, 6)

# Show middle slice of ROI from the full phantom
z_roi_mid = roi_size_z // 2
z_full_mid = z_start + z_roi_mid

# Extract the ROI region from the full phantom
roi_from_full = phantom_full[x_start:x_start+roi_size_xy, 
                             y_start:y_start+roi_size_xy, 
                             z_full_mid]

# Create side-by-side comparison
comparison = np.zeros((roi_size_xy, roi_size_xy * 2 + 10))
comparison[:, :roi_size_xy] = roi_from_full.T
comparison[:, roi_size_xy+10:] = phantom_roi[:, :, z_roi_mid].T

im6 = ax6.imshow(comparison, cmap='gray', origin='lower', vmin=0, vmax=255)
ax6.set_title('Side-by-side: Extracted from full (left) vs Saved ROI (right)', 
              fontsize=11, fontweight='bold')
ax6.axvline(x=roi_size_xy, color='white', linewidth=1)
ax6.axvline(x=roi_size_xy+10, color='white', linewidth=1)
ax6.set_xlabel('X (voxels)')
ax6.set_ylabel('Y (voxels)')
plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

plt.suptitle('Full VICTRE Phantom with ROI Location Highlighted\n' + 
             f'ROI: 256×256×20 voxels at position ({x_start}, {y_start}, {z_start})',
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the figure
output_path = '../results/current/victre_full_phantom_with_roi.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: {output_path}")

# Also create a single slice comparison at higher resolution
fig2, axes = plt.subplots(1, 3, figsize=(18, 6))

# Choose a representative slice in the middle of the ROI
z_display = z_start + roi_size_z // 2

# Full phantom slice
ax1 = axes[0]
im1 = ax1.imshow(phantom_full[:, :, z_display].T, cmap='gray', origin='lower', 
                 vmin=0, vmax=255)
rect1 = patches.Rectangle((x_start, y_start), roi_size_xy, roi_size_xy, 
                         linewidth=3, edgecolor='red', facecolor='none')
ax1.add_patch(rect1)
ax1.set_title(f'Full VICTRE Phantom (slice Z={z_display})', fontsize=12, fontweight='bold')
ax1.set_xlabel('X (voxels)')
ax1.set_ylabel('Y (voxels)')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Zoomed ROI from full phantom
ax2 = axes[1]
roi_from_full = phantom_full[x_start:x_start+roi_size_xy,
                             y_start:y_start+roi_size_xy,
                             z_display]
im2 = ax2.imshow(roi_from_full.T, cmap='gray', origin='lower', vmin=0, vmax=255)
ax2.set_title('Extracted ROI Region', fontsize=12, fontweight='bold', color='red')
ax2.set_xlabel('X (ROI coordinates)')
ax2.set_ylabel('Y (ROI coordinates)')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# The processed ROI used in reconstruction
ax3 = axes[2]
roi_slice = phantom_roi[:, :, z_display - z_start]
im3 = ax3.imshow(roi_slice.T, cmap='gray', origin='lower', vmin=0, vmax=255)
ax3.set_title('Processed ROI (used for reconstruction)', fontsize=12, fontweight='bold')
ax3.set_xlabel('X (ROI coordinates)')
ax3.set_ylabel('Y (ROI coordinates)')
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

plt.suptitle('ROI Extraction from Full VICTRE Phantom', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path2 = '../results/current/victre_roi_extraction_detail.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Saved detailed comparison to: {output_path2}")

plt.show()

print("\n" + "="*70)
print("EXPLANATION FOR DR. SIDKY:")
print("="*70)
print("""
The dark background in the reconstruction images occurs because:

1. The ROI was extracted from a specific region of the full VICTRE phantom
   (256×256×20 voxels from a 748×896×897 phantom)

2. The extracted region was chosen to maximize tissue variance and include
   lesions, focusing on glandular tissue areas

3. The background appears dark because:
   - The ROI primarily contains breast tissue (adipose and glandular)
   - Areas outside the breast tissue are air (value 0 = black)
   - The VICTRE phantom models a realistic breast with surrounding air

4. The reconstruction algorithms (single-channel and two-channel L1-DTV)
   are working on this extracted ROI, not the full phantom

This is intentional and correct - the ROI focuses on the clinically relevant
breast tissue region rather than including empty space around the breast.
""")