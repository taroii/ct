# Changes from Original Scripts

This folder (`scripts_raw`) is a duplicate of the original `scripts` folder, modified to use **raw VICTRE phantom data** instead of the synthetic ICA distribution.

## Why This Change Was Made

Dr. Sidky noticed that the slice images looked strange with a very dark background. Investigation revealed that the original `run_reconstruction_comparison.py` was **not** reconstructing the actual VICTRE phantom data. Instead, it was creating a sparse synthetic "ICA distribution" phantom:

- Original: Created an empty array, set background to 0.08 where VICTRE > 100, added small spherical tumors (value 0.4) at lesion locations
- Result: Mostly black images with tiny bright spots

The `visualize_roi_in_full_phantom.py` script showed the actual VICTRE ROI (rich tissue structure), but the reconstruction was working on completely different data.

## Key Changes in `run_reconstruction_comparison.py`

### 1. Removed ICA Distribution Creation

**Before (lines 136-157 in original):**
```python
# Create ICA distribution phantom (like Figure 8 in paper)
phantom_3d = zeros((nx, ny, nz))
glandular_mask = phantom_victre > 100
phantom_3d[glandular_mask] = background_enhancement  # 0.08
# Add tumors at lesion locations with value 0.4
```

**After:**
```python
# Normalize raw phantom to attenuation values
phantom_3d = phantom_victre.astype(np.float64) / 255.0
ATTENUATION_SCALE = 0.8
phantom_3d = phantom_3d * ATTENUATION_SCALE
```

### 2. High-Variance Slice Selection

**Before:** Selected slice with most lesions
**After:** Selected slice with highest tissue variance (more representative for raw data)

### 3. Visualization Parameters

**Before:** `vmin, vmax = 0, 0.5` (appropriate for sparse ICA)
**After:** `vmin, vmax = 0, ATTENUATION_SCALE` (0.8, shows full tissue structure)

### 4. Output Paths

All outputs now go to `../results/raw_victre/`:
- `figure_raw_victre.png` (x-y and x-z planes)
- `raw_victre_profile.png` (depth profile)
- `raw_victre_convergence.png` (iteration convergence)
- `raw_victre_performance_summary.png` (slice-by-slice comparison)
- `raw_victre_results.pkl` (saved results for post-processing)

### 5. Saved Results

The pickle file now includes additional data:
- `phantom_raw`: Original VICTRE phantom values (0-255)
- `attenuation_scale`: The scaling factor used (0.8)

## Usage

```bash
cd scripts_raw

# Full reconstruction (all 20 slices, 500 iterations each)
python run_reconstruction_comparison.py

# Quick test (1 slice, 200 iterations)
python run_reconstruction_comparison.py --quick

# Specific algorithms only
python run_reconstruction_comparison.py --algorithms single_channel
python run_reconstruction_comparison.py --algorithms two_channel
```

## Comparison: ICA vs Raw Phantom

| Aspect | Original (ICA) | Modified (Raw) |
|--------|---------------|----------------|
| Data source | Synthetic from VICTRE | Direct VICTRE values |
| Value range | 0, 0.08, 0.4 (sparse) | 0 to 0.8 (continuous) |
| Tissue structure | Not visible | Full breast anatomy |
| Background | Mostly black | Adipose tissue visible |
| Purpose | Contrast agent imaging | Anatomical imaging |

## Why the Paper Uses ICA Distribution (Not Raw Phantom)

The Sidky et al. paper "Image reconstruction algorithm for contrast-enhanced Digital Breast Tomosynthesis" is specifically designed for **Contrast-Enhanced DBT (CE-DBT)** imaging, not standard anatomical breast imaging.

### The CE-DBT Workflow

1. **Dual-Energy Acquisition**: Two X-ray images are taken at different energies (low ~20 keV, high ~50 keV)
2. **Material Decomposition**: The two images are mathematically combined to isolate iodine from soft tissue
3. **Result**: A sparse ICA distribution image showing only contrast agent uptake

### Why This Matters for the Algorithm

The L1-DTV algorithm exploits **sparsity** in its L1 regularization term. The ICA distribution is naturally sparse:
- Most pixels are zero (no contrast agent)
- Only tumor regions and blood vessels contain iodine
- This makes L1 regularization highly effective

The raw breast phantom is **not sparse**:
- Continuous tissue values throughout
- Adipose, glandular, and fibrous tissues fill the entire volume
- L1 regularization may over-penalize legitimate tissue signals

### Implications

| Application | Data Characteristics | L1 Effectiveness |
|-------------|---------------------|------------------|
| CE-DBT (ICA) | Sparse, discrete values | Excellent - matches algorithm assumptions |
| Raw Phantom | Dense, continuous values | May need parameter tuning |

## Algorithm Parameters and Tuning for Raw Phantom

The current algorithm parameters in `CONFIG` were optimized for sparse ICA imaging. For raw phantom reconstruction, these parameters may need adjustment:

### Current Parameters

```python
CONFIG = {
    # DTV (Directional Total Variation) Parameters
    'alpha': 1.75,        # Balance between in-plane (x,y) and depth (z) regularization
                          # Paper recommends 1.7-1.9; higher values = more depth smoothing

    # L1 Sparsity Parameters
    'beta': 5.0,          # L1 sparsity penalty weight
                          # Higher = more sparse solution (fewer non-zero pixels)
                          # ⚠️ MAY NEED REDUCTION for dense tissue imaging

    'l1f': 1.0,           # L1 penalty factor (multiplier for beta)

    # PDHG Algorithm Parameters
    'rho': 1.75,          # Predictor-corrector relaxation parameter
                          # Controls step size in optimization
                          # Range: 1.0-2.0 typically stable

    'eps': 0.001,         # Data discrepancy RMSE target
                          # Algorithm stops when data fit reaches this level

    # Primal-Dual Step Size Balance
    'nuxfact': 0.5,       # Primal step size factor for x-direction
    'nuyfact': 0.5,       # Primal step size factor for y-direction
    'stepbalance': 100.0, # Balance between primal/dual steps

    # Two-Channel Frequency Split Parameters
    'cutoffparm': 4.0,    # High-frequency filter cutoff (in pixels)
                          # Lower = more aggressive high-pass filtering

    'cutoffparm_lo': 8.0, # Low-frequency filter cutoff
                          # Defines the split point between channels

    'sigma_lo_scale': 4.0,# Sigma scaling for low-frequency channel
                          # Affects step size in low-frequency updates

    # Data/Noise Model
    'larc': 1.0,          # Limited angle reconstruction correction
    'addnoise': 0,        # Add Poisson noise (0=off, 1=on)
    'nph': 1.e6,          # Photon count for noise model
}
```

### Parameters Most Likely to Need Tuning

1. **`beta` (L1 sparsity weight)** - MOST CRITICAL
   - Current: 5.0 (aggressive sparsity for ICA)
   - For raw phantom: Try 0.1 - 1.0
   - Why: Dense tissue shouldn't be penalized for being non-zero

2. **`alpha` (DTV balance)**
   - Current: 1.75
   - For raw phantom: May remain similar, but tissue boundaries differ from ICA edges
   - Consider: 1.5 - 2.0 range

3. **`eps` (convergence threshold)**
   - Current: 0.001
   - For raw phantom: May need loosening (0.005 - 0.01) if dense tissue causes slower convergence

4. **`cutoffparm` and `cutoffparm_lo` (frequency split)**
   - These control the two-channel method's frequency separation
   - May need adjustment based on tissue texture vs ICA spot sizes

### Suggested Experiments

```bash
# Test with reduced L1 penalty
# Edit CONFIG['beta'] = 1.0, then run:
python run_reconstruction_comparison.py --quick

# Test with very low L1 penalty
# Edit CONFIG['beta'] = 0.1, then run:
python run_reconstruction_comparison.py --quick
```

## Notes

- The original `scripts/run_reconstruction_comparison.py` is preserved for ICA distribution experiments (matches paper Figure 8)
- This modified version is for reconstructing actual breast tissue structure
- Parameter tuning experiments are recommended before drawing conclusions about algorithm performance on raw data
