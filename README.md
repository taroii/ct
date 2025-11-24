# DBT Reconstruction with Directional-Gradient and Pixel Sparsity Regularization

Recreating results from "Accurate volume image reconstruction for digital breast tomosynthesis with directional-gradient and pixel sparsity regularization" (Sidky et al., 2025).

## Repository Structure

```
ct/
├── README.md                   # This file
├── EXPERIMENT_GUIDE.md         # Detailed experiment runner guide
├── requirements.txt            # Python dependencies
├── scripts/                    # Python scripts
│   ├── recreate_figures6_7_analytic.py           # Figures 6 & 7: Analytic phantom with overlapping spheres
│   ├── recreate_figure8_victre_ica.py            # Figure 8: VICTRE phantom with ICA distribution
│   ├── preprocess_victre_phantom.py              # Extract centered ROI from VICTRE phantom
│   ├── preprocess_victre_phantom_variance.py     # Extract high-variance ROI from VICTRE phantom
│   ├── run_reconstruction_comparison.py          # Main reconstruction comparison script
│   ├── compare_best_worst_slices.py              # Detailed slice-by-slice visualization
│   ├── plot_convergence_slices.py                # Plot convergence for specific slices
│   ├── DTVminHan.py                              # DTV minimization (Han's original)
│   └── old/                                      # Deprecated/experimental scripts
├── notebooks/                  # Jupyter notebooks
├── data/                       # Data files
│   ├── generated_roi/          # Generated VICTRE ROIs (output from preprocessing)
│   │   ├── victre_phantom_roi.npy     # Current ROI (256×256×20)
│   │   ├── victre_lesions_roi.npy     # Lesion coordinates in ROI
│   │   └── old rois/                  # Archive of previous ROIs
│   ├── phantoms_from_paper/    # Analytic phantoms from paper
│   │   ├── Phantom_Adipose.npy
│   │   ├── Phantom_Calcification.npy
│   │   └── Phantom_Fibroglandular.npy
│   ├── victre_phantom/         # VICTRE raw data (.raw, .mhd, .loc)
│   └── dataHan256/             # Han's original data
├── results/                    # Output images and results
│   ├── current/                # Latest reconstruction results
│   ├── 10.25_results_high_variance/   # Archived: Original two-channel comparison
│   └── previous_runs/          # Other archived results
└── papers/                     # Reference papers (PDFs)
    ├── Sidky_2012_Phys._Med._Biol._57_3065.pdf
    ├── ECP.pdf                 # Every Call is Precious paper
    └── ...
```

## Getting Started

### Prerequisites
```bash
# Create conda environment (Python 3.13.7 recommended)
conda create -n ct python=3.13.7
conda activate ct

# Install dependencies
conda install pip
pip install -r requirements.txt
```

## Quick Start

### **🚀 Algorithm Comparison** (Recommended)

To compare single-channel vs two-channel L1-DTV reconstruction:

```bash
cd scripts
python run_reconstruction_comparison.py --algorithms single_channel two_channel
```

This will reconstruct all 20 slices of the VICTRE phantom and generate comprehensive comparison plots in `results/current/`.

**Quick options:**
```bash
# Quick test (1 slice, 200 iterations instead of 500)
python run_reconstruction_comparison.py --quick

# Run single algorithm only
python run_reconstruction_comparison.py --algorithms single_channel
```

**Output:**
- `figure_8_victre_ica.png` - x-y and x-z plane reconstructions
- `figure_8_profile.png` - Depth profile comparison
- `figure_8_convergence.png` - Iteration convergence (high-variance slice)
- `performance_summary_all_slices.png` - Slice-by-slice performance comparison
- `fig8_victre_results.pkl` - Saved results for post-processing

**Post-processing options:**

After running the main comparison, you can generate additional visualizations:

```bash
# Detailed comparison of best/worst performing slices
python compare_best_worst_slices.py

# Convergence plots for specific slices
python plot_convergence_slices.py                    # Auto: best single, best two, median
python plot_convergence_slices.py --slices 4 9 14    # Specific slices (0-indexed)
```

---

## Detailed Workflows

### Workflow 1: Figures 6 & 7 (Analytic Phantom)

**Purpose:** Validate reconstruction algorithms using synthetic phantom with overlapping spheres

```bash
cd scripts
python recreate_figures6_7_analytic.py
```

**What it does:**
- Creates synthetic 3D phantom (256×256×10) with overlapping spheres at known depths
- Simulates DBT projections (25 views, 50° arc)
- Reconstructs with Single-channel and Two-channel L1-DTV
- Demonstrates depth resolution capability

**Output:** (saved to `results/current/`)
- `figure_6_comparison.png` - x-y and x-z plane views
- `figure_7_profile.png` - Depth profiles through overlapping spheres
- `convergence_comparison.png` - Iteration convergence

---

### Workflow 2: VICTRE Phantom Preparation (One-time Setup)

**Step 1:** Extract high-variance ROI from VICTRE phantom
```bash
cd scripts
python preprocess_victre_phantom_variance.py
```

**Output:**
- `data/generated_roi/victre_phantom_roi.npy` - Extracted 256×256×20 ROI
- `data/generated_roi/victre_lesions_roi.npy` - Lesion coordinates in ROI
- `results/current/victre_phantom_roi_variance.png` - ROI visualization

**Step 2:** Run algorithm comparison (see Quick Start above)

---

### Workflow 3: Legacy Paper Recreation

To recreate the exact Figure 8 from the paper:

```bash
cd scripts
python recreate_figure8_victre_ica.py
```

**Output:** (saved to `results/current/`)
- `figure_8_victre_ica.png` - x-y and x-z plane views
- `figure_8_profile.png` - Depth profile through phantom
- `figure_8_convergence.png` - Iteration convergence

## Methods Comparison

### Single-channel L1-DTV (Paper Method)
- Directional Total Variation (DTV) minimization
- Parameters: α=1.75, β=5.0
- PDHG algorithm with He-Yuan predictor-corrector (ρ=1.75)

### Two-channel L1-DTV (Frequency-Split Extension)
- Low/high frequency channel separation
- Low-frequency cutoff: 8.0 (fraction of Nyquist)
- σ_lo_scale = 4.0

## References

Sidky, E. Y., et al. "Accurate volume image reconstruction for digital breast tomosynthesis with directional-gradient and pixel sparsity regularization." *Journal of Medical Imaging* 12.S1 (2025): S13013.
