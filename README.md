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
│   ├── run_experiments.py                        # Main experiment runner (NEW)
│   ├── run_reconstruction_comparison.py          # Modular reconstruction comparison (NEW)
│   ├── ecp_optimizer.py                          # ECP parameter optimization (NEW)
│   ├── optimize_victre_params.py                 # ECP-VICTRE integration (NEW)
│   ├── compare_methods.py                        # Original method comparison (legacy)
│   ├── compare_methods_gpt.py                    # Two-channel method comparison (legacy)
│   └── DTVminHan.py                              # DTV minimization (Han's original)
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

### **🚀 Main Experiment Runner** (Recommended)

To run reconstruction algorithm comparisons with the new modular system:

```bash
cd scripts
python run_experiments.py
```

This will compare all available algorithms and generate comprehensive plots in `results/current/`.

**Available algorithms:**
- `single_channel` - Original L1-DTV (Sidky et al.)
- `two_channel` - Frequency-split extension
- `ecp_optimized` - ECP parameter optimization (NEW)

**Quick options:**
```bash
# Run specific algorithms
python run_experiments.py --algorithms single_channel ecp_optimized

# Quick test (fewer iterations)
python run_experiments.py --quick

# Help and options
python run_experiments.py --help
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
