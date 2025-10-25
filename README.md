# DBT Reconstruction with Directional-Gradient and Pixel Sparsity Regularization

Recreating results from "Accurate volume image reconstruction for digital breast tomosynthesis with directional-gradient and pixel sparsity regularization" (Sidky et al., 2025).

## Repository Structure

```
ct/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── scripts/                    # Python scripts
│   ├── recreate_figures6_7_analytic.py      # Figures 6 & 7: Analytic phantom with overlapping spheres
│   ├── recreate_figure8_victre_ica.py       # Figure 8: VICTRE phantom with ICA distribution
│   ├── preprocess_victre_phantom.py         # Extract ROI from VICTRE phantom data
│   ├── compare_methods.py                   # Original method comparison (legacy)
│   ├── compare_methods_gpt.py               # Two-channel method comparison (legacy)
│   └── DTVminHan.py                         # DTV minimization (Han's original)
├── notebooks/                  # Jupyter notebooks
├── data/                       # Data files
│   ├── phantoms/               # Phantom arrays (.npy files)
│   ├── victre_phantom/         # VICTRE raw data (.raw, .mhd, .loc)
│   └── dataHan256/             # Han's original data
├── results/                    # Output images and results
│   ├── current/                # Latest reconstruction results
│   └── previous_runs/          # Archived previous runs
└── papers/                     # Reference papers (PDFs)
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

## Reconstruction Workflows

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

### Workflow 2: Figure 8 (VICTRE Phantom with ICA)

**Purpose:** Demonstrate clinical applicability using realistic breast phantom with contrast-enhanced lesions

**Step 1:** Preprocess VICTRE phantom (one-time setup)
```bash
cd scripts
python preprocess_victre_phantom.py
```

**Output:**
- `data/phantoms/victre_phantom_roi.npy` - Extracted 256×256×20 ROI
- `data/phantoms/victre_lesions_roi.npy` - 6 lesion coordinates
- `results/current/victre_phantom_roi.png` - Visualization

**Step 2:** Reconstruct ICA distribution
```bash
cd scripts
python recreate_figure8_victre_ica.py
```

**What it does:**
- Loads VICTRE phantom ROI
- Creates ICA distribution (tumor=0.4, background=0.08)
- Places contrast-enhanced tumors at 6 lesion locations
- Reconstructs with both L1-DTV methods

**Output:** (saved to `results/current/`)
- `figure_8_victre_ica.png` - x-y and x-z plane views
- `figure_8_profile.png` - Depth profile through phantom
- `figure_8_convergence.png` - Iteration convergence

**Parameters:**
- Image size: 256×256×20 voxels
- Iterations: 500 per slice
- DBT geometry: 50° arc, 25 views

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
