# Accelerating Low-Frequency Convergence for Limited-Angle DBT via Two-Channel Fidelity in PDHG

Taro Iyadomi, Ricardo Parada, Anna Kim, Lily Jiang, Emil Sidky, and William Chang

## Getting Started

### Prerequisites
```bash
# Create conda environment (Python 3.13.7 recommended)
conda create -n ct python=3.13.7
conda activate ct

# Install dependencies
conda install pip
pip install -r requirements.txt
conda install -c astra-toolbox -c nvidia astra-toolbox
```

## Quick Start

### ** Algorithm Comparison** 

Compare single-channel algorithm (L1-DTV method in Sidky et al 2025) to our proposed two-channel approach across multiple image resolutions (512x512, 256x256, 128x128). 

```bash
python scripts/compare_methods_multiresolution.py
```

This script generates:
1. ```final_figures/rmse_table.txt```
2. ```final_figures/convergence_subplots.png```
3. ```final_figures/convergence_combined.png```

### ** Generate Additional Plots**

Run this script after running ```compare_methods_multiresolution.py``` to obtain other plots:

```bash
python scripts/plot_reconstruction_comparison.py
```

This script generates:
1. ```final_figures/reconstruction_2x2_256.png```
2. ```final_figures/convergence_256.png```
3. ```final_figures/convergence_256_loglog.png```
4. ```final_figures/convergence_256_sidebyside.png```

## References

Sidky, E. Y., et al. "Accurate volume image reconstruction for digital breast tomosynthesis with directional-gradient and pixel sparsity regularization." *Journal of Medical Imaging* 12.S1 (2025): S13013.
