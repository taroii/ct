# Accelerating Low-Frequency Convergence for Limited-Angle DBT via Two-Channel Fidelity in PDHG

Taro Iyadomi, Ricardo Parada, Anna Kim, Lily Jiang, Emil Sidky, and William Chang

Single- versus two-channel Chambolle–Pock reconstruction with directional total
variation for limited-angle digital breast tomosynthesis. Splitting the data
fidelity into low- and high-frequency channels and amplifying the low-frequency
dual step accelerates low-frequency convergence; a preconditioned analysis
certifies the amplified step under the sharp condition
`tau * lambda_max(M) < 1`.

The repository is split into the paper code and results (`paper/`) and the
CT-Meeting 2026 presentation (`presentation/`).

## Getting Started

```bash
conda create -n ct python=3.13.7
conda activate ct
pip install -r requirements.txt

# Only the 3D cone-beam scripts
# (presentation/src/shepp_logan_3d.py, presentation/src/victre_reconstruction.py)
# additionally require ASTRA and a GPU:
conda install -c astra-toolbox -c nvidia astra-toolbox
```

Reproduce the main results (run from the repository root):

```bash
# Operator-level stability diagnostic  -> paper/experiments/tables/stability.md
python paper/experiments/stability_diagnostic.py

# 2D convergence: single vs two-channel across all phantoms
# (breast, shepp_logan, head, jaw, defrise) -> figures + RMSE tables
python paper/experiments/run_2d_convergence.py --phantom all

# Abstract multi-resolution comparison (512/256/128) -> paper/figures/
python paper/reconstruction.py
```

Layout:

| Path | Contents |
|------|----------|
| `paper/reconstruction.py` | Core single/two-channel CP–DTV solver |
| `paper/experiments/` | Stability diagnostic, 2D convergence experiments, figures, tables |
| `paper/phantoms3d/` | Analytic 3D phantom generators (head, jaw, breast) |
| `paper/manuscripts/` | Abstract and new-paper sources and PDFs |
| `presentation/` | Deck (`main.tex`) and figure-generating scripts (`src/`) |
| `data/` | Breast phantom arrays and VICTRE phantom metadata |

## References

Sidky, E. Y., et al. "Accurate volume image reconstruction for digital breast
tomosynthesis with directional-gradient and pixel sparsity regularization."
*Journal of Medical Imaging* 12.S1 (2025): S13013.

Chambolle, A., and Pock, T. "A first-order primal-dual algorithm for convex
problems with applications to imaging." *Journal of Mathematical Imaging and
Vision* 40.1 (2011): 120–145.
