# Reconstruction Algorithm Comparison Guide

## Quick Start

To recreate the results with the new modular system:

```bash
cd scripts
python run_experiments.py
```

This will:
1. Run single-channel, two-channel, and ECP-optimized reconstructions
2. Generate comparison plots in `/results/current/`
3. Save detailed results for analysis

## Available Commands

### Run All Algorithms (Default)
```bash
python run_experiments.py
```

### Run Specific Algorithms
```bash
# Compare single-channel vs ECP-optimized
python run_experiments.py --algorithms single_channel ecp_optimized

# Test just the ECP approach
python run_experiments.py --algorithms ecp_optimized
```

### Quick Testing (Fewer Iterations)
```bash
python run_experiments.py --quick
```

### Verbose Output
```bash
python run_experiments.py --verbose
```

## Available Algorithms

| Algorithm Name | Description | Paper Reference |
|----------------|-------------|-----------------|
| `single_channel` | Original L1-DTV method | Sidky et al. 2012 |
| `two_channel` | Frequency-split approach | Your extension |
| `ecp_optimized` | ECP parameter optimization | Novel approach |

## Output Structure

After running, you'll find in `/results/current/`:

- `convergence_comparison.png` - Iteration convergence plots
- `visual_comparison.png` - Side-by-side reconstruction images  
- `profile_comparison.png` - Line profile comparisons
- `reconstruction_results.pkl` - Raw reconstruction data

## Previous Results

Your previous results are archived in:
- `/results/10.25_results_high_variance/` - Original two-channel vs single-channel

## Adding New Algorithms

To add a new algorithm:

1. **Implement the reconstruction function** in `run_reconstruction_comparison.py`:
```python
def reconstruct_my_algorithm(sinodata, phantom, geom, config, verbose=False):
    # Your reconstruction code here
    return reconstruction, error_history
```

2. **Add to the algorithm registry**:
```python
ALGORITHMS = {
    'single_channel': 'Single-channel L1-DTV',
    'two_channel': 'Two-channel L1-DTV', 
    'ecp_optimized': 'ECP-optimized Two-channel',
    'my_algorithm': 'My New Algorithm'  # Add this line
}
```

3. **Add the algorithm case** in `run_comparison()`:
```python
elif alg_name == 'my_algorithm':
    recon, errors = reconstruct_my_algorithm(
        sinodata_mid, phimage_mid, geom, CONFIG, verbose=CONFIG['verbose']
    )
```

4. **Run your new algorithm**:
```bash
python run_experiments.py --algorithms my_algorithm single_channel
```

## ECP Parameter Optimization

The ECP algorithm automatically optimizes:
- `cutoff_lo`: Frequency separation threshold (2.0-16.0)
- `sigma_scale`: Step size scaling factor (1.0-8.0)  
- `eps_ratio`: Data fidelity ratio (0.8-2.0)

During optimization, ECP:
1. Uses acceptance regions to avoid unpromising parameter combinations
2. Gradually expands search space as needed
3. Finds optimal parameters in ~15 reconstructions instead of grid search

## Performance Monitoring

The system tracks:
- **RMSE**: Reconstruction accuracy vs ground truth
- **Convergence**: Error reduction over iterations
- **Time**: Computational efficiency
- **3D Performance**: Average error across all slices

## Customization

Edit `CONFIG` in `run_reconstruction_comparison.py` to modify:
- Number of iterations (`itermax`)
- Regularization parameters (`alpha`, `beta`)
- Data fidelity weights (`eps`, `eps_hi`, `eps_lo`)
- Geometry settings (`mfact`)

## Troubleshooting

### "VICTRE phantom ROI data not found"
```bash
cd scripts
python preprocess_victre_phantom_variance.py
```

### Memory issues with large reconstructions
```bash
python run_experiments.py --quick  # Reduces iterations
```

### Import errors
Make sure you're in the `scripts/` directory:
```bash
cd /path/to/ct/scripts
python run_experiments.py
```