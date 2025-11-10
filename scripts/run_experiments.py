#!/usr/bin/env python3
"""
Experiment runner for reconstruction algorithm comparison.

Usage:
    python run_experiments.py                          # Run all algorithms
    python run_experiments.py --algorithms single two  # Run specific algorithms  
    python run_experiments.py --quick                  # Quick test (fewer iterations)
    python run_experiments.py --help                   # Show help
"""

import argparse
import sys
from pathlib import Path
from run_reconstruction_comparison import run_comparison, ALGORITHMS, CONFIG

def main():
    parser = argparse.ArgumentParser(
        description="Run reconstruction algorithm comparison experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available algorithms:
  single_channel    - Single-channel L1-DTV (Sidky's original)
  two_channel       - Two-channel L1-DTV (frequency-split)  
  ecp_optimized     - ECP-optimized Two-channel parameters

Examples:
  python run_experiments.py
  python run_experiments.py --algorithms single_channel two_channel
  python run_experiments.py --quick --algorithms ecp_optimized
        """
    )
    
    parser.add_argument('--algorithms', '-a', 
                        nargs='+',
                        choices=list(ALGORITHMS.keys()),
                        default=list(ALGORITHMS.keys()),
                        help='Algorithms to run (default: all)')
    
    parser.add_argument('--quick', '-q',
                        action='store_true',
                        help='Quick test with reduced iterations')
    
    parser.add_argument('--no-save',
                        action='store_true', 
                        help='Do not save results to disk')
    
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Verbose output during reconstruction')
    
    args = parser.parse_args()
    
    # Modify config for quick testing
    if args.quick:
        CONFIG['itermax'] = 200
        print("Quick mode: Using 200 iterations instead of 500")
    
    if args.verbose:
        CONFIG['verbose'] = True
    
    print("Running reconstruction comparison with algorithms:")
    for alg in args.algorithms:
        print(f"  - {ALGORITHMS[alg]}")
    
    # Check if we have required data
    data_path = Path('../data/generated_roi')
    if not (data_path / 'victre_phantom_roi.npy').exists():
        print("\nError: VICTRE phantom ROI data not found!")
        print(f"Expected: {data_path / 'victre_phantom_roi.npy'}")
        print("Please run preprocess_victre_phantom_variance.py first.")
        sys.exit(1)
    
    # Run the comparison
    try:
        results = run_comparison(
            algorithms_to_run=args.algorithms,
            save_results=not args.no_save
        )
        
        # Print final summary
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        
        # Sort algorithms by performance
        alg_performance = [(name, res['final_rmse']) for name, res in results.items()]
        alg_performance.sort(key=lambda x: x[1])
        
        print(f"\nRanking by RMSE (middle slice):")
        for i, (alg_name, rmse) in enumerate(alg_performance, 1):
            print(f"{i}. {ALGORITHMS[alg_name]:<25} RMSE: {rmse:.6f}")
        
        if len(alg_performance) > 1:
            best_alg, best_rmse = alg_performance[0]
            second_alg, second_rmse = alg_performance[1]
            improvement = (second_rmse - best_rmse) / second_rmse * 100
            print(f"\nBest algorithm: {ALGORITHMS[best_alg]}")
            print(f"Improvement over second-best: {improvement:.1f}%")
        
        # Show 3D results if available
        if all('avg_rmse' in res for res in results.values()):
            print(f"\nAverage RMSE (all slices):")
            alg_3d = [(name, res['avg_rmse']) for name, res in results.items()]
            alg_3d.sort(key=lambda x: x[1])
            
            for i, (alg_name, rmse) in enumerate(alg_3d, 1):
                print(f"{i}. {ALGORITHMS[alg_name]:<25} RMSE: {rmse:.6f}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()