"""
ECP (Every Call is Precious) Parameter Optimizer for Two-Channel L1-DTV Reconstruction

This module implements the ECP algorithm to optimize reconstruction parameters
without wasting expensive function evaluations.
"""

import numpy as np
from typing import Tuple, Callable, List, Dict, Optional
import time


class ECPOptimizer:
    """
    Every Call is Precious (ECP) optimizer for black-box function optimization.
    Specifically designed for optimizing expensive reconstruction parameters.
    """
    
    def __init__(self, 
                 bounds: List[Tuple[float, float]],
                 epsilon_init: float = 0.01,
                 tau: float = 1.1,
                 C: float = 100,
                 max_iters: int = 50,
                 verbose: bool = True):
        """
        Initialize ECP optimizer.
        
        Args:
            bounds: List of (min, max) tuples for each parameter
            epsilon_init: Initial acceptance region size
            tau: Growth factor for epsilon
            C: Rejection threshold before growing epsilon
            max_iters: Maximum number of evaluations
            verbose: Print progress information
        """
        self.bounds = bounds
        self.dim = len(bounds)
        self.epsilon = epsilon_init
        self.tau = tau
        self.C = C
        self.max_iters = max_iters
        self.verbose = verbose
        
        # Normalize bounds to [0, 1] for easier handling
        self.bounds_array = np.array(bounds)
        self.scale = self.bounds_array[:, 1] - self.bounds_array[:, 0]
        self.offset = self.bounds_array[:, 0]
        
        # Storage for evaluated points
        self.evaluated_points = []
        self.evaluated_values = []
        self.rejection_count = 0
        self.prev_rejection_count = 0
        
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Convert from original space to [0,1] space"""
        return (x - self.offset) / self.scale
    
    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Convert from [0,1] space to original space"""
        return x * self.scale + self.offset
    
    def _acceptance_condition(self, x: np.ndarray) -> bool:
        """
        Check if point x meets the ECP acceptance condition.
        
        Args:
            x: Point in normalized [0,1] space
            
        Returns:
            True if point should be evaluated
        """
        if len(self.evaluated_points) == 0:
            return True
            
        x_norm = self._normalize(x) if x.max() > 1 else x
        
        # Calculate min(f(xi) + epsilon * ||x - xi||)
        min_val = float('inf')
        for xi, fi in zip(self.evaluated_points, self.evaluated_values):
            xi_norm = self._normalize(xi) if xi.max() > 1 else xi
            dist = np.linalg.norm(x_norm - xi_norm)
            val = fi + self.epsilon * dist
            min_val = min(min_val, val)
        
        # Check if >= max(f(xi))
        max_val = max(self.evaluated_values)
        return min_val >= max_val
    
    def _sample_point(self) -> np.ndarray:
        """Sample a random point from the parameter space"""
        return np.random.uniform(0, 1, self.dim)
    
    def optimize(self, objective_func: Callable, 
                initial_point: Optional[np.ndarray] = None) -> Dict:
        """
        Run ECP optimization.
        
        Args:
            objective_func: Function to MINIMIZE (takes parameters, returns scalar)
                           Note: ECP maximizes, so we negate the objective
            initial_point: Optional starting point (in original space)
            
        Returns:
            Dictionary with:
                - 'best_params': Best parameters found (in original space)
                - 'best_value': Best objective value (original, not negated)
                - 'history': List of (params, value) tuples
                - 'n_evaluations': Number of function evaluations
        """
        history = []
        
        # Initial evaluation
        if initial_point is not None:
            x_init = self._normalize(initial_point)
        else:
            x_init = self._sample_point()
        
        # Evaluate first point
        params = self._denormalize(x_init)
        value = -objective_func(params)  # Negate for maximization
        self.evaluated_points.append(x_init)
        self.evaluated_values.append(value)
        history.append((params.copy(), -value))  # Store original value
        
        if self.verbose:
            print(f"Initial evaluation: params={params}, value={-value:.6f}")
        
        # Main optimization loop
        for iteration in range(1, self.max_iters):
            accepted = False
            local_rejections = 0
            
            while not accepted:
                # Sample candidate point
                x_candidate = self._sample_point()
                
                # Check acceptance condition
                if self._acceptance_condition(x_candidate):
                    # Evaluate the point
                    params = self._denormalize(x_candidate)
                    value = -objective_func(params)  # Negate for maximization
                    
                    self.evaluated_points.append(x_candidate)
                    self.evaluated_values.append(value)
                    history.append((params.copy(), -value))  # Store original value
                    
                    if self.verbose:
                        print(f"Iter {iteration}: params={params}, value={-value:.6f}, "
                              f"epsilon={self.epsilon:.4f}")
                    
                    # Reset rejection count and grow epsilon
                    self.prev_rejection_count = self.rejection_count
                    self.rejection_count = 0
                    self.epsilon *= self.tau
                    accepted = True
                    
                else:
                    # Point rejected
                    self.rejection_count += 1
                    local_rejections += 1
                    
                    # Check growth condition
                    if self.rejection_count - self.prev_rejection_count > self.C:
                        self.epsilon *= self.tau
                        self.prev_rejection_count = 0
                        self.rejection_count = 0
                        if self.verbose:
                            print(f"  Growing epsilon to {self.epsilon:.4f} due to rejections")
        
        # Find best result
        best_idx = np.argmax(self.evaluated_values)
        best_params = self._denormalize(self.evaluated_points[best_idx])
        best_value = -self.evaluated_values[best_idx]  # Convert back to minimization
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'history': history,
            'n_evaluations': len(history)
        }


def optimize_two_channel_params(reconstruction_func: Callable,
                               data_dict: Dict,
                               param_bounds: Optional[Dict] = None,
                               max_evals: int = 30,
                               verbose: bool = True) -> Dict:
    """
    Convenience function to optimize two-channel L1-DTV parameters using ECP.
    
    Args:
        reconstruction_func: Function that takes (params_dict, data_dict) and returns RMSE
        data_dict: Dictionary containing phantom data, sinograms, etc.
        param_bounds: Dictionary of parameter bounds (optional)
        max_evals: Maximum number of reconstructions to run
        verbose: Print progress
        
    Returns:
        Optimization results dictionary
    """
    
    # Default parameter bounds if not provided
    if param_bounds is None:
        param_bounds = {
            'cutoff_lo': (2.0, 16.0),      # Frequency separation
            'sigma_scale': (1.0, 8.0),      # Step size scaling
            'eps_ratio': (0.5, 2.0)         # eps_lo/eps_hi ratio
        }
    
    # Extract bounds in order
    param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in param_names]
    
    # Create objective function wrapper
    def objective(params):
        """Wrapper that converts array to dict and calls reconstruction"""
        param_dict = {name: params[i] for i, name in enumerate(param_names)}
        
        # Run reconstruction with these parameters
        rmse = reconstruction_func(param_dict, data_dict)
        return rmse  # We want to minimize RMSE
    
    # Initialize and run optimizer
    optimizer = ECPOptimizer(
        bounds=bounds,
        epsilon_init=0.01,
        tau=1.05,
        C=50,
        max_iters=max_evals,
        verbose=verbose
    )
    
    results = optimizer.optimize(objective)
    
    # Convert best params back to dictionary
    results['best_params_dict'] = {
        name: results['best_params'][i] 
        for i, name in enumerate(param_names)
    }
    
    if verbose:
        print("\n" + "="*50)
        print("OPTIMIZATION COMPLETE")
        print("="*50)
        print(f"Best parameters found:")
        for name, value in results['best_params_dict'].items():
            print(f"  {name}: {value:.4f}")
        print(f"Best RMSE: {results['best_value']:.6f}")
        print(f"Function evaluations: {results['n_evaluations']}")
    
    return results


# Example usage function for testing
def example_reconstruction_wrapper(param_dict: Dict, data_dict: Dict) -> float:
    """
    Example wrapper showing how to use optimized parameters in reconstruction.
    
    This would be replaced with actual two-channel reconstruction code.
    """
    # Extract parameters
    cutoff_lo = param_dict.get('cutoff_lo', 8.0)
    sigma_scale = param_dict.get('sigma_scale', 4.0)
    eps_ratio = param_dict.get('eps_ratio', 1.25)
    
    # In real usage, this would run the two-channel reconstruction
    # and return the RMSE
    # For now, return a dummy value
    dummy_rmse = np.random.uniform(0.01, 0.02)
    return dummy_rmse


if __name__ == "__main__":
    # Test the optimizer with a simple function
    print("Testing ECP Optimizer with Rosenbrock function...")
    
    def rosenbrock(params):
        """Rosenbrock test function (minimum at [1, 1])"""
        x, y = params
        return (1 - x)**2 + 100*(y - x**2)**2
    
    optimizer = ECPOptimizer(
        bounds=[(-2, 2), (-2, 2)],
        epsilon_init=0.01,
        tau=1.1,
        C=100,
        max_iters=50,
        verbose=True
    )
    
    results = optimizer.optimize(rosenbrock)
    print(f"\nOptimum found at: {results['best_params']}")
    print(f"Optimum value: {results['best_value']:.6f}")
    print(f"True optimum: [1.0, 1.0] with value 0.0")