"""Mathematical utility functions for geological modeling and optimization."""

import numpy as np
from typing import Tuple, List, Callable, Optional, Union, Dict
from scipy.spatial.distance import cdist


class MathUtils:
    """Mathematical and statistical utility functions for the BEL package."""
    
    @staticmethod
    def spatial_correlation_function(
        points1: np.ndarray, 
        points2: np.ndarray, 
        length_scale: Union[float, np.ndarray],
        variance: float = 1.0,
        kernel_type: str = "exponential"
    ) -> np.ndarray:
        """
        Calculate spatial correlation between two sets of points.
        
        Args:
            points1: Array of shape (n, 2) containing x, y coordinates.
            points2: Array of shape (m, 2) containing x, y coordinates.
            length_scale: Correlation length for the kernel. Can be a scalar or array with 2 elements for anisotropic kernels.
            variance: Signal variance parameter.
            kernel_type: Type of kernel to use: "exponential", "squared_exponential", or "matern".
            
        Returns:
            Correlation matrix of shape (n, m).
        """
        if isinstance(length_scale, (int, float)):
            length_scale = np.array([length_scale, length_scale])
        
        # Scale the coordinates by the length scale for anisotropic kernels
        scaled_points1 = points1 / length_scale
        scaled_points2 = points2 / length_scale
        
        # Calculate pairwise distances
        distances = cdist(scaled_points1, scaled_points2, metric='euclidean')
        
        # Apply kernel function
        if kernel_type == "exponential":
            # k(x_i, x_j) = σ² * exp(-|x_i - x_j| / l)
            return variance * np.exp(-distances)
        elif kernel_type == "squared_exponential":
            # k(x_i, x_j) = σ² * exp(-0.5 * |x_i - x_j|² / l²)
            return variance * np.exp(-0.5 * distances**2)
        elif kernel_type == "matern":
            # Matern 3/2 kernel
            # k(x_i, x_j) = σ² * (1 + √3|x_i - x_j|/l) * exp(-√3|x_i - x_j|/l)
            sqrt3_d = np.sqrt(3) * distances
            return variance * (1 + sqrt3_d) * np.exp(-sqrt3_d)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    @staticmethod
    def monte_carlo_sampling(
        mean: np.ndarray, 
        cov: np.ndarray, 
        n_samples: int = 100,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Monte Carlo samples from a multivariate normal distribution.
        
        Args:
            mean: Mean vector of the distribution.
            cov: Covariance matrix of the distribution.
            n_samples: Number of samples to generate.
            random_state: Random seed for reproducibility.
            
        Returns:
            Array of shape (n_samples, len(mean)) containing the samples.
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        return np.random.multivariate_normal(mean, cov, size=n_samples)
    
    @staticmethod
    def statistical_measures(
        samples: np.ndarray, 
        percentiles: List[float] = [5, 50, 95]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate statistical measures from samples.
        
        Args:
            samples: Array of shape (n_samples, ...) containing the samples.
            percentiles: List of percentiles to calculate.
            
        Returns:
            Dictionary with keys 'mean', 'std', 'percentiles'.
        """
        results = {
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0),
            'percentiles': {p: np.percentile(samples, p, axis=0) for p in percentiles}
        }
        return results
    
    @staticmethod
    def normalize_grid(
        grid: np.ndarray, 
        min_val: float, 
        max_val: float
    ) -> np.ndarray:
        """
        Normalize a grid to a specified range.
        
        Args:
            grid: Input grid to normalize.
            min_val: Minimum value in the output range.
            max_val: Maximum value in the output range.
            
        Returns:
            Normalized grid with values between min_val and max_val.
        """
        grid_min = np.min(grid)
        grid_max = np.max(grid)
        
        # Avoid division by zero
        if grid_min == grid_max:
            return np.ones_like(grid) * min_val
        
        normalized = (grid - grid_min) / (grid_max - grid_min)
        return normalized * (max_val - min_val) + min_val