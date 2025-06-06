"""Gaussian Process-based geological property modeling."""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import matplotlib.pyplot as plt

from ..utils.math_utils import MathUtils


class GaussianProcessGeology:
    """
    Manages spatial geological property distributions and uncertainty using Gaussian processes.
    
    This class handles the modeling of geological properties across a basin using
    Gaussian processes, allowing for updating beliefs based on new well data.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        properties: List[str],
        length_scales: Dict[str, Union[float, Tuple[float, float]]],
        property_ranges: Dict[str, Tuple[float, float]],
        kernel_type: str = "exponential",
        random_state: Optional[int] = None
    ):
        """
        Initialize the geological model.
        
        Args:
            grid_size: Tuple (nx, ny) specifying the grid dimensions.
            x_range: Tuple (x_min, x_max) specifying the x-coordinate range.
            y_range: Tuple (y_min, y_max) specifying the y-coordinate range.
            properties: List of geological property names to model.
            length_scales: Dictionary mapping property names to correlation lengths.
            property_ranges: Dictionary mapping property names to (min, max) value ranges.
            kernel_type: Type of kernel for the GP: "exponential", "squared_exponential", or "matern".
            random_state: Random seed for reproducibility.
        """
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        self.properties = properties
        self.length_scales = length_scales
        self.property_ranges = property_ranges
        self.kernel_type = kernel_type
        self.random_state = random_state
        
        # Initialize grid
        self.initialize_grid()
        
        # Initialize property distributions
        self.property_distributions = {}
        self.well_data = {prop: [] for prop in properties}
        self.initialize_prior_distributions()
    
    def initialize_grid(self):
        """Create the spatial grid for the geological model."""
        nx, ny = self.grid_size
        x = np.linspace(self.x_range[0], self.x_range[1], nx)
        y = np.linspace(self.y_range[0], self.y_range[1], ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Create flattened grid points for GP calculations
        self.grid_points = np.vstack([self.X.flatten(), self.Y.flatten()]).T
        self.n_points = len(self.grid_points)
    
    def initialize_prior_distributions(self):
        """Initialize prior distributions for all geological properties."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        for prop in self.properties:
            # Get length scale for this property
            length_scale = self.length_scales[prop]
            
            # Calculate prior covariance matrix
            K = MathUtils.spatial_correlation_function(
                self.grid_points, 
                self.grid_points, 
                length_scale=length_scale,
                kernel_type=self.kernel_type
            )
            
            # Add small jitter to diagonal for numerical stability
            K += np.eye(self.n_points) * 1e-8
            
            # Generate a random mean for the property
            mean = np.random.uniform(0.4, 0.6, self.n_points)
            
            # Store the distribution
            self.property_distributions[prop] = {
                'mean': mean,
                'cov': K,
                'L': cholesky(K, lower=True),
                'observations': [],
                'observation_points': []
            }
    
    def update_with_well_data(
        self, 
        well_locations: np.ndarray, 
        property_values: Dict[str, np.ndarray],
        measurement_noise: float = 1e-6
    ):
        """
        Update geological model with new well data.
        
        Args:
            well_locations: Array of shape (n_wells, 2) with well x, y coordinates.
            property_values: Dictionary mapping property names to arrays of observed values.
            measurement_noise: Variance of measurement noise.
        """
        for prop in self.properties:
            if prop not in property_values:
                continue
                
            # Extract observed values for this property
            y_obs = property_values[prop]
            
            # Add to observations
            self.property_distributions[prop]['observations'].extend(y_obs)
            self.property_distributions[prop]['observation_points'].extend(well_locations)
            
            # Stack all observation points and values
            X_obs = np.array(self.property_distributions[prop]['observation_points'])
            y_obs = np.array(self.property_distributions[prop]['observations'])
            
            # Calculate covariance between grid points and observation points
            length_scale = self.length_scales[prop]
            K_s = MathUtils.spatial_correlation_function(
                self.grid_points, 
                X_obs, 
                length_scale=length_scale,
                kernel_type=self.kernel_type
            )
            
            # Calculate covariance among observation points
            K_obs = MathUtils.spatial_correlation_function(
                X_obs, 
                X_obs, 
                length_scale=length_scale,
                kernel_type=self.kernel_type
            )
            K_obs += np.eye(len(X_obs)) * measurement_noise  # Add noise to diagonal
            
            # Calculate posterior mean
            L_obs = cholesky(K_obs, lower=True)
            alpha = cho_solve((L_obs, True), y_obs)
            self.property_distributions[prop]['mean'] = K_s @ alpha
            
            # Calculate posterior covariance
            v = solve_triangular(L_obs, K_s.T, lower=True)
            self.property_distributions[prop]['cov'] -= v.T @ v
            
            # Update Cholesky factor
            self.property_distributions[prop]['L'] = cholesky(
                self.property_distributions[prop]['cov'], 
                lower=True
            )
            
            # Add data to well_data collection
            for loc, val in zip(well_locations, y_obs):
                self.well_data[prop].append((loc, val))
    
    def sample_realizations(self, n_samples: int = 1) -> Dict[str, np.ndarray]:
        """
        Generate random realizations from the current geological model.
        
        Args:
            n_samples: Number of realizations to generate.
            
        Returns:
            Dictionary mapping property names to arrays of shape (n_samples, nx, ny).
        """
        samples = {}
        
        for prop in self.properties:
            # Get distribution parameters
            mean = self.property_distributions[prop]['mean']
            L = self.property_distributions[prop]['L']
            
            # Generate samples from multivariate normal using Cholesky factor
            z = np.random.normal(size=(n_samples, self.n_points))
            property_samples = mean + np.einsum('ij,kj->ki', L, z)
            
            # Apply min-max scaling to the property range
            min_val, max_val = self.property_ranges[prop]
            for i in range(n_samples):
                property_samples[i] = MathUtils.normalize_grid(
                    property_samples[i], 
                    min_val, 
                    max_val
                )
            
            # Reshape to grid dimensions
            nx, ny = self.grid_size
            samples[prop] = property_samples.reshape(n_samples, ny, nx)
        
        return samples
    
    def calculate_uncertainty(self) -> Dict[str, np.ndarray]:
        """
        Calculate uncertainty maps for all properties.
        
        Returns:
            Dictionary mapping property names to uncertainty grids.
        """
        uncertainty = {}
        
        for prop in self.properties:
            # Extract diagonal of covariance matrix as variance
            variance = np.diag(self.property_distributions[prop]['cov'])
            
            # Reshape to grid
            nx, ny = self.grid_size
            uncertainty[prop] = np.sqrt(variance).reshape(ny, nx)
        
        return uncertainty
    
    def get_property_mean(self, property_name: str) -> np.ndarray:
        """
        Get the mean map for a specific property.
        
        Args:
            property_name: Name of the property.
            
        Returns:
            2D array with the mean value of the property.
        """
        if property_name not in self.properties:
            raise ValueError(f"Property {property_name} not found in model")
            
        mean = self.property_distributions[property_name]['mean']
        nx, ny = self.grid_size
        
        # Reshape and normalize to property range
        mean_grid = mean.reshape(ny, nx)
        min_val, max_val = self.property_ranges[property_name]
        
        return MathUtils.normalize_grid(mean_grid, min_val, max_val)
    
    def plot_property(
        self, 
        property_name: str, 
        ax=None, 
        show_wells: bool = True,
        title: Optional[str] = None,
        colorbar_label: Optional[str] = None,
        cmap: str = 'viridis'
    ):
        """
        Plot the mean map for a specific property.
        
        Args:
            property_name: Name of the property to plot.
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            show_wells: Whether to show well locations on the plot.
            title: Title for the plot. If None, uses the property name.
            colorbar_label: Label for the colorbar. If None, uses the property name.
            cmap: Colormap to use for the plot.
            
        Returns:
            The matplotlib axis object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        # Get the mean property map
        mean_map = self.get_property_mean(property_name)
        
        # Plot the mean map
        im = ax.contourf(self.X, self.Y, mean_map, cmap=cmap)
        
        # Add well locations if requested
        if show_wells and self.well_data[property_name]:
            well_locs = np.array([loc for loc, _ in self.well_data[property_name]])
            well_vals = np.array([val for _, val in self.well_data[property_name]])
            
            # Plot wells as scatter points
            scatter = ax.scatter(
                well_locs[:, 0], 
                well_locs[:, 1], 
                c=well_vals, 
                cmap=cmap,
                edgecolor='black', 
                s=100, 
                zorder=10
            )
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=colorbar_label or property_name)
        
        # Add title and labels
        ax.set_title(title or f"{property_name} Map")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        return ax
    
    def plot_uncertainty(
        self, 
        property_name: str, 
        ax=None, 
        show_wells: bool = True,
        title: Optional[str] = None,
        colorbar_label: Optional[str] = None,
        cmap: str = 'plasma'
    ):
        """
        Plot the uncertainty map for a specific property.
        
        Args:
            property_name: Name of the property to plot.
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            show_wells: Whether to show well locations on the plot.
            title: Title for the plot. If None, uses the property name.
            colorbar_label: Label for the colorbar. If None, uses the property name uncertainty.
            cmap: Colormap to use for the plot.
            
        Returns:
            The matplotlib axis object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        # Get the uncertainty map
        uncertainty = self.calculate_uncertainty()[property_name]
        
        # Plot the uncertainty map
        im = ax.contourf(self.X, self.Y, uncertainty, cmap=cmap)
        
        # Add well locations if requested
        if show_wells and self.well_data[property_name]:
            well_locs = np.array([loc for loc, _ in self.well_data[property_name]])
            
            # Plot wells as scatter points
            ax.scatter(
                well_locs[:, 0], 
                well_locs[:, 1], 
                c='white', 
                edgecolor='black', 
                s=100, 
                zorder=10
            )
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=colorbar_label or f"{property_name} Uncertainty")
        
        # Add title and labels
        ax.set_title(title or f"{property_name} Uncertainty Map")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        return ax