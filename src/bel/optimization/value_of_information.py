"""Value of Information-based optimization for exploration well placement."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial.distance import cdist

from ..geological_model import GaussianProcessGeology
from ..production_model import ProductionPredictor
from ..economic_model import EconomicAssessment


class ValueOfInformation:
    """
    Determines optimal exploration well locations using value of information calculations.
    
    This class handles the calculation of value of information (VOI) surfaces
    and selection of optimal exploration well locations to maximize information gain.
    """
    
    def __init__(
        self,
        geological_model: GaussianProcessGeology,
        production_model: ProductionPredictor,
        economic_model: EconomicAssessment,
        exploration_cost: float = 10.0,  # $M per exploration well
        n_realizations: int = 50,
        n_monte_carlo: int = 100,
        random_state: Optional[int] = None
    ):
        """
        Initialize the value of information optimizer.
        
        Args:
            geological_model: Geological model for property distributions.
            production_model: Production prediction model.
            economic_model: Economic assessment model.
            exploration_cost: Cost of drilling an exploration well in millions of dollars.
            n_realizations: Number of geological realizations to generate.
            n_monte_carlo: Number of Monte Carlo simulations for economic assessment.
            random_state: Random seed for reproducibility.
        """
        self.geological_model = geological_model
        self.production_model = production_model
        self.economic_model = economic_model
        self.exploration_cost = exploration_cost
        self.n_realizations = n_realizations
        self.n_monte_carlo = n_monte_carlo
        self.random_state = random_state
        
        # Grid from geological model
        self.grid_points = self.geological_model.grid_points
        self.X = self.geological_model.X
        self.Y = self.geological_model.Y
        self.grid_size = self.geological_model.grid_size
        
        # Store results
        self.voi_surface = None
        self.uncertainty_reduction = None
        self.current_npv_dist = None
    
    def calculate_voi_surface(
        self,
        development_wells_per_realization: int = 100,
        development_years: int = 10,
        parallel: bool = False
    ) -> np.ndarray:
        """
        Calculate the value of information surface across the basin.
        
        Args:
            development_wells_per_realization: Number of development wells per realization.
            development_years: Development horizon in years.
            parallel: Whether to use parallel processing.
            
        Returns:
            2D array of VOI values across the grid.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize VOI surface
        nx, ny = self.grid_size
        voi_surface = np.zeros((ny, nx))
        uncertainty_reduction = np.zeros((ny, nx))
        
        # Generate time points for production forecasting
        time_points_years = np.linspace(0, development_years, development_years * 12 + 1)
        
        # Calculate current NPV distribution without additional information
        current_realizations = self.geological_model.sample_realizations(self.n_realizations)
        current_npv_values = np.zeros(self.n_realizations)
        
        # Iterate through realizations
        for i in range(self.n_realizations):
            # Extract geological properties for this realization
            properties = {prop: current_realizations[prop][i].flatten() for prop in current_realizations}
            
            # Predict production
            production_results = self.production_model.predict_full_well_performance(
                properties,
                time_points_years=time_points_years,
                development_years=development_years
            )
            
            # Assess economics
            economic_results = self.economic_model.assess_profitability_distribution(
                production_results['production_profiles'],
                time_points_years,
                development_wells_per_realization,
                num_simulations=self.n_monte_carlo // 10  # Reduce for computational efficiency
            )
            
            current_npv_values[i] = economic_results['npv_mean']
        
        self.current_npv_dist = current_npv_values
        current_npv_std = np.std(current_npv_values)
        
        # Iterate through potential exploration well locations
        for i in range(ny):
            for j in range(nx):
                # Get coordinates of this grid point
                x, y = self.X[i, j], self.Y[i, j]
                location = np.array([[x, y]])
                
                # Estimate uncertainty reduction from drilling at this location
                uncertainty_red = self._estimate_uncertainty_reduction(location)
                uncertainty_reduction[i, j] = uncertainty_red
                
                # Estimate improved NPV distribution with new information
                improved_npv_std = current_npv_std * (1 - uncertainty_red)
                
                # Simple VOI calculation: reduction in standard deviation minus exploration cost
                voi = (current_npv_std - improved_npv_std) - self.exploration_cost
                voi_surface[i, j] = max(0, voi)  # VOI can't be negative
        
        self.voi_surface = voi_surface
        self.uncertainty_reduction = uncertainty_reduction
        
        return voi_surface
    
    def _estimate_uncertainty_reduction(self, location: np.ndarray) -> float:
        """
        Estimate the uncertainty reduction from drilling at a specific location.
        
        Args:
            location: Coordinates of the potential exploration well.
            
        Returns:
            Estimated uncertainty reduction as a fraction (0 to 1).
        """
        # Calculate distance to all existing well data points
        existing_points = []
        for prop in self.geological_model.properties:
            if self.geological_model.well_data[prop]:
                existing_points.extend([loc for loc, _ in self.geological_model.well_data[prop]])
        
        # If no existing wells, this would provide maximum information
        if not existing_points:
            return 0.8  # Arbitrary high value for first well
        
        existing_points = np.array(existing_points)
        
        # Calculate minimum distance to existing wells
        distances = cdist(location, existing_points)
        min_distance = np.min(distances)
        
        # Calculate average length scale
        avg_length_scale = np.mean([
            np.mean(self.geological_model.length_scales[prop]) 
            if isinstance(self.geological_model.length_scales[prop], (list, tuple, np.ndarray))
            else self.geological_model.length_scales[prop]
            for prop in self.geological_model.properties
        ])
        
        # Estimate uncertainty reduction based on distance
        # Close to existing wells: less reduction, far from wells: more reduction
        uncertainty_reduction = 1.0 - np.exp(-min_distance / (2 * avg_length_scale))
        
        # Scale to reasonable range
        return min(0.8, max(0.1, uncertainty_reduction))
    
    def select_next_well_location(self) -> Tuple[float, float]:
        """
        Select the optimal location for the next exploration well.
        
        Returns:
            Tuple of (x, y) coordinates for the optimal location.
        """
        if self.voi_surface is None:
            raise ValueError("VOI surface has not been calculated. Call calculate_voi_surface() first.")
        
        # Find location with maximum VOI
        ny, nx = self.voi_surface.shape
        max_idx = np.argmax(self.voi_surface.flatten())
        i, j = max_idx // nx, max_idx % nx
        
        # Get coordinates
        x, y = self.X[i, j], self.Y[i, j]
        
        return (x, y)
    
    def simulate_information_value(
        self,
        location: Tuple[float, float],
        true_model: GaussianProcessGeology,
        development_wells_per_realization: int = 100,
        development_years: int = 10
    ) -> Dict[str, Any]:
        """
        Simulate the value of information from drilling at a specific location.
        
        Args:
            location: Coordinates (x, y) of the exploration well.
            true_model: True geological model to sample from.
            development_wells_per_realization: Number of development wells per realization.
            development_years: Development horizon in years.
            
        Returns:
            Dictionary with information value assessment results.
        """
        x, y = location
        location_array = np.array([[x, y]])
        
        # Sample true values from the true model
        true_realizations = true_model.sample_realizations(1)
        true_values = {}
        
        for prop in true_model.properties:
            # Get the value at the specified location
            i = np.argmin(np.sum((true_model.grid_points - location_array)**2, axis=1))
            true_values[prop] = true_realizations[prop][0].flatten()[i]
        
        # Update geological model with new data
        updated_model = self.geological_model
        updated_model.update_with_well_data(location_array, true_values)
        
        # Generate time points for production forecasting
        time_points_years = np.linspace(0, development_years, development_years * 12 + 1)
        
        # Calculate NPV distribution with updated model
        updated_realizations = updated_model.sample_realizations(self.n_realizations)
        updated_npv_values = np.zeros(self.n_realizations)
        
        # Iterate through realizations
        for i in range(self.n_realizations):
            # Extract geological properties for this realization
            properties = {prop: updated_realizations[prop][i].flatten() for prop in updated_realizations}
            
            # Predict production
            production_results = self.production_model.predict_full_well_performance(
                properties,
                time_points_years=time_points_years,
                development_years=development_years
            )
            
            # Assess economics
            economic_results = self.economic_model.assess_profitability_distribution(
                production_results['production_profiles'],
                time_points_years,
                development_wells_per_realization,
                num_simulations=self.n_monte_carlo // 10
            )
            
            updated_npv_values[i] = economic_results['npv_mean']
        
        # Calculate value of information
        prior_npv_mean = np.mean(self.current_npv_dist)
        prior_npv_std = np.std(self.current_npv_dist)
        posterior_npv_mean = np.mean(updated_npv_values)
        posterior_npv_std = np.std(updated_npv_values)
        
        # Probability of meeting target profit
        prior_prob = norm.sf((self.economic_model.target_profit - prior_npv_mean) / prior_npv_std)
        posterior_prob = norm.sf((self.economic_model.target_profit - posterior_npv_mean) / posterior_npv_std)
        
        # Information value metrics
        uncertainty_reduction = 1 - (posterior_npv_std / prior_npv_std)
        confidence_improvement = posterior_prob - prior_prob
        expected_value_improvement = posterior_npv_mean - prior_npv_mean
        net_value = expected_value_improvement - self.exploration_cost
        
        return {
            'location': location,
            'true_values': true_values,
            'prior_npv_mean': prior_npv_mean,
            'prior_npv_std': prior_npv_std,
            'posterior_npv_mean': posterior_npv_mean,
            'posterior_npv_std': posterior_npv_std,
            'uncertainty_reduction': uncertainty_reduction,
            'confidence_improvement': confidence_improvement,
            'expected_value_improvement': expected_value_improvement,
            'net_value': net_value,
            'prior_prob_target': prior_prob,
            'posterior_prob_target': posterior_prob
        }
    
    def plot_voi_surface(
        self,
        ax=None,
        title: str = "Value of Information Surface",
        colorbar_label: str = "VOI ($M)",
        cmap: str = 'viridis',
        show_wells: bool = True
    ):
        """
        Plot the value of information surface.
        
        Args:
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            title: Title for the plot.
            colorbar_label: Label for the colorbar.
            cmap: Colormap to use for the plot.
            show_wells: Whether to show existing well locations.
            
        Returns:
            The matplotlib axis object.
        """
        if self.voi_surface is None:
            raise ValueError("VOI surface has not been calculated. Call calculate_voi_surface() first.")
            
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the VOI surface
        im = ax.contourf(self.X, self.Y, self.voi_surface, cmap=cmap)
        
        # Add well locations if requested
        if show_wells:
            existing_points = []
            for prop in self.geological_model.properties:
                if self.geological_model.well_data[prop]:
                    existing_points.extend([loc for loc, _ in self.geological_model.well_data[prop]])
            
            if existing_points:
                existing_points = np.array(existing_points)
                ax.scatter(
                    existing_points[:, 0], 
                    existing_points[:, 1], 
                    c='white', 
                    edgecolor='black', 
                    s=100, 
                    zorder=10,
                    label='Existing Wells'
                )
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=colorbar_label)
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Add legend if wells are shown
        if show_wells and existing_points:
            ax.legend(loc='best')
        
        return ax
    
    def plot_uncertainty_reduction(
        self,
        ax=None,
        title: str = "Uncertainty Reduction Surface",
        colorbar_label: str = "Uncertainty Reduction",
        cmap: str = 'plasma',
        show_wells: bool = True
    ):
        """
        Plot the uncertainty reduction surface.
        
        Args:
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            title: Title for the plot.
            colorbar_label: Label for the colorbar.
            cmap: Colormap to use for the plot.
            show_wells: Whether to show existing well locations.
            
        Returns:
            The matplotlib axis object.
        """
        if self.uncertainty_reduction is None:
            raise ValueError("Uncertainty reduction has not been calculated. Call calculate_voi_surface() first.")
            
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the uncertainty reduction surface
        im = ax.contourf(self.X, self.Y, self.uncertainty_reduction, cmap=cmap)
        
        # Add well locations if requested
        if show_wells:
            existing_points = []
            for prop in self.geological_model.properties:
                if self.geological_model.well_data[prop]:
                    existing_points.extend([loc for loc, _ in self.geological_model.well_data[prop]])
            
            if existing_points:
                existing_points = np.array(existing_points)
                ax.scatter(
                    existing_points[:, 0], 
                    existing_points[:, 1], 
                    c='white', 
                    edgecolor='black', 
                    s=100, 
                    zorder=10,
                    label='Existing Wells'
                )
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=colorbar_label)
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Add legend if wells are shown
        if show_wells and existing_points:
            ax.legend(loc='best')
        
        return ax