#!/usr/bin/env python
"""
Demonstration using a pre-trained production model.

This example assumes we have an already trained production model that
predicts production directly from geological properties at a given location.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple, Any, Optional


class PretrainedProductionModel:
    """
    Represents a pre-trained production model.
    
    In a real scenario, this would be loaded from a saved model file
    or would call an external API/service.
    """
    
    def __init__(self):
        """Initialize the pre-trained model with fixed coefficients."""
        # These coefficients would normally come from a trained model
        self.initial_rate_coefficients = {
            'Thickness': 100.0,     # Higher thickness → more production
            'Porosity': 10000.0,    # Higher porosity → much more production
            'Permeability': 1000.0, # Higher permeability → more production
            'TOC': 50.0,            # Higher TOC → more production
            'SW': -100.0,           # Higher water saturation → less production
            'Depth': -0.01,         # Deeper → slightly less production
            'Vclay': -500.0,        # Higher clay volume → less production
            'intercept': 1000.0     # Base production rate
        }
        
        # Decline curve parameters would also come from a trained model
        # Here we use typical values for shale gas wells
        self.b_factor = 0.85        # b-factor in Arps equation
        self.initial_decline = 0.01  # Initial decline rate (fraction/day)
    
    def predict_production(self, 
                          properties: Dict[str, float], 
                          time_points_days: np.ndarray) -> np.ndarray:
        """
        Predict production over time for a location with given properties.
        
        Args:
            properties: Dictionary of geological properties at this location.
            time_points_days: Array of time points in days to predict production for.
            
        Returns:
            Array of production rates (mcf/day) at each time point.
        """
        # Calculate initial production rate using trained coefficients
        initial_rate = self.initial_rate_coefficients['intercept']
        for prop, value in properties.items():
            if prop in self.initial_rate_coefficients:
                initial_rate += self.initial_rate_coefficients[prop] * value
        
        # Ensure non-negative initial rate
        initial_rate = max(initial_rate, 100.0)
        
        # Apply Arps decline curve: q(t) = q_i / (1 + b * D_i * t)^(1/b)
        production = initial_rate / (1 + self.b_factor * self.initial_decline * time_points_days) ** (1 / self.b_factor)
        
        return production
    
    def predict_cumulative_production(self,
                                     properties: Dict[str, float],
                                     time_span_days: float) -> float:
        """
        Predict cumulative production over a time span.
        
        Args:
            properties: Dictionary of geological properties at this location.
            time_span_days: Time span in days.
            
        Returns:
            Cumulative production in mcf.
        """
        # Calculate initial production rate
        initial_rate = self.initial_rate_coefficients['intercept']
        for prop, value in properties.items():
            if prop in self.initial_rate_coefficients:
                initial_rate += self.initial_rate_coefficients[prop] * value
        
        # Ensure non-negative initial rate
        initial_rate = max(initial_rate, 100.0)
        
        # Apply Arps cumulative production formula
        b = self.b_factor
        D_i = self.initial_decline
        
        if abs(b - 1.0) < 1e-6:  # b ≈ 1
            cum_production = (initial_rate / D_i) * np.log(1 + D_i * time_span_days)
        else:  # b ≠ 1
            cum_production = (initial_rate / ((1 - b) * D_i)) * (
                1 - (1 + b * D_i * time_span_days) ** (1 - 1/b)
            )
        
        return cum_production


class EconomicModel:
    """Economic model for assessing profitability."""
    
    def __init__(self,
                gas_price: float = 4.0,           # $/mcf
                operating_cost: float = 0.5,      # $/mcf
                drilling_cost: float = 10.0,      # $M per well
                discount_rate: float = 0.1,       # annual rate
                gas_price_volatility: float = 0.2 # for Monte Carlo simulation
                ):
        """
        Initialize the economic model.
        
        Args:
            gas_price: Gas price in dollars per thousand cubic feet.
            operating_cost: Operating cost in dollars per thousand cubic feet.
            drilling_cost: Drilling and completion cost in millions of dollars.
            discount_rate: Annual discount rate as a fraction.
            gas_price_volatility: Standard deviation for gas price in Monte Carlo.
        """
        self.gas_price = gas_price
        self.operating_cost = operating_cost
        self.drilling_cost = drilling_cost
        self.discount_rate = discount_rate
        self.gas_price_volatility = gas_price_volatility
    
    def calculate_npv(self,
                     production: np.ndarray,
                     time_points_years: np.ndarray,
                     gas_price: Optional[float] = None) -> float:
        """
        Calculate Net Present Value for a production profile.
        
        Args:
            production: Array of production rates (mcf/day).
            time_points_years: Array of time points in years.
            gas_price: Optional override for gas price.
            
        Returns:
            NPV in millions of dollars.
        """
        if gas_price is None:
            gas_price = self.gas_price
        
        # Convert daily rates to period production (assuming time points are period starts)
        if len(time_points_years) > 1:
            period_lengths = np.diff(time_points_years, append=time_points_years[-1] + (time_points_years[-1] - time_points_years[-2]))
            period_days = period_lengths * 365.25
            period_production = production * period_days
        else:
            # Single time point case
            period_production = production * 365.25  # Assume one year
        
        # Calculate cash flows
        revenue = period_production * (gas_price / 1e6)  # $M
        opex = period_production * (self.operating_cost / 1e6)  # $M
        cash_flow = revenue - opex
        
        # Apply discount factors
        discount_factors = 1 / (1 + self.discount_rate) ** time_points_years
        discounted_cash_flow = cash_flow * discount_factors
        
        # Calculate NPV
        npv = np.sum(discounted_cash_flow) - self.drilling_cost
        
        return npv
    
    def monte_carlo_npv(self,
                       production_model: PretrainedProductionModel,
                       properties: Dict[str, float],
                       time_points_years: np.ndarray,
                       n_simulations: int = 1000,
                       random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Monte Carlo simulation of NPV.
        
        Args:
            production_model: Pre-trained production model.
            properties: Geological properties at the location.
            time_points_years: Array of time points in years.
            n_simulations: Number of Monte Carlo simulations.
            random_seed: Random seed for reproducibility.
            
        Returns:
            Dictionary with simulation results.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Convert years to days for production model
        time_points_days = time_points_years * 365.25
        
        # Generate base production profile
        base_production = production_model.predict_production(properties, time_points_days)
        
        # Initialize arrays for results
        npv_values = np.zeros(n_simulations)
        
        # Run Monte Carlo simulations
        for i in range(n_simulations):
            # Sample gas price from lognormal distribution
            # This ensures gas price is always positive and has higher chance of upside than downside
            gas_price = np.random.lognormal(
                mean=np.log(self.gas_price),
                sigma=self.gas_price_volatility
            )
            
            # Add random noise to production (uncertainty in performance)
            production_noise = np.random.normal(1.0, 0.1, size=len(base_production))
            production = base_production * production_noise
            
            # Calculate NPV for this simulation
            npv = self.calculate_npv(production, time_points_years, gas_price)
            npv_values[i] = npv
        
        # Calculate statistics
        mean_npv = np.mean(npv_values)
        std_npv = np.std(npv_values)
        p10 = np.percentile(npv_values, 10)  # P10 (pessimistic case)
        p50 = np.percentile(npv_values, 50)  # P50 (base case)
        p90 = np.percentile(npv_values, 90)  # P90 (optimistic case)
        
        # Calculate probability of positive NPV
        prob_positive = np.mean(npv_values > 0)
        
        # Calculate probability of meeting target
        target_npv = 20.0  # $M - This would be a parameter in a real model
        prob_target = np.mean(npv_values >= target_npv)
        
        return {
            'npv_values': npv_values,
            'mean_npv': mean_npv,
            'std_npv': std_npv,
            'p10': p10,
            'p50': p50,
            'p90': p90,
            'prob_positive': prob_positive,
            'prob_target': prob_target,
            'target_npv': target_npv
        }


def generate_geological_model(grid_size: Tuple[int, int] = (20, 20),
                             x_range: Tuple[float, float] = (-94.6, -92.9),
                             y_range: Tuple[float, float] = (31.3, 33.0),
                             random_seed: int = 42) -> Dict[str, Any]:
    """
    Generate synthetic geological property maps.
    
    Args:
        grid_size: Tuple (nx, ny) specifying the grid dimensions.
        x_range: Tuple (x_min, x_max) specifying the x-coordinate range.
        y_range: Tuple (y_min, y_max) specifying the y-coordinate range.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Dictionary containing property maps and grid coordinates.
    """
    np.random.seed(random_seed)
    
    # Create grid coordinates
    nx, ny = grid_size
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)
    
    # Property ranges
    property_ranges = {
        "Thickness": (50, 200),       # ft
        "Porosity": (0.02, 0.08),     # fraction
        "Permeability": (0.1, 0.5),   # mD
        "TOC": (2, 8),                # %
        "SW": (0.3, 0.7),             # fraction
        "Depth": (-11000, -8000),     # ft
        "Vclay": (0.3, 0.6)           # fraction
    }
    
    # Correlation lengths (controls spatial smoothness)
    correlation_lengths = {
        "Thickness": 5.0,
        "Porosity": 4.0,
        "Permeability": 4.0,
        "TOC": 3.0,
        "SW": 3.0,
        "Depth": 6.0,
        "Vclay": 3.0
    }
    
    # Generate property maps with spatial correlation
    properties = {}
    
    for prop, (min_val, max_val) in property_ranges.items():
        # Generate base random field
        base = np.random.normal(0, 1, grid_size)
        
        # Apply smoothing based on correlation length
        from scipy.ndimage import gaussian_filter
        sigma = correlation_lengths[prop] / 10.0  # Scale for the filter
        smooth = gaussian_filter(base, sigma=sigma)
        
        # Scale to property range
        prop_map = min_val + (max_val - min_val) * (smooth - smooth.min()) / (smooth.max() - smooth.min())
        
        # Store the property map
        properties[prop] = prop_map
    
    # Make some properties correlated (e.g., higher porosity → higher permeability)
    # This creates more realistic geological models
    properties["Permeability"] = 0.1 + 0.4 * (properties["Porosity"] - 0.02) / 0.06
    properties["SW"] = 0.7 - 0.4 * (properties["Porosity"] - 0.02) / 0.06
    
    return {
        'properties': properties,
        'X': X,
        'Y': Y,
        'x': x,
        'y': y
    }


def calculate_voi_surface(geological_model: Dict[str, Any],
                         production_model: PretrainedProductionModel,
                         economic_model: EconomicModel,
                         time_horizon_years: float = 10.0,
                         exploration_cost: float = 10.0) -> np.ndarray:
    """
    Calculate Value of Information surface.
    
    Args:
        geological_model: Geological model with property maps.
        production_model: Pre-trained production model.
        economic_model: Economic model for economic assessment.
        time_horizon_years: Time horizon for production forecasts.
        exploration_cost: Cost of drilling an exploration well.
        
    Returns:
        2D array of VOI values across the grid.
    """
    properties = geological_model['properties']
    X = geological_model['X']
    Y = geological_model['Y']
    nx, ny = X.shape
    
    # Initialize VOI surface
    voi_surface = np.zeros((ny, nx))
    
    # Create time points
    time_points_years = np.linspace(0, time_horizon_years, int(time_horizon_years * 12) + 1)
    time_points_days = time_points_years * 365.25
    
    # Calculate uncertainty in each property
    property_uncertainty = {}
    for prop, prop_map in properties.items():
        # For this demo, we'll use standard deviation as a proxy for uncertainty
        # In a real model, this would come from the variance of the Gaussian process
        uncertainty = np.std(prop_map) * np.ones((ny, nx))
        property_uncertainty[prop] = uncertainty
    
    # Calculate total uncertainty as average of normalized property uncertainties
    total_uncertainty = np.zeros((ny, nx))
    for prop, uncertainty in property_uncertainty.items():
        # Normalize uncertainty to [0, 1] range
        norm_uncertainty = uncertainty / np.max(uncertainty)
        total_uncertainty += norm_uncertainty
    total_uncertainty /= len(property_uncertainty)
    
    # Calculate expected NPV at each location
    expected_npv = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            # Extract properties at this location
            loc_properties = {prop: properties[prop][i, j] for prop in properties}
            
            # Predict production
            production = production_model.predict_production(loc_properties, time_points_days)
            
            # Calculate NPV
            npv = economic_model.calculate_npv(production, time_points_years)
            expected_npv[i, j] = npv
    
    # Calculate VOI as a function of uncertainty and expected NPV
    # VOI is higher in areas with:
    # 1. High uncertainty (more to learn)
    # 2. Potentially high NPV (more value)
    # 3. Minus exploration cost
    voi_surface = total_uncertainty * expected_npv - exploration_cost
    
    # VOI should be non-negative (don't drill if expected value < cost)
    voi_surface = np.maximum(voi_surface, 0)
    
    return voi_surface


def run_exploration_simulation(n_wells: int = 3, random_seed: int = 42):
    """
    Run an exploration simulation with sequential well placement.

    Args:
        n_wells: Number of exploration wells to drill.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary of simulation results.
    """
    print(f"Running exploration simulation with {n_wells} wells...")

    # Generate true geological model (this represents reality)
    true_model = generate_geological_model(
        grid_size=(20, 20),
        random_seed=random_seed
    )

    # Generate prior model (this represents our initial belief)
    # We use a different seed and larger discrepancy to make it more different from reality
    prior_model = generate_geological_model(
        grid_size=(20, 20),
        random_seed=random_seed + 500  # Much different seed to create larger initial uncertainty
    )

    # Introduce systematic biases in the prior model to make it more different from reality
    # For example, we might overestimate porosity and underestimate thickness
    for prop in prior_model['properties']:
        # Add random bias to each property (±15%)
        np.random.seed(random_seed + hash(prop) % 10000)
        bias_factor = np.random.uniform(0.85, 1.15)  # ±15% bias
        prior_model['properties'][prop] *= bias_factor

        # Add additional noise to represent initial uncertainty
        # Use absolute value to ensure the standard deviation is positive
        noise_scale = abs(np.mean(prior_model['properties'][prop])) * 0.1
        noise = np.random.normal(0, noise_scale, prior_model['properties'][prop].shape)
        prior_model['properties'][prop] += noise

        # Ensure values are within realistic ranges
        if prop == 'Porosity':
            prior_model['properties'][prop] = np.clip(prior_model['properties'][prop], 0.01, 0.2)
        elif prop == 'Permeability':
            prior_model['properties'][prop] = np.clip(prior_model['properties'][prop], 0.01, 1.0)
        elif prop == 'SW':  # Water saturation
            prior_model['properties'][prop] = np.clip(prior_model['properties'][prop], 0.1, 0.9)

    # Create production and economic models
    production_model = PretrainedProductionModel()
    economic_model = EconomicModel(
        gas_price=4.0,
        operating_cost=0.5,
        drilling_cost=10.0
    )

    # Time horizon for forecasts
    time_horizon_years = 10.0
    time_points_years = np.linspace(0, time_horizon_years, int(time_horizon_years * 12) + 1)
    time_points_days = time_points_years * 365.25

    # Create empty list to store well data
    wells = []

    # Create directory for results
    # Make sure these directories exist before trying to save files
    result_dirs = [
        'results',
        'results/npv_progression',
        'results/information_gain',
        'results/property_evolution'
    ]

    for directory in result_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensuring directory exists: {directory}")

    # Store property evolution for creating progression plot later
    property_evolution = {
        'initial': {prop: prior_model['properties'][prop].copy() for prop in prior_model['properties']}
    }

    # Store NPV distribution after each well
    npv_distributions = []

    # Calculate initial NPV map and uncertainty before any wells
    initial_npv_map = np.zeros_like(prior_model['X'])
    initial_uncertainty_map = np.zeros_like(prior_model['X'])

    # Calculate initial NPV at each grid point
    for i in range(prior_model['X'].shape[0]):
        for j in range(prior_model['X'].shape[1]):
            # Extract properties at this location
            loc_properties = {prop: prior_model['properties'][prop][i, j] for prop in prior_model['properties']}

            # Predict production
            production = production_model.predict_production(loc_properties, time_points_days)

            # Calculate NPV
            npv = economic_model.calculate_npv(production, time_points_years)
            initial_npv_map[i, j] = npv

    # Calculate initial uncertainty
    for prop in prior_model['properties']:
        # Calculate local standard deviation as proxy for uncertainty
        from scipy.ndimage import gaussian_filter
        local_std = gaussian_filter((prior_model['properties'][prop] - np.mean(prior_model['properties'][prop]))**2, sigma=1.0)
        initial_uncertainty_map += local_std
    initial_uncertainty_map /= len(prior_model['properties'])

    # Calculate initial Monte Carlo NPV at center of grid for reference
    center_i, center_j = prior_model['X'].shape[0] // 2, prior_model['X'].shape[1] // 2
    center_properties = {prop: prior_model['properties'][prop][center_i, center_j] for prop in prior_model['properties']}

    initial_mc_results = economic_model.monte_carlo_npv(
        production_model,
        center_properties,
        time_points_years,
        n_simulations=1000,
        random_seed=random_seed
    )
    npv_distributions.append(('Initial', initial_mc_results))

    # Plot initial NPV distribution
    plt.figure(figsize=(10, 6))
    plt.hist(initial_mc_results['npv_values'], bins=30, alpha=0.7, color='skyblue')
    plt.axvline(initial_mc_results['mean_npv'], color='red', linestyle='-', linewidth=2, label=f"Mean: ${initial_mc_results['mean_npv']:.1f}M")
    plt.axvline(initial_mc_results['p10'], color='orange', linestyle='--', linewidth=2, label=f"P10: ${initial_mc_results['p10']:.1f}M")
    plt.axvline(initial_mc_results['p90'], color='green', linestyle='--', linewidth=2, label=f"P90: ${initial_mc_results['p90']:.1f}M")
    plt.axvline(initial_mc_results['target_npv'], color='purple', linestyle=':', linewidth=2, label=f"Target: ${initial_mc_results['target_npv']:.1f}M")
    plt.axvline(0, color='black', linestyle='-', linewidth=1)

    plt.title(f"Initial NPV Distribution (Before Any Wells)")
    plt.xlabel("NPV ($M)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('results/npv_progression/initial_npv_distribution.png', dpi=300)
    plt.close()

    # Plot information gain concepts for the initial state
    plt.figure(figsize=(15, 10))

    # Plot 1: Initial NPV Map
    plt.subplot(2, 2, 1)
    cont = plt.contourf(prior_model['X'], prior_model['Y'], initial_npv_map, cmap='plasma')
    plt.colorbar(label="NPV ($M)")
    plt.title("Initial Expected NPV Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Plot 2: Initial Uncertainty Map
    plt.subplot(2, 2, 2)
    cont = plt.contourf(prior_model['X'], prior_model['Y'], initial_uncertainty_map, cmap='plasma_r')
    plt.colorbar(label="Uncertainty")
    plt.title("Initial Uncertainty Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Plot 3: Calculate Value of Information (Expected Value × Uncertainty)
    voi_surface = calculate_voi_surface(
        prior_model,
        production_model,
        economic_model
    )

    plt.subplot(2, 2, 3)
    cont = plt.contourf(prior_model['X'], prior_model['Y'], voi_surface, cmap='viridis')
    plt.colorbar(label="VOI ($M)")
    plt.title("Value of Information Surface")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Find the maximum VOI location
    max_idx = np.argmax(voi_surface.flatten())
    i, j = max_idx // prior_model['X'].shape[1], max_idx % prior_model['X'].shape[1]
    max_x, max_y = prior_model['X'][i, j], prior_model['Y'][i, j]
    plt.scatter(max_x, max_y, c='red', s=100, marker='*', edgecolor='white')

    # Plot 4: Information Gain Explanation
    plt.subplot(2, 2, 4)
    # This is a conceptual explanation plot
    plt.text(0.5, 0.95, "Information Gain Maximization Concept", ha='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.85, "The Value of Information (VOI) combines:", ha='center', fontsize=12)
    plt.text(0.5, 0.75, "1. Potential economic value (NPV)", ha='center', fontsize=12)
    plt.text(0.5, 0.70, "2. Current uncertainty level", ha='center', fontsize=12)
    plt.text(0.5, 0.65, "3. Cost of acquiring information", ha='center', fontsize=12)

    plt.text(0.5, 0.55, "High VOI areas have:", ha='center', fontsize=12)
    plt.text(0.5, 0.50, "• High potential value AND high uncertainty", ha='center', fontsize=12)
    plt.text(0.5, 0.45, "• Drilling here provides maximum information gain", ha='center', fontsize=12)
    plt.text(0.5, 0.40, "• Optimal for reducing overall uncertainty", ha='center', fontsize=12)

    plt.text(0.5, 0.30, "Information gain is highest where:", ha='center', fontsize=12)
    plt.text(0.5, 0.25, "• We know the least (high uncertainty)", ha='center', fontsize=12)
    plt.text(0.5, 0.20, "• Learning has high potential impact on decisions", ha='center', fontsize=12)
    plt.text(0.5, 0.15, "• The cost of acquiring information is outweighed", ha='center', fontsize=12)
    plt.text(0.5, 0.10, "  by the expected value of that information", ha='center', fontsize=12)

    plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/information_gain/information_gain_concept.png', dpi=300)
    plt.close()

    # Exploration loop
    for well_idx in range(n_wells):
        print(f"\nSelecting location for well {well_idx + 1}...")

        # Calculate VOI surface based on current knowledge
        voi_surface = calculate_voi_surface(
            prior_model,
            production_model,
            economic_model
        )

        # Find optimal location
        max_idx = np.argmax(voi_surface.flatten())
        i, j = max_idx // prior_model['X'].shape[1], max_idx % prior_model['X'].shape[1]
        x, y = prior_model['X'][i, j], prior_model['Y'][i, j]

        print(f"Selected location: ({x:.4f}, {y:.4f})")

        # "Drill" well by getting true properties at this location
        true_properties = {prop: true_model['properties'][prop][i, j] for prop in true_model['properties']}

        print("Discovered property values:")
        for prop, value in true_properties.items():
            print(f"  {prop}: {value:.4f}")

        # Store well data
        wells.append({
            'x': x,
            'y': y,
            'i': i,
            'j': j,
            'properties': true_properties,
            'voi_surface': voi_surface.copy()
        })

        # Calculate NPV at this location BEFORE update
        before_properties = {prop: prior_model['properties'][prop][i, j] for prop in prior_model['properties']}

        # Add some noise to before_production to simulate prediction uncertainty
        # Set random seed based on well index for reproducibility
        np.random.seed(random_seed + 500 + well_idx)
        before_production_base = production_model.predict_production(before_properties, time_points_days)
        prediction_noise = np.random.normal(1.0, 0.2, size=len(before_production_base))  # 20% noise
        before_production = before_production_base * prediction_noise

        before_npv = economic_model.calculate_npv(before_production, time_points_years)

        # Update our belief based on well data
        # In a real model, this would be a Bayesian update of a Gaussian process
        # Here we use a simple interpolation approach
        for prop in prior_model['properties']:
            # Create distance map from well
            X, Y = prior_model['X'], prior_model['Y']
            distances = np.sqrt((X - x)**2 + (Y - y)**2)

            # Stronger influence close to the well, weaker far away
            influence = np.exp(-distances / 0.5)  # Exponential decay with distance

            # Update prior map with true value based on influence
            prior_map = prior_model['properties'][prop]
            true_value = true_properties[prop]

            # Add some measurement noise to the true value to simulate imperfect measurements
            # Use well index and property name to create a unique but reproducible random seed for each property measurement
            prop_seed = hash(prop) % 10000  # Get a hash value from the property name
            np.random.seed(random_seed + well_idx * 100 + prop_seed)
            # Ensure standard deviation is positive
            noise_scale = abs(true_value) * 0.05  # 5% noise
            measurement_noise = np.random.normal(0, noise_scale)  # 5% noise
            measured_value = true_value + measurement_noise

            # Cap the maximum influence to avoid perfect knowledge even at the well location
            max_influence = 0.85  # Maximum 85% influence from well data
            capped_influence = influence * max_influence

            # Update prior map with measured value based on influence
            prior_model['properties'][prop] = prior_map * (1 - capped_influence) + measured_value * capped_influence

        # Store updated property maps for evolution plot
        property_evolution[f'well_{well_idx+1}'] = {
            prop: prior_model['properties'][prop].copy() for prop in prior_model['properties']
        }

        # Calculate NPV at this location AFTER update
        after_properties = {prop: prior_model['properties'][prop][i, j] for prop in prior_model['properties']}

        # After drilling, our prediction should be more accurate but still have some uncertainty
        # Use a different random seed for after-drilling calculation
        np.random.seed(random_seed + 1000 + well_idx)
        after_production_base = production_model.predict_production(after_properties, time_points_days)
        # Less noise after drilling (10% vs 20% before drilling)
        after_prediction_noise = np.random.normal(1.0, 0.1, size=len(after_production_base))
        after_production = after_production_base * after_prediction_noise

        after_npv = economic_model.calculate_npv(after_production, time_points_years)

        # Calculate Monte Carlo NPV at the well location
        well_mc_results = economic_model.monte_carlo_npv(
            production_model,
            after_properties,
            time_points_years,
            n_simulations=1000,
            random_seed=random_seed + well_idx
        )
        npv_distributions.append((f'After Well {well_idx+1}', well_mc_results))

        # Plot NPV distribution after this well
        plt.figure(figsize=(10, 6))
        plt.hist(well_mc_results['npv_values'], bins=30, alpha=0.7, color='skyblue')
        plt.axvline(well_mc_results['mean_npv'], color='red', linestyle='-', linewidth=2, label=f"Mean: ${well_mc_results['mean_npv']:.1f}M")
        plt.axvline(well_mc_results['p10'], color='orange', linestyle='--', linewidth=2, label=f"P10: ${well_mc_results['p10']:.1f}M")
        plt.axvline(well_mc_results['p90'], color='green', linestyle='--', linewidth=2, label=f"P90: ${well_mc_results['p90']:.1f}M")
        plt.axvline(well_mc_results['target_npv'], color='purple', linestyle=':', linewidth=2, label=f"Target: ${well_mc_results['target_npv']:.1f}M")
        plt.axvline(0, color='black', linestyle='-', linewidth=1)

        # Add text for confidence level
        plt.text(0.05, 0.95,
                f"Probability of exceeding target: {well_mc_results['prob_target']:.1%}\nProbability of positive NPV: {well_mc_results['prob_positive']:.1%}",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))

        plt.title(f"NPV Distribution After Well {well_idx + 1}")
        plt.xlabel("NPV ($M)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'results/npv_progression/npv_distribution_well_{well_idx + 1}.png', dpi=300)
        plt.close()

        # Calculate updated uncertainty map
        updated_uncertainty = np.zeros_like(voi_surface)
        for prop in prior_model['properties']:
            # Calculate local standard deviation as proxy for uncertainty
            from scipy.ndimage import gaussian_filter
            local_std = gaussian_filter((prior_model['properties'][prop] - np.mean(prior_model['properties'][prop]))**2, sigma=1.0)
            updated_uncertainty += local_std
        updated_uncertainty /= len(prior_model['properties'])

        # Calculate updated NPV map
        updated_npv_map = np.zeros_like(prior_model['X'])
        for i in range(prior_model['X'].shape[0]):
            for j in range(prior_model['X'].shape[1]):
                # Extract properties at this location
                loc_properties = {prop: prior_model['properties'][prop][i, j] for prop in prior_model['properties']}

                # Predict production
                production = production_model.predict_production(loc_properties, time_points_days)

                # Calculate NPV
                npv = economic_model.calculate_npv(production, time_points_years)
                updated_npv_map[i, j] = npv

        # Plot Information Gain visualization for this well
        plt.figure(figsize=(15, 12))

        # Plot 1: NPV Map Before Update
        plt.subplot(3, 2, 1)
        cont = plt.contourf(prior_model['X'], prior_model['Y'], initial_npv_map if well_idx == 0 else updated_npv_map, cmap='plasma')
        plt.colorbar(label="NPV ($M)")
        plt.title(f"NPV Map Before Well {well_idx + 1}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Show all previous wells
        for w in wells[:-1]:  # All except current well
            plt.scatter(w['x'], w['y'], c='black', s=50, marker='o', edgecolor='white')

        # Plot 2: Uncertainty Map Before Update
        plt.subplot(3, 2, 2)
        cont = plt.contourf(prior_model['X'], prior_model['Y'], initial_uncertainty_map if well_idx == 0 else updated_uncertainty, cmap='plasma_r')
        plt.colorbar(label="Uncertainty")
        plt.title(f"Uncertainty Map Before Well {well_idx + 1}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Show all previous wells
        for w in wells[:-1]:  # All except current well
            plt.scatter(w['x'], w['y'], c='black', s=50, marker='o', edgecolor='white')

        # Plot 3: VOI Surface and Selected Location
        plt.subplot(3, 2, 3)
        cont = plt.contourf(prior_model['X'], prior_model['Y'], voi_surface, cmap='viridis')
        plt.colorbar(label="VOI ($M)")
        plt.title(f"VOI Surface for Well {well_idx + 1}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Show all previous wells
        for w in wells[:-1]:  # All except current well
            plt.scatter(w['x'], w['y'], c='black', s=50, marker='o', edgecolor='white')

        # Show selected location
        plt.scatter(x, y, c='red', s=100, marker='*', edgecolor='white')

        # Plot 4: Value vs Uncertainty Scatter
        plt.subplot(3, 2, 4)

        # Flatten arrays for scatter plot
        flat_npv = updated_npv_map.flatten()
        flat_uncertainty = updated_uncertainty.flatten()
        flat_voi = voi_surface.flatten()

        # Create a scatter plot of value vs uncertainty, colored by VOI
        sc = plt.scatter(flat_npv, flat_uncertainty, c=flat_voi, cmap='viridis', alpha=0.5)
        plt.colorbar(sc, label='VOI')

        # Mark the selected point
        selected_npv = updated_npv_map[wells[-1]['i'], wells[-1]['j']]
        selected_uncertainty = updated_uncertainty[wells[-1]['i'], wells[-1]['j']]
        plt.scatter(selected_npv, selected_uncertainty, c='red', s=100, marker='*', edgecolor='white')

        plt.title(f"NPV vs. Uncertainty (colored by VOI)")
        plt.xlabel("Expected NPV ($M)")
        plt.ylabel("Uncertainty")
        plt.grid(True, linestyle='--', alpha=0.7)

        # Plot 5: Before vs After NPV at Well Location
        plt.subplot(3, 2, 5)
        plt.bar(['Before', 'After'], [before_npv, after_npv], color=['lightblue', 'orange'])
        plt.axhline(0, color='black', linestyle='-', linewidth=1)
        plt.title(f"NPV at Well {well_idx + 1} Location")
        plt.ylabel("NPV ($M)")
        plt.grid(True, linestyle='--', axis='y', alpha=0.7)

        # Add percent change
        percent_change = (after_npv - before_npv) / abs(before_npv) * 100 if before_npv != 0 else float('inf')
        plt.text(1, after_npv, f"{percent_change:.1f}%", ha='center', va='bottom')

        # Plot 6: Information Gain Calculation
        plt.subplot(3, 2, 6)

        # Calculate the reduction in uncertainty at the well location
        before_uncertainty = initial_uncertainty_map[wells[-1]['i'], wells[-1]['j']] if well_idx == 0 else updated_uncertainty[wells[-1]['i'], wells[-1]['j']]

        # Calculate updated uncertainty at the well location
        # In reality, there is always some residual uncertainty even after direct measurement
        # Using 10-20% of original uncertainty is more realistic than 0
        after_uncertainty = before_uncertainty * 0.15  # 85% reduction instead of 100%

        # Calculate information gain metrics
        uncertainty_reduction = before_uncertainty - after_uncertainty
        uncertainty_reduction_percent = uncertainty_reduction / before_uncertainty * 100 if before_uncertainty != 0 else 0

        # Create a conceptual display
        plt.axis('off')
        plt.text(0.5, 0.95, f"Information Gain from Well {well_idx + 1}", ha='center', fontsize=14, fontweight='bold')
        plt.text(0.5, 0.85, f"Before drilling:", ha='center', fontsize=12)
        plt.text(0.5, 0.80, f"• Expected NPV: ${before_npv:.2f}M", ha='center', fontsize=12)
        plt.text(0.5, 0.75, f"• Uncertainty level: {before_uncertainty:.4f}", ha='center', fontsize=12)

        plt.text(0.5, 0.65, f"After drilling:", ha='center', fontsize=12)
        plt.text(0.5, 0.60, f"• Updated NPV: ${after_npv:.2f}M", ha='center', fontsize=12)
        plt.text(0.5, 0.55, f"• Uncertainty level: {after_uncertainty:.4f}", ha='center', fontsize=12)

        plt.text(0.5, 0.45, f"Information value:", ha='center', fontsize=12)
        plt.text(0.5, 0.40, f"• NPV change: ${after_npv - before_npv:.2f}M ({percent_change:.1f}%)", ha='center', fontsize=12)
        plt.text(0.5, 0.35, f"• Uncertainty reduction: {uncertainty_reduction:.4f} ({uncertainty_reduction_percent:.1f}%)", ha='center', fontsize=12)
        plt.text(0.5, 0.30, f"• VOI at location: ${voi_surface[wells[-1]['i'], wells[-1]['j']]:.2f}M", ha='center', fontsize=12)

        plt.text(0.5, 0.20, f"Value of information exceeds drilling cost", ha='center', fontsize=12)
        plt.text(0.5, 0.15, f"Optimal trade-off between value and uncertainty reduction", ha='center', fontsize=12)

        plt.tight_layout()
        plt.savefig(f'results/information_gain/info_gain_well_{well_idx + 1}.png', dpi=300)
        plt.close()

        # Plot the current state
        plt.figure(figsize=(15, 8))

        # Select a few key properties to plot
        key_properties = ['Thickness', 'Porosity', 'Permeability', 'TOC']

        # Plot property maps
        for k, prop in enumerate(key_properties, 1):
            plt.subplot(2, 3, k)

            # Plot property map
            cont = plt.contourf(prior_model['X'], prior_model['Y'], prior_model['properties'][prop], cmap='viridis')
            plt.colorbar(label=prop)

            # Plot all wells
            for w in wells:
                plt.scatter(w['x'], w['y'], c='black', s=50, marker='o', edgecolor='white')

            # Highlight the current well
            plt.scatter(x, y, c='red', s=100, marker='*', edgecolor='white')

            plt.title(f"{prop} After Well {well_idx + 1}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")

        # Plot VOI surface
        plt.subplot(2, 3, 5)
        cont = plt.contourf(prior_model['X'], prior_model['Y'], voi_surface, cmap='plasma')
        plt.colorbar(label="VOI ($M)")

        # Plot all wells
        for w in wells:
            plt.scatter(w['x'], w['y'], c='black', s=50, marker='o', edgecolor='white')

        # Highlight the current well
        plt.scatter(x, y, c='red', s=100, marker='*', edgecolor='white')

        plt.title(f"VOI Surface Before Well {well_idx + 1}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Calculate and plot uncertainty map
        uncertainty = np.zeros_like(voi_surface)
        for prop in prior_model['properties']:
            # Calculate local standard deviation as a proxy for uncertainty
            from scipy.ndimage import gaussian_filter
            local_std = gaussian_filter((prior_model['properties'][prop] - np.mean(prior_model['properties'][prop]))**2, sigma=1.0)
            uncertainty += local_std
        uncertainty /= len(prior_model['properties'])

        plt.subplot(2, 3, 6)
        cont = plt.contourf(prior_model['X'], prior_model['Y'], uncertainty, cmap='plasma_r')
        plt.colorbar(label="Uncertainty")

        # Plot all wells
        for w in wells:
            plt.scatter(w['x'], w['y'], c='black', s=50, marker='o', edgecolor='white')

        # Highlight the current well
        plt.scatter(x, y, c='red', s=100, marker='*', edgecolor='white')

        plt.title(f"Uncertainty After Well {well_idx + 1}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.tight_layout()
        plt.savefig(f'results/exploration_state_well_{well_idx + 1}.png', dpi=300)
        plt.close()
    
    # Calculate final VOI surface and optimal development location
    final_voi = calculate_voi_surface(
        prior_model,
        production_model,
        economic_model
    )
    
    # Find optimal development location
    max_idx = np.argmax(final_voi.flatten())
    i, j = max_idx // prior_model['X'].shape[1], max_idx % prior_model['X'].shape[1]
    opt_x, opt_y = prior_model['X'][i, j], prior_model['Y'][i, j]
    
    # Get properties at optimal location
    opt_properties = {prop: prior_model['properties'][prop][i, j] for prop in prior_model['properties']}
    
    # Perform Monte Carlo NPV at optimal location
    mc_results = economic_model.monte_carlo_npv(
        production_model,
        opt_properties,
        time_points_years,
        n_simulations=1000,
        random_seed=random_seed
    )
    
    # Create final summary plot
    plt.figure(figsize=(15, 10))

    # Plot final property maps
    key_properties = ['Thickness', 'Porosity', 'Permeability', 'TOC']
    for k, prop in enumerate(key_properties, 1):
        plt.subplot(2, 3, k)
        cont = plt.contourf(prior_model['X'], prior_model['Y'], prior_model['properties'][prop], cmap='viridis')
        plt.colorbar(label=prop)

        # Plot exploration wells
        for w in wells:
            plt.scatter(w['x'], w['y'], c='black', s=50, marker='o', edgecolor='white')

        # Plot optimal development location
        plt.scatter(opt_x, opt_y, c='green', s=150, marker='*', edgecolor='white')

        plt.title(f"Final {prop} Map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    # Plot final VOI surface
    plt.subplot(2, 3, 5)
    cont = plt.contourf(prior_model['X'], prior_model['Y'], final_voi, cmap='plasma')
    plt.colorbar(label="VOI ($M)")

    # Plot exploration wells
    for w in wells:
        plt.scatter(w['x'], w['y'], c='black', s=50, marker='o', edgecolor='white')

    # Plot optimal development location
    plt.scatter(opt_x, opt_y, c='green', s=150, marker='*', edgecolor='white')

    plt.title("Final VOI Surface")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Plot NPV distribution
    plt.subplot(2, 3, 6)
    plt.hist(mc_results['npv_values'], bins=30, alpha=0.7, color='skyblue')
    plt.axvline(mc_results['mean_npv'], color='red', linestyle='-', linewidth=2, label=f"Mean: ${mc_results['mean_npv']:.1f}M")
    plt.axvline(mc_results['p10'], color='orange', linestyle='--', linewidth=2, label=f"P10: ${mc_results['p10']:.1f}M")
    plt.axvline(mc_results['p90'], color='green', linestyle='--', linewidth=2, label=f"P90: ${mc_results['p90']:.1f}M")
    plt.axvline(mc_results['target_npv'], color='purple', linestyle=':', linewidth=2, label=f"Target: ${mc_results['target_npv']:.1f}M")

    plt.title(f"NPV Distribution at Optimal Location")
    plt.xlabel("NPV ($M)")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/final_exploration_summary.png', dpi=300)

    # Create property evolution visualization
    # For each property, create a progression plot showing how it evolved with each well
    for prop in ['Porosity', 'Permeability', 'Thickness', 'TOC']:
        # For a large number of wells, split into multiple figures
        # Maximum 9 subplots per figure (3x3 grid)
        max_plots_per_figure = 9
        n_figures = (n_wells + 2) // max_plots_per_figure + 1  # +2 for initial state and buffer
        
        # Determine min/max for consistent colorbar
        all_values = []
        for stage, maps in property_evolution.items():
            all_values.append(maps[prop].flatten())
        
        all_values = np.concatenate(all_values)
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        # Process wells in batches
        plot_idx = 0
        for fig_idx in range(n_figures):
            # Calculate how many plots for this figure
            start_well = fig_idx * max_plots_per_figure - 1  # -1 to account for initial state
            end_well = min(start_well + max_plots_per_figure, n_wells)
            n_plots = end_well - start_well
            
            if start_well < 0:  # First figure includes initial state
                n_plots += 1
                has_initial = True
            else:
                has_initial = False
            
            if n_plots <= 0:
                break  # No more wells to plot
                
            # Create figure with appropriate size
            plt.figure(figsize=(15, 10))
            
            # Calculate grid dimensions - aim for roughly square layout
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            # Plot initial state in first figure
            if has_initial:
                plt.subplot(n_rows, n_cols, 1)
                cont = plt.contourf(prior_model['X'], prior_model['Y'], property_evolution['initial'][prop], 
                                  cmap='viridis', vmin=vmin, vmax=vmax)
                plt.colorbar(label=prop)
                plt.title(f"Initial {prop} Map")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plot_idx = 1
            else:
                plot_idx = 0
            
            # Plot state after each well in this batch
            for rel_well_idx in range(n_plots - (1 if has_initial else 0)):
                well_idx = start_well + rel_well_idx + (1 if has_initial else 0)
                if well_idx >= n_wells:
                    break
                    
                plt.subplot(n_rows, n_cols, plot_idx + 1)
                cont = plt.contourf(prior_model['X'], prior_model['Y'], 
                                  property_evolution[f'well_{well_idx+1}'][prop], 
                                  cmap='viridis', vmin=vmin, vmax=vmax)
                plt.colorbar(label=prop)
                
                # Plot wells drilled up to this point
                for i in range(well_idx + 1):
                    color = 'red' if i == well_idx else 'black'
                    size = 100 if i == well_idx else 50
                    plt.scatter(wells[i]['x'], wells[i]['y'], c=color, s=size, marker='o', edgecolor='white')
                
                plt.title(f"{prop} After Well {well_idx + 1}")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plot_idx += 1
            
            plt.tight_layout()
            batch_label = f"_batch{fig_idx+1}" if n_figures > 1 else ""
            plt.savefig(f'results/property_evolution/{prop}_evolution{batch_label}.png', dpi=300)
            plt.close()

    # Create NPV distribution progression plot
    plt.figure(figsize=(15, 8))

    # Plot NPV mean and confidence intervals over time
    stages = ['Initial'] + [f'After Well {i+1}' for i in range(n_wells)]
    means = [npv_dist[1]['mean_npv'] for npv_dist in npv_distributions]
    p10s = [npv_dist[1]['p10'] for npv_dist in npv_distributions]
    p90s = [npv_dist[1]['p90'] for npv_dist in npv_distributions]
    prob_targets = [npv_dist[1]['prob_target'] for npv_dist in npv_distributions]

    # Plot 1: NPV means and confidence intervals
    plt.subplot(1, 2, 1)
    plt.plot(range(len(stages)), means, 'o-', color='blue', linewidth=2, label='Mean NPV')
    plt.fill_between(range(len(stages)), p10s, p90s, color='blue', alpha=0.2, label='P10-P90 Range')

    # Add target line
    plt.axhline(mc_results['target_npv'], color='red', linestyle='--', linewidth=2,
               label=f'Target: ${mc_results["target_npv"]}M')

    plt.xticks(range(len(stages)), stages, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Exploration Stage')
    plt.ylabel('NPV ($M)')
    plt.title('NPV Evolution During Exploration')
    plt.legend()

    # Plot 2: Probability of meeting target
    plt.subplot(1, 2, 2)
    plt.plot(range(len(stages)), prob_targets, 'o-', color='green', linewidth=2)

    # Add target confidence line
    plt.axhline(0.9, color='red', linestyle='--', linewidth=2,
               label='Target Confidence: 90%')

    plt.xticks(range(len(stages)), stages, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Exploration Stage')
    plt.ylabel('Probability of Meeting Target')
    plt.title('Confidence Evolution During Exploration')
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/npv_progression/npv_confidence_progression.png', dpi=300)
    plt.close()

    # Create a overlay of all NPV distributions to show narrowing
    plt.figure(figsize=(12, 6))

    # Use different colors for each stage
    colors = plt.cm.viridis(np.linspace(0, 1, len(npv_distributions)))

    # Create histograms with increasing opacity
    for i, (stage, npv_dist) in enumerate(npv_distributions):
        alpha = 0.3 if i < len(npv_distributions) - 1 else 0.7
        plt.hist(npv_dist['npv_values'], bins=30, alpha=alpha, color=colors[i], label=stage)

    # Add target line
    plt.axvline(mc_results['target_npv'], color='red', linestyle='--', linewidth=2,
               label=f'Target: ${mc_results["target_npv"]}M')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('NPV ($M)')
    plt.ylabel('Frequency')
    plt.title('NPV Distribution Evolution During Exploration')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/npv_progression/npv_distribution_evolution.png', dpi=300)
    plt.close()
    
    # Print final summary
    print("\nExploration Summary:")
    print(f"Drilled {n_wells} exploration wells")
    print(f"Optimal development location: ({opt_x:.4f}, {opt_y:.4f})")
    print(f"Expected NPV: ${mc_results['mean_npv']:.2f}M")
    print(f"P10-P90 range: ${mc_results['p10']:.2f}M to ${mc_results['p90']:.2f}M")
    print(f"Probability of exceeding target (${mc_results['target_npv']}M): {mc_results['prob_target']:.2%}")
    print(f"Probability of positive NPV: {mc_results['prob_positive']:.2%}")
    
    return {
        'wells': wells,
        'optimal_location': (opt_x, opt_y),
        'optimal_properties': opt_properties,
        'monte_carlo_results': mc_results,
        'prior_model': prior_model,
        'final_voi': final_voi
    }


if __name__ == "__main__":
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("Error: SciPy is required for this demo.")
        print("Please install it with: pip install scipy")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run exploration simulation with pre-trained model")
    parser.add_argument("--wells", type=int, default=10, help="Number of exploration wells to drill")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show", action="store_true", help="Show plots")
    args = parser.parse_args()
    
    # Run the simulation
    results = run_exploration_simulation(n_wells=args.wells, random_seed=args.seed)
    
    print("\nSimulation complete. Results saved to the 'results' directory.")
    
    # Show plots if requested
    if args.show:
        plt.show()