#!/usr/bin/env python
"""
Simplified demonstration of the BEL package.

This is a minimal working example that demonstrates the core concepts
without any data loading or complex processing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Create a special version of the ProductionPredictor for this demo
class SimpleProductionPredictor:
    """Simplified production predictor for demo purposes."""
    
    def __init__(self):
        self.weights = {
            'Thickness': 100,
            'Porosity': 10000,
            'Permeability': 1000,
            'TOC': 50,
            'SW': -100,
            'Depth': -0.01,
            'Vclay': -500
        }
    
    def predict_initial_rate(self, properties):
        """Predict initial production rate from properties."""
        if isinstance(properties, pd.DataFrame):
            result = np.zeros(len(properties))
            for prop, weight in self.weights.items():
                if prop in properties.columns:
                    result += properties[prop].values * weight
            return result
        elif isinstance(properties, dict):
            result = np.zeros(len(next(iter(properties.values()))))
            for prop, weight in self.weights.items():
                if prop in properties:
                    result += properties[prop] * weight
            return result
        else:
            # Assume numpy array
            # Just use a simplified calculation for demo
            return np.sum(properties, axis=1) * 1000
    
    def predict_decline_parameters(self, properties):
        """Return fixed decline parameters for demo."""
        if isinstance(properties, pd.DataFrame):
            n = len(properties)
        elif isinstance(properties, dict):
            n = len(next(iter(properties.values())))
        else:
            n = properties.shape[0]
            
        return np.ones(n) * 0.8, np.ones(n) * 0.1  # b, di
    
    def forecast_production_profile(self, initial_rate, b_factor, decline_rate, time_points):
        """Forecast production using Arps decline curve."""
        return initial_rate / (1 + b_factor * decline_rate * time_points) ** (1 / b_factor)


class SimpleEconomicModel:
    """Simplified economic model for demo purposes."""
    
    def __init__(self, gas_price=4.0, operating_cost=0.5, drilling_cost=10.0):
        self.gas_price = gas_price
        self.operating_cost = operating_cost
        self.drilling_cost = drilling_cost
    
    def calculate_npv(self, production, time_points, discount_rate=0.1):
        """Calculate simple NPV."""
        revenue = production * (self.gas_price / 1000)  # $/mcf to $M/mcf
        opex = production * (self.operating_cost / 1000)  # $/mcf to $M/mcf
        cash_flow = revenue - opex
        
        # Apply discount
        discount_factors = 1 / (1 + discount_rate) ** time_points
        npv = np.sum(cash_flow * discount_factors) - self.drilling_cost
        
        return npv


def run_demo():
    """Run a simplified demonstration."""
    print("Running simplified BEL demonstration...")
    
    # Create synthetic geological properties for a reservoir
    n_points = 400  # Number of grid points (20x20)
    n_properties = 7
    property_names = ['Thickness', 'Porosity', 'Permeability', 'TOC', 'SW', 'Depth', 'Vclay']
    
    # Create random property values
    np.random.seed(42)
    properties = {}
    for i, prop in enumerate(property_names):
        # Create a smooth random field with spatial correlation
        base = np.random.normal(0, 1, (20, 20))
        # Apply smoothing
        from scipy.ndimage import gaussian_filter
        smooth = gaussian_filter(base, sigma=2.0)
        # Scale to property range
        if prop == 'Thickness':
            prop_data = 50 + 150 * (smooth - smooth.min()) / (smooth.max() - smooth.min())
        elif prop == 'Porosity':
            prop_data = 0.02 + 0.06 * (smooth - smooth.min()) / (smooth.max() - smooth.min())
        elif prop == 'Permeability':
            prop_data = 0.1 + 0.4 * (smooth - smooth.min()) / (smooth.max() - smooth.min())
        elif prop == 'TOC':
            prop_data = 2 + 6 * (smooth - smooth.min()) / (smooth.max() - smooth.min())
        elif prop == 'SW':
            prop_data = 0.3 + 0.4 * (smooth - smooth.min()) / (smooth.max() - smooth.min())
        elif prop == 'Depth':
            prop_data = -11000 + 3000 * (smooth - smooth.min()) / (smooth.max() - smooth.min())
        elif prop == 'Vclay':
            prop_data = 0.3 + 0.3 * (smooth - smooth.min()) / (smooth.max() - smooth.min())
        
        properties[prop] = prop_data
    
    # Create production model
    production_model = SimpleProductionPredictor()
    
    # Create economic model
    economic_model = SimpleEconomicModel(gas_price=4.0, operating_cost=0.5, drilling_cost=10.0)
    
    # Define grid coordinates
    x = np.linspace(-94.6, -92.9, 20)
    y = np.linspace(31.3, 33.0, 20)
    X, Y = np.meshgrid(x, y)
    
    # Calculate initial production rates across the grid
    flat_properties = {}
    for prop, data in properties.items():
        flat_properties[prop] = data.flatten()
    
    initial_rates = production_model.predict_initial_rate(flat_properties)
    initial_rates = initial_rates.reshape(20, 20)
    
    # Calculate NPV for each grid point
    time_points = np.linspace(0, 10, 121)  # 10 years, monthly
    npv_map = np.zeros((20, 20))
    
    for i in range(20):
        for j in range(20):
            # Get initial rate for this location
            qi = initial_rates[i, j]
            
            # Get decline parameters
            b, di = 0.8, 0.1
            
            # Forecast production
            production = production_model.forecast_production_profile(qi, b, di, time_points * 365.25)
            
            # Convert daily to monthly
            monthly_production = production * 30.4  # Average days per month
            
            # Calculate NPV
            npv = economic_model.calculate_npv(monthly_production, time_points)
            npv_map[i, j] = npv
    
    # Visualize the results
    plt.figure(figsize=(15, 10))
    
    # Plot property maps
    for i, (prop, data) in enumerate(properties.items(), 1):
        plt.subplot(2, 4, i)
        plt.contourf(X, Y, data, cmap='viridis')
        plt.colorbar(label=prop)
        plt.title(f"{prop}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
    
    # Plot NPV map
    plt.subplot(2, 4, 8)
    plt.contourf(X, Y, npv_map, cmap='plasma')
    plt.colorbar(label="NPV ($M)")
    plt.title("Net Present Value")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Find the optimal well location
    max_idx = np.argmax(npv_map.flatten())
    i, j = max_idx // 20, max_idx % 20
    optimal_x, optimal_y = X[i, j], Y[i, j]
    
    plt.scatter(optimal_x, optimal_y, c='red', s=100, marker='*', edgecolor='black')
    plt.text(optimal_x, optimal_y, "Optimal Well", color='white', fontweight='bold',
             verticalalignment='bottom', horizontalalignment='right')
    
    # Display summary
    print("\nSimulation Results:")
    print(f"Optimal well location: ({optimal_x:.4f}, {optimal_y:.4f})")
    print(f"Expected NPV: ${npv_map[i, j]:.2f}M")
    print(f"Expected initial rate: {initial_rates[i, j]:.2f} mcf/day")
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/simplified_demo_results.png', dpi=300)
    print("Plot saved to: results/simplified_demo_results.png")
    
    return {
        'properties': properties,
        'initial_rates': initial_rates,
        'npv_map': npv_map,
        'optimal_location': (optimal_x, optimal_y),
        'optimal_npv': npv_map[i, j]
    }


if __name__ == "__main__":
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("Error: SciPy is required for this demo.")
        print("Please install it with: pip install scipy")
        sys.exit(1)
        
    # Run the demo
    results = run_demo()
    
    # Show the plot
    plt.show()