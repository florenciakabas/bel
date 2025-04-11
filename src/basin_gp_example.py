"""
Basin Exploration Example using Gaussian Process modeling.

This script demonstrates how to use the basin_gp module for modeling 
geological properties and planning exploration wells.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from basin_gp.model import BasinExplorationGP
from basin_gp.simulation import (
    create_basin_grid, visualize_true_geology, add_random_wells,
    true_porosity, true_permeability, true_thickness, calculate_resources
)

def main():
    # Basin parameters
    basin_size = (20, 20)  # 20 km x 20 km basin
    resolution = 30  # Grid resolution

    # Visualize the true geology
    print("Creating basin model and true geology...")
    grid_tensor, x1_grid, x2_grid, true_por, true_perm, true_thick = visualize_true_geology(
        basin_size, resolution
    )

    # Initialize the exploration framework
    basin_gp = BasinExplorationGP(
        basin_size=basin_size,
        properties=['porosity', 'permeability', 'thickness']
    )

    # Add initial wells (random exploration at first)
    n_initial_wells = 3
    print(f"\nAdding {n_initial_wells} initial random wells...")
    basin_gp = add_random_wells(basin_gp, n_initial_wells, basin_size, seed=42)

    # Define economic parameters for exploration planning
    economic_params = {
        'area': 1.0e6,  # m²
        'water_saturation': 0.3,
        'formation_volume_factor': 1.1,
        'oil_price': 80,  # $ per barrel
        'drilling_cost': 8e6,  # $
        'completion_cost': 4e6  # $
    }

    # Run sequential exploration with uncertainty strategy
    n_exploration_wells = 5
    print("\nRunning uncertainty-based exploration...")
    uncertainty_history = basin_gp.sequential_exploration(
        grid_tensor,
        n_exploration_wells,
        [true_porosity, true_permeability, true_thickness],
        noise_std=0.01,
        strategy='uncertainty',
        plot=True
    )

    # Reset and try economic-based strategy
    basin_gp = BasinExplorationGP(
        basin_size=basin_size,
        properties=['porosity', 'permeability', 'thickness']
    )

    # Re-add initial wells
    basin_gp = add_random_wells(basin_gp, n_initial_wells, basin_size, seed=42)

    # Run economic-based exploration
    print("\nRunning economic-based exploration...")
    economic_history = basin_gp.sequential_exploration(
        grid_tensor,
        n_exploration_wells,
        [true_porosity, true_permeability, true_thickness],
        noise_std=0.01,
        strategy='economic',
        economic_params=economic_params,
        plot=True
    )

    # Calculate predicted resources
    print("\nCalculating resources based on exploration...")
    basin_gp.fit(verbose=False)
    mean, std = basin_gp.predict(grid_tensor)
    
    total_resources = calculate_resources(mean, resolution, basin_size)
    print(f"Estimated total recoverable resources: {total_resources/1e6:.2f} million barrels")
    
    # Calculate economic value
    total_value = total_resources * economic_params['oil_price']
    total_cost = len(basin_gp.wells) * (economic_params['drilling_cost'] + economic_params['completion_cost'])
    net_value = total_value - total_cost
    
    print(f"Estimated total economic value: ${total_value/1e9:.2f} billion")
    print(f"Total exploration cost: ${total_cost/1e6:.2f} million")
    print(f"Net value: ${net_value/1e9:.2f} billion")

if __name__ == "__main__":
    main()