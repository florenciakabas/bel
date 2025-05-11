"""
Profitability Confidence Example using Gaussian Process modeling.

This script demonstrates the Value of Information (VOI) strategy for optimizing
exploration to quickly reach confidence in project profitability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from basin_gp.model import BasinExplorationGP
from basin_gp.simulation import (
    create_basin_grid, visualize_true_geology, 
    true_porosity, true_permeability, true_thickness,
    add_knowledge_driven_wells
)

def compare_strategies(basin_size, resolution, n_wells, strategies, economic_params, show_plots=False):
    """
    Compare different exploration strategies by tracking profitability confidence.
    
    Args:
        basin_size: Basin size in (x, y) km
        resolution: Grid resolution
        n_wells: Number of exploration wells to drill
        strategies: List of strategies to compare
        economic_params: Economic parameters for value calculation
        show_plots: Whether to display plots
        
    Returns:
        results: Dictionary with results for each strategy
    """
    # Create grid
    grid_tensor, x1_grid, x2_grid, mask = create_basin_grid(basin_size, resolution)
    
    # Calculate true values for visualization 
    true_por = true_porosity(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    true_perm = true_permeability(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    true_thick = true_thickness(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    
    # Container for results
    results = {}
    
    # Run each strategy
    for strategy in strategies:
        print(f"\nRunning {strategy} strategy...")
        
        # Initialize model with 3 initial wells
        basin_gp = BasinExplorationGP(
            basin_size=basin_size,
            properties=['porosity', 'permeability', 'thickness']
        )
        
        # Add initial wells (same for all strategies)
        n_initial_wells = 3
        basin_gp = add_knowledge_driven_wells(
            basin_gp, 
            n_initial_wells, 
            basin_size, 
            [true_porosity, true_permeability, true_thickness],
            uncertainty_weight=0.3,
            seed=42
        )
        
        # Set target confidence
        basin_gp.target_confidence = 0.9
        
        # Run exploration
        history = basin_gp.sequential_exploration(
            grid_tensor,
            n_wells,
            [true_porosity, true_permeability, true_thickness],
            noise_std=0.01,
            strategy=strategy,
            economic_params=economic_params,
            plot=True,
            stop_at_confidence=False  # Continue to get full comparison data
        )
        
        # Extract results
        results[strategy] = {
            'confidence_history': [step.get('profitability_confidence', 0) for step in history],
            'n_wells': len(history),
            'final_confidence': basin_gp.profitability_confidence
        }
        
        # Print summary
        print(f"Strategy: {strategy}")
        print(f"Final confidence in profitability: {basin_gp.profitability_confidence*100:.1f}%")
        
        # Calculate how many wells were needed to reach 90% confidence
        conf_history = results[strategy]['confidence_history']
        wells_to_confidence = next((i+1 for i, conf in enumerate(conf_history) if conf >= 0.9), n_wells)
        
        print(f"Wells needed to reach 90% confidence: {wells_to_confidence}")
        results[strategy]['wells_to_confidence'] = wells_to_confidence
    
    # Plot confidence history for all strategies
    if show_plots:
        plt.figure(figsize=(12, 6))
        
        for strategy, result in results.items():
            conf_history = result['confidence_history']
            x = range(1, len(conf_history)+1)
            plt.plot(x, conf_history, marker='o', linewidth=2, label=strategy)
        
        plt.axhline(y=0.9, color='r', linestyle='--', label='90% Confidence Target')
        plt.grid(True)
        plt.xlabel('Number of Exploration Wells')
        plt.ylabel('Confidence in Profitability')
        plt.title('Strategy Comparison: Confidence in Profitability vs. Number of Wells')
        plt.legend()
        plt.ylim(0, 1)
        plt.show()
    
    return results

def str2bool(value):
    if isinstance(value, str):
        if value.lower() in ('yes', 'true', 't', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', '0'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def main(show_plots=False):
    """
    Main function to demonstrate profitability confidence-focused exploration.
    
    Args:
        show_plots: Whether to display plots during execution
    """
    # Basin parameters
    basin_size = (20, 20)  # 20 km x 20 km basin
    resolution = 30  # Grid resolution
    plt.rcParams['figure.max_open_warning'] = 0
    
    # Economic parameters
    economic_params = {
        'area': 1.0e6,  # m²
        'water_saturation': 0.3,
        'formation_volume_factor': 1.1,
        'oil_price': 80,  # $ per barrel
        'drilling_cost': 8e6,  # $
        'completion_cost': 4e6  # $
    }
    
    # Strategies to compare
    strategies = ['uncertainty', 'economic', 'balanced', 'voi']
    
    # Number of exploration wells
    n_wells = 10
    
    # Run comparison
    results = compare_strategies(
        basin_size, 
        resolution, 
        n_wells, 
        strategies, 
        economic_params,
        show_plots=show_plots
    )
    
    # Print summary
    print("\nStrategy Comparison Summary:")
    print("----------------------------")
    for strategy, result in results.items():
        print(f"{strategy.capitalize()} Strategy:")
        print(f"  Final confidence: {result['final_confidence']*100:.1f}%")
        print(f"  Wells needed for 90% confidence: {result['wells_to_confidence']}")
    
    # Determine best strategy
    best_strategy = min(results.items(), key=lambda x: x[1]['wells_to_confidence'])
    print(f"\nBest strategy: {best_strategy[0].upper()}")
    print(f"Required only {best_strategy[1]['wells_to_confidence']} wells to reach 90% confidence")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Profitability confidence exploration example')
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help='Show plots during execution (default: False)')
    
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(show_plots=args.show_plots)