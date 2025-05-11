"""
Field Exploration Example using Gaussian Process modeling.

This script demonstrates how to use the basin_gp module with field data
from the Dummy_Field_Testing.py source. It compares multiple exploration
strategies to find the one that most efficiently reaches the target 
profitability confidence.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from basin_gp.model import BasinExplorationGP
from basin_gp.field_data import (
    load_field_data, 
    field_thickness, 
    field_porosity, 
    field_permeability,
    field_toc,
    field_water_saturation,
    field_clay_volume,
    field_depth,
    visualize_field_geology,
    add_field_wells_from_samples,
    create_basin_grid
)

from basin_gp.planning import plan_next_well_balanced, plan_next_well_voi
from basin_gp_example import plot_basin_geology, plot_gp_model_evolution

def str2bool(value):
    if isinstance(value, str):
        if value.lower() in ('yes', 'true', 't', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', '0'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def compare_field_strategies(basin_size, resolution, grid_tensor, n_initial_wells, n_exploration_wells, 
                             true_funcs, economic_params, strategies, show_plots=False):
    """
    Compare different exploration strategies on field data.
    
    Args:
        basin_size: Basin size in kilometers
        resolution: Grid resolution
        grid_tensor: Grid of locations
        n_initial_wells: Number of initial wells
        n_exploration_wells: Number of exploration wells
        true_funcs: List of functions for ground truth
        economic_params: Economic parameters
        strategies: List of strategies to compare
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with results per strategy
    """
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Running {strategy.upper()} strategy...")
        print(f"{'='*50}")
        
        # Create a new model with field properties
        basin_gp = BasinExplorationGP(
            basin_size=basin_size,
            properties=['porosity', 'permeability', 'thickness', 'toc', 'water_saturation', 'clay_volume', 'depth']
        )
        
        # Set target confidence
        basin_gp.target_confidence = 0.9
        
        # Add initial wells from field data samples (same for all strategies)
        basin_gp = add_field_wells_from_samples(basin_gp, n_wells=n_initial_wells, seed=42)
        
        # Run sequential exploration
        history = basin_gp.sequential_exploration(
            grid_tensor,
            n_exploration_wells,
            true_funcs,
            noise_std=0.01,
            strategy=strategy,
            economic_params=economic_params,
            plot=True,
            confidence_target=0.9,
            stop_at_confidence=False  # Continue to get full comparison data
        )
        
        # Save results
        confidence_history = [step.get('profitability_confidence', 0) for step in history]
        wells_to_target = next((i+1 for i, conf in enumerate(confidence_history) 
                               if conf >= 0.9), n_exploration_wells+1)
        
        results[strategy] = {
            'history': history,
            'confidence_history': confidence_history,
            'final_confidence': basin_gp.profitability_confidence,
            'wells_to_target': wells_to_target,
            'model': basin_gp
        }
        
        # Print summary for this strategy
        print(f"\nStrategy: {strategy.upper()}")
        print(f"Final confidence in profitability: {basin_gp.profitability_confidence*100:.1f}%")
        print(f"Wells needed to reach 90% confidence: {wells_to_target if wells_to_target <= n_exploration_wells else 'Not reached'}")
        
    return results

def main(show_plots=False):
    """
    Main function to demonstrate field GP-based exploration using multiple strategies.
    
    Args:
        show_plots: Whether to display plots during execution
    """
    print("Loading field data...")
    df_full = load_field_data(show_plots=show_plots)
    
    # Basin parameters (use these consistently across all functions)
    basin_size = (20, 20)  # 20 km x 20 km basin
    resolution = 30  # Grid resolution
    plt.rcParams['figure.max_open_warning'] = 0
    
    # Visualize the field geology
    print("\nVisualizing field geology...")
    grid_tensor, x1_grid, x2_grid, thickness, porosity, permeability, toc, saturation, clay, depth = \
        visualize_field_geology(basin_size, resolution, show_plots=show_plots)
    
    # Define economic parameters for exploration planning
    economic_params = {
        'area': 1.0e6,  # m²
        'water_saturation': 0.5,  # Use average from field data
        'formation_volume_factor': 1.1,
        'oil_price': 80,  # $ per barrel
        'drilling_cost': 8e6,  # $
        'completion_cost': 4e6  # $
    }
    
    # Define property functions to use for sequential exploration
    true_funcs = [
        field_porosity,
        field_permeability,
        field_thickness,
        field_toc, 
        field_water_saturation,
        field_clay_volume,
        field_depth
    ]
    
    # Configuration for the comparison
    n_initial_wells = 5
    n_exploration_wells = 10
    strategies_to_compare = ['uncertainty', 'economic', 'balanced', 'voi']
    
    # Run the strategy comparison
    results = compare_field_strategies(
        basin_size=basin_size,
        resolution=resolution,
        grid_tensor=grid_tensor,
        n_initial_wells=n_initial_wells,
        n_exploration_wells=n_exploration_wells,
        true_funcs=true_funcs,
        economic_params=economic_params,
        strategies=strategies_to_compare,
        show_plots=show_plots
    )
    
    # Create plots comparing the strategies
    if show_plots:
        # Plot confidence progression
        plt.figure(figsize=(12, 8))
        for strategy, result in results.items():
            plt.plot(
                range(1, len(result['confidence_history'])+1),
                result['confidence_history'],
                marker='o',
                linewidth=2,
                label=strategy.upper()
            )
        
        plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% Confidence Target')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Number of Exploration Wells')
        plt.ylabel('Confidence in Profitability')
        plt.title('Strategy Comparison: Confidence Progression')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Determine the best strategy
    best_strategy = min(results.items(), key=lambda x: 
                        (float('inf') if x[1]['wells_to_target'] > n_exploration_wells 
                        else x[1]['wells_to_target']))
    
    print("\n" + "="*70)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*70)
    
    # Print comparison table
    print(f"{'Strategy':<12} | {'Final Confidence':<20} | {'Wells to 90% Confidence':<25}")
    print("-" * 70)
    
    for strategy, result in results.items():
        wells_needed = result['wells_to_target']
        if wells_needed > n_exploration_wells:
            wells_str = "Not reached"
        else:
            wells_str = str(wells_needed)
            
        print(f"{strategy.upper():<12} | {result['final_confidence']*100:.1f}%{' ':<13} | {wells_str:<25}")
    
    print("\nBEST STRATEGY: " + best_strategy[0].upper())
    print(f"Required only {best_strategy[1]['wells_to_target']} wells to reach 90% confidence")
    
    # If VOI strategy reached target confidence, display its final model
    if 'voi' in results and results['voi']['wells_to_target'] <= n_exploration_wells:
        voi_model = results['voi']['model']
        
        print("\nFinal VOI Strategy Model:")
        # Get predictions from VOI model
        mean, std = voi_model.predict(grid_tensor)
        
        # Print key statistics from final model
        total_porosity = torch.mean(mean[:, 0]).item()
        total_permeability = torch.mean(mean[:, 1]).item()
        total_thickness = torch.mean(mean[:, 2]).item()
        
        print(f"Average Porosity: {total_porosity:.3f}")
        print(f"Average Permeability: {total_permeability:.1f} mD")
        print(f"Average Thickness: {total_thickness:.1f} ft")
        print(f"Confidence in Profitability: {voi_model.profitability_confidence*100:.1f}%")
    
    print("\nField exploration simulation complete.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Field exploration using Gaussian Process modeling')
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help='Show plots during execution (default: False)')
    
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(show_plots=args.show_plots)