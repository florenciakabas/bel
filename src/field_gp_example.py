"""
Field Exploration Example using Gaussian Process modeling.

This script demonstrates how to use the basin_gp module with field data
from the Dummy_Field_Testing.py source, focusing on the Value of Information (VOI)
exploration strategy with customizable profitability threshold.

Features:
- Realistic field data visualization with publication-quality plots
- VOI-based exploration strategy for optimized well placement
- Customizable profitability threshold for economic analysis
- Visualization of economic model and exploration progress
- High-quality plots suitable for presentation slides
- Option to suppress plots for faster execution

Usage examples:
- Basic run: python src/field_gp_example.py
- With plots: python src/field_gp_example.py --show-plots
- Custom threshold: python src/field_gp_example.py --profit-threshold 50
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

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

def run_voi_exploration(basin_size, resolution, grid_tensor, n_initial_wells, n_exploration_wells,
                       true_funcs, economic_params, profit_threshold=30, confidence_target=0.9,
                       show_plots=False, plot_callback=None, length_scale=None):
    """
    Run field exploration using the Value of Information (VOI) strategy with a target
    profitability threshold and confidence level.

    Args:
        basin_size: Basin size in kilometers
        resolution: Grid resolution
        grid_tensor: Grid of locations
        n_initial_wells: Number of initial wells
        n_exploration_wells: Number of exploration wells
        true_funcs: List of functions for ground truth
        economic_params: Economic parameters
        profit_threshold: Minimum profit in millions $ to be confident about
        confidence_target: Target confidence level (0-1)
        show_plots: Whether to display plots
        plot_callback: Optional callback function for custom plotting
        length_scale: Optional length scale parameter for GP kernel

    Returns:
        Dictionary with exploration results
    """
    print(f"\n{'='*60}")
    print(f"VOI EXPLORATION WITH {profit_threshold}M$ PROFIT THRESHOLD")
    print(f"TARGET CONFIDENCE: {confidence_target*100:.0f}%")
    print(f"{'='*60}")

    # Update economic params with profit threshold
    economic_params['profit_threshold'] = profit_threshold * 1e6  # Convert to dollars

    # Create a new model with field properties
    basin_gp = BasinExplorationGP(
        basin_size=basin_size,
        properties=['porosity', 'permeability', 'thickness', 'toc', 'water_saturation', 'clay_volume', 'depth'],
        length_scale=length_scale
    )

    # Set target confidence
    basin_gp.target_confidence = confidence_target

    # Add initial wells from field data samples
    basin_gp = add_field_wells_from_samples(basin_gp, n_wells=n_initial_wells, seed=42)

    # Run sequential exploration with VOI strategy
    history = basin_gp.sequential_exploration(
        grid_tensor,
        n_exploration_wells,
        true_funcs,
        noise_std=0.01,
        strategy='voi',
        economic_params=economic_params,
        plot=show_plots,
        plot_callback=plot_callback,
        confidence_target=confidence_target,
        stop_at_confidence=False  # Continue to get full data
    )

    # Save results
    confidence_history = [step.get('profitability_confidence', 0) for step in history]
    wells_to_target = next((i+1 for i, conf in enumerate(confidence_history)
                          if conf >= confidence_target), n_exploration_wells+1)

    # Calculate mean property values after exploration
    mean, std = basin_gp.predict(grid_tensor)
    avg_property_values = {
        'porosity': torch.mean(mean[:, 0]).item(),
        'permeability': torch.mean(mean[:, 1]).item(),
        'thickness': torch.mean(mean[:, 2]).item(),
        'toc': torch.mean(mean[:, 3]).item() if mean.shape[1] > 3 else None,
        'water_saturation': torch.mean(mean[:, 4]).item() if mean.shape[1] > 4 else None,
        'clay_volume': torch.mean(mean[:, 5]).item() if mean.shape[1] > 5 else None,
        'depth': torch.mean(mean[:, 6]).item() if mean.shape[1] > 6 else None
    }

    results = {
        'history': history,
        'confidence_history': confidence_history,
        'final_confidence': basin_gp.profitability_confidence,
        'wells_to_target': wells_to_target,
        'model': basin_gp,
        'avg_property_values': avg_property_values,
        'economic_params': economic_params
    }

    # Print summary
    print(f"\nVOI STRATEGY RESULTS:")
    print(f"  Final confidence in profitability: {basin_gp.profitability_confidence*100:.1f}%")
    print(f"  Profit threshold: ${profit_threshold}M")
    print(f"  Wells needed to reach {confidence_target*100:.0f}% confidence: " +
          f"{wells_to_target if wells_to_target <= n_exploration_wells else 'Not reached'}")

    return results

def plot_per_well_callback(basin_gp, well_num, grid_tensor, mask=None):
    """
    Callback function for plotting exploration progress after each well.

    Args:
        basin_gp: Basin exploration model
        well_num: Current well number
        grid_tensor: Grid tensor
        mask: Optional mask
    """
    from field_visualizations import visualize_exploration_progress

    try:
        # Get grid dimensions from tensor
        resolution = int(np.sqrt(grid_tensor.shape[0]))
        x_max = torch.max(grid_tensor[:, 0]).item()
        y_max = torch.max(grid_tensor[:, 1]).item()
        x1_grid, x2_grid = np.meshgrid(
            np.linspace(0, x_max, resolution),
            np.linspace(0, y_max, resolution)
        )

        # Create directory for plots
        os.makedirs("plots/field_exploration", exist_ok=True)

        # Create exploration progress visualization for porosity
        save_path = f"plots/field_exploration/porosity_after_well_{well_num}.png"
        visualize_exploration_progress(
            None,  # Pass None instead of grid_tensor to avoid issues
            x1_grid, x2_grid, basin_gp,
            property_idx=0,  # Porosity
            well_num=well_num,
            basin_size=(20, 20),
            save_path=save_path,
            show_plot=False
        )
    except Exception as e:
        print(f"Warning: Could not create visualization for well {well_num}: {e}")
        # Continue execution even if visualization fails

def main(show_plots=False, profit_threshold=30, length_scale=None, confidence_target=0.9):
    """
    Main function to demonstrate field GP-based exploration using VOI strategy.

    Args:
        show_plots: Whether to display plots during execution
        profit_threshold: Minimum profit in millions $ to be confident about
        length_scale: Optional length scale parameter for GP kernel
        confidence_target: Target confidence level (0-1)
    """
    from field_visualizations import (
        visualize_field_properties,
        visualize_economic_model,
        visualize_exploration_progress,
        plot_confidence_progression,
        plot_field_summary
    )

    print("Loading field data...")
    df_full = load_field_data(show_plots=False)  # Always load data without showing plots initially

    # Basin parameters (use these consistently across all functions)
    basin_size = (20, 20)  # 20 km x 20 km basin
    resolution = 30  # Grid resolution
    plt.rcParams['figure.max_open_warning'] = 0

    # Create output directory for plots
    os.makedirs("plots/field_exploration", exist_ok=True)

    # Visualize the field geology
    print("\nVisualizing field geology...")
    grid_tensor, x1_grid, x2_grid, thickness, porosity, permeability, toc, saturation, clay, depth = \
        visualize_field_geology(basin_size, resolution, show_plots=False)  # Always get the data

    # If plots are enabled, create enhanced visualization
    if show_plots:
        field_data = {
            'thickness': thickness,
            'porosity': porosity,
            'permeability': permeability,
            'toc': toc,
            'water_saturation': saturation,
            'clay_volume': clay
        }

        fig = visualize_field_properties(
            grid_tensor, x1_grid, x2_grid, field_data, basin_size,
            save_path="plots/field_exploration/field_properties.png",
            show_plot=show_plots
        )

    # Define economic parameters for exploration planning
    economic_params = {
        'area': 1.0e6,  # m²
        'water_saturation': 0.5,  # Use average from field data
        'formation_volume_factor': 1.1,
        'oil_price': 80,  # $ per barrel
        'drilling_cost': 8e6,  # $
        'completion_cost': 4e6,  # $
        'profit_threshold': profit_threshold * 1e6  # Convert to dollars
    }

    # Show economic model visualization if plots are enabled
    if show_plots:
        fig = visualize_economic_model(
            economic_params,
            save_path="plots/field_exploration/economic_model.png",
            show_plot=show_plots
        )

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

    # Configuration for the VOI exploration
    n_initial_wells = 5
    n_exploration_wells = 10

    # Display exploration settings
    print(f"\nExploration Settings:")
    print(f"  Profit Threshold: ${profit_threshold}M")
    print(f"  Confidence Target: {confidence_target*100:.0f}%")
    print(f"  Initial Wells: {n_initial_wells}")
    print(f"  Max Exploration Wells: {n_exploration_wells}")
    if length_scale is not None:
        print(f"  GP Length Scale: {length_scale}")

    # Run VOI exploration with callback for plotting
    callback = plot_per_well_callback if show_plots else None

    # Run the VOI exploration
    results = run_voi_exploration(
        basin_size=basin_size,
        resolution=resolution,
        grid_tensor=grid_tensor,
        n_initial_wells=n_initial_wells,
        n_exploration_wells=n_exploration_wells,
        true_funcs=true_funcs,
        economic_params=economic_params,
        profit_threshold=profit_threshold,
        confidence_target=confidence_target,
        show_plots=show_plots,
        plot_callback=callback,
        length_scale=length_scale
    )

    # Create additional visualizations if plots are enabled
    if show_plots:
        # Plot confidence progression
        fig = plot_confidence_progression(
            results['confidence_history'],
            confidence_target=confidence_target,
            save_path="plots/field_exploration/confidence_progression.png",
            show_plot=show_plots
        )

        # Plot final exploration results
        try:
            fig = visualize_exploration_progress(
                None,  # Pass None instead of grid_tensor to avoid issues
                x1_grid, x2_grid, results['model'],
                property_idx=0,  # Porosity
                well_num=len(results['model'].wells) - n_initial_wells,  # Number of exploration wells
                basin_size=basin_size,
                save_path="plots/field_exploration/final_exploration.png",
                show_plot=show_plots
            )
        except Exception as e:
            print(f"Warning: Could not create final exploration visualization: {e}")

        # Create summary visualization
        try:
            fig = plot_field_summary(
                results, basin_size, x1_grid, x2_grid,
                save_path="plots/field_exploration/exploration_summary.png",
                show_plot=show_plots
            )
        except Exception as e:
            print(f"Warning: Could not create summary visualization: {e}")

    # Summary of well requirements
    wells_to_target = results['wells_to_target']
    if wells_to_target <= n_exploration_wells:
        print(f"\nTarget confidence of {confidence_target*100:.0f}% reached after {wells_to_target} wells")
    else:
        print(f"\nTarget confidence of {confidence_target*100:.0f}% NOT reached after {n_exploration_wells} wells")

    # Calculate final statistics
    voi_model = results['model']

    # Get predictions from VOI model
    mean, std = voi_model.predict(grid_tensor)

    # Print key statistics from final model
    total_porosity = torch.mean(mean[:, 0]).item()
    total_permeability = torch.mean(mean[:, 1]).item()
    total_thickness = torch.mean(mean[:, 2]).item()

    print("\nFinal Field Model Statistics:")
    print(f"  Average Porosity: {total_porosity:.3f}")
    print(f"  Average Permeability: {total_permeability:.1f} mD")
    print(f"  Average Thickness: {total_thickness:.1f} ft")
    print(f"  Final Confidence in Profitability: {voi_model.profitability_confidence*100:.1f}%")

    # Total cost of exploration
    drilling_cost = economic_params['drilling_cost']
    completion_cost = economic_params['completion_cost']
    total_well_cost = (drilling_cost + completion_cost) * len(voi_model.wells)

    print(f"  Total Exploration Cost: ${total_well_cost/1e6:.1f}M")

    if show_plots:
        print("\nVisualization plots created in plots/field_exploration/ directory")
        print("The following plots are available:")
        print("  - field_properties.png: Visualizes the field geological properties")
        print("  - economic_model.png: Illustrates the economic model")
        print("  - confidence_progression.png: Shows how confidence increases with wells")
        print("  - final_exploration.png: Final model predictions and uncertainty")
        print("  - exploration_summary.png: Complete summary of exploration results")
        print("  - porosity_after_well_X.png: Progress after each well (if generated)")

    print("\nField exploration simulation complete.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Field exploration using Value of Information (VOI) strategy')
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help='Show plots during execution (default: False)')
    parser.add_argument('--profit-threshold', type=float, default=30,
                        help='Profit threshold in millions $ (default: 30)')
    parser.add_argument('--confidence-target', type=float, default=0.9,
                        help='Target confidence level 0-1 (default: 0.9)')
    parser.add_argument('--length-scale', type=float, default=None,
                        help='Optional length scale parameter for GP kernel (default: None)')
    parser.add_argument('--output-dir', type=str, default='plots/field_exploration',
                        help='Directory for saving output plots (default: plots/field_exploration)')

    args = parser.parse_args()

    # Create output directory if provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Run the main function with parsed arguments
    main(
        show_plots=args.show_plots,
        profit_threshold=args.profit_threshold,
        length_scale=args.length_scale,
        confidence_target=args.confidence_target
    )