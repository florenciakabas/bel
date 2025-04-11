"""
Basin Exploration Example using Gaussian Process modeling.

This script demonstrates how to use the basin_gp module for modeling 
geological properties and planning exploration wells. It creates
publication-quality plots suitable for presentation slides.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from basin_gp.model import BasinExplorationGP
from basin_gp.simulation import (
    create_basin_grid, visualize_true_geology, add_random_wells,
    true_porosity, true_permeability, true_thickness, calculate_resources,
    prior_permeability, prior_porosity, prior_thickness, add_knowledge_driven_wells
)
from plot_styler import PlotStyler

def plot_basin_geology(grid_tensor, x1_grid, x2_grid, data, titles, cmaps, fig_title, wells=None, annotations=None, figsize=(18, 8)):
    """
    Create a professional-quality visualization of basin geology.
    
    Args:
        grid_tensor: Grid of points as tensor
        x1_grid: x-coordinates meshgrid
        x2_grid: y-coordinates meshgrid
        data: List of data arrays to plot
        titles: List of plot titles 
        cmaps: List of colormaps
        fig_title: Figure title
        wells: List of well locations to overlay
        annotations: Optional annotations to add to plots
        figsize: Figure size
    """
    styler = PlotStyler()
    resolution = x1_grid.shape[0]
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    fig.suptitle(fig_title, fontsize=16, fontweight='bold', y=0.95)
    
    for i, (d, title, cmap) in enumerate(zip(data, titles, cmaps)):
        im = axes[i].contourf(x1_grid, x2_grid, d.reshape(resolution, resolution), levels=20, cmap=cmap, alpha=0.9)
        styler.apply_style(axes[i], title, "X Distance (km)", "Y Distance (km)")
        
        # Add colorbar with custom styling
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10)
        
        # Add grid lines for clarity
        axes[i].grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Add wells if provided
        if wells is not None:
            well_x = [well['location'][0] for well in wells]
            well_y = [well['location'][1] for well in wells]
            initial_wells = [well for well in wells if 'Initial' in well['name']]
            exploration_wells = [well for well in wells if 'Initial' not in well['name']]
            
            # Plot initial wells as black X
            if initial_wells:
                init_x = [well['location'][0] for well in initial_wells]
                init_y = [well['location'][1] for well in initial_wells]
                axes[i].scatter(init_x, init_y, color='black', s=100, marker='x', linewidth=2, 
                              label='Initial Wells', zorder=10)
            
            # Plot exploration wells as red circles with numbers
            if exploration_wells:
                for well in exploration_wells:
                    axes[i].scatter(well['location'][0], well['location'][1], color='red', s=150, 
                                  marker='o', edgecolor='black', linewidth=1, alpha=0.7, zorder=10)
                    # Add well number inside the circle
                    try:
                        well_num = int(well['name'].split('_')[-1])
                        axes[i].text(well['location'][0], well['location'][1], f"{well_num}", 
                                   ha='center', va='center', color='white', fontweight='bold', zorder=11)
                    except ValueError:
                        pass
            
            if i == len(data) - 1:  # Add legend to last plot
                if initial_wells and exploration_wells:
                    axes[i].legend(loc='lower right', framealpha=0.9)
                    
        # Add annotations if provided
        if annotations and i < len(annotations) and annotations[i]:
            for ann in annotations[i]:
                axes[i].annotate(ann['text'], xy=(ann['x'], ann['y']), xytext=(ann['tx'], ann['ty']),
                               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                               fontsize=12, fontweight='bold', backgroundcolor='white', alpha=0.8)
    
    plt.tight_layout()
    return fig

def compare_prior_and_true_geology(grid_tensor, x1_grid, x2_grid, basin_size, resolution, wells=None):
    """Create comparison plots of prior beliefs versus true geology"""
    
    # Calculate true values for visualization
    true_por = true_porosity(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    true_perm = true_permeability(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    true_thick = true_thickness(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    
    # Calculate prior beliefs
    prior_por = prior_porosity(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    prior_perm = prior_permeability(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    prior_thick = prior_thickness(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    
    # Porosity comparison
    fig1 = plot_basin_geology(
        grid_tensor, x1_grid, x2_grid,
        [prior_por, true_por],
        ["Prior Porosity Model", "True Porosity Distribution"],
        ["viridis", "viridis"],
        "Porosity - Prior Belief vs. True Geology",
        wells=wells,
        annotations=[
            [{'text': 'Expected Sweet Spot', 'x': 15, 'y': 15, 'tx': 17, 'ty': 17}],
            [{'text': 'Actual Sweet Spots', 'x': 5, 'y': 15, 'tx': 3, 'ty': 17},
             {'text': 'Secondary Sweet Spot', 'x': 15, 'y': 8, 'tx': 17, 'ty': 6}]
        ]
    )
    
    # Permeability comparison
    fig2 = plot_basin_geology(
        grid_tensor, x1_grid, x2_grid,
        [prior_perm, true_perm],
        ["Prior Permeability Model", "True Permeability Distribution"],
        ["plasma", "plasma"],
        "Permeability - Prior Belief vs. True Geology",
        wells=wells,
        annotations=[
            [{'text': 'Expected Fault Zone', 'x': 10, 'y': 10, 'tx': 12, 'ty': 8}],
            [{'text': 'Actual Fault Zone', 'x': 10, 'y': 10, 'tx': 8, 'ty': 8}]
        ]
    )
    
    # Thickness comparison
    fig3 = plot_basin_geology(
        grid_tensor, x1_grid, x2_grid,
        [prior_thick, true_thick],
        ["Prior Thickness Model", "True Thickness Distribution"],
        ["cividis", "cividis"],
        "Thickness - Prior Belief vs. True Geology",
        wells=wells,
        annotations=[
            [{'text': 'Expected Structural High', 'x': 10, 'y': 10, 'tx': 13, 'ty': 11}],
            [{'text': 'Actual Structural High', 'x': 10, 'y': 10, 'tx': 13, 'ty': 11}]
        ]
    )
    
    plt.show()
    return fig1, fig2, fig3

def plot_gp_model_evolution(basin_gp, grid_tensor, x1_grid, x2_grid, property_idx=0, property_name="Porosity"):
    """
    Show the evolution of the GP model with all wells.
    
    Args:
        basin_gp: The basin exploration model (already fit)
        grid_tensor: Grid points to predict
        x1_grid, x2_grid: Coordinate grids for plotting
        property_idx: Index of property to show (0=porosity, 1=permeability, 2=thickness)
        property_name: Name of the property
    """
    resolution = x1_grid.shape[0]
    styler = PlotStyler()
    
    # Get predictions
    mean, std = basin_gp.predict(grid_tensor)
    
    # Extract the specific property
    mean_property = mean[:, property_idx].reshape(resolution, resolution).numpy()
    std_property = std[:, property_idx].reshape(resolution, resolution).numpy()
    
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Gaussian Process Model for {property_name}", fontsize=16, fontweight='bold')
    
    # Plot mean prediction
    im1 = axes[0].contourf(x1_grid, x2_grid, mean_property, levels=20, cmap='viridis', alpha=0.9)
    styler.apply_style(axes[0], f"Mean {property_name} Prediction", "X Distance (km)", "Y Distance (km)")
    
    # Add colorbar
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    
    # Plot uncertainty (standard deviation)
    im2 = axes[1].contourf(x1_grid, x2_grid, std_property, levels=20, cmap='magma', alpha=0.9)
    styler.apply_style(axes[1], f"Uncertainty (Std Dev)", "X Distance (km)", "Y Distance (km)")
    
    # Add colorbar
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    
    # Add wells
    initial_wells = [well for well in basin_gp.wells if 'Initial' in well['name']]
    exploration_wells = [well for well in basin_gp.wells if 'Initial' not in well['name']]
    
    # Plot initial wells
    if initial_wells:
        for ax in axes:
            x = [well['location'][0] for well in initial_wells]
            y = [well['location'][1] for well in initial_wells]
            ax.scatter(x, y, color='black', s=100, marker='x', linewidth=2, label='Initial Wells', zorder=10)
    
    # Plot exploration wells with sequential numbering
    if exploration_wells:
        for well in exploration_wells:
            for ax in axes:
                ax.scatter(well['location'][0], well['location'][1], color='red', s=150, 
                          marker='o', edgecolor='black', linewidth=1, alpha=0.7, zorder=10)
                # Add well number
                try:
                    well_num = int(well['name'].split('_')[-1])
                    ax.text(well['location'][0], well['location'][1], f"{well_num}", 
                           ha='center', va='center', color='white', fontweight='bold', zorder=11)
                except ValueError:
                    pass
    
    # Add annotations for key features
    low_uncertainty_regions = [
        {'x': well['location'][0], 'y': well['location'][1], 
         'tx': well['location'][0] + 1, 'ty': well['location'][1] - 1}
        for well in basin_gp.wells[:3]  # Take first few wells
    ]
    
    for region in low_uncertainty_regions[:1]:  # Just annotate one region to avoid clutter
        axes[1].annotate('Low Uncertainty\nNear Well', 
                       xy=(region['x'], region['y']), 
                       xytext=(region['tx'], region['ty']),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                       fontsize=12, fontweight='bold', backgroundcolor='white', alpha=0.8)
    
    # Find high uncertainty region (away from wells)
    high_std_idx = np.unravel_index(np.argmax(std_property), std_property.shape)
    high_x = x1_grid[high_std_idx]
    high_y = x2_grid[high_std_idx]
    
    axes[1].annotate('Highest Uncertainty', 
                   xy=(high_x, high_y), 
                   xytext=(high_x + 2, high_y + 2),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                   fontsize=12, fontweight='bold', backgroundcolor='white', alpha=0.8)
    
    # Legend
    axes[0].legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    return fig

def plot_exploration_strategy_comparison(uncertainty_basin_gp, economic_basin_gp, 
                                        grid_tensor, x1_grid, x2_grid, true_values,
                                        property_idx=0, property_name="Porosity"):
    """
    Compare different exploration strategies with side-by-side visualization.
    
    Args:
        uncertainty_basin_gp: Basin GP model using uncertainty strategy
        economic_basin_gp: Basin GP model using economic strategy
        grid_tensor: Grid points
        x1_grid, x2_grid: Coordinate grids
        true_values: True property values on the grid
        property_idx: Index of property to visualize
        property_name: Name of the property
    """
    resolution = x1_grid.shape[0]
    styler = PlotStyler()
    
    # Get predictions from both models
    mean_uncertainty, std_uncertainty = uncertainty_basin_gp.predict(grid_tensor)
    mean_economic, std_economic = economic_basin_gp.predict(grid_tensor)
    
    # Extract property values
    mean_uncertainty_prop = mean_uncertainty[:, property_idx].reshape(resolution, resolution).numpy()
    mean_economic_prop = mean_economic[:, property_idx].reshape(resolution, resolution).numpy()
    
    # Extract true property values
    true_property = true_values[property_idx].reshape(resolution, resolution)
    
    # Prediction errors
    error_uncertainty = np.abs(mean_uncertainty_prop - true_property)
    error_economic = np.abs(mean_economic_prop - true_property)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    fig.suptitle(f"Comparison of Exploration Strategies: {property_name}", 
                fontsize=16, fontweight='bold')
    
    # True property distribution (reference)
    im1 = axes[0, 0].contourf(x1_grid, x2_grid, true_property, levels=20, cmap='viridis', alpha=0.9)
    styler.apply_style(axes[0, 0], f"True {property_name}", "X Distance (km)", "Y Distance (km)")
    
    # Add colorbar
    divider1 = make_axes_locatable(axes[0, 0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im1, cax=cax1)
    
    # Uncertainty-based prediction
    im2 = axes[0, 1].contourf(x1_grid, x2_grid, mean_uncertainty_prop, levels=20, cmap='viridis', alpha=0.9)
    styler.apply_style(axes[0, 1], f"Uncertainty Strategy: {property_name}", "X Distance (km)", "Y Distance (km)")
    
    # Add colorbar
    divider2 = make_axes_locatable(axes[0, 1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im2, cax=cax2)
    
    # Economic-based prediction
    im3 = axes[1, 0].contourf(x1_grid, x2_grid, mean_economic_prop, levels=20, cmap='viridis', alpha=0.9)
    styler.apply_style(axes[1, 0], f"Economic Strategy: {property_name}", "X Distance (km)", "Y Distance (km)")
    
    # Add colorbar
    divider3 = make_axes_locatable(axes[1, 0])
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im3, cax=cax3)
    
    # Error comparison
    error_diff = error_uncertainty - error_economic
    vmax = max(np.max(np.abs(error_diff)), 0.001)  # Ensure symmetric colorbar
    
    im4 = axes[1, 1].contourf(x1_grid, x2_grid, error_diff, levels=20, 
                            cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=0.9)
    styler.apply_style(axes[1, 1], "Error Difference (Uncertainty - Economic)", 
                     "X Distance (km)", "Y Distance (km)")
    
    # Add colorbar
    divider4 = make_axes_locatable(axes[1, 1])
    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im4, cax=cax4)
    
    # Add legend explaining the error difference
    red_patch = mpatches.Patch(color='red', label='Economic strategy more accurate')
    blue_patch = mpatches.Patch(color='blue', label='Uncertainty strategy more accurate')
    axes[1, 1].legend(handles=[red_patch, blue_patch], loc='lower right', framealpha=0.9)
    
    # Add wells to all plots
    for i in range(2):
        for j in range(2):
            # Add uncertainty strategy wells to row 0, column 1
            if i == 0 and j == 1:
                initial_wells = [well for well in uncertainty_basin_gp.wells if 'Initial' in well['name']]
                exploration_wells = [well for well in uncertainty_basin_gp.wells if 'Initial' not in well['name']]
                
                # Plot initial wells
                if initial_wells:
                    x = [well['location'][0] for well in initial_wells]
                    y = [well['location'][1] for well in initial_wells]
                    axes[i, j].scatter(x, y, color='black', s=100, marker='x', linewidth=2, 
                                    label='Initial Wells', zorder=10)
                
                # Plot exploration wells with sequential numbering
                if exploration_wells:
                    for well in exploration_wells:
                        axes[i, j].scatter(well['location'][0], well['location'][1], color='red', s=150, 
                                        marker='o', edgecolor='black', linewidth=1, alpha=0.7, zorder=10)
                        # Add well number
                        try:
                            well_num = int(well['name'].split('_')[-1])
                            axes[i, j].text(well['location'][0], well['location'][1], f"{well_num}", 
                                         ha='center', va='center', color='white', fontweight='bold', zorder=11)
                        except ValueError:
                            pass
                
                axes[i, j].legend(loc='lower right', framealpha=0.9)
            
            # Add economic strategy wells to row 1, column 0
            elif i == 1 and j == 0:
                initial_wells = [well for well in economic_basin_gp.wells if 'Initial' in well['name']]
                exploration_wells = [well for well in economic_basin_gp.wells if 'Initial' not in well['name']]
                
                # Plot initial wells
                if initial_wells:
                    x = [well['location'][0] for well in initial_wells]
                    y = [well['location'][1] for well in initial_wells]
                    axes[i, j].scatter(x, y, color='black', s=100, marker='x', linewidth=2, 
                                    label='Initial Wells', zorder=10)
                
                # Plot exploration wells with sequential numbering
                if exploration_wells:
                    for well in exploration_wells:
                        axes[i, j].scatter(well['location'][0], well['location'][1], color='green', s=150, 
                                        marker='o', edgecolor='black', linewidth=1, alpha=0.7, zorder=10)
                        # Add well number
                        try:
                            well_num = int(well['name'].split('_')[-1])
                            axes[i, j].text(well['location'][0], well['location'][1], f"{well_num}", 
                                         ha='center', va='center', color='white', fontweight='bold', zorder=11)
                        except ValueError:
                            pass
                
                axes[i, j].legend(loc='lower right', framealpha=0.9)
            
            # Add only initial wells to true distribution plot
            elif i == 0 and j == 0:
                initial_wells = [well for well in uncertainty_basin_gp.wells if 'Initial' in well['name']]
                if initial_wells:
                    x = [well['location'][0] for well in initial_wells]
                    y = [well['location'][1] for well in initial_wells]
                    axes[i, j].scatter(x, y, color='black', s=100, marker='x', linewidth=2, 
                                    label='Initial Wells', zorder=10)
                    axes[i, j].legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    return fig

def resource_assessment_visualization(uncertainty_basin_gp, economic_basin_gp, 
                                     grid_tensor, basin_size, x1_grid, x2_grid,
                                     economic_params):
    """
    Visualize and compare resource assessments from different strategies.
    
    Args:
        uncertainty_basin_gp: Basin GP model using uncertainty strategy
        economic_basin_gp: Basin GP model using economic strategy
        grid_tensor: Grid points
        basin_size: Size of the basin
        x1_grid, x2_grid: Coordinate grids
        economic_params: Economic parameters
    """
    resolution = x1_grid.shape[0]
    styler = PlotStyler()
    
    # Get predictions for both strategies
    mean_uncertainty, _ = uncertainty_basin_gp.predict(grid_tensor)
    mean_economic, _ = economic_basin_gp.predict(grid_tensor)
    
    # Calculate resource values per grid cell
    hydrocarbon_saturation = 1.0 - economic_params['water_saturation']
    formation_volume_factor = economic_params['formation_volume_factor']
    oil_price = economic_params['oil_price']
    
    # Calculate resource volume and value per cell
    def calculate_cell_resources(mean_pred):
        porosity = mean_pred[:, 0]
        permeability = mean_pred[:, 1]
        thickness = mean_pred[:, 2]
        
        # Cell area in m²
        cell_area = (basin_size[0] / (resolution-1)) * (basin_size[1] / (resolution-1)) * 1e6
        
        # Original oil in place (m³)
        ooip = cell_area * thickness * porosity * hydrocarbon_saturation / formation_volume_factor
        
        # Recovery factor
        recovery_factor = 0.1 + 0.2 * torch.log10(torch.clamp(permeability, min=1.0) / 100)
        recovery_factor = torch.clamp(recovery_factor, 0.05, 0.6)
        
        # Recoverable oil (m³)
        recoverable_oil = ooip * recovery_factor
        
        # Convert to barrels
        recoverable_barrels = recoverable_oil * 6.29
        
        # Economic value
        cell_value = recoverable_barrels * oil_price
        
        return recoverable_barrels.reshape(resolution, resolution).numpy(), cell_value.reshape(resolution, resolution).numpy()
    
    # Calculate resources for both strategies
    barrels_uncertainty, value_uncertainty = calculate_cell_resources(mean_uncertainty)
    barrels_economic, value_economic = calculate_cell_resources(mean_economic)
    
    # Create resource map visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    fig.suptitle("Resource Assessment Comparison", fontsize=16, fontweight='bold')
    
    # Recoverable barrels - Uncertainty Strategy
    im1 = axes[0, 0].contourf(x1_grid, x2_grid, barrels_uncertainty / 1e3, 
                            levels=20, cmap='YlOrRd', alpha=0.9)
    styler.apply_style(axes[0, 0], "Recoverable Oil - Uncertainty Strategy", 
                     "X Distance (km)", "Y Distance (km)")
    divider1 = make_axes_locatable(axes[0, 0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Thousands of Barrels')
    
    # Recoverable barrels - Economic Strategy
    im2 = axes[0, 1].contourf(x1_grid, x2_grid, barrels_economic / 1e3, 
                            levels=20, cmap='YlOrRd', alpha=0.9)
    styler.apply_style(axes[0, 1], "Recoverable Oil - Economic Strategy", 
                     "X Distance (km)", "Y Distance (km)")
    divider2 = make_axes_locatable(axes[0, 1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label('Thousands of Barrels')
    
    # Economic value - Uncertainty Strategy
    im3 = axes[1, 0].contourf(x1_grid, x2_grid, value_uncertainty / 1e6, 
                           levels=20, cmap='Greens', alpha=0.9)
    styler.apply_style(axes[1, 0], "Economic Value - Uncertainty Strategy", 
                     "X Distance (km)", "Y Distance (km)")
    divider3 = make_axes_locatable(axes[1, 0])
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('Millions of USD')
    
    # Economic value - Economic Strategy
    im4 = axes[1, 1].contourf(x1_grid, x2_grid, value_economic / 1e6, 
                           levels=20, cmap='Greens', alpha=0.9)
    styler.apply_style(axes[1, 1], "Economic Value - Economic Strategy", 
                     "X Distance (km)", "Y Distance (km)")
    divider4 = make_axes_locatable(axes[1, 1])
    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar4.set_label('Millions of USD')
    
    # Add wells to respective plots
    # Uncertainty strategy wells
    initial_wells = [well for well in uncertainty_basin_gp.wells if 'Initial' in well['name']]
    exploration_wells = [well for well in uncertainty_basin_gp.wells if 'Initial' not in well['name']]
    
    for well in initial_wells:
        axes[0, 0].scatter(well['location'][0], well['location'][1], color='black', s=100, 
                         marker='x', linewidth=2, zorder=10)
        axes[1, 0].scatter(well['location'][0], well['location'][1], color='black', s=100, 
                         marker='x', linewidth=2, zorder=10)
    
    for well in exploration_wells:
        axes[0, 0].scatter(well['location'][0], well['location'][1], color='red', s=150, 
                         marker='o', edgecolor='black', linewidth=1, alpha=0.7, zorder=10)
        axes[1, 0].scatter(well['location'][0], well['location'][1], color='red', s=150, 
                         marker='o', edgecolor='black', linewidth=1, alpha=0.7, zorder=10)
        try:
            well_num = int(well['name'].split('_')[-1])
            axes[0, 0].text(well['location'][0], well['location'][1], f"{well_num}", 
                          ha='center', va='center', color='white', fontweight='bold', zorder=11)
            axes[1, 0].text(well['location'][0], well['location'][1], f"{well_num}", 
                          ha='center', va='center', color='white', fontweight='bold', zorder=11)
        except ValueError:
            pass
    
    # Economic strategy wells
    initial_wells = [well for well in economic_basin_gp.wells if 'Initial' in well['name']]
    exploration_wells = [well for well in economic_basin_gp.wells if 'Initial' not in well['name']]
    
    for well in initial_wells:
        axes[0, 1].scatter(well['location'][0], well['location'][1], color='black', s=100, 
                         marker='x', linewidth=2, zorder=10)
        axes[1, 1].scatter(well['location'][0], well['location'][1], color='black', s=100, 
                         marker='x', linewidth=2, zorder=10)
    
    for well in exploration_wells:
        axes[0, 1].scatter(well['location'][0], well['location'][1], color='green', s=150, 
                         marker='o', edgecolor='black', linewidth=1, alpha=0.7, zorder=10)
        axes[1, 1].scatter(well['location'][0], well['location'][1], color='green', s=150, 
                         marker='o', edgecolor='black', linewidth=1, alpha=0.7, zorder=10)
        try:
            well_num = int(well['name'].split('_')[-1])
            axes[0, 1].text(well['location'][0], well['location'][1], f"{well_num}", 
                          ha='center', va='center', color='white', fontweight='bold', zorder=11)
            axes[1, 1].text(well['location'][0], well['location'][1], f"{well_num}", 
                          ha='center', va='center', color='white', fontweight='bold', zorder=11)
        except ValueError:
            pass
    
    # Add legends
    for i in range(2):
        for j in range(2):
            if j == 0:  # Uncertainty strategy
                axes[i, j].scatter([], [], color='black', marker='x', s=100, label='Initial Wells')
                axes[i, j].scatter([], [], color='red', marker='o', s=100, label='Uncertainty Wells')
                axes[i, j].legend(loc='lower right', framealpha=0.8)
            else:  # Economic strategy
                axes[i, j].scatter([], [], color='black', marker='x', s=100, label='Initial Wells')
                axes[i, j].scatter([], [], color='green', marker='o', s=100, label='Economic Wells')
                axes[i, j].legend(loc='lower right', framealpha=0.8)
    
    # Add total resource and value information as text boxes
    total_barrels_uncertainty = calculate_resources(mean_uncertainty, resolution, basin_size)
    total_barrels_economic = calculate_resources(mean_economic, resolution, basin_size)
    
    total_value_uncertainty = total_barrels_uncertainty * economic_params['oil_price']
    total_value_economic = total_barrels_economic * economic_params['oil_price']
    
    total_cost_uncertainty = len(uncertainty_basin_gp.wells) * (economic_params['drilling_cost'] + economic_params['completion_cost'])
    total_cost_economic = len(economic_basin_gp.wells) * (economic_params['drilling_cost'] + economic_params['completion_cost'])
    
    net_value_uncertainty = total_value_uncertainty - total_cost_uncertainty
    net_value_economic = total_value_economic - total_cost_economic
    
    # Create text for uncertainty strategy (row 0, col 0 and row 1, col 0)
    uncertainty_text = (
        f"Total recoverable: {total_barrels_uncertainty/1e6:.2f}M barrels\n"
        f"Gross value: ${total_value_uncertainty/1e9:.2f}B\n"
        f"Exploration cost: ${total_cost_uncertainty/1e6:.2f}M\n"
        f"Net value: ${net_value_uncertainty/1e9:.2f}B"
    )
    
    # Create text for economic strategy (row 0, col 1 and row 1, col 1)
    economic_text = (
        f"Total recoverable: {total_barrels_economic/1e6:.2f}M barrels\n"
        f"Gross value: ${total_value_economic/1e9:.2f}B\n"
        f"Exploration cost: ${total_cost_economic/1e6:.2f}M\n"
        f"Net value: ${net_value_economic/1e9:.2f}B"
    )
    
    # Add text boxes to plots
    for j in range(2):
        text = uncertainty_text if j == 0 else economic_text
        for i in range(2):
            axes[i, j].text(0.02, 0.02, text, transform=axes[i, j].transAxes,
                          fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    return fig

def main():
    # Basin parameters
    basin_size = (20, 20)  # 20 km x 20 km basin
    resolution = 30  # Grid resolution
    plt.rcParams['figure.max_open_warning'] = 0  # Disable max figure warning
    
    # Create basin grid
    print("Creating basin model and true geology...")
    grid_tensor, x1_grid, x2_grid, true_por, true_perm, true_thick = visualize_true_geology(
        basin_size, resolution
    )
    
    # 1. SLIDE 1: True Geology Visualization
    # More elegant version of the true geology with better styling
    fig_true_geology = plot_basin_geology(
        grid_tensor, x1_grid, x2_grid,
        [true_por, true_perm, true_thick],
        ["True Porosity", "True Permeability (mD)", "True Thickness (m)"],
        ["viridis", "plasma", "cividis"],
        "Basin True Geology",
        annotations=[
            [{'text': 'Primary Sweet Spot', 'x': 5, 'y': 15, 'tx': 3, 'ty': 17},
             {'text': 'Secondary Sweet Spot', 'x': 15, 'y': 8, 'tx': 17, 'ty': 6}],
            [{'text': 'Fault Zone', 'x': 10, 'y': 10, 'tx': 8, 'ty': 8}],
            [{'text': 'Structural High', 'x': 10, 'y': 10, 'tx': 12, 'ty': 8}]
        ]
    )
    
    # Initialize the exploration framework
    basin_gp = BasinExplorationGP(
        basin_size=basin_size,
        properties=['porosity', 'permeability', 'thickness']
    )
    
    # 2. SLIDE 2: Prior Beliefs vs. True Geology
    # Add initial Wells
    n_initial_wells = 3
    print(f"\nAdding {n_initial_wells} knowledge-driven initial wells...")
    basin_gp = add_knowledge_driven_wells(
        basin_gp, 
        n_initial_wells, 
        basin_size, 
        [prior_porosity, prior_permeability, prior_thickness],
        uncertainty_weight=0.3,  # Favor value slightly over uncertainty
        seed=42
    )
    
    # Plot comparison between prior beliefs and true geology
    fig_porosity_compare, fig_perm_compare, fig_thick_compare = compare_prior_and_true_geology(
        grid_tensor, x1_grid, x2_grid, basin_size, resolution, wells=basin_gp.wells
    )
    
    # 3. SLIDE 3: Initial Model Fit
    # Fit the initial model
    print("\nFitting initial GP model...")
    basin_gp.fit(verbose=False)
    
    # Plot the initial model
    fig_initial_porosity = plot_gp_model_evolution(
        basin_gp, grid_tensor, x1_grid, x2_grid, 
        property_idx=0, property_name="Porosity"
    )
    
    fig_initial_permeability = plot_gp_model_evolution(
        basin_gp, grid_tensor, x1_grid, x2_grid, 
        property_idx=1, property_name="Permeability"
    )
    
    fig_initial_thickness = plot_gp_model_evolution(
        basin_gp, grid_tensor, x1_grid, x2_grid, 
        property_idx=2, property_name="Thickness"
    )
    
    # 4. SLIDE 4-6: Sequential Exploration - Uncertainty Strategy
    # Run uncertainty-based exploration
    n_exploration_wells = 5
    print("\nRunning uncertainty-based exploration...")
    uncertainty_basin_gp = basin_gp  # Keep reference to current state
    uncertainty_history = uncertainty_basin_gp.sequential_exploration(
        grid_tensor,
        n_exploration_wells,
        [true_porosity, true_permeability, true_thickness],
        noise_std=0.01,
        strategy='uncertainty',
        plot=False  # We'll create our own plots
    )
    
    # Plot the final model after uncertainty-based exploration
    fig_uncertainty_porosity = plot_gp_model_evolution(
        uncertainty_basin_gp, grid_tensor, x1_grid, x2_grid, 
        property_idx=0, property_name="Porosity (After Uncertainty Exploration)"
    )
    
    fig_uncertainty_permeability = plot_gp_model_evolution(
        uncertainty_basin_gp, grid_tensor, x1_grid, x2_grid, 
        property_idx=1, property_name="Permeability (After Uncertainty Exploration)"
    )
    
    fig_uncertainty_thickness = plot_gp_model_evolution(
        uncertainty_basin_gp, grid_tensor, x1_grid, x2_grid, 
        property_idx=2, property_name="Thickness (After Uncertainty Exploration)"
    )
    
    # 5. SLIDE 7-9: Sequential Exploration - Economic Strategy
    # Reset and try economic-based strategy
    basin_gp_economic = BasinExplorationGP(
        basin_size=basin_size,
        properties=['porosity', 'permeability', 'thickness']
    )
    
    # Re-add initial wells
    basin_gp_economic = add_knowledge_driven_wells(
        basin_gp_economic, 
        n_initial_wells, 
        basin_size, 
        [prior_porosity, prior_permeability, prior_thickness],
        uncertainty_weight=0.3,
        seed=42
    )
    
    # Define economic parameters for exploration planning
    economic_params = {
        'area': 1.0e6,  # m²
        'water_saturation': 0.3,
        'formation_volume_factor': 1.1,
        'oil_price': 80,  # $ per barrel
        'drilling_cost': 8e6,  # $
        'completion_cost': 4e6  # $
    }
    
    # Run economic-based exploration
    print("\nRunning economic-based exploration...")
    economic_history = basin_gp_economic.sequential_exploration(
        grid_tensor,
        n_exploration_wells,
        [true_porosity, true_permeability, true_thickness],
        noise_std=0.01,
        strategy='economic',
        economic_params=economic_params,
        plot=False
    )
    
    # Plot the final model after economic-based exploration
    fig_economic_porosity = plot_gp_model_evolution(
        basin_gp_economic, grid_tensor, x1_grid, x2_grid, 
        property_idx=0, property_name="Porosity (After Economic Exploration)"
    )
    
    fig_economic_permeability = plot_gp_model_evolution(
        basin_gp_economic, grid_tensor, x1_grid, x2_grid, 
        property_idx=1, property_name="Permeability (After Economic Exploration)"
    )
    
    fig_economic_thickness = plot_gp_model_evolution(
        basin_gp_economic, grid_tensor, x1_grid, x2_grid, 
        property_idx=2, property_name="Thickness (After Economic Exploration)"
    )
    
    # 6. SLIDE 10-12: Strategy Comparison
    # Compare exploration strategies
    fig_strategy_porosity = plot_exploration_strategy_comparison(
        uncertainty_basin_gp, basin_gp_economic,
        grid_tensor, x1_grid, x2_grid, [true_por, true_perm, true_thick],
        property_idx=0, property_name="Porosity"
    )
    
    fig_strategy_permeability = plot_exploration_strategy_comparison(
        uncertainty_basin_gp, basin_gp_economic,
        grid_tensor, x1_grid, x2_grid, [true_por, true_perm, true_thick],
        property_idx=1, property_name="Permeability"
    )
    
    fig_strategy_thickness = plot_exploration_strategy_comparison(
        uncertainty_basin_gp, basin_gp_economic,
        grid_tensor, x1_grid, x2_grid, [true_por, true_perm, true_thick],
        property_idx=2, property_name="Thickness"
    )
    
    # 7. SLIDE 13: Resource Assessment
    # Compare resource assessment between strategies
    fig_resources = resource_assessment_visualization(
        uncertainty_basin_gp, basin_gp_economic,
        grid_tensor, basin_size, x1_grid, x2_grid,
        economic_params
    )
    
    # Calculate and display final results
    print("\nFinal Results Summary:")
    
    # Calculate predicted resources for uncertainty strategy
    uncertainty_basin_gp.fit(verbose=False)
    mean_uncertainty, _ = uncertainty_basin_gp.predict(grid_tensor)
    
    total_resources_uncertainty = calculate_resources(mean_uncertainty, resolution, basin_size)
    total_value_uncertainty = total_resources_uncertainty * economic_params['oil_price']
    total_cost_uncertainty = len(uncertainty_basin_gp.wells) * (economic_params['drilling_cost'] + economic_params['completion_cost'])
    net_value_uncertainty = total_value_uncertainty - total_cost_uncertainty
    
    print("\nUncertainty Strategy Results:")
    print(f"Estimated total recoverable resources: {total_resources_uncertainty/1e6:.2f} million barrels")
    print(f"Estimated total economic value: ${total_value_uncertainty/1e9:.2f} billion")
    print(f"Total exploration cost: ${total_cost_uncertainty/1e6:.2f} million")
    print(f"Net value: ${net_value_uncertainty/1e9:.2f} billion")
    
    # Calculate predicted resources for economic strategy
    basin_gp_economic.fit(verbose=False)
    mean_economic, _ = basin_gp_economic.predict(grid_tensor)
    
    total_resources_economic = calculate_resources(mean_economic, resolution, basin_size)
    total_value_economic = total_resources_economic * economic_params['oil_price']
    total_cost_economic = len(basin_gp_economic.wells) * (economic_params['drilling_cost'] + economic_params['completion_cost'])
    net_value_economic = total_value_economic - total_cost_economic
    
    print("\nEconomic Strategy Results:")
    print(f"Estimated total recoverable resources: {total_resources_economic/1e6:.2f} million barrels")
    print(f"Estimated total economic value: ${total_value_economic/1e9:.2f} billion")
    print(f"Total exploration cost: ${total_cost_economic/1e6:.2f} million")
    print(f"Net value: ${net_value_economic/1e9:.2f} billion")
    
    # Compute strategy comparison stats
    value_diff = net_value_economic - net_value_uncertainty
    value_pct = (value_diff / net_value_uncertainty) * 100 if net_value_uncertainty != 0 else float('inf')
    
    print("\nStrategy Comparison:")
    print(f"Value difference (Economic - Uncertainty): ${value_diff/1e9:.2f} billion ({value_pct:.1f}%)")
    print("=" * 50)
    
    print("\nPlots have been generated and saved for your slide deck.")
    print("The visualizations demonstrate the entire workflow of basin exploration using Gaussian Process models.")

if __name__ == "__main__":
    main()