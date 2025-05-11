"""
Visualization module for length-scale sensitivity analysis.

This module provides functions to visualize the results of length-scale
sensitivity analysis, showing the relationship between length scales and
exploration efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_length_scale_impact(results, basin_size=(20, 20), resolution=30, 
                            figsize=(15, 10), save_path=None, show_plot=True):
    """
    Plot the relationship between length scale and wells required.
    
    Args:
        results: Results dictionary from analyze_length_scale_sensitivity
        basin_size: Size of the basin in (x, y) kilometers
        resolution: Grid resolution for visualizing well distribution
        figsize: Size of the figure in inches
        save_path: Path to save the figure (if None, figure is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        Figure object
    """
    length_scales = sorted(list(results.keys()))
    
    # Calculate average wells required for each length scale
    avg_wells = []
    std_wells = []
    for ls in length_scales:
        wells_required = results[ls]['wells_required']
        avg_wells.append(np.mean(wells_required))
        std_wells.append(np.std(wells_required))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Plot 1: Length scale vs. Wells required
    ax1 = plt.subplot(gs[0, 0])
    ax1.errorbar(length_scales, avg_wells, yerr=std_wells, marker='o', linestyle='-', 
               capsize=5, markersize=8, linewidth=2)
    ax1.set_xlabel('Length Scale Parameter')
    ax1.set_ylabel('Average Wells Required')
    ax1.set_title('Length Scale Impact on Wells Required')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Uncertainty curves
    ax2 = plt.subplot(gs[0, 1])
    for i, ls in enumerate(length_scales):
        # Average uncertainty curve across simulations
        all_curves = results[ls]['uncertainty_curves']
        # Find the max length
        max_len = max(len(curve) for curve in all_curves)
        # Pad shorter curves with NaNs
        padded_curves = []
        for curve in all_curves:
            if len(curve) < max_len:
                padded = curve + [np.nan] * (max_len - len(curve))
            else:
                padded = curve
            padded_curves.append(padded)
        
        # Convert to numpy array
        curves_array = np.array(padded_curves)
        # Calculate mean uncertainty at each well (ignoring NaNs)
        mean_curve = np.nanmean(curves_array, axis=0)
        # Plot uncertainty curve
        wells = range(1, len(mean_curve)+1)
        ax2.plot(wells, mean_curve, marker='o', label=f'LS = {ls}')
    
    ax2.set_xlabel('Number of Wells')
    ax2.set_ylabel('Mean Uncertainty')
    ax2.set_title('Uncertainty Reduction vs. Wells Drilled')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Plot 3: Well distribution maps for smallest and largest length scales
    ax3 = plt.subplot(gs[1, 0])
    _plot_well_distribution(ax3, results[length_scales[0]], basin_size, resolution, f'LS = {length_scales[0]}')
    
    ax4 = plt.subplot(gs[1, 1])
    _plot_well_distribution(ax4, results[length_scales[-1]], basin_size, resolution, f'LS = {length_scales[-1]}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def _plot_well_distribution(ax, length_scale_results, basin_size, resolution, title):
    """Helper function to plot well distribution for a specific length scale."""
    # Create grid
    x1 = np.linspace(0, basin_size[0], resolution)
    x2 = np.linspace(0, basin_size[1], resolution)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    
    # Create a heatmap of well density
    well_density = np.zeros((resolution, resolution))
    
    # Collect all well locations
    all_locations = []
    for sim_locations in length_scale_results['exploration_maps']:
        all_locations.extend(sim_locations)
    
    # Convert to numpy array
    locations = np.array(all_locations)
    
    if len(locations) > 0:
        # Count wells in each grid cell
        for loc in locations:
            # Find appropriate grid cell
            x_idx = int(np.floor(loc[0] / basin_size[0] * (resolution-1)))
            y_idx = int(np.floor(loc[1] / basin_size[1] * (resolution-1)))
            # Ensure indices are within bounds
            x_idx = min(max(x_idx, 0), resolution-1)
            y_idx = min(max(y_idx, 0), resolution-1)
            well_density[y_idx, x_idx] += 1
    
    # Plot well density
    im = ax.contourf(x1_grid, x2_grid, well_density, levels=10, cmap='YlOrRd')
    ax.set_xlabel('X Distance (km)')
    ax.set_ylabel('Y Distance (km)')
    ax.set_title(f'Well Distribution: {title}')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Well Density')
    
    return ax

def visualize_geological_smoothness(basin_size=(20, 20), resolution=30, 
                                    smoothness_values=[0.2, 0.5, 1.0, 2.0, 5.0],
                                    figsize=(15, 10), save_path=None, show_plot=True):
    """
    Visualize how different smoothness parameters affect the geological properties.
    
    Args:
        basin_size: Size of the basin in (x, y) kilometers
        resolution: Grid resolution for visualization
        smoothness_values: List of smoothness values to visualize
        figsize: Size of the figure in inches
        save_path: Path to save the figure (if None, figure is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        Figure object
    """
    from basin_gp.simulation import create_basin_grid, true_porosity, true_permeability, true_thickness
    
    # Create grid
    grid_tensor, x1_grid, x2_grid, _ = create_basin_grid(basin_size, resolution)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    n_props = 3  # Porosity, permeability, thickness
    n_smooth = len(smoothness_values)
    gs = gridspec.GridSpec(n_props, n_smooth)
    
    # Property titles and colormaps
    prop_names = ['Porosity', 'Permeability (mD)', 'Thickness (m)']
    prop_funcs = [true_porosity, true_permeability, true_thickness]
    cmaps = ['viridis', 'plasma', 'cividis']
    
    # Plot each property with different smoothness values
    for i, (prop_name, prop_func, cmap) in enumerate(zip(prop_names, prop_funcs, cmaps)):
        for j, smoothness in enumerate(smoothness_values):
            ax = plt.subplot(gs[i, j])
            
            # Calculate property values with this smoothness
            prop_values = prop_func(grid_tensor, basin_size, smoothness).reshape(resolution, resolution).numpy()
            
            # Plot
            im = ax.contourf(x1_grid, x2_grid, prop_values, levels=20, cmap=cmap)
            
            # Add labels
            if i == 0:
                ax.set_title(f'Smoothness = {smoothness}')
            if j == 0:
                ax.set_ylabel(prop_name)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def visualize_length_scale_comparison(results, property_idx=0, 
                                    basin_size=(20, 20), resolution=30,
                                    figsize=(15, 10), save_path=None, show_plot=True):
    """
    Visualize model predictions for different length scales after the same number of wells.
    
    Args:
        results: Results dictionary from analyze_length_scale_sensitivity
        property_idx: Index of property to visualize (0=porosity, 1=permeability, 2=thickness)
        basin_size: Size of the basin in (x, y) kilometers
        resolution: Grid resolution for visualization
        figsize: Size of the figure in inches
        save_path: Path to save the figure (if None, figure is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        Figure object
    """
    from basin_gp.simulation import create_basin_grid, true_porosity, true_permeability, true_thickness
    
    # Create grid
    grid_tensor, x1_grid, x2_grid, _ = create_basin_grid(basin_size, resolution)
    
    # Get property function
    property_names = ['Porosity', 'Permeability', 'Thickness']
    property_funcs = [true_porosity, true_permeability, true_thickness]
    property_name = property_names[property_idx]
    property_func = property_funcs[property_idx]
    
    # Calculate ground truth
    ground_truth = property_func(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    
    # Sort length scales
    length_scales = sorted(list(results.keys()))
    
    # Determine number of wells to compare
    min_wells_required = []
    for ls in length_scales:
        avg_wells = np.mean(results[ls]['wells_required'])
        min_wells_required.append(int(avg_wells))
    
    # Use the minimum number to ensure fair comparison
    n_wells_to_compare = min(min_wells_required)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    n_cols = len(length_scales) + 1  # +1 for ground truth
    gs = gridspec.GridSpec(2, n_cols)
    
    # Plot ground truth
    ax_truth = plt.subplot(gs[0, 0])
    im_truth = ax_truth.contourf(x1_grid, x2_grid, ground_truth, levels=20, cmap='viridis')
    ax_truth.set_title(f'True {property_name}')
    divider = make_axes_locatable(ax_truth)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im_truth, cax=cax)
    
    # Plot predictions for each length scale
    for i, ls in enumerate(length_scales):
        # Check if we have raw_data with models
        if results[ls]['raw_data'] and 'model' in results[ls]['raw_data'][0]:
            # Get model from first simulation after n_wells_to_compare wells
            sim_results = results[ls]['raw_data'][0]  # First simulation
            model = sim_results['model']

            # Get predictions
            mean, std = model.predict(grid_tensor)
            mean_prop = mean[:, property_idx].reshape(resolution, resolution).numpy()
            std_prop = std[:, property_idx].reshape(resolution, resolution).numpy()
        else:
            # Create dummy data for visualization
            print(f"No model data for length scale {ls}, using placeholder visualization")
            mean_prop = np.random.rand(resolution, resolution) * 0.2 + 0.1
            std_prop = np.random.rand(resolution, resolution) * 0.1
        
        # Plot mean prediction
        ax_mean = plt.subplot(gs[0, i+1])
        im_mean = ax_mean.contourf(x1_grid, x2_grid, mean_prop, levels=20, cmap='viridis')
        ax_mean.set_title(f'LS = {ls}: Mean {property_name}')
        divider = make_axes_locatable(ax_mean)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im_mean, cax=cax)
        
        # Plot uncertainty
        ax_std = plt.subplot(gs[1, i+1])
        im_std = ax_std.contourf(x1_grid, x2_grid, std_prop, levels=20, cmap='magma')
        ax_std.set_title(f'LS = {ls}: Uncertainty')
        divider = make_axes_locatable(ax_std)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im_std, cax=cax)
        
        # Plot well locations
        well_locations = sim_results['well_locations'][:n_wells_to_compare]
        for loc in well_locations:
            ax_mean.scatter(loc[0], loc[1], color='red', edgecolor='black', s=50)
            ax_std.scatter(loc[0], loc[1], color='red', edgecolor='black', s=50)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def plot_length_scale_impact_3d(results, figsize=(12, 10), save_path=None, show_plot=True):
    """
    Create a 3D visualization of the length scale impact on wells required.
    
    Args:
        results: Results dictionary from analyze_length_scale_sensitivity
        figsize: Size of the figure in inches
        save_path: Path to save the figure (if None, figure is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        Figure object
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Sort length scales
    length_scales = sorted(list(results.keys()))
    
    # Calculate average wells required for each length scale
    avg_wells = []
    std_wells = []
    for ls in length_scales:
        wells_required = results[ls]['wells_required']
        avg_wells.append(np.mean(wells_required))
        std_wells.append(np.std(wells_required))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create x, y coordinates for 3D surface
    x_smooth = np.linspace(min(length_scales), max(length_scales), 100)
    y_smooth = np.linspace(1, 10, 100)  # Number of wells
    X, Y = np.meshgrid(x_smooth, y_smooth)
    
    # Create Z values (uncertainty)
    Z = np.zeros_like(X)
    
    # For each length scale, interpolate uncertainty curve
    for i, ls in enumerate(length_scales):
        # Average uncertainty curve across simulations
        all_curves = results[ls]['uncertainty_curves']
        # Find the max length
        max_len = max(len(curve) for curve in all_curves)
        # Pad shorter curves with NaNs
        padded_curves = []
        for curve in all_curves:
            if len(curve) < max_len:
                padded = curve + [np.nan] * (max_len - len(curve))
            else:
                padded = curve
            padded_curves.append(padded)
        
        # Convert to numpy array
        curves_array = np.array(padded_curves)
        # Calculate mean uncertainty at each well (ignoring NaNs)
        mean_curve = np.nanmean(curves_array, axis=0)
        
        # Interpolate to match y_smooth values
        from scipy.interpolate import interp1d
        wells = np.arange(1, len(mean_curve)+1)
        if len(wells) >= 2:  # Need at least 2 points for interpolation
            f = interp1d(wells, mean_curve, bounds_error=False, fill_value=np.nan)
            Z_col = f(y_smooth)
            
            # Find index of current length scale in x_smooth
            idx = np.abs(x_smooth - ls).argmin()
            
            # Set Z values for this length scale
            Z[:, idx] = Z_col
    
    # Plot 3D surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Set labels
    ax.set_xlabel('Length Scale')
    ax.set_ylabel('Number of Wells')
    ax.set_zlabel('Mean Uncertainty')
    ax.set_title('Length Scale Impact on Uncertainty Reduction')
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Uncertainty')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig