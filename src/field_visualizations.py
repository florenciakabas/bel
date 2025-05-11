"""
Visualization module for field GP exploration

This module provides visualization functions specifically designed for creating
publication-quality plots from field exploration data. All plots use a consistent
color scheme with purples and complementary colors.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import os

# Define a consistent color scheme
COLORS = {
    'primary': '#7400B8',      # Deep purple
    'secondary': '#6930C3',    # Purple
    'tertiary': '#5E60CE',     # Blue-purple
    'quaternary': '#5390D9',   # Blue
    'highlight': '#FFBE0B',    # Amber
    'accent': '#FF006E',       # Pink
    'background': '#F8F9FA',   # Light gray
    'text': '#212529',         # Dark gray
}

# Custom colormaps aligned with our palette
CUSTOM_CMAP1 = LinearSegmentedColormap.from_list(
    'purple_amber', [COLORS['tertiary'], COLORS['highlight']], N=256)
CUSTOM_CMAP2 = LinearSegmentedColormap.from_list(
    'purple_pink', [COLORS['secondary'], COLORS['accent']], N=256)

def set_plotting_style():
    """Set consistent plotting style for all visualizations"""
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.facecolor'] = COLORS['background']
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

def visualize_field_properties(grid_tensor, x1_grid, x2_grid, field_data, 
                              basin_size, figsize=(16, 10), cmap='plasma',
                              save_path=None, show_plot=True):
    """
    Create a visualization of important field properties, suitable for presentation slides.
    
    Args:
        grid_tensor: Grid of points [n_points, 2]
        x1_grid, x2_grid: Coordinate meshgrids
        field_data: Dictionary with field properties (thickness, porosity, etc.)
        basin_size: Size of the basin (width, height)
        figsize: Figure size
        cmap: Colormap to use
        save_path: Path to save the figure
        show_plot: Whether to display the plot
    
    Returns:
        Figure object
    """
    set_plotting_style()
    resolution = x1_grid.shape[0]
    
    # Unpack field data
    thickness = field_data.get('thickness')
    porosity = field_data.get('porosity')
    permeability = field_data.get('permeability')
    toc = field_data.get('toc')
    water_saturation = field_data.get('water_saturation')
    clay_volume = field_data.get('clay_volume')
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Property titles and cmaps
    properties = [
        ('Thickness (ft)', thickness, 'plasma'),
        ('Porosity (fraction)', porosity, 'viridis'),
        ('Permeability (mD)', permeability, 'magma'),
        ('TOC (wt%)', toc, 'inferno'),
        ('Water Saturation (fraction)', water_saturation, 'cividis'),
        ('Clay Volume (fraction)', clay_volume, 'plasma'),
    ]
    
    for i, (title, data, cmap_name) in enumerate(properties):
        # Skip if data is not available
        if data is None:
            continue
            
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Plot the property
        im = ax.contourf(x1_grid, x2_grid, data, levels=20, cmap=cmap_name, alpha=0.9)
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        
        # Set labels and title
        ax.set_xlabel('X Distance (km)')
        ax.set_ylabel('Y Distance (km)')
        ax.set_title(title, fontweight='bold', pad=10)
        
        # Add basin extent
        ax.set_xlim(0, basin_size[0])
        ax.set_ylim(0, basin_size[1])
        
    fig.suptitle('Field Geological Properties', fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if not show_plot:
        plt.close(fig)
        
    return fig

def visualize_economic_model(economic_params, figsize=(12, 10), save_path=None, show_plot=True):
    """
    Create a visual representation of the economic model parameters.
    
    Args:
        economic_params: Dictionary with economic parameters
        figsize: Figure size
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        
    Returns:
        Figure object
    """
    set_plotting_style()
    
    # Extract parameters with fallbacks
    area = economic_params.get('area', 1.0e6)  # m²
    water_saturation = economic_params.get('water_saturation', 0.3)
    formation_volume_factor = economic_params.get('formation_volume_factor', 1.1)
    oil_price = economic_params.get('oil_price', 80)  # $ per barrel
    drilling_cost = economic_params.get('drilling_cost', 8e6)  # $
    completion_cost = economic_params.get('completion_cost', 4e6)  # $
    profit_threshold = economic_params.get('profit_threshold', 30e6)  # $
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Layout
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5], hspace=0.3, wspace=0.25)
    
    # 1. Parameter table - formatted as a visualization
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('tight')
    ax1.axis('off')
    
    param_names = [
        'Well Drainage Area',
        'Water Saturation',
        'Formation Volume Factor',
        'Oil Price',
        'Drilling Cost',
        'Completion Cost',
        'Profit Threshold'
    ]
    
    param_values = [
        f'{area/1e6:.1f} km²',
        f'{water_saturation:.1%}',
        f'{formation_volume_factor:.2f}',
        f'${oil_price:.2f}/bbl',
        f'${drilling_cost/1e6:.1f}M',
        f'${completion_cost/1e6:.1f}M',
        f'${profit_threshold/1e6:.1f}M'
    ]
    
    # Create table data
    table_data = [[name, value] for name, value in zip(param_names, param_values)]
    
    # Create table
    table = ax1.table(
        cellText=table_data,
        colLabels=['Parameter', 'Value'],
        loc='center',
        cellLoc='left',
        colWidths=[0.5, 0.5]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    
    # Style header cells
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor(COLORS['primary'])
        else:
            cell.set_edgecolor('lightgray')
            
            if key[1] == 0:  # Parameter name column
                cell.set_text_props(fontweight='bold')
    
    ax1.set_title('Economic Model Parameters', fontweight='bold', fontsize=14, pad=20)
    
    # 2. Revenue vs. Cost visualization
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Generate a range of porosity, thickness values
    porosity_range = np.linspace(0.05, 0.3, 10)
    thickness_range = np.linspace(20, 150, 10)
    
    # Create a mesh grid
    porosity_mesh, thickness_mesh = np.meshgrid(porosity_range, thickness_range)
    
    # Calculate revenue for each point on the grid
    hydrocarbon_saturation = 1.0 - water_saturation
    cell_volume = thickness_mesh * area  # m³
    oil_volume = cell_volume * porosity_mesh * hydrocarbon_saturation / formation_volume_factor
    barrels = oil_volume * 6.29  # 1 m³ ≈ 6.29 barrels
    revenue = barrels * oil_price
    
    # Plot revenue contours
    contour = ax2.contourf(
        porosity_mesh, thickness_mesh, revenue/1e6, 
        levels=20, cmap='plasma', alpha=0.9
    )
    
    # Add colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_label('Revenue (Million $)', rotation=90, labelpad=15)
    
    # Add labels
    ax2.set_xlabel('Porosity (fraction)')
    ax2.set_ylabel('Thickness (ft)')
    ax2.set_title('Revenue Potential', fontweight='bold', fontsize=14)
    
    # Add cost line (drilling + completion)
    total_cost = drilling_cost + completion_cost
    # Draw horizontal line for the break-even point
    break_even_thickness = {}
    for p_idx, p in enumerate(porosity_range):
        for t_idx, t in enumerate(thickness_range):
            if revenue[t_idx, p_idx] >= total_cost:
                break_even_thickness[p] = t
                break
    
    # Plot break-even line
    if break_even_thickness:
        p_vals = sorted(break_even_thickness.keys())
        t_vals = [break_even_thickness[p] for p in p_vals]
        ax2.plot(p_vals, t_vals, 'r--', linewidth=2, label=f'Break-even (${total_cost/1e6:.1f}M)')
        ax2.legend(loc='upper right')
    
    # 3. Value calculation examples
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create sample scenarios
    scenarios = [
        ('Poor', 0.08, 40, 0.4),    # Poor scenario: low porosity, thin reservoir, high water sat
        ('Average', 0.15, 80, 0.3),  # Average scenario
        ('Good', 0.25, 120, 0.2),    # Good scenario: high porosity, thick reservoir, low water sat
    ]
    
    # Calculate metrics for each scenario
    x_pos = np.arange(len(scenarios))
    width = 0.2
    metrics = []
    
    for name, por, thk, sw in scenarios:
        # Calculate original oil in place
        ooip = area * thk * por * (1 - sw) / formation_volume_factor
        recoverable = ooip * (0.2 + 0.1 * por)  # Recovery factor increases with porosity
        barrels = recoverable * 6.29
        revenue = barrels * oil_price
        profit = revenue - total_cost
        
        metrics.append({
            'name': name,
            'ooip': ooip,
            'recoverable': recoverable,
            'barrels': barrels,
            'revenue': revenue,
            'cost': total_cost,
            'profit': profit
        })
    
    # Create a condensed bar chart of the economics
    revenue_vals = [m['revenue']/1e6 for m in metrics]
    cost_vals = [m['cost']/1e6 for m in metrics]
    profit_vals = [m['profit']/1e6 for m in metrics]
    barrels_vals = [m['barrels']/1e6 for m in metrics]
    
    # Plot the bars
    ax3.bar(x_pos - width, revenue_vals, width, label='Revenue', color=COLORS['highlight'])
    ax3.bar(x_pos, cost_vals, width, label='Costs', color=COLORS['secondary'])
    ax3.bar(x_pos + width, profit_vals, width, label='Profit', color=COLORS['accent'])
    
    # Add a horizontal line for the profit threshold
    ax3.axhline(y=profit_threshold/1e6, color='r', linestyle='--', 
                label=f'Profit Threshold (${profit_threshold/1e6:.1f}M)')
    
    # Add barrel values as text on top of revenue bars
    for i, barrels in enumerate(barrels_vals):
        ax3.text(x_pos[i] - width, revenue_vals[i] + 2, 
                f'{barrels:.1f}M bbl', ha='center', fontsize=9)
    
    # Configure the plot
    ax3.set_ylabel('Value (Million $)')
    ax3.set_title('Economic Scenarios', fontweight='bold', fontsize=14)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([m['name'] for m in metrics])
    ax3.legend(loc='upper left')
    
    # Add some annotations explaining the calculations
    ax3.text(0.98, 0.02, 
             "Economic Model:\n"
             f"Recovery Factor = 0.2 + 0.1 × Porosity\n"
             f"Barrel Conversion = 6.29 bbl/m³\n"
             f"Revenue = Recoverable Oil × ${oil_price}/bbl\n"
             f"Profit = Revenue - ${(drilling_cost + completion_cost)/1e6:.1f}M",
             transform=ax3.transAxes, fontsize=10, verticalalignment='bottom', 
             horizontalalignment='right', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.8))
    
    # Add overall title
    fig.suptitle('Field Economic Model', fontsize=18, fontweight='bold', y=0.98)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if not show_plot:
        plt.close(fig)
    
    return fig

def visualize_exploration_progress(grid_tensor, x1_grid, x2_grid, basin_gp,
                                  property_idx=0, well_num=None, basin_size=(20, 20),
                                  figsize=(18, 10), cmap='plasma',
                                  save_path=None, show_plot=True):
    """
    Visualize the exploration progress with model predictions and uncertainty.

    Args:
        grid_tensor: Grid of points [n_points, 2] or can be None if x1_grid and x2_grid are provided
        x1_grid, x2_grid: Coordinate meshgrids
        basin_gp: BasinExplorationGP model
        property_idx: Index of the property to visualize
        well_num: Current well number
        basin_size: Size of the basin (width, height)
        figsize: Figure size
        cmap: Colormap to use
        save_path: Path to save the figure
        show_plot: Whether to display the plot

    Returns:
        Figure object
    """
    # If grid_tensor is not provided or we need to recreate it, do so from x1_grid and x2_grid
    if grid_tensor is None:
        grid_tensor = torch.tensor(np.column_stack([x1_grid.flatten(), x2_grid.flatten()]), dtype=torch.float32)
    set_plotting_style()
    resolution = x1_grid.shape[0]
    
    # Get property name
    property_names = basin_gp.properties
    if property_idx < len(property_names):
        property_name = property_names[property_idx].capitalize()
    else:
        property_name = f"Property {property_idx+1}"
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1],
                         hspace=0.3, wspace=0.3)
    
    # Get predictions
    mean, std = basin_gp.predict(grid_tensor)
    
    # Get property predictions
    mean_property = mean[:, property_idx].reshape(resolution, resolution).numpy()
    std_property = std[:, property_idx].reshape(resolution, resolution).numpy()
    
    # 1. Mean prediction
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(x1_grid, x2_grid, mean_property, levels=20, cmap=cmap, alpha=0.9)
    
    # Add colorbar
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    
    # Configure plot
    ax1.set_xlabel('X Distance (km)')
    ax1.set_ylabel('Y Distance (km)')
    ax1.set_title(f'Mean {property_name} Prediction', fontweight='bold')
    
    # 2. Uncertainty (standard deviation)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(x1_grid, x2_grid, std_property, levels=20, cmap='magma', alpha=0.9)
    
    # Add colorbar
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    
    # Configure plot
    ax2.set_xlabel('X Distance (km)')
    ax2.set_ylabel('Y Distance (km)')
    ax2.set_title(f'Uncertainty (Std Dev)', fontweight='bold')
    
    # 3. Value of Information
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate VOI (simple approximation - higher uncertainty means higher VOI)
    # In a real model, this would be calculated more precisely
    voi = std_property * mean_property  # Simple approximation of VOI
    
    im3 = ax3.contourf(x1_grid, x2_grid, voi, levels=20, cmap='viridis', alpha=0.9)
    
    # Add colorbar
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(im3, cax=cax3)
    
    # Configure plot
    ax3.set_xlabel('X Distance (km)')
    ax3.set_ylabel('Y Distance (km)')
    ax3.set_title('Value of Information (Approximation)', fontweight='bold')
    
    # 4. Confidence visualization
    ax4 = fig.add_subplot(gs[1, 1])
    
    # If model has profitability confidence, show a gauge visualization
    if hasattr(basin_gp, 'profitability_confidence'):
        confidence = basin_gp.profitability_confidence
        # Create a gauge-like visualization using a pie chart
        wedgeprops = {'width': 0.3, 'edgecolor': 'w', 'linewidth': 3}
        
        # Background (gray)
        ax4.pie([1], radius=1.0, colors=['lightgray'], startangle=90, counterclock=False,
              wedgeprops=wedgeprops)
        
        # Confidence level (primary color)
        ax4.pie([confidence, 1-confidence], radius=1.0, colors=[COLORS['highlight'], 'white'], 
              startangle=90, counterclock=False, wedgeprops=wedgeprops)
        
        # Target level (red line)
        target = getattr(basin_gp, 'target_confidence', 0.9)
        theta = (1-target) * 180  # Convert to angle
        ax4.plot([0, np.cos(np.radians(theta+90))], [0, np.sin(np.radians(theta+90))], 
                'r-', linewidth=3, zorder=10)
        
        # Add text in the center
        ax4.text(0, 0, f"{confidence*100:.1f}%", ha='center', va='center', 
                fontsize=24, fontweight='bold')
        
        # Add label below
        ax4.text(0, -1.5, f"Profitability Confidence\nTarget: {target*100:.0f}%", 
                ha='center', va='center', fontsize=12)
        
        ax4.set_aspect('equal')
        ax4.axis('off')
    else:
        # If no confidence data, show a placeholder
        ax4.text(0.5, 0.5, "Profitability Confidence\nNot Available", 
                ha='center', va='center', fontsize=14,
                transform=ax4.transAxes)
        ax4.axis('off')
    
    # Add wells to all plots
    for ax in [ax1, ax2, ax3]:
        # Add initial wells
        initial_wells = [well for well in basin_gp.wells if 'Initial' in well['name']]
        if initial_wells:
            x = [well['location'][0] for well in initial_wells]
            y = [well['location'][1] for well in initial_wells]
            ax.scatter(x, y, color='black', s=80, marker='x', linewidth=2, 
                     label='Initial Wells', zorder=10)
        
        # Add exploration wells
        exploration_wells = [well for well in basin_gp.wells if 'Initial' not in well['name']]
        if exploration_wells:
            for well in exploration_wells:
                ax.scatter(well['location'][0], well['location'][1], color=COLORS['accent'], 
                         s=100, marker='o', edgecolor='black', linewidth=1, alpha=0.8, zorder=10)
                
                # Add well number
                try:
                    well_num = int(well['name'].split('_')[-1])
                    ax.text(well['location'][0], well['location'][1], f"{well_num}", 
                          ha='center', va='center', color='white', 
                          fontweight='bold', zorder=11)
                except (ValueError, IndexError):
                    pass
        
        # Add legend to one of the plots
        if ax == ax1:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right', framealpha=0.7)
    
    # Add title indicating current exploration stage
    title = f"VOI-based Exploration"
    if well_num is not None:
        title += f" - After {well_num} Wells"
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if not show_plot:
        plt.close(fig)
        
    return fig

def plot_confidence_progression(confidence_history, confidence_target=0.9, 
                              figsize=(12, 6), save_path=None, show_plot=True):
    """
    Plot the confidence progression during exploration.
    
    Args:
        confidence_history: List of confidence values
        confidence_target: Target confidence level
        figsize: Figure size
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        
    Returns:
        Figure object
    """
    set_plotting_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confidence curve
    x = np.arange(1, len(confidence_history)+1)
    ax.plot(x, confidence_history, marker='o', markersize=8, linewidth=3,
          color=COLORS['tertiary'], label='Confidence Progression')
    
    # Add target line
    ax.axhline(y=confidence_target, color='r', linestyle='--', linewidth=2,
             label=f'Target ({confidence_target*100:.0f}%)')
    
    # Identify where target is reached
    target_idx = next((i for i, conf in enumerate(confidence_history) 
                      if conf >= confidence_target), None)
    
    if target_idx is not None:
        ax.scatter(target_idx+1, confidence_history[target_idx], 
                 s=150, color=COLORS['highlight'], edgecolor='k', zorder=10,
                 label=f'Target Reached: {target_idx+1} Wells')
        
        # Add annotation
        ax.annotate(f"Target reached\nafter {target_idx+1} wells",
                  xy=(target_idx+1, confidence_history[target_idx]),
                  xytext=(target_idx+1+0.5, confidence_history[target_idx]-0.1),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                  fontsize=12, fontweight='bold')
    
    # Configure plot
    ax.set_xlabel('Number of Exploration Wells')
    ax.set_ylabel('Confidence in Profitability')
    ax.set_title('Exploration Efficiency - Confidence Progression', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0.5, len(confidence_history)+0.5)
    ax.set_ylim(0, 1.05)
    
    # Add percentage labels on y-axis
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0, 1.1, 0.1)])
    
    # Add legend
    ax.legend(loc='lower right')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if not show_plot:
        plt.close(fig)
        
    return fig

def plot_field_summary(results, basin_size, x1_grid, x2_grid, figsize=(16, 12),
                      save_path=None, show_plot=True):
    """
    Create a summary visualization of field exploration results.

    Args:
        results: Results dictionary
        basin_size: Size of the basin
        x1_grid, x2_grid: Coordinate meshgrids
        figsize: Figure size
        save_path: Path to save the figure
        show_plot: Whether to display the plot

    Returns:
        Figure object
    """
    set_plotting_style()

    # Extract model and results
    basin_gp = results['model']
    confidence_history = results['confidence_history']
    wells_to_target = results['wells_to_target']
    final_confidence = results['final_confidence']
    avg_property_values = results.get('avg_property_values', {})
    economic_params = results.get('economic_params', {})

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Final property prediction (porosity)
    ax1 = fig.add_subplot(gs[0, 0])

    # Create grid tensor from x1_grid and x2_grid
    resolution = x1_grid.shape[0]
    grid_tensor = torch.tensor(np.column_stack([x1_grid.flatten(), x2_grid.flatten()]), dtype=torch.float32)

    # Get porosity predictions
    mean, std = basin_gp.predict(grid_tensor)
    porosity = mean[:, 0].reshape(x1_grid.shape).numpy()
    
    im1 = ax1.contourf(x1_grid, x2_grid, porosity, levels=20, cmap='plasma', alpha=0.9)
    
    # Add colorbar
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    
    # Add wells
    for well in basin_gp.wells:
        if 'Initial' in well['name']:
            ax1.scatter(well['location'][0], well['location'][1], color='black', 
                      s=80, marker='x', linewidth=2, zorder=10)
        else:
            ax1.scatter(well['location'][0], well['location'][1], color=COLORS['accent'], 
                      s=100, marker='o', edgecolor='black', linewidth=1, zorder=10)
            # Add well number
            try:
                well_num = int(well['name'].split('_')[-1])
                ax1.text(well['location'][0], well['location'][1], f"{well_num}", 
                       ha='center', va='center', color='white', 
                       fontweight='bold', zorder=11)
            except (ValueError, IndexError):
                pass
    
    # Add labels
    ax1.set_xlabel('X Distance (km)')
    ax1.set_ylabel('Y Distance (km)')
    ax1.set_title('Final Porosity Model', fontweight='bold')
    
    # 2. Exploration efficiency plot
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot confidence curve
    x = np.arange(1, len(confidence_history)+1)
    ax2.plot(x, confidence_history, marker='o', markersize=8, linewidth=3,
           color=COLORS['tertiary'], label='Confidence Progression')
    
    # Add target line
    target_confidence = getattr(basin_gp, 'target_confidence', 0.9)
    ax2.axhline(y=target_confidence, color='r', linestyle='--', linewidth=2,
              label=f'Target ({target_confidence*100:.0f}%)')
    
    # Configure plot
    ax2.set_xlabel('Number of Exploration Wells')
    ax2.set_ylabel('Confidence in Profitability')
    ax2.set_title('Exploration Efficiency', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(0.5, len(confidence_history)+0.5)
    ax2.set_ylim(0, 1.05)
    
    # Add percentage labels
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0, 1.1, 0.1)])
    
    # Add legend
    ax2.legend(loc='lower right')
    
    # 3. Key metrics panel
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    # Extract economic parameters
    oil_price = economic_params.get('oil_price', 80)  # $ per barrel
    profit_threshold = economic_params.get('profit_threshold', 30e6)  # $
    
    # Calculate economic values based on average properties
    porosity_avg = avg_property_values.get('porosity', 0.15)
    permeability_avg = avg_property_values.get('permeability', 100)
    thickness_avg = avg_property_values.get('thickness', 50)
    
    # Create text for metrics
    metrics_text = (
        f"EXPLORATION SUMMARY\n\n"
        f"Wells Required: {wells_to_target} wells\n"
        f"Final Confidence: {final_confidence*100:.1f}%\n"
        f"Target Confidence: {target_confidence*100:.0f}%\n\n"
        f"AVERAGE PROPERTIES\n\n"
        f"Porosity: {porosity_avg:.3f}\n"
        f"Permeability: {permeability_avg:.1f} mD\n"
        f"Thickness: {thickness_avg:.1f} ft\n\n"
        f"ECONOMIC PARAMETERS\n\n"
        f"Oil Price: ${oil_price}/bbl\n"
        f"Profit Threshold: ${profit_threshold/1e6:.1f}M"
    )
    
    # Add metrics text
    ax3.text(0.5, 0.5, metrics_text, transform=ax3.transAxes,
           fontsize=12, verticalalignment='center', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor=COLORS['background'], 
                    edgecolor=COLORS['tertiary'], alpha=0.9, pad=1.0))
    
    # 4. VOI-based well placement strategy diagram
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create a simple diagram explaining VOI-based exploration
    ax4.axis('off')
    
    # Create a box containing the explanation
    explanation = (
        "VOI-BASED EXPLORATION STRATEGY\n\n"
        "1. START with initial wells in diverse locations\n\n"
        "2. BUILD a Gaussian Process model of the field\n\n"
        "3. CALCULATE Value of Information for all potential locations:\n"
        "   VOI = Expected economic gain × Uncertainty reduction\n\n"
        "4. SELECT location with highest VOI for next well\n\n"
        "5. DRILL well, collect data, update the model\n\n"
        "6. REPEAT until confidence in profitability exceeds target\n\n"
        f"RESULT: Efficient exploration with {wells_to_target} wells"
    )
    
    # Add explanation text
    ax4.text(0.5, 0.5, explanation, transform=ax4.transAxes,
           fontsize=12, verticalalignment='center', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor=COLORS['background'], 
                    edgecolor=COLORS['tertiary'], alpha=0.9, pad=1.0))
    
    # Add main title
    fig.suptitle('Field Exploration Summary - VOI Strategy', 
               fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if not show_plot:
        plt.close(fig)
        
    return fig