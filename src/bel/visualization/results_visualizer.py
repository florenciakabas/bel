"""Visualization tools for exploration simulation results."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ResultsVisualizer:
    """
    Creates plots and maps for exploration simulation analysis.
    
    This class provides visualization tools for geological maps,
    uncertainty evolution, economic distributions, and well location maps.
    """
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (12, 10),
        style: str = "whitegrid",
        color_palette: str = "viridis",
        dpi: int = 100
    ):
        """
        Initialize the visualization tools.
        
        Args:
            figure_size: Default figure size.
            style: Seaborn style to use.
            color_palette: Default color palette.
            dpi: Dots per inch for figure resolution.
        """
        self.figure_size = figure_size
        self.style = style
        self.color_palette = color_palette
        self.dpi = dpi
        
        # Set default plotting style
        sns.set_style(style)
        plt.rcParams["figure.figsize"] = figure_size
        plt.rcParams["figure.dpi"] = dpi
    
    def plot_geological_maps(
        self,
        property_maps: Dict[str, np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        well_locations: Optional[np.ndarray] = None,
        well_values: Optional[Dict[str, np.ndarray]] = None,
        property_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        titles: Optional[Dict[str, str]] = None,
        cmaps: Optional[Dict[str, str]] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot geological property maps.
        
        Args:
            property_maps: Dictionary mapping property names to 2D arrays.
            x_grid: 2D array of x coordinates.
            y_grid: 2D array of y coordinates.
            well_locations: Optional array of well locations of shape (n_wells, 2).
            well_values: Optional dictionary mapping property names to well values.
            property_ranges: Optional dictionary mapping property names to value ranges.
            titles: Optional dictionary mapping property names to plot titles.
            cmaps: Optional dictionary mapping property names to colormaps.
            fig_size: Figure size. If None, uses default.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        properties = list(property_maps.keys())
        n_props = len(properties)
        
        # Determine grid layout
        n_cols = min(3, n_props)
        n_rows = int(np.ceil(n_props / n_cols))
        
        # Create figure
        if fig_size is None:
            fig_size = (self.figure_size[0], self.figure_size[1] * n_rows / 2)
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        
        # Handle single axis case
        if n_props == 1:
            axes = np.array([axes])
        
        # Flatten axes for easy iteration
        axes = axes.flatten()
        
        # Plot each property
        for i, prop in enumerate(properties):
            ax = axes[i]
            prop_map = property_maps[prop]
            
            # Get colormap
            cmap = cmaps.get(prop, "viridis") if cmaps else "viridis"
            
            # Get value range
            if property_ranges and prop in property_ranges:
                vmin, vmax = property_ranges[prop]
            else:
                vmin, vmax = np.min(prop_map), np.max(prop_map)
            
            # Plot contour
            cont = ax.contourf(x_grid, y_grid, prop_map, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add wells if provided
            if well_locations is not None:
                if well_values and prop in well_values:
                    # Color wells by property value
                    sc = ax.scatter(
                        well_locations[:, 0],
                        well_locations[:, 1],
                        c=well_values[prop],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        edgecolor='black',
                        s=80,
                        zorder=10
                    )
                else:
                    # Use default markers
                    ax.scatter(
                        well_locations[:, 0],
                        well_locations[:, 1],
                        marker='o',
                        edgecolor='black',
                        facecolor='white',
                        s=80,
                        zorder=10
                    )
            
            # Add colorbar
            plt.colorbar(cont, ax=ax)
            
            # Set title
            title = titles.get(prop, prop) if titles else prop
            ax.set_title(title)
            
            # Set labels
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
        
        # Hide unused subplots
        for i in range(n_props, len(axes)):
            axes[i].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_evolution(
        self,
        uncertainty_history: List[Dict[str, np.ndarray]],
        property_name: str,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        well_history: List[np.ndarray],
        stage_labels: Optional[List[str]] = None,
        cmap: str = "plasma_r",
        fig_size: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the evolution of uncertainty for a property over exploration stages.
        
        Args:
            uncertainty_history: List of dictionaries mapping property names to uncertainty arrays.
            property_name: Name of the property to visualize.
            x_grid: 2D array of x coordinates.
            y_grid: 2D array of y coordinates.
            well_history: List of arrays containing well locations for each stage.
            stage_labels: Optional list of labels for each stage.
            cmap: Colormap to use.
            fig_size: Figure size. If None, uses default.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        n_stages = len(uncertainty_history)
        
        # Determine grid layout
        n_cols = min(3, n_stages)
        n_rows = int(np.ceil(n_stages / n_cols))
        
        # Create figure
        if fig_size is None:
            fig_size = (self.figure_size[0], self.figure_size[1] * n_rows / 2)
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        
        # Handle single axis case
        if n_stages == 1:
            axes = np.array([axes])
        
        # Flatten axes for easy iteration
        axes = axes.flatten()
        
        # Find global min/max for consistent colorbar
        all_uncertainties = [uncertainty_history[i][property_name] for i in range(n_stages)]
        vmin = min(np.min(u) for u in all_uncertainties)
        vmax = max(np.max(u) for u in all_uncertainties)
        
        # Plot each stage
        for i in range(n_stages):
            ax = axes[i]
            uncertainty = uncertainty_history[i][property_name]
            
            # Plot uncertainty contour
            cont = ax.contourf(x_grid, y_grid, uncertainty, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add wells for this stage
            if i == 0:
                wells_so_far = well_history[0]
            else:
                wells_so_far = np.vstack(well_history[:i+1])
            
            # Ensure wells_so_far is 2D
            if wells_so_far.ndim == 1:
                wells_so_far = wells_so_far.reshape(1, -1)
                
            ax.scatter(
                wells_so_far[:, 0],
                wells_so_far[:, 1],
                marker='o',
                edgecolor='black',
                facecolor='white',
                s=80,
                zorder=10
            )
            
            # Highlight the newest well(s)
            if i > 0:
                newest_wells = well_history[i]
                ax.scatter(
                    newest_wells[:, 0],
                    newest_wells[:, 1],
                    marker='*',
                    edgecolor='black',
                    facecolor='yellow',
                    s=200,
                    zorder=11
                )
            
            # Set title
            title = f"Stage {i+1}" if stage_labels is None else stage_labels[i]
            ax.set_title(f"{title}\n{len(wells_so_far)} Wells")
            
            # Set labels
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
        
        # Add colorbar to the right of the subplots
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(cont, cax=cbar_ax)
        cbar.set_label(f"{property_name} Uncertainty")
        
        # Hide unused subplots
        for i in range(n_stages, len(axes)):
            axes[i].axis('off')
        
        # Add main title
        fig.suptitle(f"Evolution of {property_name} Uncertainty During Exploration",
                   fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_economic_distributions(
        self,
        economic_results: List[Dict[str, Any]],
        stage_labels: Optional[List[str]] = None,
        target_profit: Optional[float] = None,
        target_confidence: Optional[float] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the evolution of economic distributions over exploration stages.
        
        Args:
            economic_results: List of economic assessment result dictionaries.
            stage_labels: Optional list of labels for each stage.
            target_profit: Optional target profit threshold.
            target_confidence: Optional target confidence level.
            fig_size: Figure size. If None, uses default.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        n_stages = len(economic_results)
        
        # Create figure
        if fig_size is None:
            fig_size = (self.figure_size[0], self.figure_size[1] * 1.2)
            
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        
        # Histogram plot of NPV distributions
        ax_hist = fig.add_subplot(gs[0, :])
        
        # Find global min/max for consistent binning
        all_npv_values = [economic_results[i]['npv_values'] for i in range(n_stages)]
        npv_min = min(np.min(npv) for npv in all_npv_values)
        npv_max = max(np.max(npv) for npv in all_npv_values)
        
        # Create histogram bins
        bins = np.linspace(npv_min, npv_max, 30)
        
        # Plot distributions with different colors and transparency
        colors = plt.cm.viridis(np.linspace(0, 1, n_stages))
        
        for i in range(n_stages):
            npv_values = economic_results[i]['npv_values']
            stage = f"Stage {i+1}" if stage_labels is None else stage_labels[i]
            
            # Plot histogram
            ax_hist.hist(
                npv_values, 
                bins=bins, 
                alpha=0.7 if i == n_stages-1 else 0.3, 
                color=colors[i],
                label=f"{stage}: Mean=${economic_results[i]['npv_mean']:.1f}M"
            )
        
        # Add target profit line if provided
        if target_profit is not None:
            ax_hist.axvline(
                target_profit, 
                color='red', 
                linestyle='--', 
                linewidth=2,
                label=f"Target: ${target_profit}M"
            )
        
        # Add labels and legend
        ax_hist.set_xlabel("Net Present Value ($M)")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title("Evolution of NPV Distribution During Exploration")
        ax_hist.legend(loc='best')
        ax_hist.grid(True, linestyle='--', alpha=0.7)
        
        # Probability of meeting target plot
        ax_prob = fig.add_subplot(gs[1, 0])
        
        # Extract probabilities of meeting target
        stages = range(1, n_stages + 1)
        probs = [result['prob_target'] for result in economic_results]
        
        # Plot probability evolution
        ax_prob.plot(stages, probs, 'o-', linewidth=2, markersize=8)
        
        # Add target confidence line if provided
        if target_confidence is not None:
            ax_prob.axhline(
                target_confidence, 
                color='red', 
                linestyle='--', 
                linewidth=2,
                label=f"Target: {target_confidence:.0%}"
            )
        
        # Add labels
        ax_prob.set_xlabel("Exploration Stage")
        ax_prob.set_ylabel("Probability of Meeting Target")
        ax_prob.set_title("Confidence Evolution")
        ax_prob.set_xticks(stages)
        ax_prob.set_xticklabels(stage_labels if stage_labels else [f"Stage {i}" for i in stages])
        ax_prob.set_ylim(0, 1)
        ax_prob.grid(True, linestyle='--', alpha=0.7)
        
        # NPV mean and std plot
        ax_mean = fig.add_subplot(gs[1, 1])
        
        # Extract mean and std values
        means = [result['npv_mean'] for result in economic_results]
        stds = [result['npv_std'] for result in economic_results]
        
        # Calculate upper and lower bounds (P10, P90)
        lower_bounds = [result['npv_p10'] for result in economic_results]
        upper_bounds = [result['npv_p90'] for result in economic_results]
        
        # Plot mean and confidence interval
        ax_mean.plot(stages, means, 'o-', linewidth=2, markersize=8, label="Mean NPV")
        ax_mean.fill_between(
            stages, 
            lower_bounds, 
            upper_bounds, 
            alpha=0.3, 
            label="P10-P90 Range"
        )
        
        # Add target profit line if provided
        if target_profit is not None:
            ax_mean.axhline(
                target_profit, 
                color='red', 
                linestyle='--', 
                linewidth=2,
                label=f"Target: ${target_profit}M"
            )
        
        # Add labels and legend
        ax_mean.set_xlabel("Exploration Stage")
        ax_mean.set_ylabel("NPV ($M)")
        ax_mean.set_title("Mean NPV and Uncertainty")
        ax_mean.set_xticks(stages)
        ax_mean.set_xticklabels(stage_labels if stage_labels else [f"Stage {i}" for i in stages])
        ax_mean.grid(True, linestyle='--', alpha=0.7)
        ax_mean.legend(loc='best')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_well_location_maps(
        self,
        property_map: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        well_locations: np.ndarray,
        voi_surface: Optional[np.ndarray] = None,
        uncertainty_map: Optional[np.ndarray] = None,
        property_name: str = "Property",
        property_range: Optional[Tuple[float, float]] = None,
        fig_size: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Create well location maps with underlying property and optional VOI/uncertainty.
        
        Args:
            property_map: 2D array of property values.
            x_grid: 2D array of x coordinates.
            y_grid: 2D array of y coordinates.
            well_locations: Array of well locations of shape (n_wells, 2).
            voi_surface: Optional 2D array of VOI values.
            uncertainty_map: Optional 2D array of uncertainty values.
            property_name: Name of the property.
            property_range: Optional tuple of (min, max) property values.
            fig_size: Figure size. If None, uses default.
            save_path: Optional path to save the figure.
            interactive: Whether to create an interactive plotly figure.
            
        Returns:
            Matplotlib or Plotly figure object.
        """
        if interactive:
            return self._create_interactive_well_map(
                property_map, x_grid, y_grid, well_locations, voi_surface,
                uncertainty_map, property_name, property_range, save_path
            )
        else:
            return self._create_static_well_map(
                property_map, x_grid, y_grid, well_locations, voi_surface,
                uncertainty_map, property_name, property_range, fig_size, save_path
            )
    
    def _create_static_well_map(
        self,
        property_map: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        well_locations: np.ndarray,
        voi_surface: Optional[np.ndarray],
        uncertainty_map: Optional[np.ndarray],
        property_name: str,
        property_range: Optional[Tuple[float, float]],
        fig_size: Optional[Tuple[int, int]],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create a static matplotlib well location map."""
        # Determine number of subplots needed
        n_plots = 1 + (voi_surface is not None) + (uncertainty_map is not None)
        
        # Create figure
        if fig_size is None:
            fig_size = (self.figure_size[0], self.figure_size[1] * n_plots / 2)
            
        fig, axes = plt.subplots(1, n_plots, figsize=fig_size)
        
        # Handle single plot case
        if n_plots == 1:
            axes = [axes]
        
        # Plot property map
        ax = axes[0]
        
        # Get value range
        if property_range:
            vmin, vmax = property_range
        else:
            vmin, vmax = np.min(property_map), np.max(property_map)
        
        # Plot contour
        cont = ax.contourf(x_grid, y_grid, property_map, cmap="viridis", vmin=vmin, vmax=vmax)
        
        # Add wells
        ax.scatter(
            well_locations[:, 0],
            well_locations[:, 1],
            marker='o',
            edgecolor='black',
            facecolor='white',
            s=80,
            zorder=10
        )
        
        # Add colorbar
        plt.colorbar(cont, ax=ax, label=property_name)
        
        # Set title and labels
        ax.set_title(f"{property_name} Map with Well Locations")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Plot VOI surface if provided
        if voi_surface is not None:
            ax = axes[1] if n_plots > 1 else axes[0]
            
            # Plot VOI contour
            cont = ax.contourf(x_grid, y_grid, voi_surface, cmap="plasma")
            
            # Add wells
            ax.scatter(
                well_locations[:, 0],
                well_locations[:, 1],
                marker='o',
                edgecolor='black',
                facecolor='white',
                s=80,
                zorder=10
            )
            
            # Add colorbar
            plt.colorbar(cont, ax=ax, label="Value of Information ($M)")
            
            # Set title and labels
            ax.set_title("Value of Information Surface with Well Locations")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
        
        # Plot uncertainty map if provided
        if uncertainty_map is not None:
            ax = axes[-1] if n_plots > 1 else axes[0]
            
            # Plot uncertainty contour
            cont = ax.contourf(x_grid, y_grid, uncertainty_map, cmap="plasma_r")
            
            # Add wells
            ax.scatter(
                well_locations[:, 0],
                well_locations[:, 1],
                marker='o',
                edgecolor='black',
                facecolor='white',
                s=80,
                zorder=10
            )
            
            # Add colorbar
            plt.colorbar(cont, ax=ax, label=f"{property_name} Uncertainty")
            
            # Set title and labels
            ax.set_title(f"{property_name} Uncertainty with Well Locations")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _create_interactive_well_map(
        self,
        property_map: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        well_locations: np.ndarray,
        voi_surface: Optional[np.ndarray],
        uncertainty_map: Optional[np.ndarray],
        property_name: str,
        property_range: Optional[Tuple[float, float]],
        save_path: Optional[str]
    ) -> go.Figure:
        """Create an interactive plotly well location map."""
        # Determine number of subplots needed
        n_plots = 1 + (voi_surface is not None) + (uncertainty_map is not None)
        
        # Create subplot titles
        titles = [f"{property_name} Map"]
        if voi_surface is not None:
            titles.append("Value of Information Surface")
        if uncertainty_map is not None:
            titles.append(f"{property_name} Uncertainty")
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, 
            cols=n_plots, 
            subplot_titles=titles,
            specs=[[{"type": "contour"} for _ in range(n_plots)]]
        )
        
        # Flatten x and y grids for plotly
        x_flat = x_grid[0, :]
        y_flat = y_grid[:, 0]
        
        # Get value range for property
        if property_range:
            z_min, z_max = property_range
        else:
            z_min, z_max = np.min(property_map), np.max(property_map)
        
        # Add property contour
        fig.add_trace(
            go.Contour(
                z=property_map,
                x=x_flat,
                y=y_flat,
                colorscale="Viridis",
                zmin=z_min,
                zmax=z_max,
                colorbar=dict(
                    title=property_name,
                    x=0.25,
                    len=0.8
                )
            ),
            row=1, col=1
        )
        
        # Add wells to property plot
        fig.add_trace(
            go.Scatter(
                x=well_locations[:, 0],
                y=well_locations[:, 1],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=10,
                    color="white",
                    line=dict(color="black", width=1)
                ),
                name="Wells"
            ),
            row=1, col=1
        )
        
        # Add VOI contour if provided
        if voi_surface is not None:
            fig.add_trace(
                go.Contour(
                    z=voi_surface,
                    x=x_flat,
                    y=y_flat,
                    colorscale="Plasma",
                    colorbar=dict(
                        title="VOI ($M)",
                        x=0.55 if n_plots == 2 else 0.45,
                        len=0.8
                    )
                ),
                row=1, col=2
            )
            
            # Add wells to VOI plot
            fig.add_trace(
                go.Scatter(
                    x=well_locations[:, 0],
                    y=well_locations[:, 1],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=10,
                        color="white",
                        line=dict(color="black", width=1)
                    ),
                    name="Wells",
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add uncertainty contour if provided
        if uncertainty_map is not None:
            fig.add_trace(
                go.Contour(
                    z=uncertainty_map,
                    x=x_flat,
                    y=y_flat,
                    colorscale="Plasma_r",
                    colorbar=dict(
                        title=f"{property_name} Uncertainty",
                        x=0.85,
                        len=0.8
                    )
                ),
                row=1, col=n_plots
            )
            
            # Add wells to uncertainty plot
            fig.add_trace(
                go.Scatter(
                    x=well_locations[:, 0],
                    y=well_locations[:, 1],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=10,
                        color="white",
                        line=dict(color="black", width=1)
                    ),
                    name="Wells",
                    showlegend=False
                ),
                row=1, col=n_plots
            )
        
        # Update layout
        fig.update_layout(
            title=f"Basin Exploration: {property_name} and Well Locations",
            height=600,
            width=300 * n_plots + 100,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="X Coordinate")
        fig.update_yaxes(title_text="Y Coordinate")
        
        # Save figure if path provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_exploration_summary_dashboard(
        self,
        exploration_results: Dict[str, Any],
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Union[plt.Figure, Dict[str, go.Figure]]:
        """
        Create a comprehensive exploration summary dashboard.
        
        Args:
            exploration_results: Dictionary containing all exploration results.
            save_path: Optional path to save the figure.
            interactive: Whether to create interactive plotly figures.
            
        Returns:
            Matplotlib figure or dictionary of Plotly figures.
        """
        if interactive:
            return self._create_interactive_summary(exploration_results, save_path)
        else:
            return self._create_static_summary(exploration_results, save_path)
    
    def _create_static_summary(
        self,
        exploration_results: Dict[str, Any],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create a static matplotlib summary dashboard."""
        # Extract data from results
        stages = exploration_results.get("stages", [])
        n_stages = len(stages)
        
        # Create figure with complex layout
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 1])
        
        # 1. Title and key metrics
        ax_title = fig.add_subplot(gs[0, :2])
        ax_title.axis('off')
        
        # Add title and key metrics text
        ax_title.text(0.5, 0.8, "Basin Exploration Summary", 
                    fontsize=24, ha='center', va='center')
        
        # Add key metrics if available
        final_stage = stages[-1] if stages else {}
        economic_results = final_stage.get("economic_results", {})
        
        metrics_text = (
            f"Total Wells: {n_stages}\n"
            f"Final NPV (P50): ${economic_results.get('npv_p50', 0):.1f}M\n"
            f"Confidence Level: {economic_results.get('prob_target', 0):.1%}\n"
            f"Target Profit: ${economic_results.get('target_profit', 0):.1f}M"
        )
        
        ax_title.text(0.5, 0.4, metrics_text, 
                    fontsize=14, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
        
        # 2. Final property maps
        final_property_maps = final_stage.get("property_maps", {})
        property_names = list(final_property_maps.keys())
        
        if property_names:
            ax_map = fig.add_subplot(gs[0, 2:])
            
            # Select first property for the map
            prop_name = property_names[0]
            prop_map = final_property_maps[prop_name]
            
            # Get grid coordinates
            x_grid = exploration_results.get("x_grid")
            y_grid = exploration_results.get("y_grid")
            
            # Get well locations
            well_locations = np.array([stage["well_location"] for stage in stages])
            
            # Plot contour
            cont = ax_map.contourf(x_grid, y_grid, prop_map, cmap="viridis")
            
            # Add wells
            ax_map.scatter(
                well_locations[:, 0],
                well_locations[:, 1],
                marker='o',
                edgecolor='black',
                facecolor='white',
                s=80,
                zorder=10
            )
            
            # Add colorbar
            plt.colorbar(cont, ax=ax_map, label=prop_name)
            
            # Set title and labels
            ax_map.set_title(f"Final {prop_name} Map with All Wells")
            ax_map.set_xlabel("X Coordinate")
            ax_map.set_ylabel("Y Coordinate")
        
        # 3. NPV distribution evolution
        ax_npv = fig.add_subplot(gs[1, :2])
        
        # Extract NPV distributions
        npv_distributions = []
        stage_labels = []
        
        for i, stage in enumerate(stages):
            econ_results = stage.get("economic_results", {})
            if "npv_values" in econ_results:
                npv_distributions.append(econ_results["npv_values"])
                stage_labels.append(f"Well {i+1}")
        
        if npv_distributions:
            # Find global min/max for consistent binning
            npv_min = min(np.min(npv) for npv in npv_distributions)
            npv_max = max(np.max(npv) for npv in npv_distributions)
            
            # Create histogram bins
            bins = np.linspace(npv_min, npv_max, 30)
            
            # Plot distributions with different colors and transparency
            colors = plt.cm.viridis(np.linspace(0, 1, len(npv_distributions)))
            
            for i, npv_values in enumerate(npv_distributions):
                # Plot histogram
                alpha = 0.2 if i < len(npv_distributions) - 1 else 0.7
                ax_npv.hist(
                    npv_values, 
                    bins=bins, 
                    alpha=alpha, 
                    color=colors[i],
                    label=f"{stage_labels[i]}"
                )
            
            # Add target profit line if available
            target_profit = economic_results.get("target_profit")
            if target_profit is not None:
                ax_npv.axvline(
                    target_profit, 
                    color='red', 
                    linestyle='--', 
                    linewidth=2,
                    label=f"Target: ${target_profit}M"
                )
            
            # Add labels and legend
            ax_npv.set_xlabel("Net Present Value ($M)")
            ax_npv.set_ylabel("Frequency")
            ax_npv.set_title("Evolution of NPV Distribution")
            ax_npv.legend(loc='best')
            ax_npv.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Confidence evolution
        ax_conf = fig.add_subplot(gs[1, 2:])
        
        # Extract confidence levels and NPV statistics
        stage_nums = list(range(1, n_stages + 1))
        conf_levels = []
        npv_means = []
        npv_p10 = []
        npv_p90 = []
        
        for stage in stages:
            econ_results = stage.get("economic_results", {})
            conf_levels.append(econ_results.get("prob_target", 0))
            npv_means.append(econ_results.get("npv_mean", 0))
            npv_p10.append(econ_results.get("npv_p10", 0))
            npv_p90.append(econ_results.get("npv_p90", 0))
        
        # Create twin axes for confidence and NPV
        ax_npv_evol = ax_conf.twinx()
        
        # Plot confidence evolution
        conf_line = ax_conf.plot(
            stage_nums, conf_levels, 'ro-', 
            linewidth=2, markersize=8, 
            label="Confidence Level"
        )
        
        # Add target confidence line if available
        target_confidence = economic_results.get("target_confidence")
        if target_confidence is not None:
            ax_conf.axhline(
                target_confidence, 
                color='red', 
                linestyle='--', 
                linewidth=2,
                label=f"Target: {target_confidence:.0%}"
            )
        
        # Plot NPV evolution with uncertainty
        npv_line = ax_npv_evol.plot(
            stage_nums, npv_means, 'bo-', 
            linewidth=2, markersize=8, 
            label="Mean NPV"
        )
        
        ax_npv_evol.fill_between(
            stage_nums, 
            npv_p10, 
            npv_p90, 
            color='blue', 
            alpha=0.2, 
            label="P10-P90 Range"
        )
        
        # Add target profit line if available
        if target_profit is not None:
            ax_npv_evol.axhline(
                target_profit, 
                color='blue', 
                linestyle='--', 
                linewidth=2,
                label=f"Target: ${target_profit}M"
            )
        
        # Add labels
        ax_conf.set_xlabel("Exploration Well")
        ax_conf.set_ylabel("Confidence Level")
        ax_npv_evol.set_ylabel("NPV ($M)")
        ax_conf.set_title("Confidence and NPV Evolution")
        ax_conf.set_ylim(0, 1)
        ax_conf.set_xticks(stage_nums)
        ax_conf.grid(True, linestyle='--', alpha=0.7)
        
        # Combine legends
        lines1, labels1 = ax_conf.get_legend_handles_labels()
        lines2, labels2 = ax_npv_evol.get_legend_handles_labels()
        ax_conf.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # 5. Uncertainty reduction
        ax_uncert = fig.add_subplot(gs[2, :2])
        
        # Extract uncertainty values
        uncertainty_reductions = []
        
        for i, stage in enumerate(stages):
            if i > 0:  # Skip first stage (no reduction yet)
                uncertainty_reductions.append(stage.get("uncertainty_reduction", 0))
        
        if uncertainty_reductions:
            # Plot uncertainty reduction
            ax_uncert.bar(
                range(2, n_stages + 1), 
                uncertainty_reductions, 
                color='purple', 
                alpha=0.7
            )
            
            # Add labels
            ax_uncert.set_xlabel("Exploration Well")
            ax_uncert.set_ylabel("Uncertainty Reduction")
            ax_uncert.set_title("Uncertainty Reduction per Well")
            ax_uncert.set_xticks(range(2, n_stages + 1))
            ax_uncert.set_ylim(0, max(uncertainty_reductions) * 1.2)
            ax_uncert.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 6. VOI analysis
        ax_voi = fig.add_subplot(gs[2, 2:])
        
        # Extract VOI values
        voi_values = []
        
        for stage in stages:
            voi_values.append(stage.get("voi_value", 0))
        
        if voi_values:
            # Plot VOI values
            ax_voi.bar(
                range(1, n_stages + 1), 
                voi_values, 
                color='green', 
                alpha=0.7
            )
            
            # Add horizontal line at zero
            ax_voi.axhline(0, color='black', linestyle='-', linewidth=1)
            
            # Add labels
            ax_voi.set_xlabel("Exploration Well")
            ax_voi.set_ylabel("Value of Information ($M)")
            ax_voi.set_title("Value of Information per Well")
            ax_voi.set_xticks(range(1, n_stages + 1))
            ax_voi.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _create_interactive_summary(
        self,
        exploration_results: Dict[str, Any],
        save_path: Optional[str]
    ) -> Dict[str, go.Figure]:
        """Create interactive plotly summary dashboard as multiple figures."""
        # Extract data from results
        stages = exploration_results.get("stages", [])
        n_stages = len(stages)
        
        # Dictionary to store figures
        figures = {}
        
        # 1. Final property maps
        final_stage = stages[-1] if stages else {}
        final_property_maps = final_stage.get("property_maps", {})
        property_names = list(final_property_maps.keys())
        
        if property_names:
            # Get grid coordinates
            x_grid = exploration_results.get("x_grid")
            y_grid = exploration_results.get("y_grid")
            x_flat = x_grid[0, :] if x_grid is not None else None
            y_flat = y_grid[:, 0] if y_grid is not None else None
            
            # Get well locations
            well_locations = np.array([stage["well_location"] for stage in stages])
            
            # Create property map figure
            for prop_name in property_names:
                prop_map = final_property_maps[prop_name]
                
                fig = go.Figure()
                
                # Add contour
                fig.add_trace(
                    go.Contour(
                        z=prop_map,
                        x=x_flat,
                        y=y_flat,
                        colorscale="Viridis",
                        colorbar=dict(title=prop_name)
                    )
                )
                
                # Add wells
                fig.add_trace(
                    go.Scatter(
                        x=well_locations[:, 0],
                        y=well_locations[:, 1],
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=10,
                            color="white",
                            line=dict(color="black", width=1)
                        ),
                        name="Wells"
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Final {prop_name} Map with All Wells",
                    xaxis_title="X Coordinate",
                    yaxis_title="Y Coordinate",
                    height=600,
                    width=800
                )
                
                figures[f"property_map_{prop_name}"] = fig
        
        # 2. NPV distribution evolution
        npv_distributions = []
        stage_labels = []
        
        for i, stage in enumerate(stages):
            econ_results = stage.get("economic_results", {})
            if "npv_values" in econ_results:
                npv_distributions.append(econ_results["npv_values"])
                stage_labels.append(f"Well {i+1}")
        
        if npv_distributions:
            fig = go.Figure()
            
            # Find global min/max for consistent binning
            npv_min = min(np.min(npv) for npv in npv_distributions)
            npv_max = max(np.max(npv) for npv in npv_distributions)
            
            # Create histogram bins
            bins = np.linspace(npv_min, npv_max, 30)
            
            # Add histograms for each stage
            for i, npv_values in enumerate(npv_distributions):
                opacity = 0.3 if i < len(npv_distributions) - 1 else 0.7
                
                fig.add_trace(
                    go.Histogram(
                        x=npv_values,
                        name=stage_labels[i],
                        opacity=opacity,
                        xbins=dict(
                            start=npv_min,
                            end=npv_max,
                            size=(npv_max - npv_min) / 30
                        )
                    )
                )
            
            # Add target profit line if available
            target_profit = final_stage.get("economic_results", {}).get("target_profit")
            if target_profit is not None:
                fig.add_vline(
                    x=target_profit,
                    line=dict(color="red", width=2, dash="dash"),
                    annotation_text=f"Target: ${target_profit}M",
                    annotation_position="top right"
                )
            
            # Update layout
            fig.update_layout(
                title="Evolution of NPV Distribution",
                xaxis_title="Net Present Value ($M)",
                yaxis_title="Frequency",
                barmode="overlay",
                height=500,
                width=800
            )
            
            figures["npv_distribution"] = fig
        
        # 3. Confidence and NPV evolution
        stage_nums = list(range(1, n_stages + 1))
        conf_levels = []
        npv_means = []
        npv_p10 = []
        npv_p90 = []
        
        for stage in stages:
            econ_results = stage.get("economic_results", {})
            conf_levels.append(econ_results.get("prob_target", 0))
            npv_means.append(econ_results.get("npv_mean", 0))
            npv_p10.append(econ_results.get("npv_p10", 0))
            npv_p90.append(econ_results.get("npv_p90", 0))
        
        if conf_levels and npv_means:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add confidence evolution
            fig.add_trace(
                go.Scatter(
                    x=stage_nums,
                    y=conf_levels,
                    mode="lines+markers",
                    name="Confidence Level",
                    line=dict(color="red", width=2),
                    marker=dict(size=8, color="red")
                ),
                secondary_y=False
            )
            
            # Add target confidence line if available
            target_confidence = final_stage.get("economic_results", {}).get("target_confidence")
            if target_confidence is not None:
                fig.add_hline(
                    y=target_confidence,
                    line=dict(color="red", width=2, dash="dash"),
                    annotation_text=f"Target: {target_confidence:.0%}",
                    annotation_position="left",
                    secondary_y=False
                )
            
            # Add NPV evolution
            fig.add_trace(
                go.Scatter(
                    x=stage_nums,
                    y=npv_means,
                    mode="lines+markers",
                    name="Mean NPV",
                    line=dict(color="blue", width=2),
                    marker=dict(size=8, color="blue")
                ),
                secondary_y=True
            )
            
            # Add NPV confidence interval
            fig.add_trace(
                go.Scatter(
                    x=stage_nums + stage_nums[::-1],
                    y=npv_p90 + npv_p10[::-1],
                    fill="toself",
                    fillcolor="rgba(0, 0, 255, 0.2)",
                    line=dict(color="rgba(0, 0, 255, 0)"),
                    hoverinfo="skip",
                    showlegend=False
                ),
                secondary_y=True
            )
            
            # Add target profit line if available
            if target_profit is not None:
                fig.add_hline(
                    y=target_profit,
                    line=dict(color="blue", width=2, dash="dash"),
                    annotation_text=f"Target: ${target_profit}M",
                    annotation_position="right",
                    secondary_y=True
                )
            
            # Update layout
            fig.update_layout(
                title="Confidence and NPV Evolution",
                xaxis_title="Exploration Well",
                height=500,
                width=800
            )
            
            fig.update_yaxes(title_text="Confidence Level", range=[0, 1], secondary_y=False)
            fig.update_yaxes(title_text="NPV ($M)", secondary_y=True)
            
            figures["confidence_npv_evolution"] = fig
        
        # 4. Uncertainty reduction and VOI
        uncertainty_reductions = []
        voi_values = []
        
        for i, stage in enumerate(stages):
            if i > 0:  # Skip first stage (no reduction yet)
                uncertainty_reductions.append(stage.get("uncertainty_reduction", 0))
            voi_values.append(stage.get("voi_value", 0))
        
        if uncertainty_reductions or voi_values:
            fig = make_subplots(rows=2, cols=1)
            
            # Add uncertainty reduction
            if uncertainty_reductions:
                fig.add_trace(
                    go.Bar(
                        x=list(range(2, n_stages + 1)),
                        y=uncertainty_reductions,
                        marker_color="purple",
                        opacity=0.7,
                        name="Uncertainty Reduction"
                    ),
                    row=1, col=1
                )
                
                fig.update_yaxes(title_text="Uncertainty Reduction", row=1, col=1)
            
            # Add VOI values
            if voi_values:
                fig.add_trace(
                    go.Bar(
                        x=list(range(1, n_stages + 1)),
                        y=voi_values,
                        marker_color="green",
                        opacity=0.7,
                        name="Value of Information"
                    ),
                    row=2, col=1
                )
                
                # Add horizontal line at zero
                fig.add_hline(
                    y=0,
                    line=dict(color="black", width=1),
                    row=2, col=1
                )
                
                fig.update_yaxes(title_text="Value of Information ($M)", row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title="Uncertainty Reduction and Value of Information",
                xaxis_title="Exploration Well",
                height=700,
                width=800
            )
            
            figures["uncertainty_voi"] = fig
        
        # Save figures if path provided
        if save_path:
            for name, fig in figures.items():
                fig.write_html(f"{save_path}_{name}.html")
        
        return figures