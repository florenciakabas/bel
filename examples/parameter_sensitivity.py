#!/usr/bin/env python
"""
Parameter sensitivity analysis for exploration optimization.

This example demonstrates how to perform sensitivity analysis
on key parameters affecting exploration outcomes.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
from typing import Dict, List, Tuple, Any
import time
import json

from bel.geological_model import GaussianProcessGeology
from bel.production_model import ProductionPredictor
from bel.economic_model import EconomicAssessment
from bel.data_manager import DataManager
from bel.visualization import ResultsVisualizer
from bel.simulation_controller import ExplorationSimulation
from bel.utils.math_utils import MathUtils


def run_sensitivity_analysis(
    output_dir='sensitivity_results',
    random_seed=42,
    max_wells=8,
    n_repeats=3,
    show_plots=False
):
    """
    Run sensitivity analysis on key parameters.
    
    Args:
        output_dir: Directory to save results.
        random_seed: Base random seed for reproducibility.
        max_wells: Maximum number of exploration wells to drill.
        n_repeats: Number of times to repeat each parameter combination.
        show_plots: Whether to display plots interactively.
    
    Returns:
        DataFrame with sensitivity analysis results.
    """
    print("Setting up sensitivity analysis...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Base parameters
    grid_size = (50, 50)
    x_range = (-94.6, -92.9)
    y_range = (31.3, 33.0)
    
    # Geological properties
    properties = [
        "Thickness", "Porosity", "Permeability", 
        "TOC", "SW", "Depth", "Vclay"
    ]
    
    # Property ranges
    property_ranges = {
        "Thickness": (50, 200),
        "Porosity": (0.02, 0.08),
        "Permeability": (0.1, 0.5),
        "TOC": (2, 8),
        "SW": (0.3, 0.7),
        "Depth": (-11000, -8000),
        "Vclay": (0.3, 0.6)
    }
    
    # Base correlation lengths
    base_length_scales = {
        "Thickness": 0.5,
        "Porosity": 0.4,
        "Permeability": 0.4,
        "TOC": 0.3,
        "SW": 0.3,
        "Depth": 0.6,
        "Vclay": 0.3
    }
    
    # Parameters to vary
    parameter_variations = {
        "correlation_length": [0.2, 0.5, 0.8],  # Spatial correlation multiplier
        "gas_price": [2.5, 4.0, 5.5],  # $/mcf
        "drilling_cost": [8.0, 10.0, 12.0],  # $M per well
        "target_confidence": [0.7, 0.8, 0.9]  # Target confidence level
    }
    
    # Initialize data manager
    data_manager = DataManager(data_dir="data")
    
    # Generate synthetic data for production model training
    synthetic_data = data_manager.generate_synthetic_data(
        n_wells=200,
        grid_size=grid_size,
        x_range=x_range,
        y_range=y_range,
        seed=random_seed,
        save_to_file="synthetic_wells.csv"
    )
    
    # Property columns for training
    property_columns = [p for p in synthetic_data.columns 
                       if p in properties or p.replace("_norm", "") in properties]
    
    # Initialize visualizer
    visualizer = ResultsVisualizer()
    
    # Results storage
    results_list = []
    
    # Function to create a parameter combination
    def create_parameter_set(variations, base_length_scales, idx=0):
        """Create a parameter set with specific variations."""
        # Adjust correlation lengths
        corr_multiplier = variations["correlation_length"]
        length_scales = {
            prop: base_length_scales[prop] * corr_multiplier
            for prop in base_length_scales
        }
        
        # Economic parameters
        gas_price = variations["gas_price"]
        drilling_cost = variations["drilling_cost"]
        target_confidence = variations["target_confidence"]
        
        # Create the parameter set
        return {
            "length_scales": length_scales,
            "economic_params": {
                "gas_price": gas_price,
                "drilling_cost": drilling_cost
            },
            "simulation_params": {
                "target_confidence": target_confidence,
                "max_exploration_wells": max_wells
            },
            "corr_multiplier": corr_multiplier,
            "gas_price": gas_price,
            "drilling_cost": drilling_cost,
            "target_confidence": target_confidence,
            "random_seed": random_seed + idx
        }
    
    # Generate all parameter combinations
    parameter_combinations = []
    idx = 0
    
    for corr_length in parameter_variations["correlation_length"]:
        for gas_price in parameter_variations["gas_price"]:
            for drilling_cost in parameter_variations["drilling_cost"]:
                for target_confidence in parameter_variations["target_confidence"]:
                    for repeat in range(n_repeats):
                        param_set = create_parameter_set(
                            {
                                "correlation_length": corr_length,
                                "gas_price": gas_price,
                                "drilling_cost": drilling_cost,
                                "target_confidence": target_confidence
                            },
                            base_length_scales,
                            idx
                        )
                        parameter_combinations.append(param_set)
                        idx += 1
    
    total_runs = len(parameter_combinations)
    print(f"Running {total_runs} parameter combinations...")
    
    # Run simulations for each parameter combination
    for i, params in enumerate(parameter_combinations):
        print(f"\nParameter set {i+1}/{total_runs}")
        print(f"Correlation multiplier: {params['corr_multiplier']}")
        print(f"Gas price: ${params['gas_price']}/mcf")
        print(f"Drilling cost: ${params['drilling_cost']}M")
        print(f"Target confidence: {params['target_confidence']:.0%}")
        
        # Record start time
        start_time = time.time()
        
        # Create true geological model
        true_model = GaussianProcessGeology(
            grid_size=grid_size,
            x_range=x_range,
            y_range=y_range,
            properties=properties,
            length_scales=params["length_scales"],
            property_ranges=property_ranges,
            kernel_type="exponential",
            random_state=params["random_seed"]
        )
        
        # Create prior model
        prior_model = GaussianProcessGeology(
            grid_size=grid_size,
            x_range=x_range,
            y_range=y_range,
            properties=properties,
            length_scales=params["length_scales"],
            property_ranges=property_ranges,
            kernel_type="exponential",
            random_state=params["random_seed"] + 1000  # Different seed
        )
        
        # Initialize production model
        production_model = ProductionPredictor(
            model_type="linear",
            random_state=params["random_seed"]
        )
        
        # Train production model
        production_model.train(
            geological_properties=synthetic_data,
            production_data=synthetic_data,
            property_columns=property_columns,
            initial_rate_column="qi"
        )
        
        # Initialize economic model
        economic_model = EconomicAssessment(
            gas_price=params["economic_params"]["gas_price"],
            operating_cost=0.5,  # $/mcf
            drilling_cost=params["economic_params"]["drilling_cost"],
            discount_rate=0.1,  # annual rate
            development_years=10,
            drilling_cost_std=1.0,  # $M per well (standard deviation)
            target_profit=100.0,  # $M
            target_confidence=params["simulation_params"]["target_confidence"]
        )
        
        # Optimization parameters
        optimization_params = {
            "exploration_cost": params["economic_params"]["drilling_cost"],
            "n_realizations": 20,
            "n_monte_carlo": 50,
            "development_wells_per_realization": 50,
            "development_years": 10
        }
        
        # Simulation parameters
        simulation_params = {
            "max_exploration_wells": params["simulation_params"]["max_exploration_wells"],
            "target_confidence": params["simulation_params"]["target_confidence"],
            "target_profit": 100.0,  # Target profit ($M)
            "confidence_threshold": 0.05,
            "uncertainty_threshold": 0.05,
            "save_results": False,  # Don't save individual run results
            "create_plots": False   # Don't create plots for each run
        }
        
        # Run subfolder for this combination
        param_dir = os.path.join(
            output_dir, 
            f"c{params['corr_multiplier']}_g{params['gas_price']}_"
            f"d{params['drilling_cost']}_t{params['target_confidence']}_"
            f"r{params['random_seed']}"
        )
        os.makedirs(param_dir, exist_ok=True)
        
        # Initialize simulation controller
        simulation = ExplorationSimulation(
            geological_model=prior_model,
            production_model=production_model,
            economic_model=economic_model,
            data_manager=data_manager,
            visualizer=visualizer,
            true_model=true_model,
            optimization_params=optimization_params,
            simulation_params=simulation_params,
            output_dir=param_dir,
            random_state=params["random_seed"]
        )
        
        # Run exploration campaign
        sim_results = simulation.run_exploration_campaign()
        
        # Calculate run time
        run_time = time.time() - start_time
        
        # Extract key metrics
        n_wells = len(sim_results["stages"])
        final_confidence = sim_results["stages"][-1]["economic_results"]["prob_target"]
        target_met = final_confidence >= params["simulation_params"]["target_confidence"]
        final_npv_mean = sim_results["stages"][-1]["economic_results"]["npv_mean"]
        final_npv_p10 = sim_results["stages"][-1]["economic_results"]["npv_p10"]
        final_npv_p90 = sim_results["stages"][-1]["economic_results"]["npv_p90"]
        
        # Calculate average uncertainty reduction per well
        if n_wells > 1:
            avg_uncertainty_reduction = sum(
                stage.get("uncertainty_reduction", 0.0) 
                for stage in sim_results["stages"][1:]
            ) / (n_wells - 1)
        else:
            avg_uncertainty_reduction = 0.0
        
        # Record results
        result_record = {
            "run_id": i,
            "correlation_multiplier": params["corr_multiplier"],
            "gas_price": params["gas_price"],
            "drilling_cost": params["drilling_cost"],
            "target_confidence": params["target_confidence"],
            "random_seed": params["random_seed"],
            "wells_drilled": n_wells,
            "final_confidence": final_confidence,
            "target_met": target_met,
            "final_npv_mean": final_npv_mean,
            "final_npv_p10": final_npv_p10,
            "final_npv_p90": final_npv_p90,
            "avg_uncertainty_reduction": avg_uncertainty_reduction,
            "run_time_seconds": run_time
        }
        
        results_list.append(result_record)
        
        # Save incremental results
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(os.path.join(output_dir, "sensitivity_results.csv"), index=False)
        
        # Create summary plot for this run
        simulation.create_summary_plots()
        
        print(f"Run complete - Wells: {n_wells}, Confidence: {final_confidence:.2%}, "
              f"Target met: {target_met}, Runtime: {run_time:.1f}s")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Analyze and plot results
    analyze_sensitivity_results(results_df, output_dir, show_plots)
    
    return results_df


def analyze_sensitivity_results(results_df, output_dir, show_plots=False):
    """
    Analyze sensitivity analysis results and create summary plots.
    
    Args:
        results_df: DataFrame with sensitivity analysis results.
        output_dir: Directory to save analysis results.
        show_plots: Whether to display plots interactively.
    """
    print("\nAnalyzing sensitivity results...")
    
    # Save final results
    results_df.to_csv(os.path.join(output_dir, "sensitivity_results.csv"), index=False)
    
    # Create aggregate statistics
    agg_params = {
        'wells_drilled': ['mean', 'std', 'min', 'max'],
        'final_confidence': ['mean', 'std', 'min', 'max'],
        'target_met': ['mean', 'sum'],
        'final_npv_mean': ['mean', 'std', 'min', 'max'],
        'avg_uncertainty_reduction': ['mean', 'std'],
        'run_time_seconds': ['mean', 'sum']
    }
    
    # Group by each parameter
    param_groups = [
        ('correlation_multiplier', 'Correlation Length Multiplier'),
        ('gas_price', 'Gas Price ($/mcf)'),
        ('drilling_cost', 'Drilling Cost ($M)'),
        ('target_confidence', 'Target Confidence')
    ]
    
    for param, param_label in param_groups:
        # Aggregate by this parameter
        grouped = results_df.groupby(param).agg(agg_params)
        
        # Save grouped results
        grouped.to_csv(os.path.join(output_dir, f"grouped_by_{param}.csv"))
        
        # Create parameter impact plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Number of wells needed
        ax = axes[0, 0]
        means = grouped['wells_drilled']['mean']
        std = grouped['wells_drilled']['std']
        ax.bar(means.index.astype(str), means, yerr=std, capsize=5)
        ax.set_xlabel(param_label)
        ax.set_ylabel('Number of Wells')
        ax.set_title(f'Impact of {param_label} on Wells Required')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Final confidence
        ax = axes[0, 1]
        means = grouped['final_confidence']['mean']
        std = grouped['final_confidence']['std']
        ax.bar(means.index.astype(str), means, yerr=std, capsize=5)
        ax.set_xlabel(param_label)
        ax.set_ylabel('Final Confidence')
        ax.set_title(f'Impact of {param_label} on Final Confidence')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Final NPV
        ax = axes[1, 0]
        means = grouped['final_npv_mean']['mean']
        std = grouped['final_npv_mean']['std']
        ax.bar(means.index.astype(str), means, yerr=std, capsize=5)
        ax.set_xlabel(param_label)
        ax.set_ylabel('Final NPV Mean ($M)')
        ax.set_title(f'Impact of {param_label} on Final NPV')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 4: Target met ratio
        ax = axes[1, 1]
        means = grouped['target_met']['mean']
        ax.bar(means.index.astype(str), means)
        ax.set_xlabel(param_label)
        ax.set_ylabel('Target Met Ratio')
        ax.set_title(f'Impact of {param_label} on Target Met Ratio')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{param}_impact.png"), dpi=300)
        
        if not show_plots:
            plt.close(fig)
    
    # Create correlation heatmap
    numeric_cols = [
        'correlation_multiplier', 'gas_price', 'drilling_cost', 'target_confidence',
        'wells_drilled', 'final_confidence', 'final_npv_mean', 'avg_uncertainty_reduction'
    ]
    corr = results_df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.columns)), corr.columns)
    
    # Add correlation values as text
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center',
                    color='white' if abs(corr.iloc[i, j]) > 0.6 else 'black')
    
    plt.title('Parameter Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    
    if not show_plots:
        plt.close()
    
    # Show plots if requested
    if show_plots:
        plt.show()
    
    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter sensitivity analysis.")
    parser.add_argument("--output_dir", type=str, default="sensitivity_results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for reproducibility")
    parser.add_argument("--max_wells", type=int, default=8,
                        help="Maximum number of exploration wells to drill")
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="Number of times to repeat each parameter combination")
    parser.add_argument("--show_plots", action="store_true",
                        help="Display plots interactively")
    
    args = parser.parse_args()
    
    # Run sensitivity analysis
    run_sensitivity_analysis(
        output_dir=args.output_dir,
        random_seed=args.seed,
        max_wells=args.max_wells,
        n_repeats=args.n_repeats,
        show_plots=args.show_plots
    )