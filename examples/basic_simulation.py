#!/usr/bin/env python
"""
Basic exploration simulation example using the BEL package.

This example demonstrates the core functionality of the BEL package
by setting up and running a simple exploration simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pandas as pd

from bel.geological_model import GaussianProcessGeology
from bel.production_model import ProductionPredictor
from bel.economic_model import EconomicAssessment
from bel.data_manager import DataManager
from bel.visualization import ResultsVisualizer
from bel.simulation_controller import ExplorationSimulation


def run_basic_simulation(output_dir='results', random_seed=42, max_wells=10, show_plots=False):
    """
    Run a basic exploration simulation.

    Args:
        output_dir: Directory to save results.
        random_seed: Random seed for reproducibility.
        max_wells: Maximum number of exploration wells to drill.
        show_plots: Whether to display plots interactively.

    Returns:
        The simulation results.
    """
    print("Setting up basic exploration simulation...")

    # Debug info
    print(f"Python working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    
    # Basin parameters - using smaller grid for faster execution
    grid_size = (20, 20)  # Reduced from (50, 50) for faster execution
    x_range = (-94.6, -92.9)
    y_range = (31.3, 33.0)
    
    # Geological properties to model
    properties = [
        "Thickness",  # 50-200 ft
        "Porosity",   # 0.02-0.08 (fraction)
        "Permeability",  # 0.1-0.5 mD
        "TOC",        # 2-8%
        "SW",         # 0.3-0.7 (fraction)
        "Depth",      # 8000-11000 ft
        "Vclay"       # 0.3-0.6 (fraction)
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
    
    # Correlation lengths (spatial variability)
    length_scales = {
        "Thickness": 0.5,
        "Porosity": 0.4,
        "Permeability": 0.4,
        "TOC": 0.3,
        "SW": 0.3,
        "Depth": 0.6,
        "Vclay": 0.3
    }
    
    # Initialize data manager
    data_manager = DataManager(data_dir="data")
    
    # Generate synthetic geological model as the "true" model
    print("Generating synthetic geological model...")
    true_model = GaussianProcessGeology(
        grid_size=grid_size,
        x_range=x_range,
        y_range=y_range,
        properties=properties,
        length_scales=length_scales,
        property_ranges=property_ranges,
        kernel_type="exponential",
        random_state=random_seed
    )
    
    # Initialize prior model with different random seed for exploration
    prior_model = GaussianProcessGeology(
        grid_size=grid_size,
        x_range=x_range,
        y_range=y_range,
        properties=properties,
        length_scales=length_scales,
        property_ranges=property_ranges,
        kernel_type="exponential",
        random_state=random_seed + 1  # Different seed for prior
    )
    
    # Initialize production model
    print("Setting up production model...")
    production_model = ProductionPredictor(
        model_type="linear",
        random_state=random_seed
    )
    
    # Create manually for training
    print("Creating training data...")
    n_samples = 200
    np.random.seed(random_seed)
    
    # Generate geological properties
    training_data = pd.DataFrame({
        'Thickness': np.random.uniform(50, 200, n_samples),
        'Porosity': np.random.uniform(0.02, 0.08, n_samples),
        'Permeability': np.random.uniform(0.1, 0.5, n_samples),
        'TOC': np.random.uniform(2, 8, n_samples),
        'SW': np.random.uniform(0.3, 0.7, n_samples),
        'Depth': np.random.uniform(-11000, -8000, n_samples),
        'Vclay': np.random.uniform(0.3, 0.6, n_samples),
    })
    
    # Generate synthetic production data
    weights = {
        'Thickness': 100,
        'Porosity': 10000,
        'Permeability': 1000,
        'TOC': 50,
        'SW': -100,
        'Depth': -0.01,
        'Vclay': -500
    }
    
    # Calculate initial rates based on properties
    qi = np.zeros(n_samples)
    for prop, weight in weights.items():
        qi += training_data[prop] * weight
    
    # Add random noise
    qi += np.random.normal(0, 1000, n_samples)
    qi = np.maximum(qi, 100)  # Ensure positive values
    
    # Add to training data
    training_data['qi'] = qi
    training_data['well_id'] = np.arange(1, n_samples + 1)
    
    # Save the training data
    os.makedirs(data_manager.data_dir, exist_ok=True)
    training_data.to_csv(os.path.join(data_manager.data_dir, "training_data.csv"), index=False)
    
    # Define property columns for training - ensure these match exactly the DataFrame columns
    # Note: Don't use the properties list, use the actual column names in the training_data
    property_columns = ['Thickness', 'Porosity', 'Permeability', 'TOC', 'SW', 'Depth', 'Vclay']

    # Verify all columns exist in the DataFrame
    for col in property_columns:
        if col not in training_data.columns:
            print(f"Warning: Column {col} missing from training data!")
            print(f"Available columns: {list(training_data.columns)}")
            raise ValueError(f"Missing column: {col}")

    print("Training production model...")
    production_model.train(
        geological_properties=training_data,
        production_data=training_data,
        property_columns=property_columns,
        initial_rate_column="qi"
    )
    
    # Initialize economic model
    print("Setting up economic model...")
    economic_model = EconomicAssessment(
        gas_price=4.0,  # $/mcf
        operating_cost=0.5,  # $/mcf
        drilling_cost=10.0,  # $M per well
        discount_rate=0.1,  # annual rate
        development_years=10,
        drilling_cost_std=1.0,  # $M per well (standard deviation)
        target_profit=100.0,  # $M
        target_confidence=0.9  # 90%
    )
    
    # Initialize visualizer
    visualizer = ResultsVisualizer()
    
    # Optimization parameters - reduced for faster execution
    optimization_params = {
        "exploration_cost": 10.0,  # $M per exploration well
        "n_realizations": 10,  # Reduced from 20 for faster execution
        "n_monte_carlo": 20,  # Reduced from 50 for faster execution
        "development_wells_per_realization": 20,  # Reduced from 50 for faster execution
        "development_years": 10  # Years of production
    }
    
    # Simulation parameters
    simulation_params = {
        "max_exploration_wells": max_wells,
        "target_confidence": 0.9,  # Target confidence level
        "target_profit": 100.0,  # Target profit ($M)
        "confidence_threshold": 0.05,  # Stop if confidence change is less than this
        "uncertainty_threshold": 0.05,  # Stop if uncertainty reduction is less than this
        "save_results": True,
        "create_plots": True
    }
    
    # Initialize simulation controller
    print("Initializing simulation controller...")
    simulation = ExplorationSimulation(
        geological_model=prior_model,
        production_model=production_model,
        economic_model=economic_model,
        data_manager=data_manager,
        visualizer=visualizer,
        true_model=true_model,
        optimization_params=optimization_params,
        simulation_params=simulation_params,
        output_dir=output_dir,
        random_state=random_seed
    )
    
    # Run exploration campaign
    print(f"Running exploration campaign (max {max_wells} wells)...")
    results = simulation.run_exploration_campaign()
    
    # Display summary
    n_wells = len(results["stages"])
    final_confidence = results["stages"][-1]["economic_results"]["prob_target"]
    target_met = final_confidence >= simulation_params["target_confidence"]
    
    print("\nExploration Campaign Summary:")
    print(f"Wells drilled: {n_wells}")
    print(f"Final confidence level: {final_confidence:.2%}")
    print(f"Target confidence met: {target_met}")
    print(f"Results saved to: {output_dir}")
    
    # Show plots if requested
    if show_plots:
        plt.show()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a basic exploration simulation.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_wells", type=int, default=3,
                        help="Maximum number of exploration wells to drill")
    parser.add_argument("--show_plots", action="store_true",
                        help="Display plots interactively")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run simulation
    run_basic_simulation(
        output_dir=args.output_dir,
        random_seed=args.seed,
        max_wells=args.max_wells,
        show_plots=args.show_plots
    )