"""Controller for running exploration simulations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import datetime
import os
import json
import logging
from copy import deepcopy

from ..geological_model import GaussianProcessGeology
from ..production_model import ProductionPredictor
from ..economic_model import EconomicAssessment
from ..optimization import ValueOfInformation
from ..data_manager import DataManager
from ..visualization import ResultsVisualizer


class ExplorationSimulation:
    """
    Orchestrates the main exploration loop.
    
    This class manages the exploration simulation workflow,
    including well placement, model updates, and stopping criteria.
    """
    
    def __init__(
        self,
        geological_model: GaussianProcessGeology,
        production_model: ProductionPredictor,
        economic_model: EconomicAssessment,
        data_manager: Optional[DataManager] = None,
        visualizer: Optional[ResultsVisualizer] = None,
        true_model: Optional[GaussianProcessGeology] = None,
        optimization_params: Optional[Dict[str, Any]] = None,
        simulation_params: Optional[Dict[str, Any]] = None,
        output_dir: str = "results",
        random_state: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the exploration simulation.
        
        Args:
            geological_model: Prior geological model.
            production_model: Production prediction model.
            economic_model: Economic assessment model.
            data_manager: Optional data manager for I/O operations.
            visualizer: Optional visualizer for creating plots.
            true_model: True geological model for simulation. If None, uses the prior model.
            optimization_params: Parameters for the optimization engine.
            simulation_params: Parameters for the simulation.
            output_dir: Directory for saving results.
            random_state: Random seed for reproducibility.
            logger: Logger for logging messages.
        """
        self.geological_model = geological_model
        self.production_model = production_model
        self.economic_model = economic_model
        self.data_manager = data_manager or DataManager()
        self.visualizer = visualizer or ResultsVisualizer()
        self.true_model = true_model or deepcopy(geological_model)  # Use prior as truth if not provided
        self.random_state = random_state
        self.output_dir = output_dir
        
        # Set up logging
        self.logger = logger or self._setup_logger()
        
        # Default optimization parameters
        default_opt_params = {
            "exploration_cost": 10.0,  # $M per exploration well
            "n_realizations": 50,
            "n_monte_carlo": 100,
            "development_wells_per_realization": 100,
            "development_years": 10
        }
        
        # Default simulation parameters
        default_sim_params = {
            "max_exploration_wells": 20,
            "target_confidence": 0.9,
            "target_profit": 100.0,  # $M
            "confidence_threshold": 0.05,  # Stop if confidence change is less than this
            "uncertainty_threshold": 0.05,  # Stop if uncertainty reduction is less than this
            "save_results": True,
            "create_plots": True
        }
        
        # Update with provided parameters
        self.optimization_params = {**default_opt_params, **(optimization_params or {})}
        self.simulation_params = {**default_sim_params, **(simulation_params or {})}
        
        # Initialize optimizer
        self.optimizer = ValueOfInformation(
            geological_model=self.geological_model,
            production_model=self.production_model,
            economic_model=self.economic_model,
            exploration_cost=self.optimization_params["exploration_cost"],
            n_realizations=self.optimization_params["n_realizations"],
            n_monte_carlo=self.optimization_params["n_monte_carlo"],
            random_state=self.random_state
        )
        
        # Initialize results storage
        self.results = {
            "stages": [],
            "x_grid": self.geological_model.X,
            "y_grid": self.geological_model.Y,
            "simulation_params": self.simulation_params,
            "optimization_params": self.optimization_params,
            "properties": self.geological_model.properties,
            "property_ranges": self.geological_model.property_ranges
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the simulation."""
        logger = logging.getLogger("exploration_simulation")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
        
        return logger
    
    def run_exploration_campaign(
        self,
        initial_well_location: Optional[Tuple[float, float]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the full exploration campaign.
        
        Args:
            initial_well_location: Optional coordinates for the first well.
            callback: Optional callback function called after each exploration stage.
            
        Returns:
            Dictionary containing all simulation results.
        """
        self.logger.info("Starting exploration campaign")
        
        # Set random seed if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Reinitialize results
        self.results["stages"] = []
        
        # Track stopping criteria metrics
        prev_confidence = 0.0
        prev_uncertainty = 1.0
        
        # Place the first well
        if initial_well_location is not None:
            next_well_location = initial_well_location
            self.logger.info(f"Using provided initial well location: {next_well_location}")
        else:
            # Calculate VOI surface to determine first well location
            self.logger.info("Calculating initial VOI surface")
            self.optimizer.calculate_voi_surface(
                development_wells_per_realization=self.optimization_params["development_wells_per_realization"],
                development_years=self.optimization_params["development_years"]
            )
            next_well_location = self.optimizer.select_next_well_location()
            self.logger.info(f"Selected initial well location: {next_well_location}")
        
        # Main exploration loop
        max_wells = self.simulation_params["max_exploration_wells"]
        self.logger.info(f"Planning to drill up to {max_wells} exploration wells")
        
        for well_idx in range(max_wells):
            self.logger.info(f"Exploration well {well_idx + 1}/{max_wells}")
            
            # Simulate drilling the well
            stage_results = self.simulate_well_drilling(next_well_location)
            
            # Add to results
            self.results["stages"].append(stage_results)
            
            # Call callback if provided
            if callback:
                callback(stage_results)
            
            # Check stopping criteria
            economic_results = stage_results["economic_results"]
            current_confidence = economic_results["prob_target"]
            uncertainty_reduction = stage_results.get("uncertainty_reduction", 0.0)
            
            stop, stop_reason = self.check_stopping_criteria(
                current_confidence, 
                prev_confidence,
                uncertainty_reduction,
                prev_uncertainty,
                well_idx + 1
            )
            
            if stop:
                self.logger.info(f"Stopping exploration: {stop_reason}")
                self.results["stop_reason"] = stop_reason
                break
            
            # Update metrics for next iteration
            prev_confidence = current_confidence
            prev_uncertainty = 1.0 - uncertainty_reduction  # Convert reduction to remaining uncertainty
            
            # Select next well location
            self.logger.info("Calculating VOI surface for next well")
            self.optimizer.calculate_voi_surface(
                development_wells_per_realization=self.optimization_params["development_wells_per_realization"],
                development_years=self.optimization_params["development_years"]
            )
            next_well_location = self.optimizer.select_next_well_location()
            self.logger.info(f"Selected next well location: {next_well_location}")
        
        # Finalize results
        self.results["total_wells"] = len(self.results["stages"])
        self.results["final_confidence"] = self.results["stages"][-1]["economic_results"]["prob_target"]
        self.results["target_met"] = self.results["final_confidence"] >= self.simulation_params["target_confidence"]
        
        self.logger.info(f"Exploration complete. Drilled {self.results['total_wells']} wells.")
        self.logger.info(f"Final confidence: {self.results['final_confidence']:.2%}")
        self.logger.info(f"Target met: {self.results['target_met']}")
        
        # Save results
        if self.simulation_params["save_results"]:
            self.save_results()
        
        # Create summary plots
        if self.simulation_params["create_plots"]:
            self.create_summary_plots()
        
        return self.results
    
    def simulate_well_drilling(
        self,
        well_location: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Simulate drilling a well at the specified location.
        
        Args:
            well_location: Coordinates (x, y) of the well.
            
        Returns:
            Dictionary containing the stage results.
        """
        x, y = well_location
        location_array = np.array([[x, y]])
        
        # Store pre-drilling model state
        pre_voi = self.optimizer.voi_surface.copy() if self.optimizer.voi_surface is not None else None
        pre_uncertainty = self.geological_model.calculate_uncertainty()
        pre_property_maps = {}
        
        for prop in self.geological_model.properties:
            pre_property_maps[prop] = self.geological_model.get_property_mean(prop)
        
        # Sample true values from the true model
        true_realizations = self.true_model.sample_realizations(1)
        true_values = {}
        
        for prop in self.true_model.properties:
            # Get the value at the specified location
            i = np.argmin(np.sum((self.true_model.grid_points - location_array)**2, axis=1))
            true_values[prop] = true_realizations[prop][0].flatten()[i]
        
        # Update geological model with new data
        self.geological_model.update_with_well_data(location_array, true_values)
        
        # Calculate post-drilling property maps and uncertainty
        post_property_maps = {}
        post_uncertainty = self.geological_model.calculate_uncertainty()
        
        for prop in self.geological_model.properties:
            post_property_maps[prop] = self.geological_model.get_property_mean(prop)
        
        # Calculate uncertainty reduction
        uncertainty_reduction = {}
        avg_reduction = 0.0
        
        for prop in self.geological_model.properties:
            pre_uncert_sum = np.sum(pre_uncertainty[prop])
            post_uncert_sum = np.sum(post_uncertainty[prop])
            reduction = 1.0 - (post_uncert_sum / pre_uncert_sum)
            uncertainty_reduction[prop] = reduction
            avg_reduction += reduction
        
        avg_reduction /= len(self.geological_model.properties)
        
        # Generate property realizations
        realizations = self.geological_model.sample_realizations(
            self.optimization_params["n_realizations"]
        )
        
        # Predict production for development wells
        development_wells = self.optimization_params["development_wells_per_realization"]
        time_points_years = np.linspace(0, self.optimization_params["development_years"], 
                                       self.optimization_params["development_years"] * 12 + 1)
        
        # Aggregate properties for production prediction
        properties_flat = {}
        for prop in realizations:
            properties_flat[prop] = np.array([real.flatten() for real in realizations[prop]])
        
        # Predict production for each realization
        production_results = self.production_model.predict_full_well_performance(
            properties_flat,
            time_points_years=time_points_years,
            development_years=self.optimization_params["development_years"]
        )
        
        # Assess economics
        economic_results = self.economic_model.assess_profitability_distribution(
            production_results["production_profiles"],
            time_points_years,
            development_wells,
            num_simulations=self.optimization_params["n_monte_carlo"]
        )
        
        # Calculate VOI (retrospective)
        voi_value = 0.0
        if len(self.results["stages"]) > 0:
            prev_confidence = self.results["stages"][-1]["economic_results"]["prob_target"]
            current_confidence = economic_results["prob_target"]
            confidence_improvement = current_confidence - prev_confidence
            
            prev_npv_mean = self.results["stages"][-1]["economic_results"]["npv_mean"]
            current_npv_mean = economic_results["npv_mean"]
            value_improvement = current_npv_mean - prev_npv_mean
            
            voi_value = value_improvement - self.optimization_params["exploration_cost"]
        
        # Compile stage results
        stage_results = {
            "well_location": well_location,
            "true_values": true_values,
            "pre_property_maps": pre_property_maps,
            "post_property_maps": post_property_maps,
            "pre_uncertainty": pre_uncertainty,
            "post_uncertainty": post_uncertainty,
            "uncertainty_reduction": avg_reduction,
            "property_uncertainty_reduction": uncertainty_reduction,
            "pre_voi_surface": pre_voi,
            "production_results": production_results,
            "economic_results": economic_results,
            "voi_value": voi_value,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return stage_results
    
    def check_stopping_criteria(
        self,
        current_confidence: float,
        prev_confidence: float,
        uncertainty_reduction: float,
        prev_uncertainty: float,
        well_count: int
    ) -> Tuple[bool, str]:
        """
        Check if stopping criteria are met.
        
        Args:
            current_confidence: Current probability of meeting the target profit.
            prev_confidence: Previous probability of meeting the target profit.
            uncertainty_reduction: Uncertainty reduction from this well.
            prev_uncertainty: Previous uncertainty level.
            well_count: Number of wells drilled so far.
            
        Returns:
            Tuple of (should_stop, reason).
        """
        # Check if target confidence is reached
        target_confidence = self.simulation_params["target_confidence"]
        if current_confidence >= target_confidence:
            return True, f"Target confidence of {target_confidence:.1%} reached: {current_confidence:.1%}"
        
        # Check if confidence is not improving significantly
        confidence_threshold = self.simulation_params["confidence_threshold"]
        confidence_improvement = current_confidence - prev_confidence
        if well_count > 1 and confidence_improvement < confidence_threshold:
            return True, f"Confidence improvement below threshold: {confidence_improvement:.1%} < {confidence_threshold:.1%}"
        
        # Check if uncertainty reduction is diminishing
        uncertainty_threshold = self.simulation_params["uncertainty_threshold"]
        if well_count > 1 and uncertainty_reduction < uncertainty_threshold:
            return True, f"Uncertainty reduction below threshold: {uncertainty_reduction:.1%} < {uncertainty_threshold:.1%}"
        
        # Maximum well count reached
        if well_count >= self.simulation_params["max_exploration_wells"]:
            return True, f"Maximum exploration wells reached: {well_count}"
        
        # Continue exploration
        return False, ""
    
    def save_results(self) -> str:
        """
        Save simulation results to files.
        
        Returns:
            Path to the saved results.
        """
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create JSON-friendly version of results (without numpy arrays)
        json_results = deepcopy(self.results)
        
        # Remove numpy arrays and other non-serializable objects
        for stage in json_results["stages"]:
            # Remove large arrays
            keys_to_remove = [
                "pre_property_maps", "post_property_maps", 
                "pre_uncertainty", "post_uncertainty", 
                "pre_voi_surface", "production_results"
            ]
            for key in keys_to_remove:
                if key in stage:
                    del stage[key]
            
            # Convert remaining numpy arrays to lists
            for key in list(stage.keys()):
                if isinstance(stage[key], dict):
                    for subkey in list(stage[key].keys()):
                        if isinstance(stage[key][subkey], np.ndarray):
                            if len(stage[key][subkey]) < 1000:  # Only include small arrays
                                stage[key][subkey] = stage[key][subkey].tolist()
                            else:
                                del stage[key][subkey]
                elif isinstance(stage[key], np.ndarray):
                    if len(stage[key]) < 1000:  # Only include small arrays
                        stage[key] = stage[key].tolist()
                    else:
                        del stage[key]
        
        # Remove grid arrays
        if "x_grid" in json_results:
            del json_results["x_grid"]
        if "y_grid" in json_results:
            del json_results["y_grid"]
        
        # Save summary JSON
        summary_path = os.path.join(self.output_dir, f"exploration_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Saved exploration summary to {summary_path}")
        
        return summary_path
    
    def create_summary_plots(self) -> Dict[str, str]:
        """
        Create summary plots of the exploration results.
        
        Returns:
            Dictionary mapping plot names to file paths.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_paths = {}
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot final property maps
        if len(self.results["stages"]) > 0:
            final_stage = self.results["stages"][-1]
            well_locations = np.array([stage["well_location"] for stage in self.results["stages"]])
            
            for prop in self.geological_model.properties:
                property_map = self.geological_model.get_property_mean(prop)
                uncertainty_map = self.geological_model.calculate_uncertainty()[prop]
                
                # Property map
                fig = self.visualizer.create_well_location_maps(
                    property_map=property_map,
                    x_grid=self.results["x_grid"],
                    y_grid=self.results["y_grid"],
                    well_locations=well_locations,
                    uncertainty_map=uncertainty_map,
                    property_name=prop,
                    property_range=self.geological_model.property_ranges[prop],
                    interactive=False
                )
                
                path = os.path.join(plots_dir, f"{prop}_final_map_{timestamp}.png")
                fig.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                plot_paths[f"{prop}_map"] = path
            
            # Plot NPV distribution
            economic_results = [stage["economic_results"] for stage in self.results["stages"]]
            stage_labels = [f"Well {i+1}" for i in range(len(economic_results))]
            
            fig = self.visualizer.plot_economic_distributions(
                economic_results=economic_results,
                stage_labels=stage_labels,
                target_profit=self.simulation_params["target_profit"],
                target_confidence=self.simulation_params["target_confidence"]
            )
            
            path = os.path.join(plots_dir, f"economic_evolution_{timestamp}.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plot_paths["economic_evolution"] = path
            
            # Plot uncertainty evolution for first property
            prop = self.geological_model.properties[0]
            uncertainty_history = [
                {prop: stage["pre_uncertainty"][prop]} 
                for stage in self.results["stages"]
            ]
            
            # Add the final uncertainty state
            uncertainty_history.append(
                {prop: final_stage["post_uncertainty"][prop]}
            )
            
            well_history = [
                np.array([stage["well_location"]]) 
                for stage in self.results["stages"]
            ]
            
            # Skip the last well location for uncertainty plot
            well_history = well_history[:-1] + [np.array([final_stage["well_location"]])]
            
            stage_labels = [f"Initial"] + [f"After Well {i+1}" for i in range(len(self.results["stages"]))]
            
            fig = self.visualizer.plot_uncertainty_evolution(
                uncertainty_history=uncertainty_history,
                property_name=prop,
                x_grid=self.results["x_grid"],
                y_grid=self.results["y_grid"],
                well_history=well_history,
                stage_labels=stage_labels
            )
            
            path = os.path.join(plots_dir, f"uncertainty_evolution_{timestamp}.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plot_paths["uncertainty_evolution"] = path
            
            # Create exploration summary dashboard
            fig = self.visualizer.create_exploration_summary_dashboard(
                exploration_results=self.results,
                interactive=False
            )
            
            path = os.path.join(plots_dir, f"exploration_summary_{timestamp}.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plot_paths["exploration_summary"] = path
            
            self.logger.info(f"Saved {len(plot_paths)} plots to {plots_dir}")
        
        return plot_paths