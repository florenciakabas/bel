import sys
import os
import pdb
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from scipy import stats
from typing import Tuple, List


class BayesianEvidentialLearning:

    def __init__(
        self,
        basin_size:Tuple[int, int]=(10000, 10000),
        grid_resolution:int=500,
        param_names:List[str]=['depth', 'porosity', 'permeability', 'thickness', 'saturation']
    ):
        """
        Initialize the Bayesian Evidential Learning framework
        
        Parameters:
        -----------
        basin_size : tuple
            Size of the basin in meters (x, y)
        grid_resolution : int
            Resolution of the grid for modeling
        param_names : list, optional
            Names of parameters to model. Default: depth, porosity, permeability, thickness, saturation
        """

        # Add the src directory to the Python path
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        with open('data/inputparams.json', 'r') as f:
            inputs = json.load(f)
            priors = inputs['priors']
            length_scales = inputs['length_scales']
            economic_params = inputs['economic_params']

        self.basin_size = basin_size
        self.grid_resolution = grid_resolution
        self.param_names = param_names
        
        # Create grid
        self.x_grid = np.linspace(0, basin_size[0], int(basin_size[0]/grid_resolution)+1)
        self.y_grid = np.linspace(0, basin_size[1], int(basin_size[1]/grid_resolution)+1)
        self.X_grid, self.Y_grid = np.meshgrid(self.x_grid, self.y_grid)
        
        # Prior means for each parameter
        self.prior_means = {
            'depth': 2500,          # meters
            'porosity': 0.15,       # fraction
            'permeability': 50,     # mD
            'thickness': 20,        # meters
            'saturation': 0.6       # fraction
        }
        # TODO: how do I comment on json files ? add comments for units to json file 
        self.prior_means = priors
        
        # Length scales for spatial correlation
        self.length_scales = {
            'depth': 3000,          # depth varies smoothly
            'porosity': 1500,       # medium variation
            'permeability': 800,    # high local variation
            'thickness': 2000,      # medium variation
            'saturation': 1200      # medium-high variation
        }
        # TODO: how do I comment on json files ? add comments for units to json file 
        self.length_scales = length_scales
        
        # Create cross-correlation matrix (simplified)
        # Order: depth, porosity, permeability, thickness, saturation
        self.cross_correlation = np.array([
            [1.0,  -0.6,  -0.4,   0.1,   0.0],  # depth
            [-0.6,  1.0,   0.7,   0.2,   0.1],  # porosity
            [-0.4,  0.7,   1.0,   0.2,   0.0],  # permeability
            [0.1,   0.2,   0.2,   1.0,   0.0],  # thickness
            [0.0,   0.1,   0.0,   0.0,   1.0]   # saturation
        ])
        
        # Create the spatial model
        self.model = MultiOutputGP(
            param_names=self.param_names,
            length_scales=self.length_scales,
            cross_correlation_matrix=self.cross_correlation
        )
        
        pdb.set_trace()
        # Well data
        self.well_locations = []
        self.well_measurements = []
        self.well_logs = []
        
        # Economic parameters
        self.economic_params = {
            'oil_price': 60,            # $/barrel
            'discount_rate': 0.1,       # 10% annual
            'well_cost': 10e6,          # $10 million per well
            'development_cost': 500e6,  # $500 million base development cost
            'opex_per_barrel': 15,      # $15/barrel operating cost
            'recovery_factor': 0.3,     # 30% recovery
            'formation_volume_factor': 1.2,  # reservoir -> surface conversion
            'area': 2000 * 2000         # m²
        }
        self.economic_params = economic_params
        
        # True basin properties (for simulation only)
        self._generate_true_basin()
    
    def _generate_true_basin(self):
        """Generate true basin properties for simulation"""
        # This would be unknown in real life, but for simulation we need ground truth
        
        # Simple depth trend (deeper to the north)
        def true_depth(x, y):
            base = 2000
            trend = 1000 * (y / self.basin_size[1])
            return base + trend
        
        # Porosity (anticorrelated with depth + local features)
        def true_porosity(x, y):
            depth = true_depth(x, y)
            base = 0.25 - 0.05 * (depth - 2000) / 1000
            
            # Add a high porosity feature (e.g., channel)
            channel_x = self.basin_size[0] * 0.7
            channel_y = self.basin_size[1] * 0.5
            channel_width = self.basin_size[0] * 0.1
            
            dist_to_channel = np.sqrt((x - channel_x)**2 + (y - channel_y)**2)
            channel_effect = 0.1 * np.exp(-dist_to_channel**2 / (2 * channel_width**2))
            
            return min(0.35, max(0.05, base + channel_effect))
        
        # Permeability (correlated with porosity)
        def true_permeability(x, y):
            poro = true_porosity(x, y)
            
            # Permeability-porosity transform (simplified Carman-Kozeny)
            base_perm = 10**(15 * poro - 1)
            
            # Add high perm streak
            streak_x = self.basin_size[0] * 0.3
            streak_width = self.basin_size[0] * 0.05
            
            dist_to_streak = abs(x - streak_x)
            streak_effect = 100 * np.exp(-dist_to_streak**2 / (2 * streak_width**2))
            
            return max(0.1, base_perm + streak_effect)
        
        # Thickness (thicker in the center)
        def true_thickness(x, y):
            center_x = self.basin_size[0] / 2
            center_y = self.basin_size[1] / 2
            
            dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt((self.basin_size[0]/2)**2 + (self.basin_size[1]/2)**2)
            
            # Thicker in center, thinner at edges
            thickness = 35 * (1 - 0.7 * (dist_to_center / max_dist))
            
            return max(5, thickness)
        
        # Saturation (higher at structural highs - inverse of depth)
        def true_saturation(x, y):
            depth = true_depth(x, y)
            max_depth = true_depth(0, self.basin_size[1])
            min_depth = true_depth(0, 0)
            
            # Higher saturation at shallower depths
            base = 0.4 + 0.5 * (max_depth - depth) / (max_depth - min_depth)
            
            # Add local variations for realism
            local = 0.1 * np.sin(x / 1000) * np.cos(y / 1200)
            
            return min(0.9, max(0.1, base + local))
        
        # Store the true functions
        self.true_functions = {
            'depth': true_depth,
            'porosity': true_porosity,
            'permeability': true_permeability,
            'thickness': true_thickness,
            'saturation': true_saturation
        }
        
        # Pre-compute true values on the grid for visualization
        self.true_values = {}
        for param in self.param_names:
            self.true_values[param] = np.zeros_like(self.X_grid)
            for i in range(self.X_grid.shape[0]):
                for j in range(self.X_grid.shape[1]):
                    x, y = self.X_grid[i, j], self.Y_grid[i, j]
                    self.true_values[param][i, j] = self.true_functions[param](x, y)
    
    def drill_well(self, location, with_logs=True):
        """
        Drill a well at the specified location
        
        Parameters:
        -----------
        location : tuple
            (x, y) coordinates to drill
        with_logs : bool
            Whether to generate detailed logs
            
        Returns:
        --------
        measurements : dict
            Well measurements
        """
        x, y = location
        
        # Store location
        self.well_locations.append(location)
        
        # Get true values at this location
        true_values = {}
        for param in self.param_names:
            true_values[param] = self.true_functions[param](x, y)
        
        # Simulate measurements
        measurements = simulate_well_measurements(true_values)
        self.well_measurements.append(measurements)
        
        # Generate logs if requested
        if with_logs:
            # Generate a simple vertical well trajectory
            depth = true_values['depth']
            thickness = true_values['thickness']
            
            # Well points from surface to below reservoir
            n_points = 100
            well_z = np.linspace(0, depth + thickness + 50, n_points)
            well_trajectory = np.array([(x, y, z) for z in well_z])
            
            # Create true value functions for log generation
            true_log_funcs = {}
            for param in self.param_names:
                if param == 'depth':
                    # This is just the measured depth
                    true_log_funcs[param] = lambda x, y, z: z
                elif param == 'porosity':
                    # Only has values in the reservoir interval
                    true_log_funcs[param] = lambda x, y, z: \
                        self.true_functions[param](x, y) if depth <= z <= depth + thickness else 0
                elif param == 'permeability':
                    true_log_funcs[param] = lambda x, y, z: \
                        self.true_functions[param](x, y) if depth <= z <= depth + thickness else 0
                elif param == 'thickness':
                    # Not a log curve
                    true_log_funcs[param] = lambda x, y, z: 0
                elif param == 'saturation':
                    true_log_funcs[param] = lambda x, y, z: \
                        self.true_functions[param](x, y) if depth <= z <= depth + thickness else 0
            
            # Generate log
            log_data = simulate_full_well_log(true_log_funcs, well_trajectory)
            self.well_logs.append(log_data)
        
        # Update the model with new data
        self._update_model()
        
        return measurements
    
    def _update_model(self):
        """Update the spatial model with all well data"""
        # Prepare data for model
        X = np.array(self.well_locations)
        Y_dict = {param: [] for param in self.param_names}
        
        for measurements in self.well_measurements:
            for param in self.param_names:
                if param in measurements:
                    Y_dict[param].append(measurements[param])
        
        # Fit the model
        if len(X) > 1:  # Need at least 2 points
            self.model.fit(X, Y_dict)
    
    def predict_basin_properties(self):
        """
        Predict basin properties across the entire grid
        
        Returns:
        --------
        means : dict
            Mean predictions for each parameter
        uncertainties : dict
            Standard deviations for each parameter
        """
        # Reshape grid for prediction
        X_pred = np.vstack((self.X_grid.ravel(), self.Y_grid.ravel())).T
        
        # Make predictions
        means, uncertainties = self.model.predict(X_pred)
        
        # Reshape back to grid
        grid_means = {}
        grid_uncertainties = {}
        
        for param in self.param_names:
            if param in means:
                grid_means[param] = means[param].reshape(self.X_grid.shape)
                grid_uncertainties[param] = uncertainties[param].reshape(self.X_grid.shape)
        
        return grid_means, grid_uncertainties
    
    def calculate_npv(self, parameters):
        """
        Calculate Net Present Value based on parameters
        
        Parameters:
        -----------
        parameters : dict
            Dictionary of parameter values (can be means or samples)
            
        Returns:
        --------
        npv : float
            Net Present Value in dollars
        """
        # Extract parameters
        porosity = parameters['porosity']
        permeability = parameters['permeability']
        thickness = parameters['thickness']
        saturation = parameters['saturation']
        
        # Economic parameters
        oil_price = self.economic_params['oil_price']
        discount_rate = self.economic_params['discount_rate']
        development_cost = self.economic_params['development_cost']
        opex = self.economic_params['opex_per_barrel']
        recovery = self.economic_params['recovery_factor']
        fvf = self.economic_params['formation_volume_factor']
        area = self.economic_params['area']
        
        # Calculate Original Oil in Place (OOIP)
        # 7758 is the conversion factor from acre-ft to barrels
        ooip = 7758 * area * thickness * porosity * saturation / fvf
        
        # Recoverable reserves
        reserves = ooip * recovery
        
        # Simplified production profile
        # Peak rate depends on permeability, thickness, and reserves
        peak_rate = 0.1 * permeability * thickness * reserves / 1e6
        decline_rate = 0.15  # 15% annual decline
        
        # Calculate NPV
        npv = -development_cost  # Initial investment
        
        # Add cash flows from production
        for year in range(1, 21):  # 20-year project
            # Production declines exponentially
            yearly_production = peak_rate * np.exp(-decline_rate * (year-1)) * 365
            
            # Cap production at remaining reserves
            yearly_production = min(yearly_production, reserves)
            reserves -= yearly_production
            
            # Cash flow
            revenue = yearly_production * oil_price
            operating_cost = yearly_production * opex
            cash_flow = revenue - operating_cost
            
            # Discount and add to NPV
            npv += cash_flow / ((1 + discount_rate) ** year)
            
            # Stop if no more reserves
            if reserves <= 0:
                break
        
        return npv
    

    def calculate_evsi(self, candidate_location, n_simulations=100):
        """
        Calculate Expected Value of Sample Information for a potential well
        
        Parameters:
        -----------
        candidate_location : tuple
            (x, y) coordinates for the potential well
        n_simulations : int
            Number of simulations to run
            
        Returns:
        --------
        evsi : float
            Expected Value of Sample Information in dollars
        """
        import copy
        
        # Store current state
        current_locations = copy.deepcopy(self.well_locations)
        current_measurements = copy.deepcopy(self.well_measurements)
        current_logs = copy.deepcopy(self.well_logs)
        
        # Get current prediction and calculate expected NPV
        current_means, _ = self.predict_basin_properties()
        current_npv = self.calculate_npv({
            param: np.mean(current_means.get(param, 0)) 
            for param in self.param_names
        })
        
        # Current optimal decision
        develop_now = current_npv > 0
        ev_without_info = max(0, current_npv) if develop_now else 0
        
        # Run simulations
        total_ev_with_info = 0
        
        for i in range(n_simulations):
            # Restore original state
            self.well_locations = copy.deepcopy(current_locations)
            self.well_measurements = copy.deepcopy(current_measurements)
            self.well_logs = copy.deepcopy(current_logs)
            self._update_model()
            
            # Drill simulated well
            self.drill_well(candidate_location)
            
            # Get updated prediction
            updated_means, _ = self.predict_basin_properties()
            updated_npv = self.calculate_npv({
                param: np.mean(updated_means.get(param, 0)) 
                for param in self.param_names
            })
            
            # Optimal decision after information
            develop_after_info = updated_npv > 0
            ev_with_this_outcome = max(0, updated_npv) if develop_after_info else 0
            
            total_ev_with_info += ev_with_this_outcome
        
        # Restore original state
        self.well_locations = copy.deepcopy(current_locations)
        self.well_measurements = copy.deepcopy(current_measurements)
        self.well_logs = copy.deepcopy(current_logs)
        self._update_model()
        
        # Calculate EVSI
        ev_with_info = total_ev_with_info / n_simulations
        evsi = ev_with_info - ev_without_info - self.economic_params['well_cost']
        
        return evsi

    def optimize_next_well(self, n_candidates=100):
        """
        Find the optimal location for the next exploration well
        
        Parameters:
        -----------
        n_candidates : int
            Number of candidate locations to evaluate
            
        Returns:
        --------
        best_location : tuple
            (x, y) coordinates of the optimal location
        best_evsi : float
            EVSI of the optimal location
        """
        # Get current uncertainty map
        _, uncertainties = self.predict_basin_properties()
        
        # Combine uncertainties weighted by economic impact
        total_uncertainty = np.zeros_like(self.X_grid)
        if 'porosity' in uncertainties:
            total_uncertainty += 10 * uncertainties['porosity']
        if 'permeability' in uncertainties:
            total_uncertainty += 0.1 * uncertainties['permeability']
        if 'thickness' in uncertainties:
            total_uncertainty += 1 * uncertainties['thickness']
        if 'saturation' in uncertainties:
            total_uncertainty += 5 * uncertainties['saturation']
        
        # Find high uncertainty areas
        flat_uncertainty = total_uncertainty.ravel()
        flat_x = self.X_grid.ravel()
        flat_y = self.Y_grid.ravel()
        
        # Sort by uncertainty
        sorted_indices = np.argsort(flat_uncertainty)[::-1]  # Descending
        
        # Take top n_candidates
        candidate_indices = sorted_indices[:n_candidates]
        candidate_locations = [(flat_x[i], flat_y[i]) for i in candidate_indices]
        
        # Evaluate EVSI for each candidate (can be slow)
        best_location = None
        best_evsi = float('-inf')
        
        for i, location in enumerate(candidate_locations):
            # Skip if too close to existing wells
            too_close = False
            for well_loc in self.well_locations:
                distance = np.sqrt((location[0] - well_loc[0])**2 + 
                                (location[1] - well_loc[1])**2)
                if distance < self.grid_resolution * 2:  # Minimum spacing
                    too_close = True
                    break
            
            if too_close:
                continue
                
            # Calculate EVSI
            evsi = self.calculate_evsi(location, n_simulations=50)
            
            if evsi > best_evsi:
                best_evsi = evsi
                best_location = location
                
            print(f"Candidate {i+1}/{len(candidate_locations)}: EVSI = ${evsi/1e6:.2f}M")
        
        return best_location, best_evsi

    def determine_wells_needed(self, confidence_threshold=0.9, max_wells=10):
        """
        Determine how many wells are needed to reach the desired confidence level
        
        Parameters:
        -----------
        confidence_threshold : float
            Desired probability of profitability
        max_wells : int
            Maximum number of wells to consider
            
        Returns:
        --------
        results : dict
            Dictionary with exploration results
        """
        import copy
        
        # Store original state
        original_locations = copy.deepcopy(self.well_locations)
        original_measurements = copy.deepcopy(self.well_measurements)
        original_logs = copy.deepcopy(self.well_logs)
        
        # Reset if we already have wells
        if self.well_locations:
            self.well_locations = []
            self.well_measurements = []
            self.well_logs = []
            self._update_model()
        
        # Track exploration progress
        wells_drilled = 0
        exploration_path = []
        confidences = []
        npvs = []
        
        # Initial confidence
        current_means, _ = self.predict_basin_properties()
        current_params = {
            param: np.mean(current_means.get(param, 0)) 
            for param in self.param_names
        }
        current_npv = self.calculate_npv(current_params)
        
        # Monte Carlo simulation for initial confidence
        n_samples = 1000
        profitable_count = 0
        
        # Sample from current uncertainty
        _, uncertainties = self.predict_basin_properties()
        for i in range(n_samples):
            # Sample parameters
            sampled_params = {}
            for param in self.param_names:
                if param in current_means:
                    mean = np.mean(current_means.get(param, 0))
                    std = np.mean(uncertainties.get(param, 0))
                    
                    # Sample with constraints
                    if param in ['porosity', 'saturation']:
                        # Beta distribution for values between 0 and 1
                        # Convert mean/std to alpha/beta
                        mean_adj = max(0.01, min(0.99, mean))
                        std_adj = min(std, np.sqrt(mean_adj * (1 - mean_adj)))
                        
                        # Calculate alpha and beta
                        v = (mean_adj * (1 - mean_adj)) / (std_adj**2) - 1
                        alpha = mean_adj * v
                        beta = (1 - mean_adj) * v
                        
                        # Ensure valid parameters
                        alpha = max(0.5, alpha)
                        beta = max(0.5, beta)
                        
                        # Sample
                        sampled_params[param] = np.random.beta(alpha, beta)
                    
                    elif param == 'permeability':
                        # Log-normal for permeability
                        mu = np.log(mean)
                        sigma = std / mean
                        sampled_params[param] = np.random.lognormal(mu, sigma)
                    
                    else:
                        # Normal distribution for other parameters
                        # with constraints for physical values
                        sampled_params[param] = np.random.normal(mean, std)
                        
                        if param == 'thickness':
                            sampled_params[param] = max(1.0, sampled_params[param])
                        elif param == 'depth':
                            sampled_params[param] = max(100, sampled_params[param])
            
            # Calculate NPV with sampled parameters
            npv = self.calculate_npv(sampled_params)
            if npv > 0:
                profitable_count += 1
        
        # Initial confidence
        confidence = profitable_count / n_samples
        confidences.append(confidence)
        npvs.append(current_npv)
        
        print(f"Initial confidence: {confidence:.1%}, NPV: ${current_npv/1e6:.1f}M")
        
        # Drill wells until confidence threshold is reached
        while confidence < confidence_threshold and wells_drilled < max_wells:
            # Find optimal next well location
            next_location, next_evsi = self.optimize_next_well()
            
            if next_evsi <= 0:
                print(f"No profitable well locations found after {wells_drilled} wells")
                break
            
            # Drill well
            measurements = self.drill_well(next_location)
            wells_drilled += 1
            
            # Calculate new confidence
            current_means, _ = self.predict_basin_properties()
            current_params = {
                param: np.mean(current_means.get(param, 0)) 
                for param in self.param_names
            }
            current_npv = self.calculate_npv(current_params)
            
            # Monte Carlo for confidence
            profitable_count = 0
            _, uncertainties = self.predict_basin_properties()
            
            for i in range(n_samples):
                # Sample parameters
                sampled_params = {}
                for param in self.param_names:
                    if param in current_means:
                        mean = np.mean(current_means.get(param, 0))
                        std = np.mean(uncertainties.get(param, 0))
                        
                        # Sample with constraints (as above)
                        if param in ['porosity', 'saturation']:
                            mean_adj = max(0.01, min(0.99, mean))
                            std_adj = min(std, np.sqrt(mean_adj * (1 - mean_adj)))
                            v = (mean_adj * (1 - mean_adj)) / (std_adj**2) - 1
                            alpha = max(0.5, mean_adj * v)
                            beta = max(0.5, (1 - mean_adj) * v)
                            sampled_params[param] = np.random.beta(alpha, beta)
                        elif param == 'permeability':
                            mu = np.log(mean)
                            sigma = std / mean
                            sampled_params[param] = np.random.lognormal(mu, sigma)
                        else:
                            sampled_params[param] = np.random.normal(mean, std)
                            
                            if param == 'thickness':
                                sampled_params[param] = max(1.0, sampled_params[param])
                            elif param == 'depth':
                                sampled_params[param] = max(100, sampled_params[param])
                
                # Calculate NPV with sampled parameters
                npv = self.calculate_npv(sampled_params)
                if npv > 0:
                    profitable_count += 1
            
            # Update confidence
            confidence = profitable_count / n_samples
            confidences.append(confidence)
            npvs.append(current_npv)
            
            exploration_path.append({
                'location': next_location,
                'measurements': measurements,
                'evsi': next_evsi,
                'confidence': confidence,
                'npv': current_npv
            })
            
            print(f"Well {wells_drilled}: Confidence = {confidence:.1%}, " + 
                f"NPV = ${current_npv/1e6:.1f}M, EVSI = ${next_evsi/1e6:.1f}M")
            
            # Check if confidence threshold reached
            if confidence >= confidence_threshold:
                print(f"Confidence threshold {confidence_threshold:.0%} reached " + 
                    f"after {wells_drilled} wells")
                break
        
        # Restore original state if there was one
        if original_locations:
            self.well_locations = copy.deepcopy(original_locations)
            self.well_measurements = copy.deepcopy(original_measurements)
            self.well_logs = copy.deepcopy(original_logs)
            self._update_model()
        
        return {
            'wells_needed': wells_drilled,
            'final_confidence': confidence,
            'exploration_path': exploration_path,
            'confidence_history': confidences,
            'npv_history': npvs,
            'reached_threshold': confidence >= confidence_threshold
        }

    def visualize_basin(self, show_true=False, show_wells=True, show_uncertainty=True):
        """
        Visualize the basin with current knowledge
        
        Parameters:
        -----------
        show_true : bool
            Whether to show the true basin properties (only available in simulation)
        show_wells : bool
            Whether to show well locations
        show_uncertainty : bool
            Whether to show uncertainty maps
        """
        # Get current predictions
        predictions, uncertainties = self.predict_basin_properties()
        
        # Create figure
        n_params = len(self.param_names)
        n_rows = 2 if show_uncertainty else 1
        fig, axes = plt.subplots(n_rows, n_params, figsize=(n_params*4, n_rows*4))
        
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        
        for i, param in enumerate(self.param_names):
            # Customize color maps and ranges for each parameter
            if param == 'depth':
                cmap = 'terrain'
                vmin = self.prior_means[param] * 0.8
                vmax = self.prior_means[param] * 1.2
            elif param in ['porosity', 'saturation']:
                cmap = 'viridis'
                vmin = 0
                vmax = 0.4 if param == 'porosity' else 0.8
            elif param == 'permeability':
                cmap = 'plasma'
                vmin = 0
                vmax = 200
            else:  # thickness
                cmap = 'cividis'
                vmin = 5
                vmax = 40
            
            # Plot predicted means
            if param in predictions:
                im1 = axes[0, i].pcolormesh(self.X_grid, self.Y_grid, 
                                            predictions[param],
                                            cmap=cmap, vmin=vmin, vmax=vmax,
                                            shading='auto')
                plt.colorbar(im1, ax=axes[0, i])
                axes[0, i].set_title(f'{param.capitalize()} - Predicted')
                
                # Add true values as contour if requested
                if show_true and param in self.true_values:
                    contour = axes[0, i].contour(self.X_grid, self.Y_grid,
                                            self.true_values[param],
                                            colors='k', alpha=0.5, linewidths=0.5)
                    axes[0, i].clabel(contour, inline=True, fontsize=8)
            
            # Plot uncertainty
            if show_uncertainty and param in uncertainties:
                # Normalize uncertainty for visualization
                norm_uncertainty = uncertainties[param] / self.prior_means[param]
                im2 = axes[1, i].pcolormesh(self.X_grid, self.Y_grid,
                                        norm_uncertainty,
                                        cmap='hot', shading='auto')
                plt.colorbar(im2, ax=axes[1, i])
                axes[1, i].set_title(f'{param.capitalize()} - Uncertainty (%)')
            
            # Show well locations
            if show_wells and self.well_locations:
                well_x, well_y = zip(*self.well_locations)
                axes[0, i].scatter(well_x, well_y, c='white', edgecolor='black', s=80, marker='x')
                if show_uncertainty:
                    axes[1, i].scatter(well_x, well_y, c='white', edgecolor='black', s=80, marker='x')
        
        plt.tight_layout()
        plt.show()
        
        # Plot NPV map
        if len(self.well_locations) > 0:  # Only if we have some data
            fig_npv, ax_npv = plt.subplots(figsize=(10, 8))
            
            # Calculate NPV at each grid point
            npv_map = np.zeros_like(self.X_grid)
            
            for i in range(self.X_grid.shape[0]):
                for j in range(self.X_grid.shape[1]):
                    # Parameters at this location
                    params = {}
                    for param in self.param_names:
                        if param in predictions:
                            params[param] = predictions[param][i, j]
                        else:
                            params[param] = self.prior_means[param]
                    
                    # Calculate NPV
                    npv_map[i, j] = self.calculate_npv(params)
            
            # Plot NPV
            vmax = max(abs(np.min(npv_map)), abs(np.max(npv_map)))
            im_npv = ax_npv.pcolormesh(self.X_grid, self.Y_grid, npv_map/1e6,  # In millions
                                    cmap='RdBu_r', shading='auto',
                                    vmin=-vmax/1e6, vmax=vmax/1e6)
            plt.colorbar(im_npv, ax=ax_npv, label='NPV ($ Million)')
            ax_npv.set_title('Expected Net Present Value')
            
            # Show well locations
            if show_wells and self.well_locations:
                well_x, well_y = zip(*self.well_locations)
                ax_npv.scatter(well_x, well_y, c='white', edgecolor='black', s=80, marker='x')
            
            plt.tight_layout()
            plt.show()
            
            # Visualize value of information
            if show_uncertainty:
                fig_voi, ax_voi = plt.subplots(figsize=(10, 8))
                
                # Create simple proxy for VOI - uncertainty weighted by economic impact
                voi_map = np.zeros_like(self.X_grid)
                
                if 'porosity' in uncertainties:
                    voi_map += 10 * uncertainties['porosity']
                if 'permeability' in uncertainties:
                    voi_map += 0.1 * uncertainties['permeability']
                if 'thickness' in uncertainties:
                    voi_map += 1 * uncertainties['thickness']
                if 'saturation' in uncertainties:
                    voi_map += 5 * uncertainties['saturation']
                
                # Scale by expected NPV
                positive_npv = np.maximum(0, npv_map)
                voi_map = voi_map * np.sqrt(positive_npv) / 1e6  # Scale by sqrt(NPV)
                
                # Mask out areas close to existing wells
                if self.well_locations:
                    for x, y in self.well_locations:
                        for i in range(self.X_grid.shape[0]):
                            for j in range(self.X_grid.shape[1]):
                                distance = np.sqrt((self.X_grid[i, j] - x)**2 + 
                                                (self.Y_grid[i, j] - y)**2)
                                if distance < self.grid_resolution * 2:
                                    voi_map[i, j] = 0
                
                im_voi = ax_voi.pcolormesh(self.X_grid, self.Y_grid, voi_map,
                                        cmap='viridis', shading='auto')
                plt.colorbar(im_voi, ax=ax_voi, label='Value of Information Proxy')
                ax_voi.set_title('Potential Value of Information (Next Well Targets)')
                
                # Show well locations
                if show_wells and self.well_locations:
                    well_x, well_y = zip(*self.well_locations)
                    ax_voi.scatter(well_x, well_y, c='white', edgecolor='black', s=80, marker='x')
                
                plt.tight_layout()
                plt.show()

    def visualize_exploration_results(self, results):
        """
        Visualize the results of exploration
        
        Parameters:
        -----------
        results : dict
            Results from determine_wells_needed method
        """
        confidences = results['confidence_history']
        npvs = results['npv_history']
        wells = range(len(confidences))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot confidence evolution
        ax1.plot(wells, confidences, 'bo-', linewidth=2)
        ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, 
                label='90% Confidence Threshold')
        ax1.set_xlabel('Number of Wells')
        ax1.set_ylabel('Probability of Profitability')
        ax1.set_title('Confidence Evolution')
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot NPV evolution
        ax2.plot(wells, [n/1e6 for n in npvs], 'go-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7,
                label='Break-even')
        ax2.set_xlabel('Number of Wells')
        ax2.set_ylabel('Expected NPV ($ Million)')
        ax2.set_title('NPV Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Exploration path visualization if we have coordinates
        if 'exploration_path' in results and results['exploration_path']:
            # Extract well locations
            x_coords = []
            y_coords = []
            evsies = []
            
            for well_info in results['exploration_path']:
                loc = well_info['location']
                x_coords.append(loc[0])
                y_coords.append(loc[1])
                evsies.append(well_info['evsi'] / 1e6)  # In millions
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot exploration path with arrows
            for i in range(len(x_coords)-1):
                ax.annotate('',
                        xy=(x_coords[i+1], y_coords[i+1]),
                        xytext=(x_coords[i], y_coords[i]),
                        arrowprops=dict(arrowstyle="->", lw=1.5,
                                        color='blue', alpha=0.6))
            
            # Plot wells with EVSI as size and color
            scatter = ax.scatter(x_coords, y_coords, 
                            s=[max(50, e*50) for e in evsies],
                            c=evsies, cmap='viridis',
                            alpha=0.7, edgecolor='k')
            
            # Label well numbers
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                ax.text(x, y, str(i+1), fontsize=9, ha='center', va='center')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, label='EVSI ($ Million)')
            
            ax.set_title('Exploration Path')
            ax.set_xlabel('X Coordinate (m)')
            ax.set_ylabel('Y Coordinate (m)')
            ax.grid(True, alpha=0.3)
            
            # Set axis limits to basin size
            ax.set_xlim(0, self.basin_size[0])
            ax.set_ylim(0, self.basin_size[1])
            
            plt.tight_layout()
            plt.show()

from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

def create_realistic_kernel(param, length_scale):
    """Create realistic kernels for geological parameters"""
    if param == 'depth':
        # Depth tends to have smooth, large-scale trends
        return C(1.0) * Matern(length_scale=length_scale, nu=2.5)
    
    elif param in ['porosity', 'saturation']:
        # These often have both regional trends and local variations
        return C(1.0) * Matern(length_scale=length_scale, nu=1.5) + \
               C(0.3) * Matern(length_scale=length_scale*0.3, nu=0.5) + \
               WhiteKernel(noise_level=0.001)
               
    elif param == 'permeability':
        # Permeability can vary dramatically over short distances
        # Often log-normally distributed
        return C(1.0) * Matern(length_scale=length_scale*0.7, nu=0.5) + \
               WhiteKernel(noise_level=0.01)
               
    elif param == 'thickness':
        # Thickness often has smoother variations
        return C(1.0) * Matern(length_scale=length_scale, nu=1.5) + \
               WhiteKernel(noise_level=0.001)
    
    # Default
    return C(1.0) * Matern(length_scale=length_scale, nu=1.5) + \
           WhiteKernel(noise_level=0.001)


def simulate_well_measurements(true_values, measurement_biases=None):
    """
    Simulate realistic well measurements with biases and uncertainties
    
    Parameters:
    -----------
    true_values : dict
        Dictionary of true parameter values
    measurement_biases : dict, optional
        Dictionary of measurement biases
        
    Returns:
    --------
    measurements : dict
        Dictionary of measured values
    """
    if measurement_biases is None:
        measurement_biases = {param: 0.0 for param in true_values}
    
    measurements = {}
    
    # Depth - typically measured with wireline tools
    if 'depth' in true_values:
        # Slight depth bias (e.g. due to wireline stretching)
        bias = measurement_biases.get('depth', 0.0)
        noise = np.random.normal(0, 20)  # ~20m uncertainty
        measurements['depth'] = true_values['depth'] + bias + noise
    
    # Porosity - measured with logs or core analysis
    if 'porosity' in true_values:
        true_porosity = true_values['porosity']
        bias = measurement_biases.get('porosity', 0.0)
        
        # Porosity measurement error increases with porosity
        noise_scale = 0.01 + 0.1 * true_porosity
        noise = np.random.normal(0, noise_scale)
        
        # Log measurements often underestimate porosity at high values
        if true_porosity > 0.2:
            bias -= 0.02 * (true_porosity - 0.2)
            
        measurements['porosity'] = true_porosity + bias + noise
        
        # Ensure physical constraints
        measurements['porosity'] = max(0.0, min(1.0, measurements['porosity']))
    
    # Permeability - typically has large measurement uncertainty
    if 'permeability' in true_values:
        true_perm = true_values['permeability']
        bias = measurement_biases.get('permeability', 0.0)
        
        # Permeability measurements often have log-normal errors
        # Larger errors at higher permeability values
        log_noise = np.random.normal(0, 0.4)  # 0.4 log units ~ factor of 2.5
        
        # Measurement bias can increase with value
        rel_bias = bias * (1 + 0.2 * np.log10(max(1.0, true_perm)))
        
        # Apply in log space
        measurements['permeability'] = true_perm * np.exp(log_noise) * (1 + rel_bias)
        
        # Ensure physical constraints
        measurements['permeability'] = max(0.01, measurements['permeability'])
    
    # Thickness - measured from logs
    if 'thickness' in true_values:
        true_thickness = true_values['thickness']
        bias = measurement_biases.get('thickness', 0.0)
        
        # Thickness resolution depends on log sampling rate
        noise = np.random.normal(0, 0.5)  # ~0.5m for high resolution logs
        
        # Potential interpretive bias at layer boundaries
        if true_thickness < 5:
            bias += 1.0  # Thin beds often overestimated
        
        measurements['thickness'] = true_thickness + bias + noise
        
        # Ensure physical constraints
        measurements['thickness'] = max(0.1, measurements['thickness'])
    
    # Saturation - complex measurements with biases
    if 'saturation' in true_values:
        true_sat = true_values['saturation']
        bias = measurement_biases.get('saturation', 0.0)
        
        # Saturation noise depends on value (harder to measure low saturations)
        if true_sat < 0.3:
            noise_scale = 0.08
        else:
            noise_scale = 0.04
            
        noise = np.random.normal(0, noise_scale)
        
        # Saturation often underestimated in low porosity rocks
        if 'porosity' in true_values and true_values['porosity'] < 0.1:
            bias -= 0.05
        
        measurements['saturation'] = true_sat + bias + noise
        
        # Ensure physical constraints
        measurements['saturation'] = max(0.0, min(1.0, measurements['saturation']))
    
    return measurements



def simulate_well_measurements_old(true_values, measurement_biases=None):
    """
    Simulate realistic well measurements with biases and uncertainties
    
    Parameters:
    -----------
    true_values : dict
        Dictionary of true parameter values
    measurement_biases : dict, optional
        Dictionary of measurement biases
        
    Returns:
    --------
    measurements : dict
        Dictionary of measured values
    """
    if measurement_biases is None:
        measurement_biases = {param: 0.0 for param in true_values}
    
    measurements = {}
    
    # Depth - typically measured with wireline tools
    if 'depth' in true_values:
        # Slight depth bias (e.g. due to wireline stretching)
        bias = measurement_biases.get('depth', 0.0)
        noise = np.random.normal(0, 20)  # ~20m uncertainty
        measurements['depth'] = true_values['depth'] + bias + noise
    
    # Porosity - measured with logs or core analysis
    if 'porosity' in true_values:
        true_porosity = true_values['porosity']
        bias = measurement_biases.get('porosity', 0.0)
        
        # Porosity measurement error increases with porosity
        noise_scale = 0.01 + 0.1 * true_porosity
        noise = np.random.normal(0, noise_scale)
        
        # Log measurements often underestimate porosity at high values
        if true_porosity > 0.2:
            bias -= 0.02 * (true_porosity - 0.2)
            
        measurements['porosity'] = true_porosity + bias + noise
        
        # Ensure physical constraints
        measurements['porosity'] = max(0.0, min(1.0, measurements['porosity']))
    
    # Permeability - typically has large measurement uncertainty
    if 'permeability' in true_values:
        true_perm = true_values['permeability']
        bias = measurement_biases.get('permeability', 0.0)
        
        # Permeability measurements often have log-normal errors
        # Larger errors at higher permeability values
        log_noise = np.random.normal(0, 0.4)  # 0.4 log units ~ factor of 2.5
        
        # Measurement bias can increase with value
        rel_bias = bias * (1 + 0.2 * np.log10(max(1.0, true_perm)))
        
        # Apply in log space
        measurements['permeability'] = true_perm * np.exp(log_noise) * (1 + rel_bias)
        
        # Ensure physical constraints
        measurements['permeability'] = max(0.01, measurements['permeability'])
    
    # Thickness - measured from logs
    if 'thickness' in true_values:
        true_thickness = true_values['thickness']
        bias = measurement_biases.get('thickness', 0.0)
        
        # Thickness resolution depends on log sampling rate
        noise = np.random.normal(0, 0.5)  # ~0.5m for high resolution logs
        
        # Potential interpretive bias at layer boundaries
        if true_thickness < 5:
            bias += 1.0  # Thin beds often overestimated
        
        measurements['thickness'] = true_thickness + bias + noise
        
        # Ensure physical constraints
        measurements['thickness'] = max(0.1, measurements['thickness'])
    
    # Saturation - complex measurements with biases
    if 'saturation' in true_values:
        true_sat = true_values['saturation']
        bias = measurement_biases.get('saturation', 0.0)
        
        # Saturation noise depends on value (harder to measure low saturations)
        if true_sat < 0.3:
            noise_scale = 0.08
        else:
            noise_scale = 0.04
            
        noise = np.random.normal(0, noise_scale)
        
        # Saturation often underestimated in low porosity rocks
        if 'porosity' in true_values and true_values['porosity'] < 0.1:
            bias -= 0.05
        
        measurements['saturation'] = true_sat + bias + noise
        
        # Ensure physical constraints
        measurements['saturation'] = max(0.0, min(1.0, measurements['saturation']))
    
    return measurements

def simulate_full_well_log_old(true_values, well_trajectory, n_samples=50):
    """
    Simulate a full well log with multiple measurements
    
    Parameters:
    -----------
    true_values : dict of functions
        Functions that map (x, y, z) -> parameter value
    well_trajectory : np.array
        3D well path coordinates, shape (n_points, 3)
    n_samples : int
        Number of log samples to generate
        
    Returns:
    --------
    log_data : dict
        Dictionary with arrays of depth and parameter values
    """
    # Sample points along the well path
    indices = np.linspace(0, len(well_trajectory)-1, n_samples).astype(int)
    sample_points = well_trajectory[indices]
    
    log_data = {
        'measured_depth': np.array([p[2] for p in sample_points])
    }
    
    # Generate true values at each sample point
    true_log_values = {}
    for param, value_func in true_values.items():
        true_log_values[param] = np.array([
            value_func(p[0], p[1], p[2]) for p in sample_points
        ])
    
    # Add measurement noise and biases to each log point
    measured_logs = {}
    for i in range(n_samples):
        # Extract true values at this depth
        point_true_values = {
            param: true_log_values[param][i] for param in true_values
        }
        
        # Simulate measurements
        point_measurements = simulate_well_measurements(point_true_values)
        
        # Store in measured logs
        for param, value in point_measurements.items():
            if param not in measured_logs:
                measured_logs[param] = np.zeros(n_samples)
            measured_logs[param][i] = value
    
    # Combine with log data
    log_data.update(measured_logs)
    
    return log_data

def simulate_full_well_log(true_values, well_trajectory, n_samples=50):
    """
    Simulate a full well log with multiple measurements
    
    Parameters:
    -----------
    true_values : dict of functions
        Functions that map (x, y, z) -> parameter value
    well_trajectory : np.array
        3D well path coordinates, shape (n_points, 3)
    n_samples : int
        Number of log samples to generate
        
    Returns:
    --------
    log_data : dict
        Dictionary with arrays of depth and parameter values
    """
    # Sample points along the well path
    indices = np.linspace(0, len(well_trajectory)-1, n_samples).astype(int)
    sample_points = well_trajectory[indices]
    
    log_data = {
        'measured_depth': np.array([p[2] for p in sample_points])
    }
    
    # Generate true values at each sample point
    true_log_values = {}
    for param, value_func in true_values.items():
        true_log_values[param] = np.array([
            value_func(p[0], p[1], p[2]) for p in sample_points
        ])
    
    # Add measurement noise and biases to each log point
    measured_logs = {}
    for i in range(n_samples):
        # Extract true values at this depth
        point_true_values = {
            param: true_log_values[param][i] for param in true_values
        }
        
        # Simulate measurements
        point_measurements = simulate_well_measurements(point_true_values)
        
        # Store in measured logs
        for param, value in point_measurements.items():
            if param not in measured_logs:
                measured_logs[param] = np.zeros(n_samples)
            measured_logs[param][i] = value
    
    # Combine with log data
    log_data.update(measured_logs)
    
    return log_data

class MultiOutputGP:
    """
    Multi-output Gaussian Process that captures correlations between parameters
    """
    def __init__(self, param_names, length_scales, cross_correlation_matrix):
        """
        Initialize multi-output GP model
        
        Parameters:
        -----------
        param_names : list
            Names of parameters to model
        length_scales : dict
            Spatial correlation length scales for each parameter
        cross_correlation_matrix : np.array
            Matrix defining correlations between parameters
        """
        self.param_names = param_names
        self.n_params = len(param_names)
        self.length_scales = length_scales
        self.cross_corr = cross_correlation_matrix
        
        # Individual GPs for each parameter
        self.gp_models = {}
        for param in param_names:
            kernel = create_realistic_kernel(param, length_scales[param])
            self.gp_models[param] = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=3,
                alpha=1e-10  # Small noise term for numerical stability
            )
        
        # Storage for training data
        self.X_train = None
        self.Y_train = {}
        self.Y_transformed = {}
    
    def fit(self, X, Y_dict):
        """
        Fit the model to training data
        
        Parameters:
        -----------
        X : np.array
            Well locations, shape (n_wells, 2)
        Y_dict : dict
            Dictionary of parameter values, each with shape (n_wells,)
        """
        self.X_train = X.copy()
        self.Y_train = Y_dict.copy()
        
        # Apply cross-correlation transformation
        self._transform_outputs()
        
        # Fit individual GPs to transformed outputs
        for i, param in enumerate(self.param_names):
            if param in self.Y_transformed and len(self.Y_transformed[param]) > 1:
                self.gp_models[param].fit(X, self.Y_transformed[param])
    
    def _transform_outputs(self):
        """
        Apply linear transformation to capture parameter correlations
        """
        # Collect all parameter values into matrix
        Y_matrix = np.zeros((len(self.X_train), self.n_params))
        for i, param in enumerate(self.param_names):
            if param in self.Y_train and len(self.Y_train[param]) > 0:
                Y_matrix[:, i] = self.Y_train[param]
        
        # Compute Cholesky decomposition of cross-correlation matrix
        L = np.linalg.cholesky(self.cross_corr)
        
        # Transform outputs
        Y_transformed_matrix = Y_matrix @ L
        
        # Store transformed outputs
        self.Y_transformed = {}
        for i, param in enumerate(self.param_names):
            self.Y_transformed[param] = Y_transformed_matrix[:, i]
    
    def predict(self, X_new):
        """
        Predict parameters at new locations
        
        Parameters:
        -----------
        X_new : np.array
            New locations, shape (n_points, 2)
            
        Returns:
        --------
        means : dict
            Mean predictions for each parameter
        uncertainties : dict
            Standard deviations for each parameter
        """
        # Predictions from individual GPs
        transformed_means = {}
        transformed_vars = {}
        
        for param in self.param_names:
            if param in self.Y_transformed and len(self.Y_transformed[param]) > 1:
                mean, std = self.gp_models[param].predict(X_new, return_std=True)
                transformed_means[param] = mean
                transformed_vars[param] = std**2
            else:
                # Not enough data, use prior
                transformed_means[param] = np.zeros(len(X_new))
                transformed_vars[param] = np.ones(len(X_new))
        
        # Inverse transform to get original parameters
        means = {}
        uncertainties = {}
        
        # If we have all parameters, do proper inverse transform
        if all(param in transformed_means for param in self.param_names):
            # Create matrix of transformed means
            transformed_mean_matrix = np.zeros((len(X_new), self.n_params))
            transformed_var_matrix = np.zeros((len(X_new), self.n_params))
            
            for i, param in enumerate(self.param_names):
                transformed_mean_matrix[:, i] = transformed_means[param]
                transformed_var_matrix[:, i] = transformed_vars[param]
            
            # Inverse transform
            L = np.linalg.cholesky(self.cross_corr)
            L_inv = np.linalg.inv(L)
            
            mean_matrix = transformed_mean_matrix @ L_inv
            
            # Covariance transformation is more complex
            # This is simplified - in reality we'd need to account for parameter correlations
            var_matrix = transformed_var_matrix @ (L_inv**2)
            
            # Store results
            for i, param in enumerate(self.param_names):
                means[param] = mean_matrix[:, i]
                uncertainties[param] = np.sqrt(var_matrix[:, i])
        else:
            # Not all parameters have data, use simple version
            for param in self.param_names:
                if param in transformed_means:
                    means[param] = transformed_means[param]
                    uncertainties[param] = np.sqrt(transformed_vars[param])
                else:
                    means[param] = np.zeros(len(X_new))
                    uncertainties[param] = np.ones(len(X_new))
        
        return means, uncertainties





def run_exploration_workflow():

    """Run the complete basin exploration workflow"""

    # Create BEL model
    print("Initializing Bayesian Evidential Learning model...")

    bel = BayesianEvidentialLearning(
        basin_size=(5000, 5000),  # Smaller basin for faster demonstration
        grid_resolution=250       # Coarser grid for speed
    )
    
    # Visualize true basin (only available in simulation)
    print("Visualizing true basin properties...")
    bel.visualize_basin(show_true=True, show_wells=False)
    
    # Determine wells needed to reach 90% confidence
    print("Determining number of wells needed for 90% confidence...")
    results = bel.determine_wells_needed(confidence_threshold=0.9, max_wells=8)
    
    # Visualize exploration results
    print("Visualizing exploration results...")
    bel.visualize_exploration_results(results)
    
    # Final basin state
    print("Visualizing final basin knowledge...")
    bel.visualize_basin(show_true=True, show_wells=True)
    
    return bel, results

# Run the workflow
if __name__ == "__main__":

    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the exploration workflow
    bel, results = run_exploration_workflow()
    
    # Print summary
    print("\nExploration Summary:")
    print(f"Wells needed for 90% confidence: {results['wells_needed']}")
    print(f"Final confidence: {results['final_confidence']:.1%}")
    print(f"Reached threshold: {results['reached_threshold']}")
    
    # Calculate economics
    final_cost = results['wells_needed'] * bel.economic_params['well_cost']
    print(f"Total exploration cost: ${final_cost/1e6:.1f} million")
    
    if results['npv_history']:
        final_npv = results['npv_history'][-1]
        print(f"Expected NPV: ${final_npv/1e6:.1f} million")
        print(f"Return on investment: {(final_npv / final_cost):.1f}x")