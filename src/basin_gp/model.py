import torch
import gpytorch
import numpy as np

from .data import prepare_training_data
from .planning import plan_next_well

class BasinExplorationGP:
    """
    Comprehensive framework for basin exploration using multi-output GPs.
    """
    
    def __init__(self, basin_size=(10, 10), properties=['porosity', 'permeability', 'thickness']):
        """
        Initialize the Basin Exploration GP framework.
        
        Args:
            basin_size: Size of the basin in (x, y) kilometers
            properties: List of geological properties to model
        """
        self.basin_size = basin_size
        self.properties = properties
        self.n_properties = len(properties)
        
        # Model components will be initialized later
        self.model = None
        self.likelihood = None
        
        # Data storage
        self.wells = []  # List of well data
        
        # History tracking
        self.exploration_history = []
    

    def add_well(self, location, measurements, well_name=None):
        """
        Add a well to the dataset.
        
        Args:
            location: (x, y) coordinates of the well
            measurements: Dictionary mapping property names to measured values
            well_name: Optional name for the well
            
        Returns:
            The well data that was added
        """
        if well_name is None:
            well_name = f"Well_{len(self.wells) + 1}"
            
        # Convert measurements to consistent format
        property_values = []
        measurement_mask = []
        
        for prop in self.properties:
            if prop in measurements:
                property_values.append(measurements[prop])
                measurement_mask.append(True)
            else:
                property_values.append(float('nan'))
                measurement_mask.append(False)
        
        well_data = {
            'name': well_name,
            'location': np.array(location),  # Ensure location is a NumPy array
            'measurements': np.array(property_values),
            'mask': np.array(measurement_mask),
            'date_added': len(self.wells)
        }
        
        self.wells.append(well_data)
        return well_data
    
    def _create_model(self, X, Y):
        """Create the multi-output GP model."""
        # Define model class internally
        class MultitaskGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, n_properties):
                super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
                
                # Mean module
                self.mean_module = gpytorch.means.MultitaskMean(
                    gpytorch.means.ConstantMean(), num_tasks=n_properties
                )
                
                # Base kernel
                self.base_covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=1.5)
                )
                
                # Multi-task kernel
                self.covar_module = gpytorch.kernels.MultitaskKernel(
                    self.base_covar_module, num_tasks=n_properties, rank=n_properties-1
                )
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        
        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_properties)
        self.model = MultitaskGPModel(X, Y, self.likelihood, self.n_properties)
    
    def fit(self, learning_rate=0.01, iterations=500, verbose=True):
        """
        Fit the GP model to the well data.
        
        Args:
            learning_rate: Learning rate for optimization
            iterations: Number of training iterations
            verbose: Whether to print training progress
            
        Returns:
            Loss value at the end of training
        """
        # Prepare training data
        X, Y, mask = prepare_training_data(self.wells, self.n_properties)
        
        # Create model
        self._create_model(X, Y)
        
        # Training
        self.model.train()
        self.likelihood.train()
        
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Custom loss function to handle missing data
        def masked_loss(output, target, mask):
            # Convert mask to float tensor
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            
            # Mean of the output distribution
            mean = output.mean
            
            # Calculate negative log likelihood only for observed values
            masked_diff = (mean - target) * mask_tensor
            masked_diff_sq = masked_diff ** 2
            
            # Get task noise from likelihood
            task_noises = self.likelihood.task_noises.reshape(-1)
            
            # Calculate loss term for each task
            loss_terms = []
            for i in range(self.n_properties):
                # Only include terms for observed values
                observed = mask[:, i]
                observed_tensor = torch.tensor(observed, dtype=torch.bool)
                if torch.any(observed_tensor):
                    # Extract relevant parts for this task
                    observed_indices = torch.where(observed_tensor)[0]
                    y_observed = target[observed_indices, i]
                    mean_observed = mean[observed_indices, i]
                    
                    # Negative log likelihood for Gaussian
                    task_loss = 0.5 * torch.sum((y_observed - mean_observed)**2 / task_noises[i])
                    task_loss += 0.5 * torch.sum(torch.log(2 * np.pi * task_noises[i])) * observed_tensor.sum()
                    
                    loss_terms.append(task_loss)
            
            return sum(loss_terms) if loss_terms else torch.tensor(0.0, requires_grad=True)
        
        # Training loop
        final_loss = 0.0
        for i in range(iterations):
            optimizer.zero_grad()
            output = self.model(X)
            loss = masked_loss(output, Y, mask)
            loss.backward()
            optimizer.step()
            
            if verbose and (i % 100 == 0 or i == iterations - 1):
                print(f'Iteration {i+1}/{iterations} - Loss: {loss.item():.4f}')
            
            final_loss = loss.item()
        
        # Set model to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        return final_loss
    
    def predict(self, grid):
        """
        Make predictions across a grid of locations.
        
        Args:
            grid: Grid of points to predict at [n_points, 2]
            
        Returns:
            mean: Predicted means [n_points, n_properties]
            std: Predicted standard deviations [n_points, n_properties]
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        grid = torch.tensor(grid, dtype=torch.float32) if not isinstance(grid, torch.Tensor) else grid
        
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(grid))
            mean = predictions.mean
            variance = predictions.variance
            std = torch.sqrt(variance)
        
        return mean, std
        
    def plan_next_well(self, grid, strategy='uncertainty', economic_params=None):
        """
        Plan the next exploration well.
        
        Args:
            grid: Grid of candidate locations [n_points, 2]
            strategy: Strategy for well selection:
                - 'uncertainty': Maximum total uncertainty
                - 'ei': Expected improvement for property 0
                - 'economic': Maximum expected economic value
            economic_params: Economic parameters for value calculation
                
        Returns:
            best_location: Best location for next well [2]
            score: Score at the best location
            score_grid: Score across the entire grid
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        grid = torch.tensor(grid, dtype=torch.float32) if not isinstance(grid, torch.Tensor) else grid
        
        # Get predictions
        mean, std = self.predict(grid)
        
        # Delegate to strategy-specific planning function
        return plan_next_well(self, grid, mean, std, strategy, economic_params)
    
    def sequential_exploration(self, grid, n_wells, true_functions, noise_std=0.01, 
                              strategy='uncertainty', economic_params=None, plot=False):
        """
        Sequentially plan and drill exploration wells.
        
        Args:
            grid: Grid of candidate locations [n_points, 2]
            n_wells: Number of wells to drill
            true_functions: List of functions that return true property values
            noise_std: Measurement noise standard deviation
            strategy: Well planning strategy
            economic_params: Economic parameters for 'economic' strategy
            plot: Whether to plot results
            
        Returns:
            history: List of exploration steps
        """
        if not callable(true_functions[0]):
            raise ValueError("true_functions should be a list of callable functions")
        
        grid = torch.tensor(grid, dtype=torch.float32) if not isinstance(grid, torch.Tensor) else grid
        
        # Record exploration history
        history = []
        
        for i in range(n_wells):
            # Fit model to current data
            self.fit(verbose=(i==0))
            
            # Plan next well
            next_location, score, score_grid = self.plan_next_well(
                grid, strategy=strategy, economic_params=economic_params
            )
            
            # "Drill" the well (evaluate true functions with noise)
            measurements = {}
            for j, func in enumerate(true_functions):
                if callable(func):
                    value = func(next_location.reshape(1, -1))
                    if isinstance(value, torch.Tensor):
                        value = value.item() + np.random.normal(0, noise_std)
                    else:
                        value = value + np.random.normal(0, noise_std)
                    measurements[self.properties[j]] = value
            
            # Add the well to our dataset
            self.add_well(next_location.numpy(), measurements, well_name=f"Well_{len(self.wells) + 1}")
            
            # Record this step
            step_info = {
                'well_location': next_location.numpy(),
                'measurements': measurements,
                'score': score.item(),
                'strategy': strategy
            }
            history.append(step_info)
            
        return history