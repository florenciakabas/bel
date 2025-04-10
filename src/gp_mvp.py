import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
        # For the add_well function:
    

    def add_well(self, location, measurements, well_name=None):
        """
        Add a well to the dataset.
        
        Args:
            location: (x, y) coordinates of the well
            measurements: Dictionary mapping property names to measured values
            well_name: Optional name for the well
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

    def _prepare_training_data(self):
        """
        Prepare training data from wells.
        
        Returns:
            X: Well locations [n_wells, 2]
            Y: Property measurements [n_wells, n_properties]
            mask: Boolean mask for valid measurements [n_wells, n_properties]
        """
        n_wells = len(self.wells)
        
        X = np.zeros((n_wells, 2))
        Y = np.zeros((n_wells, self.n_properties))
        mask = np.zeros((n_wells, self.n_properties), dtype=bool)
        
        for i, well in enumerate(self.wells):
            X[i] = well['location']
            Y[i] = well['measurements']
            mask[i] = well['mask']
            
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        
        # Handle missing values (replace NaNs with dummy values that will be masked)
        Y_tensor[torch.isnan(Y_tensor)] = 0.0
            
        return X_tensor, Y_tensor, mask
    
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
        """
        # Prepare training data
        X, Y, mask = self._prepare_training_data()
        
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
        for i in range(iterations):
            optimizer.zero_grad()
            output = self.model(X)
            loss = masked_loss(output, Y, mask)
            loss.backward()
            optimizer.step()
            
            if verbose and (i % 100 == 0 or i == iterations - 1):
                print(f'Iteration {i+1}/{iterations} - Loss: {loss.item():.4f}')
        
        # Set model to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        

    def predict(self, grid):
        """
        Make predictions across a grid of locations.
        
        Args:
            grid: Grid of points to predict at [n_points, 2]
            
        Returns:
            mean: Predicted means [n_points, n_properties]
            std: Predicted standard deviations [n_points, n_properties]
        """
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
        grid = torch.tensor(grid, dtype=torch.float32) if not isinstance(grid, torch.Tensor) else grid
        
        # Get predictions
        mean, std = self.predict(grid)
        
        if strategy == 'uncertainty':
            # Total uncertainty across all properties
            total_uncertainty = torch.sum(std, dim=1)
            best_idx = torch.argmax(total_uncertainty)
            best_location = grid[best_idx]
            best_score = total_uncertainty[best_idx]
            score_grid = total_uncertainty
            
        elif strategy == 'ei':
            # Expected improvement for first property (e.g., porosity)
            # Get best observed value so far
            observed_values = np.array([well['measurements'][0] for well in self.wells 
                                       if well['mask'][0]])
            
            if len(observed_values) == 0:
                # If no observations yet, fall back to uncertainty
                return self.plan_next_well(grid, strategy='uncertainty')
                
            best_f = torch.tensor(np.max(observed_values), dtype=torch.float32)
            
            # Calculate improvement
            improvement = mean[:, 0] - best_f
            
            # Calculate z-score
            z = improvement / std[:, 0]
            
            # Expected improvement
            norm_dist = torch.distributions.Normal(0, 1)
            ei = improvement * norm_dist.cdf(z) + std[:, 0] * torch.exp(norm_dist.log_prob(z))
            
            # Find best location
            best_idx = torch.argmax(ei)
            best_location = grid[best_idx]
            best_score = ei[best_idx]
            score_grid = ei
            
        elif strategy == 'economic':
            if economic_params is None:
                raise ValueError("Economic parameters required for 'economic' strategy")
                
            # Calculate expected monetary value at each location
            emv = self._calculate_economic_value(grid, mean, std, economic_params)
            
            # Find best location
            best_idx = torch.argmax(emv)
            best_location = grid[best_idx]
            best_score = emv[best_idx]
            score_grid = emv
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        return best_location, best_score, score_grid
        
    def _calculate_economic_value(self, locations, mean, std, params):
        """
        Calculate expected monetary value at given locations.
        
        Args:
            locations: Locations to evaluate [n_points, 2]
            mean: Predicted means [n_points, n_properties]
            std: Predicted standard deviations [n_points, n_properties]
            params: Economic parameters
                
        Returns:
            emv: Expected monetary value [n_points]
        """
        # For simplicity, assuming:
        # Property 0: Porosity (fraction)
        # Property 1: Permeability (mD)
        # Property 2: Thickness (m)
        
        # Extract predictions
        porosity = mean[:, 0]
        permeability = mean[:, 1]
        thickness = mean[:, 2] if self.n_properties > 2 else torch.ones_like(porosity) * params.get('default_thickness', 50)
        
        # Extract uncertainties
        porosity_std = std[:, 0]
        permeability_std = std[:, 1]
        thickness_std = std[:, 2] if self.n_properties > 2 else torch.zeros_like(porosity)
        
        # Calculate hydrocarbon volume
        area = params.get('area', 1.0e6)  # m²
        hydrocarbon_saturation = 1.0 - params.get('water_saturation', 0.3)
        formation_volume_factor = params.get('formation_volume_factor', 1.1)
        
        # Original oil in place (m³)
        ooip = area * thickness * porosity * hydrocarbon_saturation / formation_volume_factor
        
        # Recovery factor based on permeability
        recovery_factor = 0.1 + 0.2 * torch.log10(torch.clamp(permeability, min=1.0) / 100)
        recovery_factor = torch.clamp(recovery_factor, 0.05, 0.6)
        
        # Recoverable oil (m³)
        recoverable_oil = ooip * recovery_factor
        
        # Convert to barrels
        barrels = recoverable_oil * 6.29
        
        # Revenue
        oil_price = params.get('oil_price', 70)  # $ per barrel
        revenue = barrels * oil_price
        
        # Cost
        base_cost = params.get('drilling_cost', 1e7) + params.get('completion_cost', 5e6)
        
        # Expected monetary value
        emv = revenue - base_cost
        
        # Apply risk adjustment based on uncertainty
        # Higher uncertainty means higher risk, which reduces EMV
        uncertainty_factor = 1.0 - torch.clamp(
            (porosity_std / porosity + permeability_std / permeability) / 2, 0, 0.5
        )
        risk_adjusted_emv = emv * uncertainty_factor
        
        return risk_adjusted_emv
        
    def sequential_exploration(self, grid, n_wells, true_functions, noise_std=0.01, 
                              strategy='uncertainty', economic_params=None, plot=True):
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
        grid = torch.tensor(grid, dtype=torch.float32) if not isinstance(grid, torch.Tensor) else grid
        
        # Setup visualization if plotting
        if plot:
            n_cols = self.n_properties + 2  # Properties + uncertainty + strategy
            fig, axes = plt.subplots(n_wells, n_cols, figsize=(n_cols * 4, n_wells * 4))
            if n_wells == 1:
                axes = axes.reshape(1, -1)
                
            # Determine grid shape for 2D plotting
            grid_shape = None
            if grid.shape[1] == 2:
                unique_x = torch.unique(grid[:, 0]).shape[0]
                unique_y = torch.unique(grid[:, 1]).shape[0]
                if unique_x * unique_y == grid.shape[0]:
                    grid_shape = (unique_y, unique_x)
                    x1_unique = torch.unique(grid[:, 0])
                    x2_unique = torch.unique(grid[:, 1])
                    x1_grid, x2_grid = torch.meshgrid(x1_unique, x2_unique, indexing='ij')
        
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
            
            # Plot results
            if plot and grid_shape is not None:
                # Make predictions with updated model
                self.fit(verbose=False)
                mean, std = self.predict(grid)
                
                # Reshape for plotting
                mean_reshaped = [mean[:, j].reshape(grid_shape).numpy() for j in range(self.n_properties)]
                std_reshaped = [std[:, j].reshape(grid_shape).numpy() for j in range(self.n_properties)]
                
                # Get well locations
                well_x = np.array([well['location'][0] for well in self.wells])
                well_y = np.array([well['location'][1] for well in self.wells])
                
                # Plot each property
                for j in range(self.n_properties):
                    ax = axes[i, j]
                    im = ax.contourf(x1_grid.numpy(), x2_grid.numpy(), mean_reshaped[j], levels=20, cmap='viridis')
                    plt.colorbar(im, ax=ax)
                    ax.scatter(well_x[:-1], well_y[:-1], color='red', s=30, marker='x')
                    ax.scatter(well_x[-1], well_y[-1], color='green', s=80, marker='o')
                    ax.set_title(f'Well #{i+1}: {self.properties[j].capitalize()}')
                    
                # Plot combined uncertainty
                total_std = torch.sum(std, dim=1).reshape(grid_shape).numpy()
                ax = axes[i, self.n_properties]
                im = ax.contourf(x1_grid.numpy(), x2_grid.numpy(), total_std, levels=20, cmap='cividis')
                plt.colorbar(im, ax=ax)
                ax.scatter(well_x[:-1], well_y[:-1], color='red', s=30, marker='x')
                ax.scatter(well_x[-1], well_y[-1], color='green', s=80, marker='o')
                ax.set_title(f'Well #{i+1}: Uncertainty')
                
                # Plot acquisition function
                ax = axes[i, self.n_properties + 1]
                score_reshaped = score_grid.reshape(grid_shape).numpy()
                im = ax.contourf(x1_grid.numpy(), x2_grid.numpy(), score_reshaped, levels=20, cmap='plasma')
                plt.colorbar(im, ax=ax)
                ax.scatter(well_x[:-1], well_y[:-1], color='red', s=30, marker='x')
                ax.scatter(well_x[-1], well_y[-1], color='green', s=80, marker='o')
                ax.set_title(f'Well #{i+1}: {strategy.capitalize()} Score')
        
        if plot:
            plt.tight_layout()
            plt.show()
                
        return history



# Create synthetic basin example
basin_size = (20, 20)  # 20 km x 20 km basin
resolution = 30  # Grid resolution

# Create grid of points
x1 = np.linspace(0, basin_size[0], resolution)
x2 = np.linspace(0, basin_size[1], resolution)
x1_grid, x2_grid = np.meshgrid(x1, x2)
grid = np.column_stack([x1_grid.flatten(), x2_grid.flatten()])
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# Define true geological functions (unknown in real life, but defined here for simulation)
def true_porosity(x):
    """True porosity function with multiple sweet spots."""
    x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # Major sweet spot
    spot1 = 0.25 * torch.exp(-0.1 * ((x_tensor[:, 0] - 5)**2 + (x_tensor[:, 1] - 15)**2))
    
    # Secondary sweet spot
    spot2 = 0.2 * torch.exp(-0.15 * ((x_tensor[:, 0] - 15)**2 + (x_tensor[:, 1] - 8)**2))
    
    # Background trend (increasing toward the north-east)
    trend = 0.05 + 0.1 * (x_tensor[:, 0] / basin_size[0] + x_tensor[:, 1] / basin_size[1]) / 2
    
    return spot1 + spot2 + trend

def true_permeability(x):
    """Permeability linked to porosity but with its own spatial patterns."""
    x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # Get porosity (main driver)
    porosity = true_porosity(x_tensor)
    
    # Additional spatial variation
    fault_effect = 500 * torch.exp(-0.2 * (x_tensor[:, 0] - 10)**2 / 4)
    
    # Convert from porosity (typical log-linear relationship)
    perm_base = 10**(porosity * 15 - 1)  # Typical transform
    
    return perm_base + fault_effect

def true_thickness(x):
    """Reservoir thickness with structural trends."""
    x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # Main structural high
    structure = 40 * torch.exp(-0.05 * ((x_tensor[:, 0] - 10)**2 + (x_tensor[:, 1] - 10)**2))
    
    # Basin deepening trend toward the south
    trend = 30 * (1 - x_tensor[:, 1] / basin_size[1])
    
    return structure + trend + 20  # Add baseline thickness

# Calculate true values for visualization
true_porosity_grid = true_porosity(grid_tensor).reshape(resolution, resolution).numpy()
true_permeability_grid = true_permeability(grid_tensor).reshape(resolution, resolution).numpy()
true_thickness_grid = true_thickness(grid_tensor).reshape(resolution, resolution).numpy()

# Visualize true geology
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im0 = axes[0].contourf(x1_grid, x2_grid, true_porosity_grid, levels=20, cmap='viridis')
axes[0].set_title('True Porosity')
axes[0].set_xlabel('X (km)')
axes[0].set_ylabel('Y (km)')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].contourf(x1_grid, x2_grid, true_permeability_grid, levels=20, cmap='plasma')
axes[1].set_title('True Permeability (mD)')
axes[1].set_xlabel('X (km)')
axes[1].set_ylabel('Y (km)')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].contourf(x1_grid, x2_grid, true_thickness_grid, levels=20, cmap='cividis')
axes[2].set_title('True Thickness (m)')
axes[2].set_xlabel('X (km)')
axes[2].set_ylabel('Y (km)')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

# Initialize the exploration framework
basin_gp = BasinExplorationGP(
    basin_size=basin_size,
    properties=['porosity', 'permeability', 'thickness']
)

# Add initial wells (random exploration at first)
np.random.seed(42)
n_initial_wells = 3

for i in range(n_initial_wells):
    location = np.random.uniform(0, basin_size, size=2)
    
    # Measure properties with noise
    #porosity = true_porosity(torch.tensor([location], dtype=torch.float32)).item() + np.random.normal(0, 0.01)
    location_np = np.array([location])
    porosity = true_porosity(torch.tensor(location_np, dtype=torch.float32)).item() + np.random.normal(0, 0.01)
    permeability = true_permeability(torch.tensor([location], dtype=torch.float32)).item() + np.random.normal(0, 50)
    thickness = true_thickness(torch.tensor([location], dtype=torch.float32)).item() + np.random.normal(0, 2)
    
    measurements = {
        'porosity': porosity,
        'permeability': permeability,
        'thickness': thickness
    }
    
    basin_gp.add_well(location, measurements, well_name=f"Initial_Well_{i+1}")

print(f"Added {n_initial_wells} initial wells")

# Define economic parameters for exploration planning
economic_params = {
    'area': 1.0e6,  # m²
    'water_saturation': 0.3,
    'formation_volume_factor': 1.1,
    'oil_price': 80,  # $ per barrel
    'drilling_cost': 8e6,  # $
    'completion_cost': 4e6  # $
}

# Run sequential exploration with different strategies
n_exploration_wells = 5

# 1. Uncertainty-based strategy
print("\nRunning uncertainty-based exploration...")
uncertainty_history = basin_gp.sequential_exploration(
    grid_tensor,
    n_exploration_wells,
    [true_porosity, true_permeability, true_thickness],
    noise_std=0.01,
    strategy='uncertainty',
    plot=True
)

# Reset and try economic-based strategy
basin_gp = BasinExplorationGP(
    basin_size=basin_size,
    properties=['porosity', 'permeability', 'thickness']
)

# Re-add initial wells
for i in range(n_initial_wells):
    location = np.random.uniform(0, basin_size, size=2)
    
    # Convert to tensor properly
    location_tensor = torch.tensor(location.reshape(1, -1), dtype=torch.float32)
    
    # Measure properties with noise
    porosity = true_porosity(location_tensor).item() + np.random.normal(0, 0.01)
    permeability = true_permeability(location_tensor).item() + np.random.normal(0, 50)
    thickness = true_thickness(location_tensor).item() + np.random.normal(0, 2)
    
    measurements = {
        'porosity': porosity,
        'permeability': permeability,
        'thickness': thickness
    }
    
    basin_gp.add_well(location, measurements, well_name=f"Initial_Well_{i+1}")

# 2. Economic-based strategy
print("\nRunning economic-based exploration...")
economic_history = basin_gp.sequential_exploration(
    grid_tensor,
    n_exploration_wells,
    [true_porosity, true_permeability, true_thickness],
    noise_std=0.01,
    strategy='economic',
    economic_params=economic_params,
    plot=True
)

# Compare final models from both strategies
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Get final predictions from uncertainty-based strategy
basin_gp.fit(verbose=False)
mean_uncertainty, std_uncertainty = basin_gp.predict(grid_tensor)

# Get well locations
well_x = np.array([well['location'][0] for well in basin_gp.wells])
well_y = np.array([well['location'][1] for well in basin_gp.wells])

# Reshape for plotting
mean_porosity = mean_uncertainty[:, 0].reshape(resolution, resolution).numpy()
mean_permeability = mean_uncertainty[:, 1].reshape(resolution, resolution).numpy()
mean_thickness = mean_uncertainty[:, 2].reshape(resolution, resolution).numpy()

# Plot uncertainty-based results
im0 = axes[0, 0].contourf(x1_grid, x2_grid, mean_porosity, levels=20, cmap='viridis')
axes[0, 0].scatter(well_x, well_y, color='red', s=30, marker='x')
axes[0, 0].set_title('Economic Strategy: Porosity')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].contourf(x1_grid, x2_grid, mean_permeability, levels=20, cmap='plasma')
axes[0, 1].scatter(well_x, well_y, color='red', s=30, marker='x')
axes[0, 1].set_title('Economic Strategy: Permeability')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].contourf(x1_grid, x2_grid, mean_thickness, levels=20, cmap='cividis')
axes[0, 2].scatter(well_x, well_y, color='red', s=30, marker='x')
axes[0, 2].set_title('Economic Strategy: Thickness')
plt.colorbar(im2, ax=axes[0, 2])

# Compare with true geology
im3 = axes[1, 0].contourf(x1_grid, x2_grid, true_porosity_grid, levels=20, cmap='viridis')
axes[1, 0].set_title('True Porosity')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].contourf(x1_grid, x2_grid, true_permeability_grid, levels=20, cmap='plasma')
axes[1, 1].set_title('True Permeability')
plt.colorbar(im4, ax=axes[1, 1])

im5 = axes[1, 2].contourf(x1_grid, x2_grid, true_thickness_grid, levels=20, cmap='cividis')
axes[1, 2].set_title('True Thickness')
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.show()

# Calculate estimated recoverable resources
def calculate_resources(mean_predictions, grid_size, basin_size):
    """Calculate total recoverable resources based on predictions."""
    # Extract predictions
    porosity = mean_predictions[:, 0]
    permeability = mean_predictions[:, 1]
    thickness = mean_predictions[:, 2]
    
    # Calculate cell area
    cell_area = (basin_size[0] / (resolution-1)) * (basin_size[1] / (resolution-1)) * 1e6  # m²
    
    # Original oil in place
    hydrocarbon_saturation = 1.0 - 0.3  # 1 - water saturation
    formation_volume_factor = 1.1
    ooip = cell_area * thickness * porosity * hydrocarbon_saturation / formation_volume_factor
    
    # Recovery factor
    recovery_factor = 0.1 + 0.2 * torch.log10(torch.clamp(permeability, min=1.0) / 100)
    recovery_factor = torch.clamp(recovery_factor, 0.05, 0.6)
    
    # Recoverable oil
    recoverable_oil = ooip * recovery_factor
    
    # Total recoverable (m³)
    total_recoverable = torch.sum(recoverable_oil).item()
    
    # Convert to barrels
    total_barrels = total_recoverable * 6.29
    
    return total_barrels

# Calculate resources based on economic-based exploration
total_resources = calculate_resources(mean_uncertainty, resolution, basin_size)
print(f"\nEstimated total recoverable resources: {total_resources/1e6:.2f} million barrels")

# Calculate economic value
total_value = total_resources * economic_params['oil_price']
total_cost = len(basin_gp.wells) * (economic_params['drilling_cost'] + economic_params['completion_cost'])
net_value = total_value - total_cost

print(f"Estimated total economic value: ${total_value/1e9:.2f} billion")
print(f"Total exploration cost: ${total_cost/1e6:.2f} million")
print(f"Net value: ${net_value/1e9:.2f} billion")