import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, List, Optional, Tuple, Union

class ExplorationGP:
    """
    A comprehensive framework for Gaussian Process-based exploration planning.
    
    This class handles:
    - Multi-output GPs for modeling related geological properties
    - Various acquisition functions for well planning
    - Visualization and evaluation utilities
    - Sequential drilling strategies
    """
    
    def __init__(
        self, 
        n_outputs: int = 2, 
        input_dim: int = 2,
        kernel_type: str = 'rbf',
        rank: int = 2,
        normalize_data: bool = True,
        learning_rate: float = 0.1,
        training_iterations: int = 200
    ):
        """
        Initialize the Exploration GP framework.
        
        Args:
            n_outputs: Number of output variables (geological properties)
            input_dim: Input dimensionality (typically 2 for spatial coordinates)
            kernel_type: Type of kernel ('rbf', 'matern', etc.)
            rank: Rank of the multi-output kernel (controls correlation complexity)
            normalize_data: Whether to normalize input and output data
            learning_rate: Learning rate for model optimization
            training_iterations: Number of iterations for model training
        """
        self.n_outputs = n_outputs
        self.input_dim = input_dim
        self.kernel_type = kernel_type
        self.rank = rank
        self.normalize_data = normalize_data
        self.learning_rate = learning_rate
        self.training_iterations = training_iterations
        
        # Will be initialized later
        self.model = None
        self.likelihood = None
        self.X = None
        self.Y = None
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        
    def _create_base_kernel(self):
        """Create the appropriate base kernel based on kernel_type."""
        if self.kernel_type.lower() == 'rbf':
            return gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)
        elif self.kernel_type.lower() == 'matern':
            return gpytorch.kernels.MaternKernel(ard_num_dims=self.input_dim, nu=2.5)
        elif self.kernel_type.lower() == 'periodic':
            return gpytorch.kernels.PeriodicKernel(ard_num_dims=self.input_dim)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
    def _initialize_model(self, X, Y):
        """Initialize the multi-output GP model."""
        # Define model class internally
        class MultitaskGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, base_kernel, n_outputs, rank):
                super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
                
                # Mean module
                self.mean_module = gpytorch.means.MultitaskMean(
                    gpytorch.means.ConstantMean(), num_tasks=n_outputs
                )
                
                # Base kernel (shared across tasks)
                self.base_covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
                
                # Multi-task kernel
                self.covar_module = gpytorch.kernels.MultitaskKernel(
                    self.base_covar_module, num_tasks=n_outputs, rank=rank
                )
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        
        # Create base kernel
        base_kernel = self._create_base_kernel()
        
        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_outputs)
        self.model = MultitaskGPModel(X, Y, self.likelihood, base_kernel, self.n_outputs, self.rank)
    
    def fit(self, X, Y, verbose=True):
        """
        Fit the GP model to observed data.
        
        Args:
            X: Input coordinates (well locations) [n_samples, input_dim]
            Y: Output values (geological properties) [n_samples, n_outputs]
            verbose: Whether to print training progress
        """
        X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        Y = torch.tensor(Y, dtype=torch.float32) if not isinstance(Y, torch.Tensor) else Y
        
        # Store original data
        self.X = X
        self.Y = Y
        
        # Normalize data if requested
        if self.normalize_data:
            self.X_mean, self.X_std = X.mean(dim=0), X.std(dim=0)
            self.Y_mean, self.Y_std = Y.mean(dim=0), Y.std(dim=0)
            
            # Avoid division by zero
            self.X_std = torch.where(self.X_std == 0, torch.ones_like(self.X_std), self.X_std)
            self.Y_std = torch.where(self.Y_std == 0, torch.ones_like(self.Y_std), self.Y_std)
            
            X_normalized = (X - self.X_mean) / self.X_std
            Y_normalized = (Y - self.Y_mean) / self.Y_std
        else:
            X_normalized = X
            Y_normalized = Y
            self.X_mean, self.X_std = 0.0, 1.0
            self.Y_mean, self.Y_std = 0.0, 1.0
        
        # Initialize model
        self._initialize_model(X_normalized, Y_normalized)
        
        # Train the model
        self.model.train()
        self.likelihood.train()
        
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss function
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Training loop
        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = self.model(X_normalized)
            loss = -mll(output, Y_normalized)
            loss.backward()
            optimizer.step()
            
            if verbose and (i % 50 == 0 or i == self.training_iterations - 1):
                print(f'Iteration {i+1}/{self.training_iterations} - Loss: {loss.item():.4f}')
        
        # Set model to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        return self
    
    def predict(self, X_new):
        """
        Make predictions at new points.
        
        Args:
            X_new: New input points [n_points, input_dim]
            
        Returns:
            mean: Predicted means [n_points, n_outputs]
            variance: Predicted variances [n_points, n_outputs]
        """
        X_new = torch.tensor(X_new, dtype=torch.float32) if not isinstance(X_new, torch.Tensor) else X_new
        
        # Normalize inputs
        if self.normalize_data:
            X_new_normalized = (X_new - self.X_mean) / self.X_std
        else:
            X_new_normalized = X_new
        
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X_new_normalized))
            mean = predictions.mean
            variance = predictions.variance
        
        # Denormalize outputs
        if self.normalize_data:
            mean = mean * self.Y_std + self.Y_mean
        
        return mean, variance
    
    def get_property_correlation(self):
        """
        Extract the learned correlation between properties.
        
        Returns:
            corr_matrix: Property correlation matrix [n_outputs, n_outputs]
        """
        with torch.no_grad():
            task_covar_matrix = self.model.covar_module.task_covar_module.covar_matrix.evaluate()
            task_corr_matrix = task_covar_matrix / torch.sqrt(
                torch.diag(task_covar_matrix).reshape(-1, 1) * torch.diag(task_covar_matrix).reshape(1, -1)
            )
        
        return task_corr_matrix.numpy()
    
    def calculate_acquisition(self, X_new, acquisition_type='variance'):
        """
        Calculate acquisition function values for candidate points.
        
        Args:
            X_new: Candidate points [n_points, input_dim]
            acquisition_type: Type of acquisition function:
                - 'variance': Total predictive variance
                - 'ivr': Integrated variance reduction
                - 'ei': Expected improvement (for targeting high property values)
                
        Returns:
            acq_values: Acquisition function values [n_points]
        """
        X_new = torch.tensor(X_new, dtype=torch.float32) if not isinstance(X_new, torch.Tensor) else X_new
        
        # Normalize inputs
        if self.normalize_data:
            X_new_normalized = (X_new - self.X_mean) / self.X_std
        else:
            X_new_normalized = X_new
            
        if acquisition_type == 'variance':
            # Simply use total predictive variance
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = self.likelihood(self.model(X_new_normalized))
                variance = predictions.variance
                total_variance = variance.sum(dim=1)  # Sum over all properties
            
            return total_variance
            
        elif acquisition_type == 'ivr':
            # Integrated variance reduction (more complex, requires fantasy models)
            X_grid_normalized = X_new_normalized
            
            # Current predictive variance at all grid points
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = self.likelihood(self.model(X_grid_normalized))
                current_var = predictions.variance.sum(dim=1)
            
            total_ivr = torch.zeros(X_new_normalized.shape[0])
            
            # For each candidate, calculate expected variance reduction
            for i in range(X_new_normalized.shape[0]):
                # Create fantasy model
                fantasy_model = self.model.get_fantasy_model(
                    X_new_normalized[i:i+1], 
                    self.likelihood(self.model(X_new_normalized[i:i+1])).mean
                )
                
                # Compute posterior variance with fantasy observation
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    fantasy_var = self.likelihood(fantasy_model(X_grid_normalized)).variance.sum(dim=1)
                
                # Integrated variance reduction
                total_ivr[i] = (current_var - fantasy_var).sum()
            
            return total_ivr
            
        elif acquisition_type == 'ei':
            # Expected improvement targets high property values (for property 0 by default)
            best_f = torch.max(self.Y[:, 0])
            
            with torch.no_grad():
                # Get predictive mean and variance
                predictions = self.likelihood(self.model(X_new_normalized))
                mean = predictions.mean[:, 0]  # For first property
                variance = predictions.variance[:, 0]
                std = torch.sqrt(variance)
                
                # If normalized, denormalize for comparison with best_f
                if self.normalize_data:
                    mean = mean * self.Y_std[0] + self.Y_mean[0]
                    std = std * self.Y_std[0]
                
                # Calculate improvement
                improvement = mean - best_f
                z = improvement / std
                
                # EI formula
                ei = improvement * torch.distributions.Normal(0, 1).cdf(z) + \
                     std * torch.distributions.Normal(0, 1).log_prob(z).exp()
                
                # Set EI to 0 where std is 0
                ei[std == 0] = 0.0
            
            return ei
        
        else:
            raise ValueError(f"Unsupported acquisition type: {acquisition_type}")
    
    def select_next_well(self, X_candidates, acquisition_type='variance'):
        """
        Select the next well location from candidate points.
        
        Args:
            X_candidates: Candidate points [n_candidates, input_dim]
            acquisition_type: Type of acquisition function
            
        Returns:
            best_point: Best candidate point [1, input_dim]
            acq_value: Acquisition value at the best point
        """
        # Calculate acquisition function
        acq_values = self.calculate_acquisition(X_candidates, acquisition_type)
        
        # Find the best candidate
        best_idx = torch.argmax(acq_values)
        best_point = X_candidates[best_idx:best_idx+1]
        best_acq_value = acq_values[best_idx]
        
        return best_point, best_acq_value
    
    def add_observation(self, X_new, Y_new, refit=True):
        """
        Add a new observation and optionally refit the model.
        
        Args:
            X_new: New observation location [1, input_dim]
            Y_new: New observation values [1, n_outputs]
            refit: Whether to refit the model after adding the observation
            
        Returns:
            self
        """
        X_new = torch.tensor(X_new, dtype=torch.float32) if not isinstance(X_new, torch.Tensor) else X_new
        Y_new = torch.tensor(Y_new, dtype=torch.float32) if not isinstance(Y_new, torch.Tensor) else Y_new
        
        # Update dataset
        self.X = torch.cat([self.X, X_new], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)
        
        # Refit model if requested
        if refit:
            self.fit(self.X, self.Y, verbose=False)
        
        return self
    
    def sequential_design(self, 
                         grid, 
                         n_wells, 
                         true_functions, 
                         noise_std=0.01, 
                         acquisition_type='variance',
                         plot=True):
        """
        Sequentially design an exploration campaign.
        
        Args:
            grid: Grid of candidate points [n_grid_points, input_dim]
            n_wells: Number of new wells to design
            true_functions: List of callables that return the true property values
            noise_std: Standard deviation of measurement noise
            acquisition_type: Type of acquisition function
            plot: Whether to plot the results at each step
            
        Returns:
            X_selected: Selected well locations [n_wells, input_dim]
            Y_selected: Property values at selected locations [n_wells, n_outputs]
        """
        grid = torch.tensor(grid, dtype=torch.float32) if not isinstance(grid, torch.Tensor) else grid
        
        # Create figure for plotting
        if plot:
            n_cols = self.n_outputs + 2  # Properties + uncertainty + acquisition
            fig, axes = plt.subplots(n_wells, n_cols, figsize=(n_cols * 4, n_wells * 4))
            if n_wells == 1:
                axes = axes.reshape(1, -1)
        
        X_selected = []
        Y_selected = []
        
        # Get grid shape for plotting
        grid_shape = None
        if plot and grid.shape[1] == 2:
            # Try to determine grid shape for 2D plots
            unique_x = torch.unique(grid[:, 0]).shape[0]
            unique_y = torch.unique(grid[:, 1]).shape[0]
            if unique_x * unique_y == grid.shape[0]:
                grid_shape = (unique_y, unique_x)
        
        for i in range(n_wells):
            # Calculate acquisition function
            acq_values = self.calculate_acquisition(grid, acquisition_type)
            
            # Select the best point
            best_idx = torch.argmax(acq_values)
            best_point = grid[best_idx:best_idx+1]
            
            # "Drill" the well (evaluate true functions with noise)
            best_values = []
            for func in true_functions:
                if callable(func):
                    value = func(best_point)
                    if isinstance(value, torch.Tensor):
                        value = value + torch.randn(1) * noise_std
                    else:
                        value = torch.tensor([value + np.random.normal(0, noise_std)], dtype=torch.float32)
                    best_values.append(value)
            
            best_values = torch.cat(best_values, dim=0).reshape(1, -1)
            
            # Store selected point and values
            X_selected.append(best_point)
            Y_selected.append(best_values)
            
            # Add observation and update model
            self.add_observation(best_point, best_values)
            
            # Make predictions on the grid
            mean, variance = self.predict(grid)
            
            # Plot results
            if plot:
                if grid.shape[1] == 2 and grid_shape is not None:
                    x1_unique = torch.unique(grid[:, 0])
                    x2_unique = torch.unique(grid[:, 1])
                    x1_grid, x2_grid = torch.meshgrid(x1_unique, x2_unique, indexing='ij')
                    
                    # Plot each property
                    for j in range(self.n_outputs):
                        mean_grid = mean[:, j].reshape(grid_shape)
                        ax = axes[i, j]
                        im = ax.contourf(x1_grid.numpy(), x2_grid.numpy(), mean_grid.numpy(), 
                                       levels=20, cmap='viridis')
                        plt.colorbar(im, ax=ax)
                        ax.scatter(self.X[:-1, 0].numpy(), self.X[:-1, 1].numpy(), 
                                 color='red', s=30, marker='x')
                        ax.scatter(best_point[:, 0].numpy(), best_point[:, 1].numpy(), 
                                 color='green', s=80, marker='o')
                        ax.set_title(f'Well #{i+1}: Property {j+1}')
                        
                    # Plot combined uncertainty
                    total_var = torch.sum(variance, dim=1).reshape(grid_shape)
                    ax = axes[i, self.n_outputs]
                    im = ax.contourf(x1_grid.numpy(), x2_grid.numpy(), total_var.numpy(), 
                                   levels=20, cmap='cividis')
                    plt.colorbar(im, ax=ax)
                    ax.scatter(self.X[:-1, 0].numpy(), self.X[:-1, 1].numpy(), 
                             color='red', s=30, marker='x')
                    ax.scatter(best_point[:, 0].numpy(), best_point[:, 1].numpy(), 
                             color='green', s=80, marker='o')
                    ax.set_title(f'Well #{i+1}: Uncertainty')
                    
                    # Plot acquisition function
                    acq_grid = acq_values.reshape(grid_shape)
                    ax = axes[i, self.n_outputs + 1]
                    im = ax.contourf(x1_grid.numpy(), x2_grid.numpy(), acq_grid.numpy(), 
                                   levels=20, cmap='plasma')
                    plt.colorbar(im, ax=ax)
                    ax.scatter(self.X[:-1, 0].numpy(), self.X[:-1, 1].numpy(), 
                             color='red', s=30, marker='x')
                    ax.scatter(best_point[:, 0].numpy(), best_point[:, 1].numpy(), 
                             color='green', s=80, marker='o')
                    ax.set_title(f'Well #{i+1}: Acquisition')
        
        if plot:
            plt.tight_layout()
            plt.show()
        
        # Concatenate all selected points and values
        X_selected = torch.cat(X_selected, dim=0)
        Y_selected = torch.cat(Y_selected, dim=0)
        
        return X_selected, Y_selected
    
    def evaluate_model(self, X_test, Y_test):
        """
        Evaluate model performance against test data.
        
        Args:
            X_test: Test input locations [n_test, input_dim]
            Y_test: True property values [n_test, n_outputs]
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        X_test = torch.tensor(X_test, dtype=torch.float32) if not isinstance(X_test, torch.Tensor) else X_test
        Y_test = torch.tensor(Y_test, dtype=torch.float32) if not isinstance(Y_test, torch.Tensor) else Y_test
        
        # Make predictions
        mean, variance = self.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        
        # Error metrics per property
        for j in range(self.n_outputs):
            # Mean Absolute Error
            mae = torch.mean(torch.abs(mean[:, j] - Y_test[:, j])).item()
            metrics[f'MAE_property_{j}'] = mae
            
            # Root Mean Squared Error
            rmse = torch.sqrt(torch.mean((mean[:, j] - Y_test[:, j])**2)).item()
            metrics[f'RMSE_property_{j}'] = rmse
            
            # Standardized log loss (negative log likelihood under Gaussian assumptions)
            std = torch.sqrt(variance[:, j])
            nll = 0.5 * torch.mean(((mean[:, j] - Y_test[:, j]) / std)**2 + torch.log(2 * np.pi * variance[:, j])).item()
            metrics[f'NLL_property_{j}'] = nll
            
            # Calculate proportion of test points within confidence intervals
            z_scores = torch.abs((mean[:, j] - Y_test[:, j]) / std)
            coverage_68 = torch.mean((z_scores <= 1.0).float()).item()
            coverage_95 = torch.mean((z_scores <= 2.0).float()).item()
            metrics[f'68%_Coverage_property_{j}'] = coverage_68
            metrics[f'95%_Coverage_property_{j}'] = coverage_95
        
        return metrics
    
    def plot_model_diagnostics(self, X_test, Y_test):
        """
        Plot comprehensive model diagnostics.
        
        Args:
            X_test: Test input locations [n_test, input_dim]
            Y_test: True property values [n_test, n_outputs]
        """
        X_test = torch.tensor(X_test, dtype=torch.float32) if not isinstance(X_test, torch.Tensor) else X_test
        Y_test = torch.tensor(Y_test, dtype=torch.float32) if not isinstance(Y_test, torch.Tensor) else Y_test
        
        # Make predictions
        mean, variance = self.predict(X_test)
        std = torch.sqrt(variance)
        
        # Create figure
        n_rows = self.n_outputs
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for j in range(self.n_outputs):
            # 1. Actual vs Predicted plot
            ax = axes[j, 0]
            ax.scatter(Y_test[:, j].numpy(), mean[:, j].numpy(), alpha=0.7)
            
            # Add perfect prediction line
            min_val = min(Y_test[:, j].min().item(), mean[:, j].min().item())
            max_val = max(Y_test[:, j].max().item(), mean[:, j].max().item())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--')
            
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')
            ax.set_title(f'Property {j+1}: Actual vs Predicted')
            
            # 2. Standardized residuals histogram
            ax = axes[j, 1]
            z_scores = ((mean[:, j] - Y_test[:, j]) / std[:, j]).numpy()
            ax.hist(z_scores, bins=30, alpha=0.7)
            
            # Add reference lines
            ax.axvline(x=-2, color='r', linestyle='--')
            ax.axvline(x=2, color='r', linestyle='--')
            
            ax.set_xlabel('Standardized Residual')
            ax.set_ylabel('Count')
            ax.set_title(f'Property {j+1}: Standardized Residuals')
            
            # 3. Uncertainty calibration plot
            ax = axes[j, 2]
            
            # Calculate quantiles for calibration
            quantiles = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0
            emp_quantiles = []
            
            for q in quantiles:
                if q == 0:
                    emp_quantiles.append(0)
                elif q == 1:
                    emp_quantiles.append(1)
                else:
                    # Theoretical: |prediction - true| / std < norm.ppf(q)
                    threshold = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + q/2))
                    emp_quantiles.append(torch.mean((torch.abs(z_scores) <= threshold).float()).item())
            
            # Plot calibration curve
            ax.plot(quantiles, emp_quantiles, 'bo-')
            ax.plot([0, 1], [0, 1], 'k--')  # Perfect calibration line
            
            ax.set_xlabel('Theoretical Quantile')
            ax.set_ylabel('Empirical Quantile')
            ax.set_title(f'Property {j+1}: Uncertainty Calibration')
        
        plt.tight_layout()
        plt.show()