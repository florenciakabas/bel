import ipdb
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Synthetic 2D exploration scenario with initial wells
n_initial_wells = 5
X_initial = torch.tensor(np.random.uniform(0, 10, size=(n_initial_wells, 2)), dtype=torch.float)

# True geological properties
def true_porosity(x):
    return 0.2 * torch.exp(-0.1 * ((x[:, 0] - 5)**2 + (x[:, 1] - 5)**2)) + 0.1

def true_permeability(x, porosity):
    # Permeability-porosity relationship (simplified Kozeny-Carman)
    return 1000 * porosity**3 / ((1 - porosity)**2) + torch.randn(porosity.size()) * 50

# Generate observations with noise
porosity_initial = true_porosity(X_initial) + torch.randn(n_initial_wells) * 0.01
permeability_initial = true_permeability(X_initial, porosity_initial)
Y_initial = torch.stack([porosity_initial, permeability_initial], dim=1)

# Current dataset
X_current = X_initial.clone()
Y_current = Y_initial.clone()

# Define a multi-output GP model
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        
        # Mean module
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        
        # Base kernel
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        
        # Multi-task kernel
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.base_covar_module, num_tasks=2, rank=2
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# Define an acquisition function for well planning
def integrated_variance_reduction(model, likelihood, X_candidate, X_grid):
    """
    Computes the integrated variance reduction if we were to sample at X_candidate
    Higher values mean better sampling locations
    """
    # Current predictive variance at all grid points
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(X_grid))
        current_var = predictions.variance.sum(dim=1)  # Sum variance across both properties
    
    total_ivr = torch.zeros(X_candidate.shape[0])
    
    # For each candidate, calculate the expected variance reduction
    for i in range(X_candidate.shape[0]):
        # Create a copy of the model for "what-if" analysis
        fantasy_model = model.get_fantasy_model(
            X_candidate[i:i+1], 
            likelihood(model(X_candidate[i:i+1])).mean  # Using mean prediction as fantasy observation
        )
        
        # Compute posterior variance if we were to sample at this location
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            fantasy_var = likelihood(fantasy_model(X_grid)).variance.sum(dim=1)
        
        # Integrated variance reduction
        total_ivr[i] = (current_var - fantasy_var).sum()
    
    return total_ivr

# Create a grid for visualization and planning
resolution = 30
x1 = torch.linspace(0, 10, resolution)
x2 = torch.linspace(0, 10, resolution)
x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing='ij')
X_grid = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1)

# Plan and drill 5 additional wells sequentially
n_new_wells = 5

# Create figure for results
fig, axes = plt.subplots(n_new_wells, 4, figsize=(20, 5*n_new_wells))

for well_idx in range(n_new_wells):
    print(f"Planning well #{well_idx+1}")
    
    # Normalize data
    X_mean, X_std = X_current.mean(dim=0), X_current.std(dim=0)
    X_normalized = (X_current - X_mean) / X_std
    X_grid_normalized = (X_grid - X_mean) / X_std
    
    Y_mean, Y_std = Y_current.mean(dim=0), Y_current.std(dim=0)
    Y_normalized = (Y_current - Y_mean) / Y_std
    
    # Initialize model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPModel(X_normalized, Y_normalized, likelihood)
    
    # Train the model
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    n_iterations = 100
    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(X_normalized)
        loss = -mll(output, Y_normalized)
        loss.backward()
        optimizer.step()
    
    # Set to evaluation mode
    model.eval()
    likelihood.eval()
    
    # Make predictions on the grid
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(X_grid_normalized))
        mean = predictions.mean
        variance = predictions.variance
    
    # Denormalize predictions
    mean = mean * Y_std + Y_mean
    
    # Reshape for plotting
    mean_porosity = mean[:, 0].reshape(resolution, resolution)
    mean_permeability = mean[:, 1].reshape(resolution, resolution)
    var_porosity = variance[:, 0].reshape(resolution, resolution)
    var_permeability = variance[:, 1].reshape(resolution, resolution)
    
    # Calculate acquisition function for all grid points
    ivr = integrated_variance_reduction(model, likelihood, X_grid_normalized, X_grid_normalized)
    ivr_grid = ivr.reshape(resolution, resolution)
    
    # Find the best location for the next well
    best_idx = torch.argmax(ivr)
    next_well = X_grid[best_idx:best_idx+1]
    
    # "Drill" the new well (sample the true function)
    next_porosity = true_porosity(next_well) + torch.randn(1) * 0.01
    next_permeability = true_permeability(next_well, next_porosity)
    # Concatenate along dimension 0 to get a 1D tensor
    next_observation = torch.cat([next_porosity, next_permeability], dim=0)
    # Then reshape to [1, 2]
    next_observation = next_observation.reshape(1, 2)

    # Plot current state
    # 1. Porosity prediction
    axes[well_idx, 0].contourf(x1_grid.numpy(), x2_grid.numpy(), mean_porosity.numpy(), levels=20, cmap='viridis')
    axes[well_idx, 0].scatter(X_current[:, 0].numpy(), X_current[:, 1].numpy(), color='red', s=50, marker='x')
    axes[well_idx, 0].scatter(next_well[:, 0].numpy(), next_well[:, 1].numpy(), color='green', s=80, marker='o')
    axes[well_idx, 0].set_title(f'Well #{well_idx+1}: Porosity Prediction')
    
    # 2. Permeability prediction
    axes[well_idx, 1].contourf(x1_grid.numpy(), x2_grid.numpy(), mean_permeability.numpy(), levels=20, cmap='plasma')
    axes[well_idx, 1].scatter(X_current[:, 0].numpy(), X_current[:, 1].numpy(), color='red', s=50, marker='x')
    axes[well_idx, 1].scatter(next_well[:, 0].numpy(), next_well[:, 1].numpy(), color='green', s=80, marker='o')
    axes[well_idx, 1].set_title(f'Well #{well_idx+1}: Permeability Prediction')
    
    # 3. Combined uncertainty
    combined_var = var_porosity + var_permeability
    axes[well_idx, 2].contourf(x1_grid.numpy(), x2_grid.numpy(), combined_var.numpy(), levels=20, cmap='cividis')
    axes[well_idx, 2].scatter(X_current[:, 0].numpy(), X_current[:, 1].numpy(), color='red', s=50, marker='x')
    axes[well_idx, 2].scatter(next_well[:, 0].numpy(), next_well[:, 1].numpy(), color='green', s=80, marker='o')
    axes[well_idx, 2].set_title(f'Well #{well_idx+1}: Combined Uncertainty')
    
    # 4. Acquisition function
    im = axes[well_idx, 3].contourf(x1_grid.numpy(), x2_grid.numpy(), ivr_grid.numpy(), levels=20, cmap='hot')
    plt.colorbar(im, ax=axes[well_idx, 3])
    axes[well_idx, 3].scatter(X_current[:, 0].numpy(), X_current[:, 1].numpy(), color='red', s=50, marker='x')
    axes[well_idx, 3].scatter(next_well[:, 0].numpy(), next_well[:, 1].numpy(), color='green', s=80, marker='o')
    axes[well_idx, 3].set_title(f'Well #{well_idx+1}: Acquisition Function')
    
    # Add the new observation to our dataset
    X_current = torch.cat([X_current, next_well], dim=0)
    Y_current = torch.cat([Y_current, next_observation], dim=0)

plt.tight_layout()
plt.close()

# Final model and predictions for evaluation
# Normalize final data
X_mean, X_std = X_current.mean(dim=0), X_current.std(dim=0)
X_normalized = (X_current - X_mean) / X_std
X_grid_normalized = (X_grid - X_mean) / X_std

Y_mean, Y_std = Y_current.mean(dim=0), Y_current.std(dim=0)
Y_normalized = (Y_current - Y_mean) / Y_std

# Initialize and train final model
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskGPModel(X_normalized, Y_normalized, likelihood)

model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iterations = 200
for i in range(n_iterations):
    optimizer.zero_grad()
    output = model(X_normalized)
    loss = -mll(output, Y_normalized)
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        print(f'Final model - Iteration {i+1}/{n_iterations} - Loss: {loss.item():.3f}')

# Set to evaluation mode for final predictions
model.eval()
likelihood.eval()

# Make final predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(X_grid_normalized))
    mean = predictions.mean
    variance = predictions.variance

# Denormalize predictions
mean = mean * Y_std + Y_mean

# Reshape for plotting
mean_porosity = mean[:, 0].reshape(resolution, resolution)
mean_permeability = mean[:, 1].reshape(resolution, resolution)
var_porosity = variance[:, 0].reshape(resolution, resolution)
var_permeability = variance[:, 1].reshape(resolution, resolution)

# Calculate true values for comparison
true_porosity_grid = true_porosity(X_grid).reshape(resolution, resolution)
porosity_for_perm = true_porosity(X_grid) + torch.randn(X_grid.shape[0]) * 0.01
true_permeability_grid = true_permeability(X_grid, porosity_for_perm).reshape(resolution, resolution)

# Final comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. True porosity
im1 = axes[0, 0].contourf(x1_grid.numpy(), x2_grid.numpy(), true_porosity_grid.numpy(), levels=20, cmap='viridis')
axes[0, 0].scatter(X_current[:, 0].numpy(), X_current[:, 1].numpy(), color='red', s=50, marker='x')
axes[0, 0].set_title('True Porosity')
plt.colorbar(im1, ax=axes[0, 0])

# 2. Predicted porosity
im2 = axes[0, 1].contourf(x1_grid.numpy(), x2_grid.numpy(), mean_porosity.numpy(), levels=20, cmap='viridis')
axes[0, 1].scatter(X_current[:, 0].numpy(), X_current[:, 1].numpy(), color='red', s=50, marker='x')
axes[0, 1].set_title('Predicted Porosity')
plt.colorbar(im2, ax=axes[0, 1])

# 3. Porosity uncertainty
im3 = axes[0, 2].contourf(x1_grid.numpy(), x2_grid.numpy(), var_porosity.numpy(), levels=20, cmap='cividis')
axes[0, 2].scatter(X_current[:, 0].numpy(), X_current[:, 1].numpy(), color='red', s=50, marker='x')
axes[0, 2].set_title('Porosity Uncertainty')
plt.colorbar(im3, ax=axes[0, 2])

# 4. True permeability
im4 = axes[1, 0].contourf(x1_grid.numpy(), x2_grid.numpy(), true_permeability_grid.numpy(), levels=20, cmap='plasma')
axes[1, 0].scatter(X_current[:, 0].numpy(), X_current[:, 1].numpy(), color='red', s=50, marker='x')
axes[1, 0].set_title('True Permeability')
plt.colorbar(im4, ax=axes[1, 0])

# 5. Predicted permeability
im5 = axes[1, 1].contourf(x1_grid.numpy(), x2_grid.numpy(), mean_permeability.numpy(), levels=20, cmap='plasma')
axes[1, 1].scatter(X_current[:, 0].numpy(), X_current[:, 1].numpy(), color='red', s=50, marker='x')
axes[1, 1].set_title('Predicted Permeability')
plt.colorbar(im5, ax=axes[1, 1])

# 6. Permeability uncertainty
im6 = axes[1, 2].contourf(x1_grid.numpy(), x2_grid.numpy(), var_permeability.numpy(), levels=20, cmap='cividis')
axes[1, 2].scatter(X_current[:, 0].numpy(), X_current[:, 1].numpy(), color='red', s=50, marker='x')
axes[1, 2].set_title('Permeability Uncertainty')
plt.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
plt.close()

# Extract and print learned correlations
with torch.no_grad():
    task_covar_matrix = model.covar_module.task_covar_module.covar_factor.matmul(
        model.covar_module.task_covar_module.covar_factor.transpose(-1, -2)
    )
    task_corr_matrix = task_covar_matrix / torch.sqrt(torch.diag(task_covar_matrix).reshape(-1, 1) * 
                                                    torch.diag(task_covar_matrix).reshape(1, -1))
    
    print("Property Correlation Matrix:")
    print(task_corr_matrix.numpy())

# Calculate error metrics to evaluate the model quality
with torch.no_grad():
    # Generate true values for entire grid
    true_porosity_values = true_porosity(X_grid)
    porosity_for_perm = true_porosity(X_grid) + torch.randn(X_grid.shape[0]) * 0.01
    true_permeability_values = true_permeability(X_grid, porosity_for_perm)
    
    # Calculate Mean Absolute Error (MAE)
    porosity_mae = torch.mean(torch.abs(mean[:, 0] - true_porosity_values))
    permeability_mae = torch.mean(torch.abs(mean[:, 1] - true_permeability_values))
    
    print(f"Porosity MAE: {porosity_mae:.4f}")
    print(f"Permeability MAE: {permeability_mae:.4f}")

# Evaluate the quality of uncertainty estimates
with torch.no_grad():
    # Calculate standardized residuals
    std_porosity = torch.sqrt(variance[:, 0])
    std_permeability = torch.sqrt(variance[:, 1])
    
    z_porosity = (mean[:, 0] - true_porosity_values) / std_porosity
    z_permeability = (mean[:, 1] - true_permeability_values) / std_permeability
    
    # Plot histograms of standardized residuals
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(z_porosity.numpy(), bins=30, alpha=0.7)
    plt.title('Standardized Residuals - Porosity')
    plt.xlabel('Standardized Residual')
    plt.ylabel('Count')
    plt.axvline(x=-2, color='r', linestyle='--')
    plt.axvline(x=2, color='r', linestyle='--')
    
    plt.subplot(1, 2, 2)
    plt.hist(z_permeability.numpy(), bins=30, alpha=0.7)
    plt.title('Standardized Residuals - Permeability')
    plt.xlabel('Standardized Residual')
    plt.ylabel('Count')
    plt.axvline(x=-2, color='r', linestyle='--')
    plt.axvline(x=2, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.close()