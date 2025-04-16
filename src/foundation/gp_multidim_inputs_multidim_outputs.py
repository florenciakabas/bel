import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Synthetic 2D exploration scenario
n_wells = 10
X = torch.tensor(np.random.uniform(0, 10, size=(n_wells, 2)), dtype=torch.float)

# Create synthetic geological properties with correlation
# Porosity
def true_porosity(x):
    return 0.2 * torch.exp(-0.1 * ((x[:, 0] - 5)**2 + (x[:, 1] - 5)**2)) + 0.1

# Permeability (related to porosity but with variation)
def true_permeability(x, porosity):
    # Permeability often has a non-linear relationship with porosity
    # (e.g., modified Kozeny-Carman relationship)
    return 1000 * porosity**3 / ((1 - porosity)**2) + torch.randn(porosity.size()) * 50

# Generate observations with noise
porosity = true_porosity(X) + torch.randn(n_wells) * 0.01
permeability = true_permeability(X, porosity)
Y = torch.stack([porosity, permeability], dim=1)

# Normalize data for better GP performance
X_mean, X_std = X.mean(dim=0), X.std(dim=0)
X_normalized = (X - X_mean) / X_std

Y_mean, Y_std = Y.mean(dim=0), Y.std(dim=0)
Y_normalized = (Y - Y_mean) / Y_std

# Define multi-output GP model with cross-correlations
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        
        # Mean module
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        
        # Base kernel (shared across tasks)
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)  # ARD allows different length scales per dimension
        )
        
        # Cross-task kernel with rank 2 (allows for more complex correlations)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.base_covar_module, num_tasks=2, rank=2
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskGPModel(X_normalized, Y_normalized, likelihood)

# Training
model.train()
likelihood.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.1)

# Loss function
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Train the model
n_iterations = 200
for i in range(n_iterations):
    optimizer.zero_grad()
    output = model(X_normalized)
    loss = -mll(output, Y_normalized)
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        print(f'Iteration {i+1}/{n_iterations} - Loss: {loss.item():.3f}')

# Set model to evaluation mode
model.eval()
likelihood.eval()

# Create a grid for predictions
resolution = 30
x1 = torch.linspace(0, 10, resolution)
x2 = torch.linspace(0, 10, resolution)
x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing='ij')
X_grid = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1)

# Normalize grid points
X_grid_normalized = (X_grid - X_mean) / X_std

# Predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(X_grid_normalized))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# Denormalize predictions
mean = mean * Y_std + Y_mean
lower = lower * Y_std + Y_mean
upper = upper * Y_std + Y_mean

# Reshape for plotting
mean_porosity = mean[:, 0].reshape(resolution, resolution)
mean_permeability = mean[:, 1].reshape(resolution, resolution)
std_porosity = (upper[:, 0] - lower[:, 0]).reshape(resolution, resolution) / 4  # Approximate std
std_permeability = (upper[:, 1] - lower[:, 1]).reshape(resolution, resolution) / 4

# Compute true values on the grid for comparison
X_grid_np = X_grid.numpy()
true_porosity_grid = true_porosity(X_grid).reshape(resolution, resolution)
porosity_for_perm = true_porosity(X_grid) + torch.randn(X_grid.shape[0]) * 0.01
true_permeability_grid = true_permeability(X_grid, porosity_for_perm).reshape(resolution, resolution)

# Plot
fig = plt.figure(figsize=(20, 12))

# 1. True Porosity
ax1 = fig.add_subplot(231, projection='3d')
surf1 = ax1.plot_surface(x1_grid.numpy(), x2_grid.numpy(), true_porosity_grid.numpy(), cmap='viridis', alpha=0.8)
ax1.scatter(X[:, 0].numpy(), X[:, 1].numpy(), porosity.numpy(), color='red', s=50)
ax1.set_title('True Porosity')
ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')
ax1.set_zlabel('Porosity')

# 2. Predicted Porosity
ax2 = fig.add_subplot(232, projection='3d')
surf2 = ax2.plot_surface(x1_grid.numpy(), x2_grid.numpy(), mean_porosity.numpy(), cmap='viridis', alpha=0.8)
ax2.scatter(X[:, 0].numpy(), X[:, 1].numpy(), porosity.numpy(), color='red', s=50)
ax2.set_title('Predicted Porosity')
ax2.set_xlabel('X coordinate')
ax2.set_ylabel('Y coordinate')
ax2.set_zlabel('Porosity')

# 3. Porosity Uncertainty
ax3 = fig.add_subplot(233)
contour3 = ax3.contourf(x1_grid.numpy(), x2_grid.numpy(), std_porosity.numpy(), cmap='cividis')
ax3.scatter(X[:, 0].numpy(), X[:, 1].numpy(), color='red', s=50)
ax3.set_title('Porosity Uncertainty (Std Dev)')
ax3.set_xlabel('X coordinate')
ax3.set_ylabel('Y coordinate')
plt.colorbar(contour3, ax=ax3)

# 4. True Permeability
ax4 = fig.add_subplot(234, projection='3d')
surf4 = ax4.plot_surface(x1_grid.numpy(), x2_grid.numpy(), true_permeability_grid.numpy(), cmap='plasma', alpha=0.8)
ax4.scatter(X[:, 0].numpy(), X[:, 1].numpy(), permeability.numpy(), color='red', s=50)
ax4.set_title('True Permeability (mD)')
ax4.set_xlabel('X coordinate')
ax4.set_ylabel('Y coordinate')
ax4.set_zlabel('Permeability (mD)')

# 5. Predicted Permeability
ax5 = fig.add_subplot(235, projection='3d')
surf5 = ax5.plot_surface(x1_grid.numpy(), x2_grid.numpy(), mean_permeability.numpy(), cmap='plasma', alpha=0.8)
ax5.scatter(X[:, 0].numpy(), X[:, 1].numpy(), permeability.numpy(), color='red', s=50)
ax5.set_title('Predicted Permeability (mD)')
ax5.set_xlabel('X coordinate')
ax5.set_ylabel('Y coordinate')
ax5.set_zlabel('Permeability (mD)')

# 6. Permeability Uncertainty
ax6 = fig.add_subplot(236)
contour6 = ax6.contourf(x1_grid.numpy(), x2_grid.numpy(), std_permeability.numpy(), cmap='cividis')
ax6.scatter(X[:, 0].numpy(), X[:, 1].numpy(), color='red', s=50)
ax6.set_title('Permeability Uncertainty (Std Dev)')
ax6.set_xlabel('X coordinate')
ax6.set_ylabel('Y coordinate')
plt.colorbar(contour6, ax=ax6)

plt.tight_layout()
plt.close()

# Calculate cross-correlation
with torch.no_grad():
    # Extract the task correlation matrix
    # task_covar_matrix = model.covar_module.task_covar_module.covar_matrix.evaluate()
    task_covar_matrix = model.covar_module.task_covar_module.covar_factor.matmul(
        model.covar_module.task_covar_module.covar_factor.transpose(-1, -2)
    )
    # Convert to correlation matrix
    task_corr_matrix = task_covar_matrix / torch.sqrt(torch.diag(task_covar_matrix).reshape(-1, 1) * 
                                                     torch.diag(task_covar_matrix).reshape(1, -1))
    
    print("Property Correlation Matrix:")
    print(task_corr_matrix.numpy())