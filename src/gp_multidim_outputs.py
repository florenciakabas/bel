import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create synthetic multi-output data for our geological problem
n_samples = 20
X = torch.linspace(0, 10, n_samples).unsqueeze(-1)  # Spatial locations
y1 = torch.sin(X * 0.5) + torch.randn(n_samples, 1) * 0.1  # E.g., porosity
y2 = torch.cos(X * 0.5) + torch.sin(X * 0.5) + torch.randn(n_samples, 1) * 0.1  # E.g., permeability
Y = torch.cat([y1, y2], dim=-1)  # Combined outputs

# Scale inputs and outputs for better GP performance
X_mean, X_std = X.mean(), X.std()
X_normalized = (X - X_mean) / X_std
Y_mean, Y_std = Y.mean(dim=0), Y.std(dim=0)
Y_normalized = (Y - Y_mean) / Y_std

# Define a multi-output GP model
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iterations = 200
for i in range(n_iterations):
    optimizer.zero_grad()
    output = model(X_normalized)
    loss = -mll(output, Y_normalized)
    loss.backward()
    if i % 50 == 0:
        print(f'Iteration {i}/{n_iterations} - Loss: {loss.item()}')
    optimizer.step()

# Testing
model.eval()
likelihood.eval()

# Create test points
test_x = torch.linspace(0, 10, 100).unsqueeze(-1)
test_x_normalized = (test_x - X_mean) / X_std

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(test_x_normalized))
    mean = predictions.mean
    # Get lower and upper confidence bounds
    lower, upper = predictions.confidence_region()

# Denormalize predictions
mean = mean * Y_std + Y_mean
lower = lower * Y_std + Y_mean
upper = upper * Y_std + Y_mean

# Plot the results
plt.figure(figsize=(12, 8))

# First output (e.g., porosity)
plt.subplot(2, 1, 1)
plt.plot(test_x.numpy(), mean[:, 0].numpy(), 'b-', label='Predicted Mean')
plt.fill_between(test_x.numpy().flatten(), 
                 lower[:, 0].numpy(), 
                 upper[:, 0].numpy(), 
                 alpha=0.2, color='blue',
                 label='95% confidence interval')
plt.scatter(X.numpy(), Y[:, 0].numpy(), color='red', label='Observations')
plt.title('Property 1 (e.g., Porosity)')
plt.legend()

# Second output (e.g., permeability)
plt.subplot(2, 1, 2)
plt.plot(test_x.numpy(), mean[:, 1].numpy(), 'g-', label='Predicted Mean')
plt.fill_between(test_x.numpy().flatten(), 
                 lower[:, 1].numpy(), 
                 upper[:, 1].numpy(), 
                 alpha=0.2, color='green',
                 label='95% confidence interval')
plt.scatter(X.numpy(), Y[:, 1].numpy(), color='red', label='Observations')
plt.title('Property 2 (e.g., Permeability)')
plt.legend()

plt.tight_layout()
plt.show()