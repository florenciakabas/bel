import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from plot_styler import PlotStyler
# Initialize the PlotStyler with the configuration file
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'plot_config.json')
styler = PlotStyler(config_path=config_path)

def rbf_kernel(X1, X2, length_scale=1.0, sigma=1.0):
    """RBF kernel function."""
    # Compute pairwise squared Euclidean distances
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma**2 * np.exp(-0.5 * sqdist / length_scale**2)

def gp_posterior(X_train, y_train, X_test, kernel_func, noise=0.1):
    """Calculate the GP posterior mean and covariance."""
    # Kernel matrices
    K = kernel_func(X_train, X_train) + noise**2 * np.eye(len(X_train))
    K_s = kernel_func(X_train, X_test)
    K_ss = kernel_func(X_test, X_test)
    
    # Calculate posterior mean
    K_inv = np.linalg.inv(K)
    mu_s = K_s.T @ K_inv @ y_train
    
    # Calculate posterior covariance
    sigma_s = K_ss - K_s.T @ K_inv @ K_s
    
    return mu_s, sigma_s

# Sample data (wells)
X = np.array([1.0, 3.0, 5.0, 6.0, 7.0, 8.0])
y = np.sin(X) + np.random.normal(0, 0.1, X.shape[0])

# Prediction points (entire field)
X_test = np.linspace(0, 10, 100)

# Fixed hyperparameters
length_scale = 1.0
sigma = 1.0
noise = 0.1

# Calculate posterior
mu_s, sigma_s = gp_posterior(X, y, X_test, lambda x1, x2: rbf_kernel(x1, x2, length_scale=length_scale, sigma=sigma), noise=0.1)

# Extract standard deviation
sigma = np.sqrt(np.diag(sigma_s))


# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X_test, mu_s, color=styler.get_color('line_color'), label='Posterior Mean')
ax.fill_between(X_test.ravel(), 
                 mu_s - 1.96 * sigma,
                 mu_s + 1.96 * sigma,
                 alpha=0.2,
                 color=styler.get_color('fill_color'),
                 label='95% confidence interval')
ax.scatter(X, y, color=styler.get_color('scatter_color'), label='Observations (Wells)')
styler.apply_style(ax, title=f'Gaussian Process Posterior, from scratch posterior update, fixed hyperparameters',
                    xlabel='Location', ylabel='Property Value')
plt.ylim(-5, 5)  # Set y-axis limits






# Approach 2: Using scikit-learn's implementation
# Create some sample data points (pretend these are measurements from existing wells)
X = np.array([[1.0], [3.0], [5.0], [6.0], [7.0], [8.0]]).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create x values for prediction (the entire field we're interested in)
X_test = np.linspace(0, 10, 100).reshape(-1, 1)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Create kernel and GP model
kernel = 1.0 * RBF(length_scale=length_scale)  # Fixed hyperparameters for comparison
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, optimizer=None)  # No optimization

# Fit the model (mainly computing required matrices for prediction)
gp.fit(X, y)

# Make predictions
mu_pred, sigma_pred = gp.predict(X_test, return_std=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X_test, mu_pred, color=styler.get_color('line_color'), label='Posterior Mean')
ax.fill_between(X_test.ravel(), 
                 mu_pred - 1.96 * sigma_pred,
                 mu_pred + 1.96 * sigma_pred,
                 alpha=0.2,
                 color=styler.get_color('fill_color'),
                 label='95% confidence interval')
ax.scatter(X, y, color=styler.get_color('scatter_color'), label='Observations (Wells)')
styler.apply_style(ax, title=f'Gaussian Process Posterior, gp.fit not optimized, fixed hyperparameters',
                    xlabel='Location', ylabel='Property Value')
plt.ylim(-5, 5)  # Set y-axis limits



# Now with hyperparameter optimization
kernel = 1.0 * RBF(length_scale=1.0)  # Initial values
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)  # Will optimize

# Fit with optimization
gp.fit(X, y)

# Now predictions use optimized hyperparameters
mu_pred, sigma_pred = gp.predict(X_test, return_std=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X_test, mu_pred, color=styler.get_color('line_color'), label='Posterior Mean')
ax.fill_between(X_test.ravel(), 
                 mu_pred - 1.96 * sigma_pred,
                 mu_pred + 1.96 * sigma_pred,
                 alpha=0.2,
                 color=styler.get_color('fill_color'),
                 label='95% confidence interval')
ax.scatter(X, y, color=styler.get_color('scatter_color'), label='Observations (Wells)')
styler.apply_style(ax, title=f'Gaussian Process Posterior, gp.fit optimized',
                    xlabel='Location', ylabel='Property Value')

plt.ylim(-5, 5)  # Set y-axis limits
plt.show()