import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from mpl_toolkits.mplot3d import Axes3D

# Generate 2D sample data (well locations with x, y coordinates)
np.random.seed(42)
X = np.random.rand(20, 2) * 10  # 20 wells with (x,y) coordinates
# Create a ground truth function (e.g., porosity distribution)
def true_function(X):
    return np.sin(0.5 * X[:, 0]) * np.cos(0.5 * X[:, 1])

y = true_function(X) + np.random.normal(0, 0.1, X.shape[0])

# Create a grid for predictions
x_range = np.linspace(0, 10, 30)
y_range = np.linspace(0, 10, 30)
xx, yy = np.meshgrid(x_range, y_range)
X_pred = np.vstack([xx.ravel(), yy.ravel()]).T

# Define and fit GP
kernel = 1.0 * RBF(length_scale=[1.0, 1.0])  # Different length scales per dimension
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gp.fit(X, y)

# Make predictions
y_pred, sigma = gp.predict(X_pred, return_std=True)

# Reshape for plotting
y_pred_grid = y_pred.reshape(xx.shape)
sigma_grid = sigma.reshape(xx.shape)

# Plot the results
fig = plt.figure(figsize=(18, 6))

# True function (if we had complete knowledge)
ax1 = fig.add_subplot(131, projection='3d')
true_y = true_function(X_pred)
ax1.plot_surface(xx, yy, true_y.reshape(xx.shape), cmap='viridis', alpha=0.8)
ax1.scatter3D(X[:, 0], X[:, 1], y, color='red', s=50)
ax1.set_title('True Function')
ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')
ax1.set_zlabel('Property Value')

# Predicted mean
ax2 = fig.add_subplot(132, projection='3d')
surf = ax2.plot_surface(xx, yy, y_pred_grid, cmap='viridis', alpha=0.8)
ax2.scatter3D(X[:, 0], X[:, 1], y, color='red', s=50)
ax2.set_title('GP Predicted Mean')
ax2.set_xlabel('X coordinate')
ax2.set_ylabel('Y coordinate')
ax2.set_zlabel('Property Value')

# Uncertainty (standard deviation)
ax3 = fig.add_subplot(133)
contour = ax3.contourf(xx, yy, sigma_grid, cmap='cividis')
ax3.scatter(X[:, 0], X[:, 1], color='red', s=50)
ax3.set_title('Prediction Uncertainty (std dev)')
ax3.set_xlabel('X coordinate')
ax3.set_ylabel('Y coordinate')
plt.colorbar(contour, ax=ax3)

plt.tight_layout()
plt.close()