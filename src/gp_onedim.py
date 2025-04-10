import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from plot_styler import PlotStyler
# Initialize the PlotStyler with the configuration file
styler = PlotStyler()

# Create some sample data points (pretend these are measurements from existing wells)
X = np.array([[1.0], [3.0], [5.0], [6.0], [7.0], [8.0]]).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create x values for prediction (the entire field we're interested in)
X_pred = np.linspace(0, 10, 100).reshape(-1, 1)

# Different kernels
ls = 0.5 #length scale
rbf_kernel = RBF(length_scale=ls)
matern_kernel = Matern(length_scale=ls, nu=1.5)

def main_plot(X_pred, y_pred, sigma, N:float=1.96) -> None:
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X_pred, y_pred, color=styler.get_color('line_color'), label='Prediction')
    ax.fill_between(
        X_pred.ravel(),
        y_pred - N * sigma,
        y_pred + N * sigma,
        alpha=0.2,
        color=styler.get_color('fill_color'),
        label='95% confidence interval'
    )
    ax.scatter(X, y, color=styler.get_color('scatter_color'), label='Observations')
    styler.apply_style(ax, title=f'Before Training - Gaussian Process Regression with {kernel} Kernel',
                       xlabel='Location', ylabel='Property Value')
    ax.legend()
    plt.ylim(-5, 5)  # Set y-axis limits

def predict(gp, X_pred, plot:bool=True):
    y_pred, sigma = gp.predict(X_pred, return_std=True)
    if plot:
        main_plot(X_pred, y_pred, sigma)

for kernel in [rbf_kernel, matern_kernel]:
    # Create the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
    # Make predictions
    # Note: This is before fitting the model to the data
    predict(gp, X_pred)
    # Fit the GP to our data
    gp.fit(X, y)
    # Make predictions
    # Note: This is after fitting the model to the data
    predict(gp, X_pred)

# Create a grid of points
x = np.linspace(-5, 5, 100).reshape(-1, 1)
x0 = np.array([[0]])  # Reference point

# Compute kernel values
k_rbf = rbf_kernel(x, x0)
k_matern = matern_kernel(x, x0)

# Plot kernel functions
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, k_rbf, color=styler.get_color('line_color'), label='RBF Kernel')
ax.plot(x, k_matern, color=styler.get_color('scatter_color'), label='Matern Kernel (nu=1.5)')
ax.axvline(x=0, color='k', linestyle='--')
styler.apply_style(ax, title='Kernel Functions (Correlation with point x=0)',
                   xlabel='x', ylabel='k(x, 0)')
ax.legend()
plt.show()