
# Gaussian Processes for Better Informed Exploration

## Introduction
This repo implements a specialized, practical version of BEL using Gaussian Processes as the underlying statistical model.
This implementation provides a systematic, data-driven approach to exploration well planning using advanced statistical methods called Gaussian Processes.
Please find a brief intro to Gaussian Processes at the bottom of the documentation. They are beautiful.

## Summary
Here's a high-level summary of what happens in the main script (gp_mvp.py):

1. Basin Modeling

Creates a statistical model of subsurface properties (porosity, permeability, thickness) based on limited initial data
Captures spatial correlations and relationships between different properties
Quantifies uncertainty in our knowledge of these properties across the basin

2. Sequential Decision Making

Evaluates thousands of potential well locations to find the optimal next drilling target
Uses smart "acquisition functions" that balance:

Exploring highly uncertain areas to gather more information
Targeting areas with high expected economic value
Considering the correlations between multiple properties

3. Value of Information Analysis

Calculates the expected economic value of drilling each potential well
Simulates how new data would update our understanding of the basin
Identifies wells that would provide the most valuable information for decision-making

4. Risk Management

Provides probability distributions rather than single estimates
Quantifies confidence levels for economic projections
Helps determine how many wells are needed to reach desired confidence threshold

5. Adaptive Learning

Continuously updates the basin model as new wells are drilled
Refines exploration strategy based on all available data
Optimizes the exploration campaign to maximize return on investment

This approach allows us to:

Reduce exploration costs through optimal well placement
Increase the probability of discovering economically viable resources
Make data-driven decisions with clear quantification of uncertainties
Systematically build knowledge of the basin with each well




## Extra - Gaussian Processes

At a high level, a Gaussian Process (GP) is a powerful non-parametric Bayesian approach to modeling functions. Instead of parameterizing a specific function (like with linear regressors or neural networks), GPs define a probability distribution over possible functions that could fit the data.

### What is a Gaussian Process?

Just as a multivariate Gaussian is fully defined by:
- A **mean vector**
- A **covariance matrix**

A Gaussian Process is fully defined by:
- A **mean function**: \( m(x) \)
- A **covariance function** (also called a kernel): \( k(x, x') \)

### Key Concepts

1. **Mean Function**: Represents the expected value of the function at any input \( x \).
   \[
   m(x) = \mathbb{E}[f(x)]
   \]

2. **Covariance Function (Kernel)**: Represents the relationship between function values at different inputs \( x \) and \( x' \).
   \[
   k(x, x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]
   \]

3. **Infinite-Dimensional Functions**: Unlike parametric models, GPs do not assume a fixed number of parameters. Instead, they define a distribution over functions.