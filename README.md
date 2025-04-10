# Gaussian Processes

At a high level, a Gaussian Process (GP) is a powerful non-parametric Bayesian approach to modeling functions. Instead of parameterizing a specific function (like with linear regressors or neural networks), GPs define a probability distribution over possible functions that could fit the data.

## What is a Gaussian Process?

Just as a multivariate Gaussian is fully defined by:
- A **mean vector**
- A **covariance matrix**

A Gaussian Process is fully defined by:
- A **mean function**: \( m(x) \)
- A **covariance function** (also called a kernel): \( k(x, x') \)

## Key Concepts

1. **Mean Function**: Represents the expected value of the function at any input \( x \).
   \[
   m(x) = \mathbb{E}[f(x)]
   \]

2. **Covariance Function (Kernel)**: Represents the relationship between function values at different inputs \( x \) and \( x' \).
   \[
   k(x, x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]
   \]

3. **Infinite-Dimensional Functions**: Unlike parametric models, GPs do not assume a fixed number of parameters. Instead, they define a distribution over functions.