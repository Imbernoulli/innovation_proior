## Research Question

In data modelling, the fitting problem and the comparison problem are not the same. For one chosen family of functions, fitting asks which parameter values best explain the observed data and how uncertain those parameter values remain. That task is familiar: write a likelihood, add a prior or regularizer if needed, find the most plausible parameter vector, and use curvature for error bars.

A separate task begins after several model families have all been fitted. A noisy data set may be interpolated by polynomials of different degrees, radial basis functions of different densities, splines with different smoothness assumptions, or a small neural network. Each family may also contain a regularization strength and a noise scale. The question is how to rank these alternatives using the data.

## Background

Bayesian inference supplies a calculus for degrees of plausibility over a defined hypothesis space. At the parameter level, one writes the posterior distribution for parameters inside a model as proportional to likelihood times prior. In ordinary parameter fitting, any factor independent of the parameters can be ignored because it does not move the optimum or change the local curvature around that optimum.

Bayesian testing and model comparison have an older tradition in Jeffreys's work, where estimation and testing are treated as distinct inferential tasks. Comparing models is a different question from estimating parameters inside one model.

Two mathematical ingredients are available. First, a sharply peaked integral can be approximated by expanding the log integrand to second order at its peak and integrating the resulting Gaussian. In `k` dimensions, the Gaussian volume is controlled by `(2*pi)^(k/2)` times the inverse square root of a Hessian determinant. Second, quadratic regularization has a Bayesian interpretation: penalizing large weights is equivalent to putting a Gaussian prior on those weights.

## Baselines

Maximum-likelihood model choice ranks each model by its best fit. The best fit is monotone in flexibility for nested or nearly nested families.

Penalized likelihood criteria add complexity costs such as a term proportional to the number of parameters, sometimes also depending on sample size. The penalty is fixed outside the geometry of the fitted posterior and the prior scale used by the model.

Description-length approaches encode the model and data in bits. They are closely related to probabilistic model comparison; a practical two-part code decides how accurately to communicate parameter values.

Capacity measures such as worst-case function-class dimension depend on the function class but not on the particular data set, the prior width of the parameters, or which parameter directions the data measure.

Validation methods reserve data or repeatedly refit on subsets. They estimate predictive performance from held-out or resampled data.

## Evaluation Settings

The cleanest setting is noisy one-dimensional interpolation. The observed data are pairs `(x_m, t_m)`, and the target values are corrupted by additive Gaussian noise. Candidate models include fixed-basis linear expansions, radial basis functions, polynomial bases, and splines with different smoothness penalties.

For a fixed model and fixed hyperparameters, the first-level fit is straightforward. With design matrix `Phi`, Gaussian noise precision `beta`, and quadratic weight penalty strength `alpha`, the objective is

```text
M(w) = alpha * E_w(w) + beta * E_n(w)
```

where `E_w = 0.5 * ||w||^2` and `E_n = 0.5 * ||Phi @ w - t||^2` in the simplest whitened case. The Hessian is

```text
A = alpha * I + beta * Phi.T @ Phi
```

and the fitted parameter vector is obtained by solving a linear system. The same Hessian gives parameter error bars.
