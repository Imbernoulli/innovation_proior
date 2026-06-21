# Bootstrap Particle Filter

Maintain the filtering distribution as a weighted empirical measure:

$$
\hat p(dx_k\mid y_{1:k})=\sum_{i=1}^N W_k^i\delta_{x_k^i}(dx_k).
$$

Initialize with `x_1^i ~ p(x_1)` and equal weights. For each observation:

1. Predict by simulation:

$$
x_k^{i,*}=f_{k-1}(x_{k-1}^i,w_{k-1}^i), \qquad
w_{k-1}^i\sim p(w_{k-1}).
$$

2. Weight by the likelihood:

$$
\tilde W_k^i = W_{k-1}^i g(y_k\mid x_k^{i,*}), \qquad
W_k^i=\frac{\tilde W_k^i}{\sum_j\tilde W_k^j}.
$$

With the transition-prior proposal, the incremental importance weight is just
the likelihood. The intractable Bayes normalizer is replaced by the finite
normalization over particles.

3. Diagnose degeneracy:

$$
\mathrm{ESS}=\left(\sum_i (W_k^i)^2\right)^{-1}.
$$

4. If `ESS < N/2`, resample `N` particles with probabilities `W_k^i`, reset
weights to `1/N`, and optionally roughen:

$$
x_k^i \leftarrow x_k^i+\epsilon_i,\qquad
\epsilon_i\sim N(0,J_k),\qquad
\sigma_j=K E_j N^{-1/d}.
$$

Here `E_j` is the current range of component `j`, `d` is state dimension, and
`K` is a small tuning constant.

The method requires only:

- sampling the initial prior,
- simulating the transition with fresh process noise,
- evaluating the observation likelihood.

It represents nonlinear and non-Gaussian filtering distributions directly as
weighted particles, and resampling prevents the sequential importance weights
from collapsing onto a tiny number of particles.

## Minimal Python Implementation

```python
import numpy as np

def effective_sample_size(weights):
    return 1.0 / np.sum(weights ** 2)

def resample(particles, weights, rng):
    indices = rng.choice(len(weights), size=len(weights), p=weights)
    return particles[indices], np.ones_like(weights) / len(weights)

def particle_filter(particles, weights, observation, model, rng):
    # Predict: push particles through the state dynamics.
    predicted = model.propagate(particles, rng)

    # Update: weight by the observation likelihood.
    log_w = model.log_likelihood(observation, predicted)
    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()

    # Resample if the effective sample size falls below half the particle count.
    if effective_sample_size(weights) < 0.5 * len(weights):
        predicted, weights = resample(predicted, weights, rng)

    return predicted, weights
```

The executable artifact is [bootstrap_particle_filter.py](/srv/home/bohanlyu/innovation_proior/methods/particle-filter-smc/code/bootstrap_particle_filter.py).
