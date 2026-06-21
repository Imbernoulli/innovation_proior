# Context: Online Bayesian Filtering Without Closed-Form Densities

## State-Space Problem

A hidden state evolves in discrete time, and only noisy measurements are observed:

$$
x_k = f_{k-1}(x_{k-1}, w_{k-1}), \qquad
y_k = h_k(x_k, v_k).
$$

The transition function, measurement function, prior distribution of the initial
state, and noise densities are known. The functions need not be linear, and the
noise need not be Gaussian. The target is the filtering distribution,

$$
p(x_k \mid y_{1:k}),
$$

updated online as each new observation arrives. A point estimate is not enough:
the full distribution is needed because the current uncertainty may be skewed,
curved, heavy-tailed, or multimodal.

## Exact Recursion

The formal Bayesian recursion has two stages. If
`p(x_{k-1} | y_{1:k-1})` is available, prediction pushes that distribution
through the state dynamics:

$$
p(x_k \mid y_{1:k-1}) =
\int p(x_k \mid x_{k-1}) p(x_{k-1} \mid y_{1:k-1})\,dx_{k-1}.
$$

The transition density is induced by the deterministic map and process noise:

$$
p(x_k \mid x_{k-1}) =
\int \delta(x_k - f_{k-1}(x_{k-1}, w_{k-1}))p(w_{k-1})\,dw_{k-1}.
$$

After observing `y_k`, Bayes' rule gives

$$
p(x_k \mid y_{1:k}) =
\frac{g(y_k \mid x_k)p(x_k \mid y_{1:k-1})}
{\int g(y_k \mid x_k)p(x_k \mid y_{1:k-1})\,dx_k}.
$$

The equations are exact, but the integrals and normalizer are usually not
available in closed form.

## Existing Baselines

The linear-Gaussian case is special. A Gaussian prior remains Gaussian after a
linear transition and Gaussian likelihood, so the distribution is represented by
mean and covariance and the Kalman equations close recursively.

The extended Kalman approach linearizes nonlinear functions around the current
estimate and still carries a single Gaussian. This is fragile when the true
posterior is not close to one Gaussian. In bearings-only tracking, range is weakly
identified from angle measurements, so the state distribution can be a curved
ridge or have multiple plausible regions. A local Gaussian approximation can
become overconfident and diverge.

Gaussian-sum filters improve expressiveness by using mixtures, but components
proliferate and must be merged or pruned. Grid filters approximate the density
directly on a fixed lattice, but the lattice does not automatically follow the
probability mass and the number of nodes grows rapidly with dimension. Earlier
accept-reject Monte Carlo updates keep a sample representation but can shrink the
sample size over time.

## Pre-Method Ingredients

A sample can stand in for a distribution. Expectations become sample averages,
region probabilities become fractions of sample points, and empirical summaries
converge as the sample size grows. This avoids committing to a Gaussian shape or
a fixed grid over all of state space.

Importance sampling gives a way to use samples from one distribution to estimate
expectations under another by assigning weights. Smith and Gelfand's
sampling-resampling perspective adds a second step: draw from the existing sample
with probabilities proportional to the importance weights, producing an
approximately unweighted sample from the tilted target. The target only needs to
be known up to proportionality, which matches Bayes' rule.

The caveat is already visible before any dynamic method is designed: if the
sampling distribution and target distribution have little overlap, a few weights
dominate and many samples contribute almost nothing.

## Starting Scaffold

The estimator receives a model interface with three operations:

```python
class StateSpaceModel:
    def sample_prior(self, n, rng):
        """Draw from p(x_1)."""

    def propagate(self, states, rng):
        """Draw from p(x_k | x_{k-1}) for each state."""

    def log_likelihood(self, y, states):
        """Evaluate log g(y_k | x_k) for each state."""
```

The missing design is the recursive representation and update rule. It must keep
an approximation to `p(x_k | y_{1:k})`, turn the prediction integral into a
bounded-cost operation, absorb each likelihood without computing the normalizing
integral, and avoid long-run collapse of the approximation.
