I want the filtering distribution $p(x_k \mid y_{1:k})$ for a hidden state that evolves according to $x_k = f_{k-1}(x_{k-1}, w_{k-1})$ and is observed only through noisy measurements $y_k = h_k(x_k, v_k)$. A point estimate is not enough for this problem, because the posterior uncertainty can be skewed, curved, heavy-tailed, or multimodal. The exact Bayesian recursion is simple to state: prediction pushes the previous posterior through the dynamics, $p(x_k \mid y_{1:k-1}) = \int p(x_k \mid x_{k-1}) p(x_{k-1} \mid y_{1:k-1}) \, dx_{k-1}$, and the new observation tilts the result, $p(x_k \mid y_{1:k}) \propto g(y_k \mid x_k) p(x_k \mid y_{1:k-1})$. The difficulty is not the probability calculus; it is that once the dynamics or observations are nonlinear or the noises are non-Gaussian, the integrals and the Bayes normalizer have no closed form.

The usual closed-form shortcuts fail exactly where the problem is interesting. In the linear-Gaussian case the posterior remains Gaussian, so mean and covariance close the recursion and the Kalman filter is pure algebra on those moments. The extended Kalman filter linearizes around the current estimate and still carries a single Gaussian, which becomes fragile when the true posterior is not close to one Gaussian. In bearings-only tracking, for example, range is weakly identified from angle measurements, so the density can become a curved ridge or develop several plausible regions, and a local Gaussian summary can become overconfident and diverge. Gaussian-sum filters add expressiveness through mixtures, but components proliferate and must be merged or pruned. Grid filters approximate the density honestly, but the lattice does not follow the moving probability mass and the number of nodes explodes with state dimension.

I propose the Bootstrap Particle Filter as the canonical way to carry the full filtering distribution without committing to a parametric shape. Instead of storing a formula for $p(x_k \mid y_{1:k})$, I store a finite empirical measure: particles $x_k^1, \dots, x_k^N$ with weights $W_k^1, \dots, W_k^N$, so that
$$
\hat p(dx_k \mid y_{1:k}) = \sum_{i=1}^N W_k^i \delta_{x_k^i}(dx_k)
$$
and any posterior expectation is read as $\sum_i W_k^i \varphi(x_k^i)$. With equal weights this is an ordinary sample average; with unequal weights the cloud can place mass in several regions at once. The object carried through time is therefore not a state estimate and not a Gaussian approximation, but an approximate filtering distribution.

Prediction now becomes a Monte Carlo realization of the Chapman-Kolmogorov integral. The transition density is defined by pushing process noise through the deterministic map, so if the current particles approximate $p(x_{k-1} \mid y_{1:k-1})$, drawing a fresh noise $w_{k-1}^i \sim p(w_{k-1})$ for each particle and applying the map,
$$
x_k^{i,*} = f_{k-1}(x_{k-1}^i, w_{k-1}^i),
$$
gives a draw from the predictive marginal. The integral over all of state space turns into one propagation per particle.

The update is where Bayes' rule meets the empirical measure. The predicted particles approximate $p(x_k \mid y_{1:k-1})$, and the new observation asks me to tilt that distribution by the likelihood. I assign each predicted particle an unnormalized weight and renormalize across the finite cloud,
$$
\tilde W_k^i = W_{k-1}^i g(y_k \mid x_k^{i,*}), \qquad
W_k^i = \frac{\tilde W_k^i}{\sum_j \tilde W_k^j}.
$$
The intractable Bayes normalizer collapses into the finite sum in the denominator. The reason the incremental weight is just the likelihood is the proposal choice. The minimum-variance proposal would sample from $p(x_k \mid y_k, x_{k-1}) \propto g(y_k \mid x_k) f(x_k \mid x_{k-1})$, because then the incremental weight would not depend on the sampled $x_k$, but sampling that distribution demands the very integral I am trying to avoid. Proposing instead from the transition prior $f(x_k \mid x_{k-1})$ makes the transition density cancel, leaving only the likelihood. This reduces the model interface to three operations: sample the prior, propagate through the dynamics, and evaluate the observation likelihood.

Carrying weighted particles forward without further action works for one update but fails over many steps. Each weight is a running product of incremental factors, so any mismatch between proposal and target compounds, and after enough updates nearly all normalized weight sits on one or a few trajectories. To see this collapse happening I monitor the effective sample size,
$$
\mathrm{ESS} = \left(\sum_i (W_k^i)^2\right)^{-1},
$$
which equals $N$ for equal weights and $1$ when a single weight is one. It measures degeneracy as a concrete concentration of mass onto too few particles rather than a vague loss of quality.

The cure is resampling, and it is the load-bearing stability device rather than a decorative extra. Using the weighted-bootstrap idea, I draw $N$ offspring from the current particles with probabilities $W_k^i$ and reset the survivors to equal weight $1/N$. This is Bayes' rule executed as selection and copying: particles that explain the observation are duplicated, and particles far from the likelihood vanish. Crucially, resampling severs the product of weights. Without it, a small weight today haunts a particle forever; with it, the weighted measure is replaced by an equally weighted one whose expected offspring counts are proportional to the weights. The next prediction then starts from a cloud already concentrated where the posterior mass is, rather than from many dead near-zero-weight particles. Resampling creates duplicates and erodes diversity in old path components, which matters for smoothing trajectories, but for filtering the current marginal it is tolerable because each prediction injects fresh process noise into the current state. I resample only when $\mathrm{ESS} < N/2$, so I do not pay the extra Monte Carlo noise when the weights are already healthy.

One finite-sample failure remains: if the likelihood barely overlaps the predictive cloud, only a few predicted particles receive meaningful weight, resampling copies them many times, and if the process noise is small those copies stay nearly identical. The response is roughening: after resampling, add small jitter $x_k^i \leftarrow x_k^i + \epsilon_i$ with $\epsilon_i \sim N(0, J_k)$, where the component scales shrink with particle count as $N^{-1/d}$. This breaks exact duplicates while vanishing as the cloud becomes denser.

What emerges is therefore a specific answer to the filtering recursion rather than generic sequential importance resampling pseudocode: prediction is simulation of the transition integral, update is likelihood weighting of the predictive empirical measure, and resampling is what prevents sequential importance weights from collapsing. Because the method never assumes a closed parametric form for the posterior, it remains valid for nonlinear dynamics, nonlinear measurements, and non-Gaussian noise.

```python
import numpy as np


class SimpleRandomWalkModel:
    def __init__(self, transition_std=0.5, likelihood_std=1.0, rng=None):
        self.transition_std = transition_std
        self.likelihood_std = likelihood_std
        self.rng = rng or np.random.default_rng(0)

    def sample_prior(self, n, rng):
        return rng.normal(loc=0.0, scale=1.0, size=n)

    def propagate(self, particles, rng):
        return particles + rng.normal(scale=self.transition_std, size=particles.shape)

    def log_likelihood(self, y, particles):
        return -0.5 * ((y - particles) / self.likelihood_std) ** 2


def effective_sample_size(weights):
    return 1.0 / np.sum(weights ** 2)


def resample(particles, weights, rng):
    idx = rng.choice(len(weights), size=len(weights), p=weights)
    return particles[idx], np.ones_like(weights) / len(weights)


def particle_filter_step(particles, weights, y, model, rng):
    predicted = model.propagate(particles, rng)
    log_w = model.log_likelihood(y, predicted)
    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()
    if effective_sample_size(weights) < 0.5 * len(weights):
        predicted, weights = resample(predicted, weights, rng)
    return predicted, weights


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    model = SimpleRandomWalkModel(rng=rng)
    n = 1000
    particles = model.sample_prior(n, rng)
    weights = np.full(n, 1.0 / n)

    true_states = [0.0]
    observations = []
    for _ in range(50):
        true_states.append(true_states[-1] + rng.normal(scale=model.transition_std))
        observations.append(true_states[-1] + rng.normal(scale=model.likelihood_std))

    estimates = []
    for k, y in enumerate(observations):
        if k > 0:
            particles = model.propagate(particles, rng)
        particles, weights = particle_filter_step(particles, weights, y, model, rng)
        estimates.append(np.average(particles, weights=weights))

    rmse = np.sqrt(np.mean([(e - t) ** 2 for e, t in zip(estimates, true_states[1:])]))
    print(f"RMSE over 50 steps: {rmse:.3f}")
```
