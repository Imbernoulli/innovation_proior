# Context: recursive Bayesian state estimation in nonlinear / non-Gaussian models

## Research question

A system evolves in discrete time. Its state vector $x_k \in \mathbb{R}^n$ moves according to a
known stochastic dynamics, and we never see it directly — we see noisy measurements
$y_k \in \mathbb{R}^p$ that depend on it. Concretely,

$$x_k = f_{k-1}(x_{k-1}, w_{k-1}), \qquad y_k = h_k(x_k, v_k),$$

where $f$ and $h$ are known (possibly nonlinear) functions and $w_k, v_k$ are independent
white-noise sequences with **known** probability densities (not necessarily Gaussian). We are also
given the prior $p(x_1)$.

The goal is online, recursive estimation of the **filtering distribution**: the full posterior
density of the current state given every measurement so far,

$$p(x_k \mid y_{1:k}), \qquad D_k = \{y_1, \dots, y_k\},$$

updated each time a new $y_k$ arrives, at a cost per step that does not grow with $k$. Having the
whole density — not just a point estimate — is what matters: it lets us report the mean, the
covariance, percentiles, highest-posterior-density regions, the probability of the state lying in
any region, and to do so even when the posterior is multimodal or sharply skewed. The difficulty:
for nonlinear $f, h$ or non-Gaussian noise there is **no closed-form expression** for this density,
so it must be represented and propagated approximately, by a scheme that stays faithful over
arbitrarily many time steps.

## Background

**The formal recursion.** The filtering posterior obeys a two-stage recursion that is exact and
model-free. Given $p(x_{k-1} \mid D_{k-1})$:

- **Prediction** (Chapman–Kolmogorov), pushing the posterior forward through the dynamics,

$$p(x_k \mid D_{k-1}) = \int p(x_k \mid x_{k-1})\, p(x_{k-1} \mid D_{k-1})\, dx_{k-1},$$

where the transition density follows from the system equation and the known noise statistics. Since
$x_k = f_{k-1}(x_{k-1}, w_{k-1})$ is deterministic once $x_{k-1}$ and $w_{k-1}$ are fixed,

$$p(x_k \mid x_{k-1}) = \int \delta\big(x_k - f_{k-1}(x_{k-1}, w_{k-1})\big)\, p(w_{k-1})\, dw_{k-1}.$$

- **Update** (Bayes), folding in the new measurement,

$$p(x_k \mid D_k) = \frac{p(y_k \mid x_k)\, p(x_k \mid D_{k-1})}{p(y_k \mid D_{k-1})}, \qquad
p(y_k \mid D_{k-1}) = \int p(y_k \mid x_k)\, p(x_k \mid D_{k-1})\, dx_k,$$

with the likelihood $p(y_k \mid x_k) = \int \delta(y_k - h_k(x_k, v_k)) p(v_k)\, dv_k$ defined by the
measurement model. These two relations are the whole problem; everything else is how to *compute*
them.

**The one tractable case.** When $f, h$ are linear and $w, v$ are additive Gaussian, the posterior
stays Gaussian at every step, so it is fully described by its mean and covariance, and the recursion
closes in finite dimension — this is the Kalman filter (see Baselines). The trouble is that "linear
and Gaussian" is a knife-edge. The moment the dynamics or the measurement bends, or the noise has
tails, or the posterior wants to be multimodal, the Gaussian family can no longer hold the true
density and the closed form is gone.

**The duality between a density and a sample.** A density generates samples; conversely, a sample
approximately recreates the density — as a histogram, a kernel estimate, an empirical CDF. A set of
draws $\{x^{(i)}\}_{i=1}^N$ from a density carries the same information as the density itself in the
limit $N \to \infty$: any expectation $\int \varphi(x) p(x) dx$ is estimated by
$\frac{1}{N}\sum_i \varphi(x^{(i)})$, with error decaying as $O(1/\sqrt N)$ *regardless of the
dimension* of the space. This is the standard Monte Carlo fact about samples standing in for
densities.

**Reweighting one sample into another (importance sampling).** If draws are easy from a density
$g$ but we want a sample from $h \propto f$ (a positive function $f$, normaliser unknown), importance
sampling weights each draw by $f(x)/g(x)$: for any test function,
$\int \varphi h \,\approx\, \sum_i W^{(i)} \varphi(x^{(i)})$ with normalised weights
$W^{(i)} \propto f(x^{(i)})/g(x^{(i)})$. The normaliser of $h$ never has to be computed — only the
ratio. The catch, knowable from the variance of the weights, is that the estimate is only as good as
the match between $g$ and $h$: the more $h$ differs from $g$, the more the weight concentrates on a
few draws, and the larger $N$ must be.

**Sampling–resampling (the weighted bootstrap).** Smith & Gelfand (1992), "Bayesian statistics
without tears," turn weights back into an unweighted sample. Given draws $\{\theta_i\}$ from $g$ and
a target $h \propto f$, form $\omega_i = f(\theta_i)/g(\theta_i)$, normalise $q_i = \omega_i / \sum_j
\omega_j$, and draw $\theta^*$ from the discrete distribution that puts mass $q_i$ on $\theta_i$.
Then $\theta^*$ is approximately distributed as $h$, with the approximation improving as $N$ grows;
their CDF argument shows
$\Pr(\theta^* \le a) \to \int_{-\infty}^a h$. This is "weighted resampling": the ordinary bootstrap
resamples with equal probability, here the probabilities are tilted by $f/g$. It needs $h$ only up
to proportionality, where $h \propto f$ with $f$ a known unnormalised function and the normaliser
unavailable. The same caveat reappears: the less $h$ resembles $g$, the larger $N$ needed.
Rubin (1988) had named this the SIR (sampling/importance-resampling) algorithm.

**The empirical motivation: where the Gaussian baseline fails.** The diagnostic failure that sets
up the problem is bearings-only tracking. A target moves in the plane under a second-order motion
model; a fixed observer sees only the bearing $z_k = \arctan(y_k/x_k) + v_k$. Bearing alone does not
pin down range, so the posterior over position is not even approximately Gaussian — it is a curved,
ridge-like, sometimes bimodal distribution. A filter that *forces* a Gaussian posterior reports a
mean and an ellipse that sit off the true ridge, becomes over-confident, and diverges, especially
around the fly-past where the informative measurements arrive. This known pathology of linearised
Gaussian filtering is the concrete reason a non-Gaussian representation is wanted.

## Baselines

- **Kalman filter** (Kalman 1960; Ho & Lee 1964; Harrison & Stevens 1976). The exact recursion for
  the linear-Gaussian model $x_k = Ax_{k-1} + w_{k-1}$, $y_k = Hx_k + v_k$, $w\sim N(0,Q)$,
  $v\sim N(0,R)$. It propagates only a mean and covariance:
  - *time update* (predict): $\hat x_k^- = A\hat x_{k-1}$, $P_k^- = A P_{k-1} A^\top + Q$;
  - *measurement update* (correct): $K_k = P_k^- H^\top (H P_k^- H^\top + R)^{-1}$,
    $\hat x_k = \hat x_k^- + K_k (y_k - H\hat x_k^-)$, $P_k = (I - K_k H) P_k^-$.
  Core idea: the posterior is Gaussian for all $k$, so two moments suffice and the update is a
  linear blend of prediction and innovation weighted by the gain. **Gap:** exact only when the model
  is linear and the noise Gaussian. Outside that knife-edge the posterior is non-Gaussian and the
  two-moment representation is wrong.

- **Extended Kalman filter (EKF)** (Jazwinski 1970). The dominant approach to nonlinear estimation:
  linearise $f, h$ about the current predicted state via first-order Taylor expansion, then apply the
  Kalman equations to the linearised system. Core idea: pretend the problem is locally
  linear-Gaussian. **Gap:** the posterior is still approximated by a single Gaussian, which can be a
  gross distortion of the true shape; linearisation error accumulates and the filter can diverge —
  exactly what happens in bearings-only tracking, where the true posterior is a curved ridge no
  Gaussian can represent.

- **Gaussian sum filter** (Alspach & Sorenson 1972). Represent the posterior as a weighted mixture
  of Gaussians and run a bank of EKFs, one per component. Core idea: a mixture can approximate
  non-Gaussian shapes. **Gap:** the number of mixture components grows multiplicatively with each
  step and must be pruned/merged heuristically; each component is still a local linear-Gaussian
  approximation, so sharp nonlinearities are handled only coarsely.

- **Grid / numerical-integration filters** (Bucy 1969; Kitagawa 1987; Kramer & Sorenson 1988; Pole
  & West 1988; Sorenson 1988). Evaluate the posterior density directly on a fixed grid of points in
  state space, performing the prediction and update integrals numerically at each node. Core idea: a
  brute-force discretisation of the exact recursion, making no Gaussian assumption. **Gap:** a fixed
  grid does not follow the moving probability mass and demands a large number of nodes; the count
  explodes with state dimension (the curse of dimensionality), and a significant computation is
  needed at every node every step. Choosing an efficient grid is itself hard.

- **Monte Carlo integration in dynamic models** (Müller 1991). Same prediction phase as a
  sample-based filter, but the update accepts each prior sample as a posterior sample with
  probability proportional to its likelihood (accept/reject). Core idea: keep the posterior as a
  sample. **Gap:** the accepted sample size is random and *decreasing* over time, so the
  representation thins out; Müller's main concern is re-expanding the prior sample from a fitted
  envelope, which is extra machinery.

- **Adaptive importance sampling with mixtures** (West 1992). Build an importance density as a large
  mixture and reweight. **Gap:** the mixtures become large and the technique is computationally
  expensive.

## Evaluation settings

The natural testbeds are nonlinear/non-Gaussian state-space models where a Gaussian filter is known
to struggle, compared against the EKF as the incumbent:

- **A univariate nonstationary growth model** (from Kitagawa 1987; Carlin, Polson & Stoffer 1992),
  $x_k = 0.5 x_{k-1} + 25 x_{k-1}/(1 + x_{k-1}^2) + 8\cos(1.2(k-1)) + w_k$, with measurement
  $y_k = x_k^2/20 + v_k$, $w_k\sim N(0,10)$, $v_k\sim N(0,1)$. Severely nonlinear in both system and
  measurement; the squared observation makes the likelihood bimodal for positive $y_k$ (modes at
  $\pm\sqrt{20 y_k}$), so the posterior is genuinely multimodal. A 50-step realisation is the test.
- **Bearings-only tracking** over a four-dimensional state $(x,\dot x, y, \dot y)$: second-order
  motion $x_k = \Phi x_{k-1} + \Gamma w_k$ with Gaussian process noise, and a single bearing
  measurement $z_k = \arctan(y_k/x_k) + v_k$ from a fixed observer, run over $\sim 24$ time steps.
  Range is unobservable from bearing alone, so the posterior is non-Gaussian (curved/bimodal).
- **A stochastic volatility model** $x_k = \alpha x_{k-1} + \sigma v_k$,
  $y_k = \beta \exp(x_k/2) w_k$ — a standard nonlinear/non-Gaussian econometric SSM used to exhibit
  filter behaviour over a few hundred steps.

Metrics are the usual ones for filtering: tracking error of the posterior-mean estimate against the
true state, coverage of stated probability regions (e.g. whether the true state falls inside the
95% interval), and the qualitative fidelity of the estimated posterior shape (uni- vs multimodal).
Diagnostics of the *approximation's* own health: the spread of the importance weights and the
effective number of distinct particles over time.

## Code framework

The pieces that already exist before any new estimator: a way to specify the model (sample the
prior, sample the dynamics noise, evaluate the measurement likelihood) and a driver loop that walks
through the measurements. The recursive estimator itself is an empty slot.

```python
import numpy as np

class StateSpaceModel:
    """A discrete-time SSM: x_k = f(x_{k-1}, w), y_k = h(x_k, v), noises with known PDFs."""

    def sample_prior(self, N):
        """Return N draws from p(x_1)."""
        raise NotImplementedError

    def propagate(self, x):
        """Push states one step through the dynamics, drawing fresh process noise:
        return f(x, w) with w ~ p(w). Vectorised over a batch of states."""
        raise NotImplementedError

    def log_likelihood(self, y, x):
        """Return log g(y | x) for measurement y and a batch of states x."""
        raise NotImplementedError


class RecursiveFilter:
    """Online estimator of p(x_k | y_{1:k}). To be designed."""

    def __init__(self, model):
        self.model = model
        # TODO: how do we represent the posterior?

    def initialise(self):
        # TODO: build the representation of p(x_1) from the prior
        pass

    def predict(self):
        # TODO: propagate the representation through the dynamics (Chapman-Kolmogorov)
        pass

    def update(self, y):
        # TODO: fold in measurement y via Bayes' rule
        pass

    def estimate(self):
        # TODO: read off a point/interval estimate from the representation
        pass


def run_filter(model, observations):
    filt = RecursiveFilter(model)
    filt.initialise()
    estimates = []
    for k, y in enumerate(observations):
        if k > 0:
            filt.predict()
        filt.update(y)
        estimates.append(filt.estimate())
    return estimates
```

The estimator must still answer how to represent the posterior, how to carry out the prediction
integral and the Bayes update on that representation, and how to keep the representation from
degrading over many steps.
