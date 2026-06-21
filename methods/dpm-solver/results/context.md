## Research question

Diffusion probabilistic models (DPMs) produce excellent samples, but generating a single sample
requires running a large neural network sequentially hundreds or thousands of times. Each step removes
a little noise; the chain is long because each step is a small, first-order correction. By contrast,
single-pass generators (GANs, VAEs) that DPMs match or beat on sample quality produce a sample in one
network evaluation.

The setting: given an already-trained DPM — a noise-prediction network — produce high-quality samples
in as few sequential network evaluations as possible, on the order of ten, using the model as-is
(training-free, plug-and-play).

## Background

**The forward process and its noise schedule.** A DPM defines a forward process that gradually adds
Gaussian noise to data. For data x_0 and time t ∈ [0,T], the conditional distribution is

  q_{0t}(x_t | x_0) = N(x_t | α_t x_0, σ_t² I),

where α_t, σ_t are smooth positive functions of t — the *noise schedule*. The schedule is chosen so the
signal-to-noise ratio α_t²/σ_t² is strictly decreasing in t (data at t=0, near-pure noise at t=T).
Kingma et al. (2021) showed this same set of marginals is realized by the stochastic differential
equation (SDE)

  dx_t = f(t) x_t dt + g(t) dw_t,   f(t) = d log α_t / dt,   g²(t) = dσ_t²/dt − 2 (d log α_t/dt) σ_t².

**Score, network parameterization.** Song et al. (2020) showed the forward SDE has a reverse-time SDE
whose only unknown is the score ∇_x log q_t(x_t). In practice a network ε_θ(x_t, t) is trained to predict
the scaled score −σ_t ∇_x log q_t(x_t); equivalently, ε_θ predicts the Gaussian noise added to x_t
(the "noise prediction model"), via the denoising objective

  E_{x_0, ε} ‖ε_θ(α_t x_0 + σ_t ε, t) − ε‖²   averaged over t,   ε ∼ N(0, I).

**Sampling as solving a differential equation.** Substituting −ε_θ/σ_t for the score gives a parameterized
reverse-time SDE; the standard ancestral sampler (Ho et al. 2020) is a first-order discretization of it.
For SDE discretizations, the step size is limited by the randomness of the Wiener process
(Kloeden & Platen 1992, Ch. 11), and the samplers use hundreds to thousands of steps. The deterministic
alternative is the *probability flow ODE* (Song et al. 2020), which shares the SDE's marginals at every t
but has no Brownian term:

  dx_t/dt = f(t) x_t + (g²(t) / 2σ_t) ε_θ(x_t, t),   solved from t=T down to t=0.

With no randomness, an ODE admits larger steps in principle, and one can bring in off-the-shelf numerical
ODE solvers. Song et al. (2020) ran a high-order adaptive black-box solver (the RK45 Runge–Kutta pair of
Dormand & Prince 1980) on this ODE and reached quality comparable to a 1000-step SDE solver in about 60
network evaluations.

**Schedule quantities.** Because the SNR α_t²/σ_t² is monotone in t, the quantity log(α_t/σ_t) is a
strictly decreasing, invertible function of t. For the variance-preserving schedules used in practice,
α_t² + σ_t² = 1, and both this quantity and its inverse have closed forms for the common linear and
cosine schedules.

## Baselines

**DDPM ancestral sampling (Ho et al. 2020).** A discrete-time Markov chain trained by a variational bound;
sampling reverses it one small step at a time and is a first-order discretization of the reverse SDE.
Needs roughly 1000 sequential network evaluations.

**Probability-flow ODE with a black-box solver (Song et al. 2020).** Writes sampling as the integral
x_t = x_s + ∫_s^t (f(τ)x_τ + (g²(τ)/2σ_τ) ε_θ) dτ and hands the whole integrand to a generic high-order
Runge–Kutta solver (RK45), using ~60 evaluations.

**DDIM (Song et al. 2020, denoising diffusion implicit models).** A deterministic, non-Markovian sampler.
For adjacent steps t_{i-1} → t_i, given a value x̃_{t_{i-1}},

  x̃_{t_i} = (α_{t_i}/α_{t_{i-1}}) x̃_{t_{i-1}} − α_{t_i} ( σ_{t_{i-1}}/α_{t_{i-1}} − σ_{t_i}/α_{t_i} ) ε_θ(x̃_{t_{i-1}}, t_{i-1}).

A first-order deterministic sampler derived from a non-Markovian inference construction, run at ~50 steps.

**Analytic-DPM (Bao et al. 2022) and learned/distilled samplers (Salimans & Ho 2022; Lam et al. 2021;
Watson et al. 2021).** Either estimate optimal reverse variances analytically, or learn noise levels /
trajectories, or distill the model into a few-step student. The distillation route can reach ~4 steps with
an extra training stage, redone per model/dataset/step-budget. Analytic and trajectory-learning samplers
use ~50 evaluations.

**Adaptive-step ODE/SDE solver (Jolicoeur-Martineau et al. 2021).** An adaptive step-size controller for
diffusion differential equations, comparing a lower- and higher-order estimate to size the next step.

**Exponential integrators / exponential Runge–Kutta (Hochbruck & Ostermann 2005, 2010).** A family of
numerical-ODE methods from the general-purpose ODE literature, part of the standard numerical-ODE toolbox
available off the shelf.

## Evaluation settings

The natural yardsticks are the standard unconditional and class-conditional image-generation benchmarks
on which DPMs were already evaluated: CIFAR-10 (32×32), CelebA 64×64, LSUN bedroom 256×256, ImageNet
64×64 and 256×256 (the last with classifier guidance). Both pre-training regimes are in scope: continuous-
time DPMs (e.g. variance-preserving models with linear / cosine schedules) and discrete-time DPMs trained
at a fixed number of steps (e.g. 1000 or 4000). Sample quality is measured by Fréchet Inception Distance
(FID). The cost axis is the number of sequential network evaluations (NFE); a fast sampler is judged by the
FID it reaches at a fixed small NFE, especially around 10. For a training-free solver, the protocol is to
reuse a model trained by others without modification, varying only the solver and its step schedule.

## Code framework

The pieces that already exist before a dedicated solver: a noise-schedule object that knows α_t, σ_t and
their derived quantities; a wrapper that turns a trained model into a noise-prediction function; and a
generic loop that discretizes the probability-flow ODE from T to a small end time. The contribution will
live entirely inside the per-step update — how to march x from one time to the next.

```python
import torch

class NoiseSchedule:
    """Known: the schedule (alpha_t, sigma_t) and quantities derived from it."""
    def marginal_log_mean_coeff(self, t):  # log(alpha_t)
        ...
    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))
    def marginal_std(self, t):             # sigma_t
        ...
    # The monotone quantity lambda_t = log(alpha_t) - log(sigma_t) and its inverse.
    # These are available schedule coordinates because SNR is monotone in t.
    def marginal_lambda(self, t):
        ...
    def inverse_lambda(self, lamb):
        ...


def wrap_model(model, noise_schedule):
    """Known: turn a trained model into a noise-prediction function eps_theta(x, t).
    Handles noise / x0 / score parameterizations and (optionally) guidance."""
    def eps_theta(x, t):
        ...  # return predicted noise at (x, t)
    return eps_theta


class ProbabilityFlowSampler:
    def __init__(self, eps_theta, noise_schedule):
        self.eps_theta = eps_theta
        self.ns = noise_schedule

    def get_time_steps(self, t_T, t_0, N, device):
        # A monotone sequence of N+1 times from T down to the end time.
        # The right spacing (uniform in t? in some derived variable?) is part of what we must decide.
        raise NotImplementedError

    def step(self, x, s, t, **kwargs):
        # Advance the ODE solution from time s to time t (t < s).
        # This is the slot the method fills in: how to discretize dx/dt = f(t)x + (g^2/2 sigma) eps_theta.
        raise NotImplementedError

    def sample(self, x_T, steps, t_T, t_0, device):
        ts = self.get_time_steps(t_T, t_0, steps, device)
        x = x_T
        for i in range(1, len(ts)):
            x = self.step(x, ts[i - 1], ts[i])
        return x
```
