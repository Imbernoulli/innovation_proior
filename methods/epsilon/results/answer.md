# Epsilon prediction (the DDPM noise-prediction parameterization), distilled

A Gaussian diffusion model fixes a forward process that gradually noises data into
`N(0, I)` and learns the reverse process. The **epsilon parameterization** trains the
reverse network to predict the *noise* added to a sample, rather than the reverse-step mean
or the clean image, and trains it with an **unweighted** mean-squared error across timesteps
(`L_simple`). This (a) gives a regression target of constant `N(0, I)` scale at every
timestep, so the problem is well-conditioned; (b) makes the reverse step a Langevin step
driven by a learned score, revealing the equivalence between variational diffusion and
denoising score matching with annealed Langevin dynamics; and (c) by dropping the
variational weight, redirects gradient from the easy low-noise terms to the hard high-noise
terms that shape image content.

## Problem it solves

Pick the reverse-process parameterization and the per-timestep loss weighting for a Gaussian
diffusion model so that a finite-time learned reverse Markov chain produces high-quality
image samples, under a fixed forward Gaussian corruption.

## Setup

Forward process with `alpha_t = 1 - beta_t`, `alpha_bar_t = prod_{s<=t} alpha_s`:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) x_{t-1}, beta_t I)
q(x_t | x_0)     = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
  =>  x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps,   eps ~ N(0, I)
```

The variational bound decomposes as `L = L_T + sum_{t>1} L_{t-1} + L_0` with
`L_{t-1} = KL(q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t))`. `L_T` is constant (the forward
process has no parameters). The forward posterior is Gaussian:

```
q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mu_tilde_t, beta_tilde_t I)
mu_tilde_t  = ( sqrt(alpha_bar_{t-1}) beta_t / (1-alpha_bar_t) ) x_0
            + ( sqrt(alpha_t)(1-alpha_bar_{t-1}) / (1-alpha_bar_t) ) x_t
beta_tilde_t = ( (1-alpha_bar_{t-1}) / (1-alpha_bar_t) ) beta_t
```

## Key idea

Fix the reverse variance to an untrained constant `sigma_t^2` (either `beta_t` or
`beta_tilde_t`, the entropy-bound extremes). Then `L_{t-1} = (1/(2 sigma_t^2)) ||mu_tilde_t -
mu_theta||^2 + C`. Substituting `x_0 = (x_t - sqrt(1-alpha_bar_t) eps)/sqrt(alpha_bar_t)`:

```
mu_tilde_t = (1/sqrt(alpha_t)) ( x_t - ( beta_t / sqrt(1-alpha_bar_t) ) eps )
```

Given `x_t`, the only unknown is `eps`. So parameterize the network to predict `eps`:

```
mu_theta(x_t,t) = (1/sqrt(alpha_t)) ( x_t - ( beta_t / sqrt(1-alpha_bar_t) ) eps_theta(x_t,t) )
```

and the loss collapses to a noise-prediction MSE with a per-timestep weight:

```
L_{t-1} - C = E_{x_0,eps}[ ( beta_t^2 / (2 sigma_t^2 alpha_t (1-alpha_bar_t)) )
                           || eps - eps_theta(sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps, t) ||^2 ]
```

**Why epsilon, not mu_tilde or x_0.** Predicting `mu_tilde` forces the network to reproduce
the input `x_t` plus a small `t`-dependent correction (a target whose scale drifts with `t`);
`x_0`-MSE equals `((1-alpha_bar_t)/alpha_bar_t)` times the corresponding epsilon error, so it
over-weights the high-noise regime where `x_0` is nearly unrecoverable.
Predicting `eps` gives a target of constant `N(0,I)` scale across all `t` — the
best-conditioned of the three (all three are linearly interchangeable reparameterizations of
the same model, but the loss is not invariant to the choice).

**Score / Langevin equivalence.** Since `∇_{x_t} log q(x_t|x_0) = -(x_t - sqrt(alpha_bar_t)
x_0)/(1-alpha_bar_t) = -eps/sqrt(1-alpha_bar_t)`, predicting `eps` is predicting the score
`s_theta = -eps_theta/sqrt(1-alpha_bar_t)`, and the reverse step is annealed Langevin
dynamics. This connects the variational diffusion to denoising score matching (Vincent 2011:
`∇_{x̃} log N(x̃;x,sigma^2 I) = -(x̃-x)/sigma^2`) and to NCSN (Song & Ermon 2019).

## Simplified objective

Drop the per-timestep weight (it peaks at small `t`, lavishing gradient on the easy
near-clean denoising and starving the hard high-noise terms):

```
L_simple(theta) = E_{t, x_0, eps}[ || eps - eps_theta(sqrt(alpha_bar_t) x_0
                                                       + sqrt(1-alpha_bar_t) eps, t) ||^2 ],
                  t ~ Uniform{1,...,T}
```

This is NCSN's equal-magnitude-per-level objective reached from the variational side; it
relatively up-weights large-`t` terms, focusing training on the hard denoising problems that
shape image content. `t=1` plays the role
of the discrete decoder `L_0`; `L_T` does not appear (fixed `beta_t`).

## Final algorithm

```
Training:
  repeat:
    x_0 ~ data;  t ~ Uniform({1,...,T});  eps ~ N(0, I)
    x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps
    gradient step on  || eps - eps_theta(x_t, t) ||^2

Sampling:
  x_T ~ N(0, I)
  for t = T,...,1:
    z = N(0, I) if t > 1 else 0
    x_{t-1} = (1/sqrt(alpha_t)) ( x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) eps_theta(x_t, t) ) + sigma_t z
  return x_0
```

Settings: `T = 1000`; `beta_t` linear from `1e-4` to `0.02` (small relative to data in
`[-1,1]`, so each reverse step is approximately Gaussian and `x_T ~ N(0,I)`); data scaled to
`[-1, 1]`; reverse variance fixed to `sigma_t^2 in {beta_t, beta_tilde_t}`.

## Working code

The parameterization reduces to a matched pair — the training target and its exact inverse —
plus the standard noise-prediction training loop.

```python
import torch


def get_schedule(betas):
    """Fixed forward-diffusion tensors (no learnable parameters)."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones_like(alphas_cumprod[:1]), alphas_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alpha": alphas_cumprod.sqrt(),                    # sqrt(alpha_bar_t)
        "sqrt_one_minus_alpha": (1.0 - alphas_cumprod).sqrt(),  # sqrt(1 - alpha_bar_t)
        "posterior_variance": posterior_variance,
        "posterior_mean_coef1": betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
    }


def q_sample(x_0, noise, t, schedule):
    """Forward process: x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps."""
    sa = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    soma = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)
    return sa * x_0 + soma * noise


# --- the epsilon parameterization: a matched (target, inverse) pair ---

def compute_training_target(x_0, noise, timesteps, schedule):
    # Epsilon prediction: the model learns to predict the added noise.
    # Target is N(0, I) at every timestep -> well-conditioned, unweighted MSE.
    return noise


def predict_x0(model_output, x_t, timesteps, schedule):
    # Recover x_0 from the predicted noise by inverting the corruption equation:
    #   x_t = sqrt(abar) x_0 + sqrt(1-abar) eps  =>  x_0 = (x_t - sqrt(1-abar) eps) / sqrt(abar)
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return (x_t - sqrt_one_minus_alpha * model_output) / sqrt_alpha.clamp(min=1e-8)


def q_posterior_mean(x_start, x_t, t, schedule):
    """Mean of q(x_{t-1} | x_t, x_0), matching the canonical sampler structure."""
    coef1 = schedule["posterior_mean_coef1"][t].view(-1, 1, 1, 1)
    coef2 = schedule["posterior_mean_coef2"][t].view(-1, 1, 1, 1)
    return coef1 * x_start + coef2 * x_t


# --- training: L_simple = E || eps - eps_theta(x_t, t) ||^2 ---

def train_step(model, x_0, schedule, T):
    B = x_0.shape[0]
    t = torch.randint(0, T, (B,), device=x_0.device)            # t ~ Uniform{0,...,T-1}
    noise = torch.randn_like(x_0)                               # eps ~ N(0, I)
    x_t = q_sample(x_0, noise, t, schedule)                     # corrupt to level t
    target = compute_training_target(x_0, noise, t, schedule)   # = eps
    pred = model(x_t, t)                                        # time-conditioned U-Net
    return ((pred - target) ** 2).mean()                       # unweighted MSE


@torch.no_grad()
def p_sample_step(model, x_t, t, schedule, sigma_t, add_noise=True):
    """One reverse step x_t -> x_{t-1}, via epsilon -> x_0 -> posterior mean."""
    eps_theta = model(x_t, t)
    x0_hat = predict_x0(eps_theta, x_t, t, schedule).clamp(-1.0, 1.0)
    mean = q_posterior_mean(x0_hat, x_t, t, schedule)
    z = torch.randn_like(x_t) if add_noise else torch.zeros_like(x_t)
    nonzero = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
    return mean + nonzero * sigma_t * z
```

The same logic in the canonical training routine: `target = noise` (`eps` prediction),
clean-image recovery `x0_hat = sqrt(1/alpha_bar_t) x_t - sqrt(1/alpha_bar_t - 1) eps_theta`
(algebraically identical to `(x_t - sqrt(1-alpha_bar_t) eps_theta)/sqrt(alpha_bar_t)`), and
the reverse step `x_{t-1} = (1/sqrt(alpha_t))(x_t - (1-alpha_t)/sqrt(1-alpha_bar_t)
eps_theta) + sigma_t z`.
