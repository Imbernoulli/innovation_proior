# Context: fast samplers for guided diffusion models (circa 2022)

## Research question

Diffusion probabilistic models (DPMs) generate an image by starting from pure Gaussian noise and
running a learned denoiser many times, each call nudging the latent a little closer to a clean
sample. The single dominant cost is the **number of function evaluations (NFE)** — each step is
one forward pass of a large U-Net, and the steps are strictly sequential, so halving the steps
roughly halves wall-clock latency. For the *unconditional* setting, dedicated solvers had pushed
this down to 10-20 NFE. But the quality that made text-to-image systems compelling comes from
**guided sampling**: the denoiser is steered toward a condition (a class, or a text prompt) by a
guidance term whose strength is set by a *guidance scale* `s`, and in practice a *large* `s` is
what buys the sharp, prompt-aligned images people want. The commonly used solver for guided
sampling, DDIM, is a first-order method that needs roughly 100-250 NFE to converge — slow.

The precise goal: a **training-free** sampler — one that plugs into an already-trained denoiser
with no retraining or distillation — that produces high-quality samples in roughly **15-20 NFE
even at a large guidance scale**, for both pixel-space DPMs (image data bounded in `[-1, 1]`) and
latent-space DPMs (Stable-Diffusion-style, where the diffusion runs in a VAE latent). A solution
must (1) take large steps without the solution diverging, and (2) keep the generated sample inside
the valid data range. Existing fast solvers achieve neither under large guidance, which is the gap.

## Background

**The diffusion forward process and its two parameterizations.** A DPM perturbs data `x_0` toward
noise: the marginal at time `t` is `q_{t0}(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I)`, so
`x_t = alpha_t x_0 + sigma_t eps` with `eps ~ N(0, I)`, where the noise schedule `(alpha_t,
sigma_t)` is chosen so the signal-to-noise ratio `alpha_t^2 / sigma_t^2` strictly decreases in `t`
(Kingma et al. 2021). A denoiser can be cast two ways that are exactly inter-convertible: the
**noise-prediction model** `eps_theta(x_t, t)` (trained to regress the added noise) and the
**data-prediction model** `x_theta(x_t, t) := (x_t - sigma_t eps_theta(x_t, t)) / alpha_t` (the
implied estimate of the clean `x_0`, i.e. Tweedie's formula). Both carry the same information; the
difference is which quantity the sampler manipulates directly.

**Sampling as solving an ODE.** Song et al. (2020) showed the reverse process can be written as a
deterministic *probability-flow ODE* with the same marginals as the diffusion. In
noise-prediction form, integrating `t` from `T` down to `0`,

```
dx_t/dt = f(t) x_t + g^2(t) / (2 sigma_t) * eps_theta(x_t, t),
f(t) = d log(alpha_t)/dt,   g^2(t) = d(sigma_t^2)/dt - 2 (d log(alpha_t)/dt) sigma_t^2.
```

The right-hand side is **semi-linear**: a linear term `f(t) x_t` plus a nonlinear term through the
network. A black-box ODE solver (e.g. RK45) is blind to this structure and discretizes both terms,
spending error budget on the linear part whose solution is actually an exact exponential.

**Exponential integrators and the half-log-SNR variable.** For semi-linear ODEs `x' = A x + N(x)`,
the *exponential-integrator* literature (Hochbruck & Ostermann 2010) solves the linear part exactly
by variation-of-constants and approximates only the integral of the nonlinear part, using the
family of `phi`-functions `phi_0(z) = e^z`, `phi_{k+1}(z) = (phi_k(z) - phi_k(0))/z`,
`phi_k(0) = 1/k!`, so `phi_1(h) = (e^h - 1)/h`, `phi_2(h) = (e^h - h - 1)/h^2`. Lu et al. (2022)
applied this to the diffusion ODE: define the **half-log-SNR** `lambda_t := log(alpha_t / sigma_t)`,
which is strictly decreasing in `t` and hence invertible (`t_lambda` its inverse). Rewriting `g^2`
via `g^2(t) = -2 sigma_t^2 (d lambda_t / dt)` and changing the integration variable to `lambda`
collapses all schedule coefficients into a single analytic exponential, yielding an *exact* solution
of the noise-prediction ODE from `s` to `t`:

```
x_t = (alpha_t / alpha_s) x_s - alpha_t * integral_{lambda_s}^{lambda_t} e^{-lambda} eps_hat_theta(x_hat_lambda, lambda) dlambda,
```

where hats denote the change-of-variable forms `x_hat_lambda := x_{t_lambda(lambda)}`. Only the
"exponentially weighted integral" of the network remains to be approximated; the linear part is now
error-free.

**Guided sampling and the guidance scale.** Conditioning is done by replacing the unconditional
`eps_theta` with a guided one. *Classifier guidance* (Dhariwal & Nichol 2021) uses a separately
trained classifier `p_phi(c | x_t, t)`:

```
eps_tilde_theta(x_t, t, c) = eps_theta(x_t, t) - s * sigma_t * grad_{x_t} log p_phi(c | x_t, t).
```

*Classifier-free guidance* (Ho & Salimans 2021) trains one model for both the conditional and an
unconditional (`c = empty`) prediction and mixes them:

```
eps_tilde_theta(x_t, t, c) = s * eps_theta(x_t, t, c) + (1 - s) * eps_theta(x_t, t, empty).
```

In both, `s > 0` is the guidance scale and a *large* `s` gives the best condition-sample alignment
in text-to-image and class-to-image use (Saharia et al. 2022; Rombach et al. 2022). Sampling then
solves the same ODE with `eps_tilde_theta` in place of `eps_theta`.

**The diagnostic findings that frame the problem.** Two observations about *existing* fast solvers
under large guidance are the motivating facts (measured on pre-trained classifier-guided ImageNet
256x256 DPMs at guidance scale `s = 8.0` with only 15 NFE):

- *Large guidance destabilizes high-order solvers.* A large `s` amplifies both the magnitude of
  `eps_tilde_theta` and its derivatives with respect to `lambda`. High-order solvers approximate
  those `lambda`-derivatives to take big steps; amplified derivatives shrink the radius within which
  the high-order approximation is valid, so the solver needs *smaller* steps to converge. At a fixed
  small NFE the high-order solvers (second- and third-order exponential-integrator solvers, and the
  pseudo-numerical PNDM solver) produce *worse* samples than first-order DDIM — and the quality
  degrades further as the solver order increases. The fast-solver speedups that hold without
  guidance do not transfer to the large-guidance regime.

- *Train-test mismatch.* Image pixel data lies in a bounded interval (`[-1, 1]`). A large `s` pushes
  `eps_tilde_theta` away from any true noise direction, so the converged `x_0` lands outside that
  interval; the decoded images come out saturated and unnatural (Saharia et al. 2022). A known
  remedy in the few-step literature is *thresholding* — clipping the running clean-image estimate
  back into the data bound at each step — but clipping is only available if the sampler actually
  has a clean-image estimate in hand to clip.

## Baselines

**DDIM (Song et al. 2021, ICLR; arXiv:2010.02502).** A deterministic, non-Markovian sampler. In
its general `eta >= 0` form, one step reads

```
x_{t_i} = alpha_{t_i} x_theta(x_{t_{i-1}}, t_{i-1}) + sqrt(sigma_{t_i}^2 - eta^2) eps_theta(x_{t_{i-1}}, t_{i-1}) + eta z,
```

with `eta = 0` the deterministic case and larger `eta` reinjecting noise. It was later identified
(Lu et al. 2022; Salimans & Ho 2022) as a **first-order** discretization of the diffusion ODE. It
is the workhorse for guided sampling and is empirically robust there. **Gap:** it is only
first-order, so it needs ~100-250 NFE to converge; there is no higher-order DDIM, and at small NFE
it is far from converged.

**DPM-Solver (Lu et al. 2022; arXiv:2206.00927).** The dedicated high-order solver for the
*noise-prediction* diffusion ODE. From the exact solution above, Taylor-expand
`eps_hat_theta(lambda)` to order `k-1` around `lambda_{t_{i-1}}`, substitute, and integrate each
term `integral e^{-lambda} (lambda - lambda_{t_{i-1}})^n / n! dlambda` analytically by repeated
integration by parts (these are the `phi`-functions). Dropping the `O(h_i^{k+1})` remainder gives a
`k`-th order solver `DPM-Solver-k` using `k` network calls per step (a *singlestep* method); `k=1`
recovers DDIM. The second-order member `DPM-Solver-2` introduces one intermediate point at the
midpoint in `lambda` (`r1 = 1/2`):

```
u_i      = (alpha_{s_i}/alpha_{t_{i-1}}) x_{t_{i-1}} - sigma_{s_i}(e^{h_i/2} - 1) eps_theta(x_{t_{i-1}}, t_{i-1})
x_{t_i}  = (alpha_{t_i}/alpha_{t_{i-1}}) x_{t_{i-1}} - sigma_{t_i}(e^{h_i} - 1) eps_theta(u_i, s_i),
```

with `h_i = lambda_{t_i} - lambda_{t_{i-1}}`, proven `k`-th order under regularity assumptions.
Reaches 10-20 NFE without guidance. **Gap:** it operates on `eps_theta`; under large guidance the
amplified `lambda`-derivatives of `eps_tilde_theta` shrink its convergence radius (the instability
finding above), and because it never forms a clean-image estimate, there is nothing to clip, so the
train-test mismatch remains.

**DEIS (Zhang & Chen 2022; arXiv:2204.13902).** A *multistep* exponential-integrator solver on
`eps_theta`, Taylor-expanding in `t` and reusing past network outputs (Adams-Bashforth style) so
each step costs only one new evaluation. **Gap:** same `eps`-parameterization, same large-guidance
instability and train-test mismatch.

**Black-box ODE / SDE solvers (Song et al. 2020; arXiv:2011.13456).** Generic RK45 on the
probability-flow ODE, or ancestral / reverse-SDE samplers. **Gap:** RK45 discretizes the linear
term it should solve exactly (wasted error) and needs ~60+ NFE; SDE samplers inject randomness that
caps the usable step size and need even more steps to converge.

**Distillation / learned-schedule samplers (Salimans & Ho 2022; Watson et al. 2021).** Train a
student to take big steps, or learn the step schedule. **Gap:** they require extra training and are
hard to scale to large pre-trained text-to-image DPMs — exactly the setting that matters here.

## Evaluation settings

The yardsticks that existed for this problem, all pre-method:

- **Pixel-space class-guided generation:** the pre-trained classifier-guided ImageNet 256x256 DPM
  (Dhariwal & Nichol 2021), varying the classifier guidance scale (`s` swept over `0, 1, ..., 8`),
  with the budget fixed by NFE (commonly 10, 15, 20, 25 for high-order solvers; 50, 100, 250 for
  DDIM). Solve from `t = 1` down to `t = 10^{-3}` (the smallest trained time).
- **Latent-space text-to-image generation:** Stable Diffusion (Rombach et al. 2022) with
  classifier-free guidance, sampling in the VAE latent space, then VAE-decoding to an image; the
  same NFE budget; a fixed shared prompt set across samplers.
- **Time-step schedule:** the time steps `{t_i}` chosen by a power-function family
  `t_i = ((M-i)/M * t_0^{1/kappa} + i/M * t_M^{1/kappa})^kappa` (with `t_M = 10^{-3}`, `t_0 = 1`),
  comparing `kappa in {1, 2, 3}` and uniform-`t` vs uniform-`lambda` spacing. Discrete-time
  denoisers are made continuous by linearly interpolating `log(alpha_t)` between trained steps.
- **Metrics:** FID against a reference image set (lower is better) for fidelity, and CLIP score
  (cosine similarity between the generated image and the text prompt; higher is better) for
  text-image alignment. The protocol holds the prompts, model weights, NFE budget, and metric
  computation fixed across samplers.

## Code framework

A training-free sampler plugs into the same iterative-denoising harness already used by DDIM and
the high-order solvers above. The substrate that already exists: a frozen denoiser exposed as a
`predict_noise` call, the noise schedule (`alpha(t)`, `sigma(t)`, the half-log-SNR `lambda(t)` and
its inverse), a latent initializer, and a VAE decoder. The outer loop walks a decreasing schedule
of times and, at each one, calls the network and then applies a **per-step update rule** to advance
the latent. That update rule — how to combine the network output(s) into the next latent so that
large steps stay accurate and stable under guidance — is exactly what is to be designed; everything
else is fixed.

```python
import torch


class GuidedDiffusionSampler:
    """Training-free sampler harness around a frozen denoiser."""

    def __init__(self, model, noise_schedule, vae, cfg_guidance=7.5):
        self.model = model                    # frozen U-Net denoiser
        self.ns = noise_schedule              # alpha(t), sigma(t), lambda(t), inverse_lambda(l)
        self.vae = vae                        # latent <-> image decoder
        self.cfg_guidance = cfg_guidance      # guidance scale s (large => best alignment)

    def predict_noise(self, x, t, uc, c):
        """One network call; returns the guided noise prediction eps_tilde(x, t)."""
        eps_uncond = self.model(x, t, uc)
        eps_cond = self.model(x, t, c)
        return eps_uncond + self.cfg_guidance * (eps_cond - eps_uncond)

    def alpha(self, t):
        return self.ns.alpha(t)               # sigma(t) = sqrt(1 - alpha(t)**2); lambda = log(alpha/sigma)

    def update_rule(self, x, t, t_next, uc, c, history):
        # TODO: implement the per-step update rule. Given the current latent
        #       x at time t and the network/schedule access above, advance to
        #       t_next under the fixed denoiser.
        #       (May keep `history` across steps if the rule needs it.)
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, prompt, timesteps):
        uc, c = self.get_text_embed(prompt)   # (unconditional, conditional) embeddings
        x = self.initialize_latent()          # x_T ~ N(0, sigma_tilde^2 I)
        history = {}
        for step, (t, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            x = self.update_rule(x, t, t_next, uc, c, history)
        img = self.vae.decode(x)
        return (img / 2 + 0.5).clamp(0, 1)
```

The harness supplies the network and the schedule; `update_rule` is the open slot, and the time loop's
spacing (and thus how the fixed NFE budget is spent) is set by `timesteps`.
