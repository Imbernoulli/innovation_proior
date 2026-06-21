## Research question

We have a pretrained generative model that transports between two arbitrary paired
distributions — a sketch and the handbag it depicts, a corrupted image and its restoration, a
masked image and a plausible completion. Unlike an ordinary diffusion model, the "prior" end is
not Gaussian noise but an *informative* endpoint `x_T` (the degraded / source image), and the
model is a *diffusion bridge*: a stochastic process pinned to start at the clean data `x_0` and
arrive almost surely at `x_T`. The bridge is trained by score matching against a tractable
forward transition kernel, and at inference one generates `x_0` from `x_T` by running the
reverse process from `t = T` down to `t = 0`.

The reverse process is a (stochastic) differential equation, and the standard way these models
are sampled is by numerically simulating that SDE/ODE with a generic solver. Each network
evaluation is one forward pass of a large U-Net, so the wall-clock cost is roughly proportional
to the number of function evaluations (NFE). The question is how to sample a *pretrained* bridge
model — reusing the learned network as is — at a small NFE budget.

## Background

**Diffusion models and their forward kernel.** A diffusion model defines a forward SDE
`dx_t = f(t) x_t dt + g(t) dw_t` over `t ∈ [0, T]` whose linear structure gives an analytic
Gaussian transition kernel `q(x_t | x_0) = N(α_t x_0, σ_t² I)`. Here `α_t, σ_t` are the *noise
schedule*, tied to the drift/diffusion by `f = d log α_t / dt` and
`g² = dσ_t²/dt − 2 (d log α_t/dt) σ_t²`, and the *signal-to-noise ratio* is `SNR_t = α_t²/σ_t²`.
Sampling reverses the SDE or the equivalent probability-flow ODE; the only unknown is the score
`∇ log q_t(x_t)`, learned by denoising score matching. A large body of dedicated solvers
(Euler/Heun on the ODE, exponential integrators) brings unconditional diffusion sampling down to
~10 NFE.

**Diffusion bridges via Doob's h-transform.** To translate between two arbitrary distributions
one conditions the diffusion to land at a chosen endpoint `x_T = y` almost surely, using Doob's
h-transform (Doob 1984): `dx_t = f(t) x_t dt + g²(t) ∇ log q(x_T = y | x_t) dt + g(t) dw_t`,
with `x_0 ~ p_data` and `x_T = y`. Conditioned on both endpoints, the process is a *bridge* and
again has an analytic Gaussian forward kernel
`q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t² I)`, with the coefficients fixed by the noise
schedule,
```
a_t = (α_t/α_T)(SNR_T/SNR_t),   b_t = α_t (1 − SNR_T/SNR_t),   c_t² = σ_t² (1 − SNR_T/SNR_t).
```
The mean is a (scaled) interpolation between the two endpoints, collapsing to a Dirac at either
end. The reverse process needs the *bridge score* `∇ log q(x_t | x_T)` (learned by a network
`s_θ(x_t, t, x_T)`, "denoising bridge score matching") plus an analytically known h-term. A
useful, equivalent view is the *data predictor* `x_θ(x_t, t, x_T)` — the network's estimate of
the clean `x_0` from a noisy bridge state — related to the score by
`s_θ = −(x_t − a_t x_T − b_t x_θ)/c_t²`.

**The marginals-only property.** A fact established for ordinary diffusion: the
score-matching / denoising training loss depends on the model only through the per-timestep
*marginals* `q(x_t | x_0)`, not through the full *joint* over the trajectory. Many different
joint inference processes share the same marginals; the network trained under any one of them is
optimal for all of them. One can swap the joint used for generation while keeping the trained
network fixed.

**Conditional spread of the bridge.** The bridge's reverse process is stochastic near the
starting point: under a *fixed* endpoint `x_T`, the state `x_t` for `t < T` is not deterministic
(a masked image admits many completions), so the marginal `p(x_t | x_T)` is not a Dirac.

## Baselines

**DDIM — denoising diffusion implicit models (Song et al. 2020).** For ordinary diffusion,
exploit the marginals-only property: index a family of inference distributions by
`σ ∈ R_{≥0}^T`, with non-Markovian reverse conditionals
```
q_σ(x_{t-1} | x_t, x_0) = N( √α_{t-1} x_0 + √(1 − α_{t-1} − σ_t²) · (x_t − √α_t x_0)/√(1 − α_t),  σ_t² I ),
```
where the mean is engineered so that `q_σ(x_t | x_0) = N(√α_t x_0, (1−α_t) I)` is preserved for
every `σ`. The generative step replaces `x_0` by the network's prediction and reads
`x_{t-1} = √α_{t-1} x̂_0 + √(1 − α_{t-1} − σ_t²) ε̂ + σ_t ε`, a "predicted-x_0 + direction-to-x_t
+ fresh noise" decomposition. `σ_t = 0` makes the process deterministic (an implicit model,
DDIM), enabling few-step sampling, latent encoding, and interpolation; the DDPM choice of `σ_t`
recovers the Markovian ancestral sampler. The family shares the DDPM training loss, so a
pretrained DDPM is reused unchanged. DDIM also rewrites as an Euler step of an ODE on
`x̄ = x/√α` with respect to `dσ`. The construction is set in the Gaussian data-to-noise
diffusion: mean `√α_t x_0`, variance `1 − α_t`, with no second endpoint.

**DDBM — denoising diffusion bridge models (Zhou et al. 2023).** The bridge framework above:
Doob h-transform, the analytic kernel `N(a_t x_T + b_t x_0, c_t² I)`, the learned bridge score,
and a reverse SDE / probability-flow ODE for generation. To sample, DDBM uses a *hybrid,
high-order sampler* adapted from the EDM Heun method (Karras et al. 2022): EDM-spaced timesteps
(`(t_max^{1/ρ} + (i/N)(t_min^{1/ρ} − t_max^{1/ρ}))^ρ`, `ρ = 7`), second-order Heun steps on the
PF-ODE, and an interleaved fraction of stochastic Euler steps ("churn") to inject noise. It is a
generic ODE/SDE discretizer applied to the bridge dynamics.

**Exponential-integrator ODE solvers for diffusion (e.g. Lu et al. 2022).** For the *ordinary*
diffusion ODE, recognize it as semi-linear, `dx = [a(t) x + b(t) F_θ] dt`, and apply
variation-of-constants to cancel the linear term exactly:
`x_t = e^{∫a} x_s + ∫ e^{∫a} b(τ) F_θ dτ`. Change the integration variable to the log-SNR
`λ_t = log(α_t/σ_t)`; the remaining integral becomes an exponentially-weighted
`∫ e^{−λ} ε̂_θ dλ`. Taylor-expand the network output in `λ`; the scalar integrals
`∫ (λ − λ_s)^n e^{−λ} dλ` are analytic, and the derivatives of the network output are estimated
by finite differences of past evaluations — yielding high-order solvers that hit good quality in
~10 NFE. This is built for the diffusion ODE `dx = f x dt − ½ g² ∇log q dt` with its single
Gaussian endpoint.

**Posterior-style bridge samplers (e.g. I²SB, Liu et al. 2023).** Variance-exploding bridges
(`f = 0`) sampled by DDPM-like ancestral sampling from a shortened bridge between `x̂_0` and the
current state, on the VE schedule.

## Evaluation settings

The natural yardsticks already in use for distribution translation:

- **Image-to-image translation** on Edges→Handbags (Isola et al. 2017, 64×64) and
  DIODE-Outdoor (Vasiljevic et al. 2019, 256×256), metrics computed against the full training
  set.
- **Image restoration / inpainting** on ImageNet (Deng et al. 2009, 256×256) with a 128×128
  center mask, metrics on 10k validation images; a separate bridge model trained with the I²SB
  noise schedule, initialized from the class-conditional ImageNet diffusion model and
  additionally conditioned on `x_T`.
- **Metric:** Fréchet Inception Distance (FID, lower is better), with Inception Score, LPIPS,
  MSE (for translation) and classifier accuracy (for inpainting) as secondary measures; a
  diversity score (per-condition pixel std over multiple samples) for inpainting.
- **Protocol:** time horizon `t_min = 1e-4`, `t_max = 1`; identical trained models compared
  across samplers; quality reported as a function of NFE. The data predictor is an EDM-style
  preconditioned network `x_θ = c_skip x_t + c_out F_θ(c_in x_t, c_noise, x_T)`. The DDBM
  baseline uses its EDM-ρ=7 Heun hybrid sampler with Euler-step ratio 0.33.

## Code framework

A sampler plugs into a fixed evaluation harness. The pretrained bridge model exposes a *data
predictor* `denoiser(x, t) → x̂_0` (the EDM-preconditioned network's estimate of clean data); the
*noise schedule* object exposes the bridge coefficients `a_t, b_t, c_t` and the raw schedule
quantities for any time, including the schedule's inverse-square-root-SNR noise level
`σ_t/α_t` when that helper names it `rho`;
the outer loop supplies the source `x_T`, a monotonically decreasing interior time schedule `ts`
whose first element is just below `t_max`, an optional `mask`, and a seeded noise generator, and it
wraps `denoiser` with a hard NFE counter. Nothing about the sampler state or the per-step update
from one time to the next is settled. The single empty slot is the state evolution computed from
the available denoiser output, schedule coefficients, and optional random numbers.

```python
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from .nn import append_dims
from .random_util import BatchedSeedGenerator


@torch.no_grad()
def sample(denoiser, diffusion, x, ts, stochasticity=1.0, mask=None, seed=None, **kwargs):
    """Generate x_0 from the source endpoint x_T = x. The first transition may
    leave t_max and land on ts[0]; the loop then walks the decreasing interior
    schedule. `denoiser(x, t)` returns the data prediction x_hat_0 and is
    NFE-counted by the harness; `diffusion.noise_schedule` exposes
    get_abc(t) -> (a_t, b_t, c_t) and get_alpha_rho(t) -> raw schedule values,
    including rho_t = sigma_t / alpha_t."""
    x_T = x
    path, pred_x0 = [], []
    ones = x.new_ones([x.shape[0]])
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))

    nfe = 0
    generator = BatchedSeedGenerator(seed)
    aux = None
    x0_hat = None

    for _, i in enumerate(indices):
        s = ts[i]            # current (larger) time
        t = ts[i + 1]        # next (smaller) time

        x0_hat = denoiser(x, s * ones)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        # schedule coefficients available for either time:
        # a_s, b_s, c_s = diffusion.noise_schedule.get_abc(s * ones)
        # a_t, b_t, c_t = diffusion.noise_schedule.get_abc(t * ones)

        # TODO: the per-step transition.
        #       From the current state x (at time s) and the data prediction x0_hat,
        #       compute the next state x (at time t < s), reusing only the pretrained
        #       network and the analytic schedule coefficients.
        x = x  # placeholder

        path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    return x, path, nfe, pred_x0, ts, aux
```

The harness fixes the I/O contract, the per-step `denoiser` call, the NFE counter, and the
return tuple `(x, path, nfe, pred_x0, ts, aux)`. The transition inside the loop is where the
method will live.
