# Context: fast sampling of diffusion bridges for image-to-image translation (circa 2023-2024)

## Research question

Image-to-image translation and restoration ask for transport between two *arbitrary*
paired distributions — a sketch and a photograph, a masked image and its completion, a
degraded scan and its clean original — not the data-to-Gaussian transport that ordinary
diffusion models perform. A recently introduced family builds a stochastic process pinned
at both ends (a *diffusion bridge*) and learns to run it in reverse, achieving strong
fidelity on these tasks. The catch is cost. To draw one sample you must simulate a reverse
stochastic (or ordinary) differential equation, and for high-resolution images this takes
well over a hundred network evaluations; even the bridge's own hybrid sampler stays in the
hundreds. Meanwhile, *ordinary* diffusion models can already produce strong samples in about
ten network calls, because a decade of work has built training-free fast samplers for them.
None of that machinery transfers directly: it is all written for a non-informative Gaussian
prior, and a bridge has a fixed, informative endpoint instead. The precise problem is to
sample a *pretrained* diffusion bridge under a strict small evaluation budget — no retraining,
no change to the learned predictor — by replacing only the rule that advances the state from
one timestep to the next. A solution must (1) reuse the existing predictor exactly; (2)
make a tiny per-sample budget viable (each network call is the dominant cost and is metered);
(3) preserve the genuine stochasticity of the task (one masked image admits many completions); and
(4) remain numerically well-behaved at the very first step, where a bridge — unlike an
ordinary diffusion — starts from a single deterministic endpoint.

## Background

**Diffusion models and their fast samplers.** An ordinary diffusion model defines a forward
SDE `dx_t = f(t) x_t dt + g(t) dw_t` whose linear-Gaussian structure gives an analytic
marginal `q(x_t|x_0) = N(alpha_t x_0, sigma_t^2 I)`, with `alpha_t, sigma_t` the noise
schedule and `f = d log alpha_t/dt`, `g^2 = d sigma_t^2/dt - 2 (d log alpha_t/dt) sigma_t^2`
(Kingma et al. 2021). Sampling reverses a probability-flow ODE (PF-ODE) or reverse SDE
(Song et al. 2021) whose only unknown is the score `nabla log q_t`, learned by denoising
score matching. The signal-to-noise ratio `SNR_t = alpha_t^2/sigma_t^2` summarizes the
schedule. Two complementary, training-free accelerations are well established for this
Gaussian-prior setting: *reduce stochasticity*, and *use higher-order information*. Both
exploit a structural fact about the diffusion ODE that prior work made precise.

**The semi-linear structure and the log-SNR change of variable.** The diffusion ODE
`dx = [f(t) x + (g^2/2 sigma) eps_theta] dt` is *semi-linear*: the part `f(t) x` is linear in
`x` and exactly integrable, and only `eps_theta` is nonlinear. Treating the whole right-hand
side as a black box (generic Runge-Kutta / Heun) incurs discretization error on *both*
parts; a dedicated solver should cancel the linear part analytically. Variation-of-constants
does exactly that: `x_t = e^{int_s^t f} x_s + int_s^t e^{int_tau^t f} (g^2/2 sigma) eps_theta
dtau`. Changing the time variable to the half-log-SNR `lambda_t = log(alpha_t/sigma_t)`
collapses the remaining integral into an *exponentially weighted integral* of the predictor,
`int e^{-lambda} eps_theta dlambda` (Lu et al. 2022a). This is the entry point both for
reducing stochasticity and for higher-order solvers; the integrals of `e^{-lambda}
(lambda - lambda_s)^n/n!` against the predictor are analytic (the φ-functions of exponential
integrators; Hochbruck & Ostermann 2010).

**Diffusion bridges and Doob's h-transform.** To pin a diffusion at a chosen endpoint
`x_T = y` almost surely, condition it via Doob's h-transform (Doob 1984): add a drift
`g^2(t) nabla_{x_t} log q(x_T = y | x_t)` to the forward SDE. With a linear base SDE this
conditioned process again has an analytic Gaussian transition kernel — now interpolating
between the two endpoints — and a reverse SDE / PF-ODE driven by the *bridge score*
`nabla_{x_t} log q(x_t | x_T)`, which is learned by denoising bridge score matching. This is
what makes high-fidelity translation possible: the prior `y` carries the source image rather
than being noise. The diagnostic that motivates the whole effort is empirical:
the bridge's reverse SDE/ODE, simulated with a generic high-order hybrid step, needs many
denoiser evaluations (over 100 at high resolution) to reach good quality, an order of
magnitude more than the ~10 that Gaussian-prior diffusion samplers need — because that
sampler is a generic ODE/SDE discretizer, not one built around the bridge's structure.

**The non-Markovian, marginal-preserving construction for ordinary diffusion.** A separate
line of work observed that the diffusion *training* objective depends only on the marginals
`q(x_t|x_0)`, never on the joint `q(x_{1:T}|x_0)`. Many different joints share those
marginals, so one can swap in an *alternative*, non-Markovian inference process — keeping the
same marginals, hence the same trained network — and read off a new, faster sampler from it
(Song et al. 2021). The family is indexed by a variance vector `sigma`; the deterministic end
of the family turns the sampler into an implicit map from a latent to a sample, and a special
nonzero `sigma` recovers the original Markovian sampler. This idea is the conceptual lever for
"reduce stochasticity," but it was developed entirely for the Gaussian-prior case with no
endpoint condition.

## Baselines

**DDBM — denoising diffusion bridge models (Zhou et al. 2023, arXiv:2309.16948).** Builds the
bridge via Doob's h-transform. With base schedule `alpha_t, sigma_t` and `SNR_t =
alpha_t^2/sigma_t^2`, the forward bridge kernel given both endpoints is
`q(x_t|x_0,x_T) = N(a_t x_T + b_t x_0, c_t^2 I)` with
`a_t = (alpha_t/alpha_T)(SNR_T/SNR_t)`, `b_t = alpha_t (1 - SNR_T/SNR_t)`,
`c_t^2 = sigma_t^2 (1 - SNR_T/SNR_t)`. The reverse SDE and PF-ODE start from `x_T = y` and
depend on the learned bridge score `s_theta(x_t,t,x_T)`; equivalently one trains a data
predictor `x_theta(x_t,t,x_T)` that recovers `x_0`. For sampling, DDBM uses an EDM-style
*hybrid Heun* sampler (Karras et al. 2022) that alternates an Euler/SDE "churn" step with a
second-order Heun correction, with an Euler step ratio (e.g. 0.33) controlling injected
stochasticity. **Gap (observed limitation):** the hybrid Heun sampler is a *generic*
ODE/SDE discretizer applied to the bridge's reverse dynamics; it is not tailored to the
bridge's semi-linear structure and offers no analytic cancellation of the easy part of the
dynamics, so it converges slowly — high-resolution translation still costs over a hundred
denoiser calls.

**DDIM — denoising diffusion implicit models (Song et al. 2021, arXiv:2010.02502).** For
ordinary (Gaussian-prior) diffusion: a family of non-Markovian forward processes
`q_sigma(x_{t-1}|x_t,x_0) = N(sqrt(alpha_{t-1}) x_0 + sqrt(1 - alpha_{t-1} - sigma_t^2)
(x_t - sqrt(alpha_t) x_0)/sqrt(1-alpha_t), sigma_t^2 I)`, all sharing the marginals
`q(x_t|x_0)`, hence the same objective `L_1` (formally `J_sigma = L_gamma + C`). The sampler
reads "predicted `x_0`" + "direction pointing to `x_t`" + `sigma_t` noise; `sigma=0` is
deterministic (an implicit model), a particular `sigma` is DDPM. Rewriting the deterministic
update exposes an Euler-ODE form `d(x/sqrt(alpha)) = eps_theta d(sqrt((1-alpha)/alpha))`.
**Gap:** the entire construction assumes a Gaussian prior with no endpoint — the mean has a
single `sqrt(alpha_{t-1}) x_0` data term and no `x_T` term, the marginals are
`N(sqrt(alpha_t) x_0, ...)`, and there is no fixed endpoint to honor — so it does not, as
written, describe transport between two arbitrary distributions.

**DPM-Solver (Lu et al. 2022a, arXiv:2206.00927).** Exploits the semi-linear diffusion ODE:
variation-of-constants cancels `f(t) x` exactly, the change of variable `lambda = log
(alpha/sigma)` turns the rest into `x_t = (alpha_t/alpha_s) x_s - alpha_t int_{lambda_s}^
{lambda_t} e^{-lambda} eps_theta dlambda`, and Taylor-expanding `eps_theta` in `lambda` makes
each term an *analytic* exponentially-weighted integral; dropping `O(h^{k+1})` gives a
`k`-th-order solver (`h = lambda_t - lambda_s`). It reaches good samples in ~10-20 calls.
**Gap:** it is built on the *noise* predictor and uses *single-step* high-order stages (an
extra intermediate evaluation per step, so `k` denoiser calls per step), and — like all of
the above — it is derived for the Gaussian-prior diffusion ODE, with no endpoint and no
bridge kernel.

**DPM-Solver++ (Lu et al. 2022b, arXiv:2211.01095).** Re-derives the exponential integrator
for the *data* predictor `x_theta` (better-behaved, thresholdable), `x_t = (sigma_t/sigma_s)
x_s + sigma_t int_{lambda_s}^{lambda_t} e^{lambda} x_theta dlambda`, and replaces single-step
stages with a *multistep* (Adams-Bashforth) scheme: the high-order derivatives of `x_theta`
are estimated from *previously computed* outputs, so each step costs exactly one denoiser
call and an `N`-call budget buys `M = N` steps (vs. `N/k` for single-step). **Gap:** still the
Gaussian-prior diffusion ODE with `lambda = log(alpha/sigma)` and the linear factor
`sigma_t/sigma_s` — there is no endpoint term, so it does not apply to a bridge whose linear
dynamics and log-SNR surrogate are different.

## Evaluation settings

The natural yardsticks for image-to-image translation and restoration that exist at the time:

- **Edges→Handbags** (Isola et al. 2017) and **DIODE-Outdoor** (Vasiljevic et al. 2019)
  translation tasks at 64×64 / 256×256; **ImageNet 256×256 center inpainting** with a
  128×128 mask (Deng et al. 2009) as a restoration task. Source `x_T` is the condition
  (edges / depth / masked image), target `x_0` the realistic image.
- **Metric: Fréchet Inception Distance (FID)** between generated and reference images, lower
  is better (Heusel et al. 2017). For restoration, a pixel-level **diversity score** (the
  standard deviation across several samples per condition; Batzolis et al. 2021) measures
  whether the sampler preserves the task's genuine one-to-many stochasticity.
- **Budget: number of function evaluations (NFE)** — the count of denoiser calls per sample,
  the dominant runtime cost (runtime is essentially proportional to NFE, since coefficient
  arithmetic is negligible). Methods are compared at matched, small NFE.
- **Protocol:** the same pretrained bridge predictor is shared across all samplers; only the
  transition rule changes. Timesteps run from `t_max = 1` down to `t_min = 1e-4`, and the
  sampler receives them as a decreasing schedule.

## Code framework

The sampler plugs into the existing bridge codebase. The trained data predictor `x_theta`
(wrapped as `denoiser`), the noise schedule (which exposes the bridge coefficients
`a_t, b_t, c_t` and the base `alpha_t, sigma_t`/`rho_t`), and the bridge forward sampler
`x = a_t x_T + b_t x_0 + c_t * noise` already exist. The denoiser is metered: it may be
called at most `len(ts)` times per sample. What does *not* exist is the rule that advances the
state from one timestep to the next under a tiny call budget — that rule is the single empty
slot. The existing bridge harness has one transition stub.

```python
import torch
import torch.distributed as dist
from tqdm.auto import tqdm


def append_dims(x, target_dims):
    """Broadcast a per-sample scalar up to the image tensor's rank."""
    return x[(...,) + (None,) * (target_dims - x.ndim)]


class NoiseSchedule:
    """Already-trained bridge schedule. Exposes the bridge transition coefficients."""

    def get_alpha_rho(self, t):
        # returns (alpha_t, alpha_bar_t, rho_t, rho_bar_t) of the base VP/VE schedule
        raise NotImplementedError

    def get_abc(self, t):
        # bridge kernel q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t^2 I)
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        a_t = (alpha_bar_t * rho_t ** 2) / self.rho_T ** 2
        b_t = (alpha_t * rho_bar_t ** 2) / self.rho_T ** 2
        c_t = (alpha_t * rho_bar_t * rho_t) / self.rho_T
        return a_t, b_t, c_t


class Diffusion:
    """Owns the schedule and the bridge forward sampler. Already trained."""

    def __init__(self, noise_schedule, t_max=1.0, t_min=1e-4):
        self.noise_schedule = noise_schedule
        self.t_max, self.t_min = t_max, t_min

    def bridge_sample(self, x0, xT, t, noise):
        a_t, b_t, c_t = [append_dims(v, x0.ndim) for v in self.noise_schedule.get_abc(t)]
        return a_t * xT + b_t * x0 + c_t * noise


@torch.no_grad()
def sample_custom_bridge(denoiser, diffusion, x, ts, mask=None, seed=None, **kwargs):
    """Advance x_T -> x_0 along the decreasing schedule `ts`, calling `denoiser`
    at most len(ts) times. The transition rule is what we design."""
    x_T = x
    path, pred_x0, nfe = [], [], 0
    ones = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        s, t = ts[i], ts[i + 1]            # current and next (smaller) timestep
        x0_hat = denoiser(x, s * ones)     # predicted clean data x_theta(x_s, s, x_T)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)
        nfe += 1

        # TODO: the transition rule we will design — given x at time s and the
        #       predicted x0_hat (and whatever per-step state we choose to keep),
        #       advance to x at time t under the bridge dynamics.
        pass

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())

    return x, path, nfe, pred_x0, ts, x_T
```

The outer loop supplies one predictor call per step; the transition rule is the slot to fill.
