# Context: sampling diffusion bridges for image-to-image translation (circa 2024)

## Research question

Image-to-image translation — sketch→photo, depth→RGB, deblurring, inpainting — asks for a model
that transports a *source* image distribution `π_cond` to a *target* image distribution `π_data`,
given paired examples. Diffusion bridges do this directly in pixel space: instead of mapping
target images to Gaussian noise and back (the standard diffusion route), they build a stochastic
path whose two ends are the source and the target. Because the source and target images already
look a lot alike — far more than a target image looks like white noise — the path is short, and
short paths are cheap to traverse and easy to learn.

The practical problem is that the bridge families that achieve high image quality are
expensive and rigid. The transition kernel that defines a bridge is parameterized by quantities
that are tied together, so the space of paths one can actually build is artificially small.
Sampling a high-quality bridge can require hundreds of network evaluations, and the fast samplers
that exist impose a positivity condition that forbids some strong stochasticity schedules.
On top of this, a translation that should be *one-to-many* — one edge map admits many plausible
handbags, in different colors and textures — collapses to near-identical outputs across random
seeds. A solution has to deliver, simultaneously: (1) a flexible path family with parameters one
can tune independently; (2) a sampler that reaches good FID in a *small* number of denoiser calls
(the regime of interest is on the order of five); (3) the freedom to set any stochasticity level
along the path, including ones the fast samplers cannot reach; (4) sharp endpoints (no residual
noise smearing the final image); and (5) a way to restore conditional diversity in one-to-many
tasks. The existing families each give a subset; none gives all five.

## Background

A diffusion bridge is built from a *reference* diffusion process `dX_t = f̄_t X_t dt + ḡ_t dW_t`
on `t ∈ [0,T]`, whose forward transition kernel `q_{t|0}(x_t|x_0) = N(x_t; a_t x_0, σ_t² I)` is
Gaussian. Doob's `h`-transform (Doob 1984; Rogers & Williams 2000) "pins" this process so that it
is conditioned to arrive almost surely at a fixed terminal point `x_T = y`: it adds a
time-inhomogeneous drift `ḡ_t² ∇ log p_{T|t}(x_T | X_t)` to the reference drift. The pinned
process is a valid bridge — it transports `π_0` (target) to `π_T` (source) and back — and its
*bridge* transition kernel, conditioned on both ends, is again Gaussian:

```
p_{t|0,T}(x_t | x_0, x_T) = N(x_t; α_t x_0 + β_t x_T, γ_t² I).
```

To generate a target from a source, one solves a reverse-time SDE or its probability-flow ODE
from `t = T` to `t = 0`, both of which need the score `∇_{x_t} log p_t(x_t | x_T)`. The score is
intractable in closed form but can be estimated by denoising bridge score matching, regressing a
network against the analytic kernel score.

Two background facts shape every design choice below.

**A denoiser is the natural reparameterization of the score.** Karras et al. (2022, EDM) observed
that for a Gaussian-corrupted variable, the score is an affine function of the L2-optimal denoiser
`D(x; σ) = E[clean | noisy]`: the score points from the current state toward the predicted clean
image. Parameterizing the network to predict the clean signal `x̂_0`, rather than the raw score,
avoids the `1/γ²` blow-up of the score at the endpoints (`γ_t → 0` as `t → 0` and `t → T`), which
otherwise makes training unstable. EDM also showed that *training and sampling can be decoupled*:
with a fixed trained denoiser, the sampler — its discretization, its noise schedule, its
time-step spacing — is a separate design problem with its own large design space, including a
stochastic sampler whose injected noise is a tunable knob.

**Many distinct dynamics can realize the same interpolating marginals.** Stochastic interpolants
(Albergo, Boffi & Vanden-Eijnden 2023) separate the specification of the time-indexed marginals
from the ODE or SDE used to realize them. Their viewpoint says that an interpolant's law can be
transported by deterministic and stochastic dynamics whose drifts are learned from data, so the
process dynamics need not be inherited from a pinned reference diffusion. What is still missing
for paired translation is the bridge-specific graft: conditioning on the observed terminal image,
using a single denoiser that sees the source image, and keeping the sampler valid under a hard
small-NFE budget.

Three diagnostic observations about existing systems set up the problem. First, the vanilla
reverse-SDE sampler of a high-quality bridge needs on the order of a hundred-plus denoiser
calls; under a five-call budget its FID is enormous. Second, the existing fast bridge sampler is
derived under a positivity condition `γ_{t-Δt}² - 2 ε_t Δt > 0`; strong stochastic schedules can
violate it, so the fast sampler has settings where its square root is undefined. Third, holding a
source image fixed and resampling, existing bridges produce visually near-identical outputs — the
injected diffusion noise barely changes the sample — so one-to-many translation loses its
diversity.

## Baselines

**DDBM — Denoising Diffusion Bridge Models (Zhou et al. 2023).** The canonical pinned-process
bridge. Builds the kernel above with VP or VE instantiations: for VP, `α_t = a_t(1 − SNR_T/SNR_t)`,
`β_t = (a_t/a_T)(SNR_T/SNR_t)`, `γ_t² = σ_t²(1 − SNR_T/SNR_t)`, where `SNR_t = a_t²/σ_t²`. Its
reverse process is

```
dX_t = [f̄_t X_t + ḡ_t²(s − h)] dt + ḡ_t dW_t   (SDE),
dX_t = [f̄_t X_t + ḡ_t²(s − ½h)] dt              (ODE),
```

with `s = ∇log p_{T|t}(x_T|X_t)`, `h = ∇log p_{t|T}(X_t|x_T)`, the score estimated by a denoiser.
Substantive and effective, but limited: the path is controlled by only the two functions `a_t`
and `σ_t`, which are *convolved* inside `α_t, β_t, γ_t` — they cannot be moved independently. And
DDBM ships exactly one reverse SDE plus its one ODE; it offers no dial for how much noise to
inject during sampling, and its vanilla SDE sampler is expensive (hundreds of NFEs), with FID
that is unusable at a five-call budget.

**DBIM — Diffusion Bridge Implicit Models (Zheng et al. 2024).** The bridge counterpart of DDIM.
Observes that the bridge score depends only on the marginals, so one can build *non-Markovian*
bridges that share the same `N` marginals as DDBM's process yet permit large jumps between
time-steps. This yields a closed-form update on the discretized schedule
`0 = t_0 < … < t_N = T`:

```
x_{t_n} = α_{t_n} x̂_0 + β_{t_n} x_T
        + √(γ_{t_n}² − ρ_{t_n}²) · (x_{t_{n+1}} − α_{t_{n+1}} x̂_0 − β_{t_{n+1}} x_T) / γ_{t_{n+1}}
        + ρ_{t_n} ε,   ε ~ N(0, I),
```

with `ρ_{t_n}` a per-step stochasticity level interpolating deterministic (`ρ = 0`) to stochastic,
and a "booting noise" first step for diversity. Up to ~25× faster than DDBM's sampler. Two
limitations matter here. It is derived *inside* DDBM's coupled kernel, so it inherits the same
restricted path family. And its stochasticity is bounded by the marginals through the
`√(γ² − ρ²)` term — equivalently the positivity condition `γ_{t-Δt}² − 2 ε_t Δt > 0`. A strong
noise schedule (one matching the full reverse-SDE injection) can violate this condition, at which
point the update's square root becomes imaginary and the sampler is undefined; it cannot reach
those settings at all.

**I2SB — Image-to-Image Schrödinger Bridge (Liu et al. 2023).** A tractable Schrödinger-bridge
construction for restoration. Strong on inverse problems, but it is one specific Markovian bridge:
its sampler is a single point in the design space (the case where the `x_T` coefficient in a
DBIM-style update vanishes), again with no independent control of the path or the sampler noise,
and it too runs at high NFE.

**Stochastic Interpolants (Albergo, Boffi & Vanden-Eijnden 2023).** Builds transport directly via
the flow map `x_t = α_t x_0 + β_t x_T + γ_t z`, `z ~ N(0,I)`, with `α_t, β_t, γ_t` *decoupled* and
free to design subject only to boundary conditions `α_0 = β_T = 1`, `α_T = β_0 = γ_0 = γ_T = 0`.
Far larger and cleaner path design space than the `h`-transform constructions, and the framework
that supplies ODE/SDE realizations for interpolating marginals. Limitation: as formulated for
generation it is *unconditional* — it does not condition on a terminal endpoint `x_T`, so for
paired translation it requires training two separate models (one per direction) and has not taken
on the bridge community's practical machinery (endpoint conditioning, denoiser preconditioning,
fast bridge samplers).

**EDM (Karras et al. 2022).** Not a bridge, but the methodological backbone: the
score↔denoiser reparameterization, the unit-variance preconditioning of network inputs/targets,
the decoupling of training from sampling, and a stochastic sampler with a controllable churn knob.
Its time-step schedule `t_i = (t_max^{1/ρ} + (i/N)(t_min^{1/ρ} − t_max^{1/ρ}))^ρ` is the standard
way to place a small number of steps non-uniformly. Limitation: it is an *unconditional* diffusion
recipe; nothing in it is about bridges or endpoint conditioning.

## Evaluation settings

The natural yardsticks already in use for bridge I2I:

- **Datasets.** Edges→Handbags (Isola et al. 2017) at 64×64 — a one-to-many task where one edge
  map admits many handbags; DIODE-Outdoor (Vasiljevic et al. 2019) at 256×256 (depth→RGB);
  ImageNet (Deng et al. 2009) deblurring; and inpainting/restoration workloads with a known-pixel
  `mask` that must be preserved.
- **Metrics.** Fréchet Inception Distance (FID; Heusel et al. 2017), lower better, as the primary
  image-quality measure; also Inception Score, LPIPS (Zhang et al. 2018), and pixel MSE. For
  conditional diversity, an Average Feature Distance over multiple samples from a fixed source.
- **Budget.** Number of function evaluations (NFE) = number of denoiser calls per sample. The
  regime of interest is small NFE (5, 10, 20), measured by an external counter that rejects any
  run exceeding the budget. The trained denoiser, the VP schedule constants, the preconditioning,
  and the time-step placement are all fixed inputs to the sampler.

## Code framework

A trained bridge denoiser and a noise schedule already exist; the open slot is the transition
that turns them into a target image under a hard cap on denoiser calls. The substrate below is the
generic bridge-sampling harness: a `NoiseSchedule` that exposes, at any time `t`, the Gaussian
kernel coefficients `(a_t, b_t, c_t)` (so `x_t = a_t x_T + b_t x_0 + c_t · noise`) and the raw
drift/diffusion `(f_t, g²_t)`; a denoiser wrapped by an NFE counter that raises once the budget is
spent; an EDM-style time-step builder; and the one empty slot — the per-step update rule — that
has to be filled.

```python
import torch
from tqdm.auto import tqdm


class NoiseSchedule:
    """Gaussian bridge kernel  x_t = a_t x_T + b_t x_0 + c_t * noise."""

    def get_alpha_rho(self, t):
        # returns (alpha_t, alpha_bar_t, rho_t, rho_bar_t) for this schedule
        raise NotImplementedError

    def get_f_g2(self, t):
        # returns (f_t, g2_t): the raw drift/diffusion of the reference process
        raise NotImplementedError

    def get_abc(self, t):
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        a_t = (alpha_bar_t * rho_t ** 2) / self.rho_T ** 2     # coef of x_T
        b_t = (alpha_t * rho_bar_t ** 2) / self.rho_T ** 2     # coef of x_0
        c_t = (alpha_t * rho_bar_t * rho_t) / self.rho_T       # coef of noise
        return a_t, b_t, c_t


def get_sigmas_karras(n, t_min, t_max, rho, device="cpu"):
    """EDM-style non-uniform time-step placement on [t_min, t_max]."""
    ramp = torch.linspace(0.0, 1.0, n, device=device)
    min_inv_rho = t_min ** (1.0 / rho)
    max_inv_rho = t_max ** (1.0 / rho)
    return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho


@torch.no_grad()
def sample_bridge(denoiser, diffusion, x, ts, eta=1.0, mask=None, seed=None, **kwargs):
    """x: initial bridge state (source image, possibly with noise). ts: decreasing schedule.
    denoiser may be called at most len(ts) times; the next call raises NFE_BUDGET_EXCEEDED.
    Must return (final_image, path, nfe, pred_x0, ts, first_noise)."""
    x_T = x
    path = [x.detach().cpu()]
    pred_x0 = []
    ones = x.new_ones([x.shape[0]])
    nfe = 0

    for i in range(len(ts) - 1):
        s, t_next = ts[i], ts[i + 1]
        x0_hat = denoiser(x, s * ones)            # one budgeted denoiser call
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        # TODO: fill in one transition from s to t_next.
        #       Inputs are x, x0_hat, x_T, and the kernel/schedule values.
        pass

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

    return x, path, nfe, pred_x0, ts, None
```

The single empty slot is the transition rule: how each denoiser prediction `x0_hat`, together
with the kernel coefficients and a stochasticity schedule, advances the state from one time-step
to the next under the call budget.
