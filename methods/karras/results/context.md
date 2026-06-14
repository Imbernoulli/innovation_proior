# Context: time-step discretization for diffusion ODE sampling (circa 2021-2022)

## Research question

Diffusion-based image generators synthesize a sample by starting from pure Gaussian noise and
sequentially denoising it down to clean data. Concretely, generation is the numerical solution of
a differential equation whose right-hand side is a learned denoiser network `D(x; sigma)`: we
follow a trajectory through a sequence of decreasing noise levels
`sigma_0 = sigma_max > sigma_1 > ... > sigma_{N-1} = sigma_min`, ending at clean data. Every step
of the solver evaluates the network once (or twice for a two-stage method), and these *neural
function evaluations* (NFE) dominate the entire cost of sampling — the network is by far the most
expensive object in the loop. So the practical figure of merit is image quality at a *small* number
of steps `N`.

The component that decides where those few steps land is the **time-step discretization** `{t_i}`,
equivalently the sequence of noise levels `{sigma_i}` (with `t_i = sigma^{-1}(sigma_i)`). It sets how
the step sizes — and therefore the per-step integration error — are distributed across the noise
range. A numerical ODE solver only ever approximates the true trajectory: each step incurs a *local
truncation error* that accumulates into a *global* error, and with a small step budget that error is
large enough to visibly corrupt the output even when the solver update rule and the trained network
are held fixed. The precise problem: choose `{sigma_i}` (and the solver) so that the accumulated
truncation error at a given `N` is as small as possible — which is what lets `N` itself be pushed
down — and do so in a way that is decoupled from how the network happened to be trained, so that the
same sampling recipe transfers across models. The schedules in use at the time were inherited from
each model's *training-time* forward noising process rather than chosen for sampling efficiency, and
none of them exposed a way to deliberately reallocate resolution between high and low noise.

## Background

The field state rests on a small number of load-bearing ideas.

**Score-based generative modeling and the probability flow ODE (Song et al. 2021).** Define a family
of mollified distributions `p(x; sigma) = p_data * N(0, sigma^2 I)` obtained by convolving the data
with isotropic Gaussian noise of standard deviation `sigma`. For `sigma >> sigma_data` this is
indistinguishable from pure noise; at `sigma = 0` it is the data. Song et al. give a forward SDE
`dx = f(t) x dt + g(t) dw` whose marginals are exactly these `p(x; sigma(t))`, and a corresponding
deterministic *probability flow ODE* with the same marginals,
`dx = [f(t) x - (1/2) g(t)^2 grad_x log p_t(x)] dt`. The term `grad_x log p(x; sigma)` is the
**score function** (Hyvarinen 2005), a vector field pointing toward higher data density at noise level
`sigma`. Reparameterizing the ODE directly in terms of the marginals (a noise-level schedule
`sigma(t)` and an optional scaling `s(t)`) turns it into
`dx = [s_dot/s x - s^2 sigma_dot sigma grad_x log p(x/s; sigma)] dt`; every realization of the flow is
then just a reparameterization of one canonical ODE, with `sigma(t)` reparameterizing time and `s(t)`
reparameterizing `x`.

**Denoising score matching (Hyvarinen 2005; Vincent 2011).** The score need not be modeled directly.
If `D(x; sigma)` minimizes the expected `L2` denoising error
`E_{y~p_data} E_{n~N(0,sigma^2 I)} || D(y + n; sigma) - y ||^2` separately for each `sigma`, then
`grad_x log p(x; sigma) = (D(x; sigma) - x) / sigma^2`. So a network trained as a denoiser supplies
the score for free, and the ODE's right-hand side is built from denoiser evaluations.

**Numerical integration of ODEs (Suli & Mayers 2003; Ascher & Petzold 1998).** Solving the ODE means
taking finite steps over `{t_i}`. Euler's method is first order: its local truncation error obeys
`||tau_i|| = O(h_i^2)` in the step size `h_i`, so halving the step cuts the error fourfold. Higher-order
Runge-Kutta methods reduce the error exponent at the cost of extra right-hand-side evaluations per step;
Heun's method (improved Euler / explicit trapezoidal rule, a two-stage second-order RK scheme) adds a
single correction evaluation and achieves `||tau_i|| = O(h_i^3)`. Under a Lipschitz right-hand side the
global error is bounded by the worst single-step error, `||e_N|| <= E max_i ||tau_i||`, with `E`
depending on `N`, the endpoints, and the Lipschitz constant. The local error scales with the curvature
of `dx/dt`, so step sizes and the schedule directly control how much error each region of the trajectory
contributes.

**Diagnostic measurements of where the error lives.** The per-step truncation error can be measured
directly: fix a step from `sigma_{i-1}` to `sigma_i`, estimate the true `x_i` by many tiny Euler
sub-steps, and compare. On a pretrained variance-exploding CIFAR-10 model this measurement shows the
single-step error is very large at low noise (root-mean-square error around 0.56 for `sigma <= 0.5` with
an evenly-spaced-in-`sigma` step of size 1.25) and much smaller at high noise. It also shows the error is
nearly constant with respect to the particular noisy sample `x_{i-1}` (the spread across samples is tiny),
so a single fixed schedule suffices and there is no benefit to varying `{t_i}` per sample. These are
pre-method facts about how an existing solver behaves on an existing model.

**Trajectory geometry.** An already-known deterministic parametrization takes `sigma(t) = t` and
`s(t) = 1`, which collapses the ODE to `dx/dt = (x - D(x; t)) / t`. A single Euler step from any
`(x, t)` straight to `t = 0` therefore lands on the denoiser output `D(x; t)`: the tangent of the
solution trajectory always points at the current denoised estimate. Since that estimate changes only
slowly with noise level over most of the range, the trajectories are close to straight at both large and
small `sigma` and have appreciable curvature only in a narrow middle band — and low curvature is exactly
what keeps truncation error small.

## Baselines

The schedules a new discretization would be compared against, each a real recipe with a concrete gap.

**Evenly spaced in `sigma` (linear).** Place the noise levels at equal spacing,
`sigma_i = sigma_max + (i/(N-1))(sigma_min - sigma_max)`. Simplest possible choice. *Limitation:* it
ignores where the integration error actually lives. The diagnostic measurement above shows the single-step
error is an order of magnitude larger at low noise than at high noise, so equal spacing pours a fixed step
size into a region (low `sigma`) that demands small steps and wastes resolution where the field is easy
(high `sigma`); with few steps the low-noise error dominates the result.

**Variance-exploding geometric schedule (Song et al. 2021; SMLD lineage).** The original variance-exploding
sampler uses noise levels equally spaced in log-`sigma`,
`sigma_bar_i = sigma_min (sigma_max/sigma_min)^{1 - i/(N-1)}`, i.e. a geometric sequence. Brought into the
common ODE framework this corresponds to `s(t) = 1`, `sigma(t) = sqrt(t)`, `t_i = sigma_bar_i^2`.
*Limitation:* it is a single fixed shape. It packs steps tightly toward low `sigma` and produces
trajectories that are strongly curved near the data and, in fact, curved throughout the range; it offers no
control to trade step resolution between high and low noise, and its shape was dictated by the forward
diffusion rather than by the integration error of the reverse solve.

**Cosine schedule from improved DDPM (Nichol & Dhariwal 2021).** DDPM (Ho et al. 2020) defines a forward
Markov chain with a variance schedule `{beta_t}` giving cumulative
`alpha_bar_t = prod_{s<=t}(1 - beta_s)`; improved DDPM replaces the linear `beta` schedule with a cosine
one, `alpha_bar_t = sin^2( (pi/2) j / (M(1 + C_2)) )` after reindexing, clamped to avoid singularities.
Mapped into noise levels this yields a sampling schedule by indexing into the model's `M = 1000` originally
trained levels (each `t_i` rounded to a supported level). *Limitation:* it is a *training-time*
discretization, designed to allocate the forward noising steps to the perceptually important content range,
and it is tied to the discrete grid the model was trained on — it is not derived from, and does not target,
the per-step integration error of the sampling ODE.

**Variance-preserving / DDPM-style schedule (Ho et al. 2020; Song et al. 2021).** The variance-preserving
formulation maps to `sigma(t) = sqrt(e^{(1/2) beta_d t^2 + beta_min t} - 1)` with steps taken uniformly in
`t`. *Limitation:* its trajectories flatten into near-horizontal lines at large `sigma` (the local gradients
only begin pointing toward data at small `sigma`), so a large fraction of the step budget is spent in a
regime that contributes little; like the others it is inherited from the forward process, not chosen to
minimize sampling error.

Across all four, the common gap is the same: the discretization is a fixed shape handed down from the
model's training-time noise process, with no single interpretable control for *how* to distribute the few
sampling steps across the noise range so as to put the integration error where it costs least.

## Evaluation settings

The natural yardsticks already in use, all pre-method:

- **Models / datasets.** Pretrained score networks spanning the different theoretical families: the
  variance-preserving and variance-exploding continuous models of Song et al. (2021) on unconditional
  CIFAR-10 at 32x32, and a class-conditional ImageNet model at 64x64 trained in the improved-DDPM
  formulation (a discrete set of `M = 1000` noise levels). The samplers are evaluated as drop-in
  replacements on these *frozen* networks, treating the denoiser as a black box.
- **Quality metric.** Frechet Inception Distance (FID; Heusel et al. 2017) between 50,000 generated images
  and the real images — lower is better.
- **Cost metric.** Number of neural function evaluations (NFE) per image, since sampling cost is dominated
  by network calls; quality is reported as a function of NFE so a schedule is judged on the quality it buys
  per unit of compute.
- **Diagnostic instrument.** Per-step local truncation error measured by comparing a single solver step
  against a finely sub-stepped reference, reported as RMSE versus noise level — used to see where the error
  concentrates and how the schedule and solver order move it.
- **Protocol.** Same frozen network and same noise-level range across schedules; only the discretization
  (and, separately, the solver) is varied, so differences are attributable to the schedule.

## Code framework

The schedule plugs into an existing diffusion sampling harness. The data pipeline, the trained denoiser
network, the ODE solver loop, and the FID evaluation already exist; the one undecided piece is the function
that, given a step budget and the noise-level endpoints, returns the sequence of noise levels the solver
will visit. That function is the single empty slot.

```python
import torch


def get_schedule(n, sigma_min, sigma_max, device="cpu"):
    """Return the sequence of noise levels the sampler steps through.

    Requirements fixed by the harness:
      - return a 1D tensor of length n + 1 (the nodes sigma_0 ... sigma_n),
      - strictly decreasing from sigma_max down to the terminal level,
      - the first element is sigma_max,
      - the tensor lives on the requested device.
    """
    # TODO: the rule that places the noise levels across [sigma_min, sigma_max].
    ...


def sample(denoiser, n, sigma_min, sigma_max, device="cpu"):
    """Existing sampling loop: integrate dx/dt = (x - D(x; sigma)) / sigma
    down the schedule, evaluating the denoiser at each node."""
    sigmas = get_schedule(n, sigma_min, sigma_max, device=device)   # the slot above
    x = torch.randn(...) * sigmas[0]                                # start at sigma_max
    for i in range(n):
        sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
        d = (x - denoiser(x, sigma_cur)) / sigma_cur               # ODE right-hand side
        x = x + (sigma_next - sigma_cur) * d                       # solver step (Euler shown)
        # (a higher-order corrector may add one more denoiser eval here)
    return x
```

The outer loop and the denoiser are fixed; `get_schedule` is where the placement rule will live.
