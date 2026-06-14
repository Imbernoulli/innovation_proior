# Context: few-step sampling of denoising / bridge generative models

## Research question

A trained denoising generative model turns noise into data by repeatedly calling a learned
denoiser `eps_theta(x_t, t)` and stepping a state `x_t` along a one-dimensional time axis
`t`, from a noisy endpoint `t_max` down to the data endpoint `t_min`. The cost of generating
a single sample is dominated entirely by the number of denoiser evaluations (NFE) — each call
is a full forward pass through a large U-Net, and the calls are sequential. The dominant
recipe spends *one denoiser call per training time index*, which means a thousand-ish
sequential calls per sample; this is the bottleneck that makes these models orders of
magnitude slower than a single-pass generator.

Suppose the denoiser is already trained and frozen, and the per-step update rule of the
sampler is fixed. The only remaining freedom is **which `t` values to evaluate the denoiser
at** — a short, ordered list of times `t_max = t_0 > t_1 > ... > t_n = t_min` at which the
sampler will stop, call the denoiser, and take a step. Under an extremely tight budget (say
`n = 5` denoiser calls), this choice becomes quality-critical even though nothing about the
model or the update rule changes. The precise problem: given the budget `n`, the interval
endpoints `t_max`, `t_min`, and a frozen model and
sampler, produce the ordered list of `n + 1` evaluation times that spends those calls well —
with the hard requirements that the list strictly decrease, start at `t_max`, end *exactly* at
`t_min`, and carry no constants secretly tuned to one dataset (it must generalize across
workloads). What that list should be is the open question.

## Background

The field state rests on a small stack of load-bearing ideas.

**Diffusion as a forward-corruption / reverse-denoising pair.** A forward process gradually
turns data `x_0` into noise; for the Gaussian case the marginal at time `t` is available in
closed form, `q(x_t | x_0) = N(sqrt(alpha_t) x_0, (1 - alpha_t) I)`, so a noisy state can be
written `x_t = sqrt(alpha_t) x_0 + sqrt(1 - alpha_t) eps` with `eps ~ N(0, I)`. A network
`eps_theta(x_t, t)` is trained to predict the noise `eps` from `x_t`; this is the denoising /
score-matching objective. Generation runs the corruption in reverse: start from noise at the
high-noise endpoint and progressively denoise down to data.

**The objective depends only on the per-time marginals.** The standard training loss is a sum
over times of `E‖eps - eps_theta(sqrt(alpha_t) x_0 + sqrt(1 - alpha_t) eps, t)‖^2`. It is a
function of the *marginals* `q(x_t | x_0)` alone — it never references how consecutive noisy
states are jointly coupled. This is the pivotal structural fact: the trained denoiser is a
function of `t` over a whole interval, not a fixed-length chain. Once trained, it can in
principle be queried at *any* set of times whose marginals it has seen.

**The continuous-time / ODE view.** Taking the number of time indices to infinity turns the
forward process into a stochastic differential equation, and there is an associated
deterministic *probability-flow ODE* `dx = [f(x, t) - (1/2) g(t)^2 ∇_x log p_t(x)] dt` whose
trajectories carry the same marginals (Song et al. 2021). The learned denoiser supplies the
score `∇_x log p_t`, so sampling becomes: numerically integrate this ODE backward in time from
`t_max` to `t_min`. A numerical ODE/SDE integrator advances the state across each interval
between two consecutive evaluation times; the evaluation times are the integrator's node
placement, and the intervals between them determine the denoiser-call count. A black-box
adaptive solver can already cut the evaluation count by a large factor by spending accuracy
where the trajectory is hard — which says plainly that *where the nodes sit on the time axis*
is the lever that controls cost-versus-accuracy.

**Bridge-style samplers.** The same scheduling interface appears when the endpoints are a paired
source and target rather than pure noise and data. The state still evolves along a scalar time
`t ∈ [t_min, t_max]`, the denoiser is still queried at chosen times, and the sampler still
marches a fixed update rule across the intervals between them. The time-placement question is
identical; only the update rule's coefficients differ.

**Local error of a one-step integrator.** A first-order (Euler-type) step across an interval
of length `Δt` incurs a local truncation error that scales like `(Δt)^2` times the local
curvature of the trajectory (its second derivative in `t`); a second-order step scales like
`(Δt)^3`. The total sampling error is the accumulation of these per-interval errors. With a
fine grid the per-interval `Δt` is tiny and the error is negligible; with only a handful of
intervals each `Δt` is large, so the error becomes first-order sensitive to how the intervals
are sized and to where the trajectory curves the most.

## Baselines

These are the prior approaches a new time-placement scheme is measured against or reacts to.

**Full-length reverse Markov chain (DDPM, Ho, Jain & Abbeel 2020).** The generative process is
a Markov chain with one learned Gaussian transition per forward time index. Sampling runs
`for t = T, ..., 1: x_{t-1} = (1/sqrt(α_t))(x_t - ((1-α_t)/sqrt(1-ᾱ_t)) eps_theta(x_t, t)) +
σ_t z`. Each reverse transition is trained to invert exactly one forward step, so the chain's
length is welded to the training length `T` (≈ 1000). Core idea: a long chain of small Gaussian
steps is easy to learn and gives high-quality samples. **Gap:** the chain must be walked in
full — every one of the `T` indices is visited, at one denoiser call apiece — because dropping
indices breaks the step-by-step inversion the transitions were trained for. Sampling is
sequential and on the order of a thousand calls; there is no built-in way to spend fewer.

**Reusing the trained denoiser on a shorter index set.** Because the training loss depends only
on the marginals, one can
construct a family of generative processes — including a deterministic one — that share the
same trained denoiser and the same loss, and that are *not* tied to the full chain. In
particular the generative process can be defined over a strictly increasing subset of the time
indices rather than all of them, and the deterministic member is an Euler step of the
probability-flow ODE. Core idea: decouple the number of sampling steps from the training
length, and run the frozen model on any chosen ordered subset of times. **Gap:** this buys the
*freedom* to use a short list of times and shows few-step sampling is possible, but it leaves
the actual *placement* of those few times as a free choice; under a tiny budget the placement
is precisely what determines whether the few large steps land well or accumulate error, and
the decoupling result itself says nothing about where to put them.

**Curvature-warped time grids (sibling schedules).** Several schedules place the few evaluation
times non-uniformly by mapping an even grid through a fixed warp, on the premise that the
trajectory varies fastest in some transformed coordinate. The EDM power-law schedule (Karras et
al. 2022) spaces times evenly in `t^{1/ρ}` (with `ρ = 7`), packing more nodes toward the
high-noise end; a log-spaced schedule places times evenly in `log t`, packing nodes toward the
data end; a cosine-spaced schedule packs nodes toward the middle of the range. Core idea: if
you know which region of the trajectory carries the curvature, concentrate the limited nodes
there to shrink the dominant `(Δt)^2` error terms. **Gap:** each warp encodes a fixed
assumption about *where* the trajectory is hardest, and each introduces a shape parameter (the
exponent `ρ`, the log base, the cosine phase). When the assumption matches the workload it
helps; when it does not, it spends nodes in the wrong place and can do worse — and the chosen
constants risk being tuned to one dataset rather than transferring. Warped closed forms also
need explicit endpoint handling when the formula itself does not land exactly on `t_min`.

## Evaluation settings

- **Image-to-image bridge workloads.** Paired-image translation tasks such as edges→handbags
  (`edges2handbags`), and depth/normal-style paired sets (DIODE), plus an ImageNet-scale paired
  setting; a single trained bridge model per workload, frozen across all schedules compared.
- **Budget.** A fixed, very small number of denoiser calls per sample (`NFE = 5`); the sampler
  update rule, the number of allowed calls, the dataset handling, and the metric computation are
  all held fixed — only the list of evaluation times changes.
- **Metric.** Fréchet Inception Distance (FID) between generated and reference images, lower is
  better; FID is known to be sensitive to residual noise and high-frequency artifacts, so a
  schedule that leaves the trajectory unfinished or oversteps a delicate region is penalized.
- **Protocol.** Same frozen model and sampler for every schedule; a schedule is expected to
  generalize across the workloads rather than encode constants fit to one of them; fixed seed.

## Code framework

The schedule plugs into a frozen bridge sampler. The harness already exists: a trained denoiser
wrapped so that, given the current state and a time, it returns a denoised prediction; a sampler
that marches a fixed update rule across consecutive times; and a top-level routine that asks for
the list of evaluation times and then hands it to the sampler. The one thing not yet decided is
the body that produces that list of times — that is the slot to fill.

```python
import torch


def get_sampling_times(n, t_min, t_max, device="cpu"):
    """Produce the ordered list of times at which the frozen sampler will call the denoiser.

    Contract enforced by the harness:
      1. Length:      return a 1-D torch.Tensor of exactly length n + 1.
      2. Monotonic:   the sequence must strictly decrease from t_max to t_min.
      3. Terminal:    the final element (index n) must equal t_min exactly.
      4. Device:      the returned tensor must live on the requested device.
    For this task n = 5 (NFE = 5).
    """
    # TODO: the rule that decides where the n + 1 evaluation times sit on [t_min, t_max].
    pass


# --- existing frozen machinery the schedule plugs into (do not change) ---

def make_denoiser(model, diffusion, clip_denoised=True, **model_kwargs):
    @torch.no_grad()
    def denoiser(x_t, t):
        _, denoised, _ = diffusion.denoise(model, x_t, t, **model_kwargs)
        return denoised.clamp(-1, 1) if clip_denoised else denoised
    return denoiser


@torch.no_grad()
def sample(denoiser, diffusion, x_T, n, t_min, t_max, device, step_fn):
    ts = get_sampling_times(n, t_min, t_max, device=device)   # <-- the slot above
    x = x_T
    for i in range(len(ts) - 1):                              # one denoiser call per interval
        x = step_fn(denoiser, diffusion, x, ts[i], ts[i + 1]) # fixed bridge update rule
    return x.clamp(-1, 1)
```

The sampler consumes the `n + 1` times and takes `n` fixed steps between them; the only thing
left to specify is how `get_sampling_times` lays those times out on the interval.
