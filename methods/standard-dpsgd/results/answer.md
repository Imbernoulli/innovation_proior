# DP-SGD, distilled

DP-SGD (Differentially Private Stochastic Gradient Descent) trains a model under a worst-case
`(ε, δ)`-differential-privacy guarantee by privatizing the *training process*: at each SGD
step it clips every per-example gradient to a fixed `L2` norm `C` (to bound each individual's
sensitivity), sums the clipped gradients over a randomly subsampled lot, adds Gaussian noise
of standard deviation `σC` to that sum, divides by the lot size, and steps. The total privacy
spent over all `T` steps is tracked by the
**moments accountant** — which composes the moments of the privacy-loss random variable rather
than just its tail — giving a far tighter budget than the generic strong composition theorem
and making single-digit `ε` on a deep, non-convex network feasible.

## Problem it solves

Train a deep, non-convex network (many layers, `10⁴`–`10⁶` parameters) on sensitive data with
a rigorous `(ε, δ)`-DP guarantee at small `ε`, against an adversary who knows the training
procedure, can read the final parameters, and may control all other records. Privatizing the
*endpoint* fails because a non-convex SGD trajectory gives no usable bound on how the final
weights depend on one record; so privacy must be injected *during* training.

## Key idea

Two coupled pieces.

**1. Per-example clipping + Gaussian noise (the mechanism).** A single per-example gradient
`∇_θ L(θ, x)` has unbounded norm, so the gradient sum has unbounded sensitivity and cannot be
privatized directly. Force finite sensitivity by clipping each per-example gradient in `L2`:

```
ḡ(x) = g(x) / max(1, ‖g(x)‖₂ / C).
```

This is the identity when `‖g‖₂ ≤ C` and rescales to norm exactly `C` otherwise. Clipping must
be **per example, before averaging** — the non-private habit of clipping the averaged batch
gradient bounds no individual's influence. After clipping, adding or removing one example
changes the sum `Σ_i ḡ(x_i)` by at most `C`, so the sum has `L2`-sensitivity `C`, and the
Gaussian mechanism adds `N(0, σ²C²I)`. `σ` is a unitless noise multiplier decoupled from `C`.
Examples are drawn as a random **lot** of expected size `L = qN` (subsampling rate `q = L/N`),
which both lowers gradient variance and amplifies privacy.

**2. The moments accountant (the accounting).** Strong composition of `T` steps is loose
because it composes only each step's `(ε, δ)` *tail*, paying a `√(log(T/δ))` factor and a `Tqδ`
term. Instead track the log moment-generating function of the privacy loss
`c(o) = log(Pr[M(d)=o]/Pr[M(d')=o])`:

```
α_M(λ) = max_{aux, d, d'} log E_{o ∼ M(aux,d)}[ exp(λ c(o)) ].
```

It **composes linearly** — `α_M(λ) ≤ Σ_i α_{M_i}(λ)` — and converts to DP via a Markov tail
bound: `M` is `(ε, δ)`-DP with `δ = min_λ exp(α_M(λ) − λε)`. For one subsampled-Gaussian step
(unit sensitivity, `q < 1/(16σ)`, `λ ≤ σ² ln(1/(qσ))`):

```
α(λ) ≤ q²λ(λ+1) / ((1−q)σ²) + O(q³λ³/σ³).
```

Summing over `T` steps and optimizing `λ` in the tail bound yields the calibration theorem.

## Final algorithm

**Algorithm (DP-SGD).** Inputs: examples `{x_1,…,x_N}`, loss `L(θ)`, learning rate `η_t`,
noise multiplier `σ`, lot size `L`, clip threshold `C`.

```
Initialize θ_0 randomly
for t = 1..T:
    L_t  ←  random sample, each example w.p. q = L/N            # subsample a lot
    for i in L_t:  g_t(x_i)  ←  ∇_θ L(θ_t, x_i)                 # per-example gradients
    ḡ_t(x_i)  ←  g_t(x_i) / max(1, ‖g_t(x_i)‖₂ / C)            # per-example L2 clip
    g̃_t  ←  (1/L) ( Σ_i ḡ_t(x_i) + N(0, σ²C²I) )               # aggregate + Gaussian noise
    θ_{t+1}  ←  θ_t − η_t g̃_t                                   # descent
output θ_T and the (ε, δ) spent via the moments accountant
```

## Privacy guarantee

The moments accountant gives:

- **Composability:** for adaptive `M_1,…,M_k`, `α_M(λ) ≤ Σ_i α_{M_i}(λ)` for every `λ`
  (condition on the public past, peel off each fresh-noise step, and upper-bound each
  conditional MGF by its worst-case log moment).
- **Tail bound:** for any `ε > 0`, `M` is `(ε, δ)`-DP with `δ = min_λ exp(α_M(λ) − λε)`
  (Markov on `exp(λ c)`, then split any output set by `{c ≥ ε}`).
- **Single-step subsampled Gaussian:** the explicit second-order term is
  `q²λ(λ+1)/((1−q)σ²)`, and the full lemma gives
  `α(λ) ≤ q²λ(λ+1)/((1−q)σ²) + O(q³λ³/σ³)`. (Binomial expansion of
  `E_{z∼μ}[(μ₀/μ)^{λ+1}]`: the `t=0` term is `1`, the `t=1` term is `0`, the `t=2`
  term reduces via `E_{z∼μ₀}[(1 − exp((2z−1)/2σ²))²] = exp(1/σ²) − 1` to
  `q²λ(λ+1)/((1−q)σ²)`; the `z≤0`, `0≤z≤1`, and `z≥1` tail cases make `t ≥ 3`
  geometrically dominated under `q < 1/(16σ)` and `λ ≤ σ² ln(1/(qσ))`; since the raw
  moment is bounded by `1+r`, `log(1+r) ≤ r` gives the log-MGF bound, and the theorem
  absorbs the remainder into constants.)

**Calibration theorem.** There exist constants `c₁, c₂` such that given `q = L/N` and `T`, for
any `ε < c₁ q²T`, DP-SGD is `(ε, δ)`-DP for any `δ > 0` if

```
σ  ≥  c₂ · q √(T log(1/δ)) / ε.
```

Strong composition would instead require `σ = Ω(q √(T log(1/δ) log(T/δ)) / ε)` — a
`√(log(T/δ))` factor worse. E.g. at `q = 0.01, σ = 4, δ = 10⁻⁵, T = 10⁴`, the moments
accountant reports `ε ≈ 1.26` versus `ε ≈ 9.34` from strong composition.

## Defaults and design choices

- **`L2` clipping** matches the Gaussian mechanism's `L2` sensitivity calibration and least
  distorts gradient direction.
- **Clip per example, before averaging** is what makes the per-example sensitivity finite;
  clipping after averaging (the non-private trick) bounds nothing per-example.
- **Noise std `σC` on the sum** (`σC/L` on the mean) is exactly `σ ×` sensitivity; keeping `σ`
  unitless lets it be calibrated once to the budget while `C` is set from the gradient scale.
- **Subsampling at rate `q`** amplifies privacy (per-step cost scales with `q`) and decouples
  the noise-averaging lot `L` from the hardware batch (compute in small batches, group into a
  lot before noising).
- **Numerics:** integrate `α(λ)` directly for tightness; the optimal `λ` is small, so a grid up
  to `λ ≈ 32` suffices. Fix `T` and the privacy parameters ahead of time and invert the
  accountant before training, rather than changing the privacy target mid-run.

## Working code

The mechanism — compute one flat norm per example, clip by `min(1, C/(||g||+1e-6))`,
add `N(0, (σC)²)` to the summed clipped gradients, then divide by the expected lot size
when using mean reduction:

```python
import torch


class PrivateGradientMechanism:
    """Per-example L2 clipping to max_grad_norm, then Gaussian noise of
    std (noise_multiplier * max_grad_norm) on the summed clipped gradients.
    """

    def __init__(self, max_grad_norm, noise_multiplier, expected_lot_size,
                 loss_reduction="mean", generator=None):
        self.max_grad_norm = max_grad_norm          # C: per-example L2 clip threshold = sensitivity
        self.noise_multiplier = noise_multiplier    # sigma: noise multiplier (privacy knob)
        self.expected_lot_size = expected_lot_size
        self.loss_reduction = loss_reduction
        self.generator = generator

    def _generate_noise(self, reference):
        return torch.normal(
            mean=0.0,
            std=self.noise_multiplier * self.max_grad_norm,
            size=reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=self.generator,
        )

    def aggregate(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # per-example flat L2 norm across all parameters
        per_param_norms = [g.reshape(batch_size, -1).norm(2, dim=1) for g in per_sample_grads]
        norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)        # [B]

        # clip factor min(1, C / (||g|| + small floor))
        clip_factor = (self.max_grad_norm / (norms + 1e-6)).clamp(max=1.0)

        noised_grads = []
        for g in per_sample_grads:
            clip_factor_on_device = clip_factor.to(g.device).to(g.dtype)
            summed = torch.einsum("i,i...", clip_factor_on_device, g)      # clipped sum
            # Gaussian mechanism: sum has sensitivity C, so std on the sum is sigma*C
            grad = summed + self._generate_noise(summed)
            if self.loss_reduction == "mean":
                grad = grad / self.expected_lot_size
            noised_grads.append(grad)

        return noised_grads

    def get_effective_sigma(self, step, epoch):
        return self.noise_multiplier
```

The accountant + calibration that fix `σ` to the budget before training:

```python
import math


def compute_epsilon(steps, sigma, q, delta, orders=None):
    """Accumulated epsilon from the sampled-Gaussian log-moment bound."""
    if orders is None:
        orders = range(1, 64)
    best_eps = float("inf")
    for lam in orders:
        if lam <= 0:
            continue
        per_step_alpha = q * q * lam * (lam + 1) / ((1.0 - q) * sigma * sigma)
        total_alpha = steps * per_step_alpha                              # composes linearly
        eps = (total_alpha + math.log(1.0 / delta)) / lam                  # Markov tail bound
        best_eps = min(best_eps, eps)
    return best_eps


def calibrate_noise_to_epsilon(target_epsilon, steps, q, delta, tol=1e-3):
    """Smallest sigma whose composed budget spends at most target_epsilon (binary search)."""
    lo, hi = 0.01, 100.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if compute_epsilon(steps, mid, q, delta) > target_epsilon:
            lo = mid                                                      # more epsilon -> more noise
        else:
            hi = mid
    return (lo + hi) / 2
```
