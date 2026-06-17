# EF-TopK (Top-K sparsification with error feedback), distilled

EF-TopK is a gradient compressor for communication-efficient distributed SGD. Each step it
transmits only the `k = max(1, int(d · ratio))` largest-magnitude coordinates of a gradient
tensor (their values and indices) and zeros the rest — but the suppressed `(d − k)`
coordinates are not discarded. They are accumulated in a local per-tensor residual ("error
feedback" / "memory") and added back into the gradient before the next compression, so
persistent suppressed signal is delayed rather than erased. This converts a *biased*
compressor — which on its own can stall or fail to converge — into one that matches plain
SGD's convergence rate.

## Problem it solves

In data-parallel distributed training, the per-step all-reduce of a full dense gradient
(millions of floats) is the wall-clock bottleneck. EF-TopK cuts the communicated volume by 1-3
orders of magnitude (e.g. keep 1% of coordinates, `100×`) without sacrificing convergence
quality.

## Key idea

- **Top-K sparsification.** Gradients are positively skewed (most coordinates near zero, a few
  large), so the top-k by magnitude holds almost all the energy `‖g‖²`. Send `k` values + `k`
  indices; apply per parameter tensor (layer-wise), since blocks live on different scales.
- **Top-K is biased**, `E[top_k(g)] ≠ ∇f`, so it is not a stochastic gradient and SGD's
  guarantees do not apply. Naively, a persistently-small coordinate is never sent and that
  direction is starved — convergence degrades or fails (the same failure as `sign(g)`, which
  can reverse the mean direction or freeze a fixed direction forever).
- **Error feedback** fixes this. Keep a residual `e_t` of all suppressed mass. Each step
  compress the error-corrected vector `p_t = γ g_t + e_t`, step with `C(p_t)`, and stash the
  remainder `e_{t+1} = p_t − C(p_t)`. Persistent suppressed signal accumulates in `e_t` until
  it can be selected — delayed rather than erased.

## Algorithm (EC-SGD with a `δ`-approximate compressor)

```
e_0 = 0
for t = 0, 1, ..., T-1:
    g_t = stochasticGradient(x_t)          # E[g_t] = grad f(x_t), E||g_t||^2 <= sigma^2
    p_t = gamma * g_t + e_t                 # error correction
    Delta_t = C(p_t)                        # compression, e.g. top-k
    x_{t+1} = x_t - Delta_t                 # update iterate
    e_{t+1} = p_t - Delta_t                 # update residual (the suppressed part)
```

A `δ`-approximate compressor satisfies `‖C(x) − x‖² ≤ (1 − δ)‖x‖²`. Top-k keeps the `k`
largest-magnitude coordinates and is a `δ = k/d` compressor: it drops no more energy than random-k,
whose expected dropped energy is `(1 − k/d)‖x‖²` (each coordinate kept with prob. `k/d`).

## Why it works (the proofs)

**Virtual iterate.** Let `x̃_t = x_t − e_t`. Then
`x̃_{t+1} = x_{t+1} − e_{t+1} = (x_t − Δ_t) − (p_t − Δ_t) = x_t − p_t = x̃_t − γ g_t`, i.e.
the virtual iterate runs *exact* SGD. The residual `e_t` is the only gap between `x̃_t` and the
real `x_t`.

**Bounded residual.** `‖e_{t+1}‖² = ‖C(p_t) − p_t‖² ≤ (1 − δ)‖e_t + γ g_t‖²`. By Young's
inequality with `η = δ/(2(1−δ))` (giving contraction factor `1 − δ/2` and injection
`1 + 1/η ≤ 2/δ`) and unrolling the geometric series from `e_0 = 0`:

```
E||e_t||^2 <= 4 (1 - delta) gamma^2 sigma^2 / delta^2     (= 0 when delta = 1)
```

**Non-convex rate (smooth `f`).** Apply `L`-smoothness on the virtual SGD iterate, trade
`∇f(x̃_t)` for `∇f(x_t)` paying `‖∇f(x_t) − ∇f(x̃_t)‖² ≤ L²‖e_t‖²`, substitute the residual
bound, and keep the Young-inequality parameter `ρ` through the telescope. For any `0 < ρ < 2`:

```
avg_t E||grad f(x_t)||^2 <= f_0 / (gamma (1-rho/2) (T+1))
                            + L gamma sigma^2 / (2-rho)
                            + 4 gamma^2 L^2 sigma^2 (1-delta) / (rho (2-rho) delta^2),
```

where `f_0 = f(x_0) - f*`. Since the average upper-bounds `min_t`, a simple fixed choice
`ρ = 1` gives:

```
min_t E||grad f(x_t)||^2 <= 2 f_0 / (gamma (T+1)) + L gamma sigma^2
                            + 4 gamma^2 L^2 sigma^2 (1-delta) / delta^2.
```

With `γ = 1/√(T+1)`, this is `O(1/√T)` with the `δ` penalty only in the `O(1/T)` term.
Letting `ρ` decrease slowly with `T` makes the first two constants approach the plain-SGD proof
constants while keeping the compression penalty higher order. This is the precise asymptotic
"compression for free" claim: compression quality does not enter the leading `O(1/√T)` term.

**Non-smooth convex rate.** Without smoothness, `∇f(x̃_t) ≈ ∇f(x_t)` fails, so `δ` enters the
leading constant. Using Cauchy-Schwarz after taking expectation, and `‖∂f‖ ≤ σ`, the average
iterate `x̄_T` satisfies

```
E[f(x_bar_T)] - f* <= ||x_0 - x*||^2 / (2 gamma (T+1)) + gamma sigma^2 (1/2 + 2 sqrt(1-delta)/delta),
```

optimizing the displayed bound over `γ` gives
`σ‖x_0 − x^⋆‖√(1 + 4√(1−δ)/δ)/√(T+1)` — still the right `1/√T` order, which the
naive biased compressor could not guarantee. For `k = 1` (top-1, `δ = 1/d`) this is a
convergent greedy-coordinate method on non-smooth functions.

**Generalization (over-parameterized least squares).** `x_t − e_t = x_0 − Σ_{i=0}^{t-1} γ g_i`
lies exactly in the gradient span when `x_0 = 0`, so
`‖x_t − Π_span(x_t)‖ ≤ ‖e_t‖` — the iterate stays within the bounded residual of where unbiased
SGD would be, recovering the min-norm / max-margin solution that naive biased compression
breaks.

## Defaults and why

- `compress_ratio` is the knob: `0.01` = keep 1% = `100×`. `k = max(1, int(d · ratio))`; the
  `max(1, ·)` guarantees even a small tensor always sends ≥ 1 coordinate (no permanent
  silence). `δ = k/d` controls the higher-order term of the rate.
- Compression is **per tensor** (layer-wise): blocks are on different scales, so a global
  top-k would let large-scale blocks monopolize the budget and starve the rest.
- The residual is **local, per-name, not communicated**; only `[values, indices]` go on the
  wire (the `2k`-number payload that yields the saving). The context `(numel, shape)` is local.

## Working code

```python
import torch


class Compressor:
    """Top-K sparsification with error feedback (EF-TopK).

    Keeps the k = max(1, int(d * compress_ratio)) largest-magnitude coordinates
    of each gradient tensor. The (d-k) suppressed coordinates are accumulated in a
    per-tensor residual and added back before the next compression, so persistent
    suppressed signal is delayed rather than erased -- which makes the biased
    top-k compressor match SGD's convergence rate."""

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        self.residuals = {}                       # e[name]: local memory, NOT communicated

    def compress(self, tensor, name):
        # error correction: p_t = g_t + e_t
        if name in self.residuals:
            tensor = tensor + self.residuals[name]

        shape = tensor.size()
        tensor_flat = tensor.flatten()
        numel = tensor_flat.numel()
        k = max(1, int(numel * self.compress_ratio))

        # top-k by magnitude
        _, indices = torch.topk(tensor_flat.abs(), k, sorted=False)
        values = torch.gather(tensor_flat, 0, indices)

        # residual = what was NOT sent: e_{t+1} = p_t - C(p_t)
        decompressed_flat = self.decompress([values, indices], (numel, shape)).flatten()
        self.residuals[name] = (tensor_flat - decompressed_flat).view(shape)

        return [values, indices], (numel, shape)

    def decompress(self, compressed_tensors, ctx):
        values, indices = compressed_tensors
        numel, shape = ctx
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)
```

This mirrors the standard GRACE call sequence in one scaffold-compatible class: compensate with
the residual, run `TopKCompressor` logic (flatten, `topk(|·|, k)`, gather values, scatter on
decompress), then run `ResidualMemory.update` logic by storing `tensor − decompressed`. The
paper equations put the learning rate inside `p_t = γg_t + e_t`; this scaffold keeps residuals
in gradient units because the optimizer applies the learning rate after decompression, which is
the usual GRACE-style integration for homogeneous compressors such as top-k.
