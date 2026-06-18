# Grokfast: accelerating grokking by amplifying slow gradients

## Problem

On small algorithmic datasets (e.g. `a · b (mod p)`), a network overfits in ~10^3 steps but its
validation accuracy only catches up after ~10^5-10^6 steps — *grokking*, a one-to-three order of
magnitude delay between memorization and generalization. Grokfast shrinks that delay with a cheap,
optimizer-agnostic add-on that touches neither the data, the loss, nor the architecture.

## Key idea

Read each parameter's gradient sequence `g(t)` over training steps as a discrete-time signal. The
parameter motion mixes a **fast-varying component** that drives overfitting and a
**slow-varying component** that drives the delayed generalization. In the frequency domain
(`G(ω) = Σ_t g(t) e^{-iωt}`), "slow" is the low-frequency content. So **amplify the
low-frequency component of the gradients**: take a low-pass-filtered copy and add it back to the
raw gradient,

```
ĝ(t) = g(t) + h(t) * g(t)   ⟹   Ĝ(ω) = (1 + H(ω)) G(ω),
```

a high-boost of the slow part with spectral gain `1 + H(ω)` — large at low `ω`, ≈1 at high `ω`,
so the fast component is kept (not denoised away) while the slow one is emphasized.

## Filter designs

**Grokfast-MA (stepping stone).** `h` is a windowed moving average of width `w`:
`h(t) = λ/w` for `0 ≤ t < w` else `0`. Per step: `ĝ_t = g_t + λ · mean(Q)`, with `Q` a queue of
the last `w` gradients. Knobs: gain `λ`, window `w` (the cutoff). Cost: stores `w` gradients per
parameter — `w×` memory.

**Grokfast-EMA (the method).** Replace the FIR window with a one-pole IIR exponential moving
average — same low-pass boost at `1/w` of the memory. Impulse response `h(t) = λ(1−α)α^t` (`t≥0`),
transfer function

```
H(ω) = λ(1−α) / (1 − α e^{-iω}),    H_amp(ω) = 1 + H(ω).
```

Low-frequency gain `H_amp(0) = 1 + λ`; Nyquist gain `H_amp(π) = 1 + λ(1−α)/(1+α) ≈ 1` for `α→1`.
`α` is the cutoff (effective window `≈ 1/(1−α)`), `λ` is the gain. Per step, per parameter:

```
μ ← α μ + (1 − α) g,    ĝ = g + λ μ.
```

Defaults `α = 0.98`, `λ = 2.0`; recommended `λ ∈ [0.1, 5]`, `α ∈ [0.8, 0.99]`. One buffer per
parameter (model-sized state).

## Why acting on the gradient is legitimate (equivalence theorem)

For the theorem-covered optimizer class, write the optimizer as a linear time-invariant system with
scalar state `x`:

```
x(t) = A x(t−1) + B g(t),    u(t) = C x(t) + D g(t),    0 < A < 1.
```
(SGD-momentum: `A=μ, B=1−τ, C=−η, D=0`; Nesterov: `A=μ, B=1−τ, C=−ημ, D=−η`, with
`m(t)=μ m(t−1)+(1−τ)g(t)`.) In frequency, `X(ω)=B G(ω)/(1−A e^{-iω})` and the optimizer's
transfer function is `H_io(ω) = U/G = BC/(1−A e^{-iω}) + D`. Feeding the filtered gradient
`Ĝ=(1+H)G` through the **same** optimizer leaves `H_io` unchanged, so

```
Û(ω)/U(ω) = Ĝ(ω)/G(ω) = 1 + H(ω)   ⟹   ĥ = h.
```

For linear SGD-family optimizers, filtering the gradient by `h` is identical to filtering the
parameter update by the same `h`: the optimizer's response cancels. This is why Grokfast can be a
hook on `p.grad` (between `backward()` and `step()`) instead of a bespoke optimizer. Adam/AdamW
support is a practical hook-level application verified empirically in the paper, not a consequence of
this exact LTI proof. It differs from momentum: the smoothed gradient is added as a **residual to
the raw gradient before the optimizer**, not consumed as the update itself.

## Code (canonical implementation)

```python
from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False, # For ablation study.
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size)
                 for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach())
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb       # g_hat = g + lambda * slow(g)
    return grads


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach()
                 for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)   # mu <- a*mu+(1-a)*g
            p.grad.data = p.grad.data + grads[n] * lamb                        # g_hat = g + lamb*mu
    return grads


# Two-line integration into any training loop:
grads = None
for batch in dataloader:
    model.zero_grad()
    loss = criterion(model(batch))
    loss.backward()
    grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
    optimizer.step()
```
