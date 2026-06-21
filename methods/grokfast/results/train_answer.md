The problem is delayed generalization, or grokking, on small algorithmic datasets such as modular addition or multiplication. A small network reaches perfect training accuracy in a few hundred steps, but validation accuracy stays near chance for tens or hundreds of thousands of steps before jumping up. The dataset, loss, architecture, and train/test split are fixed, so the only place to intervene cheaply is the gradient stream before the optimizer consumes it. Existing ideas fall short because they treat the delay indirectly. Weight decay is the standard knob: it pushes the trajectory toward lower-norm solutions and shortens the plateau, but it is a global regularizer, its size is brittle, and it still leaves a long wait. Momentum and Adam's internal EMAs smooth gradients to speed training loss convergence, but they consume that smooth signal as the step itself rather than selectively amplifying the slow component of the motion. Other regime-aware approaches describe the slow internal progress but do not turn it into a concrete, optimizer-agnostic gradient operation. The core observation missing from these baselines is that the parameter motion contains two components on two very different timescales: a fast component that drives memorization and a slow component that drives generalization. If those components can be separated and the slow one amplified, the late jump should happen sooner.

The method is Grokfast, and the practical version is Grokfast-EMA. It reads each parameter's gradient sequence as a discrete-time signal and boosts its low-frequency content. The update rule is simple: maintain an exponential moving average of the gradient and add a scaled copy back to the raw gradient before the optimizer step. Concretely, for each parameter with gradient g, keep a buffer μ and compute μ ← α μ + (1 − α) g, then hand the optimizer the modified gradient ĝ = g + λ μ. This is equivalent in the frequency domain to Ĝ(ω) = (1 + H(ω)) G(ω), where H(ω) = λ(1 − α)/(1 − α e^{-iω}). The effective gain at low frequency is 1 + λ, while at high frequency it stays near 1, so the slow component is emphasized while the fast component is preserved rather than discarded. The moving-average variant stores a queue of the last w gradients and achieves the same low-pass boost, but it costs w times the model memory. The EMA variant needs only one buffer per parameter, so it is the form used in practice. For linear SGD-family optimizers, filtering the gradient is provably equivalent to filtering the parameter update by the same filter, because the optimizer's transfer function cancels when the same optimizer is run on both the original and filtered gradients. That is why the whole thing can live in a hook between loss.backward() and optimizer.step() instead of requiring a custom optimizer. Adam and AdamW are supported empirically at the same hook point.

```python
from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {
            n: p.grad.detach().clone()
            for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None
        }
    for n, p in m.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        grads[n].mul_(alpha).add_(p.grad.detach(), alpha=1.0 - alpha)
        p.grad.add_(grads[n], alpha=lamb)
    return grads


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal["mean", "sum"] = "mean",
) -> Dict[str, deque]:
    if grads is None:
        grads = {
            n: deque(maxlen=window_size)
            for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None
        }
    for n, p in m.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        grads[n].append(p.grad.data.detach())
        if filter_type == "mean":
            avg = sum(grads[n]) / len(grads[n])
        elif filter_type == "sum":
            avg = sum(grads[n])
        else:
            raise ValueError(f"filter_type must be 'mean' or 'sum', got {filter_type}")
        p.grad.data = p.grad.data + avg * lamb
    return grads


# Integration into a standard training loop:
grads = None
for batch in dataloader:
    model.zero_grad()
    logits = model(batch)
    loss = criterion(logits, targets)
    loss.backward()
    grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
    optimizer.step()
```
