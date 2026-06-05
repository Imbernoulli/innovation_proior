# Stochastic Weight Averaging (SWA)

## Problem

Conventional SGD ships a single point that minimizes train loss, but the train-loss and test-error surfaces are shifted relative to each other, so that point is off-center for test error — and sharp optima suffer most under the shift. We want a near-free, drop-in replacement for SGD that ends at a broader, more central solution that generalizes better.

## Key idea

SGD with a high constant or cyclical learning rate keeps exploring the periphery of the high-performing region (high-dimensional samples concentrate near an ellipsoid surface, or a sphere after whitening). **Averaging the weights** of the iterates collected during this exploration moves the solution to the flat, central interior. A Taylor argument shows that averaging weights gives essentially the same predictions as ensembling — but with a single model.

## Final method

Start from a pretrained model `ŵ` (full or e.g. `0.75B` of the normal budget). Continue training with a high **constant** learning rate `α(i) = α_1`, or a **cyclical** one over batch iterations, `α(i) = (1−t(i))α_1 + t(i)α_2`, `t(i) = (1/c)(mod(i−1,c)+1)`, jumping discontinuously from the minimum back to the maximum each cycle (exploration over per-proposal accuracy). Capture a model `w_i` once per cycle (at the LR minimum) or once per epoch (constant LR), and maintain their running average:

`w_SWA ← (w_SWA · n_models + w) / (n_models + 1)`

(equivalently `w_SWA ← w_SWA + (w − w_SWA)/(n_models+1)`), so only one extra weight copy is needed.

**Batch norm.** `w_SWA` is an average of weights and was never used in a forward pass during training, so its batch-norm running statistics are stale. After training, do one extra forward pass over the data in training mode with `w_SWA` to recompute each layer's running mean and variance.

**Why it works.**
- *Width:* `w_SWA` lies in the same basin as `w_SGD` but in a flatter, more central region; it has slightly higher train loss yet lower test error, and one must step much further from it to raise the error — robust to the train→test shift.
- *Ensembling (Taylor):* for nearby points `w_i` with `Δ_i = w_i − w_SWA` (so `Σ Δ_i = 0`), linearizing the prediction `f` at `w_SWA` gives `f̄ − f(w_SWA) = ⟨∇f, (1/n)Σ Δ_i⟩ + O(Δ²) = O(Δ²)`, while proposal diversity `f(w_i) − f(w_j) = ⟨∇f, Δ_i − Δ_j⟩ + O(Δ²)` is first order. So weight-averaging matches prediction-ensembling to second order — ensemble-quality generalization from one model.

**Cost.** One extra weight copy (≈10% of training memory, since activations dominate; none afterward) and one weighted-sum update per epoch/cycle. Practically the same cost as SGD.

## Code

```python
import torch

def _set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr

def tail_lr(step, cycle_len, high_lr, low_lr=None):
    if low_lr is None:
        return high_lr
    if cycle_len <= 0:
        raise ValueError("cycle_len must be positive")
    t = ((step - 1) % cycle_len + 1) / cycle_len
    return (1 - t) * high_lr + t * low_lr

def train_tail(model, loader, loss_fn, optimizer, tail_epochs,
               high_lr, low_lr=None, cycle_len=None, device=None):
    if low_lr is not None and cycle_len is None:
        raise ValueError("cycle_len is required for a cyclical tail")

    avg_model = torch.optim.swa_utils.AveragedModel(model, device=device)
    model.train()
    step = 0

    for _ in range(tail_epochs):
        for x, y in loader:
            if device is not None:
                x, y = x.to(device), y.to(device)
            step += 1
            lr = tail_lr(step, cycle_len or 1, high_lr, low_lr)
            _set_lr(optimizer, lr)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()

            # Cyclical tail: capture exactly at the minimum LR, every c steps.
            if low_lr is not None and step % cycle_len == 0:
                avg_model.update_parameters(model)

        # Constant tail: capture one explored point per epoch.
        if low_lr is None:
            avg_model.update_parameters(model)

    if int(avg_model.n_averaged) == 0:
        avg_model.update_parameters(model)

    # Batch-norm fix: recompute running stats for the averaged weights.
    torch.optim.swa_utils.update_bn(loader, avg_model, device=device)
    return avg_model
```
