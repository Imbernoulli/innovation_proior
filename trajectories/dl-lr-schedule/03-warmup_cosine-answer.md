**Problem (from rung 2).** Cosine split the result: CIFAR-10 rose to 93.03 and FashionMNIST to 94.70
(dropping the long warmup and the non-zero floor helped the easy nets), but ResNet-56 / CIFAR-100 *fell*
to 71.07, below one-cycle's 71.57 — the bare full-rate start re-exposed the deep net's high-curvature
initialization. Warmup was protecting the deep net's init; cosine threw it out entirely.

**Key idea (linear warmup + cosine).** Put a *short* warmup back. Split start from body: a brief linear
ramp from `base_lr/5` up to `base_lr` over 5 epochs to survive the sharp initial region while
`2/lambda_max(H)` relaxes, then the same half-cosine anneal from `base_lr` to 0 that already won the easy
nets — with its clock measured from the end of warmup so the seam is continuous at `base_lr`.

**Why it works.** The deep 56-layer net has the sharpest init (smallest admissible step), so `base_lr`
from epoch 0 overshoots there specifically; a five-epoch gradual ramp lets curvature relax before the full
rate engages. The ramp must be *gradual*, not a constant prefix that snaps (a snap re-creates the
overshoot). Five epochs (2.5% of the run) is the smallest ramp that protects the deep net while costing
the body — and the easy nets, which already tolerated `base_lr` — almost nothing. Costless on the easy
nets, decisive on the hard one.

**Hyperparameters.** `warmup = 5`; ramp `base_lr*(epoch+1)/5` for `epoch < 5`; body `progress =
(epoch - warmup)/(total_epochs - warmup)`, `base_lr*0.5*(1 + cos(pi*progress))`; `eta_min = 0`. No
`arch`/`dataset` conditioning.

```python
def get_lr(epoch, total_epochs, base_lr, config):
    """Linear warmup (5 epochs) then cosine decay to 0.

    Epochs 0-4: linearly ramp from base_lr/5 to base_lr.
    Epochs 5+: cosine anneal from base_lr to 0.
    """
    warmup = 5
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / (total_epochs - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
```
