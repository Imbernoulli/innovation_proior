**Problem (from rung 1).** Constrained one-cycle landed {92.31, 71.57, 93.93} and exposed two leaks: a
sixty-epoch sub-`base_lr` warmup that ate the productive high-rate middle (worst on deep ResNet-56 /
CIFAR-100), and a `base_lr/25` floor that left residual minibatch jitter at the finish. Both are
subtractable.

**Key idea (single-cycle cosine annealing).** Drop the warmup entirely and the non-zero floor: one smooth
half-cosine from the full `base_lr` at epoch 0 down to 0 at the budget boundary. The shape is chosen from
where the rate budget should go — linger high while the gradient is informative, fall through the middle,
linger low to settle the minimum — which is exactly the half-cosine's profile (zero slope at both ends,
steepest at the midpoint).

**Why it works.** Starting at `base_lr` reclaims the productive epochs one-cycle's warmup wasted (the long
warmup underperformed, so init instability on these Kaiming+BatchNorm nets is mild enough to skip);
annealing to 0 removes the noise-floor jitter the `base_lr/25` floor left. This is the no-restart special
case of cosine annealing with warm restarts (`T_mult=1`, one cycle over the whole budget, `eta_min=0`,
`eta_max=base_lr`) — it keeps the part of one-cycle's restart idea the harness can run (the smooth anneal)
and drops the part the harness neutered (the capped kick). Two hyperparameters, no milestones, no warmup
length.

**Hyperparameters.** `eta_max = base_lr`, `eta_min = 0`, one cycle of length `total_epochs`. No
`arch`/`dataset` conditioning; sampled once per epoch. The closed form of PyTorch's `CosineAnnealingLR`
with `eta_min=0`.

```python
def get_lr(epoch, total_epochs, base_lr, config):
    """Cosine annealing from base_lr to 0.

    LR = base_lr * 0.5 * (1 + cos(pi * epoch / total_epochs))
    """
    return base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
```
