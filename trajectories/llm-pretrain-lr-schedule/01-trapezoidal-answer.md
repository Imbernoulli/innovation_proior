**Problem.** Pretrain GPT-2 Medium on FineWeb with everything frozen except the learning-rate
schedule. The scaffold default is a single-cycle cosine, which is only near-optimal when its cycle
length equals the run length: the rate at every middle step is welded to a horizon fixed in advance,
so the curve cannot be cleanly extended and its intermediate checkpoints overestimate the loss of
length-matched shorter runs. The first rung replaces the cosine with a phase-separated schedule.

**Key idea (trapezoidal / warmup–stable–decay).** Cosine fuses two separable jobs — high-rate
exploration and low-rate settling — into one fixed-length curve, which is the source of the length
commitment. Split them: linear **warmup** to the peak `η`, a long **constant plateau** at `η`
(length-agnostic exploration), and a short final **linear cooldown** that anneals the rate down to
the floor (the commit-to-a-basin phase). The plateau holds the full peak, so it accumulates more
summed learning rate than a cosine that descends throughout, and the cooldown supplies the genuine
final anneal the cosine relied on.

**Why each choice.** The constant plateau is needed for length-agnostic exploration; a flat plateau
is *not* a stalled schedule — for convex Lipschitz objectives the last iterate of a constant-rate
SGD carries an extra `ln T` factor (`η·G²·H_{T−1}`) that a cooldown removes by shrinking the late
step sizes. Linear cooldown to the floor is the worst-case-optimal last-iterate shape (Defazio et
al. 2023): `η_t = (D/(G√T))(1 − t/T)` gives `E[f(x_T) − f*] ≤ D·G/√T` with no `ln T`, "emulating
iterate averaging." Linear is the safe default here; a faster front-loaded `1 − √` tail is left for
a later rung.

**Step-1 edit (the literal scaffold fill).** Only the body of `get_lr` is replaced. There is no
`wsd_schedule` factory or `decay_type` switch in the harness — the fill is three branches written
directly against the loop's five arguments. The cooldown is taken as the **last 40%** of the horizon
(`cooldown_start = int(lr_decay_iters * 0.6)`), a deliberately generous tail for a first cut; warmup
is the loop's own `(it+1)/(warmup_iters+1)` ramp; the floor is `min_lr` (not zero).

**Hyperparameters.** Peak `learning_rate = 6e-4`, floor `min_lr = 6e-5`, `warmup_iters ≈ 481`
(4% of 12030), `lr_decay_iters = 12030`, cooldown window = last 40% of the horizon, linear cooldown
shape, single seed 42. No `CONFIG_OVERRIDES` (the schedule shape is the whole design).

**What to watch.** The trapezoid should land in the neighborhood of a tuned cosine on FineWeb val
loss — the bar for a credible floor — but is expected to be the weakest of the phase-separated
schedules, because the linear tail over a long 40% window is the least efficient way to spend the
cooldown. If so, the next rung keeps the structure and spends the cooldown more aggressively
(shorter window, faster early drop).

```python
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Trapezoidal learning rate schedule: warmup, constant plateau, cooldown."""
    if it > lr_decay_iters:
        return min_lr
    # Warmup phase
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Cooldown phase: last 40% of training
    cooldown_start = int(lr_decay_iters * 0.6)
    if it >= cooldown_start:
        t = (it - cooldown_start) / (lr_decay_iters - cooldown_start)
        return min_lr + (learning_rate - min_lr) * (1.0 - t)
    # Constant plateau
    return learning_rate
```
