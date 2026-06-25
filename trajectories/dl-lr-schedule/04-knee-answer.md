**Problem (from rung 3).** Warmup + cosine is the strongest rung at {92.71, 72.43, 94.83}: the five-epoch
ramp fixed the deep net's init (CIFAR-100 71.07→72.43) and the cosine fixed the finish. But it — like every
prior rung — *cools the rate almost immediately*, holding the peak `base_lr` for a single instant. No rung
has touched the *middle*.

**Key idea (Knee / explore-exploit).** The learning rate is the *temperature* of the search: the SGD noise
scale `g ≈ eps·N/(B·(1−m))` rises with the rate, and high noise ejects SGD from narrow basins so it settles
only in *wide* (well-generalizing) ones. Under the wide-minima-density hypothesis (wide minima are rare),
cooling early lands in an abundant narrow basin. So insert a deliberate, sustained *explore* phase — hold
the rate flat at `base_lr` for ~50% of the budget — *before* cooling, then a linear *exploit* decay to 0.

**Why it works (and why it fits where one-cycle didn't).** One-cycle wanted more regularization from a
*higher* peak, which the harness blocks (no momentum cycling, no weight-decay rebalancing). Knee gets more
wide-minimum search from a *longer* time at the *same* peak `base_lr` — entirely within the per-epoch `lr`
lever, no momentum/weight-decay change needed. Explore is constant (any decay there is premature cooling);
exploit is plain linear (the plateau carries the generalization load, the decay just settles to exactly 0
at the boundary). The warmup the ladder proved the deep net needs composes in front as init hygiene.

**Hyperparameters.** `warmup = 5`; `explore = int(0.5·total_epochs)` (= 100 for the 200-epoch budget);
`decay = total_epochs − warmup − explore` (= 95); peak `= base_lr`; exploit slope `= −base_lr/decay`,
clamped at 0. 0-indexed `epoch` mapped to the canonical 1-indexed `global_step = epoch + 1` so the phase
boundaries line up and the final epoch lands exactly at 0. No `arch`/`dataset` conditioning.

The three phases are: warmup `peak·step/warmup`, explore
constant `peak`, exploit `max(0, peak − peak/decay·(step − warmup − explore))`.

```python
def get_lr(epoch, total_epochs, base_lr, config):
    """Knee (explore-exploit) schedule.

    Three phases tiling the budget (canonical 1-indexed step = epoch + 1):
      warmup  (5 epochs):     linear ramp base_lr/5 -> base_lr
      explore (50% of budget): hold base_lr (the high-temperature wide-minimum search)
      exploit (remainder):     linear decay base_lr -> 0 (cool to settle the wide basin)
    """
    warmup_steps = 5
    explore_steps = int(0.5 * total_epochs)
    decay_steps = total_epochs - (warmup_steps + explore_steps)
    global_step = epoch + 1

    if global_step <= warmup_steps:
        return base_lr * global_step / warmup_steps
    if global_step <= warmup_steps + explore_steps:
        return base_lr
    if global_step >= total_epochs:
        return 0.0
    slope = -base_lr / decay_steps
    return max(0.0, base_lr + slope * (global_step - (warmup_steps + explore_steps)))
```
