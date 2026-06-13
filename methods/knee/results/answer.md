# The Knee (explore-exploit) learning-rate schedule, distilled

The Knee schedule is a learning-rate schedule for SGD-with-momentum training of deep nets, derived from
the geometry of the minima SGD finds. It has three phases: a short linear **warmup** to a peak rate, a
long **explore** phase that holds the peak rate constant for roughly half the budget, and an **exploit**
phase that linearly decays the rate to 0 over the remainder. The explore phase keeps the SGD "temperature"
high so the search lands in a *wide* (flat, well-generalizing) minimum, which the hypothesis says is rarer
than narrow ones and therefore needs a long search; the exploit phase cools the search to settle into the
wide basin found. At the same budget it improves accuracy over hand-tuned decays, and it reaches the
original accuracy in fewer epochs. Only the per-step learning rate changes; the optimizer, loss, data
pipeline, momentum, and weight decay are untouched.

## Problem it solves

Generalization in deep nets is governed by *which* minimum SGD lands in — wide, flat minima generalize,
sharp, narrow ones do not. The standard schedules (cosine, linear, step decay) are tuned for optimization
*speed* and begin cooling the rate from the first step, so they settle quickly and are biased toward the
abundant narrow minima. The goal: a schedule derived from the wide-minima geometry that spends a
deliberate high-rate phase searching for a wide minimum before cooling into it.

## Key ideas

- **Learning rate = temperature of the search.** The SGD noise scale `g ≈ eps·N/(B·(1−m))` rises with the
  rate `eps`. High noise ejects SGD from narrow basins and lets it settle only where the basin is wide. So
  a high rate is a wide-minimum-seeking device, and "decay the rate" is "cool the search."
- **Wide-minima density hypothesis.** Wide minima are *lower-density* than narrow ones — rare in a
  landscape full of narrow basins. (Consistent with large-batch / low-noise training settling into sharper
  minima and generalizing worse.) Finding a rare wide basin therefore requires staying *hot for a long
  time*; cooling early lands in an abundant narrow basin.
- **Explore then exploit.** Hold the rate *flat at the peak* for a long explore phase (the
  high-temperature wide-minimum search), then *linearly decay to 0* over the exploit phase (cool into the
  wide basin found). Constant during explore — any decay there is premature cooling.
- **The explore fraction is the one knob, and it should be large.** Too little explore cools before
  finding a wide basin; too much leaves too little budget to settle. Sweeping explore fractions and
  measuring both accuracy and the *width* of the minimum reached puts the optimum at ~**50% of the
  budget**.
- **Linear exploit, not cosine.** The explore phase carries the generalization load; the exploit phase
  just settles, so a plain linear decay to 0 (the strongest simple monotone budgeted decay) suffices and
  hits exactly 0 at the budget boundary.
- **Warmup composes in front** as init hygiene (a short linear ramp to the peak), complementary to the
  explore-exploit structure — important for deep nets and Transformers.

## Final schedule

Three segments tiling the budget, with `peak_lr` the reference rate, `warmup_steps` the ramp length,
`explore_steps ≈ 0.5 · total_steps`, and `decay_steps = total_steps − (warmup_steps + explore_steps)`:

```
warmup  (step ≤ warmup):                        lr = peak_lr · step / warmup_steps
explore (warmup < step ≤ warmup + explore):     lr = peak_lr
exploit (step > warmup + explore):              lr = max(0, peak_lr − peak_lr/decay_steps · (step − warmup − explore))
```

Seams are continuous in value (warmup ends at `peak_lr`, explore holds `peak_lr`, exploit opens at
`peak_lr` and reaches 0 at the budget boundary); only the slope changes at each seam.

## Working code

Faithful to the reference `KneeLRScheduler` (Iyer et al.). The schedule sets `lr` on each optimizer
parameter group; the SGD-with-momentum update is unchanged.

```python
import torch
from torch.optim.optimizer import Optimizer


class KneeLRScheduler:
    """Explore-exploit (Knee) learning-rate schedule.

    warmup : linear ramp 0 -> peak_lr over warmup_steps
    explore: hold peak_lr for explore_steps (defaults to ~50% of total budget)
    exploit: linear decay peak_lr -> 0 over the remaining decay_steps
    """

    def __init__(self, optimizer, peak_lr, warmup_steps=0, explore_steps=None, total_steps=0):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")

        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.explore_steps = int(0.5 * total_steps) if explore_steps is None else explore_steps
        self.total_steps = total_steps
        self.decay_steps = self.total_steps - (self.explore_steps + self.warmup_steps)
        self.current_step = 1

        assert self.decay_steps > 0

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.current_step)

    def get_lr(self, global_step):
        if self.warmup_steps > 0 and global_step <= self.warmup_steps:
            return self.peak_lr * global_step / self.warmup_steps
        elif global_step <= (self.explore_steps + self.warmup_steps):
            return self.peak_lr
        else:
            slope = -1 * self.peak_lr / self.decay_steps
            return max(0.0, self.peak_lr + slope * (global_step - (self.explore_steps + self.warmup_steps)))

    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.current_step)
```

## Relation to prior schedules

- **Cosine annealing / linear decay**: cool from the first step (peak rate held for one instant); Knee
  holds the peak for ~half the budget before cooling.
- **Step decay**: a high first plateau, but its length is a speed-tuned milestone and the drop is a shock;
  Knee's explore length is set by the wide-minimum search and its exploit decay is smooth-to-zero.
- **One-cycle**: a single rise-then-fall with a deep tail; Knee replaces the rise/peak-instant with a
  sustained flat explore plateau (constant peak, not a triangular/cosine peak).
- **Warmup + decay**: warmup composes in front of Knee as init hygiene; Knee adds the deliberate explore
  plateau the plain warmup+decay lacks.
