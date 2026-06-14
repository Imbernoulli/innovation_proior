# WSD (Warmup-Stable-Decay), distilled

WSD is a learning-rate schedule for large-model pretraining that replaces the single cosine
curve with three explicitly separated phases — a linear **warmup** to the peak rate `η`, an
open-ended **stable** phase holding the rate at `η`, and a short final **decay** phase that
brings `η` down so the iterate settles. Its defining property is that the rate used during
the stable phase does not depend on when the run will end, so the schedule needs no
predefined step budget: a converged model at any token count is obtained by running a cheap
decay from the stable checkpoint at that point.

## Problem it solves

The standard schedule (linear warmup → single-cycle cosine to a ~10% floor) is only optimal
when its cycle length `T` equals the run length `S`: `T < S` wastes summed learning rate
(Kaplan et al. 2020), and `T > S` hurts the final model and overestimates intermediate loss
(Hoffmann et al. 2022). Because the rate at every middle step depends on the fixed horizon
`T`, runs cannot be cleanly extended, intermediate checkpoints are never converged, and
populating an `(N, D)` scaling-law grid costs a separate full run per cell (quadratic). WSD
decouples "how long to train at high rate" from "how to decay," keeping the best-tuned
cosine as the loss target while removing the horizon dependence.

## Key idea

A `T = S` cosine wins for two separable reasons: a long high-rate phase (good exploration,
maximal summed learning rate) and a thorough final decay (a fast, late loss drop). The
cosine ties both to the single knob `T`. WSD separates them:

```
rate(s) = (s/W)·η          for s < W      (warmup: linear ramp to peak)
        = η                for W < s < T  (stable: hold at full peak, horizon-free)
        = f(s − T)·η       for T < s < S  (decay: f decreasing, f(0)≈1, → small)
```

- **Stable at the full peak `η`** (not a descending value) accumulates the maximum summed
  learning rate per step — strictly more than cosine, which descends throughout — and, being
  flat, lets stable-phase training be *reused* when the run is extended.
- **The decay is what drops the loss**, and it must be a real decay: at the high stable rate
  the optimizer overshoots curved regions and thrashes (consecutive gradients near-
  orthogonal, large weight updates but little loss progress); lowering the rate lets
  consecutive steps align and descend coherently into a basin (gradient cosine turns
  positive, curvature rises, loss falls fast). A constant rate never gets this drop.
- **Decay length ≈ 10% of the budget**: enough small-rate steps for the iterate to settle
  while keeping branch decays cheap. The required fraction can shrink for very long runs.
- **Decay shape `f` is a secondary knob** — a monotone descent from ~1 to small. Options:
  discrete drops (DeepSeek-style multi-step, ×0.316 at 80% and again at 90%, so
  0.316² ≈ 0.1); **linear** `f(u) = 1 − u`; **1 − sqrt** `f(u) = 1 − √u`;
  **exponential** `f(s − T) = 0.5^{(s−T)/D'}`; and **cosine cooldown**
  `f(u) = 0.5(1 + cos(πu))`. The HuggingFace scheduler exposes linear / `1-sqrt` /
  cosine and defaults to the cosine cooldown (`num_cycles=0.5`). The phase structure, not
  the exact cooldown shape, is the contribution.

## Scaling-law payoff

One horizon-free stable run + cheap decay branches at a series of `D` traces the whole
optimal loss envelope, making scaling-law measurement linear-cost (`O(mC)`) instead of
quadratic. Fitting `L(N, D) = C_N N^{−α} + C_D D^{−β} + L_0` with compute `C = 6ND` and
minimizing under fixed `C` (substitute `D = C/(6N)`):

```
N_opt = [(α C_N)/(β C_D)]^{1/(α+β)} · (C/6)^{β/(α+β)}
D_opt = [(β C_D)/(α C_N)]^{1/(α+β)} · (C/6)^{α/(α+β)}
N_opt / D_opt = K² (C/6)^{eta_scale},
K = ((α C_N)/(β C_D))^{1/(α+β)},   eta_scale = (β − α)/(α + β)
```

`eta_scale` is the scaling-law exponent, not the peak learning rate `η`. `α = β` ⇒
constant ratio (Chinchilla); `α < β` ⇒ emphasize parameter scaling as compute grows;
`α > β` ⇒ emphasize data scaling as compute grows. The schedule's payoff is that the
envelope points needed to estimate these exponents are cheaper to obtain.

## Working code

Filling the `get_lr` slot of the fixed pretraining loop (the schedule is the only design
choice; the loop sets this rate on every AdamW group each iteration). This adapts the
canonical HuggingFace `get_wsd_schedule` lambda to the requested interface: `lr_decay_iters`
is the total horizon, the last tenth is the decay window, warmup is linear, the stable phase
returns the peak, the default decay is a half-cosine cooldown, and `min_lr` is applied as
`min_lr_ratio = min_lr / learning_rate`.

```python
import math


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Warmup-stable-decay schedule adapted to a fixed training loop."""
    if learning_rate == 0:
        return 0.0

    num_training_steps = max(1, lr_decay_iters)
    num_warmup_steps = max(0, warmup_iters)
    num_decay_steps = max(1, int(0.1 * num_training_steps))
    if num_warmup_steps + num_decay_steps > num_training_steps:
        num_decay_steps = max(1, num_training_steps - num_warmup_steps)
    num_stable_steps = max(0, num_training_steps - num_warmup_steps - num_decay_steps)
    min_lr_ratio = max(0.0, min(1.0, min_lr / learning_rate))

    if it < num_warmup_steps:
        progress = float(it) / float(max(1, num_warmup_steps))
        factor = progress * (1.0 - min_lr_ratio) + min_lr_ratio
        return learning_rate * max(0.0, factor)

    if it < num_warmup_steps + num_stable_steps:
        return learning_rate

    if it < num_warmup_steps + num_stable_steps + num_decay_steps:
        progress = float(it - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        num_cycles = 0.5
        factor = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
        return learning_rate * max(0.0, factor)

    return min_lr
```
