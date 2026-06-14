# WSD with (1-sqrt) cooldown, distilled

The Warmup-Stable-Decay (WSD) learning-rate schedule splits LLM pretraining into three explicit
stages — a linear **warmup** to the peak rate, a long **stable** stage that holds the peak rate
constant, and a short final **decay** (cooldown) that anneals the rate down to a floor. Crucially
the stable stage has no predetermined length: it is a reusable trunk that can run indefinitely,
and a cooldown can be branched off any checkpoint on demand. The **sqrt** variant uses the
`1 - sqrt(progress)` cooldown shape, which front-loads the rate drop and outperforms the linear
cooldown in the reported 10-20% cooldown experiments.

## Problem it solves

The standard cosine schedule is near-optimal only when its cycle length `T` equals the total run
length `S`, so the token budget must be fixed before training: the run cannot be continued past
the cycle (the rate is at its floor), and studying a model at several token counts requires a
separate from-scratch run per length — `O(m^2)` runs over a model-size × data-size grid. WSD
removes the length commitment, allows continuation and checkpoint reuse, and makes scaling-law
measurement linear (`O(m)`), while matching cosine's final loss — all purely at the schedule
layer, with the model, data, optimizer, and total update budget unchanged.

## Key idea

Cosine fuses two separable jobs — high-rate *exploration* and low-rate *settling* — into one
fixed-length curve, which is what welds the cycle length to the run length. Separate them:

- **Warmup** — linear ramp from ~0 to the peak `eta`. Performance is insensitive to its length as
  long as it is long enough; a generous warmup is a safe choice.
- **Stable** — hold `eta` constant for the bulk of training. This is the reusable trunk; it can
  run for an undetermined number of steps.
- **Decay (cooldown)** — over the final short fraction of training (~10–20% of steps), anneal the
  rate from `eta` down to `min_lr`.

Why a constant trunk plus a short decay recovers cosine's loss — the **river-valley** picture:
the pretraining loss is a long valley with a flat "river" direction along its floor and steep
"hill" directions across it. A high constant rate races the iterate down the river (real progress)
but makes it overshoot and oscillate across the hill directions, so the measured loss stays
elevated — large steps, little apparent progress. Lowering the rate quenches the cross-valley
oscillation and the iterate drops the short distance into the channel — small steps, a sharp loss
drop. Because you only descend the valley's width (not the river's length), a short decay (~10%)
suffices, and pre- vs. post-decay checkpoints lie in one connected basin (linear weight
interpolation between them shows the same smooth loss drop).

Why `1 - sqrt(progress)` for the cooldown shape: the decay should quench high-rate oscillation
early, while leaving enough remaining cooldown at useful low rates. With decay-progress
`p in [0,1]`, the multiplier `1 - sqrt(p)` has derivative `-1/(2 sqrt(p))`: singular at `p = 0`
in the continuous idealization and `-1/2` at `p = 1`. Since `sqrt(p) > p` on `(0,1)`, it lies
below the linear `1 - p`, reaching any chosen low multiplier earlier. Empirically this shape
consistently beats linear in the tested 10-20% cooldown settings, with the advantage growing for
longer training. In the `1 - p^a` family, `a = 0.5` is empirical rather than algebraically forced:
very small exponents such as `0.1` and `0.2` underperform because the rate is too low for too many
steps, while the remaining tested exponents below `0.5` differ only marginally and `0.5` still
comes out on top. The final LR is an independent knob: `min_lr = 0` reproduces the canonical
cooldown-to-zero, while a small positive floor can avoid exact-zero saturation on downstream
metrics.

## Final schedule

Peak `eta`, floor `min_lr`, warmup end `W`, decay-start step `T = S - N_decay`, run end `S`,
decay-progress `p = (s - T) / (S - T)`:

```
WSD(s) =  (s / W) * eta                              if s < W        # linear warmup
          eta                                        if W <= s < T   # stable plateau
          min_lr + (eta - min_lr) * (1 - sqrt(p))    if T <= s < S   # (1-sqrt) cooldown
```

with `N_decay` commonly in the `0.1-0.2 * S` range, and sometimes smaller as a fraction for very
long runs if the absolute number of cooldown steps is large enough. At `s = T`, `p = 0` and
`WSD = eta` (continuous with the plateau); at `s >= S`, `p` is clamped to `1` and `WSD = min_lr`.

## Working code

Filling the `get_lr` slot of the nanoGPT-style training loop (signature fixed by the loop, which
calls it every iteration to set AdamW's rate):

```python
import math


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """WSD learning-rate schedule with (1-sqrt) cooldown.

    warmup -> constant stable plateau -> short final cooldown shaped as
    (1 - sqrt(progress)): drops steeply at first to quench the cross-valley
    oscillation, then flattens so most cooldown steps settle into the basin.
    """
    # Warmup: linear ramp from ~0 up to the peak learning rate.
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    # Decay window: the final fraction of the horizon (~20% of steps).
    decay_start = int(lr_decay_iters * 0.8)

    # Cooldown: 1 - sqrt(progress), front-loaded drop.
    if it >= decay_start:
        decay_len = max(1, lr_decay_iters - decay_start)
        p = (it - decay_start) / decay_len
        p = min(max(p, 0.0), 1.0)                 # clamp to [0, 1]
        coeff = 1.0 - math.sqrt(p)                # 1 at p=0, 0 at p=1
        return min_lr + (learning_rate - min_lr) * coeff

    # Stable stage: hold the peak rate — the reusable trunk, cooldown on demand.
    return learning_rate
```

Equivalent multiplicative form (matching the canonical `wsd_schedule` with `decay_type="sqrt"`):
with `final_lr_factor = min_lr / eta`, `n_anneal = int(fract_decay * N)`, and
`n_hold = N - n_anneal`, the cooldown multiplier for `n_hold <= step < N` is
`final_lr_factor + (1 - final_lr_factor) * (1 - sqrt((step - n_hold) / n_anneal))`; for
`step >= N`, it returns `final_lr_factor`.

## Relation to prior schedules

- **Cosine / SGDR** — single fused curve `0.1 eta + 0.9 eta * 0.5(1 + cos(pi * r))`; near-optimal
  only at cycle length = run length. WSD = the same exploration/settling trade-off but with the
  two phases *separated*, so the high-rate phase is length-free and the decay is a branchable tail.
- **Constant LR (no decay)** — WSD's stable trunk with no cooldown; leaves the late-anneal
  settling (the sharp loss drop) unrealized.
- **Constant + linear cooldown (trapezoidal / "infinite-LR" schedule)** — WSD with `f(p) = 1 - p`.
  The sqrt cooldown replaces the linear tail with `1 - sqrt(p)`, front-loading the rate drop; in
  the reported ablations it gives lower loss than linear, with the advantage growing for longer
  runs.
