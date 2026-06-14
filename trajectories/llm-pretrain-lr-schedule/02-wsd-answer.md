**Problem.** The trapezoid (rung 1) confirmed that a phase-separated schedule matches a tuned cosine
(FineWeb val loss 2.2512), but it used a long **40%** linear cooldown — so for nearly half the run
the rate was descending, spending less summed learning rate up high than the structure allows. The
plateau only owned the first 60%. This rung keeps the structure and reclaims cooldown budget for the
plateau.

**Key idea (WSD — warmup, stable, short decay).** The cooldown's only job is to kill the
cross-valley oscillation and let the iterate fall into the basin — descending the *width* of the
valley, not the *length* of the river — so it can be short. Shorten the linear cooldown from the last
40% to the last **20%** of the run, returning ~2400 steps to the full-peak plateau (now ~80% of
training after warmup). More summed learning rate at the peak, while a 20% (~2400-step) cooldown is
still long enough to finish the final anneal.

**Why each choice.** The river-valley picture explains "big steps, no progress / small steps, huge
progress": a high constant rate races down the flat river direction but overshoots the sharp
cross-valley directions, so the measured loss stays elevated; lowering the rate quenches the
oscillation and the iterate drops the short distance into the channel (pre- and post-cooldown
checkpoints lie in one connected basin). Because only the valley width is descended, ~10–20% of the
run suffices; below ~2.5% the cooldown ends half-quenched. The decay shape is held **linear**
(worst-case-optimal last-iterate, removes the constant-rate `ln T` term) so this rung isolates the
effect of the *window length*; the front-loaded `1 − √` shape is left for the next rung.

**Step-2 edit (the literal scaffold fill).** Only the body of `get_lr` is replaced — the same three
branches as the trapezoid with one number moved: `decay_start = int(lr_decay_iters * 0.8)` instead of
`0.6`. No scheduler factory, no `decay_type`, no `min_lr_ratio` warmup reweighting (the harness
exposes none). The redundant `if it > lr_decay_iters` guard is dropped: the loop only calls `get_lr`
for `it` up to `lr_decay_iters`, where the decay branch already returns exactly `min_lr`.

**Hyperparameters.** Peak `learning_rate = 6e-4`, floor `min_lr = 6e-5`, `warmup_iters ≈ 481`,
`lr_decay_iters = 12030`, cooldown window = last 20% of the horizon, linear cooldown, seed 42. No
`CONFIG_OVERRIDES`.

**What to watch.** With the plateau owning 80% instead of 60% (more summed rate) and the 20% cooldown
still finishing the descent, expect FineWeb val loss **below 2.2512**, with WikiText-2 (<42.31),
LAMBADA (<65.96), ARC-Easy (>55.77), and HellaSwag (>34.09) moving the same way. If a shorter window
*hurts*, 20% was too short and the next rung lengthens; if it helps, the next rung keeps 20% and
front-loads the tail with `1 − √`.

```python
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """WSD (Warmup-Stable-Decay) learning rate schedule."""
    # Warmup phase
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Decay phase: last 20% of training
    decay_start = int(lr_decay_iters * 0.8)
    if it >= decay_start:
        decay_ratio = (it - decay_start) / (lr_decay_iters - decay_start)
        return min_lr + (learning_rate - min_lr) * (1.0 - decay_ratio)
    # Stable phase: constant LR
    return learning_rate
```
