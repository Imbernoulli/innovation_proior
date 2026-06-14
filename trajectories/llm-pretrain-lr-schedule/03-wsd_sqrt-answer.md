**Problem.** WSD (rung 2) settled the phase structure and the cooldown *window* (last 20% beat the
trapezoid's 40%, FineWeb val loss 2.2474 < 2.2512). One knob was held fixed to isolate the window:
the decay *shape* inside that window, which WSD left **linear**. This rung asks whether linear is the
best way to spend the 2400 cooldown steps.

**Key idea (WSD with front-loaded inverse-sqrt cooldown + extended warmup).** The cooldown's job is
to quench the cross-valley oscillation *early*, then dwell at low rates where settling removes loss.
Linear spends equal step-density at every rate — too long at high-ish rates, too few steps left to
dwell low. Replace it with a sharply front-loaded concave tail: a rescaled inverse-sqrt
`coeff(t) = (1/√(1+9t) − 1/√10) / (1 − 1/√10)`, which runs 1 → 0 across the window and drops faster
than `1 − √t` (≈0.349 at quarter-progress vs 0.50 vs linear's 0.75). Also extend the warmup to 6% of
the horizon (a floor over the loop's 4%) as cheap insurance against early instability under the
longer plateau and harder drop-off.

**Why each choice.** River-valley mechanism: front-loading quenches the oscillation in the first part
of the window, leaving most of the 2400 steps at low rates for the productive settling. Inverse-sqrt
with `c = 9` is aggressive but not reckless — still ≈0.60 of the peak at 10% progress, only dwelling
low after the midpoint — so it avoids the over-front-loaded failure (crashing to the floor early
strands the run at a useless rate). The 6% warmup is a downside guard, not a tuning lever: final loss
is insensitive to warmup length as long as it is long enough; the 240 extra steps just let AdamW's
moments settle before the long full-peak plateau.

**Step-3 edit (the literal scaffold fill).** Only the body of `get_lr` is replaced. The harness
exposes no `decay_type` switch — the one derived shape is hardcoded, with `import math` local to the
function for the square roots. Warmup uses `effective_warmup = max(warmup_iters, int(lr_decay_iters *
0.06))`; the decay window stays the settled last 20% (`decay_start = int(lr_decay_iters * 0.8)`); the
floor is `min_lr`.

**Hyperparameters.** Peak `learning_rate = 6e-4`, floor `min_lr = 6e-5`, warmup = max(481, 721) =
721 steps (6% floor), `lr_decay_iters = 12030`, cooldown window = last 20%, rescaled inverse-sqrt
cooldown (`c = 9`), seed 42. No `CONFIG_OVERRIDES`.

**What to watch.** With everything held from WSD except the front-loaded cooldown (plus the small
warmup extension), expect FineWeb val loss **below 2.2474** — a small, diminishing-returns
improvement on an already-good schedule — with perplexities loosely tracking (low-42 / mid-64,
noisier than the in-distribution val loss) and downstream accuracy near WSD's or a hair better.
Falsifier: if it does not beat linear on val loss, either linear already finished the descent over
2400 steps or `c = 9` over-front-loaded; `c = 9` keeps a substantial rate through 10% of the window,
so it is expected to land in the productive band and edge below WSD — the strongest of the three.

```python
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """WSD with inverse-sqrt decay and extended warmup."""
    import math
    # Extended warmup: 6% of training
    effective_warmup = max(warmup_iters, int(lr_decay_iters * 0.06))
    if it < effective_warmup:
        return learning_rate * (it + 1) / (effective_warmup + 1)
    # Decay phase: last 20% uses inverse-sqrt cooldown
    decay_start = int(lr_decay_iters * 0.8)
    if it >= decay_start:
        decay_len = lr_decay_iters - decay_start
        t = (it - decay_start) / decay_len
        # Inverse-sqrt decay: gentler than linear, reaches min_lr at t=1
        coeff = (1.0 / math.sqrt(1.0 + 9.0 * t) - 1.0 / math.sqrt(10.0)) / (1.0 - 1.0 / math.sqrt(10.0))
        return min_lr + (learning_rate - min_lr) * coeff
    # Stable phase
    return learning_rate
```
