**Problem.** A constant learning rate is structurally wrong at both ends — above the curvature ceiling
`2/lambda_max(H)` at the sharp initial region (early overshoot) and too large near the minimum, where the
stochastic gradient does not vanish (perpetual jitter at the noise floor). The rate must be
small-high-small. The first rung commits to the most interesting move in the schedule lineage: a rate
that rises before it falls.

**Key idea (constrained one-cycle).** Read straight off the loss topology — steep start, long flat
valley, narrow final trough — the rate should be small to begin without overshoot, large to cross the
valley and punch through saddle plateaus, small to settle the trough: one rise, one fall. Shape both legs
with a cosine (smooth, eases into/out of the extremes), short up-leg (first 30%) and long down-leg.

**Why it works here (and what the harness amputates).** The full one-cycle policy gets its power from a
*range-test peak* an order of magnitude above the usual rate, *rebalanced* (reduced) weight decay to
offset the large rate's regularization, and momentum *cycled inversely* to the rate for stability. This
task exposes only `get_lr` returning a per-epoch float; momentum (0.9) and weight decay (5e-4) are frozen
and there is no pre-run range test. So this is the tame port: peak at `base_lr` (not above), no momentum
cycling, no weight-decay change, floor at `base_lr/25` (not orders below). It is the rise-then-fall
*shape* without the super-convergence machinery — which is why it is the floor of the ladder.

**Hyperparameters.** `pct_start = 0.3` (up-leg fraction); `div_factor = 25` ⇒ start/floor at
`base_lr/25 = 0.004`; `final_div = 25` ⇒ end at `base_lr/25`; peak `= base_lr`. Cosine on both legs.
Progress measured over `total_epochs - 1`. No `arch`/`dataset` conditioning.

```python
def get_lr(epoch, total_epochs, base_lr, config):
    """OneCycleLR schedule (Smith & Topin, 2019).

    Phase 1 (0-30%): cosine warmup from base_lr/25 to base_lr.
    Phase 2 (30-100%): cosine anneal from base_lr to base_lr/25.
    """
    pct_start = 0.3
    div_factor = 25.0
    final_div = 25.0

    min_lr = base_lr / div_factor
    final_lr = base_lr / final_div

    progress = epoch / max(total_epochs - 1, 1)

    if progress <= pct_start:
        # Warmup phase: cosine from min_lr to base_lr
        t = progress / pct_start
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (1 - t)))
    else:
        # Anneal phase: cosine from base_lr to final_lr
        t = (progress - pct_start) / (1 - pct_start)
        return final_lr + (base_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * t))
```
