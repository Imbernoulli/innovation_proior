# Context: learning-rate schedules for large-scale language-model pretraining (circa 2023-2024)

## Research question

When you pretrain a decoder-only transformer language model with a fixed model size, a
fixed optimizer (AdamW), a fixed data stream and a fixed total update budget, the one knob
that still moves the final validation loss by a meaningful margin — after the architecture
and optimizer are settled — is the *learning-rate schedule*: the function that sets the
step size at every iteration. The question is how to shape that schedule across the full
pretraining run.

## Background

By this time the dominant recipe is well established. A pretraining run does a short
**linear warmup** of the learning rate from near zero up to a peak value `η` over the first
fraction of steps (to keep the adaptive optimizer stable while its second-moment estimates
are still cold), and then **anneals `η` down a cosine curve** to a small floor over the
rest of training. Cosine annealing came into deep learning from Loshchilov & Hutter (2016),
who replaced the older hand-tuned step-decay (drop the rate by a constant factor at a few
preset epochs) with a smooth schedule
`η_t = η_min + ½(η_max − η_min)(1 + cos(π · T_cur / T_i))`,
where `T_cur` counts steps since the last restart and `T_i` is the length of the cycle.
Their construction also allowed *warm restarts*: when the cosine reaches its floor, reset to
`η_max` and decay again over a (possibly longer) cycle. The single-cycle, no-restart version
of this curve — warmup, then one cosine descent to a floor at roughly 10% of the peak — is
what large-model practice adopted (Kaplan et al. 2020; Hoffmann et al. 2022; and the GPT /
LLaMA / Qwen / Falcon families).

The load-bearing fact about this curve is empirical and well-documented:

- **Total summed learning rate matters.** Kaplan et al. (2020) report that, holding the
  schedule family fixed, the final loss improves as the learning rate summed over the whole
  run increases — provided the schedule still has a warmup and still anneals to near zero at
  the end. A practical reading: a schedule that spends most of its budget at a high rate and
  only drops at the very end accumulates more total learning rate than one that starts
  descending immediately.
- **The cosine cycle length must equal the run length.** The key hyper-parameter of the
  cosine schedule is the step `T` at which it first reaches its floor; in practice `T` is
  set to the total number of training steps `S`. Hoffmann et al. (2022, "Chinchilla")
  observed that setting `T > S` (a cosine cycle longer than the run, so training stops while
  the rate is still high) measurably *hurts* the final model, while `T = S` is both best and
  most compute-efficient; and a cosine whose cycle is far longer than the run overestimates
  the loss it would actually reach at intermediate points. Reproductions on small models
  (tens of millions of parameters) confirm the pattern: across runs of `S = 20N, 40N, 60N,
  80N` tokens, the lowest loss is always achieved by the cosine whose cycle matches that
  exact `S`; both shorter and longer cycles lose.

There is a known, parallel lever as well: instead of decaying the learning rate one can
*increase the batch size* over training (Smith et al. 2017), trading one schedule for the
other. That is a different knob on the same underlying noise-scale tradeoff and is noted
here only as an alternative axis; the question at hand is about the learning-rate curve.

## Baselines

These are the prior schedules a new schedule would be measured against and would react to.

**Single-cycle cosine with linear warmup (Loshchilov & Hutter 2016; Kaplan et al. 2020;
Hoffmann et al. 2022).** Warm up linearly to `η`, then with normalized cosine progress
`u = (s - W)/(T - W)` use
`0.1·η + 0.45·η·(1 + cos(πu))` for `W <= s <= T`, floored at `0.1·η` afterward — a
smooth descent from the peak to one tenth of the peak over a cycle of length `T = S`. With
`T` matched to the run this attains the best loss at the target step count.

**Constant learning rate, and the periodic cosine "loop" (CosineLoop).** Keep the rate at
its warmup peak indefinitely, or run the same peak-to-floor half-cosine cycles back-to-back
without ever settling. A constant high rate maximizes the summed learning rate and never
depends on a horizon, so it is trivially open-ended. The periodic loop keeps re-warming the
rate across cycles.

**Multi-step / staircase decay (Bi et al. 2024, "DeepSeek LLM").** Warm up to the peak,
hold it, then drop the rate by a constant factor at preset fractions of the run: to ≈31.6%
of the peak after 80% of the tokens, and to ≈10% (a second ×0.316) after 90%. The reported
final performance is essentially on par with a tuned cosine, and because the first,
highest-rate phase is a flat plateau, the training done in that phase can be *reused* when
the run is later extended to a different length.

**Trapezoidal / constant-rate plus cooldown schedules (Hagele et al. 2024).** Another
nearby line keeps the rate constant for most of training and reserves a final cooldown for
annealing. The cooldown is written as a monotone function over the last `N_decay` steps, with
simple forms such as a linear descent `1 - u`, a half-cosine cooldown, and a `1 - sqrt(u)`
curve where `u` is progress through the cooldown. This line is useful because it separates
the long constant-rate regime from the annealing regime and studies the cooldown shape
directly.

**Cosine annealing with warm restarts (Loshchilov & Hutter 2016).** The original cosine
proposal also resets to `η_max` at the end of each cycle and decays again over a cycle that
grows by a factor `T_mult`, optionally snapshotting the model just before each restart for
an ensemble.

## Evaluation settings

The natural yardsticks already in use for a schedule change, holding model, data, optimizer,
and total update budget fixed:

- **Small-model wind-tunnel runs** at tens of millions to a couple of billion non-embedding
  parameters (e.g. 0.009B–2B), decoder-only transformers of the LLaMA/Noam style, used to
  measure schedule effects cheaply before committing to a large run. A representative
  configuration trains a ~0.036B model across token budgets of `S = 20N, 40N, 60N, 80N`
  (where `N` is the model's parameter count) to read off how each schedule's final loss
  varies with the run length.
- **Held-out validation cross-entropy / perplexity** on the pretraining distribution and on
  standard corpora; for cross-tokenizer comparison the loss is averaged per byte rather than
  per token.
- **Scaling-law fitting** as a downstream use: fit `L(N, D) = C_N N^{-α} + C_D D^{-β} + L_0`
  to a grid of model sizes `N` and data sizes `D`, with compute `C = 6ND`, to estimate the
  compute-optimal data-to-model ratio. The cost of populating the `(N, D)` grid is itself a
  metric: the standard approach needs a separate full run per cell.
- **Optimizer / parameterization held fixed:** AdamW with decoupled weight decay, a fixed
  peak learning rate found by a hyper-parameter search (and, in some setups, a
  width-transfer parameterization so the peak transfers across sizes), fixed batch size,
  fixed warmup fraction. The schedule shape is the only thing varied.

## Code framework

The schedule is consumed by an otherwise-fixed pretraining loop: each iteration, the loop
asks a `get_lr` function for the current learning rate and writes it into every optimizer
parameter group before the step. Everything else — the model, AdamW, the data pipeline, the
gradient-accumulation and DDP machinery, the total number of iterations — is already in
place and is not what is being designed. The single empty slot is the *shape* of `get_lr`:
given the iteration index and the four numbers the loop already knows (the warmup length,
the decay-reference length, the peak rate, and a floor rate), return the rate for this step.

```python
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Return the learning rate for iteration `it`.

    Called once per iteration by the fixed training loop, which then sets this
    value on every AdamW parameter group before stepping. The total update
    budget is fixed; the schedule must not extend the number of training steps.

    Args:
        it:             current iteration index (0-based).
        warmup_iters:   length of the initial warmup ramp.
        lr_decay_iters: reference horizon for the schedule.
        learning_rate:  peak learning rate eta reached after warmup.
        min_lr:         floor learning rate.
    """
    # TODO: fill in the schedule shape.
    pass
```

The loop supplies the iteration and the four schedule numbers; `get_lr` is where the
schedule shape will live, and is exactly the slot the design fills.
