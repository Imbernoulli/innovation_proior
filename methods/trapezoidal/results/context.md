# Context: learning-rate schedules for large language model pretraining (circa 2023-2024)

## Research question

Pretraining a large language model is one long, expensive optimization run: a transformer is
trained with AdamW on tens to hundreds of billions of tokens, and the single most influential
training hyperparameter after the peak learning rate is the *schedule* — how the learning rate is
varied across the run. The prevailing schedule reaches a peak after a short warmup and then anneals
the learning rate down over the rest of training, and the quality of the final model depends on
this annealing being slow and reaching a low value by the end.

The question is how to schedule the learning rate so that training is flexible and efficient across
a range of scenarios — research scaling studies that need quality at multiple lengths, production
runs where the final length may shift, and settings where a single training run should yield
useful models at intermediate points.

## Background

**Why the learning rate must be annealed.** From optimization theory, a high learning rate is good
for *exploration* — it lets the iterate move freely across the loss landscape and escape poor
regions — while a *slow annealing* of the learning rate toward the end of training is what lets a
deep network settle into a good minimum rather than rattling around it (Smith 2017; Loshchilov &
Hutter 2016). A schedule is therefore a trade-off: spend most of the budget exploring at a high
rate, then cool down to commit to a basin.

**The cosine schedule and its length-coupling.** The cosine decay was introduced by Loshchilov &
Hutter (2016) for *cyclic* schedules with warm restarts in vision, where the rate is repeatedly
reset to a high value and annealed down over each cycle to escape bad minima across many epochs.
Within one cycle the rate follows

```
eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * T_cur / T_i)),
```

where `T_cur` counts steps since the last restart and `T_i` is the cycle length; at `T_cur = 0` the
rate is `eta_max`, at `T_cur = T_i` it is `eta_min`. For language models, where data is effectively
unbounded and training is a single pass, this was adopted as a *single* cosine cycle: warmup to the
peak, then one smooth cosine decay over the whole run, conventionally ending at 10% of the peak.
The shape is parameterized by the cycle length `T_i`, which is set to match the total training
length.

**The diagnostic finding of Hoffmann et al. (2022).** For a fixed family of models, the best final
loss at a given token count is obtained by a cosine cycle whose length matches that token count. For
a cosine schedule planned to decay over (say) 130B tokens, the loss measured at an intermediate
point `D' << 130B` is an overestimate of the loss a run planned to stop at `D'` would have
reached — because the bulk of the loss improvement comes from the decay, which has not happened yet
at the intermediate point. The methodologically clean approach they used was to train a separate
model, with a length-matched cosine, for every length of interest.

**A known alternative shape from vision.** In the vision-transformer scaling work of Zhai et al.
(2021), schedules were studied that keep the rate roughly constant (or on a reciprocal-square-root
path) for the main body of training and add a short final phase in which the rate is annealed down.
They report that this kind of held-high-then-cool schedule allows "indefinite training and
evaluating multiple training durations from just one run," and found a reciprocal-square-root main
phase with such a final annealing to perform best in their setting.

**The optimization-theory view of the final phase.** For convex Lipschitz objectives there is a
sharp result about the *last iterate* of SGD as a function of the step-size schedule. Defazio et al.
(2023) prove a last-iterate suboptimality bound and show that the schedule minimizing its worst case
is a *linear* decay of the step size to zero: with `eta_t = (D / (G * sqrt(T))) * (1 - t/T)`, the
last iterate satisfies `E[f(x_T) - f*] <= D*G/sqrt(T)`, achieving the optimal `O(1/sqrt(T))` rate
*without* the extra `log T` factor that the last iterate of a constant-step SGD carries. Their
reading is that a linear decay "emulates the effects of iterate averaging."

**Weight averaging as a noise-reducer.** Polyak (1992) averaging of iterates, and its deep-learning
form stochastic weight averaging (Izmailov et al. 2018), improve generalization by averaging out
the noise that a high-LR iterate carries. Sandler et al. (2023) show an equivalence between weight
averaging and learning-rate-decay schedules: averaging recent iterates produces an effect comparable
to having decayed the rate.

## Baselines

**Single-cycle cosine decay (Loshchilov & Hutter 2016, as used by Radford et al. 2018 onward).**
The de-facto standard for LLM pretraining: linear warmup to `eta_max`, then one cosine cycle
decaying to `eta_min` (typically `0.1 * eta_max`) over the whole run, per the equation above. Core
idea: a smooth, slow anneal that spends a long tail at progressively lower rates.

**Inverse-square-root schedule (as in T5, Raffel et al. 2019; PaLM, Chowdhery et al. 2023).**
Warmup then `eta ∝ 1/sqrt(step)`, with no dependence on a planned total length. Core idea: a
length-agnostic decay.

**Stepwise / restart schedules (cyclic SGDR; stepwise as in DeepSeek, Bi et al. 2024).** Drop the
rate by a constant factor at fixed intervals, or restart it. Core idea: discrete annealing.

**Linear decay to zero (Defazio et al. 2023).** Warmup then `eta_t ∝ (1 - t/T)` to zero. Core idea:
the worst-case-optimal last-iterate schedule from convex theory, beating cosine on average in their
large comparison.

**Stochastic weight averaging (Izmailov et al. 2018; Sandler et al. 2023).** Keep a running average
of the parameters along a constant-rate (or any) trajectory; the averaged model generalizes better
and behaves like a decayed-rate model. Core idea: replace decay with averaging.

## Evaluation settings

The natural yardsticks already in use for this comparison:

- **Models.** Decoder-only transformers (LLaMA/Noam-style: SwiGLU, RoPE, RMSNorm), trained with
  AdamW (`beta = (0.9, 0.95)`, decoupled weight decay `0.1`, gradient clipping `1.0`), across a
  range of sizes (tens of millions up to several billion parameters), with short warmups (a few
  hundred to a few thousand steps).
- **Data.** Standard pretraining corpora — SlimPajama, OpenWebText2, FineWeb / FineWeb-Edu — with a
  GPT-2-style BPE tokenizer, a held-out validation split, and several total token counts spanning
  the compute-optimal ratio (around 20 tokens per parameter) for scaling-law fits.
- **Metrics.** Validation cross-entropy / perplexity as the primary signal; downstream benchmark
  accuracy (MMLU, ARC, HellaSwag, PIQA, WinoGrande, etc.) as the eventual quantity of interest.
- **Protocol.** Sweep the peak learning rate for each schedule; for scaling, fit the loss as a power
  law `L(N, D) = A/N^alpha + B/D^beta + E` in parameters `N` and tokens `D`; account compute in
  FLOPs and wall-clock GPU hours. Compare schedules at matched model, data, optimizer, and total
  step budget.

## Code framework

A learning-rate schedule plugs into the standard pretraining loop as a single pure function that the
training loop calls once per iteration to set the optimizer's current rate. Everything around it
already exists — the model, the AdamW optimizer, the data pipeline, the loop that does forward /
backward / `optimizer.step()` — and is held fixed. The only thing to design is the body of `get_lr`:
given the iteration index and the run's warmup length, total-decay horizon, peak rate, and floor
rate, return the rate to use at that step. The signature is fixed by the loop; the body is one empty
slot.

```python
import math


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Return the learning rate to use at iteration `it`.

    Called once per training step. The total update budget `lr_decay_iters` is fixed and the
    number of steps is not extended. Arguments:
      it            : current iteration index (0-based)
      warmup_iters  : length of the initial linear warmup
      lr_decay_iters: horizon over which the schedule is defined
      learning_rate : peak learning rate (after warmup)
      min_lr        : floor learning rate
    """
    # TODO: the schedule we will design.
    #   Map the iteration `it` (relative to warmup_iters / lr_decay_iters) to a rate between
    #   min_lr and learning_rate. The shape of that map is exactly the contribution.
    pass


# existing training loop the schedule plugs into (fixed)
def train(model, optimizer, data_loader, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    it = 0
    for inputs, targets in data_loader:
        lr = get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)
        for group in optimizer.param_groups:    # set the rate for this step
            group["lr"] = lr
        optimizer.zero_grad()
        loss = model(inputs, targets)           # existing model + loss
        loss.backward()                         # backprop
        optimizer.step()                        # AdamW update at the current rate
        it += 1
```

The loop hands `get_lr` the iteration and the run's fixed `(warmup_iters, lr_decay_iters,
learning_rate, min_lr)`; the body returning the per-step rate is where the schedule will live.
