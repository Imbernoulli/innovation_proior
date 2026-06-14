# Context: learning-rate schedules for large language model pretraining (circa 2023-2024)

## Research question

Pretraining a large language model is one long, expensive optimization run: a transformer is
trained with AdamW on tens to hundreds of billions of tokens, and the single most influential
training hyperparameter after the peak learning rate is the *schedule* — how the learning rate is
varied across the run. The prevailing schedule reaches a peak after a short warmup and then anneals
the learning rate down over the rest of training, and the quality of the final model depends on
this annealing being slow and reaching a low value by the end.

This couples two things that practitioners would much rather keep separate. The schedule's *shape*
is tied to the *total number of steps*: the annealing is stretched to span exactly the planned
training length, so the schedule cannot be specified without first committing to how long training
will run. That commitment is costly in two regimes. First, in research that needs the model's
quality *at several different training lengths* — scaling-law fits, data-mixture comparisons,
architecture ablations — a separate model must be trained from scratch for each length, because the
loss read off partway through a long run is not the loss of a run that was *planned* to stop there.
Second, for a production run, the stopping point must be fixed in advance, and continuing past it,
or stopping early, both require either re-running or paying a penalty.

The precise goal is a schedule that (1) matches the final-model quality of the standard annealing
schedule under the same model, data, optimizer, and total step budget; (2) does *not* require the
total number of steps to be known up front, so a single run can yield optimal models at many
lengths and can be extended at will; and (3) is reliable across model sizes, so it can be trusted
in scaling studies. Achieving (1)+(2)+(3) together would cut the cost of scaling research by
roughly the number of lengths studied, and remove the need to predetermine a stopping point.

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
The defining property — and the source of the problem above — is that the shape is parameterized by
the cycle length `T_i`, which must equal the total training length for the decay to be stretched
correctly.

**The diagnostic finding that pins down the pain.** Hoffmann et al. (2022) established that, for a
fixed family of models, the best final loss at a given token count is obtained by a cosine cycle
whose length *matches* that token count. The flip side, which they report explicitly, is the
problem: for a cosine schedule planned to decay over (say) 130B tokens, the loss measured at an
intermediate point `D' << 130B` is an *overestimate* of the loss a run *planned* to stop at `D'`
would have reached — because the bulk of the loss improvement comes from the decay, which has not
happened yet at the intermediate point. Reading scaling behavior off intermediate checkpoints of a
single long cosine run therefore *underestimates* model quality at shorter lengths. The
methodologically clean fix they used — train a separate model, with a length-matched cosine, for
every length of interest — is exactly what makes scaling studies expensive: a family of model sizes
times several lengths, each trained from scratch.

**Continuation is brittle.** Because a cosine run is built to bottom out at its planned end, its
final rate is too low to make large further progress, so naively extending it stalls; and re-warming
the rate to continue causes loss spikes that training only slowly recovers from, and has been
reported to hurt performance and induce forgetting in the continual-learning setting (Ibrahim et al.
2024). So a finished cosine run is hard to either extend or read intermediate quality from.

**A known alternative shape from vision.** In the vision-transformer scaling work of Zhai et al.
(2021), schedules were studied that — to train for several durations and evaluate from a *single*
run — keep the rate roughly constant (or on a reciprocal-square-root path) for the main body of
training and add a short final phase in which the rate is annealed down. They report that this kind
of held-high-then-cool schedule allows "indefinite training and evaluating multiple training
durations from just one run," and found a reciprocal-square-root main phase with such a final
annealing to perform best in their setting. The decoupling of the main phase from a final annealing
phase is the load-bearing idea this body of work rests on; what such a schedule does for
*language-model* pretraining, and which annealing form and length are right there, was open.

**The optimization-theory view of the final phase.** For convex Lipschitz objectives there is a
sharp result about the *last iterate* of SGD as a function of the step-size schedule. Defazio et al.
(2023) prove a last-iterate suboptimality bound and show that the schedule minimizing its worst case
is a *linear* decay of the step size to zero: with `eta_t = (D / (G * sqrt(T))) * (1 - t/T)`, the
last iterate satisfies `E[f(x_T) - f*] <= D*G/sqrt(T)`, achieving the optimal `O(1/sqrt(T))` rate
*without* the extra `log T` factor that the last iterate of a constant-step SGD carries. Their
reading is that a linear decay "emulates the effects of iterate averaging." This convex theory is a
yardstick for what a final annealing phase should look like and why a constant rate alone leaves
something on the table.

**Weight averaging as a noise-reducer.** Polyak (1992) averaging of iterates, and its deep-learning
form stochastic weight averaging (Izmailov et al. 2018), improve generalization by averaging out
the noise that a high-LR iterate carries. Sandler et al. (2023) show an equivalence between weight
averaging and learning-rate-decay schedules: averaging recent iterates produces an effect comparable
to having decayed the rate. This is the relevant background for whether the explicit final-annealing
phase could be replaced by averaging instead.

## Baselines

**Single-cycle cosine decay (Loshchilov & Hutter 2016, as used by Radford et al. 2018 onward).**
The de-facto standard for LLM pretraining: linear warmup to `eta_max`, then one cosine cycle
decaying to `eta_min` (typically `0.1 * eta_max`) over the whole run, per the equation above. Core
idea: a smooth, slow anneal that spends a long tail at progressively lower rates. *Limitation:* the
cosine cycle is parameterized by its length and must be set to the total training duration to be
optimal (Hoffmann et al. 2022); a single run's intermediate checkpoints overestimate the loss of
length-matched shorter runs, so it cannot serve as a one-run source of quality estimates across
lengths, and it cannot be cleanly extended past its planned end (re-warming spikes the loss).

**Inverse-square-root schedule (as in T5, Raffel et al. 2019; PaLM, Chowdhery et al. 2023).**
Warmup then `eta ∝ 1/sqrt(step)`, with no dependence on a planned total length. Core idea: a
length-agnostic decay. *Limitation:* it has no distinct final phase that drives the rate to a low
value matched to a chosen stopping point, so its end-of-run loss does not match a length-matched
cosine, and stopping quality varies with where one happens to stop.

**Stepwise / restart schedules (cyclic SGDR; stepwise as in DeepSeek, Bi et al. 2024).** Drop the
rate by a constant factor at fixed intervals, or restart it. Core idea: discrete annealing.
*Limitation:* the step boundaries and factors are themselves a schedule that must be planned, and
restarts re-introduce the re-warming loss spikes; quality between drops is not length-matched.

**Linear decay to zero (Defazio et al. 2023).** Warmup then `eta_t ∝ (1 - t/T)` to zero. Core idea:
the worst-case-optimal last-iterate schedule from convex theory, beating cosine on average in their
large comparison. *Limitation:* the `1 - t/T` ramp begins immediately after warmup and is again
parameterized by the total length `T`; the rate spends the whole run below the peak, and the run is
no more extendable or one-run-reusable than cosine.

**Stochastic weight averaging (Izmailov et al. 2018; Sandler et al. 2023).** Keep a running average
of the parameters along a constant-rate (or any) trajectory; the averaged model generalizes better
and behaves like a decayed-rate model. Core idea: replace decay with averaging. *Limitation:* in
the LLM setting the size of the resulting improvement relative to an explicit final annealing, and
whether it fully closes the gap, was not established; averaging adds a copy of the parameters and
its window is a hyperparameter.

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
