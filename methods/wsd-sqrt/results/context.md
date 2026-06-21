## Research question

Pretraining a decoder-only transformer language model means running one very long stochastic
optimization with AdamW, and the single curve that governs how far the optimizer is allowed to
move at each step — the learning-rate schedule — has an outsized effect on the final loss. The
prevailing recipe is a linear warmup followed by a cosine decay of the learning rate down to a
small floor, with the cosine cycle stretched to cover the whole run. The empirical finding is that
the schedule is near-optimal when the cycle length `T` is set equal to the total number of
training steps `S`. The question is: what learning-rate schedule should be used for LLM
pretraining across different token budgets, and how should the schedule shape be chosen?

## Background

By this period the cosine learning-rate schedule is the de-facto standard for LLM pretraining.
It comes from warm-restart SGD (Loshchilov & Hutter, "SGDR", 2016, arXiv:1608.03983), where a
half-cosine drives the rate from its peak down to a floor; in the LLM setting a single cycle is
used, typically decaying to about 10% of the peak. After warmup the rate follows

```
0.1 * eta + 0.9 * eta * 0.5 * (1 + cos(pi * r)),   r = (s-W)/(T-W),
```

with peak `eta`, warmup end `W`, and cycle length `T`; for `s > T` it stays at the floor `0.1 eta`
(the "CosineLoop" variant instead lets the cosine keep oscillating forever). The general wisdom
behind any such curve — going back to the warm-restart and large-batch literature (Smith et al.
2017; Loshchilov & Hutter 2016) — is a trade-off: a *high* learning rate early aids exploration of
the loss landscape, while a *slow anneal* to a low rate at the end is what lets a deep network
settle into a good minimum.

The motivating empirical facts about this setting, established before any new schedule:

- **The cosine cycle length and run length.** Kaplan et al. (2020, arXiv:2001.08361) observe
  that the loss decreases as the *summed* learning rate over the whole run increases, which means
  setting `T < S` (decaying too early, so the rate spends less total mass high) is suboptimal.
  Hoffmann et al. ("Chinchilla", 2022, arXiv:2203.15556) observe the other side: setting `T > S`
  (decay not finished by the end) drops performance, while `T = S` gives the best loss and the
  most efficient training. Together these pin the optimum at `T = S`: the cosine must be exactly
  as long as the run. Reproductions on small (~0.036B) models confirm that for runs of
  `S = 20N, 40N, 60N, 80N` tokens the lowest loss is always at the cosine with `T = S`, and both
  `T < S` and `T > S` are worse.
- **A constant rate followed by a short final decay was already in use.** Zhai et al. ("Scaling
  Vision Transformers", 2021, arXiv:2106.04560) used an "infinite learning rate" schedule —
  warmup, then a constant (or reciprocal-root) main phase, then a short sharp linear decay — so
  that the final decay could be branched off at different lengths. DeepSeek-LLM (Bi et al. 2024,
  arXiv:2401.02954) used a multi-step rate that, of prior LLM schedules, most resembles a
  staged constant-then-decay shape. These are the constant-plus-cooldown ancestors.

The natural mathematical frame for "how good is a schedule" is the data-model scaling law
(Hoffmann et al. 2022): fit `L(N, D) = C_N N^{-alpha} + C_D D^{-beta} + L_0` for model size `N`
and data size `D`, from which a compute-optimal data-to-model ratio at fixed compute `C = 6 N D`
follows.

## Baselines

These are the prior schedules a new schedule is measured against and reacts to.

**Cosine schedule with linear warmup (Loshchilov & Hutter 2016; standard LLM practice).** Warmup
linearly to the peak `eta`, then a half-cosine from `eta` down to a floor (commonly `0.1 eta`)
over a cycle of length `T`:

```
get_lr(s):  (s/W) * eta                          if s < W      # linear warmup
            0.1*eta + 0.9*eta * 0.5*(1+cos(pi*r))  if W <= s <= T  # r = (s-W)/(T-W)
            0.1*eta                              if s > T
```

A single smooth curve that interleaves the high-rate exploration phase and the decay into one
shape. The cycle length `T` is set to match the total training steps `S`.

**Constant learning rate (no decay).** Hold `eta` fixed after warmup. The run can stop or
continue anywhere without a predetermined length commitment.

**Constant rate with a final linear cooldown (the "infinite-LR" / trapezoidal shape; Zhai et al.
2021; DeepSeek-LLM 2024).** Warmup, then a long constant plateau, then a short final phase that
decays the rate linearly to (near) zero:

```
get_lr(s):  (s/W) * eta                          if s < W              # warmup
            eta                                  if W <= s < S - D     # constant plateau
            eta * (1 - (s-(S-D))/D)              if S - D <= s < S     # linear cooldown
```

This decouples the high-rate phase (now unbounded in length) from the anneal (a short tail), so
the cooldown can be started from a plateau checkpoint at any step.

## Evaluation settings

The natural yardsticks already in place at this time, for a schedule-only change with the model,
data, optimizer, and total update budget held fixed:

- **Architecture and optimizer:** decoder-only GPT-style transformers (in the small-LM regime,
  models from ~0.04B to a few B non-embedding parameters), trained with the AdamW optimizer
  (Kingma & Ba 2014; Loshchilov & Hutter decoupled weight decay 2017), with linear warmup.
- **Data:** large deduplicated English (and code) web corpora used for LLM pretraining (e.g.
  SlimPajama, C4, the Pile, FineWeb-style web text) tokenized with a standard BPE tokenizer;
  reported in tokens and, when comparing tokenizers, in loss-per-byte.
- **Primary metric:** held-out cross-entropy / validation loss (equivalently perplexity) on the
  pretraining distribution, lower is better — this is the quantity a schedule is judged on.
- **Auxiliary metrics:** perplexity on held-out corpora (e.g. WikiText, LAMBADA) and zero/few-shot
  downstream accuracy on standard benchmarks (e.g. ARC, HellaSwag, PIQA, WinoGrande).
- **Scaling-law protocol:** fit `L(N, D) = C_N N^{-alpha} + C_D D^{-beta} + L_0` across a grid of
  model and data sizes via least-squares curve fitting, and read off the compute-optimal
  data-to-model ratio at fixed compute `C = 6 N D`. The cost of populating that grid — one run
  per `(N, D)` pair under cosine — is itself part of what is being evaluated.
- **Protocol:** identical model, data ordering, optimizer, batch size, and total step budget
  across schedules; only the per-step learning-rate value differs. Decay length is expressed as a
  fraction of the total steps and swept across short and long cooldowns.

## Code framework

The schedule plugs into an existing nanoGPT-style pretraining loop. The model, the AdamW
optimizer, the batch sampler, the loss, and the total step budget all already exist and are
fixed; the only thing the training loop asks for at each iteration is a single scalar learning
rate, which it writes into the optimizer's parameter groups before that step. So the substrate is
just: a function `get_lr(it, ...)` returning the rate for iteration `it`, and the loop that calls
it. The shape of that function — how the rate moves over the course of training — is the one open
slot.

```python
import math

# Fixed: GPT-style model, AdamW optimizer, data sampler, loss, total step budget.
# The only knob exposed at the schedule layer is the per-iteration learning rate.

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Return the learning rate for iteration `it`.

    it             : current iteration (0-indexed)
    warmup_iters   : number of linear-warmup iterations
    lr_decay_iters : iteration by which the schedule has fully decayed (~ total steps)
    learning_rate  : peak learning rate (eta)
    min_lr         : floor learning rate
    """
    # Linear warmup is the one settled piece: ramp from ~0 up to the peak.
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Open slot: decide how the rate should move after warmup.
    raise NotImplementedError("learning-rate schedule shape is unspecified")


# existing training loop the schedule plugs into
def train(model, optimizer, data, total_iters,
          warmup_iters, lr_decay_iters, learning_rate, min_lr):
    for it in range(total_iters):
        lr = get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)
        for group in optimizer.param_groups:   # write the scalar rate into AdamW
            group['lr'] = lr
        x, y = data.next_batch()               # fixed sampler
        loss = model(x, y)                      # fixed model + loss
        loss.backward()
        optimizer.step()                        # fixed AdamW
        optimizer.zero_grad(set_to_none=True)
```

The loop supplies the iteration index and the four schedule parameters; `get_lr` is where the
schedule shape will live.
