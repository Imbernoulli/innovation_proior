## Research question

Pretrain a GPT-2 Medium language model (24 layers, 16 heads, d=1024, ~355M params) on FineWeb,
with the model, the data stream, the AdamW optimizer, the batch construction, and the total update
budget all frozen. The one lever still free to move the final validation loss is the
**learning-rate schedule** — the function `get_lr` that the training loop calls every iteration to
set the rate on every optimizer parameter group. The single thing being designed is the *shape* of
that schedule: how the rate is warmed up, held, and decayed across the 12,030 iterations of the run.
Everything else about the pipeline is fixed and must not be touched.

## Prior art before the first rung (learning-rate-schedule lineage)

The schedule the first rung reacts to — and the lineage the whole ladder climbs out of — is the
single-cycle cosine. These are the prior shapes; the editable interface below ships with the cosine
as its default fill.

- **Step decay (the pre-deep-learning default).** Hold the rate constant, then multiply it by a
  fixed factor (e.g. ×0.1) at a few hand-picked epoch boundaries. Simple and robust, but the drop
  points and factors are discrete hyperparameters tuned per run, and the abrupt jumps jolt the
  optimizer rather than easing the rate down. Gap: coarse, hand-tuned, no principled shape.
- **Cosine annealing (Loshchilov & Hutter 2016).** Replace the staircase with a smooth curve,
  `η_t = η_min + ½(η_max − η_min)(1 + cos(π·T_cur/T_i))`, where `T_cur` counts steps since the last
  restart and `T_i` is the cycle length. The single-cycle, no-restart version — warmup, then one
  cosine descent to a floor at ~10% of the peak — is what large-model pretraining adopted (Kaplan
  et al. 2020; Hoffmann et al. 2022; the GPT/LLaMA/Qwen families). Gap: the curve is welded to its
  cycle length `T`, and it is only near-optimal when `T` exactly equals the run length `S`.
- **The two load-bearing facts that pin `T = S`.** Kaplan et al. (2020): final loss improves as the
  learning rate *summed over the run* grows, provided there is still a warmup and a final anneal —
  so a cosine with `T < S` decays too early and bleeds summed rate. Hoffmann et al. (2022,
  Chinchilla): a cosine with `T > S` (stops while the rate is still high) measurably *hurts* the
  final model and overestimates intermediate-checkpoint loss. Both directions lose; the optimum sits
  exactly on the diagonal `T = S`. Gap: the rate at every middle step depends on a horizon fixed in
  advance, so runs cannot be cleanly extended and intermediate checkpoints are never converged.
- **Cosine with warm restarts (Loshchilov & Hutter 2016).** Reset the rate to `η_max` at the end of
  each cycle and decay again over a (growing) cycle, optionally snapshotting for an ensemble. Gap:
  in single-objective LLM pretraining the rewarming spikes spend summed rate on re-exploration and
  introduce instability; targets multimodality/ensembling, not a single horizon-free run.

The ladder below replaces the welded cosine curve with explicitly separated phases — a warmup, a
flat stable phase at the peak, and a short final decay — and then asks what the decay phase should
actually look like.

## The fixed substrate

A nanoGPT-style single-file pretraining loop is frozen. It builds GPT-2 Medium, configures AdamW
(decoupled weight decay 0.1, `β=(0.9, 0.95)`, fused on CUDA, no weight decay on 1-D params),
streams FineWeb through a memmap `get_batch`, runs the AMP/bfloat16 forward-backward with
gradient-clipping at 1.0, and drives a 2-GPU DDP loop for `max_iters = 12030` iterations with
micro-batch 96 and gradient accumulation 6 (≈1.18M tokens per iteration, ~7.1B training tokens
total). The loop computes, once per iteration,
`lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)` and writes that value
into every AdamW parameter group before stepping. The four numbers it hands the schedule are fixed
upstream: `learning_rate = 6e-4` (the peak `η`), `min_lr = learning_rate / 10 = 6e-5` (the floor),
`warmup_iters = int(max_iters * 0.04)` (≈481), and `lr_decay_iters = max_iters = 12030` (the
horizon). Evaluation (FineWeb val loss, WikiText-2 / LAMBADA perplexity, and the downstream
lm-eval-harness tasks) is run after training and is not part of the edit surface.

## The editable interface

Exactly one region is editable — the body of `get_lr` in `nanoGPT/custom_pretrain.py` (the
scaffold also exposes a `CONFIG_OVERRIDES` dict for a handful of optimizer scalars, but the ladder
leaves it empty: the schedule *shape* is the whole design). The contract is fixed:

- The signature must stay `get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)`.
- It is called every iteration with `it` the 0-based step index, `learning_rate` the peak `η`,
  `min_lr` the floor, `warmup_iters` the warmup length, and `lr_decay_iters` the schedule horizon.
- It returns the scalar rate for this step. The total update budget is fixed; the schedule must not
  extend the number of training steps.

The starting point is the scaffold default: **single-cycle cosine with linear warmup**. Each rung
of the ladder replaces exactly this function body and nothing else.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 191-201) — default fill: cosine
import math

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

## Evaluation settings

A single seed (42), one fixed configuration. Metrics, run after training on the final model:

- **Validation loss** — cross-entropy on held-out FineWeb (lower is better; **primary**).
- **Perplexity** — WikiText-2 and LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy and HellaSwag via lm-eval-harness (higher is better); PIQA and
  WinoGrande are tracked but hidden.

The model, data, optimizer, batch construction, total iteration budget, and warmup/peak/floor
numbers handed to `get_lr` are all held fixed across every rung; only the schedule shape varies.
