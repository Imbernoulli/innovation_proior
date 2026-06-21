## Research question

Pretrain a GPT-2 Medium language model (24 layers, 16 heads, d=1024, ~355M params) on FineWeb. The model, the data stream, AdamW, the batch construction, and the total update budget are frozen. The only free lever for the final validation loss is the **learning-rate schedule** — the function `get_lr` the training loop calls every iteration. The design target is the *shape* of that schedule over the 12,030 iterations of the run. Nothing else in the pipeline may be changed.

## Prior art / Background / Baselines

- **Step decay.** Hold the rate constant, then multiply it by a fixed factor (e.g. ×0.1) at hand-picked epoch boundaries.
- **Cosine annealing.** Replace the staircase with a smooth cosine curve over a fixed cycle length `T`. The single-cycle, no-restart version — warmup, then one cosine descent to a floor near 10% of the peak — is what large-model pretraining currently uses. The horizon `T` must be fixed in advance.
- **Cosine with warm restarts.** Reset the rate to its peak at the end of each cycle and decay again, optionally snapshotting for an ensemble.

## Fixed substrate / Code framework

A nanoGPT-style single-file pretraining loop is frozen. It builds GPT-2 Medium, configures AdamW (decoupled weight decay 0.1, `β=(0.9, 0.95)`, fused on CUDA, no weight decay on 1-D params), streams FineWeb through a memmap `get_batch`, runs AMP/bfloat16 forward-backward with gradient-clipping at 1.0, and drives a 2-GPU DDP loop for `max_iters = 12030` iterations with micro-batch 96 and gradient accumulation 6 (~1.18M tokens per iteration, ~7.1B training tokens total). Once per iteration it computes

```
lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
```

and writes that value into every AdamW parameter group before stepping. The four numbers passed to the schedule are fixed upstream: `learning_rate = 6e-4` (peak η), `min_lr = learning_rate / 10 = 6e-5` (floor), `warmup_iters = int(max_iters * 0.04)` (~481), and `lr_decay_iters = max_iters = 12030` (horizon). Evaluation (FineWeb validation loss, WikiText-2 / LAMBADA perplexity, and downstream lm-eval-harness tasks) is run after training and is not part of the edit surface.

## Editable interface

Only one region is editable: the body of `get_lr` in `nanoGPT/custom_pretrain.py` (the scaffold also exposes a `CONFIG_OVERRIDES` dict, but it is left empty). The contract is fixed:

- The signature must stay `get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)`.
- It is called every iteration with `it` as the 0-based step index, `learning_rate` as the peak η, `min_lr` as the floor, `warmup_iters` as the warmup length, and `lr_decay_iters` as the schedule horizon.
- It returns the scalar rate for this step. The total update budget is fixed; the schedule must not extend the number of training steps.

The starting point is the scaffold default: **single-cycle cosine with linear warmup**. Each run replaces exactly this function body and nothing else.

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
- **Downstream accuracy** — ARC-Easy and HellaSwag via lm-eval-harness (higher is better); PIQA and WinoGrande are tracked but hidden.

The model, data, optimizer, batch construction, total iteration budget, and warmup/peak/floor numbers handed to `get_lr` are all held fixed across every run; only the schedule shape varies.
