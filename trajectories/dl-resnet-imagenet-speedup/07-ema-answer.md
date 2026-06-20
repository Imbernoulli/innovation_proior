**Problem (from step 6).** SGD converges to a noisy distribution around the minimum, not a point, so the
final iterate is one jittery draw and validation accuracy bounces between checkpoints. For a recipe trying to
shorten the schedule to the minimum that still clears 76.6%, that jitter forces a wasteful safety margin.
Need cheap end-of-training variance reduction.

**Key idea — EMA (Exponential Moving Average).** Maintain a shadow copy of the weights and pull it a small
fraction toward the live weights each step: `θ_EMA ← s·θ_EMA + (1 − s)·θ`, with smoothing `s` close to 1 (a
half-life of ~1000 batches). This is a recency-weighted average — old (bad) iterates fade as `s^k`, recent
(good) ones dominate — so θ_EMA sits nearer the center of the basin than any single iterate. Evaluate the
*averaged* model, not the training model.

**Why it works.** Near a minimum the loss is bowl-shaped, so averaging weights scattered around the bowl lands
nearer the bottom and smooths per-step jitter, giving less-noisy validation metrics (and sometimes genuine
extra generalization). Two correctness points: the EMA must track batchnorm **buffers** too, not just
parameters, or evaluation gets mismatched normalization stats; and evaluation must use the EMA model (a real
trap — evaluating the training model gives none of the benefit). Cost is small: one weighted-add over the
params (optionally only every `update_interval` steps, fine as long as half-life ≫ interval) plus extra memory
for the shadow copy. Model-averaging methods don't compose — use EMA *or* SWA, not both.

**Change / code.** The functional core `compute_ema` updates the shadow copy in place over both parameters and
buffers.

```python
def compute_ema(model, ema_model, smoothing: float = 0.99) -> None:
    model_context_manager = get_model_context_manager(model)
    with model_context_manager:
        with torch.no_grad():
            if isinstance(ema_model, torch.nn.Module):
                ema_params = ema_model.state_dict()
                # iterate over BOTH parameters and buffers (buffers = batchnorm stats)
                for name, param in itertools.chain(model.named_parameters(),
                                                   model.named_buffers()):
                    if name in ema_params:
                        ema_params[name].copy_(
                            ema_params[name] * smoothing + param.data * (1. - smoothing))
            ...

# Usage: ema = EMA(half_life='1000ba'); evaluate with the averaged weights:
#   model = ema.ema_model
```
