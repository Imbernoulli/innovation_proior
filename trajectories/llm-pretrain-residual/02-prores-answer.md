**Problem (from step 1).** Vanilla Pre-LN trains fine (val_loss 2.2763) but its residual stream is a
fixed unit-weight accumulator: the stream variance climbs with depth, shrinking deep-block Jacobians
toward identity so the deepest layers go half-dead, and the branch fires full-strength from init. The
floor leaves depth capacity on the table; the next rung must keep the clean highway but control *when
and how much* each branch writes.

**Key idea.** ProRes (Progressive Residual Warmup): multiply each layer's residual *branch* by a
**predefined, non-learnable** scalar `α(l,t)` that starts at 0 (exact identity at init) and ramps to 1
over training, with deeper layers warming up more slowly. Branches switch on in a shallow-to-deep wave;
deep layers engage at full strength only once the layers beneath them have stabilized. Default linear
schedule `α(l,t) = min(t/(T·l), 1)`, `T = 1000` (the first-layer warm-up length), so layer `l` reaches
full strength at step `T·l` and thereafter the block runs exactly as vanilla.

**Why it works.** *Identity at init* (`α(l,0)=0`): no early variance blowup, well-conditioned gradients
— ReZero's good start, made exact and parameter-free. *Bounded update across depth and time*: early on
only shallow layers have nonzero `α`, and the constraint relaxes itself layer-by-layer — tight in the
chaotic warm-up, fully released in the stable phase (unlike a constant init-time bound or static
`1/√l`). *Shallow-before-deep ordering*: a deep branch stays near 0 while the shallow layers below do
their large early updates, so deep layers build on stabilized inputs instead of injecting random noise
into the representations above and the gradients below — the order is *imposed*, not left to the
optimizer (the ReZero gap).

**Hyperparameters.** `T = 1000`, linear schedule, untuned; the skip stays at weight 1 (scale the branch
only, so `α=0` is exactly identity and `α=1` exactly vanilla). **Zero learnable parameters** — the
optimizer and `CONFIG_OVERRIDES` are unchanged.

**Scaffold wrinkle.** `α(l,t)` depends on the global step, which `Block` does not see, so the schedule
lives in the `GPT.forward` loop, carried by a non-trainable `_prores_step` buffer. The block returns
`block_out = x + delta` (full Pre-LN residual), so the branch contribution is `delta = block_out − x`,
scaled by `α`; the step is read before the loop (first forward is `t=0`) and incremented after.

**What to watch.** A *modest* val-loss drop into the low-2.27s (conditioning fix, not new capacity),
clearest on perplexity — WikiText-2 and especially LAMBADA at or below 44.28 / 70.09. Risk: at 24
layers / 13.5k steps the variance explosion may be mild, so ProRes lands near the floor. If so, the next
rung should *learn* the per-layer weight and add an embedding-injection route the schedule never gives.

```python
# EDITABLE regions of custom_pretrain.py — step 2: ProRes (progressive residual warmup)
# (each region shown inside its enclosing method exactly as spliced into the file)

# Block: unchanged — vanilla Pre-LN residual.

class GPT(nn.Module):
    def _init_prores(self, config):  # GPT.__init__ residual region:
        # ── ProRes: progressive residual warmup ──
        # T controls the warmup period; deeper layers take T*layer_idx steps
        # to reach full contribution.  step counter is a non-parameter buffer.
        self.prores_T = 1000
        self.register_buffer('_prores_step', torch.zeros(1, dtype=torch.long))

    def _forward_block_loop(self, x):  # GPT.forward block loop:
        # ── ProRes: progressive residual warmup per block ──
        # Increment step counter once per forward (training only).
        if self.training:
            self._prores_step += 1
        step = self._prores_step.item()
        T = self.prores_T
        for i, block in enumerate(self.transformer.h):
            block_out = block(x)
            if self.training and step < T * (i + 1):
                # alpha ramps from 0 to 1 over T * layer_idx steps
                layer_idx = i + 1
                alpha = min(step / (T * layer_idx), 1.0)
                # block_out = x + delta (Pre-LN residual), so delta = block_out - x
                x = x + alpha * (block_out - x)
            else:
                x = block_out
        return x

# GPT.configure_optimizers: unchanged (no new learnable params).
# CONFIG_OVERRIDES = {}   (no override).
```
