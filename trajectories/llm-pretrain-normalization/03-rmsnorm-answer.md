**Problem (from rmsnorm_post).** The sandwich rung beat the parallel floor by only 0.0008 val_loss (2.3104
vs 2.3112) while *raising* the perplexities (WikiText-2 46.8, LAMBADA 72.08) and PIQA dropping to 62.46, at
the slowest wall-clock on the board. That is variance insurance, not improvement: four norms per block
over-constrained a 24-layer model that did not need it. Both restructuring experiments — parallel
(simplify, lost cross-talk) and sandwich (elaborate, over-constrained) — failed to buy quality. The one
change safe every time was the normalization *rule*.

**Key idea.** Stop restructuring the block. Keep **RMSNorm** and return to the **plain sequential pre-norm**
block — the default block with `LayerNorm` swapped for `RMSNorm` and nothing else:
`x = x + Attn(RMSNorm(x))`, then `x = x + MLP(RMSNorm(x))`. The minimal change from the default fill.

**Why.** This sits at the sweet spot the two restructured rungs straddled. Sequential ordering restores the
intra-block cross-talk the parallel rung lost (the MLP reads the post-attention residual). One norm per
sublayer removes the output-variance constraint that over-constrained the sandwich and pushed its perplexities
up. The clean pre-norm identity path keeps the depth-stable, warmup-free gradient. RMSNorm itself is the safe
swap: drop the mean-subtraction (controls location, `var(a−μ)=var(a)`, not spread) and the bias (nothing to
recenter, and `bias=False` anyway), keep `a/RMS(a)·γ` whose quadratic form gives a self-regulating gradient
(`∂L/∂W` invariant to input scale, inversely proportional to weight scale — larger weights, smaller steps).
Same parameter count as the others (fewer, no biases), so the win is placement, not capacity — expected to be
the *best* val_loss of the three with the perplexities *reversing* the sandwich's regression. This is the
block the substrate's `1/√(2·n_layer)` init and schedule were tuned for, so no init-vs-wiring mismatch.

**Hyperparameters.** RMSNorm `eps=1e-5`, gain init 1, no bias. Two RMSNorms per block (default sequential
Pre-LN block, unchanged). No `CONFIG_OVERRIDES` (kept at substrate defaults precisely because this is the
arrangement they were tuned for). Same GPT-2 Medium (24L/16H/1024d), 12,030 iters, seed 42.

```python
# EDITABLE regions of nanoGPT/custom_pretrain.py — rmsnorm (plain Pre-LN)

# ── Normalization (lines 22–31) — REPLACED ───────────────────────────────────
class LayerNorm(nn.Module):
    """RMSNorm — Root Mean Square Layer Normalization."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = 1e-5

    def forward(self, input):
        rms = input.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (input * rms).type_as(input) * self.weight


# ── Transformer Block (lines 88–100) — UNCHANGED from the default fill ────────
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```
