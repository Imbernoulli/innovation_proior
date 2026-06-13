**Problem (from rmsnorm_parallel).** The parallel rung came in fastest but at the worst validation loss
(2.3112): it summed two un-normalized branch outputs into the residual and lost intra-block cross-talk at
355M. Pure pre-norm never normalizes a sublayer's *output* before adding it back, so residual-stream
variance grows monotonically with depth — the documented reason pre-norm leaves final loss on the table — and
the parallel rung exaggerated it. There is quality budget to reclaim by spending structure on normalization
rather than saving it.

**Key idea.** Keep RMSNorm (from rung 1) and replace the block with the **sandwich** placement: *two* norms
per sublayer — a pre-norm on the input (for the depth-stable gradient) and a **post-norm on the sublayer
output, applied before the residual add** — with the residual threaded between:
`x = x + LN_post(Attn(LN_pre(x)))`, then `x = x + LN_post2(MLP(LN_pre2(x)))` (CogView-style, Ding et al.
2021). This is *not* main-path post-LN (`LN(x + Attn(x))`); the residual path stays norm-free between blocks.

**Why.** `LN_pre` keeps the input distribution controlled (bounded, depth-independent gradient — the pre-norm
property kept). `LN_post` makes each block's *contribution* to the residual controlled in scale, bounding the
variance growth the way post-norm does, without putting a norm on the main residual path. So it recovers
post-norm's variance control and the parallel rung's lost cross-talk at once. Caveat: `LN_post`'s learned
gain can re-inflate the contribution, so the variance control is a soft prior — at 24 layers this may be
stability insurance more than a final-loss lever, so it is expected to beat the parallel floor but not
necessarily a plain sequential RMSNorm block. It is the most norm-heavy rung (four RMSNorms/block) and so the
slowest. The `1/√(2·n_layer)` init still matches the two residual writes per block; `LN_post` gains init 1
make initial contributions conservative, not larger.

**Hyperparameters.** RMSNorm `eps=1e-5`, gain init 1, no bias. Four RMSNorms per block
(`ln_pre1/ln_post1/ln_pre2/ln_post2`). No `CONFIG_OVERRIDES`. Same GPT-2 Medium, 12,030 iters, seed 42.

```python
# EDITABLE regions of nanoGPT/custom_pretrain.py — rmsnorm_post (sandwich)

# ── Normalization (lines 22–31) ──────────────────────────────────────────────
class LayerNorm(nn.Module):
    """RMSNorm — Root Mean Square Layer Normalization."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = 1e-5

    def forward(self, input):
        rms = input.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (input * rms).type_as(input) * self.weight


# ── Transformer Block (lines 88–100) ─────────────────────────────────────────
class Block(nn.Module):
    """Sandwich-Norm: Pre-LN + Post-LN with RMSNorm (CogView style)."""
    def __init__(self, config):
        super().__init__()
        self.ln_pre1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_post1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_pre2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.ln_post2 = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        x = x + self.ln_post1(self.attn(self.ln_pre1(x)))
        x = x + self.ln_post2(self.mlp(self.ln_pre2(x)))
        return x
```
