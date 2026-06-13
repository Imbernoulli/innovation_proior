**Problem.** The default block runs two sequential sublayers, each behind a full `LayerNorm` (mean,
variance, subtraction, gain, bias) — two norms per block, a strict serial dependency between attention and
MLP, four reductions per block per token, across 24 blocks and ~7B tokens. The first rung attacks the
largest structural inefficiency the edit surface exposes.

**Key idea.** Bundle two changes into the cheapest, most simplified block. (1) Replace `LayerNorm` with
**RMSNorm**: drop the mean-subtraction (which controls location, not spread) and the bias (which only
restores a location after recentering), keeping only the re-scaling normalization `a/RMS(a)·γ` — one
reduction instead of two, the re-scaling invariance and the self-regulating quadratic gradient preserved.
(2) Replace the sequential block with the **parallel** block: one shared norm feeds attention and MLP, and
both outputs are summed into the residual in a single update, `x = x + Attn(LN(x)) + MLP(LN(x))`.

**Why.** RMSNorm removes the part of the norm not responsible for stabilization, so quality should be ~neutral
on the norm rule. The parallel wiring halves the norms and shortens the critical path (the shared-norm /
shortened-dependency portion of the GPT-J/PaLM speed win; the fused input projection is inside the
out-of-edit `CausalSelfAttention`/`MLP` and is *not* realized here). The cost is representational: the two
sublayers no longer see each other within a block, a small-scale quality tax expected to be visible at 355M,
plus a slightly hotter residual stream from summing two un-normalized branches. This is the ladder's floor by
design: maximal simplification, lowest expected quality, fastest wall-clock. The `1/√(2·n_layer)` residual
init still matches the two residual writes per block, so the wiring change does not break the init contract.

**Hyperparameters.** RMSNorm `eps=1e-5`, gain init 1, no bias. One shared RMSNorm per block. No
`CONFIG_OVERRIDES` (learning rate, schedule, weight decay all left at the substrate defaults). Same GPT-2
Medium (24L/16H/1024d), 12,030 iters, seed 42.

```python
# EDITABLE regions of nanoGPT/custom_pretrain.py — rmsnorm_parallel

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
    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # Parallel: single norm, attention and MLP operate in parallel
        h = self.ln(x)
        x = x + self.attn(h) + self.mlp(h)
        return x
```
