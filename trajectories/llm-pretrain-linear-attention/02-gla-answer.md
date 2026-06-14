**Problem (from rung 1).** RetNet trained stably but landed weak (val_loss 2.4795, loose perplexities,
HellaSwag 31.12) because its forget gate is a single *fixed* scalar `γ` per head — a data-independent
gate that cannot choose its forgetting rate from content, exactly the weakness gated-RNN experience
warns against. The fix is to make the decay data-dependent without re-breaking the matmul chunkwise
training form that the scalar `γ` protected.

**Key idea (gated linear attention).** Replace the scalar decay with a **per-key-channel,
data-dependent forget gate**: `Sₜ = Diag(αₜ) S_{t−1} + kₜᵀvₜ`, `oₜ = (qₜ/√d_k) Sₜ`, with
`αₜ ∈ (0,1)^{d_k}` computed from `xₜ` alone (input-only gating, so the recurrence stays linear and
parallelizable). Unrolling, the per-step gates telescope into a cumulative product `bₜ = ∏_{j≤t} α_j`,
and the layer collapses to plain linear attention on *preconditioned* tensors `Q⊙B`, `K/B` — so the
chunkwise matmul form survives where a full-rank data-dependent transition would not. `bₜ` underflows,
so scores are computed in log space (`log Bₜ = Σ log α_j`, a data-dependent relative-position factor);
a two-level chunk tiling confines the full-precision log-space work to small diagonal sub-blocks while
off-diagonal blocks and the inter-chunk recurrence stay on tensor cores.

**Why it should beat RetNet.** A diagonal data-dependent gate can hold the *right* facts at the *right*
rate instead of an exponential average at a fixed rate, which is what cross-entropy on natural text
rewards — so val_loss should fall and the loose perplexities (where a fixed rate discarded salient
tokens) should tighten most. Open risk for the next rung: the gate is still *multiplicative/diagonal*,
so it forgets per-channel but cannot remove the specific stored association a new key collides with — a
residual recall failure would point at the write rule, not the decay.

**Scaffold edit / hyperparameters.** Import `fla.layers.GatedLinearAttention`. `mode = 'chunk'`,
`hidden_size = n_embd = 1024`, `num_heads = n_head = 16`, `expand_k = 0.5` (`d_k = d/2`),
`expand_v = 1.0` (`d_v = d`, full-width state for memory capacity), `use_output_gate = True`,
`gate_fn = 'swish'`. The low-rank (rank-16) log-space gate `logsigmoid(logits)/gate_logit_normalizer`
with temperature `τ = 16`, per-head RMSNorm, and the secondary-tiling chunk kernel are internal to the
FLA layer. Wrap the FLA call in a `@torch.compiler.disable` helper (torch.compile is off for this task;
the guard keeps the Triton kernel from ever being traced). `self.use_pos_emb = False` (the cumulative
decay encodes relative position). `Block` unchanged from the scaffold default — only the mixer differs
from rung 1.

```python
# EDITABLE region 1 of nanoGPT/custom_pretrain.py (lines 33-70) — GLA (gated linear attention)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from fla.layers import GatedLinearAttention
        self.attn = GatedLinearAttention(
            mode='chunk',
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            expand_k=0.5,                 # d_k = d/2
            expand_v=1.0,                 # d_v = d  (full-width state = memory capacity)
            use_output_gate=True,         # swish output gate
            gate_fn='swish',
        )
        self.use_pos_emb = False          # cumulative data-dependent decay encodes relative position

    @torch.compiler.disable
    def _attn_forward(self, x):
        return self.attn(x)

    def forward(self, x):
        o, _, _ = self._attn_forward(x)
        return o


# EDITABLE region 2 of nanoGPT/custom_pretrain.py (lines 88-100) — standard pre-norm block (unchanged)
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
