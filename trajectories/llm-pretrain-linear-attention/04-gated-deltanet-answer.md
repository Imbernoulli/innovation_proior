**Problem (from rung 3).** DeltaNet's error-correcting write won the recall axis (val_loss 2.3481,
LAMBADA 70.48, HellaSwag 32.77) but its Householder transition `I − βₜ kₜ kₜᵀ` has **no decay term** —
it erases exactly the overwritten direction and leaves everything else intact forever. So it cannot do
*global* memory fading, and WinoGrande (the most coreference/over-a-span task) stalled at 49.17, below
GLA's 49.88 and RetNet's 52.01. The ladder fixed the decay axis (GLA) and the write axis (DeltaNet)
separately; neither rung had both.

**Key idea (the gated delta rule).** Apply both fixes on their orthogonal axes of the same recurrence —
a data-dependent global scalar decay composed with the content-addressed delta write:

`Sₜ = αₜ (I − βₜ kₜ kₜᵀ) S_{t−1} + βₜ vₜ kₜᵀ`,  `oₜ = Sₜ qₜ`,

with `αₜ ∈ (0,1]` a **per-head scalar** data-dependent decay and `βₜ = σ(W_β xₜ)` the learned writing
strength. It is a strict generalization: `αₜ = 1` recovers DeltaNet; setting `βₜ = 0` drops the delta
write and leaves the pure scalar-gated decay skeleton `Sₜ = αₜ S_{t−1}` shared with the Mamba2 / GLA
family (that family adds its own write back; here the gate is the part being shared). The gate is the
global eraser DeltaNet lacked; the delta
rule is the local scalpel GLA lacked.

**Why it should clear the bar.** A scalar `αₜ` keeps the chunkwise UT-transform training form intact (a
scalar telescopes cleanly, like RetNet's `γ`), so the matmul-rich kernel survives; it is parameterized
Mamba2-style `αₜ = exp(−exp(A_log)·softplus(a_proj(xₜ)+dt_bias))`, near 1 at init (long-memory prior,
the gate does not collapse). With L2-normalized keys the combined contractive factor `αₜ(1−βₜ) ∈ [0,1]`
is stable. Expectation against DeltaNet: hold its recall gains (val_loss ≤ 2.3481, LAMBADA ≤ 70.48,
since at `αₜ≈1` it *is* DeltaNet) **and** recover the eraser deficit (WinoGrande back above 49.17 toward
RetNet's 52.01). The output gate that rung 3 dropped is **re-added** (`use_gate=True`): the finale is the
union of both lineages, not the write-rule axis in isolation, so the GLA/RetNet output-gate recipe
applies again.

**Scaffold edit / hyperparameters.** Import `fla.layers.GatedDeltaNet`. Head shaping follows the Mamba2
convention `num_heads·head_dim = 0.75·hidden_size` when `use_gate=True` — for GPT-2 Medium
(`hidden_size=1024`) pass `head_dim=128`, `num_heads=6` (`6·128 = 768`), `expand_v=2.0` (value head dim
256, value width 1536), landing at the documented ~`6·d²` budget. `use_gate=True` (re-added swish output
gate via fused gated RMSNorm), `use_short_conv=True`, `conv_size=4`. The data-dependent log-decay
`g = −exp(A_log)·softplus(a_proj(x)+dt_bias)`, `β = sigmoid(b_proj(x))`, SiLU+short-conv on q/k/v, and
in-kernel q/k L2-norm are all internal to the FLA layer. `self.use_pos_emb=False` (decay + recurrence
encode position). `Block` unchanged from the scaffold default.

```python
# EDITABLE region 1 of nanoGPT/custom_pretrain.py (lines 33-70) — Gated DeltaNet (gated delta rule)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from fla.layers import GatedDeltaNet
        # num_heads * head_dim = 0.75 * hidden_size when use_gate=True (Mamba2 budget convention):
        # 6 * 128 = 768 = 0.75 * 1024
        self.attn = GatedDeltaNet(
            hidden_size=config.n_embd,
            head_dim=128,
            num_heads=6,
            expand_v=2.0,                 # value head dim 256 (value width 1536)
            use_gate=True,                # re-add the swish output gate (fused gated RMSNorm)
            use_short_conv=True,
            conv_size=4,
        )
        self.use_pos_emb = False          # scalar decay + delta recurrence encode position

    def forward(self, x):
        o, _, _ = self.attn(x)            # FLA returns (output, attn_weights, past_kv)
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
