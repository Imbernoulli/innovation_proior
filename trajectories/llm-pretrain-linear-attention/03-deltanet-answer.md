**Problem (from rung 2).** GLA's data-dependent diagonal gate improved bulk language modeling (val_loss
2.4482, WikiText-2 64.32, ARC-Easy 53.11) but *regressed on recall*: LAMBADA rose to 84.73 (worse than
RetNet's 82.36) and WinoGrande fell to 49.88. A multiplicative/diagonal gate scales the whole state but
cannot remove the *specific stored association* a new key collides with — the failure is no longer in
the decay, it is in the additive Hebbian **write rule** whose readout
`S k_j = v_j(k_jᵀk_j) + Σ_{i≠j}(k_iᵀk_j) v_i` is corrupted by cross-talk once `L` exceeds the key dim.

**Key idea (the delta rule).** Replace the additive write with an *error-correcting* one — one
LMS/Widrow-Hoff gradient step on `½‖S kₜ − vₜ‖²`:
`Sₜ = S_{t−1}(I − βₜ kₜ kₜᵀ) + βₜ vₜ kₜᵀ`, `oₜ = Sₜ qₜ`, with learned writing strength
`βₜ = σ(W_β xₜ) ∈ (0,1)`. Equivalently: retrieve `vₜ^old = S_{t−1} kₜ`, blend
`vₜ^new = βₜ vₜ + (1−βₜ) vₜ^old`, swap it in. The write scales with the *error*, so it deallocates a
stale association for a colliding key instead of accumulating interference. With L2-normalized keys the
only non-unit eigenvalue is `1 − βₜ ∈ [0,1]` (always stable), and at `βₜ = 1` the transition
`I − kₜ kₜᵀ` is an orthogonal *projection* — surgical, content-addressed forgetting.

**Why it parallelizes.** The state stays additive, `Sₜ = Σ uᵢ kᵢᵀ` with pseudo-values
`uₜ = βₜ(vₜ − Σ_{i<t} uᵢ(kᵢᵀkₜ))`; the same triangular recurrence gives the WY pseudo-keys `wₜ`. Both
are one lower-triangular system, solved in closed form `T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β)`,
`U = T V`, `W = T K`, with `I+L` inverted by forward substitution (UT transform). Everything becomes
matmuls — `O(LCd + Ld²)` FLOPs, `O(L/C)` sequential steps — the same hardware profile as the GLA chunk
kernel.

**Why it should be the strongest rung.** Higher capacity and cleaner retrieval help bulk loss, but the
real test is recall: **LAMBADA should fall well below both 84.73 (GLA) and 82.36 (RetNet)**, WikiText-2
below 64.32, and HellaSwag (stuck at 31.1 on both prior rungs) and ARC-Easy should rise — because the
projection-style erase is exactly the clean retrieval those tasks reward.

**Scaffold edit / hyperparameters.** Import `fla.layers.DeltaNet`. `hidden_size = n_embd = 1024`,
`num_heads = n_head = 16`, `use_beta = True` (the learned `βₜ`), `use_short_conv = True`, `conv_size = 4`
(local-mixing short conv), `qk_activation = 'silu'`, `qk_norm = 'l2'` (SiLU-then-L2 → exact projection
at `βₜ = 1`). Defaults `expand_k = 1.0`, `expand_v = 1.0` (state `d×d`, matched width) and
`use_gate = False` (**no** swish output gate, unlike RetNet/GLA — this rung isolates the write-rule
change; only the per-head output RMSNorm is kept). `self.use_pos_emb = False` (the recurrence + short
conv handle ordering). `Block` unchanged from the scaffold default.

```python
# EDITABLE region 1 of nanoGPT/custom_pretrain.py (lines 33-70) — DeltaNet (delta-rule linear attention)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from fla.layers import DeltaNet
        self.attn = DeltaNet(
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            use_beta=True,                # learned writing strength beta_t (the delta rule)
            use_short_conv=True,
            conv_size=4,                  # depthwise short conv for local token comparison
            qk_activation='silu',
            qk_norm='l2',                 # L2-normalized keys -> transition is a projection at beta=1
        )
        self.use_pos_emb = False          # DeltaNet handles sequence ordering internally

    def forward(self, x):
        o, _, _ = self.attn(x)
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
