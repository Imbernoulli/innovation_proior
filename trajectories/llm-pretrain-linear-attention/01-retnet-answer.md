**Problem.** Softmax attention is `O(L²)` in compute and memory and grows an unbounded KV cache at
inference. The first rung of the ladder is the cheapest *credible* subquadratic mixer: linear attention
with the simplest forget mechanism that does not break parallel training — a single fixed scalar decay
per head.

**Key idea (multi-scale retention).** Derive the mixer from a linear recurrence with a matrix state,
`Sₙ = A s_{n−1} + kₙᵀvₙ`, `oₙ = qₙ Sₙ`; unrolled it is already a causal weighted sum
`Σ_{m≤n} qₙ A^{n−m} kₘᵀ vₘ`. Diagonalizing `A` and absorbing the eigenvectors into `W_Q, W_K` turns the
matrix power into a scalar decay `γ^{n−m}` times a rotary phase `e^{i(n−m)θ}` (an xPos-style relative
position encoding). This gives one function with three equivalent faces: a parallel form
`(QKᵀ ⊙ D) V` with the combined causal-decay matrix `D_{nm}=γ^{n−m}` for training; a recurrent form
`Sₙ = γ S_{n−1} + KₙᵀVₙ`, `oₙ = Qₙ Sₙ` for `O(1)` inference; and a chunkwise form for linear-time
long-sequence training (what the FLA kernel runs). A **different `γ` per head** spans multiple memory
horizons; per-head normalization balances the heads, and a **swish output gate** restores the
nonlinearity that deleting softmax removed.

**Why it is the floor.** Retention's decay is *fixed and data-independent* — one `γ` per head, chosen a
priori. It is a real forget gate (strictly better than no decay) but it cannot choose its forgetting
rate from content, which is exactly the data-independent-gate weakness the gated-RNN literature warns
against. So it should train stably and land in the credible range, but be the **weakest** rung — the
least information-adaptive memory — with the gap showing most on perplexity and recall-flavored
downstream tasks.

**Scaffold edit / hyperparameters.** Import `fla.layers.MultiScaleRetention` into
`CausalSelfAttention`. Match the softmax parameter budget at the scaffold's fixed `4·d` GELU FFN by
using the **unwidened** value: `expand_k = 1.0`, `expand_v = 1.0` (so `d_k = d_v = d`, state `d×d`).
`hidden_size = n_embd = 1024`, `num_heads = n_head = 16`, `use_output_gate = True`, `gate_fn = 'swish'`.
Set `self.use_pos_emb = False` (the decay + rotary phase already encode relative position, so the loop
skips `wpe`). The `Block` is left exactly as the scaffold default — only the mixer is swapped, keeping
the comparison fair. (Per-head normalization and the chunkwise kernel are internal to the FLA layer.)

```python
# EDITABLE region 1 of nanoGPT/custom_pretrain.py (lines 33-70) — RetNet (multi-scale retention)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from fla.layers import MultiScaleRetention
        self.attn = MultiScaleRetention(
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            expand_k=1.0,                 # d_k = d  (matched to softmax width)
            expand_v=1.0,                 # d_v = d  (state d x d; no value widening)
            use_output_gate=True,         # swish output gate = the restored nonlinearity
            gate_fn='swish',
        )
        self.use_pos_emb = False          # gamma^{n-m} decay + rotary phase encode position

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
