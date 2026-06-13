**Problem.** GeGLU established the gating *structure* but left the gate's activation as an unexamined
carry-over from the default MLP (it reused GELU because that was the incumbent activation). Its row sagged
on the long-range and commonsense metrics (LAMBADA ppl 68.73, hellaswag 32.90). The only editable slot is
the `MLP` class; the goal is to lower FineWeb `val_loss` further at the **same** matched budget by turning
the one knob GeGLU chose casually — the gate's activation — and nothing else.

**Key idea.** Keep the GLU structure and the 8/3 matched budget exactly; replace the gate activation
`GELU → SiLU` (Swish₁): `( SiLU(xW) ⊗ (xV) ) W2` — SwiGLU, the gated FFN form that PaLM, LLaMA, DeepSeek,
and Qwen converged on for this precise slot. The change is literally one function on the gate path.

**Why.** (1) *Controlled swap* — SwiGLU and GeGLU share the identical three-matrix layout, 8/3 width, and
bias-free `Linear`s; only the gate's `f` differs, so any `val_loss` delta is attributable to the
activation alone. (2) *Gradient highway intact* — the value path stays linear, so `∇[X⊗f(X)]`'s leading
term `∇X⊗f(X)` still scales the upstream gradient by the gate *value*, not by an activation derivative;
switching Φ-shaped → σ-shaped only reshapes which units the highway is open on. (3) *Softer moderate-regime
gate* — `SiLU(z)=z·σ(z)` is smoother and slightly more generous than `GELU(z)=z·Φ(z)` near and below the
origin (the two curves are close for large positive `z`), letting a touch more moderate-magnitude content
pass at near-unit gain — the kind of change that nudges the long-range (LAMBADA) and commonsense
(hellaswag) signals where GeGLU sagged.

**Matched budget.** Identical to GeGLU: three matrices, `d_ff' = (8/3)·d`. At `n_embd=1024`,
`int(8/3·1024)=2730` rounded up to a multiple of 64 → 2752; biases omitted.

**Hyperparameters.** Gate `w1`, value `w3` = `Linear(n_embd, hidden)`; `c_proj = Linear(hidden, n_embd)`;
`hidden = ((int(8/3·n_embd)+63)//64)·64`. Activation `F.silu` on the gate path only. `CONFIG_OVERRIDES`
empty — only the gate activation changed.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 2: SwiGLU MLP (8/3 width)
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 / 3 * config.n_embd)
        # Round to nearest multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: SiLU(xW1) * (xW3) then project back
        return self.dropout(self.c_proj(F.silu(self.w1(x)) * self.w3(x)))


# training-setup hook left at the default — only the gate activation changed:
CONFIG_OVERRIDES = {}
```
