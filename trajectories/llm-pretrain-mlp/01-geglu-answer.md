**Problem.** The MLP sublayer holds most of a GPT-2 Medium's parameters and compute, yet forms every
hidden unit from a *single* learned linear view of the token bent by one fixed pointwise function (the
default GELU). The only editable slot is the `MLP` class; the goal is to lower FineWeb `val_loss` at
**matched** parameters and FLOPs, with a drop-in change confined to the FFN — same `(B, T, n_embd)`
contract, nothing touched in attention, normalization, data, optimizer, or evaluation.

**Key idea.** Every standard activation is already a "value × gate": `ReLU(z)=z·1[z>0]`, `GELU(z)=z·Φ(z)`,
`Swish(z)=z·σ(z)` — but value and gate are tied to the same projection `xW1`. *Untie them.* Use two
up-projections, a gate `W` and a value `V`, and form the hidden as their elementwise product, GELU on the
gate, value carried linearly: `( GELU(xW) ⊗ (xV) ) W2` — GeGLU, the GLU variant whose gate reuses the
activation this MLP already runs.

**Why.** (1) *Multiplicative interaction* — each hidden unit multiplies two independent linear views of
`x`, expressing degree-2 couplings a single-projection FFN cannot. (2) *Clean gradient path* — the value
path is linear, so for `X⊗σ(X)`, `∇ = ∇X⊗σ(X) + X⊗σ'(X)∇X`: the first term scales the upstream gradient by
the gate *value*, not by an activation derivative (contrast the both-paths-nonlinear `tanh(X)⊗σ(X)`, every
term shrunk by `tanh'≤1` or `σ'≤¼`). So the nonlinearity goes on the gate, the value stays linear. (3)
*GELU gate* — consistent with the MLP's existing activation, and richer than a sigmoid: `GELU(z)=zΦ(z)` can
pass content at greater-than-unit gain and softly sign-flip, where σ only attenuates.

**Matched budget (the 2/3 / 8/3 rule).** Three matrices replace two, so the hidden width shrinks to keep
the budget fixed: `3·d·d_ff' = 2·d·(4d) ⇒ d_ff' = (8/3)·d`. FLOPs scale identically. At `n_embd=1024`
this is `int(8/3·1024)=2730`, rounded up to a multiple of 64 (2752) for matmul-friendly shapes; biases
omitted (`config.bias=False`).

**Hyperparameters.** Gate `w1` and value `w3` are `Linear(n_embd, hidden)`; down-projection `c_proj` is
`Linear(hidden, n_embd)`; `hidden = ((int(8/3·n_embd)+63)//64)·64`. Activation `F.gelu` on the gate path
only. `CONFIG_OVERRIDES` left empty — only the MLP form changes.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 1: GeGLU MLP (8/3 width)
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 / 3 * config.n_embd)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # GeGLU: GELU(xW1) * (xW3) then project back
        return self.dropout(self.c_proj(F.gelu(self.w1(x)) * self.w3(x)))


# training-setup hook left at the default — only the MLP form changed:
CONFIG_OVERRIDES = {}
```
