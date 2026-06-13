**Problem.** GeGLU→SwiGLU improved `val_loss` only 0.0029 — two near-identical smooth gate curves in the
identical GLU structure must land in the same band, so the *gate axis* is exhausted. The only editable
slot is the `MLP` class; the goal is to lower FineWeb `val_loss` further by questioning what the whole GLU
family shares and never varied: the *shape of the nonlinearity itself*, inside the plain two-matrix FFN,
at the same matched budget.

**Key idea.** Leave the GLU family. Go back to the default two-matrix 4d FFN and replace the smooth
sigmoid-derived activation with a rectified *quadratic*: `(ReLU(z))² = max(0,z)²` (Primer-EZ's squared
ReLU). It keeps ReLU's hard sparsity floor (zero for `z≤0`) but grows *faster than linearly* for `z>0`,
sharpening the per-unit response curve — a different *kind* of nonlinearity, not another gate.

**Why.** (1) *Selectivity / effective sparsity* — squaring amplifies the contrast between strongly- and
weakly-firing units (a `z=2` unit gives 4, a `z=0.5` unit gives 0.25, a 16× ratio vs ReLU's 4×), so the
down-projection sees a hidden vector dominated by the few units that genuinely matched the token — soft,
learned feature selection baked into the activation. (2) *Zero structural cost* — unlike the GLU variants
it adds no third matrix and no width re-sizing: it keeps the default's exact two matrices and the **full
4d width** (recovering the 33% of hidden width the 8/3 GLU variants gave up), with the square a cheap fused
elementwise op. Matched-budget to the default *and* to GeGLU/SwiGLU. (3) *Stable without a schedule change*
— the quadratic is buffered against blow-up by small init preactivations (`c_fc` ~ `N(0,0.02)` on a
LayerNormed input), the residual-scaled `c_proj` init (`N(0, 0.02/√(2·n_layer))`), and the frozen
`grad_clip=1.0`, so `CONFIG_OVERRIDES` stays empty and any `val_loss` change is the activation shape alone.

**Hyperparameters.** `c_fc = Linear(n_embd, 4·n_embd)`, `c_proj = Linear(4·n_embd, n_embd)`, both bias-free
(`config.bias=False`); activation `F.relu(x).square()`; dropout on the output; full 4d width.
`CONFIG_OVERRIDES` empty.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 3: ReLU² (squared-ReLU) MLP, 4x expansion
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU²
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# training-setup hook left at the default — only the activation shape changed:
CONFIG_OVERRIDES = {}
```
