# Looped Transformers as Programmable Computers

## Problem

A transformer of `L` layers performs a fixed `L` stages of computation per forward pass, so
running an iterative algorithm for `T` steps the naive way needs depth growing with `T`. The
goal is a **fixed-depth** transformer that, **placed in a loop**, executes an arbitrary
iterative program — a calculator, numerical linear algebra, an in-context learning algorithm —
where only the *number of loop iterations* scales with the computation, not the number of
layers, and the weights are given explicitly.

## Key idea

Treat the fixed-depth transformer as a CPU and the loop as its clock. Run `X ← TF(W; X)` for
`T` iterations, feeding the output back as input. The network's depth only needs to suffice for
**one instruction**; program length lives in `T`. Make this work by:

1. **Punchcard input layout.** Columns are `d`-dim embedding vectors partitioned into
   Scratchpad `S` | Memory `M` | Commands `C`. Each column carries a `±1` binary positional code
   `p_i` (length `log n`) of its index, used as both program counter and data pointers:
   `p_i^T p_i = log n`, and `p_i^T p_j ≤ log n − 2` for `i ≠ j` (codes differing in `k ≥ 1`
   coordinates give `log n − 2k`).
2. **Read/write via attention as an approximate permutation.** Set key = query projecting onto
   the position rows, so the score matrix is the Gram matrix `p_i^T p_j`; a softmax temperature
   `λ ≥ log(n^3/ε)` sharpens it to within `ε` of a permutation that selects the addressed
   column. The value matrix routes the selected contents to the scratchpad; a large-constant
   ReLU gate keyed on a scratchpad-indicator bit `b` performs the overwrite. `write` is the same
   with the gate sign flipped. Each is 1 layer, width `O(log n + d)`.
3. **Arithmetic and control in the ReLU MLP.** Binary pointer increment (1-hidden-layer ReLU,
   `8d` units); two's-complement subtraction (bit-flip `2·ReLU(−b)−1`, add 1, add); a `≤ 0`
   flag from the sign bit and an all-`−1` test; a conditional jump selecting between the
   incremented PC and the jump target; a piecewise-linear error-correction layer that snaps
   `±ε`-noisy values back to exact `{−1, 0, 1}`.

These assemble into one **`SUBLEQ(a,b,c)`** instruction in a **9-layer, 2-head** looped
transformer (`SUBLEQ` is a one-instruction-set computer, hence Turing complete). Generalizing
the single operation to a menu of `M` hardcoded **function blocks** gives **`FLEQ`**:
`mem[c] = f_m(mem[a], mem[b]); if mem[flag] ≤ 0 goto p`, executed by a `9 + max_m l_m`-layer
transformer with `Σ_m h_m` heads and width `O(Md + log n)`. Setting `flag` to a constant-1 cell
gives a pure function call; setting `a,b,c` to dummy cells gives a pure branch.

**Nonlinear functions come from the softmax itself.** Putting slopes `a_{ji}` in the key,
coefficients `c_{ji}` in the value, identity in the query, and padding each `a_{ji}` with a bias
`−log(3d−1)` (and `x` with a trailing `1`) makes the per-column softmax denominator
`(3d−1) + e^{a^T x}`, so each head outputs `c_{ji} σ(a_{ji}^T x)`; summing `m` heads gives
`Σ_i c_{ji} σ(a_{ji}^T x)`. By Barron (1993) this approximates any `f ∈ Γ_{C,B}, f(0)=0` with
error `O(m^{-1/2})` for `τ ≥ m^{1/2} ln m` — a universal approximator at constant depth (vs a
ReLU MLP needing depth `O(log 1/ε)` for `x^2`). The same large-constant padding read the other
way *linearizes* the softmax, giving matrix multiplication `A^T B + εM` (2 layers). Transpose,
matrix inversion (Newton) and power iteration follow, each in a 13-layer looped transformer.

**SGD with backprop is a `FLEQ` program.** Linear SGD (`w ← w − η Σ_i (w^T x_i − y_i) x_i`) and
two-layer-net backprop (`z=W_1x+b_1`, `a=σ(z)`, `o=W_2a+b_2`; `δ_2=o−y`,
`δ_1=σ'(z)⊙W_2δ_2` with `σ'(z)=σ(z)(1−σ(z))`; outer-product gradients; subtract `η·`gradient)
each run in a 13-layer, 1-head looped transformer of width `O(log D + d)`; the number of loops
scales with `T·D`. All constructed transformers have depth `≤ 13`.

## Final form (the looped construction)

```python
import torch
import torch.nn as nn


class TFLayer(nn.Module):
    """Standard transformer layer with a tunable softmax temperature `lam` and a position-wise
    ReLU MLP. All weights are SET (hardcoded), not trained."""

    def __init__(self, d_model, n_heads, lam):
        super().__init__()
        self.lam = lam
        self.K = nn.ParameterList(nn.Parameter(torch.zeros(d_model, d_model)) for _ in range(n_heads))
        self.Q = nn.ParameterList(nn.Parameter(torch.zeros(d_model, d_model)) for _ in range(n_heads))
        self.V = nn.ParameterList(nn.Parameter(torch.zeros(d_model, d_model)) for _ in range(n_heads))
        self.W1 = nn.Parameter(torch.zeros(4 * d_model, d_model))
        self.b1 = nn.Parameter(torch.zeros(4 * d_model))
        self.W2 = nn.Parameter(torch.zeros(d_model, 4 * d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, X):                                   # X: [d_model, n], columns = tokens
        out = X.clone()
        for K, Q, V in zip(self.K, self.Q, self.V):
            scores = (K @ X).transpose(0, 1) @ (Q @ X)      # p_i^T p_j (read) or a^T x (sigmoid)
            P = torch.softmax(self.lam * scores, dim=0)     # high lam: approx permutation / sigmoid
            out = out + V @ X @ P
        ones = torch.ones(1, X.shape[1])
        return out + self.W2 @ torch.relu(self.W1 @ out + self.b1[:, None] @ ones) + self.b2[:, None] @ ones


def pos_encoding(i, log_n):
    """+/-1 binary code of column index i: self-product log_n, off-diagonal <= log_n - 1."""
    return torch.tensor([1.0 if (i >> k) & 1 else -1.0 for k in range(log_n)])


def compute_flag_bits(bits, N):
    """0/1 indicator of (two's-complement integer) <= 0 from N +/-1 bits b_N..b_1 (b_N = MSB)."""
    b_N = bits[-1]
    return torch.relu(b_N) + torch.relu(1.0 - N - bits.sum())   # negative OR all-bits-(-1) == 0


def next_pc(p_inc, p_jump, flag):
    """Conditional jump: flag=1 -> p_jump, flag=0 -> p_inc (both +/-1 codes)."""
    return 2.0 * torch.relu(p_inc - flag) + 2.0 * torch.relu(p_jump - (1.0 - flag)) - 1.0


def fix_bits(b, eps):
    """Snap softmax (not hardmax) residue back to exact {-1, 0, 1}; |b - exact| < eps < 0.5."""
    s = 1.0 / (1.0 - 2.0 * eps)
    return (s * (torch.relu(b + 1 - eps) - torch.relu(b + eps))
            + s * (torch.relu(b - eps) - torch.relu(b - 1 + eps)) - 1.0)


def build_fleq_computer(d_model, lam, function_block_specs):
    """9-layer SUBLEQ skeleton (read instr; read mem[a],mem[b] with 2 heads; subtract; write
    back; flag; branch; error-correct) + the layers of the deepest function block f_m.
    Depth is constant in program length; heads are summed across blocks. Weights set per lemmas."""
    skeleton = [TFLayer(d_model, n_heads=2, lam=lam) for _ in range(9)]
    if function_block_specs:
        depth = max(l for (l, h) in function_block_specs)
        heads = sum(h for (l, h) in function_block_specs)
        body = [TFLayer(d_model, n_heads=heads, lam=lam) for _ in range(depth)]
    else:
        body = []
    return skeleton + body


def run(layers, X, T):
    """Loop = clock: T iterations execute T program instructions on a fixed-depth network."""
    for _ in range(T):
        for layer in layers:
            X = layer(X)
    return X
```

## Why the choices

- **Loop, not a deep stack:** depth would otherwise scale with the number of program lines; a
  CPU reuses one ALU across cycles. The loop is the only mechanism that makes program length
  free of depth.
- **`±1` binary positional codes:** width `log n` (not `n` as one-hot), a strict margin of `≥2`
  between the diagonal and off-diagonal Gram entries that the temperature can exploit, appended
  as a suffix for clean block-matrix arguments.
- **Key = query for reads:** makes the attention score depend only on whether two columns share
  an address, peaking on the matching column.
- **Temperature `λ ≥ log(n^3/ε)`:** turns softmax into a near-permutation; hardmax would be
  exact but non-differentiable, so softmax is used and the residual error is cleaned later.
- **Two's complement:** subtraction = negate (flip + add 1) + add, and the sign test is a single
  MSB read, so branching is cheap.
- **Sigmoid from softmax (bias `−log(3d−1)`):** reuses the existing nonlinearity instead of
  adding MLP depth; Barron then makes `m` heads a universal approximator at constant depth.
- **One function block per `f_m`:** simpler bookkeeping; `M` is small in practice so width `Md`
  is acceptable. Restricting global attention to scratchpad columns drops cost `O(n^2 d) → O(nd)`.
