# Triangular multiplicative update and triangular self-attention on the pair representation

## Problem

The pair representation `z_ij ∈ ℝ^{c_z}` (`c_z = 128`) is an edge feature on the complete graph
of residues: it encodes the relation between residues `i` and `j` and is read out into a
distogram. Updating each edge independently — or with axis-local row/column (axial) attention —
cannot enforce that the implied pairwise distances are realizable in 3D. The binding constraint
is geometric: every triple `(i, j, k)` must satisfy the triangle inequality
`d_ij ≤ d_ik + d_kj`, so edge `(i, j)` is constrained by the *other two edges* `(i, k)` and
`(k, j)` of every triangle it belongs to. The operation must let each edge feel those
constraints, cheaply, inside a deep trunk.

## Key idea

Update edge `(i, j)` by routing a message through every third node `k`, combining the two
incident edges of the triangle `{i, j, k}` and aggregating over `k`. Two complementary families:

- **Triangular multiplicative update** — the cheap, symmetric workhorse. Combine the two edges
  with a gated elementwise product and sum over `k` (an `einsum`/matmul). Because the pair map is
  directed (`z_ij ≠ z_ji`), there are two non-equivalent versions: **outgoing**, where `i` and `j`
  share the *target* `k`, and **incoming**, where they share the *source* `k`. Apply both.
- **Triangular self-attention** — content-routed refinement. Edge `(i, j)` attends over the third
  node, choosing which `k` matter by query–key similarity, with an additive bias from the *third*
  edge so the decision is made over the whole triangle, not two of its sides. **Starting-node** and
  **ending-node** versions cover both orientations.

## Final algorithms

**Multiplicative update (outgoing, `c = 128`):**

1. `z_ij ← LayerNorm(z_ij)`
2. `a_ij = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij)`, `b_ij = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij)`, with `a_ij, b_ij ∈ ℝ^c`
3. `g_ij = sigmoid(Linear(z_ij)) ∈ ℝ^{c_z}`
4. `z̃_ij = g_ij ⊙ Linear( LayerNorm( Σ_k a_ik ⊙ b_jk ) )`

**Incoming** is identical except step 4 uses `Σ_k a_ki ⊙ b_kj`.
Outgoing einsum: `ikc,jkc->ijc`; incoming einsum: `kjc,kic->ijc`.

**Triangular self-attention (starting node, `c = 32`, `N_head = 4`):**

1. `z_ij ← LayerNorm(z_ij)`
2. `q^h_ij, k^h_ij, v^h_ij = LinearNoBias(z_ij)`
3. `b^h_ij = LinearNoBias(z_ij)` (scalar per head)
4. `g^h_ij = sigmoid(Linear(z_ij))`
5. `a^h_ijk = softmax_k( (1/√c) q^h_ij·k^h_ik + b^h_jk )`
6. `o^h_ij = g^h_ij ⊙ Σ_k a^h_ijk v^h_ik`
7. `z̃_ij = Linear(concat_h o^h_ij)`

**Ending node** replaces the affinity with `(1/√c) q^h_ij·k^h_kj + b^h_ki` and values with `v^h_kj`.

In the pair stack, each is a residual sub-layer, in order: outgoing mult → incoming mult →
starting-node attention → ending-node attention → pair transition (a 4×-wide MLP).

## Why each piece

- **Route through a third node `k`** — independent / axial updates never combine `(i, k)` and
  `(k, j)` together, so they cannot express the triangle inequality. Routing the message through
  `k` makes both constraining edges present.
- **Multiplicative `Σ_k a ⊙ b`** — the cheapest symmetric way to combine two edge vectors and
  aggregate over the apex; one matmul, `O(N³ c)`. Gets the wide hidden width (`c = 128`) as the
  per-block workhorse.
- **Outgoing and incoming** — `z` is directed; the two contractions read different entries (rows
  vs columns) and cover both triangle orientations relative to the directed edge.
- **Gates on `a`, `b`** — a differentiable soft mask over which edges `k` should carry signal into
  the product.
- **Center LayerNorm after the sum** — the sum over `N` products has variance growing with `N`;
  renormalizing removes the sequence-length dependence (input LayerNorm separately conditions the
  projections).
- **Output gate `g_ij`** — lets a confident edge limit how much triangle-routed update it absorbs
  (residual-friendly).
- **Triangular attention on top** — the multiplicative update is content-blind across `k`;
  attention lets `(i, j)` concentrate on the decisive `k` by `q·k` similarity. Narrower
  (`c = 32`, 4 heads) since it is the more expensive op.
- **Third-edge bias `b^h_jk`** — vanilla row attention sees only sides `ij` and `ik`; adding the
  `jk` logit injects the missing side so the affinity depends on the full triangle.

## Code

```python
import torch
import torch.nn as nn
from functools import partialmethod

# Linear (with init modes), LayerNorm, Attention (gated MHA with additive biases
# and 1/sqrt(c) scaling), and permute_final_dims are trunk primitives.


class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_z, c_hidden, _outgoing=True):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.layer_norm_in = LayerNorm(c_z)
        self.layer_norm_out = LayerNorm(c_hidden)

        self.linear_a_p = Linear(c_z, c_hidden)
        self.linear_a_g = Linear(c_z, c_hidden, init="gating")
        self.linear_b_p = Linear(c_z, c_hidden)
        self.linear_b_g = Linear(c_z, c_hidden, init="gating")

        self.linear_z = Linear(c_hidden, c_z, init="final")
        self.linear_g = Linear(c_z, c_z, init="gating")
        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self, a, b):
        # outgoing: 'ikc,jkc->ijc' ; incoming: 'kjc,kic->ijc'
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))   # [*, c, i, k]
            b = permute_final_dims(b, (2, 1, 0))   # [*, c, k, j]
        else:
            a = permute_final_dims(a, (2, 1, 0))   # [*, c, k, i]
            b = permute_final_dims(b, (2, 0, 1))   # [*, c, k, j]
        p = torch.matmul(a, b)                     # sum over k
        return permute_final_dims(p, (1, 2, 0))    # [*, i, j, c]

    def forward(self, z, mask=None):
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask * self.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = mask * self.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)
        x = self._combine_projections(a, b)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        return x * g


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)

class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        super().__init__()
        self.starting = starting
        self.inf = inf
        self.layer_norm = LayerNorm(c_in)
        self.linear = Linear(c_in, no_heads, bias=False, init="normal")
        self.mha = Attention(c_in, c_in, c_in, c_hidden, no_heads)

    def forward(self, x, mask=None):
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if not self.starting:                       # ending node = work on columns
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1)).unsqueeze(-4)
        biases = [mask_bias, triangle_bias]         # logit = q·k/sqrt(c) + b_jk + mask

        x = self.mha(q_x=x, kv_x=x, biases=biases)
        if not self.starting:
            x = x.transpose(-2, -3)
        return x


class TriangleAttentionStartingNode(TriangleAttention):
    __init__ = partialmethod(TriangleAttention.__init__, starting=True)

class TriangleAttentionEndingNode(TriangleAttention):
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
```

Pair stack within one trunk block (each a residual sub-layer):

```python
z = z + TriangleMultiplicationOutgoing(c_z, 128)(z)
z = z + TriangleMultiplicationIncoming(c_z, 128)(z)
z = z + TriangleAttentionStartingNode(c_z, 32, 4)(z)
z = z + TriangleAttentionEndingNode(c_z, 32, 4)(z)
z = z + PairTransition(c_z, n=4)(z)
```
