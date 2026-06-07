# Triangular multiplicative update and triangular self-attention on pair features

## Problem

The pair representation `z_ij` is an edge feature on the complete residue graph. It is read out
into distance and confidence heads, so updating it as `N^2` independent cells is geometrically weak:
the distance implied by edge `(i, j)` is constrained by the two other sides of every triangle
through a third residue `k`. A useful pair update should let `z_ij` receive information routed
through those third-residue triangles while remaining cheap enough for a deep trunk.

## Key idea

Use two complementary triangle-shaped pair updates.

- **Triangular multiplicative update:** combine two directed reads of the other triangle sides with
  a gated elementwise product and sum over `k`. Because `z_ij` is directed, use both directed
  contractions: outgoing `ikc,jkc->ijc` and incoming `kjc,kic->ijc`.
- **Triangular self-attention:** let edge `(i, j)` choose useful third residues by query-key
  attention, but add a per-head scalar bias from the missing third edge so the logit depends on all
  three sides, not just two.

These are learned geometric inductive biases, not a hard projection onto Euclidean distance
matrices.

## Algorithms

For multiplicative outgoing with hidden width `c = 128`:

1. `z_ij <- LayerNorm(z_ij)`
2. `a_ij = sigmoid(Linear(z_ij)) * Linear(z_ij)`, `b_ij = sigmoid(Linear(z_ij)) * Linear(z_ij)`
3. `g_ij = sigmoid(Linear(z_ij))`
4. `z_tilde_ij = g_ij * Linear(LayerNorm(sum_k a_ik * b_jk))`

Incoming is identical except step 4 uses `sum_k a_ki * b_kj`.

For starting-node attention with per-head width `c = 32` and `N_head = 4`:

1. `z_ij <- LayerNorm(z_ij)`
2. `q^h_ij, k^h_ij, v^h_ij = LinearNoBias(z_ij)`
3. `b^h_ij = LinearNoBias(z_ij)`, a scalar per head
4. `g^h_ij = sigmoid(Linear(z_ij))`
5. `a^h_ijk = softmax_k((1/sqrt(c)) q^h_ij dot k^h_ik + b^h_jk)`
6. `o^h_ij = g^h_ij * sum_k a^h_ijk v^h_ik`
7. `z_tilde_ij = Linear(concat_h o^h_ij)`

Ending-node attention uses `softmax_k((1/sqrt(c)) q^h_ij dot k^h_kj + b^h_ki)` and values
`v^h_kj`.

Within one pair block:

`outgoing multiplication -> incoming multiplication -> starting-node attention -> ending-node attention -> pair transition`.

The first three residual updates use rowwise dropout; the ending-node attention uses columnwise
dropout in the original orientation. The pair transition is a 4x-wide MLP.

## Code

```python
import torch
import torch.nn as nn
from functools import partialmethod

# Linear, LayerNorm, Attention, DropoutRowwise, DropoutColumnwise,
# PairTransition, permute_final_dims, and is_fp16_enabled are trunk primitives.


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
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))   # [*, c, i, k]
            b = permute_final_dims(b, (2, 1, 0))   # [*, c, k, j]
        else:
            a = permute_final_dims(a, (2, 1, 0))   # [*, c, k, i]
            b = permute_final_dims(b, (2, 0, 1))   # [*, c, k, j]
        p = torch.matmul(a, b)
        return permute_final_dims(p, (1, 2, 0))    # [*, i, j, c]

    def forward(self, z, mask=None):
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask * self.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = mask * self.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)

        if is_fp16_enabled():
            a_std = a.std()
            b_std = b.std()
            if a_std != 0.0 and b_std != 0.0:
                a = a / a_std
                b = b / b_std
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
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
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1)).unsqueeze(-4)
        x = self.mha(q_x=x, kv_x=x, biases=[mask_bias, triangle_bias])

        if not self.starting:
            x = x.transpose(-2, -3)
        return x


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)


class PairStack(nn.Module):
    def __init__(self, c_z, pair_dropout=0.25):
        super().__init__()
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, 128)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, 128)
        self.tri_att_start = TriangleAttentionStartingNode(c_z, 32, 4)
        self.tri_att_end = TriangleAttentionEndingNode(c_z, 32, 4)
        self.pair_transition = PairTransition(c_z, n=4)
        self.dropout_row = DropoutRowwise(pair_dropout)
        self.dropout_col = DropoutColumnwise(pair_dropout)

    def forward(self, z, pair_mask):
        z = z + self.dropout_row(self.tri_mul_out(z, mask=pair_mask))
        z = z + self.dropout_row(self.tri_mul_in(z, mask=pair_mask))
        z = z + self.dropout_row(self.tri_att_start(z, mask=pair_mask))
        z = z + self.dropout_col(self.tri_att_end(z, mask=pair_mask))
        z = z + self.pair_transition(z, mask=pair_mask)
        return z
```
