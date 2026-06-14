# Multi-head self-attention encoder block

The attention encoder replaces recurrence and convolution with **self-attention**: a layer in
which every position of a sequence attends to every other position by content, in one
parallel, constant-path-length operation. The encoder block stacks a multi-head
self-attention sub-layer and a position-wise feed-forward sub-layer, each wrapped in a
residual connection and layer normalization. It connects any two positions with `O(1)`
sequential operations and an `O(1)` maximum path length, and every operation is a dense
matrix multiply.

## Problem it solves

Map a variable-length sequence `(x_1, ..., x_n)`, `x_i in R^d`, to a new sequence
`(z_1, ..., z_n)` where each `z_i` may draw, by content, on all positions — with no
left-to-right dependency chain (so it parallelizes across positions) and a path length
between any two positions that does not grow with their distance (so long-range dependencies
are easy to learn). Recurrence gives `O(n)` sequential operations and `O(|i-j|)` path length;
convolution gives `O(1)` sequential depth but a path length that still grows (`O(n/k)` or
`O(log_k n)`). Self-attention gives `O(1)` for both.

## Key idea

For each position form a **query**, and for every position a **key** and a **value**, via
learned linear maps `Q = X W^Q`, `K = X W^K`, `V = X W^V`. The output at a position is a
content-weighted average of the value vectors, with weights from matching its query against
all keys:

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

In **self-attention**, `Q`, `K`, `V` are all projections of the same input, so each position
attends to every position of its own sequence.

- **Scaling by `1/sqrt(d_k)`.** If query/key components are independent with mean 0 and
  variance 1, the dot product `q.k = sum_{c=1}^{d_k} q_c k_c` has mean 0 and
  `Var(q.k) = sum_c Var(q_c k_c) = sum_c E[q_c^2]E[k_c^2] = d_k`, so the logits have standard
  deviation `sqrt(d_k)`. Large logits push the softmax toward one-hot; at a one-hot vector
  the softmax Jacobian `diag(p) - p p^T` is zero. Dividing by `sqrt(d_k)` restores
  unit-variance logits independent of width — the exact scale that cancels the width
  dependence, not a tuned constant.

- **Multi-head.** One softmax expresses a single relation, and its averaging blurs several.
  Run `h` attention functions in parallel in `d_k = d_v = d_model/h`-dimensional subspaces,
  concatenate, and re-project:

  ```
  head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
  ```

  With per-head width `d_model/h`, the cost stays at the full-width scale. For each of the
  query, key, and value projection families,
  `h * d_model * (d_model/h) = d_model^2`; the output projection is another `d_model^2`.
  For the score matrices, `h * n^2 * (d_model/h) = n^2 * d_model`. So this keeps the
  leading-order score cost equal to a single full-width head while giving separate learned
  subspaces; it also restores the resolution a single averaged head would lose. `W^Q, W^K`
  decide who attends to whom;
  `W^V, W^O` decide what is carried and how it is repackaged.

- **Position-wise feed-forward.** Attention is linear in the values (only the weights are
  nonlinear), so add a per-position network with a ReLU, inner width `d_ff = 4 * d_model`:
  `FFN(x) = max(0, x W_1 + b_1) W_2 + b_2`. It gives each position a real nonlinear transform
  of what it gathered.

- **Residual + layer normalization (post-norm).** Each sub-layer output is
  `LayerNorm(x + Sublayer(x))`. The residual identity path keeps a deep stack trainable
  (the sub-layer learns a correction). Layer normalization normalizes across the feature
  dimension of each position — `mu = (1/H) sum_c a_c`, `sigma = sqrt((1/H) sum_c (a_c-mu)^2)`,
  `(g/sigma)(a-mu)+b` — so it is batch-size independent and works at batch size 1 and for
  variable-length inputs (where batch normalization cannot). Normalizing *after* the residual
  hands every downstream block a fixed-scale input regardless of depth.

## Encoder block and defaults

One block has two sub-layers; an encoder is a stack of `N` identical blocks:

```
a = LayerNorm( x + MultiHeadSelfAttention(x) )   # gather across positions, then normalize
z = LayerNorm( a + FFN(a) )                       # transform per position, then normalize
```

Standard defaults: `N = 6` blocks, `d_model = 512`, `h = 8` heads (`d_k = d_v = 64`),
`d_ff = 2048`, dropout 0.1.

## Complexity Summary

| Layer type | Mixing complexity per layer | Sequential ops | Max path length |
|---|---|---|---|
| Self-attention | `O(n^2 d)` | `O(1)` | `O(1)` |
| Recurrent | `O(n d^2)` | `O(n)` | `O(n)` |
| Convolutional | `O(k n d^2)` | `O(1)` | `O(n/k)` contiguous, `O(log_k n)` dilated |

Self-attention wins on path length and parallelism, and is cheaper per layer than recurrence
when comparing the mixing terms in the usual `n < d` regime for sub-word / short-sequence
representations. The projections add `O(n d^2)` work, and the position-wise feed-forward
part adds `O(n d d_ff)` work, just as the block code below makes explicit.

## Working code (canonical encoder block)

```python
import math
import torch
import torch.nn as nn


def attention(query, key, value, mask=None, dropout=None):
    "Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) V."
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # custom mask: 1 keeps, 0 blocks
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        sigma = torch.sqrt(var + self.eps)
        return self.a_2 * (x - mean) / sigma + self.b_2


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class EncoderLayer(nn.Module):
    "Post-norm block: LayerNorm(x + self-attn(x)) then LayerNorm(x + FFN(x))."
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )

    def forward(self, x, mask=None):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, mask)
        return h
```

## PyTorch module equivalent

`nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)` accepts tensors shaped
`(batch, seq, feature)`, uses packed input projection weights for `Q`, `K`, and `V` when the
dimensions match, splits `embed_dim` across `num_heads`, applies scaled dot-product attention,
concatenates the heads, and applies `out_proj`. Its `dropout` argument drops attention
weights, matching the attention-dropout placement above. With `need_weights=False`, it can
use the optimized scaled-dot-product path while preserving the same sub-layer semantics.
Boolean `key_padding_mask` entries are `True` at padded key positions to ignore; boolean
`attn_mask` entries likewise mark disallowed query-key pairs.

```python
import torch
import torch.nn as nn


class TorchMHAEncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=h,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x
```
