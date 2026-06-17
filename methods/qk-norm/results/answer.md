# QKNorm (Query-Key Normalization), distilled

QKNorm bounds the per-head attention scores by replacing the scaled dot product with a cosine
similarity and then stretching it with a learnable scalar. Concretely, it `l2`-normalizes each
query and key along the head dimension (after the multi-head split) so each score is a cosine in
`[−1, 1]`, and multiplies the resulting score matrix by a learnable scalar `g` instead of dividing
by `sqrt(d_k)`. This keeps the input to the attention softmax in a controlled range — magnitudes can
no longer saturate it into winner-take-all — while `g` restores the head's ability to be sharp when
it should be.

## Problem it solves

The softmax in attention depends only on the *differences* between its input scores, and the
scaled dot product feeding it is unbounded. Query/key norm growth can turn a slight, possibly
accidental lead into a large absolute gap
(`softmax([760,752,750]) = softmax([12,4,2]) = [0.99962,0.00034,0.00005]`), pushing heads into
saturated, peaky, low-gradient distributions and limiting the diffuse attention patterns a head can
learn — most costly when data is scarce (low-resource translation). The
`1/sqrt(d_k)` factor only normalizes the score *variance at initialization* (under independent,
unit-variance `q`, `k`, `q·k` has variance `d_k`); it does not bound the scores as the projections
drift during training.

## Key idea

In polar form a score is `q · k = ||q|| ||k|| cos(angle)`. The magnitudes can drown out the angular
agreement the head should reason with — the same structural issue seen in the dot-product output
softmax of language models, where embedding norms dominate angles. So discard the magnitudes and
keep the cosine:

- **Bound the score.** `q_hat = q/||q||`, `k_hat = k/||k||`, normalized along the **head dimension**
  (after the split), for **queries and keys only** (values are averaged, not scored). Then each
  entry `q_hat · k_hat` is a cosine in `[−1, 1]`, for all of training — not just at init.
- **Restore expressivity.** Cosines squeezed into `[−1, 1]` prevent sharp attention on long rows:
  with row length `m` and `g=1`, even the best case `1` versus all `−1`s gives top probability
  `e^2/(e^2+m−1)`. Multiply the bounded scores by a **learnable scalar `g`** (one per attention
  module) before the softmax, so the range can be stretched back out for sharp distributions. This
  replaces the fixed `1/sqrt(d_k)`.

The attention operation changes from
`softmax(QK^T / sqrt(d_k)) V` to `softmax(g · Q_hat K_hat^T) V`,
with `Q_hat`, `K_hat` the `l2`-normalized (along head dim) queries and keys and `g` learnable.

The learnable `g` is the load-bearing complement to the bound: fixing `g = 1` leaves long-row
attention unable to put most of its mass on one token.

## Initialization of `g` and why

`g_0 = log2(L^2 − L)`, with `L` the 97.5th-percentile sequence length across the training data
(source and target). The exact row-length bound with scale `g` is
`p_max = 1/(1+(m−1)e^(−2g))`, so larger rows need larger scale to permit a concentrated softmax.
The paper's `L^2−L` is the off-diagonal count of a typical full `L × L` score matrix, used as a
coarse empirical proxy for similarity-matrix size rather than a derivation from row competitors.
`log2` compresses that count. For `L = 72...79`, `g_0` is `12.32...12.59`, stretching `[−1, 1]` to
about `[−12.5, 12.5]`. It is only an initialization; the model tunes `g` per attention module.

## Properties

- **Bounded scores throughout training**, not merely calibrated at init — a guarantee the fixed
  `1/sqrt(d_k)` cannot give.
- **Head-count ablation was stable in the paper**: on English→Vietnamese, the reported score stays
  in the same broad range from 2 to 32 heads, including head dimension 16. This is empirical, not a
  theorem; the mechanism-level reason to expect stability is that the score remains a per-head cosine
  and no longer relies on a `sqrt(d_k)` variance calibration.
- **Complements, not replaces, residual-stream normalization** (vanilla LayerNorm or ScaleNorm);
  it acts strictly inside attention on the scores. Pairs naturally with PreNorm (residual path
  kept an identity map) and FixNorm (unit-length word embeddings — the same magnitude-vs-direction
  principle at the input side).

## Working code

Filling the score slot of the per-head attention module: `l2`-normalize `q` and `k` along the head
dimension, score with a batched matmul, and stretch by the learnable `g`.

```python
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


class ScaleUp(nn.Module):
    """Learnable scalar g that scales the bounded cosine scores before softmax."""
    def __init__(self, scale):
        super().__init__()
        self.scale = Parameter(torch.tensor(scale))

    def forward(self, x):
        return x * self.scale


class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = args.att_dropout

        self.use_bias = args.use_bias
        self.mha_sup = args.mha_sup
        self.seq_len_threshold = args.seq_len_threshold

        # packed q/k/v/o projections; smaller-than-Xavier std for stability
        self.weights = Parameter(torch.Tensor(4 * self.embed_dim, self.embed_dim))
        if self.use_bias:
            self.biases = Parameter(torch.Tensor(4 * self.embed_dim))
        nn.init.normal_(self.weights, mean=0.0, std=(2 / (5 * self.embed_dim)) ** 0.5)
        if self.use_bias:
            nn.init.constant_(self.biases, 0.)

        if self.mha_sup:
            self.mha_scale = ScaleUp(np.log2(self.seq_len_threshold ** 2 - self.seq_len_threshold))
        else:
            self.scale = self.head_dim ** -0.5

    def _split_heads(self, x):
        bsz, length, _ = x.size()
        return (x.reshape(bsz, length, self.num_heads, self.head_dim)
                 .transpose(1, 2)
                 .reshape(bsz * self.num_heads, -1, self.head_dim))

    def forward(self, q, k, v, mask, do_proj_qkv=True):
        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        if self.mha_sup:
            # QKNorm: cosine score in [-1, 1], then learned scale-up.
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            att = self.mha_scale(torch.bmm(q, k.transpose(1, 2)))
        else:
            att = torch.bmm(q, k.transpose(1, 2)) * self.scale

        bsz_x_num_heads, src_len, tgt_len = att.size()
        bsz = bsz_x_num_heads // self.num_heads
        att = att.reshape(bsz, self.num_heads, src_len, tgt_len)
        if mask is not None:
            att.masked_fill_(mask, -1e9)
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        out = torch.bmm(att.reshape(bsz_x_num_heads, src_len, tgt_len), v)   # values left un-normalized
        out = (out.reshape(bsz, self.num_heads, src_len, self.head_dim)
                  .transpose(1, 2)
                  .reshape(bsz, src_len, self.embed_dim))
        return self.proj_o(out), att

    # proj_qkv / proj_o: existing packed linear projections
```

## Relation to prior methods

- **Scaled dot-product attention** (`softmax(QK^T/sqrt(d_k))V`): QKNorm replaces the unbounded dot
  product with a bounded cosine and the fixed `1/sqrt(d_k)` with a learnable `g` — turning an
  init-time variance calibration into a training-long bound plus a learned temperature.
- **ScaleNorm** (`g·x/||x||` along the embedding dim on the residual-stream vector before attention
  projections, replacing LayerNorm): QKNorm normalizes along the **head** dim **after** the split,
  on **Q and K only**, and **complements** LayerNorm instead of replacing it — targeting the
  per-head scores that the softmax actually sees. It reuses ScaleNorm's idea of a single learnable
  scale.
- **RMSNorm**: re-scaling-only normalization (`a/RMS(a)·g`); since the Euclidean norm and RMS
  differ only by a constant factor `sqrt(head_dim)` (absorbable into `g`), normalizing Q/K by RMS
  along the head dim is the same construction within the re-scaling family.
- **Cosine normalization** (Luo et al. 2018): bounds inner products to `[−1, 1]` in
  fully-connected layers; QKNorm applies the bounded-similarity idea inside per-head attention and
  adds the learnable scale-up that a softmax over a variable-length sequence requires.
