# QKNorm (Query-Key Normalization), distilled

QKNorm bounds the per-head attention scores by replacing the scaled dot product with a cosine
similarity and then stretching it with a learnable scalar. Concretely, it `l2`-normalizes each
query and key along the head dimension (after the multi-head split) so each score is a cosine in
`[−1, 1]`, and multiplies the resulting score matrix by a learnable scalar `g` instead of dividing
by `sqrt(d)`. This keeps the input to the attention softmax in a controlled range — magnitudes can
no longer saturate it into winner-take-all — while `g` restores the head's ability to be sharp when
it should be.

## Problem it solves

The softmax in attention depends only on the *differences* between its input scores, and the
scaled dot product feeding it is unbounded. Large scores let even a slight, possibly accidental,
lead dominate a row (`softmax([760,752,750]) = softmax([12,4,2]) = [0.99962,0.00034,0.00005]`),
pushing heads into saturated, peaky, low-gradient distributions and limiting the diffuse attention
patterns a head can learn — most costly when data is scarce (low-resource translation). The
`1/sqrt(d)` factor only normalizes the score *variance at initialization* (under independent,
unit-variance `q`, `k`, `q·k` has variance `d`); it does not bound the scores as the projections
drift during training.

## Key idea

In polar form a score is `q · k = ||q|| ||k|| cos(angle)`. The magnitudes can drown out the angular
agreement the head should reason with — the same structural issue seen in the dot-product output
softmax of language models, where embedding norms dominate angles. So discard the magnitudes and
keep the cosine:

- **Bound the score.** `q_hat = q/||q||`, `k_hat = k/||k||`, normalized along the **head dimension**
  (after the split), for **queries and keys only** (values are averaged, not scored). Then each
  entry `q_hat · k_hat` is a cosine in `[−1, 1]`, for all of training — not just at init.
- **Restore expressivity.** Cosines squeezed into `[−1, 1]` make softmax nearly uniform (a head
  can no longer concentrate). Multiply the bounded scores by a **learnable scalar `g`** (one per
  attention layer) before the softmax, so the range can be stretched back out for sharp
  distributions. This replaces the fixed `1/sqrt(d)`.

The attention operation changes from
`softmax(QK^T / sqrt(d)) V` to `softmax(g · Q_hat K_hat^T) V`,
with `Q_hat`, `K_hat` the `l2`-normalized (along head dim) queries and keys and `g` learnable.

The learnable `g` is the load-bearing complement to the bound: fixing `g = 1` (scores stuck in
`[−1, 1]`) makes attention essentially non-functional rather than merely worse.

## Initialization of `g` and why

`g_0 = log2(L^2 − L)`, with `L` the 97.5th-percentile sequence length across the training data
(source and target). `g` must make it *possible* for the max entry of a softmax row to dominate;
how hard that is grows with the number of competitors, and `L^2 − L` is the off-diagonal count of
an `L × L` score matrix (more competitors with longer sequences). `log2` compresses that count so
the init stretches `[−1, 1]` to roughly `[−12.5, 12.5]` for a typical `L ≈ 72-79` — wide enough to
concentrate, narrow enough to stay diffuse. Like `sqrt(d)` in scaled dot product it is a rule of
thumb (found empirically; the 97.5th percentile beats lower/higher percentiles), but here it only
initializes a *learnable* parameter, which the model then tunes per layer.

## Properties

- **Bounded scores throughout training**, not merely calibrated at init — a guarantee the fixed
  `1/sqrt(d)` cannot give.
- **Robust to the number of heads** (stable from a few heads up to many small heads, e.g. head
  dim 16): the per-head cosine never relied on `d` being large.
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

        # packed q/k/v/o projections; smaller-than-Xavier std for stability
        self.weights = Parameter(torch.Tensor(4 * self.embed_dim, self.embed_dim))
        nn.init.normal_(self.weights, mean=0.0, std=(2 / (5 * self.embed_dim)) ** 0.5)

        # g_0 = log2(L^2 - L): L^2 - L is the off-diagonal count of an L x L score matrix;
        # log2 compresses it so the max entry can still softmax toward 1.
        L = args.seq_len_threshold                      # 97.5th-percentile sequence length
        self.mha_scale = ScaleUp(np.log2(L ** 2 - L))

    def _split_heads(self, x):
        bsz, length, _ = x.size()
        return (x.reshape(bsz, length, self.num_heads, self.head_dim)
                 .transpose(1, 2)
                 .reshape(bsz * self.num_heads, -1, self.head_dim))

    def forward(self, q, k, v, mask):
        q, k, v = self.proj_qkv(q, k, v)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        # QKNorm: cosine score in [-1, 1] (normalize q, k along head dim; q and k only),
        # then stretch by learnable g.
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        att = self.mha_scale(torch.bmm(q, k.transpose(1, 2)))   # g * q_hat k_hat^T

        bh, src_len, tgt_len = att.size()
        att = att.reshape(-1, self.num_heads, src_len, tgt_len)
        if mask is not None:
            att.masked_fill_(mask, -1e9)
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        out = torch.bmm(att.reshape(bh, src_len, tgt_len), v)   # values left un-normalized
        out = (out.reshape(-1, self.num_heads, src_len, self.head_dim)
                  .transpose(1, 2)
                  .reshape(-1, src_len, self.embed_dim))
        return self.proj_o(out), att

    # proj_qkv / proj_o: existing packed linear projections
```

## Relation to prior methods

- **Scaled dot-product attention** (`softmax(QK^T/sqrt(d))V`): QKNorm replaces the unbounded dot
  product with a bounded cosine and the fixed `1/sqrt(d)` with a learnable `g` — turning an
  init-time variance calibration into a training-long bound plus a learned temperature.
- **ScaleNorm** (`g·x/||x||` along the embedding dim, on Q, K, V, before the head split, replacing
  LayerNorm): QKNorm normalizes along the **head** dim **after** the split, on **Q and K only**,
  and **complements** LayerNorm instead of replacing it — targeting the per-head scores that the
  softmax actually sees. It reuses ScaleNorm's idea of a single learnable scale.
- **RMSNorm**: re-scaling-only normalization (`a/RMS(a)·g`); since the Euclidean norm and RMS
  differ only by a constant factor `sqrt(head_dim)` (absorbable into `g`), normalizing Q/K by RMS
  along the head dim is the same construction within the re-scaling family.
- **Cosine normalization** (Luo et al. 2018): bounds inner products to `[−1, 1]` in
  fully-connected layers; QKNorm applies the bounded-similarity idea inside per-head attention and
  adds the learnable scale-up that a softmax over a variable-length sequence requires.
