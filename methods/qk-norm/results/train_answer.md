When I train attention on small translation sets I keep watching heads collapse. The attention heatmaps show a head dumping almost all of its weight on a single token, row after row — a near one-hot. Sometimes that is correct; a head that copies, or that tracks a single antecedent, *should* be peaky. But it happens even where a diffuse, smeared-out distribution would clearly help, and once a head is that sharp it has stopped being able to express anything else. The suspicion is that something in the mechanism is *forcing* heads toward winner-take-all rather than letting the data decide. The mechanism is the softmax itself. For one query the attention weights are a softmax over a row of compatibility scores, and the softmax depends only on the *differences* between its entries — add a constant to every entry and nothing changes. Now stare at a concrete row: $\mathrm{softmax}([760, 752, 750]) = [0.99962, 0.00034, 0.00005]$, essentially one-hot, and $\mathrm{softmax}([12, 4, 2])$ is the *same* distribution because the gaps (8 and 2) are identical. So a gap of 8 between the top score and the next all-but-silences everything else, and whether it rides on a baseline of 6 or of 750 is irrelevant. The danger is that when scores live at large magnitudes — hundreds — an 8-point gap is barely over one percent in relative terms yet produces a near-deterministic choice. The scaled dot product feeding the softmax is unbounded, so query/key norm growth can turn a slight, even accidental lead into a large absolute gap and saturate the head into a low-gradient, peaky regime. This is most costly exactly where data is scarce, in low-resource translation, where a head that can only concentrate is a head whose capacity is wasted.

The obvious objection is that the $1/\sqrt{d_k}$ factor is supposed to handle this, so let me re-derive precisely what it does. Model the components of $q$ and $k$ as independent, mean zero, variance one. Then $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$, each product $q_i k_i$ has mean $0$ and variance $1$, so the sum has mean $0$ and variance $d_k$; the typical magnitude grows like $\sqrt{d_k}$, and dividing by $\sqrt{d_k}$ pulls the variance back to $1$. But read what that proves: it controls the scale *under the initialization assumptions* — independent, unit-variance, uncorrelated $q$ and $k$. The moment training moves the projection matrices those assumptions rot, the query and key vectors grow norms and their components correlate, and $q \cdot k$ is free to grow again. The factor is a one-time variance normalization at $t=0$; it sets the expected scale of a fresh layer and then does nothing. It is a calibration, not a *bound*. The residual-stream normalizations in the strong low-resource recipe do not close the gap either: ScaleNorm ($g \cdot x/\lVert x\rVert$ along the embedding dimension) acts on the whole pre-split residual vector before the attention projections, and normalizing that full vector to unit length does not make the per-head query/key slices unit length. None of these touch the one place the saturation originates — the per-head scores entering the softmax.

What I want is a *bound* on each score that holds for all of training. The shape of this disease is familiar from the output softmax of language models: the logit for word $i$ is $z_i = \lVert x_i\rVert\,\lVert h\rVert\,\cos\theta_i + b_i$, the polar form of the dot product, and the *norm* term empirically dominates the *angle* term — word embeddings spread their norms over a wide range while their angles fall into a narrow band, so the magnitudes, not the directions, decide the ranking. The attention score $q \cdot k = \lVert q\rVert\,\lVert k\rVert\cos(\text{angle})$ is the same object one layer down, and the magnitudes can swamp the angular agreement the head should actually reason with. So I propose QKNorm — query-key normalization: discard the magnitudes and keep the cosine. Replace $q \cdot k$ with $(q \cdot k)/(\lVert q\rVert\,\lVert k\rVert)$, which means $\ell_2$-normalizing each query and each key to unit length before the dot product:

$$\hat q = q/\lVert q\rVert, \qquad \hat k = k/\lVert k\rVert, \qquad \text{score} = \hat q \cdot \hat k.$$

Now every score is a cosine in $[-1, 1]$ no matter what training does to the projection norms — a guarantee that is a property of the operation, not a calibration at init, and one $1/\sqrt{d_k}$ could never give. Two design choices are load-bearing about *which* vectors and *along which dimension*. The normalization must happen *after* the multi-head split, *along the head dimension*, because the dot product that feeds the softmax is contracted per head over the length-$d_k$ slices $\hat q$ and $\hat k$; bounding the cosine over that contracted dimension is what bounds the score, and normalizing the full pre-split vector (what ScaleNorm does) does not. And it applies to $q$ and $k$ only, never $v$: the values are *averaged* by the attention weights, not scored against each other, so bounding them would deform the representation being averaged for no benefit. This is also why QKNorm *complements* rather than replaces residual-stream normalization — ScaleNorm and LayerNorm act on the residual path; QKNorm lives strictly inside attention on the scores — so I keep vanilla LayerNorm on the stream, and it pairs naturally with PreNorm (residual path stays an identity map, keeping warmup-light low-resource training stable) and FixNorm (unit-length word embeddings — the very same magnitude-vs-direction principle applied at the input side).

Bounding the score to $[-1, 1]$ kills saturation-by-magnitude but creates the mirror-image disease: it makes concentration impossible. With no extra scale, even the best possible row — one entry at $1$, every competitor at $-1$, over $m$ keys — gives the top entry probability $e^1/(e^1 + (m-1)e^{-1}) = e^2/(e^2 + m - 1)$, which for a row of length about 75 is only $\approx 0.091$, and real cosine gaps are smaller than the full width $2$, so typical rows are flatter still. So the bound is necessary but not sufficient; I have to be able to *expand* the bounded range back out. Note the direction has flipped relative to scaled dot product — there the dot products were too big and I *divided*, here the cosines are too small and I must *multiply* by some $g > 1$. The attention becomes

$$\mathrm{softmax}\!\left(g \cdot \hat Q \hat K^\top\right) V$$

in place of $\mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$. The cosine gives the floor (scores cannot run away); $g$ buys back the ceiling (scores can stretch far enough apart for a sharp selection), and together they give the head the full range from diffuse to peaky. Fixing $g=1$ is not a half-measure — it removes the head's ability to make near-copy selections — so the learned re-scaling is the indispensable complement to the bound, not decoration. The right amount of stretch is not a fixed property of the head dimension; it depends on row length and on how sharp a given module's heads need to become, which the model is better placed to discover than I am. So $g$ is a *learnable* scalar, one per attention module, tuned by backprop — a learned temperature on the bounded cosines, the same move ScaleNorm makes with its single learned scale.

I still must initialize $g$, since a bad start on a learnable parameter wastes training budget before it crawls to a usable value. The job of $g$ is to make it *possible* for the maximum entry in a row to dominate; with scale $g$ the best-case top probability is

$$p_{\max} = \frac{e^g}{e^g + (m-1)e^{-g}} = \frac{1}{1 + (m-1)e^{-2g}},$$

so longer sequences need more scale. A purely derived row-length constant would still be crude, because real cosine matrices are not one $1$ and many $-1$s, so the initialization should be an empirical rule of thumb. I use the off-diagonal count of a typical full $L \times L$ similarity matrix as a coarse proxy for matrix size and compress it with a logarithm,

$$g_0 = \log_2\!\left(L^2 - L\right),$$

with $L$ the 97.5th-percentile sequence length over the training data (source and target) — a robust stand-in for typical matrix size rather than the maximum, which a few outlier-length sentences would inflate. For the benchmark $L$ of 72 through 79 this gives $g_0$ from $\log_2(72^2-72) = 12.32$ to $\log_2(79^2-79) = 12.59$, stretching $[-1,1]$ to roughly $[-12.5, 12.5]$ — but only as a starting point; the parameter is then free to move. There is a clean head-count sanity check: because the score is normalized per head it remains a cosine over the $d_k$-dimensional slice whether $d_k$ is 16 or 256, so the mechanism never relied on the $\sqrt{d_k}$ variance argument and should not break as the head dimension shrinks. The causal chain holds together end to end: heads collapse because the softmax sees only differences and the dot product is unbounded; $1/\sqrt{d_k}$ bounds nothing; the cure is to keep the cosine and drop the magnitudes, normalizing $q,k$ per head along the head dimension; that bounds the score for all of training but forbids sharpness, which a learnable per-module $g$, initialized by $\log_2(L^2-L)$, buys back.

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
