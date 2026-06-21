# Context: the self-attention softmax and its scaling (circa 2019-2020)

## Research question

Self-attention scores every pair of token representations by a compatibility function and turns
those scores into a distribution with a softmax; the distribution is then used to mix the value
vectors. The compatibility function in use is a scaled dot product, and the softmax that sits on
top of it has a property that is easy to forget: it depends only on the *differences* between its
inputs, and those inputs are unbounded. A large score difference can therefore dominate the whole
row of attention weights and silence every other connection, even when that difference is small
relative to the absolute score magnitudes. Concretely, `softmax([760, 752, 750])` and
`softmax([12, 4, 2])` are the same distribution, `[0.99962, 0.00034, 0.00005]` — a near one-hot,
"winner-take-all" assignment.

The question is how the per-head query and key vectors should be turned into the scores that feed
the softmax — especially in low-resource settings (roughly ten thousand to a few hundred thousand
sentence pairs) where the model has limited signal to learn rich attention structure.

## Background

By this time the Transformer (Vaswani et al. 2017) is the dominant architecture for neural machine
translation, and a steady stream of architectural and normalization-centric modifications has been
made at exactly the layers in question — the multi-head attention and the normalization around it.
Several threads of the field's state are load-bearing here.

**Why a scaling factor is there at all.** Scaled dot-product attention computes
`softmax(QK^T / sqrt(d_k)) V`. The `1/sqrt(d_k)` is not cosmetic. If the components of a query and
a key are modeled as independent, mean-zero, unit-variance random variables, then their dot
product `q · k = sum_{i=1}^{d_k} q_i k_i` has mean 0 and variance `d_k`, so its typical magnitude
grows like `sqrt(d_k)`. For large head dimension the raw dot products grow large, push the softmax
into regions of vanishingly small gradient, and stall learning; dividing by `sqrt(d_k)` rescales
the logits' variance back toward 1 at initialization.

**Layer normalization and its variants.** For NLP models, layer normalization (Ba et al. 2016)
outperforms batch normalization, in part because language models show more variance in batch
statistics; it boosts deep networks chiefly by controlling gradients. It re-scales and re-centers
the activation distribution over the feature dimension, `LayerNorm(a) = (a − mu)/sigma · g + b`,
with `mu` and `sigma` the per-vector mean and standard deviation. A subsequent observation is that
the re-centering may be unnecessary: root-mean-square layer normalization (Zhang & Sennrich 2019)
keeps only the re-scaling, `RMSNorm(a) = a / RMS(a) · g` with `RMS(a) = sqrt((1/n) sum_i a_i^2)`,
dropping the mean subtraction. This forces the summed inputs onto a `sqrt(n)`-scaled sphere,
preserves the re-scaling invariance that is hypothesized to be the real source of LayerNorm's
benefit, and runs 7-64% faster. The Euclidean norm and RMS differ only by a factor of `sqrt(n)`.
Both the *type* of normalization and *where* it sits in the Transformer are known to matter a lot.

**The dot-product softmax can let magnitude drown out direction.** A closely related diagnostic
comes from the output softmax of language models. There the logit for word `i` is
`z_i = ||x_i|| · ||h|| · cos(theta_i) + b_i`, the polar form of the dot product between a word
vector `x_i` and the prediction vector `h`. Demeter, Kimmel & Downey (2020) find empirically that
the embedding-norm term dominates the angular term `cos(theta_i)`: word vectors spread their norms
over a wide range while their angles relative to a reference vector fall into a narrow band, so the
norms decide the ranking. Words with smaller norms — and, they prove, any word whose embedding is
interior to the convex hull of the vocabulary embeddings — are systematically less likely to be
predicted, regardless of context, a structural ceiling they call the stolen-probability effect.
The structural lesson is general: when a *dot product* (a product of two norms and a cosine) is
fed into a softmax, the magnitudes can overwhelm the directional signal that the model presumably
wanted to act on.

**Bounded compatibility and unconstrained attention.** Cosine normalization (Luo et al. 2018)
replaces the inner product `w · x` inside fully-connected layers with the cosine similarity
`w · x / (||w|| ||x||)`, bounding each pre-activation to `[−1, 1]`. Separately, Richter &
Wattenhofer (2020) note that the softmax constrains every attention output to the convex hull of
the value vectors, which they argue limits the functions a head can express, and propose replacing
the softmax with a normalization altogether. Both are reactions to the same family of concerns
about what the softmax-over-dot-products can and cannot represent.

**The state-of-the-art low-resource recipe.** The strongest published results on the five
low-resource pairs of interest come from Nguyen & Salazar (2019), who combine three
normalization-centric changes, each contributing about +0.3 BLEU for roughly +1.1 BLEU total:
- *PreNorm* moves layer normalization to the input of each sublayer, `x_{l+1} = x_l +
  F_l(LayerNorm(x_l))`, instead of after the residual addition (PostNorm,
  `x_{l+1} = LayerNorm(x_l + F_l(x_l))`). Keeping the residual path an identity map stops the
  normalization from injecting terms into the residual gradient that can explode or vanish, which
  makes deep and low-resource training more stable and largely removes the need for learning-rate
  warmup. They find PreNorm helps in low-resource but not high-resource settings.
- *FixNorm* (Nguyen & Chiang 2018) fixes word embeddings to unit length, `g · w/||w||`, which aids
  rare-word translation — otherwise a word embedding's norm drowns out its direction.
- *ScaleNorm* replaces layer normalization with a scaled `l2` normalization,
  `ScaleNorm(x; g) = g · x/||x||`, a single learned scalar `g` (initialized to `sqrt(d)`) in place
  of LayerNorm's `2d` scale-and-shift parameters. It projects each activation onto a
  `(d−1)`-sphere with a learned radius, expressing the inductive bias that each sublayer's
  activations have an ideal global scale. In the low-resource Transformer codepath, this acts on the
  whole embedding-dimension residual vector before the attention projections and head split. RMSNorm
  can be viewed as the related rescaling-only family member with per-unit gains; tying those gains
  and dividing by `sqrt(d)` recovers ScaleNorm.

## Baselines

These are the prior methods a new attention scheme would be measured against.

**Scaled dot-product multi-head attention (Vaswani et al. 2017).** For each head, project the
input into queries, keys and values of head dimension `d_k`; form scores `QK^T`, divide by
`sqrt(d_k)`, softmax over the key axis, and mix the values: `softmax(QK^T/sqrt(d_k)) V`. The
scaling sets the logits' variance to about 1 at initialization under the independent-component
model.

**LayerNorm-based Transformer (Ba et al. 2016; Vaswani et al. 2017).** Normalize each activation
vector over its feature dimension with a learned scale-and-shift, placed after the residual
addition (PostNorm) in the original architecture. Stabilizes deep training by controlling
gradients.

**ScaleNorm + FixNorm + PreNorm (Nguyen & Salazar 2019).** The state-of-the-art low-resource
recipe described above: PreNorm for stable warmup-light training, FixNorm for rare words, and
ScaleNorm — `g · x/||x||` along the embedding dimension on the residual-stream vector before the
attention projections and multi-head split — as a fast single-scalar replacement for LayerNorm.
ScaleNorm+FixNorm at the final linear layer coincides with cosine normalization with a learned
scale.

**Cosine normalization (Luo et al. 2018) and softmax-free attention (Richter & Wattenhofer 2020).**
The first bounds inner products in fully-connected layers to `[−1, 1]` via cosine similarity; the
second removes the softmax from self-attention entirely in favor of a normalization, on the
grounds that the simplex constraint limits expressivity.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **Five low-resource translation pairs.** Galician→English and Slovak→English (TED Talks corpus,
  the two lowest-resource pairs at roughly 10k and 61k examples), Arabic→English and English→Hebrew
  (TED Talks, 100-300k examples), and English→Vietnamese (IWSLT'15, 133k examples) — spanning
  resource levels, language families, writing directions, and English-as-source vs. -as-target.
- **Architecture and optimization.** Transformer with hidden dimension 512 and feed-forward
  dimension 2048, multi-head attention, dropout on sublayer outputs / ReLU / attention weights,
  label smoothing; trained with Adam under an inverse-square-root learning-rate schedule with
  linear warmup (on the order of 8,000 steps) and a validation-based decay, training to the
  minimum learning rate and reporting test BLEU at the epoch with the best validation BLEU.
- **Subword preprocessing.** Byte-pair encoding via fastBPE, with the number of merge operations
  scaled to the data size; the target input embedding and the final linear layer weight are tied.
- **Metric.** BLEU, scored with the Moses `multi-bleu.perl` and `multi-bleu-detok.perl` scripts
  and cross-checked with SacreBLEU, with statistical significance assessed by bootstrap resampling.
- **A diagnostic visualization setting.** The Annotated Transformer implementation trained on
  IWSLT'16 German→English, used to inspect per-head attention heatmaps qualitatively.

## Code framework

The new attention scheme plugs into the existing multi-head attention module of the Transformer
codebase used for the baselines. What already exists: the query/key/value projections, the split
into heads, a softmax over the key axis, the value mixing, the output projection, and the
residual-stream normalization (LayerNorm, or ScaleNorm) around each sublayer. What is *not* settled
is how the per-head query and key vectors are turned into the scores that feed the softmax — that
is the slot to be designed. The substrate below is the generic per-head attention machinery; the
single empty slot is the score computation.

```python
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


class MultiheadAttention(nn.Module):
    """Per-head scaled attention. Projections, head split, softmax, value mixing
    and the output projection already exist. How the per-head q and k become the
    scores fed to the softmax is the open slot."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = args.att_dropout

        # q/k/v/o projections, packed
        self.weights = Parameter(torch.Tensor(4 * self.embed_dim, self.embed_dim))
        nn.init.normal_(self.weights, mean=0.0, std=(2 / (5 * self.embed_dim)) ** 0.5)

        # TODO: any state the score function needs.
        self.score_state = None

    def _split_heads(self, x):
        bsz, length, _ = x.size()
        return (x.reshape(bsz, length, self.num_heads, self.head_dim)
                 .transpose(1, 2)
                 .reshape(bsz * self.num_heads, -1, self.head_dim))

    def _score(self, q, k):
        # q, k: [bsz * num_heads, len, head_dim]
        # TODO: turn the per-head queries and keys into the scores that go to the
        #       softmax. This is the slot the method fills.
        raise NotImplementedError

    def forward(self, q, k, v, mask):
        q, k, v = self.proj_qkv(q, k, v)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        att = self._score(q, k)                       # [bsz*heads, len, len]
        bh, src_len, tgt_len = att.size()
        att = att.reshape(-1, self.num_heads, src_len, tgt_len)
        if mask is not None:
            att.masked_fill_(mask, -1e9)
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        out = torch.bmm(att.reshape(bh, src_len, tgt_len), v)
        out = (out.reshape(-1, self.num_heads, src_len, self.head_dim)
                  .transpose(1, 2)
                  .reshape(-1, src_len, self.embed_dim))
        return self.proj_o(out), att

    # proj_qkv / proj_o: existing packed linear projections (omitted)
```

The forward path — split, score, mask, softmax, dropout, mix, project — is fixed; `_score` is the
one stub the method will fill in.
