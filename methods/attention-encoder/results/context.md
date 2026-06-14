# Context: connecting positions in a sequence (circa 2014-2017)

## Research question

I have a variable-length sequence of element representations `(x_1, ..., x_n)` with each
`x_i` a `d`-dimensional vector - tokens in a sentence, frames in a signal, or a group of
interaction tuples I want to summarize - and I need a layer that produces a new sequence
`(z_1, ..., z_n)` in which information from distant positions can affect each output without
being forced through a long serial chain. The pain is that the dominant way to build such a
layer threads the elements through sequential steps, so the computation cannot be
parallelized across positions and the path a signal must travel from position `i` to
position `j` grows with `|i - j|`, which is exactly what makes long-range dependencies hard
to learn. The precise goal is a single, stackable layer that simultaneously: (1) keeps the
number of sequential operations from growing with the sequence length; (2) keeps the maximum
path length between any two positions short, so gradients between distant positions do not
have to survive a long chain; (3) is cheap per layer and maps cleanly to matrix hardware;
(4) remains trainable when many such layers are stacked into a deep model; and (5) computes
normalization statistics without depending on the batch size or on other examples, so it
works even when the "batch" is a single variable-length group. Existing layers achieve some
of these but not all at once; closing that gap is the problem.

## Background

By this time the established engine for sequence modeling and transduction is the recurrent
neural network — LSTMs (Hochreiter & Schmidhuber 1997) and GRUs (Cho et al. 2014) — which
process a sequence by maintaining a hidden state `h_t = f(h_{t-1}, x_t)` advanced one
position at a time. This is state of the art for language modeling and machine translation
(Sutskever et al. 2014; Bahdanau et al. 2015; Wu et al. 2016), but it has a structural cost:
because `h_t` depends on `h_{t-1}`, the positions must be visited in order, so there is no
parallelism *within* a single example, and at long sequence lengths the memory needed to
batch across examples runs out. The number of sequential operations is `O(n)`, and the
shortest path through the network between two positions `i` and `j` is `O(|i - j|)`; a
well-known observation is that the longer the path a forward or backward signal must
traverse, the harder the corresponding dependency is to learn (Hochreiter et al. 2001).

A second line of work attacks the sequential bottleneck by replacing recurrence with
convolution — the Extended Neural GPU (Kaiser & Sutskever 2016), ByteNet (Kalchbrenner et
al. 2017), and ConvS2S (Gehring et al. 2017) — all of which compute hidden representations
for all positions in parallel. These have `O(1)` sequential depth, but a convolution with
kernel width `k < n` only connects positions within a local window, so to connect two
distant positions you must stack many layers: the path length grows like `O(n/k)` for
contiguous kernels or `O(log_k n)` for dilated ones. The per-layer cost is `O(k · n · d^2)`,
more expensive than recurrence by a factor of `k`; separable convolutions (Chollet 2016)
reduce this to `O(k · n · d + n · d^2)`.

The other load-bearing idea is the attention mechanism. Bahdanau, Cho & Bengio (2015)
introduced it to relieve the fixed-length-vector bottleneck of encoder-decoder translation:
instead of compressing the whole source into one vector, the decoder, at each output step
`i`, computes a weight `alpha_ij = softmax_j(e_ij)` over every source annotation `h_j`,
where `e_ij = a(s_{i-1}, h_j)` is a small feed-forward scoring network, and forms a context
vector `c_i = sum_j alpha_ij h_j`. This is a soft, differentiable, content-based,
distance-agnostic way to connect one position to all others, and an attention layer can link
a large number of positions at low sequential cost. The catch is that in nearly every system
of the time attention is used *in conjunction with* a recurrent network — the annotations
`h_j` it weights are themselves produced by an RNN encoder, so the sequential cost is still
there.

Two facts about the design space matter here and are knowable before any new layer is built.
First, the cost of a content-weighted average of value vectors scales with the number of
position pairs: forming all pairwise compatibilities for a sequence of length `n` with
representation width `d` costs on the order of `n^2 · d`, which is *cheaper* than the
recurrent `n · d^2` precisely when `n < d` — the common regime for the sub-word
representations used by strong translation systems (Wu et al. 2016; Sennrich et al. 2015),
where sequences are short relative to the hidden width. Second, the compatibility score
between a query vector and a key vector can be computed either additively (a feed-forward net
with a hidden layer, as in Bahdanau) or multiplicatively (a dot product, optionally with a
learned matrix in between). The two are comparable in accuracy, but the multiplicative form
is a single matrix multiplication, so it is much faster and more memory-efficient in practice
because it maps onto highly optimized dense-matrix kernels — a concrete observed property of
the two scoring functions, independent of any particular model.

A diagnostic fact about the multiplicative score is also already on record and will shape any
design that uses it: as the dimension of the vectors being dotted grows, the magnitude of the
dot product grows with it, and feeding very large values into a softmax pushes it toward a
near-one-hot distribution where its gradient is tiny — so without some control, dot-product
scoring degrades for large vector widths even though it ties with additive scoring for small
ones (Britz et al. 2017).

Finally, two general-purpose components for training deep stacks are available. Residual
connections (He et al. 2016) wrap a sub-computation `F` as `y = F(x) + x`, so an identity
path runs around `F`; this lets very deep stacks train, because gradients reach early layers
through the identity even when `F`'s Jacobian is small. And layer normalization (Ba, Kiros &
Hinton 2016) normalizes the summed inputs of a layer across the *feature* dimension within a
single training case: with `H` features it computes `mu = (1/H) sum_i a_i`,
`sigma = sqrt((1/H) sum_i (a_i - mu)^2)`, then `h = (g/sigma) ⊙ (a - mu) + b` with learned
gain `g` and bias `b`. Because the statistics are taken over the features of one example and
not across the batch, layer normalization introduces no dependency between training cases and
works unchanged at batch size 1 and for variable-length inputs — the regime where batch
normalization, which needs cross-example statistics, struggles.

## Baselines

These are the prior layer types a new sequence-mixing layer would be measured against and
would react to.

**Recurrent layers (LSTM, GRU; Hochreiter & Schmidhuber 1997; Cho et al. 2014;
Sutskever et al. 2014).** A hidden state is advanced one position at a time,
`h_t = f(h_{t-1}, x_t)`, so the representation at each position is a function of all earlier
positions through the recurrence. Per layer the cost is `O(n · d^2)` (a `d × d` matrix
multiply at each of `n` steps). **Gap:** the update at `t` cannot start until the update at
`t-1` finishes, so there are `O(n)` sequential operations and no parallelism within an
example; and the shortest path between positions `i` and `j` is `O(|i - j|)`, so a dependency
between far-apart positions must survive a long chain of transformations, which is hard to
learn and limits how long the sequences can be before training becomes impractical.

**Convolutional sequence layers (ConvS2S, ByteNet, Neural GPU; Gehring et al. 2017;
Kalchbrenner et al. 2017; Kaiser & Sutskever 2016).** Apply a width-`k` convolution to
compute all positions' hidden states in parallel, giving `O(1)` sequential depth. **Gap:** a
single layer with `k < n` only connects positions inside a window of size `k`; relating two
positions farther apart than `k` requires stacking `O(n/k)` layers (contiguous kernels) or
`O(log_k n)` (dilated kernels), so the longest path between positions still grows with the
sequence, and the per-layer cost `O(k · n · d^2)` carries the extra factor of `k`. Separable
convolutions cut the constant but not the growing-path-length behavior.

**Additive attention on top of an RNN (Bahdanau et al. 2015).** The first content-based,
distance-agnostic connection between positions: `e_ij = a(s_{i-1}, h_j)` with `a` a
feed-forward net with one hidden layer, `alpha_ij = softmax_j(e_ij)`,
`c_i = sum_j alpha_ij h_j`. It removes the fixed-length bottleneck and connects any pair of
positions in `O(1)` sequential steps. **Gap:** the annotations `h_j` it operates on are
produced by a recurrent encoder, so the layer does not by itself remove the `O(n)` sequential
cost — it is an addition to recurrence, not a replacement; and the scoring function is a
per-pair feed-forward network, which does not reduce to a single dense matrix multiply.

**Multiplicative (dot-product) attention on top of an RNN (Luong et al. 2015).** Replaces
the additive scorer with a dot product: `score(h_t, h̄_s) = h_t^T h̄_s` (or
`h_t^T W_a h̄_s` with a learned matrix), so all compatibilities are one matrix multiply
`Q K^T`. Faster and more memory-efficient than additive scoring at equal accuracy on small
widths. **Gap:** still attached to a recurrent encoder, so it inherits the sequential cost;
it is single-scored (one weighted average per query, blending whatever relations matter into
a single distribution); and at large vector widths the unscaled dot products grow large and
saturate the softmax, where gradients vanish.

**Feed-forward attention between sequences, and intra-sequence attention (Parikh et al. 2016;
Cheng et al. 2016; Lin et al. 2017).** Parikh et al. compute alignments between two sentences
with no recurrence at all: `e_ij = F(ā_i)^T F(b̄_j)`, with `F` a feed-forward network applied
to each element independently, then a dot product — a content-based, fully parallelizable
alignment at cost `O(ℓ^2 d)` plus `O(ℓ d^2)`, and they note an optional *intra-sentence*
variant that aligns a sequence to itself, `f_ij = F_intra(a_i)^T F_intra(a_j)`,
`a'_i = sum_j softmax_j(f_ij + d_{i-j}) a_j`. Cheng et al. (2016) likewise relate positions
of a single sequence to each other (intra-attention) but inside an LSTM, and Lin et al.
(2017) embed a sentence with several parallel attention distributions (a 2-D attention
matrix). **Gap:** these use such within-sequence content matching as a *sub-component* - for
a pairwise classification task, or inside a recurrent cell, or to pool a sentence into one
fixed vector - so they do not remove the general sequence-transduction stack's dependence on
other layer types; the dot-product scores are unscaled and single-headed; and they are not
packaged with the residual/normalization machinery needed to stack many layers deep.

## Evaluation settings

The natural yardsticks already in use for a sequence-transduction layer:

- **WMT 2014 English-to-German** machine translation, about 4.5M sentence pairs, encoded with
  byte-pair encoding into a shared source-target vocabulary of roughly 37,000 tokens. The
  standard metric is BLEU on the held-out `newstest2014` set; perplexity per token is tracked
  during development on `newstest2013`.
- **WMT 2014 English-to-French**, the larger benchmark, about 36M sentence pairs split into a
  32,000 word-piece vocabulary. Same BLEU metric.
- Sentence pairs are batched by approximate length so that each batch holds a comparable token
  count; sequence lengths are short relative to the model width in these sub-word
  representations, the regime where all-pairs compatibility computations are plausible.
- **English constituency parsing** (Penn Treebank WSJ; Marcus et al. 1993), used to check
  that a sequence model generalizes beyond translation, in both a large-data and a
  limited-data setting.
- Protocol: train on matrix hardware (multi-GPU), compare layer types on translation quality
  and on per-layer complexity / sequential-operation count / maximum path length as the
  intrinsic measures of the layer itself.

## Code framework

The existing harness is deliberately plain: an embedding turns input elements into
`d`-dimensional vectors, a stack of identical blocks refines those vectors, and a downstream
head consumes the encoded sequence. The internal architecture of the block is
the open part; the surrounding substrate only fixes shapes, parameter registration, and the
loop that applies the same kind of block repeatedly.

```python
import torch
import torch.nn as nn


class SequenceBlock(nn.Module):
    """One encoder block. Maps (batch, n, d_model) to (batch, n, d_model)."""

    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.d_model = d_model
        # TODO: fill the block body.

    def forward(self, x, mask=None):
        # x: (batch, n, d_model)
        # TODO: return an encoded sequence with the same shape.
        raise NotImplementedError


class Encoder(nn.Module):
    """Existing harness: embed the input elements, run a stack of identical blocks,
    return the encoded sequence."""

    def __init__(self, input_size, d_model, num_layers, **kwargs):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList(
            [SequenceBlock(d_model, **kwargs) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        # x: (batch, n, input_size)
        h = self.embed(x)                  # (batch, n, d_model)
        for layer in self.layers:
            h = layer(h, mask)             # each block refines the sequence
        return h                           # (batch, n, d_model)
```

The harness supplies the embedded sequence and stacks the blocks; `SequenceBlock` is the
single empty slot.
