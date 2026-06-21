# Context: content-addressed sequence mixing for neural transduction (circa 2014-2017)

## Research question

A sequence model has to let information flow between positions: when producing the
representation (or the output token) at one position, it must be able to pull in information
from the other positions, however far away they sit. One way to phrase the operation we want is
a single, reusable layer that updates each position's vector as a *content-addressed* read over
the whole sequence — every position emits something that says "this is what I am looking for,"
every position advertises "this is what I hold," and the layer returns, for each reader, a
weighted blend of what the others hold, with the blend decided by how well each holder matches
the reader's request. We want this read to be usable as the primary mixing operation of a deep
stack, run in every layer rather than as a one-off bolt-on, so its constant factors and
hardware utilization matter as much as its asymptotic cost. We want the weights to be a proper
normalized blend, so the output is a convex combination of the things being read and the whole
thing is smoothly differentiable end to end. And we want it to support causal use: in an
autoregressive decoder, the read at position `i` must see positions `<= i` only, never the
future. Designing such a layer, and using it as the cross-position mechanism of a stacked
non-recurrent model, is the question.

## Background

By this time the dominant way to move information along a sequence is **recurrence**: an RNN,
typically an LSTM (Hochreiter & Schmidhuber 1997) or GRU (Cho et al. 2014), threads a hidden
state `h_t = f(h_{t-1}, x_t)` left to right, so position `t` sees the past through a
single fixed-width state that is updated at every step. It is **inherently sequential** —
`h_t` is computed from `h_{t-1}` — and the path a signal travels between two positions `i` and
`j` is `O(|i-j|)` steps long, with forward and backward signals decaying along long paths
(Hochreiter et al. 2001).

A parallel line of work replaces recurrence with **convolution** —
the Extended Neural GPU, ByteNet (Kalchbrenner et al. 2016), ConvS2S (Gehring et al. 2017) all
compute all positions at once with `O(1)` sequential operations. A convolution with kernel
width `k < n` connects positions within a window; relating two far-apart positions
requires a stack of layers, so the path length grows with distance — linearly for
contiguous kernels, `O(log_k n)` for dilated ones (ByteNet).

A useful way to compare candidate mixing layers, made
explicit in this period, is along three axes: the **per-layer compute**, the **number of
sequential operations** (how parallel it is), and the **maximum path length** between any two
positions (how easy long-range learning is). A recurrent layer costs `O(n d^2)` compute but
`O(n)` sequential ops and `O(n)` path length. A convolutional layer costs `O(k n d^2)`,
`O(1)` sequential, but `O(log_k n)` path. A layer that connects *all* pairs of positions in
`O(1)` sequential ops and `O(1)` path length — where any position reaches any other in a single
hop — would have compute scaling with the number of pairs, `O(n^2 d)`, which is favorable
whenever the sequence length `n` is below the representation width `d` — the common regime for
the sentence-length, subword-tokenized sequences of the day (word-piece, Wu et al. 2016;
byte-pair, Sennrich et al. 2015).

The mechanism that already realizes "read from anywhere in one hop" is **attention**, which by
this point is the integral ingredient that made neural machine translation work: it lets a
decoder draw on any source position without regard to distance, at low sequential cost. A
large-scale study of translation architectures (Britz et al. 2017) compares the two standard
scoring rules across attention widths from 128 to 1024: the multiplicative (dot-product) score
is competitive at widths 128-512 (BLEU around 22 on newstest2013) but measures 18.22 at width
1024, while the additive score stays near 22 across the whole range.

## Baselines

The prior content-addressed reads a new layer would be measured against and react to.

**Additive attention (Bahdanau, Cho & Bengio 2015).** The original soft-alignment mechanism
for translation. For a reader state `s_{i-1}` and each holder annotation `h_j`, score the pair
with a small feed-forward network — a learned energy

```
e_ij = a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j),
```

normalize the energies over holders with a softmax, `alpha_ij = exp(e_ij) / sum_k exp(e_ik)`,
and return the weighted blend `c_i = sum_j alpha_ij h_j` (an "expected annotation"). It is fully
differentiable. The score is an MLP — a matrix multiply, a `tanh`, and an inner product with
`v_a` — evaluated for every (reader, holder) pair. It is used as a single read per decoder step
over the source.

**Multiplicative / dot-product attention (Luong, Pham & Manning 2015).** Simplify the score to
a bilinear or plain inner product. Global attention computes `alpha(s) = softmax(score(h_t,
h_s))` with `score` one of `h_t^T h_s` (dot), `h_t^T W_a h_s` (general), or a concat-MLP, then
blends the source states. The dot form is a single matrix multiplication and expresses
compatibility directly as vector alignment. It is presented and used unscaled, and is attached
to a recurrent decoder as an auxiliary read alongside the RNN.

**Softmax normalization (standard).** Given a vector of scores `z`, `softmax(z)_i = exp(z_i) /
sum_j exp(z_j)` maps them to a nonnegative, sum-to-one weighting that is smooth and
differentiable. It is the standard way to turn compatibility scores into a blend. Its gradient
is largest when the inputs are of moderate, similar magnitude and shrinks toward zero once one
logit dominates and the distribution goes near one-hot.

**Recurrent and convolutional mixing (LSTM/GRU; ByteNet, ConvS2S).** The incumbent
cross-position mechanisms, described above. Recurrence costs `O(n)` sequential ops and
`O(n)` path length; convolution parallelizes but its path length grows with distance
(`O(n/k)` or `O(log_k n)`).

## Evaluation settings

The natural yardsticks already in use for sequence transduction. Machine translation is the
standard task: the WMT 2014 English-to-German and English-to-French benchmarks, scored by
**BLEU** on the held-out newstest sets, with sequences tokenized into subword units (byte-pair
encoding, Sennrich et al. 2015; word-piece, Wu et al. 2016) so vocabulary size stays bounded
and sequence lengths stay in the regime where `n < d`. Training is on stacked encoder-decoder
models with residual connections and layer normalization, on multi-GPU machines, with training
wall-clock and parallelizability themselves treated as figures of merit. The candidate mixing
layers are also compared analytically along the three axes above — per-layer compute, sequential
operations, maximum path length — as a pre-experimental way to argue one layer should be easier
to optimize and faster to run than another.

## Code framework

The layer plugs into a standard sequence-model harness: token embeddings go in, a stack of
residual sublayers mixes information across positions and transforms it in place, and a final
projection reads out. The substrate below is only the generic machinery that already exists —
layer normalization, residual wiring, learned linear maps, and an optional positional mask —
with one empty slot for the cross-position computation.

```python
import torch.nn as nn


class SequenceMixingLayer(nn.Module):
    """One residual sublayer of the stack."""

    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def mix_positions(self, readers, addresses, payloads, mask=None):
        # TODO: design the cross-position computation.
        raise NotImplementedError

    def forward(self, x, mask=None):
        h = self.norm(x)
        readers, addresses, payloads = self.in_proj(h).chunk(3, dim=-1)
        out = self.mix_positions(readers, addresses, payloads, mask=mask)
        return x + self.o_proj(out)
```

The harness supplies a sequence of vectors and an optional mask; the body of `mix_positions` is
the slot to fill.
