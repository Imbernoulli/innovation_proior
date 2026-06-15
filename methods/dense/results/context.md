# Context: content-addressed sequence mixing for neural transduction (circa 2014-2017)

## Research question

A sequence model has to let information flow between positions: when producing the
representation (or the output token) at one position, it must be able to pull in whatever it
needs from the other positions, however far away they sit. The precise object we want is a
single, reusable layer that updates each position's vector as a *content-addressed* read over
the whole sequence — every position emits something that says "this is what I am looking
for," every position advertises "this is what I hold," and the layer returns, for each reader,
a weighted blend of what the others hold, with the blend decided by how well each holder
matches the reader's request. Several properties have to hold at once for this to be usable as
the workhorse of a deep stack. The read has to be **cheap enough to run in every layer** —
not a small bolt-on used once, but the primary mixing operation, so its constant factors and
hardware utilization matter as much as its asymptotic cost. The weights have to be a proper
**normalized blend** (nonnegative, summing to one), so the output stays a convex combination
of the things being read and the whole thing is smoothly differentiable end to end. The
scoring rule that decides the blend has to **stay numerically well-behaved as the vectors get
wide** — a high-dimensional model must not, merely by being high-dimensional, drive the
weighting into a degenerate all-or-nothing regime where learning stalls. And the layer has to
support **causal use**: in an autoregressive decoder, the read at position `i` must be allowed
to see positions `<= i` only, never the future, with the masking folded into the same
operation rather than bolted on around it. Building a single layer that meets all of these,
and meets them well enough to be stacked dozens deep and used as the *only* cross-position
mechanism, is the problem.

## Background

By this time the dominant way to move information along a sequence is **recurrence**: an RNN,
typically an LSTM (Hochreiter & Schmidhuber 1997) or GRU (Cho et al. 2014), threads a hidden
state `h_t = f(h_{t-1}, x_t)` left to right, so position `t` sees the past only through a
single fixed-width state that has been overwritten at every step. This has two well-known
costs. It is **inherently sequential** — `h_t` cannot be computed until `h_{t-1}` is — which
prevents parallelization within a training example and becomes the binding constraint at long
sequence lengths. And it forces every long-range dependency to survive a long chain of state
overwrites: the path a signal travels between two positions `i` and `j` is `O(|i-j|)` steps
long, and long paths are exactly what make distant dependencies hard to learn, because the
forward and backward signals decay along them (Hochreiter et al. 2001).

A parallel line of work replaces recurrence with **convolution** to get parallelism back —
the Extended Neural GPU, ByteNet (Kalchbrenner et al. 2016), ConvS2S (Gehring et al. 2017) all
compute all positions at once with `O(1)` sequential operations. But a convolution with kernel
width `k < n` only connects positions within a window; relating two far-apart positions
requires a stack of layers, so the path length still grows with distance — linearly for
contiguous kernels, `O(log_k n)` for dilated ones (ByteNet). Distance is reduced, not removed.

A useful way to compare candidate mixing layers, then, is along three axes that were made
explicit in this period: the **per-layer compute**, the **number of sequential operations**
(how parallel it is), and the **maximum path length** between any two positions (how easy
long-range learning is). A recurrent layer costs `O(n d^2)` compute but `O(n)` sequential ops
and `O(n)` path length. A convolutional layer costs `O(k n d^2)`, `O(1)` sequential, but
`O(log_k n)` path. The open slot in this table is a layer that connects *all* pairs of
positions in `O(1)` sequential ops and `O(1)` path length — a layer where any position can
reach any other in a single hop. Its compute would scale with the number of pairs, `O(n^2 d)`,
which is favorable whenever the sequence length `n` is below the representation width `d` — the
common regime for the sentence-length, subword-tokenized sequences of the day (word-piece, Wu
et al. 2016; byte-pair, Sennrich et al. 2015).

The mechanism that already realizes "read from anywhere in one hop" is **attention**, which by
this point is the integral ingredient that made neural machine translation work: it lets a
decoder draw on any source position without regard to distance, at low sequential cost. The
empirical fact that fixes the central numerical risk comes from a large-scale study of
translation architectures (Britz et al. 2017): comparing the two standard scoring rules across
attention widths from 128 to 1024, the multiplicative (dot-product) score is competitive at
widths 128-512 (BLEU around 22 on newstest2013) but **degrades sharply to 18.22 at width
1024**, while the additive score stays near 22 across the whole range. So a dot-product score
is fine at moderate width and breaks at large width; an additive score whose energy passes
through a bounded nonlinearity does not. That contrast — a width-dependent collapse of one
scoring rule but not the other — is the diagnostic that any wide-vector content-addressed read
has to contend with, and it is knowable before our layer exists.

## Baselines

The prior content-addressed reads a new layer would be measured against and react to.

**Additive attention (Bahdanau, Cho & Bengio 2015).** The original soft-alignment mechanism
for translation. For a reader state `s_{i-1}` and each holder annotation `h_j`, score the pair
with a small feed-forward network — a learned energy

```
e_ij = a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j),
```

normalize the energies over holders with a softmax, `alpha_ij = exp(e_ij) / sum_k exp(e_ik)`,
and return the weighted blend `c_i = sum_j alpha_ij h_j` (an "expected annotation"). It works
beautifully and is fully differentiable. **Gap:** the score is an MLP — a matrix multiply, a
`tanh`, and an inner product with `v_a` — evaluated for *every* (reader, holder) pair. It does
not collapse into a single dense matrix multiply, so its constant factor and memory traffic are
high. That is tolerable when attention is a single bolt-on used once per decoder step over the
source; it is the wrong cost profile for a layer meant to run many times inside a deep stack.

**Multiplicative / dot-product attention (Luong, Pham & Manning 2015).** Simplify the score to
a bilinear or plain inner product. Global attention computes `alpha(s) = softmax(score(h_t,
h_s))` with `score` one of `h_t^T h_s` (dot), `h_t^T W_a h_s` (general), or a concat-MLP, then
blends the source states. The dot form is a single matrix multiplication, far cheaper than the
additive MLP, and it expresses compatibility directly as vector alignment. **Gaps:** it is
presented and used **unscaled**, and as shown by the width-1024 collapse above, the raw inner
product's magnitude grows with the vector dimension and pushes the downstream softmax toward a
saturated, tiny-gradient regime as the model gets wide; and like additive attention it is
always **attached to a recurrent decoder**, used as an auxiliary read alongside the RNN rather
than as the standalone, repeatable mixing layer of a non-recurrent stack.

**Softmax normalization (standard).** Given a vector of scores `z`, `softmax(z)_i = exp(z_i) /
sum_j exp(z_j)` maps them to a nonnegative, sum-to-one weighting that is smooth and
differentiable. It is the standard way to turn compatibility scores into a blend. **Property
that becomes the constraint:** its gradient is largest when the inputs are of moderate, similar
magnitude and collapses toward zero once one logit dominates (the distribution goes near
one-hot), so whatever feeds it must keep the logit scale under control.

**Recurrent and convolutional mixing (LSTM/GRU; ByteNet, ConvS2S).** The incumbent
cross-position mechanisms, described above. **Gaps:** recurrence pays `O(n)` sequential ops and
`O(n)` path length; convolution parallelizes but its path length still grows with distance
(`O(n/k)` or `O(log_k n)`), so neither connects all positions in one hop.

## Evaluation settings

The natural yardsticks already in use for sequence transduction. Machine translation is the
standard task: the WMT 2014 English-to-German and English-to-French benchmarks, scored by
**BLEU** on the held-out newstest sets, with sequences tokenized into subword units (byte-pair
encoding, Sennrich et al. 2015; word-piece, Wu et al. 2016) so vocabulary size stays bounded
and sequence lengths stay in the regime where `n < d`. Training is on stacked encoder-decoder
models with residual connections and layer normalization, on multi-GPU machines, with training
wall-clock and parallelizability themselves treated as figures of merit (a layer that removes
the sequential bottleneck is interesting partly because it trains faster). The candidate mixing
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
