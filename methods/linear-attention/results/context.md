# Context

## Research question

Self-attention has become the central computational primitive of sequence
models, but it carries a cost that grows quadratically with the sequence
length. For a sequence of $N$ positions, the attention layer forms an
$N \times N$ matrix of pairwise similarities, then uses it to mix value
vectors. This is $\mathcal{O}(N^2)$ in both time and memory — and the memory
cost is not incidental, because the full similarity matrix has to be retained
in order to backpropagate through it.

Training and full-sequence inference are quadratic, so in practice the usable
context length is bounded: long-range dependencies spanning very long sequences
are harder to exploit. Autoregressive generation is sequential by nature —
each new token is produced from all the tokens before it — and under softmax
attention the per-step cost grows with the number of tokens already generated,
because the new query must be compared against every past key. Producing a
sequence of length $N$ this way costs $\mathcal{O}(N^2)$ overall.

The question is how to design an attention mechanism whose computational cost
scales more favourably with sequence length, while preserving the quality of
the softmax baseline and supporting causal masking for autoregressive training.

## Background

**Self-attention.** A sequence of $N$ feature vectors $x \in \mathbb{R}^{N
\times F}$ is projected into queries, keys and values $Q = xW_Q$, $K = xW_K$,
$V = xW_V$. The attention output places, at each position, a weighted average
of all value vectors, weighted by a similarity between that position's query
and every key:
$$ V' = \text{softmax}\!\left(\frac{QK^T}{\sqrt{D}}\right) V . $$
The softmax is applied rowwise. The scaling by $1/\sqrt{D}$ keeps the variance
of the dot products $\sim\mathcal{O}(1)$ so the softmax does not saturate into a
near-one-hot regime where gradients vanish. A full transformer layer wraps this
in a residual connection and a position-wise two-layer feed-forward network
$f_l$: $T_l(x) = f_l(A_l(x) + x)$. The feed-forward part acts on each position
independently; attention is the only component that mixes across positions, and
it is the only quadratic component.

**The cost, precisely.** Computing $QK^T$ is $\mathcal{O}(N^2 D)$ and yields an
$N \times N$ matrix; multiplying by $V$ is $\mathcal{O}(N^2 M)$. The matrix must
be stored for the backward pass, so memory is $\mathcal{O}(N^2)$ as well. This
is a measured, structural fact about softmax attention, not a contingent
implementation detail.

**Autoregressive decoding.** With causal masking, position $i$ may attend only
to positions $j \le i$. Training is fully parallel: the whole ground-truth
sequence is known, so all positions are processed at once with a triangular
mask, which is exactly why transformers train faster than recurrent networks.
Inference is the opposite — the output at step $i$ is the input at step $i+1$,
so it cannot be parallelized, and the cost of the attention at step $i$ grows
with $i$ because the new query is compared against all previously emitted keys.
Even when past keys and values are cached, the per-step work still grows with
the context.

**The kernel view of attention.** Attention can be read as a kernel smoother:
for any similarity function the output is
$$ V'_i = \frac{\sum_{j=1}^N \text{sim}(Q_i, K_j)\, V_j}
               {\sum_{j=1}^N \text{sim}(Q_i, K_j)} , $$
which reduces to softmax attention when $\text{sim}(q,k) = \exp(q^Tk/\sqrt{D})$.
This kernel framing (Tsai et al., 2019) makes clear that softmax is just one
admissible choice: the only thing the similarity must satisfy for this to be a
well-defined weighted average is that it be non-negative, which includes any
kernel $k(x,y): \mathbb{R}^{2\times F} \to \mathbb{R}_+$.
Polynomial and RBF-kernel variants of attention have been shown to perform
comparably to the exponential one.

**Linearizing softmax elsewhere.** softmax over a large number of categories
has long been a bottleneck in classification (hierarchical and sampled softmax;
Goodman 2001; Morin & Bengio 2005; Mnih & Teh 2009). A recent line (Blanc &
Rippel, 2017; Rawat et al., 2019) approximates the exponential by a dot product
of feature maps, $\exp(u^Tv) \approx \phi(u)^T\phi(v)$, so that the expensive
normalization can be sampled efficiently.

**Recurrent networks.** Before attention, sequence models were recurrent: an
RNN or LSTM (Hochreiter & Schmidhuber, 1997) carries a fixed-size hidden state,
updates it from the current input in constant time per step, and reads its
prediction off that state. Their per-step cost is constant and their memory is
independent of how much sequence has been seen. Recurrent networks were largely
displaced by attention on quality grounds.

## Baselines

**Full softmax attention (Vaswani et al., 2017).** The reference mechanism,
$V' = \text{softmax}(QK^T/\sqrt{D})V$. Strong quality, fully parallel training,
$\mathcal{O}(N^2)$ time and memory, and $\mathcal{O}(N^2)$ to generate a
full sequence autoregressively. This is the quality bar.

**Sparse Transformer (Child et al., 2019).** Replaces the dense attention
matrix with sparse factorizations — each position attends to a structured
subset of others — reducing complexity to $\mathcal{O}(N\sqrt{N})$ and enabling
generative modelling of long sequences.

**Reformer (Kitaev et al., 2020).** Uses locality-sensitive hashing to bucket
queries and keys that are likely to have high similarity, computing dot
products only within buckets, for $\mathcal{O}(N\log N)$ complexity. To hash
queries and keys into the same buckets it forces the keys to be identical to
the queries.

**Context-extension methods.** Transformer-XL (Dai et al., 2019) caches and
attends to representations from previous segments to learn dependencies past a
fixed window, and adaptive-span attention (Sukhbaatar et al., 2019) learns a
per-head attention span.

**Recurrent baselines (LSTM).** Constant per-step cost and bounded memory at
generation time, lower quality on these tasks than attention and sequential
to train.

## Evaluation settings

The natural yardsticks are:

- **Synthetic.** A sequence-duplication / copy task with causal masking (a short
  alphabet plus a separator symbol, sequences up to length 128), used to check
  that an attention variant converges as cleanly as softmax. Plus direct
  micro-benchmarks of peak GPU memory and wall-clock time for an
  attention-plus-gradient computation across sequence lengths $N \in \{2^9,
  \dots, 2^{16}\}$, batch size scaled inversely with length, time and memory
  reported per sample.
- **Autoregressive image generation.** Predicting images pixel by pixel and
  reporting bits-per-dimension, on MNIST ($784$ positions) and CIFAR-10
  (over $3000$ positions, with a per-pixel mixture-of-logistics output head).
  Generation throughput (images/second) and single-image latency on CPU and GPU
  are the efficiency metrics of interest.
- **Automatic speech recognition.** Frame-level phoneme prediction on the
  $80$-hour WSJ corpus with $40$-dimensional mel filterbank features (sequences
  averaging $\sim 800$ frames, up to $2400$), trained with the CTC loss,
  reported as phoneme error rate and training time per epoch — a
  non-autoregressive setting to confirm the mechanism is general.

Standard optimizers and schedules of the time (Adam / RAdam) are assumed.

## Code framework

The primitives that already exist: an attention layer that owns the
query/key/value projections and the output projection and delegates the actual
mixing to a pluggable inner module; a softmax attention as the baseline inner
module; a feed-forward block and residual wrapper forming a transformer layer;
and a standard training loop. The inner mixing module is left as an open slot to
be filled in.

```python
import torch
from torch.nn import Module, Linear

class AttentionLayer(Module):
    def __init__(self, inner_attention, d_model, n_heads,
                 d_keys=None, d_values=None, d_model_keys=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        d_model_keys = d_model_keys or d_model
        self.inner_attention = inner_attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask,
                query_lengths, key_lengths):
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        q = self.query_projection(queries).view(N, L, H, -1)
        k = self.key_projection(keys).view(N, S, H, -1)
        v = self.value_projection(values).view(N, S, H, -1)
        out = self.inner_attention(
            q, k, v, attn_mask, query_lengths, key_lengths
        ).view(N, L, -1)
        return self.out_projection(out)


class FullAttention(Module):
    def __init__(self, softmax_temp=None):
        super().__init__()
        self.softmax_temp = softmax_temp

    def forward(self, queries, keys, values, attn_mask,
                query_lengths, key_lengths):
        from math import sqrt
        N, L, H, E = queries.shape
        temp = self.softmax_temp or 1. / sqrt(E)
        QK = torch.einsum("nlhe,nshe->nhls", queries * temp, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        if not key_lengths.all_ones:
            QK = QK + key_lengths.additive_matrix[:, None, None]
        A = torch.softmax(QK, dim=-1)
        return torch.einsum("nhls,nshd->nlhd", A, values).contiguous()


class CandidateAttention(Module):
    # The inner mixing rule to be designed, plugged in where FullAttention is.
    def __init__(self, query_dimensions, eps=1e-6):
        super().__init__()
        pass  # TODO

    def forward(self, queries, keys, values, attn_mask,
                query_lengths, key_lengths):
        pass  # TODO
```
