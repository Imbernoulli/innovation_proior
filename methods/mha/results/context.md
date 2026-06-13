# Context: sequence transduction and the cost of recurrence (circa 2014–2017)

## Research question

The dominant problem is neural sequence transduction: map a variable-length input sequence
`(x_1, ..., x_n)` to an output sequence `(y_1, ..., y_m)` — machine translation is the flagship
case, but the same shape covers language modeling, summarization, and parsing. A good layer for
this has to do one thing well: let the representation at every position be informed by the
relevant content at every other position, including far-away ones, because meaning in language
is non-local (agreement, coreference, long-distance dependencies). Two costs make the existing
solutions painful at scale.

First, **training throughput**. The state-of-the-art layer factors its computation along the
sequence: it produces a hidden state at position `t` as a function of the hidden state at
position `t-1` and the input at `t`. That dependency chain is inherently sequential — you cannot
compute position `t` until position `t-1` is done — so within a single training example there is
nothing to parallelize. At long sequence lengths, memory limits how many examples you can batch
together, so the sequential bottleneck directly throttles training over large corpora, exactly
when you most want to scale.

Second, **the length of the path a signal must travel** between two positions that depend on
each other. The easier it is for forward and backward signals to traverse the network between
two positions, the easier it is to learn the dependency between them; long paths make
long-range dependencies hard to learn because gradients have many steps to survive. In a layer
that walks the sequence one step at a time, information from position `j` reaches position `i`
only after `|i - j|` steps.

The precise goal, then, is a layer that simultaneously: (1) relates any two positions with a
number of *sequential* operations that does not grow with the sequence length or the distance
between them; (2) keeps the maximum path length between any pair of positions short and roughly
constant; (3) is fully parallelizable across positions within an example; and (4) is
computationally affordable — its per-layer cost must be competitive with the recurrent and
convolutional layers it would replace. Each existing layer below achieves some of these and
pays on the others.

## Background

**Recurrence.** Recurrent networks — and in particular LSTM and gated-recurrent variants — are
the established workhorses for sequence modeling and transduction. A recurrent layer maps
`(x_1,...,x_n)` to `(z_1,...,z_n)` by `h_t = f(h_{t-1}, x_t)`, threading a single hidden state
through the sequence. Per layer this costs `O(n · d^2)` (an `n`-long walk, each step a `d×d`
matrix-vector product), but the number of *sequential* operations is `O(n)`, and the maximum
path length between positions `i` and `j` is `O(|i-j|)` — both grow with the sequence. The first
is the throughput wall; the second is the long-range-dependency wall.

**Convolution.** Convolutional sequence models (ByteNet, ConvS2S) replace recurrence with stacked
1-D convolutions, computing all positions in parallel, so the number of sequential operations
drops to `O(1)`. But a single convolution of kernel width `k < n` only connects positions within
its receptive field; to connect *all* pairs you need a stack of `O(n/k)` layers for contiguous
kernels, or `O(log_k n)` for dilated ones, so the maximum path length still grows with distance.
A convolutional layer also costs a factor of `k` more than a recurrent one, `O(k · n · d^2)`;
separable convolutions reduce this to `O(k · n · d + n · d^2)`.

**Attention as a bridge over distance.** The crucial prior idea is the attention mechanism,
introduced to remove the fixed-vector bottleneck of plain encoder-decoder models. Instead of
compressing the whole source into one vector, the decoder forms, at each output step `i`, a
context vector as a *content-weighted sum* of all encoder annotations: `c_i = Σ_j α_ij h_j`,
where the weights `α_ij = softmax_j(e_ij)` come from a learned compatibility score
`e_ij = a(s_{i-1}, h_j)` between the current decoder state and each source position. Because the
weights are produced by a differentiable softmax, gradients flow through the soft alignment and
the whole thing trains end to end. The decisive structural property: this connects output
position `i` to *any* input position `j` through a single weighted sum — a path of constant
length, independent of `|i-j|`, computed for all `j` in parallel. Attention layers, used as the
cross-positional channel inside otherwise-recurrent models, are by this time an essential
ingredient of competitive translation systems. In nearly all of those systems, though, the
attention is bolted onto a recurrent (or convolutional) backbone rather than used on its own.

**Self-attention / intra-attention.** Attention has also been turned inward — relating different
positions *of the same sequence* to compute a representation of it — in reading comprehension,
summarization, entailment, and sentence-embedding work (Cheng et al. 2016; Parikh et al. 2016;
Lin et al. 2017), and recurrent attention over an external memory (end-to-end memory networks,
Sukhbaatar et al. 2015) performs well on simple QA and language modeling. These show attention
can stand in for sequence-aligned recurrence in pieces of a model.

**Diagnostic finding on dot-product scores at large dimension.** The compatibility score `a` can
be computed several ways. A feed-forward (additive) score and a multiplicative (dot-product)
score have similar theoretical cost, but their behavior diverges with the per-position
dimension: an empirical NMT study (Britz et al. 2017) reports that while the two perform
similarly for small key dimension `d_k`, additive attention outperforms raw dot-product attention
as `d_k` grows. Raw dot-product magnitude is therefore a practical concern when choosing a
high-dimensional score function.

## Baselines

These are the prior layers a new cross-positional layer would be measured against and would
react to.

**Recurrent encoder-decoder with attention (Bahdanau et al. 2014; Luong et al. 2015).** A
bidirectional RNN encodes the source into per-position annotations `h_j`; a recurrent decoder
emits the target, and at each step `i` reads from the source through attention. Bahdanau's
*additive* alignment model parametrizes the score as a one-hidden-layer feed-forward net,
`e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)`, then `α_ij = softmax_j(e_ij)`, `c_i = Σ_j α_ij h_j`.
Luong's *multiplicative* family keeps the same weighted-sum structure but offers cheaper scores:
`dot: s_t^T h̄_s`, `general: s_t^T W_a h̄_s`, and a `concat` variant equivalent to the additive
form. The dot score is just an inner product between the decoder state and each source state.
**Gap:** the cross-positional channel is excellent, but it rides on an RNN, so the model as a
whole keeps the recurrent layer's `O(n)` sequential operations and `O(n)` path length on the
self-sequence side — the parallelization and long-range-on-the-same-sequence problems are
untouched. The additive score also evaluates a small MLP for every query-key pair, which does
not map onto a single dense matrix multiply.

**Convolutional sequence models (ByteNet, Kalchbrenner et al. 2016; ConvS2S, Gehring et al.
2017).** Stacked convolutions give `O(1)` sequential operations and full parallelism across
positions, and ConvS2S was a strong translation system. **Gap:** the maximum path length between
two positions still grows with their distance (`O(n/k)` or `O(log_k n)`), so distant
dependencies remain comparatively hard, and the per-layer cost carries the extra factor of `k`.

**Plain encoder-decoder (Sutskever et al. 2014; Cho et al. 2014).** Encode the entire source
into a single fixed-length vector, then decode from it. **Gap:** one fixed vector is a hard
bottleneck for long inputs; this is precisely what attention was introduced to remove, so it is
a baseline the attention mechanisms above already improve on.

## Evaluation settings

The natural yardsticks already in use for this problem:

- **Machine translation benchmarks.** WMT 2014 English-to-German and English-to-French, the
  standard large-scale translation corpora; quality reported as BLEU on the standard test sets
  (newstest), with a development set (newstest2013) for tuning. Sub-word vocabularies are
  standard — byte-pair encoding (Sennrich et al. 2015) and word-piece (Wu et al. 2016) — which
  keep the sequence representation dimension `d` comparable to or larger than typical sentence
  lengths `n`.
- **Training protocol.** Encoder-decoder models trained with token-level cross-entropy, on
  accelerator hardware, with optimizer schedules and regularization treated as tunable settings;
  throughput measured as wall-clock time under a matched benchmark setup.
- **Layer-level analysis.** Per-layer computational complexity, the minimum number of sequential
  operations, and the maximum path length between any two positions — the analytic yardstick on
  which recurrent, convolutional, and attention layers are directly compared (`n` is sequence
  length, `d` representation dimension, `k` kernel width).

## Code framework

A standard sequence-model harness supplies token and position
embeddings, repeated residual blocks around a cross-positional sub-layer and a position-wise
feed-forward sub-layer, normalization, and a final projection to vocabulary logits trained by
cross-entropy. The substrate exposes one empty slot: how each position's representation is updated
using positions in the sequence.

```python
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class SequenceMixing(nn.Module):
    """The cross-positional sub-layer: update each position's representation
    using information from positions in the sequence. The internal mechanism is what
    we will design; it must connect any two positions cheaply and in parallel."""

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        # TODO: the cross-positional mechanism we will design, and whatever
        #       projections / parameters it needs.

    def forward(self, x):                      # x: (batch, seq_len, n_embd)
        # TODO: return a (batch, seq_len, n_embd) tensor in which each position
        #       has been updated from positions in the sequence.
        raise NotImplementedError


class FeedForward(nn.Module):
    """Existing position-wise sub-layer: same MLP applied independently per position."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    """Existing residual block: norm -> sub-layer -> add, twice."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mix = SequenceMixing(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.mix(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class SequenceModel(nn.Module):
    """Existing harness: embed tokens (+ positions), stack blocks, project to logits."""

    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.vocab_proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.vocab_proj(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

The harness supplies the embeddings, the residual/feed-forward scaffolding, and the training
objective; the single empty slot is `SequenceMixing` — how a position's representation gets
updated from positions in the sequence.
