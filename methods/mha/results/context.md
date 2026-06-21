# Context: sequence transduction and the cost of recurrence (circa 2014–2017)

## Research question

The dominant problem is neural sequence transduction: map a variable-length input sequence
`(x_1, ..., x_n)` to an output sequence `(y_1, ..., y_m)` — machine translation is the flagship
case, but the same shape covers language modeling, summarization, and parsing. A good layer for
this has to do one thing well: let the representation at every position be informed by the
relevant content at every other position, including far-away ones, because meaning in language
is non-local (agreement, coreference, long-distance dependencies).

The precise goal is a layer that relates any two positions cheaply and in parallel, with a short
maximum path length between any pair of positions. The analytic measures on which layers are
compared are: the number of sequential operations per layer, the maximum path length between any
two positions, and the per-layer computational cost.

## Background

**Recurrence.** Recurrent networks — and in particular LSTM and gated-recurrent variants — are
the established workhorses for sequence modeling and transduction. A recurrent layer maps
`(x_1,...,x_n)` to `(z_1,...,z_n)` by `h_t = f(h_{t-1}, x_t)`, threading a single hidden state
through the sequence. Per layer this costs `O(n · d^2)` (an `n`-long walk, each step a `d×d`
matrix-vector product), with `O(n)` sequential operations and a maximum path length of
`O(|i-j|)` between positions `i` and `j`.

**Convolution.** Convolutional sequence models (ByteNet, ConvS2S) replace recurrence with stacked
1-D convolutions, computing all positions in parallel, so the number of sequential operations
drops to `O(1)`. A single convolution of kernel width `k < n` connects positions within its
receptive field; to connect all pairs you need a stack of `O(n/k)` layers for contiguous kernels,
or `O(log_k n)` for dilated ones. A convolutional layer costs `O(k · n · d^2)`; separable
convolutions reduce this to `O(k · n · d + n · d^2)`.

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
ingredient of competitive translation systems. In nearly all of those systems, the attention is
bolted onto a recurrent (or convolutional) backbone rather than used on its own.

**Self-attention / intra-attention.** Attention has also been turned inward — relating different
positions *of the same sequence* to compute a representation of it — in reading comprehension,
summarization, entailment, and sentence-embedding work (Cheng et al. 2016; Parikh et al. 2016;
Lin et al. 2017), and recurrent attention over an external memory (end-to-end memory networks,
Sukhbaatar et al. 2015) performs well on simple QA and language modeling.

**Diagnostic finding on dot-product scores at large dimension.** The compatibility score `a` can
be computed several ways. A feed-forward (additive) score and a multiplicative (dot-product)
score have similar theoretical cost, but their behavior diverges with the per-position
dimension: an empirical NMT study (Britz et al. 2017) reports that while the two perform
similarly for small key dimension `d_k`, additive attention outperforms raw dot-product attention
as `d_k` grows.

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

**Convolutional sequence models (ByteNet, Kalchbrenner et al. 2016; ConvS2S, Gehring et al.
2017).** Stacked convolutions give `O(1)` sequential operations and full parallelism across
positions, and ConvS2S was a strong translation system.

**Plain encoder-decoder (Sutskever et al. 2014; Cho et al. 2014).** Encode the entire source
into a single fixed-length vector, then decode from it.

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
