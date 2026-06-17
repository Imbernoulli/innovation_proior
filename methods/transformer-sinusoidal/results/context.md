## Research question

A sequence model built entirely out of self-attention and per-position feed-forward
layers has one structural property that becomes a problem the moment you remove
recurrence and convolution: it has no notion of order. Self-attention computes each
output as a content-based weighted average over all positions, and that average is
**permutation-equivariant** — permute the input tokens and the outputs come out
permuted the same way, with identical values. A per-position feed-forward layer acts on
each position independently and cannot see order either. So a stack of these layers sees
its input as a *set* of token vectors, not a sequence: "the cat sat" and "sat cat the"
are indistinguishable to it. Yet the task is sequence transduction, where order is
meaning.

The precise problem is therefore: design a representation of a token's position in the
sequence that can be injected into a width-`d_model` model so that order becomes
recoverable, subject to several constraints that a usable scheme must meet at once.
(1) It has to be a fixed-width-`d_model` object so it can ride alongside the token
embeddings without changing the rest of the architecture. (2) Its magnitude must stay
bounded as the position index grows, so that long sequences don't drive activations
outside the range the network was trained in. (3) Distinct positions need distinct
codes, so the model can in principle tell any two positions apart. (4) It should make
*relative* position easy to use, because what matters linguistically is usually "how far
back" a token is, not its absolute index — and that relationship should look the same
everywhere in the sequence. (5) Ideally it should be defined for positions **beyond the
longest sequence seen in training**, so the model has a chance to run on longer inputs at
test time. No scheme in use at the time meets all five together.

## Background

The field state is the move *away* from recurrence. The established sequence-transduction
recipe is the recurrent encoder-decoder (Sutskever et al. 2014; Bahdanau et al. 2014;
Cho et al. 2014), where order is never a separate concern: a recurrent net consumes
tokens one at a time, `h_t = f(h_{t-1}, x_t)`, so position is encoded implicitly in *when*
a token is processed. Order is free, but it is bought with a sequential dependency along
the time axis that cannot be parallelized within an example — the binding throughput
constraint at long sequence lengths.

The reaction has been to replace the sequential recurrence with operations that compute
all positions in parallel. Convolutional sequence models — the Extended Neural GPU
(Kaiser & Sutskever 2016), ByteNet (Kalchbrenner et al. 2017), ConvS2S (Gehring et al.
2017) — do this with stacked convolutions; attention-based mechanisms (Bahdanau et al.
2014; Luong et al. 2015) route information between arbitrary positions in one hop. The
load-bearing concept underneath all of this is **attention**: an output is a weighted sum
of values, the weights a softmax of a compatibility score between a query and each key.
The crucial structural fact about it — the thing that creates the present problem — is
that the score depends only on the *contents* of the query and key vectors, not on where
they sit. Permute the keys and values and the softmax weights permute with them and the
output is unchanged. Attention dissolved the long-range-dependency problem (any position
reaches any other in one hop) precisely *because* it is blind to distance and order; that
same blindness is what now has to be repaired by something external.

There is a known diagnostic about position counters that any candidate scheme has to
respect. A raw integer index `t` used as a feature is unbounded: it keeps growing with
sequence length, so a model trained on short sequences sees position values at test time
it was never exposed to, and large-magnitude inputs destabilize the downstream linear
layers. Normalizing the index into `[0, 1]` by dividing by the sequence length removes the
unboundedness but breaks consistency: the same step of one position corresponds to a
different numeric delta in a short sequence than in a long one, so "one token later" has
no fixed meaning across examples. Both of these are pre-method facts about how naive
position counters behave, independent of any particular architecture.

## Baselines

**Implicit position via recurrence (Sutskever et al. 2014; Bahdanau et al. 2014).** In a
recurrent encoder-decoder, position is not represented at all — it is carried by the order
of computation, since state `t` is produced after state `t-1`. Core idea: `h_t =
f(h_{t-1}, x_t)`, with the decoder attending over encoder states. Gap: the very mechanism
that supplies order, the step-by-step recurrence, is the sequential bottleneck the field
is trying to eliminate. Take recurrence away and the order information vanishes with it,
leaving nothing in its place.

**Learned absolute position embeddings (Gehring et al. 2017, ConvS2S).** The first clean
"position as an addable vector" scheme. Keep a learned table `p_1, …, p_L ∈ ℝ^d`, one
trainable vector per absolute index up to a maximum length `L`, and add `p_t` to the token
embedding at position `t`: the input to position `t` is `embed(x_t) + p_t`. Convolutions
only see a local window, so without this the model cannot recover absolute order; the
added embeddings give it back, at negligible cost (a single addition). This is the direct
template — order injected as a width-`d` vector summed into the embeddings. Gaps: (1) the
table is only defined for indices `1…L`; position `L+1` has no vector, so the scheme has
literally nothing to say about any position longer than the longest training sequence — it
cannot extrapolate. (2) Each `p_t` is learned independently of every other, so there is no
built-in relationship between the code for position `t` and the code for position `t+k`;
any regularity relating nearby positions has to be discovered from data rather than
guaranteed by the construction, and a relationship learned at small indices need not hold
at large ones.

**A single bounded periodic counter.** A natural fix for the unbounded-index problem is to
pass the index through one bounded periodic function, e.g. `sin(ω t)`, giving a value in
`[-1, 1]` for every `t`. Core idea: boundedness and definedness everywhere come for free
from periodicity. Gap: a single sinusoid aliases — `sin(ω t)` returns to the same value
every period, so distinct positions collide and become indistinguishable, and a single
scalar channel carries far too little information to identify a position among hundreds.

## Evaluation settings

The natural yardstick at the time is machine translation on the **WMT 2014** shared task,
English→German (newstest2014 test set; ~4.5M training sentence pairs) and the larger
English→French (~36M pairs), scored by **BLEU** against reference translations, with
training cost (wall-clock / accelerator-hours to reach a quality) tracked because the
whole motivation is parallel throughput. Inputs are subword units — byte-pair encoding
(Sennrich et al. 2015) with a shared ~37k source-target vocabulary for EN-DE, word-piece
(Wu et al. 2016) with ~32k for EN-FR — which keeps the sequence length `n` (tens to low
hundreds of tokens) below the model width `d`. A position scheme's headline question —
whether it lets a model behave sensibly on sequences *longer* than those seen in training
— has as its natural yardstick a setup where a model is trained at one length and then run
at strictly longer lengths and scored for whether quality holds up.

## Code framework

The position scheme plugs into a decoder-only causal sequence model whose every other part
already exists. The substrate is: a token embedding table; a stack of pre-norm decoder
blocks, each a causal-masked multi-head self-attention sub-layer (scaled dot-product
scores with a `-∞` upper-triangular mask, so position `t` only attends to `≤ t`) plus a
position-wise feed-forward sub-layer, both wrapped in residual connections; a final norm;
and a tied output projection to the vocabulary. The attention scores depend only on token
content, so as written the whole stack is order-blind — the one empty slot is *how order
information enters the model*. The slot can be filled either by producing a per-position
vector that is added to the token embeddings at the bottom of the stack, or by producing
an additive bias on the attention scores; the scaffold exposes both hooks and leaves their
bodies empty.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionScheme:
    """Holds the optional hooks by which order information enters the model.
    At least one hook must be non-None, otherwise the model is order-blind."""

    def __init__(self, token_add=None, attn_bias=None, extra_modules=None):
        # token_add(positions) -> [..., d_model] additive position vector, or None
        self.token_add = token_add
        # attn_bias(T, device, dtype) -> [n_heads or 1, T, T] additive score bias, or None
        self.attn_bias = attn_bias
        # learnable params, if any, registered so the optimizer sees them
        self.extra_modules = extra_modules or nn.ModuleList()


def build_position_scheme(d_model, max_len) -> PositionScheme:
    # Empty order-signal hook.
    return PositionScheme(token_add=None, attn_bias=None)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores + attn_mask        # causal mask (+ optional position bias)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class SeqModel(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers, max_len, scheme):
        super().__init__()
        self.d_model = d_model
        self.tok = nn.Embedding(vocab, d_model)
        self.scheme = scheme
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight     # tied embeddings

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.tok(tokens)
        # Empty order-signal slot: add a token-level signal and/or attention bias here.
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=tokens.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal)
        return self.head(self.ln_f(x))
```

The single unfilled slot is the position signal: `build_position_scheme` and the marked
line in `forward` are the unresolved design surface.
