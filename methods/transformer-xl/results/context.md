# Context

## Research question

The task is autoregressive language modeling: given a corpus of tokens
$\mathbf{x}=(x_1,\dots,x_T)$, estimate the joint probability through the chain
factorization $P(\mathbf{x})=\prod_t P(x_t\mid x_{<t})$, so the problem reduces
to modeling each next-token conditional from its left context. The standard
neural recipe encodes the context $x_{<t}$ into a fixed-size hidden state, which
is multiplied with the word embeddings to produce logits, and a softmax turns
those into a categorical distribution over the next token.

The hard part is *long-term dependency*. Many predictions hinge on information
that appeared hundreds or thousands of tokens earlier (the subject of a clause,
a name introduced pages ago, a topic established at the start of a document).
The question is how to encode an *arbitrarily long* context into the
representation used for the current token, under a fixed compute and memory
budget, so that the *effective* dependency length extends well past a few hundred
tokens while keeping training and evaluation tractable.

## Background

**Recurrent models.** Recurrent neural networks, in particular the LSTM
(Hochreiter & Schmidhuber 1997), have been the default solution. They carry a
hidden state forward step by step, so in principle nothing bounds how far back
information can travel. Training by backpropagation through time multiplies many
Jacobians together, and the product either vanishes or explodes
(Hochreiter et al. 2001), so gradients carrying long-range credit shrink toward
zero. Gating (the LSTM cell) and gradient clipping (Graves 2013) soften this.
A diagnostic measurement makes the effective range concrete: a trained LSTM
language model uses on average only about 200 words of context
(Khandelwal et al. 2018) — feeding it more history beyond that barely changes
its predictions.

**Self-attention as the alternative.** Attention (Bahdanau et al. 2014;
Vaswani et al. 2017) builds *direct* connections between any pair of positions:
the representation of one position is a weighted sum over all others, with the
weight given by a query-key dot product. Because any two positions are one hop
apart, the gradient path between distant tokens is short and constant-length
rather than growing with their separation, so the vanishing-gradient pressure
that limits recurrent models is largely absent.

**The fixed-length-segment regime.** To apply a self-attention decoder to a
long corpus under a finite budget, the standard move is to chop the corpus into
separate segments of a few hundred tokens and train the model independently
within each segment, with no information passing across segment boundaries in
either the forward or backward pass (Al-Rfou et al. 2018 trained deep
character-level decoders this way and beat LSTMs by a wide margin). At test
time, one can slide the window forward by a single position at each step and
re-encode the entire segment from scratch so the predicted token always sits at
the end of a full-length context.

**Absolute positional encoding.** A self-attention layer is permutation-
invariant in its inputs, so order has to be injected explicitly. The standard
device (Vaswani et al. 2017) is a set of absolute positional encodings
$\mathbf{U}\in\mathbb{R}^{L_\text{max}\times d}$ whose $i$-th row encodes
absolute position $i$ within the sequence, built from sinusoids of geometrically
spaced frequencies, $\text{inv\_freq}_k = 1/10000^{2k/d}$, with the even
dimensions holding $\sin$ and the odd dimensions $\cos$. This encoding is added
to the word embeddings once, at the input, before any attention layer. The
sinusoidal form is chosen deliberately: a fixed linear map sends the encoding of
position $i$ to that of position $i+\Delta$, so relative offsets are expressible
and the scheme extends past lengths seen in training.

**Carrying state across segment boundaries in recurrent training.** Recurrent
language models face the same "the corpus is longer than what fits" problem and
solve it with truncated backpropagation through time (Mikolov et al. 2010): the
last hidden state of the previous segment is passed into the next segment as a
fixed (gradient-free) input, so the forward pass carries history across the
boundary while the backward pass stays inside the current segment. This lets
recurrent models exploit context well beyond a single training segment, though
it passes only a single summary vector forward.

## Baselines

**Vanilla fixed-segment self-attention decoder (Al-Rfou et al. 2018).** A deep
stack of masked (causal) self-attention layers trained independently on
fixed-length segments. Within a segment, layer $n$ computes, for each position,
queries/keys/values $q,k,v$ by linear projections of the previous layer's
hidden states, an attention score $q_i^\top k_j$ scaled by $1/\sqrt{d_k}$, a
causal-masked softmax over $j\le i$, a weighted sum of values, a residual add
with layer normalization, and a position-wise feed-forward sublayer. Absolute
sinusoidal encodings are added at the input.

**LSTM language model (Hochreiter & Schmidhuber 1997).** A recurrent cell with
input/forget/output gates carries a state $c_t,h_t$ forward;
$h_t$ is read out into a softmax over the vocabulary. A gated additive state
path keeps gradients from vanishing as fast as in a plain RNN.

**Truncated BPTT for recurrent LMs (Mikolov et al. 2010).** Pass the last
hidden state of the previous segment forward as a fixed input, reusing history
across segment boundaries cheaply.

**Relative position in attention for translation/music (Shaw et al. 2018;
Huang et al. 2018).** Inject the relative distance between query and key into the
attention computation rather than adding absolute positions at the input. Shaw et
al. add a learned per-distance vector to the keys, effectively keeping a content-
content term and one content-position term while folding the position projection
into a single trainable per-distance matrix.

## Evaluation settings

The natural yardsticks are standard language-modeling benchmarks spanning
word-level and character-level granularity: WikiText-103 and Penn Treebank
(word-level perplexity), One Billion Word (word-level, predominantly short-range,
sentence-shuffled), and enwik8 and text8 (character-level, scored in bits per
character). The metric is per-token negative log-likelihood reported as
perplexity (word) or bits-per-character (character). Beyond raw likelihood, the
quantity of interest is the *effective* context length — how far back added
context still measurably lowers loss — measured by gradually extending the
context/attention span at test time until the loss stops improving, and by
comparing the loss of models given a short versus a long context on the same
positions. Evaluation cost (how much computation each prediction requires given
its context budget) is itself a reported quantity.

## Code framework

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- existing primitives ---------------------------------------------------

class PositionwiseFF(nn.Module):
    """Per-position MLP sublayer: Linear -> ReLU -> Linear, residual + LayerNorm."""
    def __init__(self, d_model, d_inner, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_inner, d_model), nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layer_norm(x + self.net(x))


class CausalSelfAttention(nn.Module):
    """Standard masked self-attention: q,k,v projections, scaled dot-product
    score q_i . k_j, causal softmax, weighted value sum, output proj, residual+LN."""
    def __init__(self, n_head, d_model, d_head, dropout):
        super().__init__()
        self.n_head, self.d_head = n_head, d_head
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / (d_head ** 0.5)

    def forward(self, h, attn_mask=None):
        raise NotImplementedError


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(n_head, d_model, d_head, dropout)
        self.ff = PositionwiseFF(d_model, d_inner, dropout)

    def forward(self, x, attn_mask=None):
        return self.ff(self.attn(x, attn_mask=attn_mask))


# --- to be filled in -------------------------------------------------------

class PositionEncoding(nn.Module):
    """How sequence order enters the model. The existing recipe adds an absolute
    sinusoidal vector to the input embeddings."""
    def __init__(self, d_model):
        super().__init__()
        pass

    def forward(self, *args, **kwargs):
        pass


class LanguageModel(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner, dropout):
        super().__init__()
        self.word_emb = nn.Embedding(n_token, d_model)
        self.pos = PositionEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(n_head, d_model, d_head, d_inner, dropout)
            for _ in range(n_layer)
        ])
        self.out = nn.Linear(d_model, n_token)

    def forward(self, data, target, *carry):
        # TODO: turn a stream of fixed-length segments into per-token logits.
        raise NotImplementedError


# --- training loop ----------------------------------------------------------

def train(model, segment_iter, optimizer, n_steps):
    carry = tuple()
    for data, target in segment_iter:           # consecutive fixed-length chunks
        ret = model(data, target, *carry)
        loss, carry = ret[0], ret[1:]
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step(); optimizer.zero_grad()
```
