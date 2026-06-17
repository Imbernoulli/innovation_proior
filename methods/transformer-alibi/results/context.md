# Context: position representation and length extrapolation in transformer LMs (circa 2021)

## Research question

When a transformer language model is built, one design number dominates its cost: the length
`L` of the training input subsequence. The self-attention sublayer compares every token with
every other token in its input, so both compute and memory grow quadratically in `L`. Longer
inputs give each prediction more left context and lower the loss, but they make every training
step dramatically more expensive. The wish is to have it both ways: train on short
subsequences (cheap), then at inference feed sequences *longer* than `L` and still predict
well — ideally better, because more context should help.

Call this ability **extrapolation**: a model trained at length `L` keeps (and hopefully
improves) its held-out perplexity when evaluated at length `L_valid > L`. The recurrent
language models that preceded transformers were routinely trained at short lengths and simply
run on longer contexts at test time, and this was assumed to generalize; the transformer was
hoped to inherit the same property. The precise problem: does a transformer LM trained at `L`
keep its perplexity when scored at `L_valid > L`, and if not, *why* not — and can the
position-representation mechanism be changed so that it does, **without** spending any extra
runtime, memory, or parameters relative to the cheapest existing position method?

A solution has to clear three bars simultaneously: (1) it must genuinely extrapolate to much
longer inputs; (2) it must be no slower and no more memory-hungry than the cheapest current
position method; (3) ideally it adds no learned parameters, so a single recipe transfers
across model sizes and datasets without retuning.

## Background

A transformer layer's functions are agnostic to input length. The embedding lookup, the
position-wise feedforward sublayer, and the softmax classifier act independently per vector;
the attention sublayer's parameters do not depend on length and it already must handle
variable-length inputs (because of causal masking). So a layer maps an arbitrary number of
input vectors to the same number of output vectors, and *nothing* in the parameters cares how
many tokens arrived. Causal (left-to-right) language modeling is enforced by adding a **causal
mask** — an upper-triangular matrix of `-inf` — to the attention scores before the softmax, so
position `i`'s prediction sees only tokens `1..i`. Concretely, for query `q_i` and the first
`i` keys `K`, the attention sublayer computes `softmax(q_i K^T)` and uses those weights to
average the value vectors; the standard implementation also divides the scores by `sqrt(d_k)`,
the head dimension, to keep the dot-product variance in a sane range.

Because the layers are length-agnostic, sequential **order has to be injected explicitly**.
The original recipe adds a position signal to the token embeddings at the bottom of the
network, so two identical tokens at different positions enter the stack as different vectors.
Everything the model learns about "where" a token is, it learns from how this injected signal
behaves over the positions `[0, L)` that appear during training.

**Inference on long text.** Training and perplexity evaluation process many predictions at
once under the causal mask. A document longer than `L` is handled by cutting it into
`L`-length pieces scored independently — **nonoverlapping inference**; a prediction then sees
context only back to the start of its own segment. An alternative, **sliding-window
inference**, advances a width-`L` window by a small stride `S`, re-encoding `L-S` tokens each
pass so that every prediction has up to `L` left-context tokens. It is far more accurate at
segment boundaries but much slower, since the same tokens are re-encoded many times.

**The early token curse.** Under nonoverlapping inference, the predictions near the *start* of
each segment have very little left context (the relevant tokens sit at the end of the
discarded previous segment). These context-starved predictions carry high loss and inflate the
reported perplexity. Feeding longer inputs at inference dilutes the curse: with a larger
`L_valid`, a smaller *fraction* of predictions are context-starved. So a model that can merely
*accept* longer inputs gains perplexity even if it learns nothing new about long-range
structure — provided it does not break when handed more tokens than it trained on.

**The diagnostic finding that frames everything.** A transformer LM with the standard added
position signal, trained at `L` and evaluated at `L+k`, improves perplexity only for very
small `k` (a few dozen tokens), holds briefly, then *degrades sharply* as `k` grows. The model
does not gracefully use the extra length; it breaks. Holding architecture, data, seed, and
training budget fixed and changing *only* the position method changes this behavior, which
pins the failure on the position representation rather than on the transformer as such. The
mechanism is traceable: at positions beyond `L`, the position signal the model is asked to
interpret is one it never saw in training — it is out of distribution, and the responses the
model tuned to the signals in `[0, L)` do not apply.

A second, architectural observation matters for the design space. In the original
added-signal scheme, position information mixes into the *values* and so flows into every
layer's output. In some later relative schemes, position influences only the query–key
comparison (the attention weights) and never the values, so the attention output — a weighted
average of value vectors — carries no explicit absolute-position component.

## Baselines

These are the prior position methods a new method would be measured against and would react
to. Each is given with its actual mechanism and the specific limitation it leaves open.

**Learned absolute position embeddings.** Store a trainable vector for each position `0..L-1`
and add it to the token embedding there. Simple and effective inside the training range. The
limitation for this goal: there is no vector for any position `>= L`, so the method has *no
defined behavior* beyond the training length.

**Sinusoidal position embeddings (Vaswani et al., 2017, §3.5).** Replace the learned
per-position vectors with fixed, non-learned vectors whose coordinates are sines and cosines
of the position at a geometric range of frequencies, added to the token embeddings at the
input. Parameter-free, cheap, and — being a function defined for every position — it *can* in
principle produce a signal past `L`; Vaswani et al. speculated it might therefore extrapolate.
Empirically it barely does: trained at `L`, perplexity improves for only the first ~20–50
extra tokens, holds briefly, then degrades, because the continued-but-novel combinations of
sinusoid phases past `L` are still out of distribution. Leaves the gap: cheap and
parameter-free, but does not extrapolate.

**Rotary position embeddings (Su et al., 2021; popularized via GPT-J).** Rather than adding a
vector at the input, rotate the query and key vectors in *each* attention layer by an angle
proportional to their absolute position, across a geometric range of frequencies. A query at
`i` and a key at `j` then interact through a rotation by `i-j`, so the scheme is effectively
relative; it injects position at every layer rather than only the first, and never into the
values. It extrapolates somewhat better than sinusoidal (perplexity keeps improving for
roughly the first hundred-plus extra tokens), but the per-layer rotations make training and
inference slower and heavier than the sinusoidal baseline. Leaves the gap: better
extrapolation, but not efficient.

**Relative position bias on attention scores (Shaw et al., 2018; the T5 variant of Raffel et
al., 2020).** Add no signal to the token embeddings. Instead, after each query–key dot-product
score, add a *learned scalar bias* that depends only on the relative distance between query
and key, shared across the network and tuned per head. Distances are **bucketed**: nearby
distances each get their own learned bias, while distances beyond a cutoff are pooled into
shared buckets — which intuitively might help past training length, since a never-seen
distance falls into an already-learned far bucket. Like rotary, it injects position at every
layer through the query–key comparison only, never into the values. Of the existing methods it
extrapolates best (perplexity improving for several hundred extra tokens). Its limitation is
efficiency: computing and gathering the bucketed relative bias makes it at least roughly twice
as slow as the sinusoidal baseline in a PyTorch/GPU setting, and it adds learned parameters.
So even though it proves that *changing the position method can buy extrapolation*, it cannot
deliver the intended payoff — train short and cheap, then extrapolate — because the method
itself is expensive.

The ladder, then: extrapolation is demonstrably reachable by swapping the position method (the
relative-bias result shows it), but every method that extrapolates well is slower and heavier
than the cheapest method that does not, and the position signals that *are* cheap (sinusoidal,
learned absolute) go out of distribution past `L`.

## Evaluation settings

- **WikiText-103** (Merity et al., 2016): ~103M tokens of English Wikipedia. The standard LM
  here is the adaptive-input model of Baevski & Auli (2018): 16 transformer layers, model
  dimension 1024, 8 attention heads, feedforward inner dimension 4096, tied input-embedding and
  softmax matrices, trained for 205 epochs. The natural way to study extrapolation is to fix
  `L_train` and evaluate over a range of `L_valid`.
- **Toronto Book Corpus** (Zhu et al., 2015): a books-domain corpus used to check that
  conclusions from Wikipedia transfer to another domain at the same hyperparameters.
- **CC100 + RoBERTa corpus**: a ~461 GB combination used for a 1.3-billion-parameter model: 25
  layers, dimension 2048, 16 heads, feedforward 8192, one epoch (50k updates) on 128 V100 GPUs.
- **Metric and protocol.** Perplexity on held-out text. Nonoverlapping inference is the
  default; stride-1 sliding-window inference serves as a slow but accurate analysis tool, used
  to separate "the model truly uses longer context" from "longer inputs merely reduce the early
  token curse." Extrapolation is measured by holding `L_train` fixed and increasing `L_valid`.

## Code framework

The position mechanism plugs into the same decoder-only transformer LM harness already used
for the baselines. The token embedding, the per-head scaled dot-product attention with its
causal mask, the feedforward sublayer, and the softmax output projection all already exist.
What is *not* settled is how positional order enters the model — that is exactly what is to be
designed — so the substrate exposes only the generic pieces that exist before any position
method is chosen, with one empty slot for the contribution.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T, device):
    # upper-triangular -inf so query i attends only to keys 1..i
    m = torch.full((T, T), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)


class Attention(nn.Module):
    """Multi-head scaled dot-product attention with a causal mask.
    Position information, if any, is injected by the scheme below."""

    def __init__(self, d_model, n_heads, pos):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.pos = pos                      # the position mechanism we will design

    def forward(self, x):                   # x: [B, T, d_model]
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)   # [B, H, T, d_k]
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)     # [B, H, T, T]
        scores = scores + causal_mask(T, x.device)
        # TODO: the position mechanism we will design enters here / below.
        attn = F.softmax(scores, dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, -1))


class PositionScheme(nn.Module):
    """How sequential order enters the model. The whole design space of this
    problem lives in this one object; nothing about it is decided yet."""

    def __init__(self, config):
        super().__init__()
        # TODO: the position mechanism we will design.

    def apply(self, *args, **kwargs):
        # TODO: how (and where) position information is injected.
        raise NotImplementedError


class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = PositionScheme(config)
        self.layers = nn.ModuleList(
            Block(config.d_model, config.n_heads, self.pos) for _ in range(config.n_layers)
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, tokens):              # tokens: [B, T]
        x = self.tok(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))      # [B, T, vocab_size]
```

`Block` is a residual attention + feedforward pair (standard); the only undecided piece is
`PositionScheme` and where in `Attention` it acts.
