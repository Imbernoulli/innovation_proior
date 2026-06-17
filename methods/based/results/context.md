# Context: efficient sequence mixing and the recall bottleneck (circa 2023-2024)

## Research question

A language model's sequence mixer — the layer that lets each position read information from
earlier positions — is judged on two axes at once: how well it *recalls* (grounds a generation
in a specific token seen earlier in the context), and how cheaply it runs, especially during
autoregressive generation. Softmax attention wins the first axis decisively but loses the second:
to generate each new token it attends over a growing cache of all past keys and values (the
"KV-cache"), so the per-token cost and the memory footprint both grow linearly with how much
context has already been seen. On long sequences this KV-cache dominates memory and throttles
throughput. The precise problem is to build a sequence mixer whose *recurrent state during
generation is bounded* (so memory and per-token cost do not grow with sequence length) yet that
still performs in-context key-value lookup as well as attention — and ideally one whose state
size is a tunable knob, so the same design can be slid from "cheap and forgetful" to "expensive
and recall-perfect." The hard part is that recall and a small state appear to be in direct
tension, and it is not obvious whether that tension is fundamental or just an artifact of the
particular efficient architectures tried so far.

## Background

By this time, sub-quadratic alternatives to attention have proliferated — gated convolutions
(H3, Hyena), input-dependent recurrences and state-space models (RWKV, Mamba) — and several
match attention in *aggregate* perplexity while running much faster. But aggregate perplexity
hides skill-level differences. A specific, well-documented one is *associative recall*: the
ability to retrieve a value bound to a key earlier in the context. The canonical stress test is
**Multi-Query Associative Recall (MQAR)** (Arora, Eyuboglu et al. 2023): the input is a stream of
key-value token pairs followed by query keys, and at each query position the model must emit the
value that was bound to that key. A reported diagnostic finding is that full softmax attention
solves MQAR with a model dimension that is *constant* in sequence length, while gated-convolution
mixers need their hidden dimension to grow at least linearly with sequence length to keep up — and
empirically these convolutional mixers (Hyena, H3) sit well below the recall-vs-memory frontier,
whereas an input-dependent recurrence (Mamba) makes the best use of a fixed memory budget among the
prior sub-quadratic options but still degrades on MQAR as the number of key-value pairs grows,
because it must compress every key seen into one fixed-size hidden state.

Two structural facts about the design space are load-bearing. First, what governs generation cost
is the size of the model's *recurrent state* — the quantity it must carry forward from one
position to the next. For attention that state is the whole KV-cache (it grows with N); for a
recurrence or linear attention it is a fixed-shape buffer. Second, there is a clean way to reason
about the *minimum* state any causal model needs to solve recall, via communication complexity.
In the *index problem*, Alice holds a bit string x in {0,1}^N, Bob holds an index i, and they must
output x_i with only a single one-way message from Alice to Bob; the randomized one-way
communication complexity of this is known to be Omega(N) (Jayram et al. 2008). This is the tool
that converts "how much must a causal model remember" into a hard lower bound.

Alongside this, two efficiency primitives are already on the table that each *bound* attention's
state in a different way. (i) *Linear attention* removes the softmax so that the attention scores
factor through a feature map; matrix-product associativity then lets the layer carry a
fixed-shape running summary instead of a growing cache. (ii) *Sliding window attention* restricts
each query to attend over only the last w keys, capping the cache at w. Each, on its own, is known
to leave a gap (below). Finally, a separate line of analysis of *feature maps* for linear attention
observes that softmax attention's effective weights are "spiky" — low-entropy, concentrated on a
few keys — and monotonic in the query-key dot product, and that linear-attention feature maps which
fail to reproduce this spikiness tend to underperform on recall (Zhang et al. 2024).

## Baselines

**Softmax attention (Vaswani et al. 2017).** With projections q, k, v = xW_q, xW_k, xW_v, the
causal output is

```
y_i = sum_{j<=i} [ exp(q_i^T k_j / sqrt(d)) v_j ] / sum_{m<=i} exp(q_i^T k_m / sqrt(d)).
```

It is recall-perfect and, with IO-aware kernels, trains in O(N) memory. **Gap:** generation keeps a
KV-cache {k_i, v_i} that grows with the sequence; producing token n costs O(nd) over a state whose
size is unbounded in N, so memory and throughput degrade on long contexts.

**Linear attention (Katharopoulos et al. 2020, "Transformers are RNNs").** Replace the softmax
kernel exp(q^T k) with a feature-map dot product phi(q)^T phi(k), phi: R^d -> R^{d~}. The attention
output becomes, using associativity,

```
y_i = phi(q_i) ( sum_{j<=i} phi(k_j)^T v_j ) / ( phi(q_i) sum_{j<=i} phi(k_j)^T ),
```

so the layer trains in O(N d~^2) instead of O(N^2), and during generation it carries a *fixed-shape*
recurrent state: a KV-state S_i = S_{i-1} + phi(k_i)^T v_i in R^{d~ x d} and a normalizer
z_i = z_{i-1} + phi(k_i)^T in R^{d~}, with y_i = phi(q_i) S_i / (phi(q_i) z_i) — O(1) per token. The
original feature map is phi(x) = elu(x) + 1 (chosen to keep the kernel non-negative so weights stay
positive). **Gap:** with such generic feature maps, linear attention underperforms softmax on recall;
the weights it produces are flat/high-entropy rather than spiky, and it lacks the precision for local
token shifts and comparisons.

**Feature maps for linear attention (Zhang et al. 2024; Choromanski et al. 2020).** A range of phi
have been proposed — random Fourier features that approximate the softmax kernel in expectation
(Performer), ReLU with cosine reweighting (cosFormer), and learned single-layer MLPs trained to
match softmax weights (Hedgehog). **Gap:** random-feature maps are unbiased only in expectation and
need many features to be accurate; learned maps add parameters and a training stage; and the ones
that map R^d -> R^{d^2} to better approximate softmax incur O(N d^3) time/space and a large state.

**Sliding window attention (Parmar et al. 2018; Child et al. 2019; Beltagy et al. 2020; used in
Mistral 7B).** Each query q_i attends exactly (softmax) only over keys {k_{i-w+1}, ..., k_i}, giving
O(Nw) cost and a KV-cache capped at w. **Gap:** anything bound more than w tokens back is invisible,
so associative-recall range is limited by the window; widening w to recover long-range recall grows
the state and the cost linearly again (deployed windows are large, e.g. w=4096).

**Input-dependent state-space models (Mamba; Gu & Dao 2023).** A selective SSM with a fixed-size
recurrent hidden state h, input-dependent transition and input matrices, run as a linear recurrence
h_i = A-bar_i h_{i-1} + B-bar_i u_i, z_i = C_i^T h_i. Fixed state -> O(1)/token generation, strong
aggregate quality, and the best recall-per-memory among prior sub-quadratic mixers. **Gap:** it must
compress all keys into one fixed hidden state, so its recall falls off as the number of key-value
pairs grows; its state cannot be cheaply dialed up to attention's recall without losing the
efficiency that motivated it.

**Gated convolutions and the BaseConv canonical form (Arora et al. 2023).** A broad class built from
gating (Hadamard products) and (possibly long) convolutions; BaseConv,
z = (uW^B + b^B) ⊙ (K * u + b^K), is a minimal operator that can simulate any such architecture.
**Gap:** this class provably cannot solve MQAR in a constant number of layers / with dimension
independent of sequence length, which is why convolutional mixers fall below the recall-memory
frontier.

## Evaluation settings

- **MQAR synthetic.** Sequences of key-value token pairs followed by query keys; the model must emit
  the value for each query key. Train on shorter sequences with a moderate number of key-value pairs
  and evaluate on longer sequences with more pairs to probe recall capacity. Metric: exact-match
  next-token accuracy at the query positions only (non-query positions are masked out of the loss).
  Standard protocol sweeps a hyperparameter that controls each mixer's recurrent-state size (feature
  dimension for linear attention, window width for SWA, model/state dimension for SSMs) and plots
  recall accuracy against that state size — the memory-recall tradeoff curve.
- **Recurrent-state-size accounting.** For each mixer, the inference state size is computed
  explicitly (KV-cache size for attention; phi-expanded KV-state for linear attention; window-sized
  cache for SWA; hidden-state size for SSMs) so different architecture classes can be placed on one
  axis.
- **Language modeling at scale.** Autoregressive LM on a large text corpus, perplexity vs. model
  size; a held-out slice that is recall-intensive (associative-recall hits within real text);
  downstream zero-/few-shot evaluation harnesses; throughput/latency benchmark configurations
  during generation against optimized attention and recurrent baselines.

## Code framework

The mixer plugs into a fixed Transformer-style residual stack: token embedding, a small number of
pre-norm residual blocks (each = pre-norm residual sequence-mixer + pre-norm residual MLP), then a
tied output head; the optimizer (AdamW, cosine decay), the loss (token-level cross-entropy with the
non-query positions ignored), and the MQAR data generator are all fixed. The open slot is a causal
sequence mixer with a bounded generation state; the substrate exposes only generic primitives that
already exist in efficient sequence models (linear projections, causal masking, local attention,
depthwise 1-D convolution, and elementwise operations) plus the empty mixer stub.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMixer(nn.Module):
    """Causal sub-quadratic sequence mixer. Maps [B, T, d_model] -> [B, T, d_model];
    output at position t may depend only on inputs at positions <= t. Generation cost
    and recurrent-state size should not grow with sequence length."""

    def __init__(self, d_model: int, seq_len: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # generic primitives that already exist for a mixer of this shape
        # (linear projections, a depthwise conv, etc.) may be instantiated here
        # TODO: the sequence-mixing operator we will design.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        # TODO: produce the causal mixed output [B, T, d_model].
        raise NotImplementedError


# fixed residual-stack harness the mixer plugs into
class Block(nn.Module):
    def __init__(self, d_model, seq_len, mixer_cls):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = mixer_cls(d_model, seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))   # pre-norm residual mixer
        x = x + self.mlp(self.norm2(x))     # pre-norm residual MLP
        return x
```

The single empty slot is the mixer operator: given the per-position projections it chooses to form,
mix information causally across the sequence and return `[B, T, d_model]`, while keeping the
quantity carried forward from one position to the next bounded in sequence length.
