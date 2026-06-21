# Context: subquadratic sequence mixing for autoregressive language models (circa 2023-2024)

## Research question

Softmax attention is the workhorse primitive of sequence modeling. Computing
`o_t = sum_{i<=t} softmax(k_i^T q_t) v_i` over a length-`L` sequence costs `O(L^2 d)` and, at
inference, keeps every past key/value vector — a "KV cache" that grows linearly in `L`. A family
of subquadratic alternatives replaces the recurrence so that the per-step state is a *fixed-size*
object, giving `O(Ld^2)` training and constant-memory inference. These models are evaluated both
on plain language modeling and on *associative recall*, where the model must fetch a value it
stored earlier keyed on some token it has now seen again.

The question this work addresses is how to design a sequence-mixing layer with a fixed-size state
that runs in subquadratic time and constant inference memory, performs well on associative
recall, and admits a *training* algorithm that is hardware-efficient on modern GPUs — rich in
matrix multiplications so it can use tensor cores, and parallelizable across the sequence
dimension for high occupancy. A layer can be cheap in FLOPs and still run one step at a time on
the elementwise units, so the form of the training algorithm is part of the design, not just the
form of the recurrence.

## Background

**Linear attention as a matrix-valued linear RNN.** Katharopoulos et al. (2020) observed that
if one replaces the exponential kernel `exp(k_i^T q_t)` in softmax attention with a dot product
`phi(k_i)^T phi(q_t)` of feature-mapped vectors, the sum over the past can be re-associated:

```
o_t = sum_{i<=t} ( phi(q_t)^T phi(k_i) ) v_i / sum_{j<=t} ( phi(q_t)^T phi(k_j) )
    = S_t phi(q_t) / ( z_t^T phi(q_t) ),   S_t = sum_{i<=t} v_i phi(k_i)^T,  z_t = sum_{i<=t} phi(k_i).
```

`S_t` is a `d x n` matrix-valued hidden state and `z_t` a normalizer vector. The model is now a
linear RNN: `S_t = S_{t-1} + v_t phi(k_t)^T`, `z_t = z_{t-1} + phi(k_t)`, read out by
`o_t = S_t phi(q_t)/(z_t^T phi(q_t))`. The whole past lives in the fixed-size `S_t`, giving
constant-memory inference. Katharopoulos used `phi = elu(.)+1` (positive, with nonzero gradient
on the negatives, unlike ReLU). Their feature map preserves the key dimension, so the state size
— and hence memory capacity — is unchanged from the input key dimension. The denominator
`z_t^T phi(q_t)` is often dropped in linear-attention variants, and the feature map is often taken
to be the identity, leaving the bare recurrence `S_t = S_{t-1} + v_t k_t^T`, `o_t = S_t q_t`.

**Two computational forms.** With `Q, K, V in R^{L x d}` stacked, the identical output can be
computed in a *parallel form*, `O = (Q K^T ⊙ M_L) V`, where `M_L` is the causal mask. The
parallel form costs `O(L^2 d + L d^2)` FLOPs but runs in `O(1)` sequential steps and is pure
matmul, so it saturates tensor cores and GPU occupancy. The *recurrent form*
`S_t = S_{t-1}+v_t k_t^T`, `o_t = S_t q_t` costs only `O(L d^2)` FLOPs but is strictly sequential
in `t` and its elementwise updates do not use matmul accelerators. The *chunkwise parallel form*
(Hua 2022; Sun et al. 2023; Yang et al. 2023) interpolates: split the sequence into `L/C` chunks
of size `C`, carry a chunk-level state `S_[t]`, do intra-chunk work with the parallel form and
inter-chunk propagation with the recurrence,

```
S_[t+1] = S_[t] + V_[t]^T K_[t],
O_[t]   = Q_[t] S_[t]^T + (Q_[t] K_[t]^T ⊙ M_C) V_[t].
```

This costs `O(LCd + Ld^2)` FLOPs in `O(L/C)` sequential steps; `C=L` recovers the parallel form
and `C=1` the recurrent form. With `C` a small constant (64 or 128), it is subquadratic *and*
matmul-rich, and the intra-chunk states need never be materialized. I/O-aware kernels in the
FlashLinearAttention style make this fast in practice even at moderate sequence lengths.

**Additive memory.** The additive update `S_t = S_{t-1}+v_t k_t^T` is a Hebbian / outer-product
associative memory — in the fast-weight-programmer view (Schmidhuber 1992; Schlag et al. 2021),
`S_t` is a "fast weight" matrix written by outer products and read by matrix-vector product. Such
memories have bounded capacity (McEliece et al. 1987). Concretely, reading with a stored key `k_j`
gives `S k_j = (sum_i v_i k_i^T) k_j = v_j (k_j^T k_j) + sum_{i != j} (k_i^T k_j) v_i`: the
intended value plus cross-talk from every non-orthogonal key. In a `d`-dimensional space there
are at most `d` mutually orthogonal keys. Additive linear-attention variants underperform softmax
attention on language modeling and, more sharply, on recall-intensive synthetic and real tasks.

**Gated linear variants and the general associative-RNN view.** Data-dependent *gating*
multiplies the state by an input-dependent decay before the additive write. Many recent models
fall into a single template of an associative RNN with matrix-valued state,
`S_t = S_{t-1} • M_t + v_t k_t^T`, `o_t = S_t q_t`, where `•` is an associative operator and
`M_t`, `v_t`, `k_t`, `q_t` are functions of `x_t`. With `• = ⊙` (Hadamard) and various structured
`M_t` this recovers, among others: Gated Linear Attention (`M_t = beta_t alpha_t^T`, Yang et al.
2023), HGRN2 (Qin et al. 2024), RWKV-6 (Peng et al. 2024), RetNet (`M_t = gamma`, a fixed scalar
decay, Sun et al. 2023), mLSTM (Beck et al. 2024), and Mamba/Mamba-2 (selective state-space
models, Gu & Dao 2023; Dao & Gu 2024 — reparameterizable as a gated linear transformer). The
Hadamard choice is *elementwise* (`O(dn)` per step) and supports parallel scan; a genuinely
matrix-valued `S_{t-1} M_t` with unstructured `M_t` would cost `O(dn^2)` per step. These gated
variants are competitive with strong transformer baselines on plain language modeling. The
elementwise decay forgets per channel.

**The delta rule / Widrow-Hoff as an error-correcting write.** A classical alternative to the
Hebbian write is the *delta rule* (Widrow & Hoff 1960), later known as the Least Mean Squares
(LMS) algorithm in adaptive signal processing. Read the fast-weight state as a regressor that
should map `k_t` to `v_t`, and write by a single gradient step on the squared error:

```
L_t(S) = 1/2 || S k_t - v_t ||^2,    S_t = S_{t-1} - beta_t * grad_S L_t = S_{t-1} - beta_t (S_{t-1} k_t - v_t) k_t^T,
```

with `beta_t` a (here, dynamic) learning rate. Equivalently, retrieve the old value
`v_t^old = S_{t-1} k_t`, form a new value `v_t^new = beta_t v_t + (1-beta_t) v_t^old`, and swap
it in: `S_t = S_{t-1} - v_t^old k_t^T + v_t^new k_t^T`. The write is proportional to the
*prediction error* `v_t - v_t^old`, so a key already well-represented in memory produces little
change, and `beta_t` controls how strongly the current pair overwrites the colliding one. The
delta-rule fast weight has better memory capacity than the Hebbian one (Gardner 1988; Prados &
Kak 1989) and, applied as a sequence-mixing layer (Schlag et al. 2021, with
`beta_t = sigma(W_beta x_t)`), it improves associative recall and small-scale language modeling
and machine translation over additive linear attention.

**Rank-one state transitions.** A matrix of the form `I - beta k k^T` has eigenvalue `1` on the
orthogonal complement of `k` and `1 - beta ||k||_2^2` along the span of `k`; if `k` is
L2-normalized this non-unit eigenvalue becomes `1 - beta`. Numerical linear algebra has standard
representations for working with near-identity rank-one transformations, and a sequential
recurrence written as a lower-triangular linear system can be handled by forward substitution.

## Baselines

A new layer for this problem would be measured against the following.

**Softmax-attention transformer (Vaswani et al. 2017; LLaMA-architecture "Transformer++",
Touvron et al. 2023).** `o_t = sum_{i<=t} softmax_i(k_i^T q_t / sqrt(d)) v_i`, with RoPE,
SwiGLU FFN, pre-norm. The quality bar and the recall bar; `O(L^2 d)` compute and a KV cache that
grows linearly with the sequence.

**Linear attention (Katharopoulos et al. 2020).** The matrix-valued linear RNN above with
`phi = elu+1`. Linear-time training (in its chunkwise form), constant-memory inference; the
additive Hebbian write.

**Gated linear attention / RetNet / Mamba / RWKV-6 / HGRN2 / mLSTM.** The
`S_t = S_{t-1} ⊙ M_t + v_t k_t^T` family with data-dependent (or, for RetNet, fixed) decay,
trained with chunkwise or scan algorithms; competitive with transformers on plain LM. The decay
is elementwise (scalar or diagonal).

**Delta-rule fast-weight layer (Widrow & Hoff 1960; Schlag et al. 2021).** The error-correcting
write `S_t = S_{t-1} - beta_t(S_{t-1} k_t - v_t) k_t^T` with `beta_t = sigma(W_beta x_t)`,
feature map `elu+1` and an L1-style key/query normalization. Its training algorithm is the
sequential, memory-efficient recurrent procedure inherited from Katharopoulos (their §3.3.1):
because the value being written, `v_t^old = S_{t-1} k_t`, depends on the running state, the
per-step writes are computed one step at a time on the elementwise units.

## Evaluation settings

The natural yardsticks for subquadratic language-modeling layers:

- **Associative-recall / in-context-learning synthetics**: Multi-Query Associative Recall
  (MQAR; Arora et al. "Zoology" 2023), the Mechanistic Architecture Design suite (MAD; Poli et
  al. 2024), and RegBench (Akyürek et al. 2024), which tests inferring a probabilistic
  finite-automaton language in context.
- **Language modeling**: WikiText perplexity, and zero-shot accuracy on common-sense reasoning
  — LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-easy and ARC-challenge — via the standard LM
  evaluation harness.
- **Recall-intensive real tasks** (Arora et al. 2024): FDA (key-value extraction from PDFs),
  SWDE (structured extraction from HTML), and SQuAD reading comprehension.
- **Scale and optimization protocol**: GPT/LLaMA-style decoder models at ~340M (≈15B tokens),
  1.3B (≈100B tokens), and 3B (≈1T tokens); AdamW at peak learning rate 3e-4 with a cosine
  schedule and warmup, weight decay 0.01, gradient clipping at 1.0, head dimension 128, and a
  short-convolution kernel size of 4.
- **Throughput / kernel speed**: training tokens-per-second and per-kernel wall-clock for the
  recurrent vs. chunkwise vs. dense-attention forms, swept over sequence length `L` and head
  dimension `d_head`, on GPU.

## Code framework

The layer plugs into a standard decoder-only transformer harness: token embedding, a stack of
pre-norm blocks each with a token-mixing sublayer and a SwiGLU feed-forward sublayer, a final
norm and the tied LM head. Everything except the token-mixing sublayer already exists. The open
slot is a causal mixer whose fixed-size state can be updated across the sequence and trained
efficiently on the GPU.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMixer(nn.Module):
    """The sequence-mixing layer to design. Maps x in R^{B x L x d} to o in R^{B x L x d}
    causally (o_t may depend on x_1..x_t only), with a fixed-size per-step state so that
    inference is constant-memory and training is subquadratic in L. The internal recurrence /
    state update and a hardware-efficient (matmul-rich, sequence-parallel) training algorithm
    for it are exactly what is to be designed."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        # query/key/value projections exist for any attention-like mixer
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # TODO: additional parameters the mixer may need.

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # TODO: the causal sequence-mixing recurrence and its hardware-efficient
        #       training path. Produce o from q, k, v.
        o = None
        return self.o_proj(o)


class Block(nn.Module):
    """Pre-norm transformer block: token mixing then channel mixing, each residual."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mixer = TokenMixer(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)  # SwiGLU feed-forward, already defined

    def forward(self, x):
        x = x + self.mixer(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

A completed mixer supplies the `forward` body, any extra projections or normalization it needs,
and the corresponding state-update path for training and inference.
