# Context: subquadratic sequence mixing for language-model pretraining (circa 2023)

## Research question

Standard autoregressive Transformers mix tokens with softmax attention, which costs
`O(L^2)` compute and memory in the sequence length `L`, and at inference attend each new
token to a KV cache that grows with `L`. The question is how to design a sequence-mixing
layer whose training and inference scale better than `O(L^2)` — linear in `L` for inference
and subquadratic for training — while remaining competitive in language-model quality with a
strong softmax-attention Transformer trained on the same data and token budget, and fast in
measured wall-clock throughput on the GPUs people train on.

## Background

By this time the dominant sequence layer is **softmax attention** (Vaswani et al. 2017). For
input `X ∈ R^{L×d}`, it forms `Q,K,V = X W_Q, X W_K, X W_V` and
`O = softmax(Q K^T / sqrt(d_k) + A) V`, where `A` is an additive causal mask
(`A_{ij}=0` if `i≥j`, else `-∞`). This *parallel form* trains on the whole sequence at once and
is quadratic in `L`. At inference it uses the equivalent *recurrent form*, attending each new
token to a cache of all past keys and values (the "KV cache"), whose size grows with `L`.

**Linear attention** (Katharopoulos et al. 2020, "Transformers are RNNs") is the line of work
that makes this cheap. Replace the exponential similarity `exp(q_t · k_i)` with a kernel
`k(x,y) = ⟨φ(x), φ(y)⟩` for some feature map `φ`. Then
`o_t = (Σ_{i≤t} φ(q_t) φ(k_i)^T v_i) / (Σ_{i≤t} φ(q_t) φ(k_i)^T)`, and because `φ(q_t)`
factors out of the sums, one can carry running summaries
`S_t = Σ_{i≤t} φ(k_i)^T v_i ∈ R^{d×d}` and `z_t = Σ_{i≤t} φ(k_i)^T ∈ R^{d×1}` and write a
recurrence `S_t = S_{t-1} + φ(k_t)^T v_t`, `z_t = z_{t-1} + φ(k_t)^T`, `o_t = φ(q_t) S_t /
(φ(q_t) z_t)`. This is a **linear RNN with a matrix-valued hidden state** `S_t`, updated by the
outer product `φ(k_t)^T v_t` — a form of "fast weights" (Schmidhuber 1992; Ba et al. 2016).
Inference is `O(1)` per step in time, with a fixed-size state instead of a growing cache.
The original feature map was `φ(x) = elu(x) + 1` (a positive map); pre-existing variants also
use an identity map `φ = I` with no normalizer, leaving the bare unnormalized update
`S_t = S_{t-1} + k_t^T v_t`, `o_t = q_t S_t`.

For *training*, causality interacts with associativity. The non-causal product can be
reassociated — `(Q K^T) V = Q (K^T V)` — turning `O(L^2 d)` into `O(L d^2)`. The causal
parallel form `O = ((Q K^T) ⊙ M) V`, with binary lower-triangular `M`, applies the mask
*elementwise* to `Q K^T`, so the reassociation does not apply directly and this form is
`O(L^2 d)`.

The **chunkwise parallel form** (Hua et al.
2022; Sun et al. 2023) interpolates between the two. Split the sequence into `N = L/C` non-overlapping chunks of length
`C`. Carry the matrix state across chunks by a recurrence,
`S_{[i+1]} = S_{[i]} + K_{[i]}^T V_{[i]}` (computed by a `C×d` matmul), and within a chunk
compute the output as an inter-chunk term plus an intra-chunk term,
`O_{[i+1]} = Q_{[i+1]} S_{[i]} + ((Q_{[i+1]} K_{[i+1]}^T) ⊙ M) V_{[i+1]}`. The intra term is the
small `C×C` masked parallel form; the inter term carries everything before the chunk through
the single state `S_{[i]}`. Total training cost is `O(L C d + L d^2)`, below `O(L^2 d)` once
`L > d`. Setting `C = L` recovers the quadratic parallel form; `C = 1` recovers the recurrent
form; `C` interpolates.

Two empirical facts about this design space are established. First, in language modeling
**plain linear attention scores below softmax attention**, often by a wide margin (Kasai et
al. 2021). Second, in 1D recurrent networks a **forget gate is central**, and a
*data-dependent* one — its value computed from the current input — is a long-standing element
of recurrent models; this is the lesson of the LSTM forget gate (Hochreiter & Schmidhuber
1997; Gers et al. 2000) and of analyses placing much of the model's capacity in the forget
gate (van der Westhuizen & Lasenby 2018). A related observation (Martin & Cundy 2018) is that
if the forget value depends only on the *current input* (not the previous hidden state), the
recurrence stays linear and parallelizable; HGRN (Qin et al. 2023) used this at moderate
scale.

A third fact is about hardware. An algorithm is only fast if it respects the GPU: keep
**occupancy** high (use enough streaming multiprocessors — when batch is small, parallelizing
over the sequence dimension is what keeps the SMs busy), route as much work as possible
through **tensor cores** (half-precision matmuls run roughly an order of magnitude faster than
the same FLOPs on general CUDA cores, but only matmul-shaped work qualifies, and tile sizes
want to be multiples of 16), and minimize traffic to **HBM** (the large, slow global memory)
by keeping reused tensors in the small fast on-chip SRAM. FlashAttention (Dao et al. 2022;
Dao 2023) makes exact softmax attention fast by tiling the computation so the `L×L`
score matrix is never written to HBM, recomputing it in the backward pass instead, and
(in its second version) adding sequence-level parallelism. The elementwise recurrent form has
the lowest FLOPs of all but has low arithmetic intensity and does not use tensor cores;
existing chunkwise linear-attention implementations vary in how I/O-aware they are.

## Baselines

These are the prior sequence-mixing methods a new layer would be measured against.

**Softmax-attention Transformer (Vaswani et al. 2017), modern recipe ("Transformer++").**
The reference for quality. The strong contemporary form is the LLaMA architecture (Touvron et
al. 2023): rotary position embeddings, a SwiGLU feed-forward network, RMSNorm, pre-norm
residual blocks. Core math as above; `O = softmax(QK^T/sqrt(d_k)+A)V`. Cost is `O(L^2)` in
compute and memory, with an inference KV cache that grows linearly with context.

**Plain linear attention (Katharopoulos et al. 2020).** The recurrence `S_t = S_{t-1} +
φ(k_t)^T v_t`, `o_t = φ(q_t) S_t`, with the chunkwise training form above. Constant-memory
linear-time inference, subquadratic training. The update adds outer products into a fixed-size
matrix state.

**Linear attention with a global decay — RetNet (Sun et al. 2023), TransNormerLLM (Qin et al.
2023).** Multiply the state by a single fixed scalar before each update:
`S_t = γ S_{t-1} + k_t^T v_t`, with `γ ∈ (0,1)` a global, **data-independent** constant (this
is essentially linear attention with an ALiBi-style recency bias). The single `γ` keeps the
attention-style parallel and chunkwise forms for efficient training, and gives an improvement
over no decay. RetNet also adds per-head output normalization and an output gate.

**Mamba (Gu & Dao 2023).** A selective state-space model whose transition and input
projections are made *input-dependent*, giving a full-rank, data-dependent update. The
full-rank selective update is computed with a parallel scan rather than a matmul, materializing
each time step's state; the per-channel state expansion is set around 16 so those states stay
in SRAM.

**Fine-grained matrix-valued gates (Mao 2022; Katsch 2023 GateLoop).** Give linear attention a
2D, data-dependent gate via a low-rank outer product, `G_t = α_t^T β_t`, so the state update
is `S_t = G_t ⊙ S_{t-1} + k_t^T v_t` with `d·d_k + d·d_v` gate parameters. These training
procedures materialize the matrix-valued hidden state for every time step in HBM.

## Evaluation settings

The natural yardsticks already in use for this comparison:

- **Language-model pretraining corpus.** A large deduplicated web/text mixture such as
  SlimPajama (Cerebras 2023), a ~627B-token corpus from which a fixed subset (e.g. 100B
  tokens) is drawn, tokenized with a standard subword tokenizer (e.g. the Mistral tokenizer;
  Jiang et al. 2023). Models are trained from scratch at fixed parameter scales (e.g. ~340M
  and ~1.3B) for a fixed token budget, so every architecture sees identical data and tokens.
- **Optimizer / schedule.** AdamW (Loshchilov & Hutter 2018), peak learning rate ~3e-4,
  cosine decay to a small final rate, a warmup of a fraction of a billion tokens, weight decay
  ~0.01, gradient clipping at 1.0 — applied identically across all architectures.
- **Quality metrics.** Perplexity on held-out text such as WikiText and LAMBADA; zero-shot
  downstream accuracy on common-sense and QA benchmarks — LAMBADA, PiQA, HellaSwag
  (length-normalized accuracy), WinoGrande, ARC-easy, ARC-challenge (length-normalized),
  OpenbookQA — scored with a standard LM evaluation harness (Gao et al. 2021).
- **Recall-intensive probes.** The synthetic multi-query associative recall task MQAR
  (Arora et al. 2023, "Zoology"), a harder multi-query version of the induction-head task, and
  real information-extraction / reading-comprehension tasks (FDA, SWDE, SQuAD).
- **Long-context / length extrapolation.** Training on long contexts directly (e.g. 8K) and on
  longer contexts (e.g. 24K) via truncated backpropagation through time over short segments
  (carrying the recurrent state across segments without backpropagating through the carry),
  evaluated on long-document corpora such as PG19.
- **Throughput.** Measured training tokens/second and memory footprint versus a FlashAttention
  baseline and versus other subquadratic models, across sequence lengths, on the same GPUs.

## Code framework

The new sequence-mixing layer drops into the standard pre-norm Transformer training harness
already used for the baselines: token + (optional) position embeddings, a stack of residual
blocks each composed of a token-mixing sublayer and a SwiGLU feed-forward sublayer, a final
norm and a language-model head, trained with the usual cross-entropy loop and AdamW. Everything
except the token-mixing sublayer already exists; the token mixer itself — how it summarizes the
past and produces each position's output subquadratically — is exactly what is to be designed,
so it is left as a single empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMixer(nn.Module):
    """The sequence-mixing sublayer to be designed. Consumes a sequence
    x of shape (batch, length, d) and returns a contextualized sequence of
    the same shape. It must be cheaper than O(L^2) in the sequence length L,
    work autoregressively (position t may only depend on positions <= t),
    and stay competitive in quality with softmax attention."""

    def __init__(self, config):
        super().__init__()
        self.d = config.n_embd
        self.n_head = config.n_head
        # whether this mixer needs absolute position embeddings added upstream
        self.use_pos_emb = True
        # TODO: the token-mixing mechanism we will design.

    def forward(self, x):                      # x: (B, L, d)
        # TODO: produce the contextualized sequence subquadratically in L.
        raise NotImplementedError


class SwiGLU(nn.Module):
    """Existing gated feed-forward sublayer (LLaMA-style)."""

    def __init__(self, config):
        super().__init__()
        hidden = config.ffn_mult * config.n_embd
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    """Existing pre-norm residual block: token mixer, then feed-forward."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mixer = TokenMixer(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.mixer(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# existing language-model pretraining loop the block plugs into
def train(model, data_loader, optimizer):
    for batch in data_loader:                  # (B, L) token ids
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)                 # embed -> stack of Block -> norm -> LM head
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()                        # AdamW, lr warmup+cosine, grad clip 1.0
        optimizer.step()
```

The harness supplies embeddings, the SwiGLU FFN, the residual structure, and the training
loop; the token mixer's `__init__` and `forward` are the empty slot the method fills.
