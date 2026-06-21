# Context: efficient sequence modeling and the state-compression tradeoff

## Research question

Foundation models across language, audio, genomics, and time series are built on a sequence
model as their backbone. The dominant choice is self-attention, which routes information densely
between every pair of positions in a context window. Training scales quadratically in sequence
length, because all `L²` pairwise interactions are formed; autoregressive decoding keeps the entire
context resident as a key/value cache, so per-step time and memory grow with how much has already
been generated; and modeling is confined to a fixed window.

A large family of subquadratic architectures — linear attention, gated convolutions, recurrent
networks, structured state space models — offers linear or near-linear scaling in length and a
constant-size inference state. The question is how to build a sequence model backbone that is
competitive with attention on information-dense discrete modalities such as text while retaining
linear-time scaling and a fixed-size inference state.

## Background

**Classical continuous state space models.** A continuous linear system maps a scalar input
signal `x(t)` to an output `y(t)` through an `N`-dimensional latent state `h(t)`:

```
h'(t) = A h(t) + B x(t),    y(t) = C h(t),
```

with `A ∈ ℝ^{N×N}`, `B ∈ ℝ^{N×1}`, `C ∈ ℝ^{1×N}`. This is the object behind Kalman filters and
linear dynamical systems (Kalman 1960). To run it on a discrete sequence it is discretized with
a step size `Δ`, turning the continuous `(Δ, A, B)` into discrete `(Ā, B̄)` by a fixed rule. The
most common rule is zero-order hold (ZOH) — hold the input constant across each step of length
`Δ` and integrate the linear ODE exactly:

```
Ā = exp(Δ A),    B̄ = (Δ A)^{-1} (exp(Δ A) − I) · Δ B.
```

Once discretized, the system is the linear recurrence `h_t = Ā h_{t−1} + B̄ x_t`, `y_t = C h_t`.

**HiPPO and principled long-range memory.** A particular choice of `A` (Gu et al. 2020, HiPPO)
makes the hidden state a near-optimal online summary of the input history — the coefficients of
the input projected onto a basis of orthogonal polynomials. This gives state space models a
principled long-range-memory mechanism and supplies a special initialization, rather than a
heuristic decay.

**Two computation modes and the LTI property.** When `(Δ, A, B, C)` are held constant across all
time steps — the *linear time-invariant* (LTI) case — the recurrence has a second, equivalent
form. Unrolling it from `h_0 = 0` gives a convolution by a single fixed kernel:

```
K̄ = (C B̄, C Ā B̄, C Ā² B̄, …, C Ā^k B̄, …),    y = x ∗ K̄.
```

The recurrence is convenient for step-by-step autoregressive inference (constant work per step);
the convolution is convenient for training, because with the whole sequence in hand it is
evaluated in parallel with an FFT in `O(L log L)`. This duality — that an LTI linear recurrence
*is* a global convolution — is the efficiency engine of this whole family, and it holds *only*
because the dynamics are constant in time: one kernel can be reused at every position precisely
because `(Ā, B̄, C)` do not change.

**Structured SSMs (S4 and descendants).** Computing the convolution kernel for the HiPPO `A` is
not trivial because that `A` cannot be stably diagonalized; S4 (Gu et al. 2021/2022) writes `A`
as normal-plus-low-rank and reduces the kernel to a diagonal-plus-low-rank computation via a
truncated generating function, a Cauchy kernel, and the Woodbury identity, all in `O(N + L)`.
Later work (DSS; S4D, Gu et al. 2022) showed the low-rank correction can be dropped: a purely
*diagonal* `A` works essentially as well, so `A` reduces to `N` numbers per channel, with simple
initializations such as `A_n = −(n+1)` (real) or `A_n = −1/2 + n·i` (complex). Applied with a
state of size `N` to a length-`L`, `D`-channel, batch-`B` input, the SSM runs independently per
channel; the full hidden state then has size `B·L·D·N`, a factor of `N` larger than the input.

**The parallel associative scan.** A first-order linear recurrence `h_t = a_t h_{t−1} + b_t`
need not be computed sequentially even when `a_t, b_t` vary with `t`. Pairs `(a, b)` compose
under the operator

```
(a, b) • (a', b') = (a' a,  a' b + b'),
```

and a work-efficient parallel scan (Blelloch 1990; Martin & Cundy 2018) computes all prefixes in
`O(L)` work and `O(log L)` depth. S5 (Smith et al. 2023) used exactly this to compute a diagonal
SSM as a recurrence rather than a convolution, switching the per-channel SISO formulation to a
MIMO one to keep the materialized state small.

**Synthetic diagnostic tasks.** Two small synthetic tasks probe specific capabilities of sequence
models:

- *Copying* (Arjovsky et al. 2016): reproduce a block of tokens after a *fixed* offset. The
  input-to-output spacing is constant, so the task tests time-awareness; a kernel that is a spike
  at the right lag, or a fixed-delay recurrence, suffices.
- *Selective Copying* (the denoising variant, Jing et al. 2019): the tokens to memorize are
  placed at *random* positions, interspersed with noise tokens. The spacing between a relevant
  input and its output varies per example, and the model must decide per token whether to keep it.
- *Induction Heads* (Olsson et al. 2022): having seen a bigram such as "Harry Potter", on the
  next occurrence of "Harry" the model must emit "Potter". This associative recall at a
  content-determined moment is strongly predictive of in-context learning in large language models.

**Classical gated RNNs.** LSTM and GRU control information flow with input-dependent gates, e.g.
`h_t = (1 − g_t) h_{t−1} + g_t x_t` with `g_t = σ(Linear(x_t))`. Time-wise-linear gated RNNs
(QRNN, Bradbury et al. 2016; SRU, Lei et al. 2017) made the recurrence parallelizable via the
scan.

**IO-aware kernels.** On GPUs most operations other than dense matrix multiply are bounded by
memory bandwidth, not arithmetic (the roofline model, Williams et al. 2009). FlashAttention (Dao
et al. 2022) exploited this: fuse the computation into a single kernel that keeps intermediates
in fast on-chip SRAM, never writing the large score matrix to slow high-bandwidth memory (HBM),
and recompute what is needed in the backward pass rather than storing it.

## Baselines

**Self-attention / Transformer** (Vaswani et al. 2017; attention of Bahdanau et al. 2015). Each
position attends to all others through a softmax over query-key dot products, and the output is
the weighted sum of values — maximally expressive content-based routing with `O(L²)` time and
memory at training, an `O(L)` key/value cache at inference, and a fixed context window.

**Linear attention** (Katharopoulos et al. 2020). Replaces the softmax with a kernel feature map
so attention can be rewritten as a linear recurrence with a matrix-valued state, giving `O(L)`
inference; in effect a degenerate LTI state space model, making the attention-recurrence duality
explicit.

**S4 / S4D** (Gu et al. 2021/2022; Gupta et al. 2022). Structured SSMs computed as global
convolutions for training and recurrences for inference, with HiPPO-based memory; strong on
continuous-signal and long-range benchmarks. Parameters `(A, B, C)` are fixed across all time
steps.

**S5** (Smith et al. 2023). A diagonal SSM computed recurrently with the parallel scan instead
of a convolution, switching SISO→MIMO to keep the materialized state small. Parameters remain
constant across time steps.

**H3** (Dao et al. 2023). An SSM architecture block: an SSM flanked by two multiplicative gated
connections, preceded by a short local convolution framed as a shift-SSM, and interleaved with a
*separate* MLP block. Generalizes linear attention to use S4.

**Hyena** (Poli et al. 2023). The H3 block with the S4 layer replaced by a long convolution
whose kernel is parameterized by an MLP.

**Gated RNNs / QRNN / SRU** (LSTM; Bradbury et al. 2016; Lei et al. 2017). Input-dependent
gating gives content-dependent behavior, and the time-wise-linear variants admit the parallel
scan.

## Evaluation settings

- *Synthetic diagnostics.* Selective Copying — long sequences (e.g. length 4096), a small
  vocabulary including a noise token, a fixed number of data tokens to memorize, small 2-layer
  models. Induction Heads — trained at a short sequence length (e.g. 256), then evaluated for
  *length generalization* across `2^6`…`2^20` to probe extrapolation.
- *Language modeling.* Pretraining on the Pile, GPT-2/NeoX tokenizers, GPT-3-style model sizes
  (≈125M–1.3B) and recipes (AdamW, cosine schedule with linear warmup, gradient clipping,
  weight decay 0.1), scaling-law curves of perplexity versus compute, and zero-shot downstream
  common-sense reasoning (LAMBADA, HellaSwag, PIQA, ARC-easy/challenge, WinoGrande) via a
  standard evaluation harness.
- *Audio.* Autoregressive waveform modeling (YouTubeMix, mu-law 8-bit) and speech generation
  (SC09), measured by bits-per-byte / negative log-likelihood and sample-quality metrics, across
  sequence lengths up to roughly `10^6`.
- *Genomics.* DNA modeling on the human genome (HG38), next-base-pair prediction, and downstream
  species classification across context lengths from `2^10` to `2^20`.
- *Efficiency.* Throughput and memory of the core sequence operator versus optimized attention
  and FFT convolution, on A100 GPUs in BF16; end-to-end generation throughput and training
  memory versus a same-size Transformer.

## Code framework

The scaffold is a generic causal sequence-model harness: a token embedding, a residual stack of
pre-normalized blocks, an output head, the diagonal-SSM recurrence primitives that already
exist, and one empty sequence-mixing slot. The recurrence primitive below is the LTI building
block known from prior structured SSMs; different mixers can choose how to instantiate the
parameters and how to evaluate the recurrence.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- diagonal-SSM recurrence primitives (already known) -------------------

def discretize_zoh(delta, A, B):
    # delta: (..., D), A: (D, N) diagonal entries, B: (D, N)
    delta_A = delta.unsqueeze(-1) * A
    dA = torch.exp(delta_A)                            # Ā = exp(Δ A)
    # B̄ = ((exp(ΔA) - 1) / A) B, with the A -> 0 limit equal to ΔB.
    scale = torch.where(A.abs() > 1e-8, torch.expm1(delta_A) / A, delta.unsqueeze(-1))
    dB = scale * B
    return dA, dB


def ssm_scan(dA, dB, C, x):
    # dA, dB: (B, L, D, N); C: (..., N); x: (B, L, D) -> y: (B, L, D)
    # Sequential reference for the linear recurrence h_t = Ā_t h_{t-1} + B̄_t x_t.
    B_, L, D, N = dA.shape
    h = x.new_zeros(B_, D, N)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
        ys.append((h * C_at(C, t)).sum(-1))
        # TODO: how to evaluate this recurrence efficiently on a GPU
    return torch.stack(ys, dim=1)


def C_at(C, t):
    # TODO: how C enters the read-out is part of the design
    pass


# --- the open slot: the sequence mixer ------------------------------------

class SequenceMixer(nn.Module):
    """The layer that moves information along the sequence dimension.
    Must be causal and run up to long sequence lengths."""

    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        # TODO: define the mixer's parameters and how it produces its dynamics

    def forward(self, x):              # x: (B, L, D)
        # TODO: produce y: (B, L, D)
        raise NotImplementedError


# --- scaffolding: residual stack, head, training loop ---------------------

class Block(nn.Module):
    def __init__(self, d_model, norm_cls=nn.LayerNorm):
        super().__init__()
        self.norm = norm_cls(d_model)
        self.mixer = SequenceMixer(d_model)

    def forward(self, x):
        return x + self.mixer(self.norm(x))            # pre-norm residual


class LMBackbone(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Block(d_model) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.norm_f(h))


def train_step(model, batch, opt):
    logits = model(batch["input_ids"])
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        batch["input_ids"][:, 1:].reshape(-1),
    )
    loss.backward(); opt.step(); opt.zero_grad()
    return loss
```
