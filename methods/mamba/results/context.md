# Context: Linear-Time Sequence Modeling

## Research question

Sequence models are the backbone of foundation models across language, audio, genomics, and more. The dominant choice — self-attention — routes information densely between every pair of positions in a context window. That density is exactly what makes it effective, but it carries three structural costs: it cannot attend to anything outside a fixed window; training scales quadratically in sequence length; and autoregressive decoding must keep the entire context resident as a key/value cache, so per-step memory and time grow with how much has been generated.

The question is whether a sequence model can match attention's modeling quality while scaling *linearly* in sequence length, with a *constant-size* per-step state at inference. A large literature of subquadratic architectures — linear attention, gated convolutions, recurrent models, structured state space models — answers the efficiency half: they are cheap and they scale. But on information-dense discrete modalities, language above all, they have consistently fallen short of attention. So the real question is sharper: what is the *specific* capability these efficient models lack, and can it be restored without giving up their linear-time scaling?

A useful way to frame it: sequence modeling is the problem of compressing context into a state. Attention sits at one extreme — it compresses nothing (it stores the whole context), which is why it is both powerful and expensive. A recurrent model with a finite state sits at the other extreme — cheap, but only as good as what its fixed-size state managed to keep. The goal is a model whose bounded state nonetheless retains everything that matters, which means the act of compression has to depend on the *content* of the data, not just its position.

## Background

**Classical state space models.** A continuous linear system maps a scalar input signal x(t) to an output y(t) through a hidden state h(t) ∈ ℝ^N:

  h'(t) = A h(t) + B x(t),  y(t) = C h(t).

This is the object behind Kalman filters and linear dynamical systems. To use it on discrete sequences it is discretized with a step size Δ, turning the continuous (Δ, A, B) into discrete (Ā, B̄) via a fixed rule. The most common rule is zero-order hold (ZOH):

  Ā = exp(Δ A),  B̄ = (Δ A)^{-1} (exp(Δ A) − I) · Δ B.

Once discretized, the system is a linear recurrence h_t = Ā h_{t−1} + B̄ x_t, y_t = C h_t.

**HiPPO and principled long-range memory.** A particular choice of A (Gu et al., 2020, HiPPO) makes the hidden state an optimal online compression of the input history into coefficients of orthogonal polynomials. This is what gives state space models a principled mechanism for long-range dependencies and supplies their special initialization.

**Two computation modes and the LTI property.** When (Δ, A, B, C) are held constant across all time steps — the *linear time-invariant* (LTI) case — the same recurrence has a second, equivalent form. Unrolling it gives a convolution by a single fixed kernel:

  K̄ = (C B̄, C Ā B̄, C Ā² B̄, …, C Ā^k B̄, …),  y = x ∗ K̄.

The recurrence is convenient for step-by-step autoregressive inference (constant work per step); the convolution is convenient for training, because the whole sequence is known in advance and the convolution can be evaluated in parallel with an FFT. This duality — that an LTI linear recurrence *is* a global convolution — is the engine of efficiency for this whole family. It holds only because the dynamics are constant in time: a single kernel can be reused at every position precisely because (Ā, B̄, C) do not change.

**Structured SSMs (S4 and descendants).** Computing the convolution kernel for the HiPPO A is not trivial, because that A cannot be stably diagonalized. S4 (Gu et al., 2021/2022) resolves this by writing A in normal-plus-low-rank form and reducing to a diagonal-plus-low-rank computation, then evaluating the kernel through a truncated generating function, a Cauchy kernel, the Woodbury identity, and an inverse FFT, all in O(N + L). Later work (DSS; S4D, Gu et al., 2022) showed the low-rank correction can be dropped: a purely *diagonal* A works essentially as well, so A reduces to N numbers per channel, with simple initializations such as A_n = −(n+1) (real) or A_n = −1/2 + n·i (complex). Applied with a state of size N to a length-L, D-channel, batch-B input, the SSM runs independently per channel; the full hidden state has size B·L·D·N, which is N times larger than the input — the seed of an efficiency problem if it ever had to be materialized.

**The parallel associative scan.** A first-order linear recurrence h_t = a_t h_{t−1} + b_t need not be computed sequentially even when a_t, b_t vary with t. Pairs (a, b) compose under an associative operator

  (a, b) • (a', b') = (a' a,  a' b + b'),

and a work-efficient parallel scan (Blelloch, 1990; Martin & Cundy, 2018) computes all prefixes in O(L) work and O(log L) depth. S5 (Smith et al., 2023) used exactly this to compute a diagonal SSM as a recurrence rather than a convolution, trading the SISO formulation for a MIMO one to keep the materialized state small.

**The LTI limitation, seen on diagnostic tasks.** The constant-dynamics property that buys the convolution also imposes a ceiling. Two synthetic tasks expose it:

- *Copying* (Arjovsky et al., 2016): reproduce a set of tokens after a fixed offset. The spacing is constant, so the task needs only time-awareness; an LTI model solves it trivially — a convolution kernel of exactly the right length, or a fixed-delay recurrence.
- *Selective Copying* (the Denoising task of Jing et al., 2019): the tokens to memorize are placed at *random* positions, interspersed with noise tokens to ignore. Now the model must decide, based on each token's content, whether to keep it. A static convolution kernel cannot model the varying input-to-output spacing, and constant recurrent dynamics cannot choose what to retain. LTI models fail.
- *Induction Heads* (Olsson et al., 2022): having seen a bigram such as "Harry Potter", on the next occurrence of "Harry" the model must emit "Potter". This associative recall at the right moment is strongly predictive of in-context learning in large language models, and again demands content- and context-dependent behavior.

A complementary observation about state size: a larger recurrent state N should compress more context and improve quality, but in a naive recurrence it multiplies the materialized state and the memory traffic by N. This is the expressivity-versus-speed tension — one would like to expand the state without paying for it.

**Gating and its hidden connection to discretization.** Classical gated RNNs (LSTM, GRU) control information flow with input-dependent gates, e.g. h_t = (1 − g_t) h_{t−1} + g_t x_t with g_t = σ(Linear(x_t)). Such gates are powerful but were historically tied to small states (N = 1), heuristic forms, and vanishing-gradient/efficiency problems from their sequential nature. It is a known result (Funahashi & Nakamura, 1993; Tallec & Ollivier, 2018) that gating is what an input-dependent discretization step Δ looks like — i.e. gates are discretized continuous dynamics. Older gated, time-wise-linear RNNs (QRNN, Bradbury et al., 2016; SRU, Lei et al., 2017) already made the recurrence parallelizable via the scan, but kept N = 1 and used heuristic gates.

**IO-aware kernels.** On GPUs most operations other than dense matrix multiply are bounded by memory bandwidth, not arithmetic (Williams et al., 2009). FlashAttention (Dao et al., 2022) exploited this: fuse the attention computation into a single kernel that keeps intermediates in fast on-chip SRAM, never writing the large score matrix to slow high-bandwidth memory (HBM), and recompute what is needed in the backward pass rather than storing it. This is the template for making any memory-bound sequence operator fast.

## Baselines

**Self-attention / Transformer** (Vaswani et al., 2017; attention of Bahdanau et al., 2015). Each position attends to all others via softmax over query-key dot products; output is the weighted sum of values. Maximally expressive routing. Gaps: O(L²) time and memory at training, an O(L) key/value cache at inference, and no mechanism for context beyond the window.

**Linear attention** (Katharopoulos et al., 2020). Replaces softmax with a kernel feature map so that attention can be rewritten as a linear recurrence with a matrix-valued state, giving O(L) inference. It is, in effect, a degenerate LTI state space model and made the attention-recurrence duality explicit. Gap: quality lags softmax attention, particularly on language.

**S4 / S4D** (Gu et al., 2021/2022; Gupta et al., 2022). Structured SSMs computed as global convolutions for training and recurrences for inference, with HiPPO-based memory. Strong on continuous-signal and long-range benchmarks. Gap: strictly LTI — the constant dynamics that enable the convolution prevent any content-based selection, so they underperform on dense discrete data like text.

**S5** (Smith et al., 2023). A diagonal SSM computed recurrently with the parallel scan instead of a convolution; switches SISO→MIMO to keep the materialized state small. Gap: the MIMO move shrinks the effective per-channel recurrent state, and the model is still LTI.

**H3** (Dao et al., 2023). The standard SSM architecture block: an SSM flanked by two multiplicative gated connections, preceded by a short local convolution framed as a shift-SSM, and interleaved with a separate MLP block. Generalized linear attention to use S4. Gap: built from LTI SSMs, so it inherits the selection limitation, and it carries a two-block (mixer + MLP) structure.

**Hyena** (Poli et al., 2023). The H3 block with the S4 layer replaced by a long convolution whose kernel is parameterized by an MLP. Gap: still a global convolution — time-aware but not content-aware — and cannot do fast autoregressive inference directly.

**Gated RNNs / QRNN / SRU** (LSTM; Bradbury et al., 2016; Lei et al., 2017). Input-dependent gating gives content-dependent behavior and, in the time-wise-linear variants, admits the parallel scan. Gaps: no state expansion (N = 1), heuristic gate forms not grounded in SSM theory, and historically poor parameterizations/initializations.

## Evaluation settings

- *Synthetic diagnostics.* Selective Copying — sequences of length 4096, vocabulary of 16 tokens (including a noise token), 16 data tokens to memorize, small 2-layer models (D = 64). Induction Heads — trained at sequence length 256, vocabulary 16, 2-layer models, then tested for length generalization across 2^6 up to 2^20 to probe extrapolation.
- *Language modeling.* Pretraining on the Pile, GPT-2/NeoX tokenizers, GPT-3-style model sizes (125M–1.3B) and training recipes (AdamW, cosine schedule with linear warmup, gradient clipping, weight decay 0.1), scaling-law curves of perplexity versus compute, and zero-shot downstream common-sense reasoning (LAMBADA, HellaSwag, PIQA, ARC-easy/challenge, WinoGrande) via the EleutherAI evaluation harness.
- *Audio.* Autoregressive waveform modeling (YouTubeMix, mu-law 8-bit, vocabulary 256) and speech generation (SC09), measured by bits-per-byte / negative log-likelihood and sample-quality metrics, across sequence lengths up to ~10^6.
- *Genomics.* DNA modeling on the human genome (HG38), next-base-pair prediction, and downstream species classification across context lengths from 2^10 to 2^20.
- *Efficiency.* Throughput and memory of the core sequence operator versus optimized attention (FlashAttention-2) and FFT convolution, on A100 GPUs in BF16; end-to-end generation throughput and training memory versus a same-size Transformer.

## Code framework

The scaffold has a token embedding, a residual stack of normalized blocks, a language-model head, an optimizer and loss, diagonal-SSM recurrence primitives, and one generic sequence-mixing slot.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# --- SSM primitives (diagonal, discretized) -------------------------------

def discretize_diagonal_step(delta, A, B):
    # delta: (..., D), A: (D, N), B: (..., N)
    dA = torch.exp(delta.unsqueeze(-1) * A)          # Ā = exp(ΔA)
    dB = delta.unsqueeze(-1) * B.unsqueeze(-2)       # B̄ ≈ ΔB
    return dA, dB

def ssm_recurrence(dA, dB, C, x):
    # dA, dB: (B, L, D, N); C: (..., N); x: (B, L, D)  -> y: (B, L, D)
    # Sequential reference for a linear recurrence h_t = Ā_t h_{t-1} + B̄_t x_t.
    B_, L, D, N = dA.shape
    h = x.new_zeros(B_, D, N)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
        ys.append((h * C_at(C, t)).sum(-1))
        # TODO: efficient computation path for this recurrence
    return torch.stack(ys, dim=1)

def C_at(C, t):
    # TODO: how C indexes time (constant vs per-step) is a design choice
    pass


# --- the open slot: the sequence mixer ------------------------------------

class SequenceMixer(nn.Module):
    """The layer that moves information along the sequence dimension."""
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        # TODO: architecture, parameters, input-dependent choices, and the
        #       efficient computation path for the recurrence.

    def forward(self, x):           # x: (B, L, D)
        # TODO: produce y: (B, L, D)
        raise NotImplementedError


# --- scaffolding: residual stack, head, training loop ----------------------

class Block(nn.Module):
    def __init__(self, d_model, norm_cls=nn.LayerNorm):
        super().__init__()
        self.norm = norm_cls(d_model)
        self.mixer = SequenceMixer(d_model)
    def forward(self, x):
        return x + self.mixer(self.norm(x))   # pre-norm residual

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
