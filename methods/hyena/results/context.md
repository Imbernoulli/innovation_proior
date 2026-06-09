## Research question

Sequence models built on self-attention have become the default across language, vision, audio and biology. The operator at their core mixes information across a length-`L` sequence by forming, for every pair of positions, a similarity weight — an `L × L` interaction matrix. This gives any-to-any context but costs `O(L²)` in time and memory. That quadratic scaling is the binding constraint: it caps how much context a model can hold, which rules out whole-textbook contexts, long-form audio, and high-resolution images processed as long sequences.

The question is whether there is a token-mixing operator whose cost grows *subquadratically* in `L` while matching the modeling quality of full attention — ideally with no attention component at all, so that the cost ceiling is genuinely removed rather than merely lowered on average. A satisfactory answer must reproduce what makes attention effective: an operator whose action depends on the input (so a single layer realizes a whole family of transformations), a parameter count that does not grow with `L` (so capacity can be spent elsewhere), and unrestricted context (no built-in locality cutoff).

## Background

**What makes attention work.** Scaled self-attention maps `u ∈ R^{L×D}` through `A(u) = softmax(u Mq Mkᵀ uᵀ / √D)` and `y = A(u) u Mv`. Three properties stand out. (1) It is a *data-controlled* linear operator: `y = A(u) v`, a linear map in `v` whose matrix entries are a nonlinear function of the input — one block encodes an entire family of linear functions and selects among them per input, a mechanism associated with in-context learning. (2) Its parameters (the projections `Mq, Mk, Mv ∈ R^{D×D}`) are *decoupled from `L`*. (3) It has *unrestricted context*: any position can influence any other, subject only to causal masking in autoregressive use. The cost of these properties is the `O(L²)` materialization of `A(u)`. Mechanistic-interpretability probes (associative recall, induction) suggest much of attention's quadratic capacity is unused for language, motivating the search for cheaper operators that retain the three properties.

**Convolutions.** A discrete (aperiodic) convolution `y_t = (h * u)_t = Σ_n h_{t-n} u_n` is a linear map; written as a matrix-vector product it is multiplication by the Toeplitz matrix `S_h` induced by the filter `h`. For a filter to reach arbitrarily far back — the unrestricted-context property — it must be as long as the sequence (a *long* convolution), unlike the short, local filters of standard CNNs.

**Explicit vs. implicit filters.** The classical (CNN / FIR) approach stores the filter as `M` tap values; its memory — how far back the output depends on the input, `∂y_t/∂u_{t-n} = h_n` — equals the filter size `M`, and its parameter count scales linearly with `M`. A length-`L` explicit filter therefore costs `L` parameters per channel, defeating parameter decoupling. The alternative is an *implicit* parametrization `h_t = γ_θ(t)`: the filter value at position `t` is the output of a parametric function of `t`, decoupling filter length from parameter count. Two known families:

- *State-space models* (SSMs; e.g. HiPPO, S4, `gu2020hippo`, `gu2021efficiently`): a linear recurrence `x_{t+1}=A x_t + B u_t`, `y_t = C x_t + D u_t` has impulse response `h_t = C Aᵗ B + D δ_t`. Memory extent is governed by the spectral radius of `A` and is tunable; parameter count is sublinear in `L`. Materializing the kernel requires a structured algebraic form for `A` and careful numerics.
- *Continuous / neural-implicit kernels* (CKConv, FlexConv, `romero2021ckconv`): `γ_θ` is a small feed-forward network mapping a positional encoding of `t` to the filter value, building on neural implicit representations. Free-form filters; evaluated by a single forward pass. Networks have a low-frequency (spectral) bias (`basri2020frequency`), so periodic (sine) activations are used to let the kernel carry high-frequency content. Decaying multi-scale long filters have been found effective (`li2022makes`).

**Fast convolution.** Direct evaluation of a length-`L` convolution is `O(L²)`. The Cooley–Tukey FFT plus the DFT convolution theorem reduces it to `O(L log L)`: zero-pad input and filter to convert the aperiodic convolution into a circular one, whose Toeplitz-becomes-circulant kernel is diagonalized by the DFT, `Ŝ_h = W⁻¹ D_H W`. Then `y = iFFT(D_H · FFT(u))`, computed *without materializing* the `L × L` matrix. The dual identity is also useful: under the common unnormalized-DFT convention, `widehat{x ⊙ u}_ℓ = (1/L) Σ_r x̂_r û_{ℓ-r}`, so pointwise multiplication in time is convolution in frequency up to the normalization factor.

## Baselines

- **Self-attention** (`vaswani2017attention`): `y = A(q,k) v`, the data-control template; sublinear parameters and unrestricted context. Gap: `O(L²)`.
- **State-space sequence models / S4** (`gu2021efficiently`): a long convolution with sublinear parameters and long, tunable memory. Gap: the filter is a fixed linear time-invariant map — the *same* operator for every input, so it is **not data-controlled**; and it requires a structured (normal-plus-low-rank) parametrization and careful conditioning to materialize, with low hardware utilization from iterative kernel construction.
- **Attention-Free Transformers** (`zhai2021attention`): build the operator from gating combined with either softmax (AFT-full) or a single explicit convolution (AFT-conv). Gap: the explicit convolution is local — limited memory and restricted context.
- **Gated State Spaces** (`mehta2022long`): gating composed with one long SSM convolution. Gap: a single gate-and-convolution is a limited data-controlled operator; it underperforms on associative recall.
- **H3** (`dao2022hungry`): motivated by the recall gap of the above, uses two gates and two convolutions — a short (shift-SSM) convolution and a long (diagonal-SSM) one — `z_t = k_t (φ * v)_t`, `y_t = q_t (ψ * z)_t`. Its surrogate interaction matrix factorizes as `A(q,k) = D_q S_ψ D_k S_φ` (diagonal gates `D_q, D_k` times Toeplitz convolution matrices), evaluable in `O(L log L)`. Gap: fixed to three projections and two convolutions, and tied to SSM (not free-form) filters.
- **Frequency-domain and continuous-kernel convolutions**: FNO (`li2020fourier`, filters parametrized by a fixed number of frequency modes) and CKConv (`romero2021ckconv`, FFN-parametrized continuous kernels). Gap: used as plain convolutions — long-range but not data-controlled.
- **Structured fast-matvec decompositions** (butterfly / Monarch, `dao2019learning`, `dao2022monarch`): represent a dense matrix as a product of sparse factors, with the number of factors trading off against expressivity.

## Evaluation settings

- **Mechanistic / in-context-learning synthetics** (`elhage2021mathematical`, `garg2022can`): associative recall (extract the value for a queried key — a data-controlled shift), majority voting / counting (densely active interactions), in-context learning of linear functions (real-valued `x_1, w x_1, …, x_n → w x_n`), and arithmetic (digit addition). Difficulty scaled by sequence length (`1k`–`131k`) and vocabulary size (`10`–`40`). Small probes: 2-layer, width-64 models; AdamW, cosine schedule with linear warmup.
- **Autoregressive language modeling**: WikiText103 (GPT-2 tokenizer, vocabulary `50257`) and The Pile (`gao2020pile`) at the `125M`–`1.3B` scale; sequence length `2024`; metric perplexity, with FLOPs accounted per-layer (`hoffmann2022training`).
- **Downstream**: SuperGLUE (`wang2019superglue`) and LAMBADA (`paperno2016lambada`), evaluated by greedy generation or logit scoring.
- **Vision**: ImageNet-1k from scratch with a ViT backbone whose token-mixing layer is swapped out (`dosovitskiy2020image`, `yuan2021tokens`); CIFAR-10 in sequential and 2D settings, compared against S4 / S4ND (`nguyen2022s4nd`).
- **Efficiency**: wall-clock and memory of the mixing layer versus dense attention and FlashAttention (`dao2022flashattention`) at lengths up to `64k`.

Natural comparison points: Transformer / FlashAttention, H3, GSS, AFT-conv, RWKV, GPTNeo; and, for the convolution-parametrization study, explicit `Conv1d`, FNO, SSM (S4), transfer-function, and CKConv filters.

## Code framework

The pieces below already exist: linear projections, an FFT-based long convolution, depthwise `Conv1d`, and a standard residual block / training loop. The mixing operator is the empty slot.

```python
import math
import torch
import torch.nn as nn
from einops import rearrange


def fftconv(u, k, D):
    # Long aperiodic convolution via the DFT convolution theorem, O(L log L).
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen                       # zero-pad: circular FFT conv == linear conv
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    return (y + u * D.unsqueeze(-1)).to(dtype=u.dtype)


class LongFilter(nn.Module):
    # A filter as long as the sequence, with a parameter count independent of L.
    def __init__(self, d_model, seq_len, **kwargs):
        super().__init__()
        raise NotImplementedError

    def filter(self, L):
        # TODO: produce the length-L filter.
        raise NotImplementedError

    def forward(self, x, L, k=None, bias=None):
        # TODO: apply a generated length-L filter to x with fftconv.
        raise NotImplementedError


class SequenceMixer(nn.Module):
    # Token-mixing operator: maps u (B, L, D) -> y (B, L, D).
    # TODO: a subquadratic operator meeting the three properties above.
    def __init__(self, d_model, l_max, order=2, filter_order=64, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, u):
        # TODO: fill in the mixer.
        raise NotImplementedError


class Block(nn.Module):
    # Standard pre-norm residual block; the mixer is the only unknown.
    def __init__(self, d_model, l_max, ffn_mult=4, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = SequenceMixer(d_model, l_max, **kwargs)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```
