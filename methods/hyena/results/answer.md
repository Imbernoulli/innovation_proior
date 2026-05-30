# Hyena Hierarchy

## Problem

Self-attention is the dominant sequence-mixing operator, but it costs `O(L²)` time and memory in the sequence length `L`. That quadratic cost caps the context a model can attend to. Existing subquadratic substitutes (linearized / low-rank / sparse attention, attention-free transformers, gated state spaces) trade away expressivity and, in practice, need to be hybridized with real attention to reach Transformer quality. The goal is a subquadratic token-mixing operator that matches attention's quality with no attention at all.

## Key idea

Keep the three properties that make attention work and drop the quadratic cost.

Attention is a **data-controlled** linear operator: `y = A(u) v` with `A(u) = softmax(u Mq Mkᵀ uᵀ / √D)`, a matrix whose entries are a nonlinear function of the input. It also has **sublinear parameter scaling** (its parameters are the projections, independent of `L`) and **unrestricted context** (any position can influence any other).

Hyena reproduces all three from two subquadratic primitives:

1. **Long implicit convolutions.** A convolution `(h * u)` reaches arbitrarily far back if the filter `h` is as long as the sequence — that gives unrestricted context. To keep the parameter count independent of `L`, the filter is not stored as taps but *generated*: `h_t = Window(t) · (FFN ∘ PositionalEncoding)(t)`. A small FFN maps a positional encoding of `t` to the filter value, so the filter has length `L` with `O(1)`-in-`L` parameters. The convolution is evaluated by FFT in `O(L log L)`.
2. **Data-controlled gating.** Element-wise multiplication by input-dependent projections `xⁿ = uMⁿ`. This is the cheap (`O(L)`) mechanism that makes the operator depend nonlinearly on the input.

Interleaving `N` gates with `N` long convolutions gives the order-`N` Hyena operator.

## The operator

Take `N+1` linear projections of the input `u`: `(x¹, …, x^N, v)`. With learnable filters `h¹, …, h^N`:

```
z¹  = v
z^{n+1}_t = x^n_t · (h^n * z^n)_t ,   n = 1, …, N
y   = z^{N+1}
```

equivalently

```
y = x^N ⊙ (h^N * (x^{N-1} ⊙ (h^{N-1} * ( … (x¹ ⊙ (h¹ * v)))))).
```

In matrix form, with `D_x^n = diag(x^n)` and `S_h^n` the Toeplitz matrix of filter `h^n`,

```
y = H(u) v = D_x^N S_h^N ··· D_x¹ S_h¹ v.
```

`H(u)` is data-controlled (its entries are functions of `u`), has unrestricted context (the long convs), and sublinear parameters (the implicit filters). It is evaluated **without materializing** `H(u)`, in `O(N L log L)` time; the full cost on `u ∈ R^{L×D}` is `O(N D L(log L + D))`.

Order `N = 1` recovers Gated State Spaces; order `N = 2` recovers the H3 mechanism `A(q,k) = D_q S_ψ D_k S_φ` — both as special cases.

## Filter parametrization

- `h_t = Window(t) · FFN(PositionalEncoding(t))`.
- `PositionalEncoding(t)` is a truncated complex-exponential (Fourier-feature) basis `[t, Re rho_0(t), ..., Re rho_{K-1}(t), Im rho_0(t), ...]`, with `rho_j(t) = exp(-i f_j 2πt/L)` and frequencies `f_j` spread from almost zero to the chosen band limit; `K` bands set the filter's spectral cut-off at init.
- The FFN uses **sine** activations so it can represent high-frequency filter content, countering networks' low-frequency bias.
- `Window(t) = exp(-α t) (+ bias)` biases filters toward exponential decay; `α` varies across channels, giving a multi-scale set of filter lengths.
- **Causality** (for autoregressive LM): evaluate `h` at `t = 0, …, L-1` and zero-pad input and filter to at least `2L-1` before the FFT (implemented below as `2L`), so the FFT computes the linear (aperiodic) causal convolution with no future leakage.
- A short explicit depthwise convolution (size 3) is applied to the projections before the long convolutions, supplying cheap local/shift mixing.
- In the implementation, the short depthwise convolution supplies the first local convolutional factor; the implicit filter module therefore generates `order - 1` long filters, with the first projection used as the final gate.

## Code

```python
import math
import torch
import torch.nn as nn
from einops import rearrange


def fftconv(u, k, D):
    # Long (aperiodic) convolution via the DFT convolution theorem, O(L log L).
    # Pad to 2L so the circular FFT convolution equals the linear/causal one.
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    return (y + u * D.unsqueeze(-1)).to(dtype=u.dtype)  # learned residual term


class Sin(nn.Module):
    # High-frequency periodic activation: lets the filter FFN represent
    # high-frequency content, countering the low-frequency bias of MLPs.
    def __init__(self, dim, w=10):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    # Truncated complex-exponential (Fourier-feature) encoding of position t.
    def __init__(self, emb_dim, seq_len):
        super().__init__()
        t = torch.linspace(0, 1, seq_len)[None, :, None]          # 1, L, 1
        bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)                # 1, L, emb_dim
        self.register_buffer("z", z)
        self.register_buffer("t", t)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    # Window(t) = exp(-t * delta) + shift, with delta varied across channels
    # so filters have a spread of effective lengths at initialization.
    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5,
                 target=1e-2, shift=0.0):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register_buffer("deltas", deltas)

    def forward(self, t, h):
        decay = torch.exp(-t * self.deltas.abs())
        return h * (decay + self.shift)


class LongFilter(nn.Module):
    # Implicit long filter: h_t = Window(t) * FFN(PositionalEncoding(t)).
    # Filter length is L but parameter count is independent of L.
    def __init__(self, d_model, emb_dim=3, order=64, seq_len=1024,
                 w=10, num_inner_mlps=2):
        super().__init__()
        assert emb_dim % 2 != 0 and emb_dim >= 3
        self.d_model = d_model
        self.bias = nn.Parameter(torch.randn(d_model))            # learned residual term
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)
        act = Sin(dim=order, w=w)
        self.implicit_filter = nn.Sequential(nn.Linear(emb_dim, order), act)
        for _ in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)
        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
        self.modulation = ExponentialModulation(d_model)

    def filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)        # FFN(PositionalEncoding(t))
        h = self.modulation(t, h)          # * Window(t)
        return h

    def forward(self, x, L, k=None, bias=None):
        if k is None:
            k = self.filter(L)
        if k.dim() == 3:
            k = rearrange(k, "1 l d -> d l")   # (d_model, L) filter
        bias = self.bias if bias is None else bias
        return fftconv(x, k, bias)


class SequenceMixer(nn.Module):
    def __init__(self, d_model, l_max, order=2, filter_order=64,
                 short_filter_order=3, dropout=0.0, **filter_args):
        super().__init__()
        assert order >= 2
        self.d_model, self.l_max, self.order = d_model, l_max, order
        # N+1 projections (the x^n and v), produced jointly.
        self.in_proj = nn.Linear(d_model, (order + 1) * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Short explicit depthwise conv: cheap local / shift mixing on projections.
        self.short_filter = nn.Conv1d(
            (order + 1) * d_model, (order + 1) * d_model,
            kernel_size=short_filter_order, groups=(order + 1) * d_model,
            padding=short_filter_order - 1,
        )
        # One implicit long filter per long convolution (order - 1 of them; the
        # first projection gates the final output without a preceding conv).
        self.filter_fn = LongFilter(
            d_model * (order - 1), order=filter_order, seq_len=l_max, **filter_args
        )

    def forward(self, u):
        L = u.size(-2)
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")
        uc = self.short_filter(u)[..., :L]                 # local mix, then truncate pad
        *x, v = uc.split(self.d_model, dim=1)              # x^1..x^N, v

        k = self.filter_fn.filter(L)                       # (1, L, d_model*(order-1))
        k = rearrange(k, "1 l (o d) -> o d l", o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, "(o d) -> o d", o=self.order - 1)

        # Recurrence over the implicit long filters; x[0] is the remaining gate.
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)                      # data-controlled gate  D_x^n
            v = self.filter_fn(v, L, k=k[o], bias=bias[o]) # long conv  S_h^n  (FFT)

        y = rearrange(v * x[0], "b d l -> b l d")          # final gate
        return self.out_proj(y)


class Block(nn.Module):
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
