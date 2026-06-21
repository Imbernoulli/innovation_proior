Self-attention has become the default sequence-mixing operator because it is a data-controlled linear map with unrestricted context and a parameter count that does not grow with sequence length. The trouble is that the interaction matrix it builds is L × L, so time and memory scale as O(L²). That quadratic wall caps the context length of practical models. Earlier subquadratic ideas — linearized or sparse attention, attention-free transformers, gated state spaces — lower the cost, but they usually trade away expressivity and end up needing real attention layers bolted back on to recover quality. The reason is that they approximate the attention matrix rather than reproduce the property that makes attention work.

The useful property is not the softmax matrix itself, but the fact that the operator is a family of linear maps selected by the input. A single attention layer computes y = A(u) v, which is linear in v but whose mixing matrix A(u) depends nonlinearly on u. Any substitute must keep that data control, plus unrestricted context and parameter decoupling from L, while avoiding the O(L²) materialization of the mixing matrix. The way forward is to build the operator out of cheap primitives that already have these properties, rather than to approximate attention.

The method I propose is Hyena, from the Hyena Hierarchy. It keeps the three attention properties by interleaving two subquadratic primitives: long implicit convolutions and input-dependent gating. A long convolution y = h * u has unrestricted context as long as the filter h is as long as the sequence, and it can be evaluated in O(L log L) by the FFT convolution theorem without ever forming the L × L Toeplitz matrix. To keep parameters independent of L, the filter is not stored as taps; instead it is generated from the position index by a small feed-forward network: h_t = Window(t) · FFN(PositionalEncoding(t)). The positional encoding uses truncated Fourier features so the filter can carry high-frequency structure, and the FFN uses sine activations to counter the low-frequency bias of ordinary MLPs. An exponential-decay window with per-channel decay rates biases filters toward smooth decay while allowing a floor of long-range response, giving different channels different effective memory lengths at initialization.

Convolution alone, however, is a fixed linear time-invariant map; it is not data-controlled. Hyena injects data dependence by gating the signal between convolutions with element-wise projections of the input. Because a time-domain diagonal matrix does not commute with the DFT basis, a gate sandwiched between two convolutions cannot be folded into a single convolution. The resulting operator is genuinely input-conditioned. For order N, Hyena takes N+1 linear projections of the input (x¹, ..., x^N, v) and applies the recurrence z¹ = v, z^{n+1} = x^n ⊙ (h^n * z^n), y = z^{N+1}. In matrix form this is H(u) = D_x^N S_h^N ··· D_x¹ S_h¹, where each D_x^n is a data-controlled diagonal and each S_h^n is a long-convolution Toeplitz matrix. The operator is evaluated right-to-left on v, never materialized, at O(N L log L) cost.

This construction generalizes prior work: order one recovers a gated state-space layer, and order two recovers the H3 mechanism. In the order-two case the surrogate interaction matrix is A(q,k) = D_q S_ψ D_k S_φ, which is a dense data-controlled L × L matrix exactly like attention but never formed explicitly. Causality for autoregressive modeling follows automatically by using causal filters and zero-padding to length 2L before the FFT, so the circular convolution equals the linear causal one. A short explicit depthwise convolution is applied to the projections first to supply cheap local mixing, after which the implicit long filters handle the global context.

The full layer cost on u ∈ R^{L×D} is O(N D L(log L + D)), subquadratic in L, with no attention component at all. The implementation below realizes the projections, the short depthwise convolution, and the order-N recurrence of gates plus implicit FFT-based long convolutions.

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
    return (y + u * D.unsqueeze(-1)).to(dtype=u.dtype)


class Sin(nn.Module):
    # Periodic activation lets the filter MLP represent high-frequency content.
    def __init__(self, dim, w=10):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    # Truncated complex-exponential (Fourier-feature) encoding of position t.
    def __init__(self, emb_dim, seq_len):
        super().__init__()
        t = torch.linspace(0, 1, seq_len)[None, :, None]
        bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register_buffer("z", z)
        self.register_buffer("t", t)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    # Window(t) = exp(-t * delta) + shift, with delta varied across channels.
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
    def __init__(self, d_model, emb_dim=3, order=64, seq_len=1024,
                 w=10, num_inner_mlps=2):
        super().__init__()
        assert emb_dim % 2 != 0 and emb_dim >= 3
        self.d_model = d_model
        self.bias = nn.Parameter(torch.randn(d_model))
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
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None):
        if k is None:
            k = self.filter(L)
        if k.dim() == 3:
            k = rearrange(k, "1 l d -> d l")
        bias = self.bias if bias is None else bias
        return fftconv(x, k, bias)


class SequenceMixer(nn.Module):
    def __init__(self, d_model, l_max, order=2, filter_order=64,
                 short_filter_order=3, dropout=0.0, **filter_args):
        super().__init__()
        assert order >= 2
        self.d_model, self.l_max, self.order = d_model, l_max, order
        self.in_proj = nn.Linear(d_model, (order + 1) * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.short_filter = nn.Conv1d(
            (order + 1) * d_model, (order + 1) * d_model,
            kernel_size=short_filter_order, groups=(order + 1) * d_model,
            padding=short_filter_order - 1,
        )
        self.filter_fn = LongFilter(
            d_model * (order - 1), order=filter_order, seq_len=l_max, **filter_args
        )

    def forward(self, u):
        L = u.size(-2)
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")
        uc = self.short_filter(u)[..., :L]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(L)
        k = rearrange(k, "1 l (o d) -> o d l", o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, "(o d) -> o d", o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, L, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], "b d l -> b l d")
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
