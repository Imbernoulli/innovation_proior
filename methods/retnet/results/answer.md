# RetNet - Retentive Network

## Problem

A foundation sequence architecture should hold three properties at once: train in parallel
(like attention), decode in \(O(1)\) time and memory per step (like an RNN), and match
Transformer quality. Attention has parallel training and strong quality but \(O(N)\)-per-step
decoding with a key-value cache that grows linearly; RNNs decode in \(O(1)\) but do not
parallelize over time; prior efficient variants recover one corner while giving up another.
The "impossible triangle" is to occupy all three corners simultaneously.

## Key idea

Replace multi-head attention with **multi-scale retention (MSR)**, an operator with one
function and three equivalent computation forms. It starts from a linear recurrence with a
state matrix:
\[
s_n = A s_{n-1} + k_n^\top v_n,\qquad
o_n = q_n s_n = \sum_{m\le n} q_n A^{n-m} k_m^\top v_m .
\]
Making \(Q=XW_Q,\,K=XW_K\) content-aware and diagonalizing
\(A=\Lambda\,\mathrm{diag}(\gamma e^{i\theta})\Lambda^{-1}\), with the basis absorbed into
the projections, turns the power \(A^{n-m}\) into a relative-position factor. With scalar
\(\gamma\) per head:
\[
o_n=\sum_{m\le n}\gamma^{n-m}(Q_n e^{in\theta})(K_m e^{im\theta})^\dagger V_m .
\]
The dagger supplies the conjugate phase on the key side; in code this is
implemented by applying the same RoPE rotation to both \(q\) and \(k\) and then taking their
real dot product.

## The three forms

**Parallel training.** With \(Q=(XW_Q)\odot\Theta\),
\(K=(XW_K)\odot\overline{\Theta}\), \(\Theta_n=e^{in\theta}\), and
\[
D_{nm}=\begin{cases}\gamma^{n-m}, & n\ge m\\ 0,& n<m,\end{cases}
\]
the retention output is
\[
\mathrm{Retention}(X)=(QK^\top\odot D)V .
\]

**Recurrent inference.**
\[
S_n=\gamma S_{n-1}+K_n^\top V_n,\qquad
\mathrm{Retention}(X_n)=Q_n S_n .
\]
Unrolling gives \(S_n=\sum_{m\le n}\gamma^{n-m}K_m^\top V_m\), so row \(n\) matches the
parallel form exactly.

**Chunkwise long-sequence training.** Let a chunk have length \(B\), query local index \(j\),
and key local index \(j'\):
\[
R_i=K_{[i]}^\top(V_{[i]}\odot\zeta)+\gamma^B R_{i-1},\qquad
\zeta_{j'}=\gamma^{B-1-j'} ,
\]
\[
\mathrm{Retention}(X_{[i]})=(Q_{[i]}K_{[i]}^\top\odot D)V_{[i]}
 +(Q_{[i]}R_{i-1})\odot\xi,\qquad \xi_j=\gamma^{j+1}.
\]
The exponents \((B-1-j')+(j+1)\) add to \(B+j-j'\), the true distance from a key in the
previous chunk to a query in the current chunk. This gives linear-in-\(N\) long-sequence
training while preserving within-chunk parallelism.

## Gated multi-scale retention

Use \(h\) heads, each with a different fixed decay. The default is
\[
\gamma = 1-2^{-5-\mathrm{arange}(0,h)} ,
\]
while the large-model regime uses a log-spaced range
\[
\gamma = 1-\exp(\mathrm{linspace}(\log(1/32),\log(1/512),h)).
\]
Each head runs retention at its own scale:
\[
\mathrm{head}_i=\mathrm{Retention}(X,\gamma_i),\qquad
Y=\mathrm{GroupNorm}_h(\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)),
\]
\[
\mathrm{MSR}(X)=\big(\mathrm{swish}(XW_G)\odot Y\big)W_O .
\]
Conceptually this is per-head GroupNorm; I realize it in code
as per-head `RMSNorm(head_dim, elementwise_affine=False)`. Its scale-invariance is what makes
the stabilizers function-preserving: \(QK^\top/\sqrt d\), \(D_{nm}/\sqrt{\sum_i D_{ni}}\), and
row-magnitude normalization of \(R=QK^\top\odot D\).

The block is pre-norm residual MSR followed by FFN. In the full implementation I
use RMSNorm pre-norm, a GLU feed-forward layer, DeepNorm residual scaling when enabled, and
`W_V,W_G` projecting to the configured value dimension, typically \(2d\), with the FFN reduced to
\(2d\) for parameter matching.

## Reference-shaped code

This is the core artifact in implementation form. A production wrapper would wrap the projections in a
`MultiwayWrapper`, but the retention math below is the same.

```python
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x if self.weight is None else x * self.weight


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RetNetRelPos(nn.Module):
    def __init__(self, embed_dim, num_heads, chunk_size=512):
        super().__init__()
        angle = 1.0 / (
            10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2)
        )
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(
            1 - 2 ** (-5 - torch.arange(num_heads, dtype=torch.float))
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = chunk_size

    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            return (sin, cos), self.decay.exp()

        index = torch.arange(slen).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])

        if chunkwise_recurrent:
            b = self.recurrent_chunk_size
            block_index = torch.arange(b).to(self.decay)
            tri = torch.tril(torch.ones(b, b).to(self.decay))
            raw = torch.masked_fill(
                block_index[:, None] - block_index[None, :],
                ~tri.bool(),
                float("inf"),
            )
            raw = torch.nan_to_num(torch.exp(raw * self.decay[:, None, None]))

            value_inner_decay = raw[:, -1] / raw[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = raw.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = raw / scale

            cross_decay = torch.exp(self.decay * b)[:, None, None]
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (
                scale / raw[:, -1].sum(dim=-1)[:, None, None]
            )
            return (sin, cos), (
                inner_mask,
                cross_decay,
                query_inner_decay,
                value_inner_decay,
            )

        tri = torch.tril(torch.ones(slen, slen).to(self.decay))
        raw = torch.masked_fill(index[:, None] - index[None, :], ~tri.bool(), float("inf"))
        mask = torch.nan_to_num(torch.exp(raw * self.decay[:, None, None]))
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
        return (sin, cos), mask


class MultiScaleRetention(nn.Module):
    def __init__(self, embed_dim, value_dim, num_heads, gate_fn="swish", layernorm_eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = value_dim // num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.gate_fn = F.silu if gate_fn == "swish" else F.gelu

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)
        self.group_norm = RMSNorm(
            self.head_dim, eps=layernorm_eps, elementwise_affine=False
        )

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, _ = v.size()
        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        qk = (qr @ kr.transpose(-1, -2)) * mask
        qk = qk / qk.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        return (qk @ vr).transpose(1, 2)

    def recurrent_forward(self, qr, kr, v, decay, incremental_state):
        bsz = v.size(0)
        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            old = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(
                self.num_heads, 1, 1
            )
            new = kv / scale.sqrt().view(self.num_heads, 1, 1)
            kv = old + new
        else:
            scale = torch.ones_like(decay)
        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale
        return torch.sum(qr * kv, dim=3)

    def chunk_recurrent_forward(self, qr, kr, v, inner_mask):
        mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
        bsz, tgt_len, _ = v.size()
        chunk_len = mask.size(1)
        assert tgt_len % chunk_len == 0
        num_chunks = tgt_len // chunk_len

        qr = qr.view(
            bsz, self.num_heads, num_chunks, chunk_len, self.key_dim
        ).transpose(1, 2)
        kr = kr.view(
            bsz, self.num_heads, num_chunks, chunk_len, self.key_dim
        ).transpose(1, 2)
        v = v.view(
            bsz, num_chunks, chunk_len, self.num_heads, self.head_dim
        ).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)
        qk = (qr @ kr_t) * mask
        inner_scale = qk.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk = qk / inner_scale
        inner_output = qk @ v

        kv = kr_t @ (v * value_inner_decay)
        kv_recurrent, cross_scale = [], []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = (
                kv_state.detach()
                .abs()
                .sum(dim=-2, keepdim=True)
                .max(dim=-1, keepdim=True)
                .values
                .clamp(min=1)
            )
        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        all_scale = torch.maximum(inner_scale, cross_scale)
        inner_output = inner_output / (all_scale / inner_scale)
        cross_output = ((qr * query_inner_decay) @ kv_recurrent) / (
            all_scale / cross_scale
        )
        return (inner_output + cross_output).transpose(2, 3)

    def forward(self, x, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k = k * self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)

        output = self.group_norm(output).reshape(
            bsz, tgt_len, self.head_dim * self.num_heads
        )
        return self.out_proj(self.gate_fn(g) * output)
```
