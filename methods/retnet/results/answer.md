# RetNet — Retentive Network

## Problem

A foundation sequence architecture should hold three properties at once: train in parallel
(like attention), decode in \(O(1)\) time and memory per step (like an RNN), and match
Transformer quality. Attention has parallel training and strong quality but \(O(N)\)-per-step
decoding with a key–value cache that grows linearly; RNNs decode in \(O(1)\) but do not
parallelize over time; prior efficient attention variants recover one corner while giving up
another. The "impossible triangle" is to occupy all three corners simultaneously.

## Key idea

Replace multi-head attention with **multi-scale retention (MSR)**, an operator with one
function that has three equivalent computation forms. It is derived from a linear recurrence
with a state matrix,
\[
s_n = A s_{n-1} + k_n^\top v_n,\qquad o_n = q_n s_n
= \sum_{m\le n} q_n A^{\,n-m} k_m^\top v_m,
\]
which, unrolled, is already a causal weighted sum over the past. Making \(Q=XW_Q,\,K=XW_K\)
content-aware and diagonalizing \(A=\Lambda\,\mathrm{diag}(\gamma e^{i\theta})\,\Lambda^{-1}\)
(absorbing \(\Lambda\) into the projections) turns \(A^{n-m}\) into a scalar decay
\(\gamma^{n-m}\) times a rotary phase \(e^{i(n-m)\theta}\) — an xPos-style relative position
encoding. Deleting softmax and using the decay in its place is what makes the recurrent form
possible.

## The three forms (one function)

**Parallel (training).** With \(Q=(XW_Q)\odot\Theta\), \(K=(XW_K)\odot\overline\Theta\),
\(\Theta_n=e^{in\theta}\), and the combined causal-decay matrix
\[
D_{nm}=\begin{cases}\gamma^{\,n-m}, & n\ge m\\ 0,& n<m,\end{cases}\qquad
\mathrm{Retention}(X)=(QK^\top\odot D)\,V.
\]

**Recurrent (inference, \(O(1)\)).**
\[
S_n=\gamma S_{n-1}+K_n^\top V_n,\qquad \mathrm{Retention}(X_n)=Q_n S_n.
\]
Equivalence: \(S_n=\sum_{m\le n}\gamma^{n-m}K_m^\top V_m\), so
\(Q_nS_n=\sum_{m\le n}\gamma^{n-m}(Q_nK_m^\top)V_m\), which is row \(n\) of \((QK^\top\odot D)V\).

**Chunkwise (long-sequence training, linear time).** Chunk length \(B\), local index \(j\):
\[
R_i=K_{[i]}^\top(V_{[i]}\odot\zeta)+\gamma^{B}R_{i-1},\quad \zeta_j=\gamma^{\,B-1-j},
\]
\[
\mathrm{Retention}(X_{[i]})=\underbrace{(Q_{[i]}K_{[i]}^\top\odot D)V_{[i]}}_{\text{inner-chunk}}
+\underbrace{(Q_{[i]}R_{i-1})\odot\xi}_{\text{cross-chunk}},\quad \xi_j=\gamma^{\,j+1}.
\]
The within-chunk decay-to-boundary \(\gamma^{B-1-j}\) and boundary-to-query decay
\(\gamma^{j+1}\) sum to the true relative decay \(\gamma^{B+j-j'}\), so this equals the parallel
form. Cost \(O(N(B+d)d)\), linear in \(N\).

## Gated multi-scale retention

Use \(h\) heads, each with a **different** decay \(\gamma\) (multi-scale memory horizons),
\(\gamma = 1-2^{-5-\mathrm{arange}(0,h)}\). Normalize each head with GroupNorm (variances
differ across \(\gamma\)), and add a swish gate to restore the nonlinearity softmax provided:
\[
\mathrm{head}_i=\mathrm{Retention}(X,\gamma_i),\quad
Y=\mathrm{GroupNorm}_h(\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)),
\]
\[
\mathrm{MSR}(X)=\big(\mathrm{swish}(XW_G)\odot Y\big)W_O.
\]
GroupNorm's scale-invariance makes three numeric stabilizers free: \(QK^\top/\sqrt d\),
row-normalizing \(D\), and clamped row-magnitude normalization of \(R=QK^\top\odot D\).

Block layout (pre-norm residual, like Transformer): \(Y^l=\mathrm{MSR}(\mathrm{LN}(X^l))+X^l\),
\(X^{l+1}=\mathrm{FFN}(\mathrm{LN}(Y^l))+Y^l\), \(\mathrm{FFN}(X)=\mathrm{gelu}(XW_1)W_2\).
Parameter allocation keeps counts matched to a Transformer: \(W_Q,W_K\in\mathbb{R}^{d\times d}\),
\(W_V,W_G\in\mathbb{R}^{d\times 2d}\), \(W_O\in\mathbb{R}^{2d\times d}\), FFN intermediate \(2d\).
Train with the parallel/chunkwise forms; decode with the recurrent form.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_every_two(x):
    x1, x2 = x[:, :, :, ::2], x[:, :, :, 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RetNetRelPos(nn.Module):
    """Rotary (sin, cos) + per-head decay gamma / decay-mask D."""
    def __init__(self, embed_dim, num_heads, chunk_size=512):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(num_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.chunk_size = chunk_size

    def forward(self, slen, activate_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            return (sin, cos), self.decay.exp()
        index = torch.arange(slen).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])
        mask = torch.tril(torch.ones(slen, slen).to(self.decay))
        mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
        mask = torch.exp(mask * self.decay[:, None, None])          # gamma^{n-m}
        mask = torch.nan_to_num(mask)
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()         # row-normalize D
        return (sin, cos), mask


class MultiScaleRetention(nn.Module):
    def __init__(self, embed_dim, value_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.key_dim = embed_dim // num_heads
        self.head_dim = value_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, num_heads, affine=False)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, _ = v.size()
        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        qk = (qr @ kr.transpose(-1, -2)) * mask                    # (QK^T) (.) D
        qk = qk / qk.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        return (qk @ vr).transpose(1, 2)

    def recurrent_forward(self, qr, kr, v, decay, state):
        bsz = v.size(0)
        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v                                                # K_n^T V_n
        if "prev" in state:
            kv = state["prev"] * decay.view(self.num_heads, 1, 1) + kv   # gamma S_{n-1} + .
        state["prev"] = kv
        return torch.sum(qr * kv, dim=3)                           # Q_n S_n

    def chunk_recurrent_forward(self, qr, kr, v, inner_mask):
        mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
        bsz, tgt_len, _ = v.size()
        chunk_len = mask.size(1)
        num_chunks = tgt_len // chunk_len
        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)
        inner = ((qr @ kr.transpose(-1, -2)) * mask) @ v           # inner-chunk parallel
        kv = kr.transpose(-1, -2) @ (v * value_inner_decay)        # V (.) zeta, zeta=gamma^{B-1-j}
        R = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        R_list = []
        for i in range(num_chunks):
            R_list.append(R)                                       # chunk i reads R_{i-1}
            R = R * cross_decay + kv[:, i]                         # R_i = gamma^B R_{i-1} + .
        R_recurrent = torch.stack(R_list, dim=1)
        cross = (qr * query_inner_decay) @ R_recurrent             # (Q R_{i-1}) (.) xi, xi=gamma^{j+1}
        return (inner + cross).transpose(2, 3)

    def forward(self, x, rel_pos, chunkwise_recurrent=False, state=None):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos
        q, k, v, g = self.q_proj(x), self.k_proj(x), self.v_proj(x), self.g_proj(x)
        k = k * self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        qr, kr = theta_shift(q, sin, cos), theta_shift(k, sin, cos)
        if state is not None:
            out = self.recurrent_forward(qr, kr, v, inner_mask, state)
        elif chunkwise_recurrent:
            out = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            out = self.parallel_forward(qr, kr, v, inner_mask)
        out = self.group_norm(out.reshape(-1, self.num_heads, self.head_dim).transpose(1, 2))
        out = out.transpose(1, 2).reshape(bsz, tgt_len, self.head_dim * self.num_heads)
        return self.out_proj(F.silu(g) * out)                      # (swish(XW_G) (.) Y) W_O


class RetNetBlock(nn.Module):
    def __init__(self, embed_dim, value_dim, num_heads, ffn_dim):
        super().__init__()
        self.retention = MultiScaleRetention(embed_dim, value_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x, rel_pos, chunkwise_recurrent=False, state=None):
        y = self.retention(self.ln1(x), rel_pos, chunkwise_recurrent, state) + x
        return self.fc2(F.gelu(self.fc1(self.ln2(y)))) + y


class RetNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_dim=None):
        super().__init__()
        value_dim = embed_dim * 2                       # value width = 2 * model dim
        ffn_dim = ffn_dim or embed_dim * 2              # FFN intermediate = 2d (param-matched)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rel_pos = RetNetRelPos(embed_dim, num_heads)
        self.layers = nn.ModuleList(
            [RetNetBlock(embed_dim, value_dim, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        rel_pos = self.rel_pos(tokens.size(1))          # parallel training path
        for layer in self.layers:
            x = layer(x, rel_pos)
        return self.lm_head(self.norm(x))
```

Train with `parallel_forward` (or `chunk_recurrent_forward` for long sequences); decode
autoregressively with `recurrent_forward`, carrying the per-layer state for \(O(1)\) cost.
