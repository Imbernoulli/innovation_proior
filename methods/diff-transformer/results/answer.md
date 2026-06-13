# Differential Transformer (DIFF Transformer)

## Problem

A single-softmax attention head produces a strictly-positive probability distribution over all context
positions. It therefore cannot assign exactly zero weight to irrelevant tokens and has no mechanism to
take mass back off a position — so over a long context a real fraction of the attention mass leaks onto
irrelevant tokens (attention "noise", the lost-in-the-middle failure). We want a drop-in attention
operator that concentrates on the relevant context and *cancels* the common noise floor, at matched
parameters and FLOPs.

## Key idea

Form attention as the **difference of two softmax maps**, the way a differential amplifier rejects
common-mode noise:

```
DiffAttn(X) = ( softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d)) ) V
```

with `Q = [Q1; Q2]`, `K = [K1; K2]`. Both maps carry the same correlated noise floor over irrelevant
tokens (common-mode), so the subtraction cancels it; the signal, where the first map spikes, survives.
The result is a *signed* attention pattern — it can drive an irrelevant token's contribution to zero or
below, which a single positive softmax structurally cannot.

## The pieces that make it work

- **Re-parameterized, depth-scaled `lambda`.** Learn
  `lambda = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init` (four learnable
  `head_dim` vectors, init `N(0, 0.1)`), so at init `lambda ~ lambda_init` with well-scaled, signed
  gradients. The init grows with depth: `lambda_init = 0.8 - 0.6 * exp(-0.3 * (l - 1))` (0.2 at layer 1,
  rising toward 0.8) — gentle cancellation early, strong cancellation deep.
- **Matched budget by halving the head dim.** Each logical head uses `head_dim = d_model / h / 2`, with
  `2h` query/key sub-heads of that dimension and `h` value heads of dimension `2*head_dim`. Total q/k/v
  width is `d_model` — same parameters and FLOPs as a vanilla `h`-head Transformer; the doubling is
  absorbed by the halving.
- **Per-head normalization + fixed gain.** The subtraction makes heads heterogeneous in scale, so apply
  a per-head RMSNorm (GroupNorm across heads) to each head's `2*head_dim` output, then rescale by the
  *fixed* constant `(1 - lambda_init)` to compensate the gain lost to subtraction. Fixed (not the learned
  `lambda`) so the compensation is stable, letting vanilla Transformer hyperparameters transfer.

Everything else — q/k/v projections, RoPE, causal mask, output projection, residual placement — is the
vanilla block. Only the score-formation step changes, from one positive softmax to a difference of two.

## Why it cancels noise rather than adding a free knob

On irrelevant tokens the two halves see the same content, so their floor patterns are correlated and
`A1 - lambda A2 -> 0`. On relevant tokens the model has an incentive to make the halves disagree, so the
difference is large. The trivial `lambda = 0` (back to single softmax) is not the optimum, because
cancelling the floor produces a cleaner value average and genuinely lowers loss. The signed, potentially
sparse pattern expresses "attend to these, actively ignore those."

## Code

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def lambda_init_fn(depth):
    # depth: 0-based layer index. Paper schedule (1-based l): 0.8 - 0.6*exp(-0.3*(l-1)).
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out


class MultiheadDiffAttn(nn.Module):
    def __init__(self, d_model, n_heads, depth):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads // 2          # halved so doubled q/k is budget-matched
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, rope, attn_mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, 2 * self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, 2 * self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, 2 * self.head_dim)

        q, k = rope(q, k)
        q = q.transpose(1, 2)                            # (B, 2H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)                            # (B,  H, T, 2*head_dim)

        q = q * self.scaling
        att = torch.matmul(q, k.transpose(-1, -2))       # (B, 2H, T, T)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=att.dtype), 1)
        att = att + attn_mask
        att = F.softmax(att, dim=-1, dtype=torch.float32).type_as(att)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        att = att.view(B, self.n_heads, 2, T, T)
        att = att[:, :, 0] - lambda_full * att[:, :, 1]  # (B, H, T, T), signed

        o = torch.matmul(att, v)                         # (B, H, T, 2*head_dim)
        o = self.subln(o)
        o = o * (1.0 - self.lambda_init)                 # fixed gain compensation
        o = o.transpose(1, 2).reshape(B, T, self.n_heads * 2 * self.head_dim)
        return self.out_proj(o)
```

Reference: Ye, Dong, Xia, Sun, Zhu, Huang, Wei, "Differential Transformer", 2024, arXiv:2410.05258
(ICLR 2025). Canonical implementation: `microsoft/unilm`, `Diff-Transformer/multihead_diffattn.py`.
