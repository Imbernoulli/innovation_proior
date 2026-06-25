# Gated DeltaNet — the gated delta rule

Gated DeltaNet is a subquadratic sequence-mixing layer whose fixed-size matrix state is updated by
**both** a data-dependent global decay **and** the delta-rule (error-correcting, content-addressed)
write, trained by a hardware-efficient chunkwise (UT-transform) algorithm. It unifies two previously
separate fixes of additive linear attention: gating (which fades memory globally but cannot localize
removal) and the delta rule (which removes the colliding association precisely but has no decay).

## Problem it solves

A fixed-size-state, constant-memory-inference replacement for softmax attention that needs *both* rapid
global forgetting (to let stale context fade) and targeted content-addressed updates (to overwrite the
specific association a new key collides with). Scalar/diagonal gated linear attention has the first and
not the second; DeltaNet has the second and not the first.

## Key idea

Compose the two mechanisms on their orthogonal axes of one recurrence:

```
S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t v_t k_t^T,   o_t = S_t q_t
```

- `alpha_t in (0,1]` — a **per-head scalar** data-dependent decay (the global eraser).
- `beta_t = sigma(W_beta x_t) in (0,1)` — the learned writing strength of the delta rule (the local
  scalpel); `I - beta_t k_t k_t^T` is a generalized Householder transition.

It is a strict generalization: `alpha_t = 1` recovers **DeltaNet**; `beta_t = 0` drops the delta write
and leaves the pure scalar-gated decay skeleton `S_t = alpha_t S_{t-1}` — the gating part of the
**scalar-gated linear attention / Mamba2** family (that family adds its own write back on top).

## Why the choices

- **Scalar (not diagonal) decay.** The delta rule already gives per-direction control, so the gate only
  needs to do uniform global fading; a scalar does exactly that and, being scalar, telescopes cleanly so
  the delta rule's chunkwise UT-transform training form survives (the decay folds into a chunk-local
  cumulative-sum of log-gates).
- **Mamba2-style gate parameterization.** `alpha_t = exp(g_t)`,
  `g_t = -exp(A_log) * softplus(a_proj(x_t) + dt_bias) <= 0`, with `dt_bias` initialized so the gate
  starts near 1 — a long-memory prior, so the gate does not collapse to halving the state at init.
- **Stability.** With L2-normalized keys the contractive factor along `k_t` is `alpha_t(1 - beta_t)` and
  along orthogonal directions `alpha_t`, both in `[0,1]`.
- **DeltaNet stabilizers kept.** SiLU + L2-norm on q/k, learned `beta_t`, depthwise short conv
  (kernel 4) on q/k/v. **Gated output** RMSNorm + swish gate re-added (the data-dependent decay gives
  head-varying output scale, so the output gate is useful again).

## Chunkwise training (matmul-rich)

DeltaNet's pseudo-value reduction and WY/UT-transform inverse, with the scalar decay threaded through
the chunk-local cumulative log-decay `decay = cumsum(g)`: the key-key similarities that build the
triangular system are weighted by `exp(decay_i - decay_j)`; the carried state is decayed by
`exp(decay_last)` and the chunk keys by `exp(decay_last - decay)`; the inter-chunk read scales the query
by `exp(decay)`. `O(LCd + Ld^2)` FLOPs, `O(L/C)` sequential steps, chunk states recomputed in the
backward.

## Working code

Faithful to `fla.layers.GatedDeltaNet` / `fla.ops.gated_delta_rule`: gate
`g = -A_log.exp() * softplus(a_proj(x) + dt_bias)`, `beta = sigmoid(b_proj(x))`, SiLU short conv on
q/k/v, in-kernel q/k L2-norm, `chunk_gated_delta_rule` for training and `fused_recurrent` for decoding,
gated RMSNorm output. The reference recurrence makes the math explicit; the chunk kernel computes the
same outputs. Head shaping follows `num_heads * head_dim = 0.75 * hidden_size` when `use_gate=True`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale=None):
    """Reference recurrence the chunk kernel parallelizes.
    q, k: (B, H, T, d_k); v: (B, H, T, d_v); beta, g: (B, H, T); g = log decay (<= 0).
    S_t = exp(g_t) (I - beta_t k_t k_t^T) S_{t-1} + beta_t v_t k_t^T,  o_t = S_t q_t."""
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    scale = d_k ** -0.5 if scale is None else scale
    q = q * scale
    S = q.new_zeros(B, H, d_k, d_v)
    o = torch.zeros_like(v)
    for t in range(T):
        k_t, v_t, b_t = k[:, :, t], v[:, :, t], beta[:, :, t]
        S = S * g[:, :, t].exp()[..., None, None]            # alpha_t S_{t-1}  (scalar decay)
        v_old = (S * k_t[..., None]).sum(-2)                 # decayed-state old value
        delta = (v_t - v_old) * b_t[..., None]               # beta_t (v_t - v_old)
        S = S + k_t[..., None] * delta[..., None, :]         # rank-one delta write
        o[:, :, t] = (q[:, :, t][..., None] * S).sum(-2)     # S_t q_t
    return o


def chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=64):
    """Chunkwise UT-transform form with the scalar decay folded into the chunk-local cumulative
    log-decay. q,k: (B,H,T,d_k); v: (B,H,T,d_v); beta,g: (B,H,T)."""
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        q, k, v_beta, k_beta = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v_beta, k_beta))
        g = F.pad(g, (0, pad))
    q, k, v_beta, k_beta, g = (rearrange(x, 'b h (n c) ... -> b h n c ...', c=chunk_size)
                               for x in (q, k, v_beta, k_beta, g))
    decay = g.cumsum(-1)
    dexp = decay.exp()[..., None]
    Lm = (decay[..., :, None] - decay[..., None, :]).tril().exp().tril()
    eye = torch.eye(chunk_size, dtype=q.dtype, device=q.device)
    mask0 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -((k_beta @ k.transpose(-1, -2)) * Lm).masked_fill(mask0, 0)
    for i in range(1, chunk_size):                            # forward substitution -> (I + L)^{-1}
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone()
                                               * attn[..., :, :i].clone()).sum(-2)
    attn = attn + eye                                        # T = (I + L)^{-1}
    u = attn @ v_beta                                        # pseudo-values
    w = attn @ (k_beta * dexp)                               # decay-weighted pseudo-keys
    S = q.new_zeros(B, H, d_k, d_v)
    o = torch.zeros_like(v_beta)
    mask1 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for i in range(q.shape[2]):
        q_i, k_i = q[:, :, i], k[:, :, i]
        intra = (q_i @ k_i.transpose(-1, -2) * Lm[:, :, i]).masked_fill(mask1, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S                    # effective write, corrected for carried state
        o_inter = (q_i * dexp[:, :, i]) @ S                  # query scaled by exp(decay)
        o[:, :, i] = o_inter + intra @ u_i
        d_last = decay[:, :, i, -1]
        S = S * d_last[..., None, None].exp() + \
            (k_i * (d_last[..., None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ u_i
    o = rearrange(o, 'b h n c d -> b h (n c) d')[:, :, :T]
    return o


class GatedDeltaNet(nn.Module):
    def __init__(self, hidden_size, num_heads=6, head_dim=128, expand_v=2.0,
                 conv_size=4, norm_eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * head_dim                  # = 0.75 * hidden_size by convention
        self.value_dim = num_heads * self.head_v_dim
        self.use_pos_emb = False                             # decay + recurrence encode position

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)   # writing strength
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)   # gate input -> Delta_t
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)  # swish output gate
        A = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A));            self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(num_heads) * (torch.log(torch.tensor(0.1))
                       - torch.log(torch.tensor(1e-3))) + torch.log(torch.tensor(1e-3)))
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.q_conv = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, padding=conv_size - 1, bias=False)
        self.k_conv = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, padding=conv_size - 1, bias=False)
        self.v_conv = nn.Conv1d(self.value_dim, self.value_dim, conv_size, groups=self.value_dim, padding=conv_size - 1, bias=False)
        self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.conv_size = conv_size

    def _short_conv(self, x, conv):
        T = x.shape[1]
        return F.silu(conv(x.transpose(1, 2))[..., :T].transpose(1, 2))   # causal depthwise conv + SiLU

    def forward(self, x):                                    # x: (B, T, d)
        q = self._short_conv(self.q_proj(x), self.q_conv)
        k = self._short_conv(self.k_proj(x), self.k_conv)
        v = self._short_conv(self.v_proj(x), self.v_conv)
        q, k = (rearrange(t, 'b s (h d) -> b h s d', d=self.head_k_dim) for t in (q, k))
        v = rearrange(v, 'b s (h d) -> b h s d', d=self.head_v_dim)
        q = F.normalize(q, dim=-1, p=2)                      # L2-normalize q, k (SiLU already applied)
        k = F.normalize(k, dim=-1, p=2)
        beta = rearrange(self.b_proj(x).sigmoid(), 'b s h -> b h s')
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)
        g = rearrange(g, 'b s h -> b h s')                   # log decay <= 0

        # training: chunk kernel; recurrence below computes the same outputs.
        o = chunk_gated_delta_rule(q, k, v, beta, g)

        o = self.o_norm(o)
        o = rearrange(o, 'b h s d -> b s (h d)')
        o = o * F.silu(self.g_proj(x))                       # swish output gate
        return self.o_proj(o)
```
