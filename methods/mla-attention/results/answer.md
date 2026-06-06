# Multi-Head Latent Attention (MLA)

## Problem

Autoregressive decoding in a Transformer is memory-bandwidth bound on its **KV cache**. Standard multi-head attention (MHA) caches `2·n_h·d_h·l` scalars per token (keys + values, all heads, all layers); each generation step streams that whole cache through the accelerator to do a tiny amount of arithmetic. The cache caps batch size and context length and dominates decode latency. The cheap fixes — **MQA** (all heads share one K/V head) and **GQA** (G groups share K/V heads) — shrink the cache by deleting per-head key/value subspaces, which measurably hurts quality on hard benchmarks. The goal: a small cache *and* MHA-level quality, with RoPE intact.

## Key idea

Don't make heads *share* K and V — **compress** them. Down-project each token's input into one small joint latent and cache only that; reconstruct full per-head K and V on the fly via up-projections, which can be **absorbed** into the surrounding query/output projections at inference so K and V are never materialized. Because RoPE's position-dependent rotation breaks that absorption, **decouple position**: carry RoPE on a small separate key/query pair concatenated to the compressed content, so the content branch stays absorbable and a tiny branch carries position.

## Method

Let `h_t ∈ R^d` be the layer input, `n_h` heads of content width `d_h`, decoupled width `d_h^R`, KV/query compression dims `d_c`, `d_c'`.

**Joint KV compression** (cache only `c_t^{KV}`):
- `c_t^{KV} = W^{DKV} h_t` ∈ R^{d_c}   — *cached*
- `k_t^C = W^{UK} c_t^{KV}`, `v_t^C = W^{UV} c_t^{KV}`  (per-head, full-rank fan-out — no head sharing)

**Query compression** (saves training activation memory, not cache):
- `c_t^Q = W^{DQ} h_t`, `q_t^C = W^{UQ} c_t^Q`

**Decoupled RoPE** (`k_t^R` shared across heads, MQA-style; `q_{t,i}^R` per-head):
- `q_{t,i}^R = RoPE(W^{QR} c_t^Q)`,  `k_t^R = RoPE(W^{KR} h_t)` ∈ R^{d_h^R}   — *`k_t^R` cached*
- `q_{t,i} = [q_{t,i}^C ; q_{t,i}^R]`,  `k_{t,i} = [k_{t,i}^C ; k_t^R]`

**Attention** (note the `√(d_h + d_h^R)` scale):
- `o_{t,i} = Σ_{j≤t} softmax_j( q_{t,i}·k_{j,i} / √(d_h + d_h^R) ) · v_{j,i}^C`
- `u_t = W^O [o_{t,1}; … ; o_{t,n_h}]`

**Why it works.** The score splits as `q_{t,i}^C·k_{j,i}^C + q_{t,i}^R·k_j^R`. The content term is `c_t^{Q,T}(W^{UQ}_i)^T W^{UK}_i c_j^{KV}`: the fixed product `(W^{UQ}_i)^T W^{UK}_i` is precomputed (absorb `W^{UK}` into the query projection), so scoring reads only the cached latent `c_j^{KV}` — `K` is never built. On the value side `o_{t,i} = W^{UV}_i Σ_j a_{ij} c_j^{KV}`, so `W^{UV}` folds into `W^O` and `V` is never built. RoPE only lives on the small decoupled branch, where the rotation legitimately sits between the two small vectors and gives the relative-position property without obstructing absorption.

**Cache:** `(d_c + d_h^R)·l` per token. With `d_c = 4 d_h`, `d_h^R = d_h/2`, that is `4.5·d_h·l` — equal to GQA with 2.25 groups, far below MHA's `2 n_h d_h l`, while keeping full per-head content subspaces.

**Sizes used at scale:** `n_h=128`, `d_h=128`, `d_c=512`, `d_c'=1536`, `d_h^R=64`. RMSNorm is applied to the latents `c^{KV}`, `c^Q` (the narrow bottlenecks); RMSNorms and up-projections may be recomputed in back-prop to save activation memory; long-context RoPE rescaling touches only the small decoupled RoPE branch.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    bsz, n_heads, seq_len, head_dim = x.shape
    x = x.view(bsz, n_heads, seq_len, head_dim // 2, 2).transpose(4, 3).reshape_as(x)
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.q_lora_rank = config.q_lora_rank           # d_c'
        self.kv_lora_rank = config.kv_lora_rank         # d_c  -> the cached latent
        self.qk_nope_head_dim = config.qk_nope_head_dim # d_h   (content, absorbable)
        self.qk_rope_head_dim = config.qk_rope_head_dim # d_h^R (decoupled RoPE, shared key)
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # query: W^{DQ} -> RMSNorm -> fused [W^{UQ}; W^{QR}]
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        # KV: fused [W^{DKV}; W^{KR}] -> latent ++ shared rope key
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)  # [W^{UK}; W^{UV}]

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)
        self.softmax_scale = self.q_head_dim ** -0.5

    def forward(self, hidden_states, cos, sin, position_ids,
                past_key_value=None, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)  # shared across heads

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, self.num_heads,
                     self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_pe = apply_rope(q_pe, cos, sin, position_ids)
        k_pe = apply_rope(k_pe, cos, sin, position_ids)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)

        if past_key_value is not None:
            # Generic eager caches store reconstructed tensors.
            key_states, value_states = past_key_value.update(key_states, value_states)

        scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)

        out = torch.matmul(probs, value_states)
        out = out.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        return self.o_proj(out)
```

This eager forward materializes K/V for training/prefill and for a generic reconstructed-tensor cache. In the compressed decode path, the `kv_b_proj` up-projections fold into the query and output projections, so the durable cache is `compressed_kv` plus the rotated `k_pe`.
