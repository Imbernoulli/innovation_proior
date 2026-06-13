# Multi-head Latent Attention (MLA), distilled

MLA is a causal self-attention structure that shrinks the inference KV cache by caching a small
*joint low-rank latent* per token instead of the realized per-head keys and values, then
reconstructing distinct per-head keys/values from that latent on the fly. The reconstruction
up-projections absorb into the query and output projections at inference, so the latent is as
cheap to attend over as a single shared key/value head — giving multi-query-scale cached bytes
with multi-head-scale capacity. A small *decoupled* rotary slice carries position so the
absorption survives RoPE.

## Problem it solves

Autoregressive decoding caches keys and values for every past token, head, and layer. For MHA
that is `2 · n_h · d_h · l` scalars per token, and it dominates long-context / large-batch
inference: it caps batch and sequence length, and incremental decoding is memory-bandwidth-bound
because each step reloads the whole K,V cache (the memory-to-compute ratio scales like `n/d`).
Prior fixes (MQA, GQA) shrink the cache only by reducing the *count* of key/value heads, paying
in quality. MLA decouples cached bytes from head count.

## Key idea

Cache a joint latent `c_t^{KV} = W^{DKV} h_t ∈ R^{d_c}` with `d_c ≪ n_h d_h`. Reconstruct each
head's key and value from it: `k_{t,i}^C = W^{UK}_i c_t^{KV}`, `v_{t,i}^C = W^{UV}_i c_t^{KV}`.

- **Each head gets its own up-projection**, so all heads read the same cached latent but
  reconstruct *different* keys/values — the per-head diversity MQA destroyed, kept at a tiny
  cache.
- **Absorption (inference).** The content score is
  `q_{t,i}^T (W^{UK}_i c_j^{KV}) = (W^{UK}_i{}^T q_{t,i})^T c_j^{KV}`; equivalently,
  `(W^{UK}_i{}^T q_{t,i})^T c_j^{KV} = q_{t,i}^T W^{UK}_i c_j^{KV}`. With the query low-rank path this is
  `(W^{UQ}_i c_t^Q)^T (W^{UK}_i c_j^{KV}) = c_t^Q{}^T (W^{UQ}_i{}^T W^{UK}_i) c_j^{KV}`,
  so `W^{UK}` folds into the query projection — keys are never materialized. Likewise `W^{UV}`
  folds into `W^O`. At inference MLA attends directly over the latent, like one shared head.
- **Query low-rank** (`c_t^Q = W^{DQ} h_t`, `q^C = W^{UQ} c^Q`) cuts training activation memory
  only; it does not affect the (uncached) query at inference.

## Decoupled RoPE (the RoPE fix)

RoPE rotates queries and keys by position. The relative-rotation identity is
`R_t^T R_j = R_{j-t}`. If the reconstructed content key is rotated, the score contains
`c_t^Q{}^T W^{UQ}_i{}^T R_t^T R_j W^{UK}_i c_j^{KV}
= c_t^Q{}^T W^{UQ}_i{}^T R_{j-t} W^{UK}_i c_j^{KV}`. The position-dependent `R_{j-t}`
now sits *between* `W^{UQ}_i{}^T` and `W^{UK}_i`, so it cannot be precomputed into one absorbed
matrix; honoring RoPE naively would require rebuilding and rotating prefix keys.
Fix: split position into a small slice. Add a per-head rotary query
`q_{t,i}^R = RoPE(W^{QR} c_t^Q)_i` and a **single shared** rotary key `k_t^R = RoPE(W^{KR} h_t)`,
each of dim `d^R_h`. Concatenate onto the content parts; the score splits:

```
q_{t,i}^T k_{j,i} = (q_{t,i}^C)^T k_{j,i}^C  +  (q_{t,i}^R)^T k_j^R
                    └ position-free, absorbable ┘   └ carries relative position ┘
```

The content term keeps absorption; the tiny rotary term carries all positional info. Only
`c^{KV}` and the shared `k^R` are cached, so cache = `(d_c + d^R_h) l`. The softmax scale becomes
`1/sqrt(d_h + d^R_h)` because the key is now `[content ; rotary]`. RMSNorm is applied to `c^{KV}`
and `c^Q` before up-projection to keep the low-rank bottleneck from destabilizing training.

## KV cache comparison

The per-layer cache count is `d_c + d^R_h` versus GQA's `2 n_g d_h`; multiplying by `l` gives
the all-layer cache per token.

| Structure | KV cache / token across `l` layers | Capacity |
|---|---|---|
| MHA | `2 n_h d_h l` | strong |
| GQA (`n_g` groups) | `2 n_g d_h l` | moderate |
| MQA | `2 d_h l` | weak |
| MLA | `(d_c + d^R_h) l` | strong |

With `d_c = 4 d_h`, `d^R_h = d_h/2`, MLA's cache `= 4.5 d_h l` equals GQA with only ~2.25
groups, near the MQA end of the cache spectrum while preserving per-head up-projections.

## Why MLA is richer than GQA at equal cache

Written as an up-projection from a cached vector, GQA's shared key head is a *replication map*:
one `d_h`-vector copied unchanged to every query head in the group — rank-deficient. MLA's
`W^{UK}, W^{UV}` are unconstrained low-rank up-projections from the same-sized latent, so each
head gets a distinct key/value. At equal cached bytes, MQA and GQA are constrained replication
special cases of MLA's parameterization; full MHA is recovered only when the cached state grows
to the realized per-head K/V state. The useful separation is that cache size is no longer tied
to the number of query heads.

## Reference dimensions

A full-scale instance: `n_h = 128`, `d_h = 128`, KV latent `d_c = 512 = 4 d_h`, query latent
`d'_c = 1536`, decoupled `d^R_h = 64 = d_h/2`; RoPE base `θ = 10000`.

## Working code (from-scratch nanoGPT attention block)

A from-scratch nanoGPT realization (not a checkpoint-conversion pipeline). The latent acts as one
shared KV head (`build_kv_heads -> (1, head_dim)`); per-head distinctness lives in the
up-projections. Ranks are scaled to the small hidden size while preserving the relative schedule;
cached bytes are proportional to `kv_lora_rank + qk_rope_head_dim`.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


def build_kv_heads(config):
    head_dim = config.n_embd // config.n_head
    return 1, head_dim                       # latent = one shared KV head


def cross_layer_share(layer_idx, config):
    return False


def latent_kv_project(k, v, config):
    return k, v, 1.0                         # compression lives inside the block


class MLARMSNorm(nn.Module):
    """RMSNorm on the latent: stabilizes the low-rank bottleneck."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
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


def build_rotary_cache(seq_len, dim, device, dtype, theta=10000.0):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).view(1, 1, seq_len, dim)
    sin = emb.sin().to(dtype).view(1, 1, seq_len, dim)
    return cos, sin


def apply_rotary_pos_emb_interleave(q, k, cos, sin):
    # half-split cache + rotate_half is the correct from-scratch form
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        self.n_kv_head, self.head_dim = build_kv_heads(config)
        self.share_across_layers = False

        self.qk_rope_head_dim = min(64, self.head_dim)
        self.qk_rope_head_dim = max(16, self.qk_rope_head_dim)
        if self.qk_rope_head_dim % 2 != 0:
            self.qk_rope_head_dim -= 1
        self.qk_nope_head_dim = self.head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = self.head_dim
        self.q_lora_rank = min(self.n_embd, 12 * self.head_dim)
        self.kv_lora_rank = max(16, self.head_dim // 2)

        self.q_a_proj = nn.Linear(config.n_embd, self.q_lora_rank, bias=False)
        self.q_a_layernorm = MLARMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.n_head * self.qk_head_dim, bias=config.bias
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            config.n_embd, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.bias
        )
        self.kv_a_layernorm = MLARMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_head * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.n_head * self.v_head_dim, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )
        self.use_pos_emb = False
        self.head_sharing_ratio = float(self.n_head)
        self.scaling = self.qk_head_dim ** -0.5

    def forward(self, x):
        bsz, seq_len, _ = x.size()

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q_states = q_states.view(bsz, seq_len, self.n_head, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv_latent, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_states = self.kv_b_proj(self.kv_a_layernorm(kv_latent))
        kv_states = kv_states.view(
            bsz, seq_len, self.n_head, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, value_states = torch.split(
            kv_states, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        k_rot = k_rot.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        cos, sin = build_rotary_cache(seq_len, self.qk_rope_head_dim, x.device, q_rot.dtype)
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        # new_empty + slice-assign: avoids expand/cat of the shared rope key;
        # slice setitem is autograd-safe (k_rot broadcasts over the head axis).
        query_states = q_states.new_empty(bsz, self.n_head, seq_len, self.qk_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_rot

        key_states = q_states.new_empty(bsz, self.n_head, seq_len, self.qk_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_rot       # broadcasts over n_head

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=self.scaling,
            )
        else:
            att = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = torch.matmul(att, value_states)

        latent_ratio = self.kv_lora_rank / self.qk_head_dim
        storage_ratio = (self.kv_lora_rank + self.qk_rope_head_dim) / (2 * self.head_dim)
        self._last_latent_rank_ratio = float(latent_ratio)
        self._last_kv_storage_ratio = float(storage_ratio)
        self._uses_latent_compression = True

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_head * self.v_head_dim)
        y = self.resid_dropout(self.o_proj(y))
        return y
```
