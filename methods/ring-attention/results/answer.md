# Ring Attention with Blockwise Parallel Transformers

## Problem

Train and run Transformers on near-arbitrarily long sequences, limited by total device count rather than any single device's memory, computing *exact* attention with no communication overhead. Blockwise attention + FFN (online-softmax attention and block-by-block FFN) already cut per-layer working memory to ~2·b·s·h, but each device must still store the full layer output (b·s·h) and cannot discard it, because self-attention is n-to-n — so the maximum sequence length stays capped by one device's memory.

## Key idea

Split the sequence into blocks, one per device. The FFN is position-wise, so each device runs it on its local block with no communication. For attention, exploit that the online-softmax inner loop over key/value blocks is **permutation-invariant** (the running max, numerator, denominator merge correctly in any order). Arrange devices in a **ring**: each device keeps its query block fixed, and the key/value blocks **rotate around the ring** — while a device computes blockwise attention on the key/value block it currently holds, that same key/value block is sent onward and the next one is received. After each query folds in its local block plus the N−1 blocks received by rotation, it has attended to the whole sequence (exact full attention), and a device ever holds only ~6 blocks.

## Final method

- **Per-device memory:** a device stores its query block, the current key/value blocks, the incoming key/value blocks, and the output accumulator (shape of the query block) — **6·b·c·h** bytes (block size c), **independent of sequence length s**. So context scales linearly with device count.
- **Online-softmax update:** for one query row with old state `(m, l, a)` and a new key/value block with scores `z_j` and values `v_j`, set `m' = max(m, max_j z_j)`, `l' = exp(m-m')l + sum_j exp(z_j-m')`, and `a' = exp(m-m')a + sum_j exp(z_j-m')v_j`; the output is `a'/l'`.
- **Exactness:** that state depends only on the set of processed key/value rows, not their order, so reordering blocks around the ring does not approximate attention.
- **Causal masking and balance:** every host executes the same number of ring steps (`axis_size`). At step `idx`, the current key block is `(axis_index - idx) % axis_size`; the implementation slices `attn_bias` along only the key axis at `k_block_idx * kv_len`, then passes global `q_chunk_idx_start` and `k_chunk_idx_start` into the blockwise kernel. Causal legality is enforced by that bias/chunk-index path rather than by skipping whole ring steps, so the host schedule stays balanced.
- **Overlap condition:** blockwise attention for one query-block × one kv-block costs **4dc²** FLOPs (2dc² for QKᵀ, 2dc² for scores·V); rotating the key+value blocks moves 4cd bytes. Communication is fully hidden behind compute only when 4dc²/F ≥ 4cd/B, i.e. **block size c ≥ F/B** (compute throughput over bandwidth) — independent of s. With the six-block memory footprint, the per-host sequence length threshold is **s = 6c**; on A100/NVLink, c ≈ 1K tokens and s ≈ 6K tokens suffice.
- **Backward:** identical ring structure; key/value blocks and their gradients rotate together, overlapped with the blockwise backward, which uses the saved (denominator, running max). The distributed attention defines its own custom VJP.

## Code (JAX)

```python
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from einops import rearrange

def _ring_attention_fwd(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(q.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    axis_size = lax.psum(1, axis_name)
    block_size = q_len  # pre-sharded inside shard_map
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]

    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, k, v = carry
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        # Slice the current key block; q/k chunk starts carry global positions.
        attn_bias_slice = lax.dynamic_slice_in_dim(
            attn_bias, k_block_idx * kv_len, kv_len, axis=-1)
        q_block_idx = lax.axis_index(axis_name)
        q_chunk_idx_start = q_block_idx * (block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (block_size // key_chunk_size)
        numerator, denominator, max_score = _blockwise_attention_fwd(
            q, k, v, (numerator, denominator, prev_max_score),
            q_chunk_idx_start, k_chunk_idx_start,
            bias=attn_bias_slice, **blockwise_kwargs)
        k, v = map(lambda x: lax.ppermute(
            x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]),
            (k, v))
        return (max_score, numerator, denominator, k, v), None

    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)
    (max_score, numerator, denominator, _, _), _ = lax.scan(
        scan_kv_block,
        init=(prev_max_score, numerator, denominator, k, v),
        xs=jnp.arange(0, axis_size))
    output = numerator / rearrange(denominator, 'b h q -> b q h')[..., None]
    return output.astype(v.dtype), (output, q, k, v, attn_bias, denominator, max_score)

def _ring_attention_bwd(axis_name, float32_logits, blockwise_kwargs, res, g):
    output, q, k, v, attn_bias, denominator, max_score = res
    batch, kv_len, num_heads, dim_per_head = k.shape
    axis_size = lax.psum(1, axis_name)
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]
    block_size = q.shape[1]  # pre-sharded inside shard_map

    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        # Same key-axis bias slice and global chunk offsets as in the forward.
        attn_bias_slice = lax.dynamic_slice_in_dim(
            attn_bias, k_block_idx * kv_len, kv_len, axis=-1)
        q_block_idx = lax.axis_index(axis_name)
        q_chunk_idx_start = q_block_idx * (block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (block_size // key_chunk_size)
        dq, dk, dv = _blockwise_attention_bwd(
            q, k, v, g, (dq, dk, dv, output, denominator, max_score),
            q_chunk_idx_start, k_chunk_idx_start,
            bias=attn_bias_slice, **blockwise_kwargs)
        k, v, dk, dv = map(lambda x: lax.ppermute(
            x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]),
            (k, v, dk, dv))
        return (dq, dk, dv, k, v), None

    (dq, dk, dv, _, _), _ = lax.scan(
        scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    return dq, dk, dv, None

@partial(jax.custom_vjp, nondiff_argnums=[4, 5, 6])
def ring_attention(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    y, _ = _ring_attention_fwd(
        q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs)
    return y
ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)
```

Each transformer layer projects its local block to q/k/v, calls `ring_attention`, then applies the position-wise blockwise FFN locally. Compose with FSDP (to shard model weights) and tensor parallelism (to bound global batch size) for end-to-end long-context training.
