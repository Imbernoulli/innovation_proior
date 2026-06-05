# Ring Attention with Blockwise Parallel Transformers

## Problem

Train and run Transformers on near-arbitrarily long sequences, limited by total device count rather than any single device's memory, computing *exact* attention with no communication overhead. Blockwise attention + FFN (online-softmax attention and block-by-block FFN) already cut per-layer working memory to ~2·b·s·h, but each device must still store the full layer output (b·s·h) and cannot discard it, because self-attention is n-to-n — so the maximum sequence length stays capped by one device's memory.

## Key idea

Split the sequence into blocks, one per device. The FFN is position-wise, so each device runs it on its local block with no communication. For attention, exploit that the online-softmax inner loop over key/value blocks is **permutation-invariant** (the running max, numerator, denominator merge correctly in any order). Arrange devices in a **ring**: each device keeps its query block fixed, and the key/value blocks **rotate around the ring** — every step a device computes blockwise attention on the key/value block it holds while simultaneously sending it to the next device and receiving the previous one. After N−1 rotations every query block has attended to the whole sequence (exact full attention), and a device ever holds only ~6 blocks.

## Final method

- **Per-device memory:** a device stores its query block, the current key/value blocks, the incoming key/value blocks, and the output accumulator (shape of the query block) — **6·b·c·h** bytes (block size c), **independent of sequence length s**. So context scales linearly with device count.
- **Exactness:** reordering a commutative-associative online-softmax accumulation does not change the result — no approximation.
- **Overlap condition:** blockwise attention for one query-block × one kv-block costs **4dc²** FLOPs (2dc² for QKᵀ, 2dc² for scores·V); rotating the key+value blocks moves 4cd bytes. Communication is fully hidden behind compute when 4dc²/F ≥ 4cd/B, i.e. **block size c ≥ F/B** (compute throughput over bandwidth) — independent of s. On A100/NVLink, c ≈ 1K tokens suffices.
- **Backward:** identical ring structure; key/value blocks and their gradients rotate together, overlapped with the blockwise backward, which uses the saved (denominator, running max). The distributed attention defines its own custom VJP.

## Code (JAX)

```python
import jax, jax.numpy as jnp
from jax import lax
from functools import partial
from einops import rearrange

def _ring_attention_fwd(q, k, v, axis_name, blockwise_kwargs):
    batch, q_len, num_heads, dim = q.shape
    numerator   = jnp.zeros((batch, q_len, num_heads, dim), dtype=q.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len),       dtype=q.dtype)
    prev_max    = jnp.full((batch, num_heads, q_len), -jnp.inf, dtype=q.dtype)
    axis_size = lax.psum(1, axis_name)                       # devices on the ring

    def step(carry, idx):
        prev_max, numerator, denominator, k, v = carry
        numerator, denominator, new_max = blockwise_attention_fwd(   # online softmax
            q, k, v, (numerator, denominator, prev_max), **blockwise_kwargs)
        k, v = lax.ppermute((k, v), axis_name,                       # rotate, overlapped
            perm=[(i, (i + 1) % axis_size) for i in range(axis_size)])
        return (new_max, numerator, denominator, k, v), None

    (max_score, numerator, denominator, _, _), _ = lax.scan(
        step, (prev_max, numerator, denominator, k, v), xs=jnp.arange(axis_size))
    output = numerator / rearrange(denominator, 'b h q -> b q h')[..., None]
    return output, (output, q, k, v, denominator, max_score)

def _ring_attention_bwd(axis_name, blockwise_kwargs, res, g):
    output, q, k, v, denominator, max_score = res
    axis_size = lax.psum(1, axis_name)
    dq, dk, dv = jnp.zeros_like(q), jnp.zeros_like(k), jnp.zeros_like(v)
    def step(carry, idx):
        dq, dk, dv, k, v = carry
        dq, dk, dv = blockwise_attention_bwd(
            q, k, v, g, (dq, dk, dv, output, denominator, max_score), **blockwise_kwargs)
        k, v, dk, dv = lax.ppermute((k, v, dk, dv), axis_name,
            perm=[(i, (i + 1) % axis_size) for i in range(axis_size)])
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, _, _), _ = lax.scan(step, (dq, dk, dv, k, v), xs=jnp.arange(axis_size))
    return dq, dk, dv

@partial(jax.custom_vjp, nondiff_argnums=[3, 4])
def ring_attention(q, k, v, axis_name, blockwise_kwargs):
    y, _ = _ring_attention_fwd(q, k, v, axis_name, blockwise_kwargs)
    return y
ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)
```

Each transformer layer projects its local block to q/k/v, calls `ring_attention`, then applies the position-wise blockwise FFN locally. Compose with FSDP (to shard model weights) and tensor parallelism (to bound global batch size) for end-to-end long-context training.
