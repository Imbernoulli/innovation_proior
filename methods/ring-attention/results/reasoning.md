Let me think about what's actually stopping me from feeding a Transformer a sequence of, say, a hundred million tokens. The naive answer is the s×s attention matrix — quadratic in sequence length, blows up immediately. But that one's been beaten: I can compute attention block by block with an online softmax, never materializing the full score matrix, so the attention activation drops to about the block size instead of s. And the feedforward sublayer, whose intermediate is 8·b·s·h because of its 4× inner width, can likewise be done block by block, pulling its peak activation down to about 2·b·s·h. So with blockwise attention and blockwise FFN, the *per-layer working memory* is no longer the bottleneck. Good.

So why can't I just crank s to a hundred million? Let me find what's still scaling with s. Each layer still has to *store its output* — one vector per position, the whole sequence's worth, b·s·h. And I can't throw any of it away, because self-attention is n-to-n: the next layer's attention has every query attending to every key, so every position's output from this layer is needed as input to the next. So even with all the blockwise tricks, I'm pinned at storing the full layer output, which scales with s. Concretely, 100M tokens at hidden 1024 is over 1000 GB just for one layer's output, and a device has well under 100 GB of HBM, and HBM isn't growing much. That's the real wall. Single-device memory caps the sequence length, full stop.

The thing is, if the wall is "one device's memory," then the fix isn't a cleverer kernel — it's to stop confining the sequence to one device. Split the sequence across devices. Put block 1 of the sequence on device 1, block 2 on device 2, and so on, N devices for N blocks. Now each device only stores its own block's worth of activations and its own block's layer output, so per-device memory for the output is b·s/N·h instead of b·s·h. Add devices, get longer sequences. That's the scaling I want — context proportional to device count.

The FFN is no problem under this split: it's position-wise, so each device just runs the FFN on its own block, zero communication. The trouble is attention. A query block on device i must attend to *all* key/value blocks — n-to-n — but those other blocks live on the other devices. So device i needs the key/value blocks it doesn't hold. The naive move: every device fetches all the other blocks before computing. But that re-accumulates all N blocks in memory — exactly the O(s) storage I just escaped, straight back through the front door. Dead end. And even setting memory aside, while device i waits to receive a block from device j it just sits idle. So I have two problems stacked: don't accumulate all the blocks, and don't stall.

Let me look harder at the structure of the blockwise attention computation, because I have a feeling it's more forgiving than it looks. In the online-softmax loop, a query block iterates over key/value blocks and maintains running statistics — a running max for numerical stability, a running numerator (the weighted sum of values), and a running denominator (the sum of exponentials). For one query row, suppose the old state is `(m, l, a)`: max score so far, softmax normalizer so far, and weighted-value numerator so far. A new key/value block gives scores `z_j` and values `v_j`. I set `m' = max(m, max_j z_j)`, rescale the old state into that new max with `exp(m-m')`, and update `l' = exp(m-m')l + sum_j exp(z_j-m')` and `a' = exp(m-m')a + sum_j exp(z_j-m')v_j`. The output row at the end is `a'/l'`. The key thing: does the *order* of the key/value blocks matter? Let me check. The final max is the max over all scores regardless of order; the final normalizer and numerator are sums of each block's contribution expressed under that final max; and addition is commutative. So no — the inner loop over key/value blocks is permutation-invariant. A query block can attend to its key/value blocks in *any order*, as long as the statistics are merged correctly for rescaling.

That permutation invariance is the lever. If the order doesn't matter, then I don't need device i to have all the blocks at once — I just need each key/value block to *pass through* device i at some point, in any order, so device i can fold it into its running statistics and then let it go. So picture the devices as a ring: device 1, device 2, …, device N, with device N's "next" wrapping back to device 1. Each device permanently holds its own query block. The key/value blocks circulate around the ring. At each visit, device i computes the online-softmax update of its query block against whichever key/value block it currently holds while that key/value block is sent onward and a new one is received from the previous device. Count the visits carefully: first the local key/value block, then N−1 received blocks. Every key/value block has visited every device exactly once, so every query block has attended to the entire sequence — exact full attention, no approximation. And at any instant a device holds only its query block plus a couple of key/value blocks in flight, never all N. The O(s) storage is gone and the n-to-n requirement is satisfied.

If the attention is causal, I shouldn't turn that into a variable-length ring where early query blocks skip future key blocks and late query blocks do more work; that would unbalance the hosts. The cleaner move is to keep the same number of block visits on every host and let the bias path decide which q/k pairs are legal. The global query block index is the device index. At ring step `idx`, the key block currently in hand is `(axis_index - idx) % axis_size`, so the local code slices `attn_bias` only along the key axis at `k_block_idx * kv_len` for `kv_len` entries. The query side is carried by `q_chunk_idx_start`, and the key side by `k_chunk_idx_start`, as the blockwise kernel walks its smaller chunks. So the kernel still sees the correct global query/key positions for the causal mask, but every host executes exactly `axis_size` block visits.

The "simultaneously" is doing real work, and I need to check it actually hides the communication, because if devices stall waiting for blocks then I've traded a memory problem for a latency problem. The send-to-next/receive-from-previous is a ring permute. While that transfer is in flight, the device is busy computing attention on the block it already has. If the compute takes at least as long as the transfer, the communication is completely hidden — zero overhead. So when is compute ≥ communication? Let me size it. Let block size be c, hidden d, each device has F FLOPs/s, and the inter-device bandwidth is B. The attention compute for one query-block-against-one-kv-block: the scores QKᵀ cost 2dc² FLOPs, and multiplying scores by V costs another 2dc², so 4dc² FLOPs total. (I'll ignore the q/k/v projections and the FFN — they add compute but no inter-host communication, so including them only makes the overlap easier; leaving them out is the stricter condition.) The communication per step: I have to send the key block and the value block, each c·d elements, so 2cd elements, and at, say, 2 bytes each that's 4cd bytes. The overlap condition is compute-time ≥ communication-time:

4dc² / F ≥ 4cd / B.

Cancel 4dc from both sides: c ≥ F / B. So the block size just has to be at least the ratio of compute throughput to bandwidth. That's a clean, satisfying condition — it doesn't depend on the sequence length s at all, only on the hardware. For an A100 over NVLink, F/B is about 1000, so blocks of ~1K tokens per device suffice; because the memory accounting below needs six blocks, the corresponding per-host sequence length threshold is s = 6c, about 6K tokens on that link. So on high-bandwidth interconnects, the communication is fully hidden behind compute as long as each device's block is roughly 1K tokens — totally achievable. On low-bandwidth links like InfiniBand the required block is much bigger, but it is still a hardware constant, not a function of the total context length.

Now let me make sure the per-device memory is genuinely independent of s, because that's the whole point. What does a device hold during ring attention? Its query block (1 block), the current key and value blocks it's computing on (2 blocks), the incoming key and value blocks it's receiving (2 blocks), and the output accumulator, which has the shape of the query block (1 block). That's 6 blocks — 6·b·c·h bytes. And the blockwise FFN's peak activation is about 2·b·c·h, which is dominated by those 6 blocks, so the total maximum activation stays at 6·b·c·h. It's linear in the block size c and *independent of s*. That's exactly the property I wanted: per-device memory set by the block size, not the sequence length, so the sequence length scales with the number of devices with no per-layer memory penalty.

The backward pass: it's the same structure. Attention's gradient still needs every query block to interact with every key/value block, so I run the same ring — circulate the key/value blocks (and now also accumulate the key/value gradients as they pass) around the devices, overlapping the permute with the blockwise backward computation. The online-softmax backward uses the saved statistics (the denominator and the running max from the forward) to recompute the needed quantities per block. Same permutation invariance, same overlap argument, so the backward also scales for free. I'll define a custom forward/backward so the distributed attention owns its own VJP.

Let me also place this against the obvious alternative I rejected. I could have tried to approximate attention — sparsify it, low-rank it — to dodge the n-to-n cost. But I wanted *exact* attention and full quality; the ring gives me that, because reorganizing the order of a commutative-associative accumulation changes nothing about the result. And I could have used a ring purely to cut communication cost without the blockwise memory formulation, but that wouldn't touch the real wall, which is storing the full per-layer output on one device. Coupling the ring to the blockwise online-softmax is what makes both the memory and the communication disappear at once.

Let me write the forward. Each device holds q, k, v for its block; it takes `axis_size` block visits, updating online-softmax statistics against the current key/value block while the key/value tensors are permuted to the next device.

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
    # online-softmax running statistics for THIS device's query block
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(q.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    axis_size = lax.psum(1, axis_name)  # number of devices on the ring = N
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
        # fold the CURRENT key/value block into the running statistics, exact
        # (permutation invariant: any visiting order gives the same result)
        numerator, denominator, max_score = _blockwise_attention_fwd(
            q, k, v, (numerator, denominator, prev_max_score),
            q_chunk_idx_start, k_chunk_idx_start,
            bias=attn_bias_slice, **blockwise_kwargs)
        # rotate key/value to the next device; hidden as long as c >= F / B
        k, v = map(lambda x: lax.ppermute(
            x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]),
            (k, v))
        return (max_score, numerator, denominator, k, v), None

    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)
    (max_score, numerator, denominator, _, _), _ = lax.scan(
        scan_kv_block,
        init=(prev_max_score, numerator, denominator, k, v),
        xs=jnp.arange(0, axis_size))
    # finalize: output is the normalized accumulator (shape of the query block)
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
        # blockwise backward against the current k/v block, using saved stats
        dq, dk, dv = _blockwise_attention_bwd(
            q, k, v, g, (dq, dk, dv, output, denominator, max_score),
            q_chunk_idx_start, k_chunk_idx_start,
            bias=attn_bias_slice, **blockwise_kwargs)
        # rotate k, v AND their grads around the ring, overlapped with compute
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

def transformer_layer(x_block, attn_bias, axis_name, float32_logits, blockwise_kwargs):
    q, k, v = qkv_projection(x_block)
    attn = ring_attention(q, k, v, attn_bias, axis_name, float32_logits, blockwise_kwargs)
    return blockwise_ffn(attn)   # position-wise, on the local block, no communication
```

The causal chain, top to bottom: blockwise attention and FFN already kill the per-layer working memory, but each device still has to store the full b·s·h layer output and can't drop it because attention is n-to-n — that's the wall, and it scales with s; split the sequence across N devices so each holds only a 1/N block, making per-device output memory independent of s; the FFN is position-wise so it needs no communication, but attention needs every query block to see every key/value block living on other devices; the online-softmax inner loop is permutation-invariant, so the key/value blocks can simply circulate a ring, each query folding in each block in whatever order it arrives, giving exact full attention while a device ever holds only ~6 blocks (6·b·c·h, independent of s); and the send-to-next/receive-from-previous permute is fully overlapped with the 4dc² FLOPs of blockwise attention whenever the block size c ≥ F/B, so scaling the context across more devices costs no extra wall-clock time.
