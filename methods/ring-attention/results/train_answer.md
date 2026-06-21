Transformers have become the backbone of sequence modeling, and the pressure to feed them ever-longer contexts — whole books, codebases, long dialogues, long-horizon trajectories — keeps mounting. The obstacle is memory. The first thing that explodes is the $s \times s$ score matrix of self-attention, which costs $O(s^2)$ memory in the sequence length $s$, but that wall has already been knocked down: I can compute attention block by block with an online softmax that never materializes the full score matrix, dropping the attention activation to roughly the block size rather than $s$, and I can do the position-wise feedforward sublayer block by block too, pulling its peak intermediate from $8 \cdot b \cdot s \cdot h$ down to about $2 \cdot b \cdot s \cdot h$. So the per-layer *working* memory is no longer the bottleneck. The problem is what remains. Each layer still has to store its full output — one vector per position, $b \cdot s \cdot h$ in all — and none of it can be thrown away, because self-attention is n-to-n: the next layer has every query attend to every key, so every position's output from this layer is an input to the next. Concretely, 100M tokens at hidden size 1024 is over 1000 GB just for one layer's output, against well under 100 GB of high-bandwidth memory on a device, and HBM is not growing much. That stored output scales with $s$, and it is the real wall. So the goal is sharp: train and run on sequences whose length is bounded only by the total number of devices, computing *exact* full attention with no sparsity or low-rank approximation, and with the inter-device communication fully hidden behind computation so that scaling the context costs nothing in wall-clock time. Memory-efficient and blockwise approaches (Rabe & Staats 2021; Dao et al. 2022; the Blockwise Parallel Transformer of Liu et al. 2023) reach the state of the art for single-device efficiency but still store the full per-layer output and so stay capped by one device's memory; and earlier ring-topology attention (Li et al. 2021) targeted only communication cost and left that storage untouched.

If the wall is "one device's memory," the fix is not a cleverer kernel but to stop confining the sequence to one device. I propose Ring Attention with Blockwise Parallel Transformers. Split the sequence into blocks and place one block on each of $N$ devices, so a device stores only its own block's activations and output, $b \cdot \tfrac{s}{N} \cdot h$ instead of $b \cdot s \cdot h$; add devices, get longer sequences, with context proportional to device count. The feedforward sublayer is trivial under this split because it is position-wise — each device just runs the FFN on its own block with zero communication. The difficulty is attention: a query block on device $i$ must attend to *all* key/value blocks, but the other blocks live on the other devices. Fetching all of them re-accumulates the whole sequence in memory, which is exactly the $O(s)$ storage I just escaped, and waiting on those fetches stalls the device. The way out comes from looking hard at the online-softmax loop itself. For one query row, carry the state $(m, l, a)$ — the running max score, the running normalizer, and the running weighted-value numerator. A new key/value block contributes scores $z_j$ and values $v_j$, and the merge is

$$m' = \max\!\big(m,\ \max_j z_j\big), \qquad l' = e^{m - m'}\,l + \sum_j e^{z_j - m'}, \qquad a' = e^{m - m'}\,a + \sum_j e^{z_j - m'}\,v_j,$$

with the output row equal to $a'/l'$ at the end. The decisive observation is that this state depends only on the *set* of key/value rows processed, not their order: the final max is the max over all scores regardless of order, the normalizer and numerator are sums of per-block contributions all re-expressed under that final max, and addition is commutative. The inner loop over key/value blocks is permutation-invariant.

That invariance is the lever. If order does not matter, device $i$ never needs all the blocks at once — each key/value block only needs to *pass through* device $i$ once, in any order, to be folded into the running statistics and then released. So arrange the devices in a ring, $1, 2, \ldots, N$ with $N$ wrapping back to $1$. Every device keeps its own query block fixed; the key/value blocks circulate. At each step a device computes the online-softmax update of its query block against whichever key/value block it currently holds, *while* that key/value block is sent onward to the next device and a new one is received from the previous one. Counting the visits — first the local key/value block, then $N-1$ received blocks — every key/value block has visited every device exactly once, so every query block has attended to the entire sequence. This is exact full attention, not an approximation, precisely because reordering a commutative-associative accumulation changes nothing. And at any instant a device holds only its query block plus a couple of key/value blocks in flight, never all $N$. This is why I rejected the obvious alternatives: sparsifying or low-ranking attention would dodge the n-to-n cost but sacrifice exactness, and a ring used merely to cut communication without the blockwise online-softmax formulation would never touch the real wall, which is storing the full per-layer output on one device. Coupling the ring to the blockwise softmax makes both the memory and the communication vanish at once.

Two quantitative checks make it real. First, memory: during ring attention a device holds its query block (1), the current key and value blocks it is computing on (2), the incoming key and value blocks being received (2), and the output accumulator shaped like the query block (1) — six blocks, $6 \cdot b \cdot c \cdot h$ bytes for block size $c$, which dominates the FFN's $\approx 2 \cdot b \cdot c \cdot h$ and leaves the total maximum activation at $6 \cdot b \cdot c \cdot h$, linear in $c$ and *independent of $s$*. Exactly the property I wanted. Second, overlap: the send-to-next/receive-from-previous is a ring permute, and while it is in flight the device computes on the block it already holds, so if compute time is at least transfer time the communication is fully hidden. Sizing it with block size $c$, head dimension $d$, per-device throughput $F$ FLOPs/s and interconnect bandwidth $B$: blockwise attention for one query-block against one key/value block costs $2dc^2$ for $QK^\top$ and $2dc^2$ for scores$\cdot V$, so $4dc^2$ FLOPs (ignoring the q/k/v projections and FFN, which add compute but no inter-host communication, making this the stricter condition); the transfer moves the key and value blocks, $2cd$ elements or about $4cd$ bytes. The overlap condition $4dc^2/F \ge 4cd/B$ cancels to

$$c \ \ge\ \frac{F}{B},$$

a clean threshold that does not depend on $s$ at all — only on hardware. On an A100 over NVLink $F/B \approx 1000$, so blocks of about 1K tokens per device suffice, and since the memory accounting needs six blocks the corresponding per-host sequence-length threshold is $s = 6c \approx 6\text{K}$ tokens; on a low-bandwidth link the required block is larger but is still a hardware constant, never a function of the total context.

A few details keep it correct and balanced. For causal attention I do not turn the ring into a variable-length schedule where early query blocks skip future keys — that would unbalance the hosts — so every host executes exactly `axis_size` block visits and the bias path decides which q/k pairs are legal. The global query block index is the device index; at ring step `idx` the key block in hand is $(\text{axis\_index} - \text{idx}) \bmod \text{axis\_size}$, so the local code slices `attn_bias` only along the key axis at `k_block_idx * kv_len`, and passes the global `q_chunk_idx_start` and `k_chunk_idx_start` into the blockwise kernel so it sees the correct global positions for the mask while every host runs the same number of steps. The backward pass has the same structure: the gradient still needs every query block to interact with every key/value block, so the same ring circulates the key/value blocks — now also accumulating their gradients as they pass — overlapping the permute with the blockwise backward, which uses the saved denominator and running max to recompute what it needs per block. The distributed attention owns its own custom forward/backward (VJP), so both directions scale for free.

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
