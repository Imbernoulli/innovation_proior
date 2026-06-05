# Context

## Research question

Transformers are the backbone of modern sequence models, and there is mounting demand to feed them *very* long contexts — books, codebases, long dialogues, long-horizon RL trajectories, interleaved multi-document inputs. Context lengths in deployed systems have already crept from a few thousand tokens to tens and hundreds of thousands. But the context a single device can handle is intrinsically bounded by its memory, and that bound is tight: even modest models cannot hold the activations for sequences in the millions of tokens.

The precise problem: **can we train and run a Transformer on sequences whose length is essentially unbounded — limited only by the total number of devices, not by any single device's memory — without approximating attention and without paying a communication penalty that erases the benefit?** A solution must (a) make the per-device memory for a layer independent of the full sequence length, (b) compute *exact* full attention (no sparsity, no low-rank approximation), and (c) distribute the sequence across devices such that the inter-device communication is fully hidden behind computation, so scaling the context costs nothing in wall-clock overhead.

## Background

**Self-attention and the feedforward sublayer.** For a layer with queries, keys, values Q, K, V ∈ ℝ^{s×d} (sequence length s, head dimension d), attention is softmax(QKᵀ/√d)·V, softmax applied row-wise. Each attention sublayer is followed by a position-wise feedforward network FFN(x) = max(0, xW₁ + b₁)W₂ + b₂, typically with an inner dimension 4× the hidden size.

**The memory wall.** A vanilla attention layer materializes the full s×s score matrix, costing O(s²) memory — quadratic in sequence length and the first thing to explode. The FFN's intermediate activation is also large (its 4× inner width gives an 8·b·s·h activation for batch b, hidden h). For 100M tokens at a hidden size of 1024, just storing one layer's output is over 1000 GB, while GPUs/TPUs offer well under 100 GB of high-bandwidth memory, and HBM cannot grow much due to physical and cost limits.

**Memory-efficient / blockwise attention (the immediate ancestors).** A line of work removes the need to materialize the full attention matrix by computing attention *block by block* with online (streaming) softmax rescaling: tile Q, K, V into blocks; for each query block, loop over key/value blocks, accumulating a running output while tracking per-row running statistics (a running max for numerical stability and a running normalizer), rescaling the accumulator as each new block arrives. This (Rabe & Staats 2021; Dao et al. 2022, FlashAttention) brings attention activation memory down to roughly the block size, not s. Blockwise Parallel Transformer (Liu et al. 2023, BPT) extends the same block-by-block discipline to the FFN — computing it per query block instead of materializing the full 8·b·s·h intermediate — cutting the maximum activation from 8·b·s·h down to 2·b·s·h per layer.

**The remaining wall, and the key property.** Even with blockwise attention+FFN, each device must still store the *full output of each layer* — the n-to-n nature of self-attention means the next layer's attention needs all positions' outputs, so they cannot be discarded. That output storage is what caps the sequence length: it still scales with s. The crucial mathematical fact that the next step exploits: because the running-softmax statistics of each key/value block can be combined in any order, **the inner loop over key/value blocks is permutation-invariant** — a query block can attend to its key/value blocks in any sequence as long as the statistics are merged correctly.

**Prior ring-topology attention.** Earlier work proposed arranging devices in a ring to pass attention computation around and reduce communication cost (Li et al. 2021, sequence parallelism). It targeted communication, not the memory of storing full per-layer outputs, and did not couple the ring to a blockwise memory-efficient formulation.

## Baselines

- **Vanilla Transformer attention.** Materializes the s×s score matrix: self-attention activation 2·b·n·s² (n heads), FFN activation 8·b·s·h. Exact attention but O(s²) memory — the hard quadratic wall.

- **Memory-efficient attention (Rabe & Staats 2021; FlashAttention, Dao et al. 2022).** Block-by-block attention with online-softmax rescaling, never forming the full score matrix. Self-attention activation drops to ~2·b·s·h + 4·b·c·h (block size c); FFN still 8·b·s·h. Exact, but the FFN intermediate and the stored per-layer output still scale with s.

- **Blockwise Parallel Transformer / BPT (Liu et al. 2023).** Also computes the FFN block-by-block, fusing it with blockwise attention; maximum activation per layer ≈ 2·b·s·h. The state of the art for single-device memory efficiency. Its remaining limitation: it still stores the full layer output of size proportional to s, so the maximum context is still bounded by one device's memory.

## Evaluation settings

- **Architecture.** LLaMA-style Transformers at 3B, 7B, 13B, 30B parameters; the model architecture is unchanged — only the computation is reorganized.
- **Hardware.** GPUs (single 8×A100 DGX server; distributed 32×A100) and TPUs (v3, v4, v5e), with their respective interconnects (NVLink / InfiniBand between GPUs; ICI torus between TPUs).
- **Baselines for comparison.** Vanilla attention, memory-efficient attention (+ its fused CUDA kernel), and blockwise attention+FFN (BPT).
- **Protocol.** Full gradient checkpointing on both attention and FFN (following the memory-efficient-attention works). Metrics: maximum supported sequence length within device memory; model FLOPs utilization (MFU) and throughput; and (separately) downstream task quality on long-context language modeling and in-context RL — used to confirm that reorganizing the computation does not change the result.

## Code framework

The pre-existing primitives: a blockwise/online-softmax attention kernel that, given a query block and a key/value block plus running (max, numerator, denominator) statistics, returns updated statistics without materializing the full score matrix; a blockwise feedforward routine; a device mesh with collective communication, including a ring-permute primitive that simultaneously sends a tensor to the next device and receives from the previous one; and a custom forward/backward (VJP) hook so the distributed attention can define its own gradient. The scaffold is the blockwise attention loop with empty slots for *where the key/value blocks live* and *how they move between devices*.

```python
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

# --- existing blockwise (online-softmax) primitives ---
def blockwise_attention_fwd(q, k_block, v_block, carry, ...):
    """One step of online-softmax attention against ONE key/value block.
    carry = (running_max, numerator, denominator); returns updated carry.
    Never materializes the full s x s score matrix."""
    ...

def blockwise_ffn(x_block):
    """Position-wise FFN computed block-by-block (max activation ~2*b*c*h)."""
    ...

def ring_permute(x, axis_name):
    """Send x to the next device on the ring, receive from the previous one."""
    return lax.ppermute(x, axis_name, perm=[(i, (i + 1) % N) for i in range(N)])

# --- the slot the contribution fills ---
def distributed_attention_fwd(q, k, v, axis_name):
    """Each device holds the q/k/v for ONE block of the sequence.
    A query block must attend to ALL key/value blocks (n-to-n), but the
    other blocks live on other devices.
    TODO: how does a device obtain the key/value blocks it does not hold,
          WITHOUT (a) waiting idle for them, and (b) accumulating all of
          them in memory (which would re-introduce the O(s) storage)?
    """
    running_max = jnp.full(..., -jnp.inf)
    numerator   = jnp.zeros(...)
    denominator = jnp.zeros(...)
    # TODO: iterate over all key/value blocks in some order, updating the
    #       online-softmax statistics; overlap fetching the next block with
    #       computing on the current one.
    ...
    output = numerator / denominator
    return output

def transformer_layer(x_block, axis_name):
    attn_out = distributed_attention_fwd(*qkv(x_block), axis_name)
    return blockwise_ffn(attn_out)   # FFN on the local block, no communication
```
