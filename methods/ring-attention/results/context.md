# Context

## Research question

Transformers are the backbone of modern sequence models, and there is mounting demand to feed them *very* long contexts — books, codebases, long dialogues, long-horizon RL trajectories, interleaved multi-document inputs. Context lengths in deployed systems have crept from a few thousand tokens to tens and hundreds of thousands. The context a single device can handle is bounded by its memory: even modest models cannot hold the activations for sequences in the millions of tokens.

The question: **how can a Transformer be trained and run on sequences whose length is limited by the total number of devices rather than by any single device's memory, while computing exact full attention distributed across those devices?**

## Background

**Self-attention and the feedforward sublayer.** For a layer with queries, keys, values Q, K, V ∈ ℝ^{s×d} (sequence length s, head dimension d), attention is softmax(QKᵀ/√d)·V, softmax applied row-wise. Each attention sublayer is followed by a position-wise feedforward network FFN(x) = max(0, xW₁ + b₁)W₂ + b₂, typically with an inner dimension 4× the hidden size.

**Memory of a layer.** A vanilla attention layer materializes the full s×s score matrix, costing O(s²) memory. The FFN's intermediate activation is also large (its 4× inner width gives an 8·b·s·h activation for batch b, hidden h). For 100M tokens at a hidden size of 1024, just storing one layer's output is over 1000 GB, while GPUs/TPUs offer well under 100 GB of high-bandwidth memory.

**Memory-efficient / blockwise attention.** A line of work computes attention *block by block* with online (streaming) softmax rescaling: tile Q, K, V into blocks; for each query block, loop over key/value blocks, accumulating a running output while tracking per-row running statistics (a running max for numerical stability and a running normalizer), rescaling the accumulator as each new block arrives. This (Rabe & Staats 2021; Dao et al. 2022, FlashAttention) brings attention activation memory down to roughly the block size, not s. Blockwise Parallel Transformer (Liu et al. 2023, BPT) extends the same block-by-block discipline to the FFN — computing it per query block instead of materializing the full 8·b·s·h intermediate — cutting the maximum activation from 8·b·s·h down to 2·b·s·h per layer.

**Per-layer output.** With blockwise attention+FFN, each device stores the full output of each layer; the n-to-n nature of self-attention means the next layer's attention needs all positions' outputs. That output storage scales with s.

**Sequence parallelism.** Earlier work arranged devices to pass attention computation between them to reduce communication cost (Li et al. 2021, sequence parallelism).

## Baselines

- **Vanilla Transformer attention.** Materializes the s×s score matrix: self-attention activation 2·b·n·s² (n heads), FFN activation 8·b·s·h. Exact attention, O(s²) memory.

- **Memory-efficient attention (Rabe & Staats 2021; FlashAttention, Dao et al. 2022).** Block-by-block attention with online-softmax rescaling, never forming the full score matrix. Self-attention activation drops to ~2·b·s·h + 4·b·c·h (block size c); FFN still 8·b·s·h. Exact attention.

- **Blockwise Parallel Transformer / BPT (Liu et al. 2023).** Also computes the FFN block-by-block, fusing it with blockwise attention; maximum activation per layer ≈ 2·b·s·h. The state of the art for single-device memory efficiency; it stores the full layer output of size proportional to s.

## Evaluation settings

- **Architecture.** LLaMA-style Transformers at 3B, 7B, 13B, 30B parameters; the model architecture is unchanged — only the computation is reorganized.
- **Hardware.** GPUs (single 8×A100 DGX server; distributed 32×A100) and TPUs (v3, v4, v5e), with their respective interconnects (NVLink / InfiniBand between GPUs; ICI torus between TPUs).
- **Baselines for comparison.** Vanilla attention, memory-efficient attention (+ its fused CUDA kernel), and blockwise attention+FFN (BPT).
- **Protocol.** Full gradient checkpointing on both attention and FFN (following the memory-efficient-attention works). Metrics: maximum supported sequence length within device memory, model FLOPs utilization (MFU), throughput, and downstream task quality on long-context language modeling and in-context RL.

## Code framework

The pre-existing primitives: a blockwise/online-softmax attention kernel that, given a query block and a key/value block plus running (max, numerator, denominator) statistics, returns updated statistics without materializing the full score matrix; a blockwise feedforward routine; a device mesh with collective communication, including a collective permute primitive that moves tensors according to a device permutation; and a custom forward/backward (VJP) hook so the distributed attention can define its own gradient. The scaffold is the blockwise attention loop with an empty slot for the distributed attention.

```python
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

# --- existing blockwise (online-softmax) primitives ---
def blockwise_attention_fwd(q, k_block, v_block, carry, ...):
    """One step of online-softmax attention against ONE key/value block.
    carry contains (numerator, denominator, running_max); returns updated carry.
    Never materializes the full s x s score matrix."""
    ...

def blockwise_ffn(x_block):
    """Position-wise FFN computed block-by-block (max activation ~2*b*c*h)."""
    ...

def collective_permute(x, axis_name, permutation):
    """Move x across devices according to a supplied permutation."""
    return lax.ppermute(x, axis_name, perm=permutation)

# --- distributed-attention scaffold ---
def distributed_attention_fwd(q, k, v, axis_name):
    """Each device holds the q/k/v for ONE block of the sequence.
    A query block must attend to ALL key/value blocks (n-to-n), but the
    other blocks live on other devices.
    TODO: distribute the n-to-n attention across devices.
    """
    numerator   = jnp.zeros(...)
    denominator = jnp.zeros(...)
    running_max = jnp.full(..., -jnp.inf)
    ...
    output = numerator / denominator
    return output

def transformer_layer(x_block, axis_name):
    attn_out = distributed_attention_fwd(*qkv(x_block), axis_name)
    return blockwise_ffn(attn_out)   # FFN on the local block, no communication
```
