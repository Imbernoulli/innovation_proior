# Research question

High-throughput LLM serving requires batching many requests so the cost of moving model weights is amortized across the batch. The batch size that fits in GPU memory is largely determined by the **key-value cache** (KV cache): the stored key and value vectors for every token of every in-flight request, which must be kept around because each new token attends to all previous tokens. The KV cache for one request grows token-by-token, its final length is unknown in advance, and it can be large.

How should a serving system organize and manage GPU memory for the KV cache so that many requests can share the GPU and throughput scales with available hardware?

# Background

**Autoregressive generation and the KV cache.** A Transformer LM factorizes $P(x)=\prod_i P(x_i\mid x_{<i})$ and generates one token at a time. Each self-attention layer computes, for position $i$, $q_i=W_qx_i,\ k_i=W_kx_i,\ v_i=W_vx_i$, then $a_{ij}=\mathrm{softmax}_j(q_i^\top k_j/\sqrt d)$ over $j\le i$ and output $o_i=\sum_{j\le i}a_{ij}v_j$. Generating token $i$ needs the keys and values of *all* previous tokens, so they are cached. Serving has two phases: a **prompt/prefill phase** that processes the whole input prompt in parallel (matrix–matrix, compute-efficient) and produces the prompt's KV cache plus the first output token; and an **autoregressive generation phase** that emits one token per iteration (matrix–vector, memory-bound, GPU-underutilized) and appends one new key/value per step.

**KV cache size.** For a 13B OPT model, one token's KV cache is $2\ (\text{K,V})\times 5120\ (\text{hidden})\times 40\ (\text{layers})\times 2\ (\text{bytes, FP16}) = 800$ KB; a 2048-token request needs up to 1.6 GB. With tens-of-GB GPUs, only a few tens of requests fit even if all memory went to KV cache. Hardware trends amplify the constraint — compute throughput is growing faster than memory capacity (A100→H100 roughly doubled FLOPS while memory stayed at 80 GB), making serving increasingly memory-bound.

**Iteration-level (continuous) batching.** Fine-grained batching mechanisms (cellular batching, iteration-level scheduling à la Orca) operate at the granularity of a single decoding iteration: after each step, finished requests leave the batch and new ones join, eliminating most queueing delay and padding waste. This is the batching substrate for modern LLM serving.

**Complex decoding and prefix sharing.** Many decoding algorithms create requests that share token prefixes: parallel sampling (multiple samples from one prompt share the prompt's KV cache), beam search (beam candidates share large, dynamically-changing prefixes), and shared system prompts (many requests share a long fixed prefix).

# Baselines

**Orca (iteration-level scheduling).** State-of-the-art throughput via continuous batching at the iteration level. Allocates KV cache as a contiguous per-request chunk.

**FasterTransformer.** Highly optimized inference engine with fast fused attention kernels. Uses request-level, contiguous, max-length KV-cache allocation.

**Contiguous max-length pre-allocation (the common design).** Reserve a contiguous tensor of size (max sequence length × per-token KV size) per request. Simple and kernel-friendly. Compaction of the KV region has been proposed but moving the massive KV cache around in a latency-sensitive loop is costly.

# Evaluation settings

- **Models / hardware.** OPT (13B on one A100-40GB, 66B on 4×A100, 175B on 8×A100-80GB) and LLaMA, served under Megatron-LM-style tensor parallelism for multi-GPU.
- **Workloads.** Real request traces with realistic, highly variable input/output length distributions (e.g. ShareGPT — long, and Alpaca — short), replayed at varying request rates.
- **Decoding algorithms.** Greedy/basic sampling, parallel sampling, beam search (beam width $k$), and shared-prefix prompting — chosen to exercise different sharing patterns.
- **Metrics.** Serving throughput (requests/s sustainable) at a fixed latency target (normalized latency per output token), against FasterTransformer and Orca; plus KV-cache memory utilization, and ablations on block size and on recovery mechanism (recomputation vs. swapping).

# Code framework

The primitives that already exist: an iteration-level scheduler that picks which requests run each step, a Transformer model executor with an attention kernel, and a chunk of GPU DRAM set aside for KV cache. Today the KV cache is one contiguous per-request tensor; the manager that decides *where* each token's KV lives is the empty slot.

```python
class KVCacheManager:
    """Owns the GPU KV-cache region. Must serve each request's growing KV cache
    as new tokens arrive, support requests whose tokens coincide, and reclaim
    storage when requests finish or must be evicted."""
    def __init__(self, total_kv_bytes):
        # TODO: how to organize and hand out the KV-cache region
        pass

def attention(q_i, K_cache, V_cache):
    # standard: a_ij = softmax(q_i . k_j / sqrt(d)) over j<=i ; o_i = sum_j a_ij v_j
    # TODO: read K/V from whatever layout the manager uses
    pass

# scheduler loop: each iteration, select runnable sequences, ask the manager for
# storage for the new tokens, run the model + attention, append KV, free finished.
```
