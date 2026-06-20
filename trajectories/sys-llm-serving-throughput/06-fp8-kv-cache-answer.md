**Problem (from step 5).** After paging (near-zero waste), continuous batching (full occupancy), chunked prefill (no prefill stalls), prefix caching (no redundant shared work), and speculative decoding (more tokens per decode pass), throughput at fixed latency is again set by the same thing it was at the start: **how many requests' KV caches fit in GPU memory**. Every memory rung made the *bookkeeping* of that memory efficient but left the one quantity underneath untouched — the **bytes per cached element** (FP16/BF16 = 2 bytes), which multiplies the entire cache.

**Key idea — FP8 KV cache.** Store KV entries in **1 byte (FP8)** instead of 2. The cache is consumed only by attention (query·keys → softmax → value-weighting), which is forgiving of low precision because softmax averages many score contributions and small per-element errors wash out. Crucially, FP8 is a *storage* format, not a *compute* format: keep the cache at one byte, but **convert each element back up to the compute dtype on read** and do the matmuls at full precision. The only error is the one-time rounding when KV is written into 8 bits. Default to **E4M3** (4 exp / 3 mantissa — finer steps, what averaged values need) over E5M2 (5 exp / 2 mantissa — wider range, coarser), with a per-tensor `scale` so values land in the format's well-represented range.

**Why it works.** Halving bytes-per-element ≈ doubles the tokens that fit in the same memory ≈ doubles the achievable concurrent batch / context length. Since throughput at fixed latency tracks the live batch size (decode is bandwidth-bound) and the batch is gated by exactly this memory, halving the per-token cost directly raises the ceiling that *every other rung* presses against. It composes for free: block management (free list, block tables, ref counting, prefix-hash sharing, preemption) never cared what is *inside* a block, so paging / continuous batching / chunked prefill / prefix caching / speculative decoding all keep working verbatim on a cache holding 2× the tokens. Only two spots touch the bytes — convert-on-write (activation dtype → FP8 with scale) and convert-on-read (FP8 → compute dtype). Gain and any quality effect are **config-sensitive** (most for long-context / high-concurrency memory-bound deployments), reproducible via the throughput benchmark with the FP8 flag, not a fixed multiplier.

**The fixed task this whole ladder served.** Maximize generated tokens/sec for a fixed LLM on fixed GPU(s) under a fixed per-request latency budget. Each rung attacked whatever then capped the GPU-busy batch at fixed latency — paging (memory waste), continuous batching (occupancy), chunked prefill (prefill stalls), prefix caching (redundant work), speculative decoding (one-token decodes). FP8 KV storage attacks the deepest cap — the bytes-per-token of the cache itself — so each token costs one byte instead of two and the fixed GPU serves a larger batch at the same latency.

**Change / code.** Make the cache element type a 1-byte FP8 type; convert on write (with scale) and on read.

```cpp
// csrc/.../cache_kernels.cu — reshape_and_cache write path (excerpt).
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;
  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);                            // unquantized copy
    } else {
      dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale); // -> 1-byte FP8
    }
  }
};
// cache_t is now a 1-byte FP8 type: half the bytes/token -> ~2x the capacity.
// Block layout, block tables, free list, prefix sharing all unchanged.
// On read, paged attention scaled_converts FP8 back up before the QK dot.
```

```cpp
// csrc/attention/dtype_fp8.cuh — the 8-bit KV storage formats.
enum class Fp8KVCacheDataType {
  kAuto = 0,     // unquantized (2-byte FP16/BF16)
  kFp8E4M3 = 1,  // 4 exp / 3 mantissa -- finer steps; the default ("fp8")
  kFp8E5M2 = 2,  // 5 exp / 2 mantissa -- wider range, coarser
};
// Selected at serve time by --kv-cache-dtype fp8 (== fp8_e4m3) or fp8_e5m2.
```
