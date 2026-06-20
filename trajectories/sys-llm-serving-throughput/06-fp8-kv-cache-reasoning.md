Let me take stock of where the whole stack stands and what's still binding. Paging gave me near-zero KV waste; continuous batching keeps occupancy at the ceiling; chunked prefill stops long prompts from stalling decodes and fills idle compute; prefix caching skips redundant shared-prefix work; speculative decoding claims the bandwidth slack in the decode step. After all of that, the throughput at fixed latency is set, once again, by the same thing it was set by at the very start: **how many requests' KV caches fit in GPU memory**. Every memory rung I added made the *bookkeeping* of that memory efficient — no fragmentation, no over-reservation, no duplicate prefixes. But I never changed the one quantity underneath all of it: how many *bytes each cached token costs*. The cache is the binding resource, and I've been packing it perfectly while leaving its per-element size untouched.

So look at that per-element size. KV entries are stored in the model's activation dtype — FP16 or BF16, **two bytes** per element. The total cache footprint is (number of cached tokens) × (layers × heads × head_dim × 2 for K and V) × (bytes per element). Everything except that last factor is fixed by the model and the live workload. The bytes-per-element is the one knob I haven't turned, and it multiplies the entire cache. Halve it and I *double* the number of tokens — hence roughly the number of concurrent requests, or the context length — that fit in the same memory. Since throughput at fixed latency tracks the live batch size, and the batch is gated by exactly this memory, halving the per-token cost is a direct lever on the ceiling that all the other rungs are pushing against.

Can I store KV in **one** byte instead of two? That's an 8-bit float — FP8. The question is whether 8-bit KV is *good enough*, and the answer turns on what the KV cache is actually used for. The cached keys and values are consumed by one operation: attention. A query dots against the cached keys to make scores, softmax normalizes them, and the result weights the cached values. Two facts make this forgiving of low precision. First, softmax is a *smoothing* over many score contributions — small per-element key errors get averaged across the context and largely wash out in the normalized weights; attention is not a sharp function of any single key element. Second, and more important: I do not have to *compute* in FP8. I only have to *store* in FP8. I keep the cache at one byte per element, but when the attention kernel reads a cached element it converts it up to the compute dtype and does the matmuls at full precision. FP8 is a *storage* format here, not a compute format — the only error introduced is the rounding when a KV element is written into 8 bits, and that single quantization step is what softmax then smooths over. So the precision cost is a small, bounded rounding error on the stored KV, not an accumulating low-precision computation.

There's a real subtlety in *which* 8-bit float, and I should pick deliberately rather than defaulting. An 8-bit float splits its byte between exponent and mantissa, and the two standard splits trade range against precision: **E5M2** (5 exponent bits, 2 mantissa) has wide dynamic range but only 2 mantissa bits — coarse; **E4M3** (4 exponent, 3 mantissa) has 3 mantissa bits — finer steps — but narrower range. KV activations are not heavy-tailed in a way that needs huge dynamic range; what hurts attention is *coarse rounding* of the values it averages. So I want the finer-mantissa format, E4M3, as the default — more bits where it matters for the averaged quantity, at the cost of range I don't need. And to use the available range well rather than waste it, I scale: keep a per-tensor (or per-channel) scale factor so the FP8 values land in the format's well-represented range, dividing by the scale on write and multiplying back on read. That's the `k_scale`/`v_scale` the kernel carries. So "FP8 KV cache" concretely means: store as E4M3 by default (E5M2 available when range is the concern), with a scale, and convert-on-read for the attention matmul.

Now the implementation, and the beautiful part is how little has to change because of what's already in place. The cache is a pool of fixed-size blocks (rung 1). Making it FP8 is just declaring the block storage to be one byte per element instead of two — the *block management* (free list, block tables, ref counting, prefix-hash sharing, preemption) is completely untouched; it never cared what was *inside* a block, only how blocks are allocated and addressed. So paging, continuous batching, chunked prefill, prefix caching, and speculative decoding all keep working verbatim on a cache that now holds twice as many tokens per byte. The only two places that touch the *bytes* are: (1) **on write** — when a freshly computed K/V element is stored into a cache slot, convert it from the activation dtype to FP8 with the scale; (2) **on read** — when the attention kernel loads a cached element, convert it back up to compute precision. Both are single per-element conversions at the cache boundary.

The write path is where the KV is reshaped into the paged layout and stored, so the conversion lives right there: instead of `dst = static_cast<OutT>(src)` (a plain FP16→FP16 copy), do `dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale)` — scale and round the activation-dtype element down into the FP8 byte as it goes into the block. The cache element type `OutT`/`cache_t` becomes a one-byte type (`uint8_t` storage carrying the FP8 bits), so each cached element is half the size; everything downstream that addresses blocks by stride keeps working because the block dimensions and strides are computed from that element size. The read path mirrors it: the paged attention kernel that does the logical→physical block lookup (rung 1) loads the FP8 byte and `scaled_convert`s it up to the compute dtype before the QK dot and the value weighting, so the math is still full precision over de-quantized values.

I should be honest about where this can cost something and pick the default accordingly. The error is the one-time rounding of stored KV; for most models and workloads it's negligible after softmax smoothing, which is why E4M3 (finer mantissa) is the default and the scale is there to use the range well. But the magnitude of any quality effect, and the exact memory-into-throughput payoff, depend on the model, the sequence-length distribution, and how memory-bound the deployment is — a long-context or high-concurrency workload that is memory-bound gains the most (more tokens per byte directly buys more batch / longer context), while a workload already compute-bound gains less. So like the middle rungs, the throughput gain here is config-sensitive: "halving KV bytes per element roughly doubles the cache's token capacity, raising the achievable batch / context within the same memory; the realized throughput-at-fixed-latency gain is reproducible by running the throughput benchmark with the FP8 KV-cache flag on a given model/hardware/workload," not a fixed multiplier — and the quality side should be checked, not assumed.

Which brings the whole ladder back to its single fixed task and closes it. The goal was never "use a clever cache layout" or "decode several tokens at once" for their own sake — it was one thing throughout: **serve a fixed LLM on fixed GPU(s) at the maximum generated-tokens-per-second while holding per-request latency under a fixed budget.** Every rung was an attack on whatever was, at that moment, capping the batch the GPU could keep busy at fixed latency: paging removed the memory waste that capped the batch; continuous batching kept the batch full; chunked prefill kept prefill from stalling it; prefix caching stopped it re-doing shared work; speculative decoding squeezed more tokens out of each decode pass. And now FP8 KV storage attacks the deepest cap of all — the bytes-per-token of the cache itself — doubling how many requests fit in the same memory, which is the ceiling all the other rungs are pressing against. Each token now costs one byte to remember instead of two; the cache holds twice as much; the batch the fixed GPU can serve at the fixed latency budget grows accordingly. That is the last lever, because it changes the cost of the binding resource itself.

The core is the convert-on-write into the FP8-typed paged block.

```cpp
// csrc/.../cache_kernels.cu — reshape_and_cache write path (excerpt), plus the
// per-element op. The KV cache element type (cache_t) is now a 1-byte FP8 type
// (default E4M3, see dtype_fp8.cuh) instead of 2-byte FP16/BF16: half the bytes
// per cached token -> ~2x the tokens fit in the same GPU memory. Block layout,
// strides, block tables, free list, prefix-hash sharing all unchanged.

// One element: copy with an optional FP8 scaled conversion at the cache boundary.
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;
  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);                       // unquantized: plain copy
    } else {
      dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale);  // -> 1-byte FP8
    }
  }
};

// In reshape_and_cache_kernel: build the per-element ops with the K/V scales
// (scale only used when not kAuto), then store K and V into the paged block.
float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;
CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, x, 0, 1, k_op);  // K -> FP8 block
for (int i = 0; i < x; i++)
  v_op(value_dst[i * block_size], value_src[i]);                      // V -> FP8 block
// On read, the paged attention kernel scaled_converts the FP8 byte back up to
// the compute dtype before the QK dot and value weighting -- store in FP8,
// compute in full precision.
```

```cpp
// csrc/attention/dtype_fp8.cuh — the 8-bit KV storage formats.
enum class Fp8KVCacheDataType {
  kAuto = 0,     // unquantized (2-byte FP16/BF16)
  kFp8E4M3 = 1,  // 4 exp / 3 mantissa: finer steps -- the default ("fp8")
  kFp8E5M2 = 2,  // 5 exp / 2 mantissa: wider range, coarser
};
// Selected at serve time by --kv-cache-dtype fp8 (== fp8_e4m3) or fp8_e5m2.
```
