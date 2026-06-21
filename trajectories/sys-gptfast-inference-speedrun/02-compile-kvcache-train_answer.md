The eager baseline measured 25.5 tok/s, and the bandwidth diagnostic confirms the diagnosis: it achieves only about 340 GB/s against the A100's ~2 TB/s ceiling, so the device is busy less than a fifth of the time. That flips the mental model of the problem. The eventual enemy is the 13.5 GB weight read, but right now that read is not even the binding constraint — the GPU is *idle*, waiting on the host to hand it the next kernel. The first large multiplier is not in memory bandwidth at all; it is in CPU overhead. I have to make the GPU run flat-out before "bandwidth bound" is even an accurate description.

The host time is going into launching kernels: one decode step is hundreds of tiny CUDA kernels dispatched through eager PyTorch, each one a Python call, a dispatcher resolution, an allocator request, and a launch, and the GPU work each triggers finishes before the next launch is ready. The fix must attack the *number of launches and the per-launch cost*, not the arithmetic, which was never the limiter. I propose two changes that have to be pulled together: **`torch.compile` in `reduce-overhead` (CUDA-graph) mode plus a static KV-cache**.

The first lever is to stop dispatching op-by-op in Python and hand the whole decode step to a compiler that sees the entire forward pass at once. Tracing `decode_one_token` and lowering it through an inductor-style backend lets it **fuse** chains of pointwise ops into single kernels — the RMSNorm's square-mean-rsqrt-scale collapses from five kernels into one, the SiLU-gated MLP fuses its elementwise parts — cutting the kernel *count* directly. Once the step is compiled I go further and capture the whole kernel sequence into a **CUDA graph** and replay it: a graph records the launches once and re-issues them as a single device-side operation, so the per-step host cost of launching hundreds of kernels collapses to essentially one call. That is exactly what `mode="reduce-overhead"` does, and `fullgraph=True` is deliberate — it forces the entire decode step to compile as one graph with no break back into Python, because a graph break would reopen the host round-trip I am trying to delete and would split the CUDA graph, so I want the compile to fail loudly rather than silently leave a break in.

CUDA graphs carry a hard precondition that the rest of the design must satisfy: a graph records *specific* kernels on *specific* memory addresses with *specific* shapes, and replay only works if none of that changes between steps. The naive KV-cache violates this. Growing the cache by concatenation makes the key/value tensors a different size at a different address every step, and the attention matmul sees a sequence dimension that ticks up by one each iteration — dynamic shapes that cannot be captured into a static graph. So the cache must become *static*: I preallocate it once, before the loop, at the full `max_seq_length` I will ever decode to, as a fixed `(max_batch_size, n_heads, max_seq_length, head_dim)` zero-filled buffer per layer, and each step *writes its key/value into the slot indexed by the current position* rather than appending, returning the whole fixed-shape buffer for attention to read. Shapes never change, addresses never change, so the compiled graph can be captured and replayed. This is what makes `setup_caches` real work: it is the one-time allocation that turns the decode step into something graph-able, and it also precomputes the rotary-embedding table so that read, too, is a static index by `input_pos`. The static cache is not an independent optimization but the *precondition that unlocks* the graph — the two only work together.

This is a pure scheduling change, so the output distribution is untouched; the only cost is a one-time compilation latency on the first call, which the benchmark warms up and which amortizes over a long generation, the regime that matters for serving. The expectation is that nearly the entire gap between 340 GB/s and the ~2 TB/s ceiling is host overhead, so CUDA-graph replay over a fused, static graph should be the single largest multiplier in the ladder — roughly $4\times$, landing near 100 tok/s — by converting the workload from latency-bound-on-the-CPU into genuinely bandwidth-bound-on-the-GPU. Once that happens the device is streaming weights as fast as it can, and the only way to go faster from here is to *reduce the bytes per token*, which is what every subsequent rung does.

```python
# model.py
class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out

class Transformer(nn.Module):
    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length,
                                           self.config.n_local_heads, head_dim, dtype)
        self.freqs_cis = precompute_freqs_cis(self.config.block_size,
            self.config.dim // self.config.n_head, self.config.rope_base, dtype, self.config.rope_scaling)

# generate.py — inside generate(): preallocate before the loop
with torch.device(device):
    model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

# generate.py — inside main(): capture the decode step into a CUDA graph
if compile:
    global decode_one_token, prefill
    decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
    if compile_prefill:
        prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
```
