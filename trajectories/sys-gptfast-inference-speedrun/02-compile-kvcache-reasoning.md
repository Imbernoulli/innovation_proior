25.5 tokens/s, and the bandwidth diagnostic confirms what I argued before running it: this is achieving
maybe ~340 GB/s against a ~2 TB/s ceiling. So the device is busy less than a fifth of the time. That
flips my mental model of the problem completely. I came in thinking the enemy was the 13.5 GB weight
read, and it is — eventually — but right now the weight read isn't even the binding constraint, because
the GPU is *idle*, waiting on the host to hand it the next kernel. The first 4× to 5× of speedup isn't
sitting in memory bandwidth at all; it's sitting in CPU overhead. I have to make the GPU run flat-out
before "bandwidth bound" is even a description that fits.

So: where is the host time going, and how do I delete it? It is going into launching kernels. One
decode step is dozens of CUDA kernels per layer — the QKV projection, the rotary embedding, the
attention, the output projection, two RMSNorms, three feed-forward matmuls and the SiLU — times 32
layers, each one dispatched through PyTorch eager: a Python call, the dispatcher resolving the op, the
allocator finding memory for the output, the launch. Each of those is microseconds of CPU, and there
are hundreds of them per token, and the GPU work each one triggers is so small it finishes before the
next launch is ready. The fix has to attack the *number of launches and the per-launch cost*, not the
arithmetic.

Two levers, and they have to be pulled together. The first is to stop dispatching op-by-op in Python
and instead hand the whole decode step to a compiler that can see the entire forward pass at once. If I
trace `decode_one_token` and let an inductor-style backend lower it, it can **fuse** chains of pointwise
ops into single kernels (the RMSNorm's square-mean-rsqrt-scale collapses into one kernel instead of
five; the SiLU-gated MLP fuses its elementwise parts), which cuts the kernel *count* directly. And once
the step is compiled, I can go further: capture the whole sequence of kernels into a **CUDA graph** and
replay it. A CUDA graph records the launches once and re-issues them as a single device-side operation,
so the per-step CPU cost of launching hundreds of kernels collapses to essentially one host call. That
is the `reduce-overhead` mode — it exists precisely for this regime, where the launches dominate.

But CUDA graphs come with a hard precondition that the rest of the design has to satisfy: a graph
records *specific* kernels operating on *specific* memory addresses with *specific* shapes. Replay only
works if nothing about that changes between steps — same tensor sizes, same buffers, every iteration.
And that is exactly what my naive KV-cache violates. Growing the cache by concatenation means the key
and value tensors are a different size on every step, allocated at a different address, with the
attention matmul seeing a sequence dimension that ticks up by one each time. Dynamic shapes like that
can't be captured into a static graph, and even the compiler alone would have to re-specialize or guard
on every new length. The growing cache is the thing standing between me and the overhead deletion.

So the cache has to become *static*. Instead of growing it, I preallocate it once at full size — for
the whole `max_seq_length` I'll ever decode to — and then each step *writes into a slot* rather than
appending. The cache is a fixed `(max_batch_size, n_heads, max_seq_length, head_dim)` buffer per layer,
zero-filled up front; the update for the new token scatters its key and value into the row indexed by
the current position and returns the whole (fixed-shape) buffer for attention to read. Shapes never
change, addresses never change, so the compiled graph can be captured and replayed. This is what makes
`setup_caches` a real piece of work: it's the one-time allocation, before the loop, that turns the
decode step into something graph-able.

```python
class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D] — scatter into preallocated slots, return full buffer
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out
```

`setup_caches` walks the layers and hangs one of these on each attention module, sized for the run, and
also precomputes the rotary-embedding table so that, too, is a static read indexed by `input_pos`:

```python
def setup_caches(self, max_batch_size, max_seq_length):
    head_dim = self.config.dim // self.config.n_head
    max_seq_length = find_multiple(max_seq_length, 8)
    self.max_seq_length = max_seq_length
    self.max_batch_size = max_batch_size
    for b in self.layers:
        b.attention.kv_cache = KVCache(max_batch_size, max_seq_length,
                                       self.config.n_local_heads, head_dim, dtype)
    self.freqs_cis = precompute_freqs_cis(self.config.block_size,
                                          self.config.dim // self.config.n_head, self.config.rope_base, dtype)
```

With the cache static, I compile the decode step in reduce-overhead mode so it gets captured into a CUDA
graph, and I leave prefill on its own path (it sees the whole prompt at once, a different shape, and runs
once — not worth graphing the same way):

```python
decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
```

`fullgraph=True` is deliberate: it forces the entire decode step to compile as a single graph with no
break back into Python. A graph break would re-introduce exactly the host round-trip I'm trying to
delete, and it would split the CUDA graph, so I want the compile to fail loudly rather than silently
leave a break in. There is one cost to be honest about: the first call pays a real compilation latency
(the trace, the lowering, the graph capture), so the benchmark warms up with a throwaway pass and only
times steady state — the compile cost is amortized over a long generation, which is the regime that
matters for serving.

Now the prediction, against the 25.5-tok/s, ~340-GB/s baseline. The argument is that the entire gap
between 340 GB/s and the ~2 TB/s ceiling is host overhead, and CUDA-graph replay over a fused, static
graph deletes almost all of it — so the GPU stops waiting and starts running near bandwidth. I expect
this to be the single largest multiplier in the whole ladder, roughly a 4× jump, landing somewhere near
100 tok/s, because it converts the workload from *latency-bound-on-the-CPU* to actually
*bandwidth-bound-on-the-GPU*. And that reframing is the important part: once this rung lands, the device
really is streaming weights as fast as it can, the bandwidth diagnostic should be a large fraction of
peak, and from here on the only way to go faster is to *reduce the bytes per token* — which is what every
subsequent rung does. If the number comes in near 100 and the achieved bandwidth jumps with it, the bet
is right and the problem has changed shape. The change is the static `KVCache` + `setup_caches` plus the
`reduce-overhead` compile of the decode step; full scaffold code in the answer.
