25.5 tokens/s, and the bandwidth diagnostic confirms what I argued before running it: this is achieving
maybe ~340 GB/s against a ~2 TB/s ceiling. Let me turn that into a duty cycle, because the ratio is the
whole story. `340 / 2000 ≈ 0.17` — the device is doing useful streaming work about a sixth of the time
and idle the other five-sixths. Put it in the time domain: 25.5 tok/s is `1/25.5 ≈ 39 ms` per token, and
of that only the ideal weight read — `13.5 GB / 2 TB/s ≈ 6.75 ms` — is unavoidable. So roughly `39 − 6.75
≈ 32 ms` of every token is *not* weight streaming. That 32 ms is the host overhead I argued about at the
baseline, now quantified: about five times larger than the actual work. That flips my mental model of the
problem completely. I came in thinking the enemy was the 13.5 GB weight read, and it is — eventually — but
right now the weight read isn't even the binding constraint, because the GPU is *idle*, waiting on the
host to hand it the next kernel. The first 4× to 5× of speedup isn't sitting in memory bandwidth at all;
it's sitting in CPU overhead. I have to make the GPU run flat-out before "bandwidth bound" is even a
description that fits.

So: where is the host time going, and how do I delete it? It is going into launching kernels. One decode
step is dozens of CUDA kernels per layer — the QKV projection, the rotary embedding, the attention, the
output projection, two RMSNorms, three feed-forward matmuls and the SiLU — times 32 layers, each one
dispatched through PyTorch eager: a Python call, the dispatcher resolving the op, the allocator finding
memory for the output, the launch. Each of those is microseconds of CPU, and there are hundreds of them
per token, and the GPU work each one triggers is so small it finishes before the next launch is ready.
The 32 ms breaks down as roughly `(number of launches) × (per-launch host cost)`: if there are ~400
launches at ~5-10 µs of host time apiece plus the per-token Python bookkeeping, that lands in exactly the
tens-of-milliseconds range I'm measuring. So the fix has to attack the *number of launches and the
per-launch cost*, not the arithmetic — the arithmetic is already ~150× cheaper than the memory traffic
and touching it would be pointless.

Let me lay out the design space honestly, because there's more than one way to attack launch overhead and
I want to pick by argument, not reflex. Option one: hand-fuse the hot kernels myself — write a fused
RMSNorm, a fused SwiGLU — in Triton or CUDA. That would cut the kernel count, but it's a huge amount of
bespoke code, it's confined to the ops I bother to fuse, and it doesn't touch the *per-launch* cost of the
kernels I leave alone; I'd be chipping at the 400 rather than collapsing it, and I'm restricted to the
scaffold's editable slots anyway. Option two: `torch.compile` in its default mode. Tracing the decode step
and letting an inductor backend lower it would fuse the pointwise chains automatically — the RMSNorm's
square/mean/rsqrt/scale collapses into one kernel instead of five, the SiLU-gated MLP fuses its elementwise
parts — which cuts the kernel *count* substantially and is essentially free to apply. That's a real win,
but it leaves the *residual* per-launch cost of the fused kernels: even a well-fused decode step still
issues a few dozen kernels per token through the normal launch path, and at batch 1 those launches can
still outrun the tiny GPU work. Option three: capture the whole step into a **CUDA graph** and replay it.
A CUDA graph records the entire sequence of launches once and re-issues them as a single device-side
operation, so the per-step host cost of launching hundreds (or, post-fusion, dozens) of kernels collapses
to essentially one host call. That's the one that attacks both terms — count *and* per-launch cost — and
it's exactly the regime CUDA graphs exist for.

The two levers, fusion and graph capture, aren't mutually exclusive; they compose, and the right move is
to pull them together. `torch.compile` with `mode="reduce-overhead"` is precisely fusion-plus-CUDA-graph:
it lowers and fuses the traced step, then wraps it in CUDA-graph capture-and-replay. So I hand the whole
decode step to the compiler, let it fuse the pointwise chains into single kernels to cut the count, and
let it capture the fused sequence into a graph so the per-step launch cost of what remains collapses to
one host call. If my 32-ms-of-overhead estimate is right and reduce-overhead deletes almost all of it,
the per-token time should fall from ~39 ms toward the ~6.75 ms weight-read floor — call it a ~4× jump.

But CUDA graphs come with a hard precondition that the rest of the design has to satisfy, and this is
where the KV-cache decision I flagged at the baseline comes back to bite. A graph records *specific*
kernels operating on *specific* memory addresses with *specific* shapes. Replay only works if nothing
about that changes between steps — same tensor sizes, same buffers, every iteration. And that is exactly
what my naive KV-cache violates. Growing the cache by concatenation means the key and value tensors are a
different size on every step, allocated at a different address, with the attention matmul seeing a
sequence dimension that ticks up by one each time. Dynamic shapes like that can't be captured into a
static graph, and even the compiler alone would have to re-specialize or guard on every new length,
re-tracing and defeating the point. The growing cache is the thing standing between me and the overhead
deletion — it is not a side issue, it is *the* blocker, and it's why compile and static-cache have to
land as one rung rather than two.

So the cache has to become *static*. Instead of growing it, I preallocate it once at full size — for the
whole `max_seq_length` I'll ever decode to — and then each step *writes into a slot* rather than appending.
Let me size it concretely to be sure it fits and to understand what I'm spending. Per layer the cache is a
`(max_batch_size, n_heads, max_seq_length, head_dim)` buffer for keys and another for values. At batch 1,
32 heads, head_dim 128, and say a generous `max_seq_length` of a few hundred, that's `1 × 32 × ~300 × 128`
elements × 2 bytes ≈ a few MB per buffer per layer, times 2 (K and V) times 32 layers — on the order of a
few hundred MB total. That's trivial against the 80 GB card and trivial against the 13.5 GB of weights, so
preallocating at full size costs me almost nothing in memory and buys me the static shapes the graph needs.
The update for a new token scatters its key and value into the row indexed by the current position and
returns the whole (fixed-shape) buffer for attention to read. Shapes never change, addresses never change,
so the compiled graph can be captured and replayed. This is what makes `setup_caches` a real piece of work:
it's the one-time allocation, before the loop, that turns the decode step into something graph-able.

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

Let me trace the shapes through `update` once by hand to make sure the static-buffer story actually holds
and I'm not fooling myself. On a decode step `input_pos` is a single position, so `S = 1`; `k_val` comes
in as `[B=1, H=32, S=1, D=128]`. The line `k_out[:, :, input_pos] = k_val` indexes the third axis at that
one position and writes the `[1, 32, 1, 128]` slice in place — the buffer `k_out` keeps its full
`[1, 32, max_seq_length, 128]` shape and its address throughout, and I return that full buffer. So the
attention downstream always sees a fixed `[1, 32, max_seq_length, 128]` key tensor, every step, regardless
of how far into the sequence I am. That's the invariant the CUDA graph needs, confirmed by walking the
indices. There's a subtlety I should be honest about: attention now reads the whole preallocated buffer,
including slots at positions I haven't written yet, so correctness relies on the causal mask forbidding
attention to future/empty slots — the mask is what makes reading the fixed-size buffer equivalent to
reading only the valid prefix. That's exactly right and it's the price of static shapes: I trade a tiny
bit of wasted attention compute over masked-out slots (negligible — attention arithmetic is not the
bottleneck) for a graph-able, address-stable step.

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

The `find_multiple(max_seq_length, 8)` rounds the cache length up to a multiple of 8, which keeps the
buffer's inner strides aligned for the kernels — a small detail, but it's the kind of thing that, left
ragged, produces an unaligned tensor the compiler has to special-case, so I round it. And precomputing
`freqs_cis` once and indexing it by `input_pos` turns the rotary embedding from a per-step recomputation
(more kernels) into a static gather from a fixed table — same static-read discipline as the cache,
applied to the other position-dependent input.

With the cache static, I compile the decode step in reduce-overhead mode so it gets captured into a CUDA
graph, and I leave prefill on its own path (it sees the whole prompt at once, a different shape, and runs
once — not worth graphing the same way):

```python
decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
```

`fullgraph=True` is deliberate, and worth dwelling on because it encodes the whole precondition as an
assertion. It forces the entire decode step to compile as a single graph with no break back into Python. A
graph break would re-introduce exactly the host round-trip I'm trying to delete — control returns to the
interpreter mid-step, a launch goes out the slow path — and worse, it would *split* the CUDA graph into
two captured segments with an un-captured gap between them, so I'd lose the single-host-call property
precisely where the break sits. By demanding `fullgraph=True` I make the compile *fail loudly* if anything
in the step is un-traceable, rather than silently leaving a break in that quietly caps my speedup. If it
does fail, that failure is information: it tells me exactly which op broke the graph so I can make it
static, which is far better than a mysterious 2× instead of 4×.

Prefill is compiled separately and differently. It runs once per generation over the whole prompt, so its
shape depends on the prompt length and isn't worth capturing into a fixed graph; I compile it with
`dynamic=True` so the compiler specializes on a symbolic length rather than a fixed one, and I don't ask
for reduce-overhead there because a single prefill launch already does enough GPU work to hide its own
dispatch — prefill was never the starved regime. Keeping prefill and decode on separate compiled paths is
the clean division: the graph discipline is spent exactly where it pays, the decode loop.

There is one cost to be honest about: the first call pays a real compilation latency — the trace, the
lowering, and the graph capture all happen on the first invocation, and that's seconds, not microseconds.
That would wreck a benchmark that timed it. But the fix is the same warmup I already established at the
baseline: run a throwaway generation first to trigger the compile and fill the caches, then time only
steady state. Over a 200-token generation — and over any realistic serving run — the one-time compile is
amortized to nothing, so timing steady state is the honest measurement of what a served model would
sustain. This is the subtlety I flagged at the baseline as "mattering for the next rung," and this is the
rung: eager had no compile, so its warmup was only allocator warmup; here the warmup also pays the compile,
and steady-state is what counts.

Before I commit I want to check whether the static cache *alone*, without the compile, would buy much —
because if it did, I'd be conflating two wins in one rung and should think harder about attribution. It
wouldn't, and the reason is instructive. Making the cache static removes the per-step reallocation and
copy of the growing-`cat` version, which is real but small: a few MB of copy per step against a 13.5 GB
weight read is under a percent of the traffic, and the allocator churn it saves is a slice of the host
overhead, not the bulk. In pure eager, a static cache still dispatches all ~400 kernels through the slow
path every token, so the device stays starved and tokens/s barely moves. Symmetrically, the compile
*alone* — without a static cache — can't capture the CUDA graph at all, because the shapes keep changing,
so I'd get only the fusion half of reduce-overhead and lose the graph-replay half. Neither piece is worth
much without the other; the win is genuinely joint, which is exactly why they belong in one rung. The
static cache isn't a throughput optimization in its own right — it's the *enabling condition* that lets
the compile do the thing that actually pays.

One more precondition of CUDA-graph replay I have to respect: the graph pins not just the cache addresses
but the *input* tensor's address too. Replay re-runs the recorded kernels against fixed memory, so the
`cur_token` I feed the compiled step each iteration must live at a stable address that the graph writes
into, rather than a freshly allocated tensor per step. reduce-overhead handles this by managing static
input buffers under the hood — it copies my input into the captured buffer before replay — but it means
the decode loop has to keep feeding single-token, fixed-shape inputs, which it does: `cur_token` is always
`[B, 1]`. This is why the whole design hangs together only if *everything* the graph touches is
shape-stable — the cache (via preallocation), the rotary table (via precompute-and-index), and the input
token (via fixed `[B,1]` shape). Any one of them going dynamic re-opens the host round-trip. Walking that
list is how I convince myself `fullgraph=True` will actually succeed rather than throw.

The `.clone()` calls in the scaffold's `decode_n_tokens` loop, which I noted at the baseline as Python
bookkeeping overhead, turn out to be load-bearing once the step is a replayed CUDA graph, and I want to be
sure I understand why before I compile — otherwise I might "optimize" them away and get silent corruption.
A CUDA graph writes its outputs into the *same* fixed buffers on every replay. So the `next_token` tensor
the compiled `decode_one_token` returns is not a fresh tensor each step — it's a view onto the graph's
static output buffer, which the *next* replay will overwrite. If I appended that tensor to `new_tokens`
without cloning, every entry in the list would end up aliasing the same buffer and reading back the last
token 200 times. `new_tokens.append(next_token.clone())` copies the value out of the static buffer into a
fresh tensor before the next replay clobbers it, which is exactly correct. So the `.clone()`s I called
overhead at the baseline are cheap insurance that becomes *necessary* here; the small per-step copy is the
price of the graph reusing its buffers, and it's negligible against the weight read. This is the kind of
correctness trap that only shows up once graph replay is in play, and spotting it now is why I trace the
data flow rather than just flipping on reduce-overhead and hoping.

I should also state plainly what this rung does *not* change: the numerics. Fusion re-associates and
co-schedules the same operations; CUDA-graph capture re-issues the same kernels in the same order; the
static cache stores the same keys and values I would have concatenated, just in preallocated slots read
under a causal mask. None of that alters a single output value — it's a pure scheduling and memory-layout
change, so the model's output distribution is bit-for-bit (modulo the usual non-associativity of floating
point, which fusion can perturb at the last ulp, well below any behavioral threshold) what the eager
baseline produced. That's why this rung needs no accuracy check under the evaluation settings: it's the
"purely scheduling" case. The quality bar set at the baseline is preserved by construction, and the entire
gain is throughput.

Now the prediction, against the 25.5-tok/s, ~340-GB/s baseline. The argument is arithmetic: ~32 ms of the
39-ms token is host overhead, and CUDA-graph replay over a fused, static graph should delete almost all of
it, leaving the token dominated by the ~6.75 ms weight read plus whatever irreducible launch and sampler
cost survives. If overhead went to near zero I'd approach the 148 tok/s ideal; realistically some
overhead and some bandwidth inefficiency remain, so I expect to land short of that — somewhere near 100
tok/s, a roughly 4× jump, the single largest multiplier I expect in the whole ladder. Equally important is
what should happen to the *diagnostic*: the achieved bandwidth should leap from ~340 GB/s toward a large
fraction of 2 TB/s, because the device stops idling and starts streaming weights continuously. That
bandwidth jump is the real signal — it's the direct evidence that the constraint has shifted from
host-overhead to memory-bandwidth. And that reframing is the important part: once this rung lands, the
device really is streaming weights as fast as it can, and from here on the only way to go faster is to
*reduce the bytes per token* — because once the achieved bandwidth is a large fraction of a fixed 2 TB/s
ceiling, scheduling has almost nothing left to give, and the weight read I dismissed as "not yet the
constraint" becomes the entire game. If the
number comes in near 100 and the achieved bandwidth jumps with it, the bet is right and the problem has
changed shape. If tokens/s rose but achieved bandwidth stayed low, then a graph break or a residual sync
is still starving the device and I'd hunt it down. The change is the static `KVCache` + `setup_caches`
plus the `reduce-overhead` compile of the decode step; full scaffold code in the answer.
