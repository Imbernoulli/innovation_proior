25.5 tokens/s, and the bandwidth diagnostic confirms what I argued before running it: ~340 GB/s against a
~2 TB/s ceiling. Turn that into a duty cycle — `340 / 2000 ≈ 0.17` — the device is streaming useful work
about a sixth of the time and idle the rest. In the time domain, 25.5 tok/s is `39 ms` per token, of which
only the ideal weight read `13.5 GB / 2 TB/s ≈ 6.75 ms` is unavoidable; the other ~32 ms is *not* weight
streaming. That 32 ms is the host overhead I argued about at the baseline, now quantified at about five
times the actual work. It flips my mental model: I came in thinking the enemy was the weight read, and it
is eventually, but right now the weight read isn't even the binding constraint because the GPU is idle
waiting on the host. The first 4-5× isn't in memory bandwidth at all; it's in CPU overhead.

The host time is going into launching kernels — dozens per layer times 32, each dispatched through eager
PyTorch, each triggering GPU work so small it finishes before the next launch is ready. The 32 ms is
roughly `(number of launches) × (per-launch host cost)`: ~400 launches at ~5-10 µs plus per-token Python
bookkeeping lands in exactly the tens-of-milliseconds range I measure. So the fix attacks the *count and
per-launch cost of kernels*, not the arithmetic (already ~150× cheaper than the memory traffic).

There's more than one way to attack launch overhead and I want to pick by argument. Hand-fusing the hot
kernels myself in Triton or CUDA would cut the count, but it's bespoke, confined to the ops I bother to
fuse, doesn't touch the per-launch cost of what I leave alone, and I'm restricted to the scaffold's slots
anyway. `torch.compile` default-mode fuses the pointwise chains automatically — the RMSNorm's five
elementwise kernels collapse to one, the SiLU-gated MLP fuses — cutting the count for free, but it leaves
the residual per-launch cost of the fused kernels, and at batch 1 even a few dozen launches can outrun the
tiny GPU work. Capturing the whole step into a **CUDA graph** and replaying it collapses the per-step cost
of launching hundreds (post-fusion, dozens) of kernels to essentially one host call — it attacks both terms.
Fusion and graph capture compose, and `torch.compile(mode="reduce-overhead")` is precisely
fusion-plus-CUDA-graph: it lowers and fuses the traced step, then wraps it in capture-and-replay. So I hand
the decode step to the compiler and let it do both. If my 32-ms estimate is right and reduce-overhead
deletes almost all of it, the per-token time falls from ~39 ms toward the ~6.75 ms floor — call it ~4×.

But CUDA graphs come with a hard precondition, and this is where the KV-cache decision I flagged at the
baseline comes back to bite. A graph records *specific* kernels on *specific* addresses with *specific*
shapes; replay only works if none of that changes between steps. My naive concatenating cache violates all
three — the key/value tensors are a different size and address every step, and the attention matmul sees a
sequence dimension ticking up by one. Dynamic shapes can't be captured, and the compiler alone would have
to re-specialize on every new length, re-tracing and defeating the point. The growing cache is *the*
blocker, which is why compile and static-cache have to land as one rung rather than two.

So the cache becomes static: preallocate once at full `max_seq_length`, then each step *write into a slot*
indexed by the current position and return the whole fixed-shape buffer. Size it to be sure it fits — per
layer, `(max_batch_size, n_heads, max_seq_length, head_dim)` for K and another for V; at batch 1, 32 heads,
head_dim 128, a few hundred positions, that's a few MB per buffer per layer, ×2 ×32 layers, on the order of
a few hundred MB total — trivial against 80 GB and against the 13.5 GB of weights. So preallocating buys
the static shapes the graph needs at almost no memory cost.

Tracing `update` by hand pins the invariant. On a decode step `input_pos` is one position, so `k_val` comes
in `[B=1, H=32, S=1, D=128]`; `k_out[:, :, input_pos] = k_val` writes that slice in place while `k_out`
keeps its full `[1, 32, max_seq_length, 128]` shape and address, and I return that full buffer. Attention
downstream always sees a fixed-size key tensor, every step — the invariant the graph needs. One subtlety:
attention now reads the whole preallocated buffer, including slots I haven't written yet, so correctness
relies on the causal mask forbidding attention to future/empty slots. That's the price of static shapes — a
tiny bit of wasted attention compute over masked slots (negligible; attention arithmetic isn't the
bottleneck) for a graph-able, address-stable step. `setup_caches` does the one-time work: it walks the
layers hanging a `KVCache` on each attention module sized for the run, rounds `max_seq_length` up to a
multiple of 8 (`find_multiple`) so the buffer's inner strides stay aligned rather than forcing the compiler
to special-case a ragged tensor, and precomputes the rotary `freqs_cis` table once so it too becomes a
static gather indexed by `input_pos` instead of a per-step recomputation. Full code is in the answer.

With the cache static I compile the decode step in reduce-overhead so it gets captured, and leave prefill on
its own path:

```python
decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
```

`fullgraph=True` encodes the whole precondition as an assertion: it forces the step to compile as a single
graph with no break back into Python. A graph break would reintroduce the host round-trip I'm deleting and
*split* the CUDA graph into two captured segments with an un-captured gap, losing the single-host-call
property precisely where the break sits. Demanding fullgraph makes the compile *fail loudly* if anything is
un-traceable rather than silently leaving a break that caps my speedup at 2× instead of 4× — and the
failure is information, naming exactly which op broke the graph so I can make it static. Prefill compiles
separately with `dynamic=True`: it runs once per generation over the whole prompt, its shape depends on
prompt length so it isn't worth a fixed graph, and it doesn't need reduce-overhead because a single prefill
launch already does enough GPU work to hide its own dispatch — prefill was never the starved regime.

Would the static cache *alone*, without compile, buy much? No, and the reason is instructive. It removes
the per-step reallocation and copy of the growing-`cat` version — real but small, a few MB against a 13.5 GB
read, under a percent of traffic — while in pure eager it still dispatches all ~400 kernels through the slow
path, so the device stays starved and tokens/s barely moves. Symmetrically, compile *alone* without a static
cache can't capture the graph at all, because the shapes keep changing, so I'd get only the fusion half of
reduce-overhead. Neither piece is worth much without the other; the static cache isn't a throughput
optimization in its own right, it's the *enabling condition* that lets the compile do the thing that pays.

Graph replay also pins the *input* tensor's address, not just the cache. reduce-overhead handles this by
managing static input buffers — it copies my input into the captured buffer before replay — but it means the
loop must keep feeding single-token, fixed-shape `[B, 1]` inputs, which it does. And it means the `.clone()`
calls I dismissed as bookkeeping at the baseline become *load-bearing*. A CUDA graph writes its outputs into
the same fixed buffers on every replay, so the `next_token` the compiled step returns is a view onto the
graph's static output buffer, which the next replay overwrites. Appending it without cloning would leave
every entry in `new_tokens` aliasing that one buffer, reading back the last token 200 times.
`new_tokens.append(next_token.clone())` copies the value out before the next replay clobbers it — cheap
insurance that becomes necessary here, negligible against the weight read. This is the kind of correctness
trap that only shows up once graph replay is in play, which is why I trace the data flow rather than just
flip on reduce-overhead and hope.

None of this changes the numerics. Fusion re-associates and co-schedules the same operations, graph capture
re-issues the same kernels in the same order, and the static cache stores the same keys and values I would
have concatenated, read under a causal mask. So the output distribution is what the eager baseline produced
(modulo the last-ulp non-associativity of floating point that fusion can perturb, well below any behavioral
threshold) — the pure-scheduling case that needs no accuracy check.

The prediction, against 25.5 tok/s at ~340 GB/s: ~32 ms of the 39-ms token is host overhead, and
CUDA-graph replay over a fused static graph should delete almost all of it, leaving the token dominated by
the ~6.75 ms weight read plus whatever irreducible launch and sampler cost survives. If overhead went to
zero I'd approach 148 tok/s; realistically some remains, so I expect somewhere near 100, a roughly 4× jump —
the largest multiplier I expect in the whole ladder. Equally important is the diagnostic: achieved
bandwidth should leap from ~340 GB/s toward a large fraction of 2 TB/s as the device stops idling and starts
streaming continuously. That bandwidth jump is the real signal that the constraint has shifted from host
overhead to memory bandwidth — and once it lands, the only way to go faster is to *reduce the bytes per
token*, because scheduling has almost nothing left to give against a fixed 2 TB/s ceiling. If tokens/s rose
but achieved bandwidth stayed low, a graph break or a residual sync is still starving the device and I'd
hunt it down. The change is the static `KVCache` + `setup_caches` plus the reduce-overhead compile of the
decode step; full scaffold code in the answer.
