Before I change anything, I want to know what the most direct implementation actually costs, because
the cost structure is the whole game here and I'd rather measure it than guess. The temptation, staring
at a decode loop everyone already knows is slow, is to skip straight to the fix — reach for a fused
attention kernel, or start compiling the step — but that would leave me without a clean zero to read the
first real win against. So I take the loop as it stands: eager PyTorch, no compilation, the cache
extended by concatenation each step, and I just run it. The prompt goes through `prefill`, then
`decode_n_tokens` calls `decode_one_token` once per token, samples, and appends, exactly as written in
the scaffold. I want the honest floor first.

What regime does this workload sit in? One decoded token at batch 1 does two matmuls per attention (QKV
up-projection and output projection) and three per feed-forward (w1, w3 up, w2 down), across 32 layers,
plus embedding and head. The arithmetic is roughly `2 × params` FLOP — a multiply and an add per weight
touched — which for ~6.7B parameters is ~13-14 GFLOP for the whole token, matching the context's "~14
GFLOP." To do that arithmetic I read every weight once, and the weights are bf16, 2 bytes each, so ~13.5
GB crosses HBM per token. The ratio is the arithmetic intensity: `14e9 / 13.5e9 ≈ 1` FLOP per byte. One.
That is the single most important fact about this problem. The A100 does ~312 TFLOP/s of bf16 and streams
~2 TB/s from HBM, so its roofline ridge — the intensity where a kernel stops being memory-bound — is
`312e12 / 2e12 ≈ 156` FLOP per byte. My workload sits at intensity ~1, more than two orders of magnitude
to the left of the ridge. No scheduling and no kernel moves a workload at intensity 1 into the
compute-bound regime; it is memory-bound by an enormous margin. What matters is how fast I can move 13.5
GB. Streaming it at the full 2 TB/s would take `13.5e9 / 2e12 ≈ 6.75 ms` per token, ~148 tokens/s — the
"~150 tokens/s ideal" the context quotes, now rederived from the byte count. Against that, the compute is
`14e9 / 312e12 ≈ 45 µs` if the matmuls ran at peak: the arithmetic is ~150× cheaper than the weight read.
I am not fighting the math, I am fighting the memory system and the machinery around it.

But that 148 tok/s ceiling assumes the GPU is *busy* streaming weights the whole time, and in eager mode
it almost certainly isn't. Every op dispatches through PyTorch's Python machinery one at a time: a Python
call, the dispatcher resolving the kernel, the allocator finding output memory, then the launch. Counting
the kernels: a transformer block is an RMSNorm (several elementwise kernels — square, mean, rsqrt,
multiply, scale), the QKV matmul, rotary embedding (a couple more), the attention, the output projection,
a residual add, a second RMSNorm, then w1, w3, SiLU and the gating multiply, w2, another residual — a
dozen or more kernels per layer, times 32, plus embedding and head: on the order of 400 launches to
produce one token. Each launch is microseconds of *CPU* work that must finish before the GPU can start
that kernel, and the GPU work per kernel is tiny. The QKV matmul at batch 1 is a `1 × 4096` activation
times a `4096 × 12288` weight; its FLOP is ~0.1 GFLOP (microseconds on the A100), while reading its
`4096 × 12288 × 2 B ≈ 100 MB` weight costs ~50 µs at 2 TB/s. So each kernel is at most tens of
microseconds of GPU time, and the host has to prepare and enqueue the next one within that window,
serially. If eager dispatch costs even 5-10 µs of host time apiece across ~400 kernels, the host spends
milliseconds per token just *issuing* work, and the GPU repeatedly drains its queue and stalls waiting.
The expensive A100 spends a large fraction of every decode step idle — not because it is slow but because
it is starved.

That gives me a falsifiable prediction I can check the moment the number comes back. If the eager loop
were already bandwidth-bound — pinned near 2 TB/s — the host would have to keep the launch queue full, i.e.
each dispatch cheaper than the GPU kernel it feeds; but the kernels here are microseconds, among the
smallest useful GPU work there is, and eager dispatch is not microsecond-cheap. So I should expect the
eager number *well below* the 148 ceiling — not because bandwidth is the limiter but because the device is
idle. I compute achieved bandwidth as `model_size × tokens_per_sec`, and if it comes in at a small
fraction of 2 TB/s the device was starved, and that gap is host overhead I have not yet paid to remove. If
it came back near peak, my whole mental model would be wrong.

The two phases of generation sit in different regimes and I don't want to conflate them. Prefill runs the
whole prompt once: for length `L` the QKV matmul is `L × 4096` times `4096 × 12288`, so its arithmetic
scales with `L` while the weight read stays 13.5 GB — intensity ~`L`, not ~1. Even a modest prompt pushes
prefill toward compute-bound, and one prefill kernel does enough GPU work to hide the launch before it.
Prefill is not my problem. Decode is: there `L = 1` every step, intensity pinned at ~1, and the 200 tokens
are 200 serial intensity-1 passes. The benchmark's tiny 5-token prompt is deliberate — it isolates the
decode regime where every free variable in the editable interface lives. And this is exactly why batch 1
is the adversarial setting. If I could batch 32 requests, the one 13.5 GB weight read would amortize over
32 tokens, intensity would climb to ~32, and launch overhead would hide behind real GPU work. Batch 1
forbids that amortization: every token pays the full weight read alone. So every lever I have must either
cut the bytes a token reads, cut the passes a token costs, or add parallel bandwidth — never "pack more
tokens per pass."

There's a second decision buried in this baseline I want to make deliberately: the KV-cache. The naive
thing — no thought required — is to grow it by concatenation: each step `cat` the new key/value onto the
running tensors so attention sees the whole history. I keep exactly that for the baseline, but it's a
*choice* with costs I'm choosing to eat. Concatenating a row every step reallocates a one-larger tensor
and copies the old contents — allocator traffic that grows with sequence length — and the shapes the model
sees change every iteration, the attention sequence dimension ticking up by one each time. For a floor
whose job is to expose overhead that's fine; it's *more* overhead, on-brand. But I note the instability
explicitly, because a growing, reallocating, shape-shifting cache is exactly what would obstruct any scheme
that wants the decode step identical every iteration — so when I come to attack launch overhead, the cache
layout is back on the table.

The `decode_n_tokens` loop adds its own overhead on top of the forward. Each iteration does `input_pos +=
1`, appends `next_token.clone()` and `next_prob.clone()` to Python lists, invokes the callback, reassigns
`cur_token`. The `.clone()`s are small device copies, the appends are Python-object churn, and it's an
interpreted `for` loop reaching the CUDA boundary afresh every token — CPU work interleaved with the same
host thread the GPU is waiting on. So eager overhead has two stacked sources — per-op dispatch inside the
forward, and per-token Python bookkeeping around it — both scaling with the step count, which is why a
200-token generation makes the effect so visible. I keep every line as written; I just want it named so I
know what a later rung collapses. (The sampler uses the exponential/argmax trick,
`multinomial_sample_one_no_sync`, specifically to avoid a `torch.multinomial` CPU-GPU sync every token —
the right shape for a throughput loop, so I keep it; it's a handful of small kernels on a 32000-vector,
not where the time goes.)

Why is *this* the right zero rather than some slightly-optimized start? Because a baseline's only job is to
isolate one thing so the next rung's win reads cleanly. If I folded a fused attention kernel or a static
cache in, then when I later delete the host overhead I couldn't attribute the speedup. Pure eager with the
naive cache isolates exactly the per-token CPU launch overhead, and the first real rung's gain reads as
precisely "what removing that overhead buys." I'll measure by generating 200 tokens from the fixed 5-token
prompt, discarding the first run as warmup (allocator growth and cache filling are one-time), and taking
the mean over several runs. Eager has no compile step, so warmup here is only reaching steady allocator
state — a subtlety that will matter for the *next* rung, not this one. I report both tokens/s and the
derived achieved bandwidth, because the pair is diagnostic: tokens/s says "slow," but achieved bandwidth
relative to 2 TB/s says *why* — starved versus bandwidth-bound — and only the "why" tells me what to fix
first.

The baseline changes nothing numerically — native bf16, exact attention, the true temperature-0.8,
top-k-200 sampler — so its outputs *are* the reference behavior every later rung is checked against, and it
needs no accuracy check. My expectation, from the roofline and the launch-overhead argument, is that
tokens/s comes in low and achieved bandwidth at a small fraction of 2 TB/s: eager decode at batch 1 leaves
the device mostly idle waiting on the host. If that's what I see, the first thing worth doing is not
touching the arithmetic or the bytes at all, but making the GPU run continuously instead of starved. The
default fill is the scaffold's decode loop, run as written; it is in the answer.
