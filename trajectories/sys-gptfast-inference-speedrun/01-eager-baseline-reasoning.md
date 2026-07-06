Before I change anything, I want to know what the most direct implementation actually costs, because
the cost structure is the whole game here and I'd rather measure it than guess. The temptation, staring
at a decode loop that everyone already knows is slow, is to skip straight to the fix — reach for a fused
attention kernel, or start compiling the step — but that would leave me without a clean zero to read the
first real win against. So I take the loop as it stands: eager PyTorch, no compilation, the cache
extended by concatenation each step, and I just run it. The prompt goes through `prefill`, then
`decode_n_tokens` calls `decode_one_token` once per token, samples, and appends, exactly as written in
the scaffold. I want the honest floor first.

Let me be quantitative about what regime I think this workload sits in, because that framing decides
everything that follows. One decoded token at batch 1 does two matmuls per attention (QKV up-projection
and the output projection) and three per feed-forward (w1, w3 up, w2 down), across 32 layers, plus the
embedding lookup and the final head. The arithmetic is roughly `2 × params` FLOP — two flops (a multiply
and an add) per weight touched — which for ~6.7B parameters is about 13-14 GFLOP for the whole token.
That matches the context's "~14 GFLOP of useful compute." Now the bytes: to do that arithmetic I have to
read every weight once, and the weights are bf16, 2 bytes each, so ~13.5 GB crosses HBM per token. The
ratio of those two numbers is the arithmetic intensity, and it is about `14e9 / 13.5e9 ≈ 1` FLOP per
byte. One. That is a stunningly low number, and it is the single most important fact about this problem.

To see why it's so damning I put it on a roofline. The A100 does on the order of 312 TFLOP/s of bf16 on
its tensor cores and streams about 2 TB/s from HBM. The ridge point — the arithmetic intensity at which
a kernel stops being memory-bound and starts being compute-bound — is `312e12 / 2e12 ≈ 156` FLOP per
byte. My workload sits at intensity ~1. That is more than two orders of magnitude to the *left* of the
ridge. There is no scheduling, no kernel, no cleverness that moves a workload at intensity 1 into the
compute-bound regime; it is memory-bound by an enormous margin, and it will stay memory-bound no matter
what I do to the arithmetic. The tensor cores are almost irrelevant here. What matters is how fast I can
move 13.5 GB.

So what is the ideal? If I could stream the weights at the full 2 TB/s with zero waste and perfectly
overlapped compute, one token would take `13.5e9 / 2e12 ≈ 6.75 ms`, which is about 148 tokens/s. That is
the ceiling the context quotes as "~150 tokens/s ideal for a perfect implementation," and now I've
rederived it from the byte count and the bandwidth, which reassures me I understand where it comes from.
Contrast that with the compute time: `14e9 / 312e12 ≈ 45 µs` per token if the matmuls ran at peak.
Forty-five microseconds of arithmetic against six-point-seven-five milliseconds of unavoidable weight
reading — the arithmetic is ~150× cheaper than the memory traffic. Even if my matmuls ran at a tenth of
peak I'd still be swamped by the weight read. This is the whole reason the problem is interesting: I am
not fighting the math, I am fighting the memory system and the machinery around it.

But here is the twist I actually expect to find when I run the eager loop, and it's why the naive version
is worth measuring rather than skipping. That 148 tok/s ceiling assumes the GPU is *busy* streaming
weights the whole time. In eager mode it almost certainly isn't. Every operation in the forward pass
dispatches through PyTorch's normal Python machinery one at a time: a Python-level call, the dispatcher
resolving which kernel to run, the caching allocator finding memory for the output, then the actual
launch onto the CUDA stream. Let me count the kernels. A transformer block is, roughly: an RMSNorm
(which by itself is several elementwise kernels — square, mean, rsqrt, multiply, scale), the QKV matmul,
the rotary embedding (a couple of elementwise kernels), the attention itself, the output projection, a
residual add, a second RMSNorm, then w1, w3, the SiLU and the gating multiply, w2, and another residual.
Call it a dozen or more kernels per layer, times 32 layers, plus embedding and head — on the order of 400
kernel launches to produce a single token. Each launch is microseconds of *CPU* work that must complete
before the GPU can start that kernel.

Now weigh those against each other. The GPU work per kernel is tiny. The QKV matmul at batch 1 is a
tall-skinny GEMM: a `1 × 4096` activation times a `4096 × 12288` weight (query, key, value stacked). Its
FLOP is `2 × 1 × 4096 × 12288 ≈ 0.1 GFLOP`, which at even a fraction of peak the A100 finishes in a few
microseconds; its *memory* cost is reading that `4096 × 12288 × 2 B ≈ 100 MB` weight, which at 2 TB/s is
~50 µs. So each of these kernels is at most tens of microseconds of GPU time — and the CPU has to prepare
and enqueue the next one within that window. If dispatching a kernel through eager PyTorch costs even
5-10 µs of host time, and there are ~400 of them, the host spends on the order of a few milliseconds per
token just *issuing* work, and crucially it issues serially, so the GPU repeatedly drains its queue and
stalls waiting for the CPU to hand it the next launch. The expensive A100 spends a large fraction of
every decode step idle — not because it is slow but because it is starved.

Let me stress-test that reasoning against the alternative reading, because if I'm wrong about which
constraint binds I'll optimize the wrong thing on the next rung. Could the eager loop already be
bandwidth-bound — actually pinned near 2 TB/s, achieving something close to the 148 tok/s ideal? For that
to be true the host would have to keep the launch queue full so the GPU never waits, which requires each
CPU dispatch to be cheaper than the GPU kernel it feeds. But the GPU kernels here are microseconds, among
the smallest useful GPU work there is, and eager dispatch is not microsecond-cheap once you count Python,
the dispatcher, and the allocator. So the queue-full assumption fails, and I should expect the eager
number to land *well below* the 148 ceiling — not because bandwidth is the limiter but because the GPU is
idle. That is a falsifiable prediction I can check the moment the number comes back: I compute the
achieved bandwidth as `model_size × tokens_per_sec`, and if it is a small fraction of 2 TB/s, the device
was starved, and the gap between achieved and peak is host overhead I have not yet paid to remove. If
instead the achieved bandwidth came back near peak, my whole mental model would be wrong and I'd have to
look for the stall somewhere else.

It's worth separating the two phases of generation here, because they sit in completely different regimes
and I don't want to conflate them. Prefill runs the whole prompt through the model once: for a length-`L`
prompt the QKV matmul is `L × 4096` times `4096 × 12288`, so its arithmetic scales with `L` while the
weight read stays fixed at 13.5 GB. That means prefill has arithmetic intensity ~`L`, not ~1 — even a
modest prompt of a few dozen tokens pushes prefill toward the compute-bound side of the roofline, and one
prefill kernel does enough GPU work to hide the launch that precedes it. Prefill is not my problem. The
decode loop is: there `L = 1` on every step, intensity is pinned at ~1, and the 200 tokens I generate are
200 separate, serial, intensity-1 passes. The benchmark deliberately uses a tiny 5-token prompt precisely
so the measurement is dominated by the decode loop and not by prefill — it isolates the hard regime, and
that is the regime every free variable in the editable interface lives in.

And this is exactly why batch 1 is the adversarial setting rather than an easy one. If I were allowed to
batch, say, 32 requests together, the weight read would be shared across all 32 tokens in a step — one
13.5 GB read amortized over 32 tokens — and the arithmetic intensity would climb to ~32, dragging the
workload toward the compute-bound regime where the tensor cores earn their keep and the launch overhead is
hidden behind real GPU work. Batch 1 forbids that amortization. Every token pays the full weight read by
itself, intensity stays at 1, and the GPU kernels stay small enough that host overhead can starve them.
So the batch-1 constraint in the evaluation settings isn't incidental; it is the thing that makes this a
latency/bandwidth problem rather than a throughput-batching problem, and it means every lever I have must
either cut the bytes a single token reads, cut the number of passes a token costs, or add more parallel
bandwidth — not "pack more tokens per pass," which the batch-1 rule takes off the table.

Let me trace one decode step as a timeline to make the starvation concrete, using only the structure and
the per-kernel estimates above. The host wakes up, dispatches the embedding gather (a few µs of CPU),
the GPU does a trivial lookup (sub-µs) and goes idle. The host dispatches the first RMSNorm's several
elementwise kernels one by one; each is a tiny GPU op that finishes before the next dispatch arrives, so
the GPU idles between them. Then the QKV matmul: ~5-10 µs of CPU to launch, ~50 µs of GPU to read the
100 MB weight — this one the GPU actually spends time on, but by the time it finishes the host has moved
on to rotary and attention. Across the ~400 kernels of the token, the pattern is a sawtooth: short bursts
of GPU work separated by gaps where the device waits on the serial CPU dispatch stream. If the sum of the
gaps is comparable to or larger than the sum of the GPU-busy time, then the achieved bandwidth — which
only counts wall-clock — comes in at a fraction of peak, and that fraction is roughly the GPU's duty
cycle. My whole argument is that at batch 1, with kernels this small, that duty cycle is low. I can't
know the exact number until I run it, but the mechanism predicts "well under peak," and the achieved-GB/s
diagnostic will quantify exactly how far under.

I should also be clear-eyed about why the production frameworks the context lists — FasterTransformer,
DeepSpeed-Inference, vLLM — are fast, because it tells me which of their ideas are even applicable to my
constrained interface. They win largely by attacking this same launch overhead with hand-written fused
C++/CUDA kernels (one big kernel per layer instead of a dozen small ones), by managing the KV-cache in
optimized or paged layouts, and by batching many requests to raise intensity. But the batching win is
closed to me by the batch-1 rule, and I'm confined to the editable slots of a PyTorch scaffold — I'm not
going to hand-write fused CUDA. So the question that actually matters for the next rung is: can I get the
*effect* of fused kernels and a launch-overhead-free step, at batch 1, from within PyTorch? That's a
sharper question than "how do I make it fast," and I can only pose it because I first measured — or am
about to measure — that the binding constraint is host overhead, not the weight read. Establishing that
ordering of constraints is the entire payoff of running the honest eager baseline.

There is a second design decision buried in this baseline that I want to make deliberately rather than by
default: the KV-cache. The scaffold leaves the cache layout as a TODO, and the naive thing — the thing
that requires no thought — is to grow it by concatenation: each step, take the new token's key and value
and `cat` them onto the running key and value tensors, so attention sees the whole history. I'll keep
exactly that naive form for this baseline, and I want to be honest that it's a *choice* with costs I'm
choosing to eat. Concatenating a new row every step means allocating a fresh, one-larger tensor each
iteration and copying the old contents into it — allocator traffic and memory copies that grow with the
sequence length. It also means the shapes the model sees change every single step: the attention matmul's
sequence dimension ticks up by one each time, so the tensor sizes are never stable across iterations. For
a pure eager baseline whose job is to expose overhead, that's fine — it's *more* overhead, which is
on-brand for a floor. But I'm noting the instability explicitly, because a growing, reallocating,
shape-shifting cache is exactly the kind of thing that would obstruct any scheme that wants the decode
step to look identical every iteration, and when I come to attack the launch overhead the cache layout is
going to be back on the table.

Why is *this* the right zero, as opposed to some slightly-optimized starting point? Because a baseline's
only job is to isolate one thing so the next rung's win reads cleanly. If I folded a fused attention
kernel or a static cache into the baseline, then when I later delete the host overhead I couldn't tell how
much of the speedup came from that versus from the cache versus from the kernel — the measurement would be
muddied. By running pure eager with the naive cache, I isolate exactly the per-token CPU launch overhead.
The first real rung's improvement then reads as precisely "what removing that overhead buys," with nothing
else moving. I won't quantize, won't compile, won't touch the cache layout — pure eager decode, the
scaffold's loop run as written.

The `decode_n_tokens` loop itself deserves a second look, because the overhead I'm blaming isn't only in
the transformer forward — some of it is in the Python driving the forward. Each iteration calls
`decode_one_token`, then does `input_pos += 1`, appends `next_token.clone()` and `next_prob.clone()` to
Python lists, invokes the callback, and reassigns `cur_token = next_token.clone()`. The `.clone()` calls
are small device allocations and copies per step, the list appends are Python-object churn, and the whole
thing is an interpreted `for` loop that has to reach the C++/CUDA boundary afresh every token. None of
this is GPU work; it is CPU work interleaved with the kernel dispatch, and at ~400 kernels plus a handful
of Python-level operations per token it all lands on the same host thread that the GPU is waiting on. So
the eager overhead has two sources stacked on top of each other — the per-op dispatch inside the forward,
and the per-token Python bookkeeping around it — and both scale with the number of decode steps, which is
why a 200-token generation makes the effect so visible. I'm keeping every line of this loop exactly as
written for the baseline; I just want to have named all the overhead now so that when a later rung
collapses it, I know precisely what got collapsed.

Let me also sanity-check the sampling path, since it sits on the decode step and I don't want a hidden
synchronization polluting the measurement. The sampler uses the exponential/argmax trick —
`multinomial_sample_one_no_sync` draws `q ~ Exponential(1)` and takes `argmax(probs / q)` — specifically
to avoid a `torch.multinomial` call that would force a CPU-GPU sync every token. That's the correct shape
for a throughput loop: a sync would stall the host on the device and vice versa, and here every stall is
visible. I'll keep it. The `logits_to_probs` does the temperature scale and top-k mask (temperature 0.8,
top-k 200 per the settings), and `sample` slices the last position's logits and calls it. None of this is
where the time goes — it's a handful of small kernels on a 32000-vector — but I want it clean so the
number I get reflects the transformer forward, not a sampler artifact.

One more unit check before I commit, because I've been throwing around GB and TB/s and I want the
diagnostic formula to be dimensionally right. The context defines achieved bandwidth as
`model_size × tokens_per_sec / 1e9`, i.e. GB × (tokens/s) → GB/s, treating one token as "one full pass
over the model_size bytes." That's the correct accounting for a bandwidth-bound decode where each token
reads the whole weight set once: bytes-per-token times tokens-per-second is bytes-per-second. So if I
later see this number climb toward 2000 GB/s, the device is nearing its HBM ceiling; if it sits at a few
hundred, the device is idle. That single derived number will tell me, unambiguously, whether the binding
constraint on any given rung is host overhead (achieved ≪ peak) or memory bandwidth (achieved ≈ peak),
and it's the compass I'll steer the whole ladder by.

One nuance in the forward I want to make sure the baseline handles honestly is the rotary embedding and
the attention mask, since they're indexed by `input_pos`. In the growing-cache eager version the mask is
whatever causal mask the attention builds for the current sequence length, and the rotary table is either
recomputed or sliced per step — both are cheap relative to the weight read, but both are extra small
kernels that add to the launch count, and both are shape-dependent on the ever-growing sequence length.
That's consistent with everything else about this baseline: correct, simple, and maximally exposed to the
per-op overhead. I'm not going to optimize the rotary or the mask here; I only note that they're part of
the ~400-kernel budget and part of the shape instability, so they too will be in scope the moment I try
to make the step look identical across iterations.

I also want to fix in my head what "correct" means for this rung so I have a reference the later,
numerics-changing rungs can be checked against. The baseline runs the model in its native bf16 with no
approximation anywhere — the weights are the model's own, the attention is exact, the sampler draws from
the true temperature-0.8, top-k-200 distribution. So whatever token stream and whatever downstream
accuracy this produces *is* the reference behavior; there's nothing to validate because nothing has been
approximated. That matters later: the evaluation settings say scheduling-only and provably
distribution-preserving changes need no accuracy check, while anything that alters the numerics has to be
validated on hellaswag/winogrande. The baseline is the trivial case of the former — it changes nothing —
and it establishes the quality bar (the model's own outputs) that every faster rung will either match
exactly or be measured against.

A word on how I'll measure, because a bandwidth-bound micro-benchmark is easy to get wrong. I generate
200 new tokens from the fixed 5-token prompt, and I discard the first run entirely as a warmup — the very
first pass pays one-time costs (allocator growth to steady-state, any lazy initialization, the caches
filling) that don't represent steady decode, and folding them in would understate the true throughput. I
then take the mean tokens/s over several runs so a single noisy run doesn't set the number. Because eager
mode has no compilation step there's no compile latency to amortize here — that's a subtlety that will
matter for the *next* rung, not this one — so for the baseline the warmup is really just about reaching a
steady allocator and cache state. And I report both the headline tokens/s and the derived achieved
bandwidth, because the pair is what's diagnostic: tokens/s alone tells me "slow," but the achieved
bandwidth relative to 2 TB/s tells me *why* — starved versus bandwidth-bound — and only the "why" tells me
what to fix first.

So the configuration to start from is settled, and I'm confident it's the *right* zero: pure eager, naive
growing cache, no compilation, no quantization. What I'm watching is two numbers. The headline is
tokens/s. The diagnostic is the bandwidth this run *achieves* — the rate at which it is actually streaming
the 13.5 GB of weights from HBM. My expectation, from the launch-overhead argument above and the roofline
that says the arithmetic is ~150× cheaper than the memory traffic, is that tokens/s comes in low and the
achieved bandwidth comes in at a small fraction of the ~2 TB/s ceiling — that eager decode at batch 1
leaves the device mostly waiting on the host rather than streaming weights flat-out. If that's what I see,
then the first thing worth doing is not touching the arithmetic or the bytes at all, but making the GPU
run continuously instead of starved — closing the gap between achieved bandwidth and peak before I do
anything about the weight read itself. The default fill is in the answer; it is the scaffold's decode
loop, run as written.
