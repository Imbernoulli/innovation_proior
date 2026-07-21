104.9 tokens/s, and the achieved bandwidth is 1397 GB/s — about 70% of the A100's ~2 TB/s. That second
number tells me the problem has fundamentally changed. At the eager baseline the GPU was idle at ~340 GB/s
and the win was deleting host overhead. Now the GPU is busy streaming weights at a serious fraction of peak,
and tokens/s tracks the bandwidth almost linearly: a token takes `1/104.9 ≈ 9.5 ms`, and reading 13.5 GB at
1397 GB/s takes `13.5/1397 ≈ 9.66 ms` — the two agree, direct confirmation that the token time is now
essentially *the weight read* and nothing else. The 32 ms of host overhead is gone. What's left between me
and the ceiling is the gap from 1397 to 2000 GB/s, `~1.43×`, and even hitting the ceiling perfectly tops me
out at 148 tok/s. Scheduling tricks won't get past that wall — I've basically paid for them — because to
generate one token I read all 13.5 GB of bf16 weights once, ~6.75 ms at peak no matter how I schedule it.

So the lever has to move. I can't make the bytes arrive faster than HBM allows; I can only make there be
*fewer bytes*. Tokens/s in this regime is `bandwidth / bytes_per_token`, and bytes_per_token is dominated by
the weight read — the batch-1 activations are a `1 × 4096` vector, kilobytes, and the KV-cache reads are a
few hundred MB, both negligible against gigabytes of weights. So if I halve the weight bytes I should roughly
double tokens/s, for free on the bandwidth account: the matmuls were never the limiter and I'm 150× to the
left of the roofline ridge, so I can afford to *add* arithmetic if it buys fewer bytes. The weights are bf16,
2 bytes each; the move is to store them in fewer bits.

"Fewer bits" has flavors that trade off differently, and the activation question decides the character of
the win. I could quantize both weights and activations to int8 and run genuine int8×int8 tensor-core
matmuls, but the arithmetic isn't my bottleneck — accelerating it buys nothing here, and I'd pay with the
delicate numerics of quantizing dynamic-range activations, which have outliers and per-token ranges. The
thing crossing the bottleneck is the *weight bytes*, so the right move is **weight-only**: quantize the
weights for storage, leave activations and the matmul in bf16. That sidesteps the activation-outlier problem
entirely and targets exactly the scarce resource. On width, int8 is the natural first stop — one byte, a
clean halving of bf16, integers the hardware and compiler both handle well, dequant a single multiply, and
256 levels is ample resolution for a trained weight matrix. Going below 8 bits is possible but riskier, and
I want the *safe* halving first: establish that halving the bytes buys the throughput I expect at no quality
cost before gambling on anything more aggressive. This isn't training, the weights are frozen, and I'm not
allowed to make the model dumber to make it faster.

Granularity is where quality is won or lost, and it's why int8 weight-only is safe if I choose it right. A
weight matrix's entries within a single output channel — one row, mapping the whole input to one output
feature — live in a fairly narrow range. If I take that row's abs-max `m`, pick a scale `s = m/127` mapping
it onto [−128, 127], round, and store int8 weights plus one fp scale per row, the rounding step is `s` and
round-to-nearest gives per-weight error ≤ `s/2 = m/254`, under half a percent of full-scale. Crucially those
errors are roughly independent across the 4096 input dimensions of the dot product, so in `y_j = Σ_i W_ji
x_i` the noise is a sum of 4096 small terms growing like `√4096 ≈ 64` while the coherent signal grows
faster, and the *relative* output perturbation is far smaller than `1/254`. Put a concrete row on it, since a
worked number is what convinces me "lossless" isn't wishful. A 4096-wide row typically has σ ≈ 0.02 and
abs-max a few sigma out, say `m ≈ 0.1`; then `s ≈ 7.9e-4`, RMS rounding error `s/√12 ≈ 2.3e-4`, and its
output contribution has std `2.3e-4 × √4096 × σ_x = 2.3e-4 × 64 × σ_x ≈ 0.015 σ_x`, against a signal `y_j`
of magnitude `√4096 × 0.02 × σ_x ≈ 1.3 σ_x`. That's an output SNR of `~85`, quantization noise ~1% of the
signal per layer — the kind of perturbation the residual stream and softmax tolerate without changing which
token wins. If real weights had far heavier tails than my `0.1` assumption the per-row scale would inflate
and the SNR degrade, and *that's* the scenario the accuracy check exists to catch; my arithmetic says it
won't happen at int8, but the harness is the arbiter.

The contrast is what makes the choice. A single per-*tensor* scale would have to accommodate the largest row
across the whole matrix; if one row's entries are ten times a quiet row's — common — the global scale is set
by the loud row and the quiet row lands in a handful of the 256 levels, its structure destroyed. Per-channel
granularity is the fix: each row gets a scale sized to its own abs-max, so no row is crushed by another's
outliers. And it costs almost nothing — one bf16 scale per row is `4096 × 2 B = 8 KB` against `16 MB` of
int8 weights for a `4096 × 4096` matrix, 0.05% overhead. On symmetric versus asymmetric: asymmetric keeps a
zero-point to map `[min, max]` rather than `[−|max|, |max|]`, recovering the level symmetric "wastes" when a
distribution isn't centered. With 256 levels to spend that recovered fraction is a rounding detail, and in
exchange symmetric makes dequant a single multiply with no zero-point tensor to store or stream. The trade
only tips toward asymmetric when levels are *scarce* — I put that in my back pocket for when the bit-width
gets tight — but at 256 levels symmetric is the cheaper, right call.

So the quantizer takes each row's abs-max, sets the scale so it lands at the top of the int8 range, rounds,
clamps. Checking the boundary — `quant_max − quant_min = 127 − (−128) = 255`, `255/2 = 127.5`, so `scales =
|max| / 127.5` and the row's abs-max divided by that gives `127.5`, rounding to 128 and clamping to 127, so
the largest weight lands exactly at the top of the range using the full span symmetrically without overflow.
The neg/pos folding computes the symmetric abs-max robustly for all-positive or all-negative rows, and
`clamp(min=eps)` guards a degenerate all-zero row from divide-by-zero. That quantizer runs once, offline,
over every `nn.Linear` — the QKV and output projections, the three feed-forward matmuls, the LM head, ~96% of
the parameters and where the 13.5 GB lives. I leave the embeddings and RMSNorm weights in bf16: the embedding
is a lookup not a hot-path matmul, and the norm weights are tiny and numerically sensitive — quantizing
either risks quality for no bandwidth gain.

The runtime is a drop-in linear storing int8 weights and per-row scales, whose forward casts the int8
weights up to the activation dtype *before* `F.linear` and applies the scale in the epilogue *after* — so I'm
not doing int8×int8 tensor-core math, the multiply is bf16×bf16 after an upcast. The *win is purely the
memory read*: the weights live in HBM as 1 byte each, stream at half the byte count, and widen to bf16 inside
the kernel. That's exactly right for a bandwidth-bound problem — I don't need int8 *arithmetic*, I need int8
*storage*, because storage is what crosses the bottleneck. Because the decode step is already compiled, the
upcast and epilogue scale fuse into the matmul kernel, costing no extra launches. And the order is exact, not
approximate: `(W_int8 · x) · s_row = (W_int8 · s_row) · x` because the per-row scale factors out of the dot
product. The swap is a recursive walk replacing every `nn.Linear` with the int8 module of matching
in/out_features; the swapped module is a plain `nn.Module` with fixed-shape int8 and scale buffers, so it
traces and captures into the same CUDA graph, keeping the entire scheduling win from the previous rung
stacked under the byte reduction. Full code in the answer.

I account for the *actual* per-token bytes, because it isn't a clean 6.75 GB. The linear weights are ~6.47B
of the ~6.74B params; at 1 byte each that's ~6.47 GB of int8, plus the still-bf16 embeddings, head, and norms
at ~0.27B × 2 B ≈ 0.54 GB. So the effective model streamed per token is ~7.0 GB, not 6.75 — the un-quantized
tail matters, and the real bf16-to-int8 byte ratio is `13.5 / 7.0 ≈ 1.93×`. Even a perfectly efficient int8
read gives ~1.93×, not 2×, and real inefficiency pulls it further — which is why I predict ~1.5× rather than
a clean double. (I briefly consider quantizing the KV-cache too, but decide against: it's a few-hundred-MB
read, an order of magnitude smaller than the ~7 GB weights, so halving it buys little, while K/V precision
loss feeds the attention softmax where it's harder to bound than weight rounding. The cost/benefit is
upside-down — weights are the big, quality-robust read; the cache is the small, quality-sensitive one.)

The prediction, against 104.9 tok/s at 1397 GB/s. Halving the weight bytes should carry roughly twice the
tokens, so ~2× naively; I expect *less*, ~1.5×, and the reason is worth stating as a falsifiable claim: the
achieved bandwidth will *drop*, not stay at 1397. Reading half-size int8 with an upcast-and-dequant is a less
bandwidth-efficient access pattern, and the effective model is ~7 GB not 6.75, so I won't realize the full
ideal doubling. The bandwidth-bound signature is *tokens/s up because bytes/token down, even though GB/s
achieved goes down*; if GB/s instead stayed near 1397 and tokens/s doubled cleanly I'd suspect a measurement
artifact. The guardrail is the accuracy harness: per-row symmetric int8 should show no observable
degradation on hellaswag/winogrande, and if it does the likely culprit is a per-tensor scale sneaking in or a
layer with a nastier row distribution than I assumed, and I'd back it out. My bet, from the `1/254`
error-per-weight bound averaging over 4096-wide dot products, is no observable quality loss. The change is the
per-channel int8 quantizer, the `WeightOnlyInt8Linear`, and the handler that swaps the model's linears; full
scaffold code in the answer.
