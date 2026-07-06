104.9 tokens/s, and the achieved bandwidth is 1397 GB/s — about 70% of the A100's ~2 TB/s. That second
number is the one I keep staring at, because it tells me the problem has fundamentally changed since the
eager baseline. Back then the GPU was idle at ~340 GB/s and the win was deleting host overhead. Now the
GPU is busy: it is streaming weights from HBM at a serious fraction of peak, and the tokens/s is tracking
the bandwidth almost linearly. Let me pin the arithmetic. At 104.9 tok/s a token takes `1/104.9 ≈ 9.5 ms`,
and reading 13.5 GB at the achieved 1397 GB/s takes `13.5/1397 ≈ 9.66 ms` — those two numbers agree, which
is the direct confirmation that the token time is now essentially *the weight read* and nothing else. The
32 ms of host overhead is gone. What's left between me and the ceiling is the gap from 1397 to 2000 GB/s,
a factor of `2000/1397 ≈ 1.43×`, and even if I hit the ceiling perfectly I top out at `148` tok/s and
stop. Scheduling tricks won't get me past that wall — I've basically already paid for them. The wall *is*
the wall: to generate one token I read all 13.5 GB of bf16 weights once, and at 2 TB/s that read takes
~6.75 ms no matter how cleverly I schedule it.

So the lever has to move. I can't make the bytes arrive faster than HBM allows; I can only make there be
*fewer bytes*. Tokens/s in this regime is essentially `bandwidth / bytes_per_token`, and bytes_per_token
is dominated by the weight read — the activations at batch 1 are a `1 × 4096` vector, kilobytes, utterly
negligible against gigabytes of weights, and the KV-cache reads are a few hundred MB, small. So the token
time is set by the weight bytes, and if I halve the weight bytes I should, to first order, roughly double
the tokens/s — for free on the bandwidth account, because the matmuls were never the limiter and have
arithmetic to spare (I'm 150× to the left of the roofline ridge, so I can afford to *add* arithmetic if it
buys fewer bytes). The weights are stored in bf16, 2 bytes each. The obvious move is to store them in
fewer bits.

Let me walk the representation choices deliberately rather than jumping at the first idea, because "fewer
bits" has several flavors and they trade off differently. There's fp8, int8, or lower; there's weight-only
versus also quantizing activations; and there's the granularity of the scale — per-tensor, per-channel,
per-group. Take the activation question first, because it's the one that decides the *character* of the
win. I could quantize both weights and activations to int8 and run genuine int8×int8 tensor-core matmuls
(the classic "dynamic quantization" path). That would speed up the arithmetic — but the arithmetic isn't
my bottleneck, it's already ~150× cheaper than the memory, so accelerating it buys me nothing here, and I'd
pay for it with the harder numerics of quantizing dynamic-range activations (activations have outliers,
per-token dynamic ranges, and quantizing them well is genuinely delicate). The thing crossing the
bottleneck is the *weight bytes*, not the activation arithmetic. So the right move is **weight-only**:
quantize only the weights, for storage, and leave the activations and the matmul in bf16. That sidesteps
the activation-outlier problem entirely and targets exactly the resource that's scarce.

Now the width. fp8 and int8 are both one byte, a clean halving of the 2-byte bf16 footprint. int8 is the
natural first stop: integers are a representation the hardware and the compiler both handle well, the
dequant is a single multiply, and 256 levels is a lot of resolution for a distribution as well-behaved as
a trained weight matrix's. Going below 8 bits is possible but riskier, and I want the *safe* halving first
— establish that halving the bytes buys the throughput I expect, at no quality cost, before I gamble on
anything more aggressive. So int8 weight-only it is, and the whole question collapses to one thing: can I
drop a linear layer's weights to int8 *without observably changing the model's outputs*? Because the point
of this rung is throughput at no quality cost — this isn't training, the weights are frozen, and I am not
allowed to make the model dumber to make it faster.

That leaves the granularity of the scale, which is where the quality is won or lost. Here is why int8
weight-only is safe *if I choose the granularity right*, and why a naive per-tensor cast would not be. A
weight matrix's entries, within a single output channel (a single row, mapping the whole input to one
output feature), live in a fairly narrow numerical range. If I find, *per row*, the maximum absolute
value, I can pick a scale that maps that row's range onto the int8 range [−128, 127], round each weight to
the nearest integer, and store the int8 weights plus one fp scale per row. At compute time I cast the int8
weights back up to bf16 and multiply, then multiply the result by the per-row scale to undo the
normalization. Let me bound the error. For a row with abs-max `m`, the scale is `s = m/127`, so the
rounding step is `s`, and round-to-nearest gives a per-weight error of at most `s/2 = m/254`. Relative to
the row's own scale that's under half a percent of full-scale per weight. And crucially those errors are
roughly independent across the 4096 input dimensions of the dot product, so in the output `y_j = Σ_i W_ji
x_i` the quantization noise is a sum of 4096 roughly-independent small terms — it grows like `√4096 ≈ 64`
in magnitude while the signal grows like the coherent sum, so the *relative* output perturbation is far
smaller than the per-weight `1/254`. The rounding noise that survives averages out. This is why per-row
symmetric int8 is essentially lossless in practice.

Contrast that with the granularity I'm rejecting. A single per-*tensor* scale would have to accommodate
the largest-magnitude row across the *whole* matrix. If one row has entries ten times larger than a quiet
row — which is common — then the global scale is set by the loud row, and the quiet row's weights, when
divided by that big scale, land in just a handful of the 256 levels. The quiet row gets quantized to near
nothing, its structure destroyed. Per-*channel* (per-row) granularity is precisely the fix: each row gets
a scale sized to *its own* abs-max, so a row with small weights gets a small step and a row with large
weights gets a large one, and no row is crushed by another row's outliers. That's the design choice that
makes the halving lossless, and it costs almost nothing in storage: one bf16 scale per row is 4096 scales
for a `4096 × 4096` matrix, i.e. `4096 × 2 B = 8 KB` against `4096 × 4096 × 1 B = 16 MB` of int8 weights —
0.05% overhead. Essentially free.

One more choice to defend: symmetric versus asymmetric (affine, with a zero-point). Asymmetric quantization
keeps both a scale and an offset per row, mapping `[min, max]` rather than `[−|max|, |max|]` onto the
integer grid, which recovers the level that symmetric "wastes" when a distribution isn't centered at zero.
For int8, with 256 levels to spend, that recovered fraction is a rounding detail — a centered-ish weight
row loses essentially nothing by assuming zero maps to zero, and in exchange the dequant is a single
multiply with no add, and there's no zero-point tensor to store or stream. The trade only tips toward
asymmetric when levels are *scarce*, because then the one wasted level and the centering assumption cost a
larger relative slice of the available resolution. At 256 levels they don't, so symmetric is the right,
cheaper call here. For now: symmetric, per output channel — and I note only that the level-budget is what
makes this call easy, so it's the first thing I'd re-examine if the resolution ever got tight.

So I quantize symmetrically (zero stays zero — no zero-point, which keeps the dequant a single multiply
and is the right choice for weight distributions that are roughly centered) and per output channel. The
per-channel quantizer: take each row's absolute max, set the scale so that max lands at the top of the
int8 range, round, clamp:

```python
def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # symmetric, per-row (axis 0)
    eps = torch.finfo(torch.float32).eps
    min_val, max_val = torch.aminmax(x, dim=1)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)   # map |max| -> 127
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=x.device)
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    quant = torch.clamp(x_round, quant_min, quant_max).to(target_dtype)
    return quant, scales, zero_points
```

Let me sanity-check the scale formula on the boundary, because an off-by-one here quietly clips or wastes
a level. `quant_max - quant_min = 127 - (-128) = 255`, and `255/2 = 127.5`, so `scales = |max| / 127.5`.
Then the row's abs-max weight, divided by that scale, gives `|max| / (|max|/127.5) = 127.5`, which rounds
to 128 and clamps to 127 — so the largest weight lands exactly at the top of the int8 range, using the
full [−128, 127] span symmetrically without overflow. The formula is right: it maps the row's dynamic
range onto all 256 levels, which is what maximizes resolution. The `torch.aminmax` with the neg/pos
folding computes the symmetric abs-max robustly (handling all-positive or all-negative rows), and the
`clamp(min=eps)` guards a degenerate all-zero row from a divide-by-zero.

Let me make the error argument concrete with a plausible row rather than leaving it as an inequality,
because a worked number is what convinces me the "lossless" claim isn't wishful. Trained transformer
weights in a `4096`-wide row typically have a standard deviation on the order of `0.02` and an abs-max a
handful of sigma out — say `m ≈ 0.1`. Then the scale is `s = 0.1/127 ≈ 7.9e-4`, the rounding step, and the
worst-case per-weight error is `s/2 ≈ 3.9e-4`. A typical weight of magnitude `0.02` carries a relative
error of `~2%` at worst, but that's the *worst* weight; averaged, the RMS rounding error is `s/√12 ≈
2.3e-4`. Now propagate it to the output: `y_j = Σ_i W_ji x_i` over 4096 terms, and the error contribution
is `Σ_i (ΔW_ji) x_i` with `ΔW_ji` roughly independent zero-mean of RMS `2.3e-4`. Its standard deviation is
`2.3e-4 × ||x|| ≈ 2.3e-4 × √4096 × σ_x = 2.3e-4 × 64 × σ_x ≈ 0.015 σ_x`. Meanwhile the signal `y_j` has
magnitude on the order of `||W_row|| × σ_x ≈ √4096 × 0.02 × σ_x ≈ 1.3 σ_x`. So the output SNR is roughly
`1.3 / 0.015 ≈ 85`, i.e. the quantization noise is around 1% of the output signal per layer. That's small,
and it's the kind of perturbation the residual stream and the softmax tolerate without changing which
token wins — which is why I expect no observable harness movement. If the real weights had far heavier
tails than my `0.1` abs-max assumption, the per-row scale would inflate and the SNR would degrade, and
*that's* the scenario the accuracy check exists to catch; my arithmetic says it won't happen at int8, but
the harness is the arbiter.

That quantizer runs once, offline, over every `nn.Linear` weight to produce an int8 state dict plus its
scales — the QKV projection, the output projection, the three feed-forward matmuls, the LM head. These are
the big matrices; quantizing them is where the 13.5 GB lives (they're ~96% of the parameters). I leave the
embeddings and RMSNorm weights alone: the embedding is a lookup, not a matmul on the hot path, and the
norm weights are tiny and numerically sensitive — quantizing either would risk quality for no bandwidth
gain, since neither is a big weight read on the decode path.

The runtime side is a drop-in linear that stores int8 weights and the per-row scales and applies the scale
after the matmul:

```python
class WeightOnlyInt8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input):
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales
```

and a handler that swaps every `nn.Linear` for this and builds the quantized weights:

```python
class WeightOnlyInt8QuantHandler:
    def __init__(self, mod): self.mod = mod
    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(), -128, 127, torch.int8)
                cur_state_dict[f"{fqn}.weight"] = int8_weight
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)
        return cur_state_dict
    def convert_for_runtime(self):
        replace_linear_weight_only_int8_per_channel(self.mod)
        return self.mod
```

One subtlety the forward reveals, and it's the crux of why this is the right kind of quantization for this
problem: the dequant multiply by `scales` happens *after* `F.linear`, and the int8 weights are cast up to
the activation dtype (`self.weight.to(dtype=input.dtype)`) just *before* the matmul. So I'm not, in this
version, doing int8×int8 tensor-core math — the multiply is bf16×bf16 after an upcast. The *win is purely
the memory read*. The weights live in HBM as 1 byte each, get streamed at half the byte count, and are
widened to bf16 inside the kernel for the multiply. That's exactly right for a bandwidth-bound problem: I
don't need int8 *arithmetic*, I need int8 *storage*, because storage is what crosses the bottleneck. And I
can lean on the compiler here — since the decode step is already compiled, the upcast and the epilogue
scale multiply fuse into the matmul kernel, so the dequant costs no extra kernel launches and the extra
elementwise ops are hidden behind the (bandwidth-bound) matmul. The mathematical order matters too:
`(W_int8 · x) · s_row` equals `(W_int8 · s_row) · x` because the scale is per output row and factors out of
the dot product, so applying it in the epilogue is exact, not an approximation.

Let me also account for the *actual* per-token bytes after quantizing, because that's what sets the
throughput and it isn't a clean 6.75 GB. The linear weights are ~6.47B of the ~6.74B parameters; at 1 byte
each that's ~6.47 GB of int8, plus the per-row scales (a few MB, negligible), plus the still-bf16
embeddings, output head, and norms at ~0.27B × 2 B ≈ 0.54 GB. So the effective model size streamed per
token is roughly `6.47 + 0.54 ≈ 7.0 GB`, not 6.75 — the un-quantized tail matters. That's why I say the
halving is "clean" only for the bulk; the real ratio of bf16-model to int8-model bytes is `13.5 / 7.0 ≈
1.93×`, so even a *perfectly* bandwidth-efficient int8 read would give ~1.93×, not 2×, and real
inefficiency pulls it further down. This arithmetic is why I'm predicting ~1.5× rather than a clean double,
and it also tells me where a future rung would have to look: the un-quantized 0.54 GB tail becomes a
larger *fraction* of the token as I shrink the quantized bulk, so at ever-lower bit-widths it starts to
matter, though at int8 it's still small.

I briefly consider whether I should also quantize the KV-cache to int8 while I'm here, since it's another
thing crossing HBM on the decode path. I decide against it for this rung. The cache read at these sequence
lengths is a few hundred MB per token — an order of magnitude smaller than the ~7 GB weight read — so
halving it buys little throughput, while the K/V values feed directly into the attention softmax where
precision loss can distort the attention distribution in ways that are harder to bound than weight
rounding. The cost/benefit is upside-down compared to weights: weights are the big read and are
quality-robust; the cache is the small read and is quality-sensitive. So I leave the cache in bf16 and
spend my quantization budget entirely on the weights, which is where the bytes and the safety both are.

The `replace_linear_weight_only_int8_per_channel` swap is a recursive walk over `named_children`, replacing
each `nn.Linear` in place with a `WeightOnlyInt8Linear` of matching `in/out_features` and recursing into
non-Linear submodules. Doing it structurally like this — rather than by name-matching — means it catches
every linear in the 32 blocks plus the head uniformly, and it composes with the compile+cache rung
unchanged: the swapped module is still a plain `nn.Module` whose `forward` the compiler can trace and
capture into the same CUDA graph, so I keep the entire scheduling win from the previous rung and stack the
byte reduction on top. Nothing about the graph discipline breaks — the int8 buffer and its scales are
fixed-shape, fixed-address, exactly what replay needs.

The prediction, against 104.9 tok/s at 1397 GB/s. Halving the weight bytes should let the same bandwidth
carry roughly twice the tokens, so a naive read says ~2×, into the 200s. I expect *less* than that — call
it ~1.5×, into the 150s — and the reason is instructive and worth stating as a falsifiable claim: the
achieved bandwidth will actually *drop*, not stay pinned at 1397. Reading half-size int8 weights is a
somewhat less bandwidth-efficient access pattern than reading bf16 (the upcast and dequant add per-byte
work, and the effective per-token model size is now ~7 GB rather than a clean 6.75), so I won't realize the
full ideal doubling. The right way to read the result is the bandwidth-bound signature: *tokens/s up
because bytes/token down, even though GB/s achieved goes down*. If instead the achieved GB/s stayed near
1397 and tokens/s doubled cleanly, I'd be surprised and would suspect a measurement artifact; the honest
expectation is the sub-2× gain with falling GB/s. And the guardrail is the accuracy harness: per-row
symmetric int8 should show no observable degradation on hellaswag/winogrande, and if it does, the
per-channel argument was wrong — the likely culprit would be a per-tensor scale sneaking in, or a layer
whose row distribution is nastier than I assumed — and I'd back it out. My bet, from the `1/254`
error-per-weight bound averaging over 4096-wide dot products, is no observable quality loss. The change is
the per-channel int8 quantizer, the `WeightOnlyInt8Linear`, and the handler that swaps the model's linears;
full scaffold code in the answer.
