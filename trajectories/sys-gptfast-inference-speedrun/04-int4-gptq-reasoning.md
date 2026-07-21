155.58 tokens/s at 1069 GB/s. The int8 bet paid off exactly as the bandwidth story predicted: I halved the
weight bytes and throughput rose `155.58/104.9 ≈ 1.48×`, with achieved GB/s dropping from 1397 to 1069
rather than rising. Reading 1069 back to a model size, `1069 / 155.58 ≈ 6.87 GB` per token, matches the ~7 GB
int8 model (6.47 GB weights plus the still-bf16 tail). So the mechanism is confirmed — throughput is
`achieved_bandwidth / bytes_per_token` and the win came entirely from the denominator — and the lever is
proven. The obvious next pull: if 8 bits bought 1.48×, what about 4? int4 halves the weight footprint again,
~6.5 GB of int8 weights toward ~3.3 GB, and on a bandwidth-bound workload that's another `~1.3-1.5×`. The
arithmetic budget is nowhere near binding (still ~150× left of the ridge, and the matmul stays bf16 after
upcast), so the only question, again, is quality: can I drop the weights to 4 bits without the model getting
observably worse?

Here the trick that made int8 safe stops being enough, and I should quantify why. int8 had 256 levels: the
rounding step was `m/127`, the per-weight error `~m/254`, propagating to ~1% output perturbation — a rounding
error. At 4 bits I have *16* levels. A single scale per row now covers the whole row's range with sixteen
rungs, so the step jumps to ~`m/8` — roughly 16× coarser than int8. Redo the SNR estimate with a 16× larger
step and the ~1% output noise becomes ~16%, and 16% per layer compounded over 32 layers isn't a rounding
error, it's the model getting observably dumber. And it's worse than uniform: a 4096-wide row isn't all in
one band — there are local clusters and occasional large entries, so sixteen levels stretched across a range
set by the largest entry leaves the typical weight only a few usable levels. Round-to-nearest at 4 bits, per
row, would show on the harness. Two things have to change, attacking two different parts of the problem.

First, finer scale granularity — the asymmetric-plus-finer note I put in my back pocket at int8, now cashed
in. Instead of one scale per *row*, quantize in small *groups* of consecutive weights along the input
dimension, group size `G`, each group getting its own scale and zero-point. With `G=32`, every block of 32
weights is fit to the 4-bit grid with a scale and offset tuned to *that block's* local range — a quiet block
gets a fine step, a loud block a coarse one, and no block is forced onto a grid sized for an outlier four
thousand columns away. This is affine now, not symmetric: at 16 levels the one level symmetric wastes and the
centering assumption cost a real slice of resolution when a group isn't centered, so I keep a min and a scale
per group. Pricing the overhead, because "finer" isn't free: each group of 32 weights is `16 B` packed int4
plus `2 × 2 = 4 B` of scale+zero metadata, i.e. `4/32 = 0.125 B` per weight on top of the `0.5 B` packed —
`0.625 B/weight`, a 25% overhead on the raw int4 storage. Contrast `G=128`: `4 B` per 128 weights is `0.031
B/weight`, cheaper metadata but four times coarser grouping, exactly the resolution I can't spare at 4 bits.
`G=32` is the aggressive choice — a few percent more bytes to buy back the resolution 4 bits threw away. The
group quantizer maps each group's `[min, max]` onto the 16 integer levels `0..15` (`scales = (max−min)/15`,
zero-point at grid-center), so both endpoints land exactly on grid and all 16 levels are used within each
block; code in the answer.

Finer groups shrink the step, but 16 levels is still 16 levels, and even with per-group scales the SNR is a
few times worse than int8 — the residual error summed over 32 layers still drifts the outputs. The second
change is the one that matters, and it comes from asking the right question. Round-to-nearest minimizes the
error of *each weight in isolation*. But I don't care about weights in isolation — I care about the layer's
*output*, `y = Wx`. The thing to minimize is `E||Wx − Ŵx||²` over the real input distribution, and RTN is
the wrong objective for it: when two input features are correlated, an over-estimate on one weight can be
compensated by an under-estimate on the correlated one, and RTN, blind to input statistics, throws that
cancellation away. Expand the output error: `E[||(W−Ŵ)x||²] = tr((W−Ŵ) H (W−Ŵ)ᵀ)` with `H = E[x xᵀ]`, the
second-moment of the inputs, estimated by running a calibration set (a couple thousand short wikitext
sequences) through the layer and accumulating `x xᵀ`. For the quadratic `||Wx−Ŵx||²` the second derivative in
the weights *is* `2H`, so `H` isn't an approximation of curvature, it *is* the curvature. The quantization
error is now a quadratic form weighted by `H`: directions the inputs excite strongly are expensive, ones they
rarely excite are cheap, and I want to be greedy about it.

This is GPTQ. Quantize the columns of `W` one at a time in order; when column `i` incurs error `(w_i − ŵ_i)`,
push it forward onto the not-yet-quantized columns so the layer's output is corrected for what I just did.
The exact push is dictated by `H` — minimizing the quadratic form with column `i` fixed to its quantized
value has a closed form that distributes the residual along inverse-Hessian-weighted directions, with the
per-column denominator the diagonal of the inverse Hessian. So per column: quantize, compute local error
`err = (w − dq) / d` with `d` the inverse-Hessian diagonal entry, and subtract a rank-1 update `err ⊗
Hinv[i, i:]` from the remaining columns. Dividing by `d` is what makes the update the *optimal* correction
rather than a heuristic; the index only ever moves forward and the update touches only columns `≥ i`, a clean
triangular sweep. Working with the *Cholesky* of the inverse Hessian makes that sweep stable and ordered
instead of a fresh linear solve per column — `O(columns³)` once rather than `O(columns⁴)` — and a small
`percdamp · mean(diag H)` on the diagonal keeps `H` invertible when a dead column the calibration never
excited would otherwise make it singular. Full runner in the answer.

Make the "correlated columns cancel" claim concrete on a two-column toy, because it's the whole
justification for GPTQ over RTN. Take `H = [[1, 0.9], [0.9, 1]]` (unit variance, correlation 0.9) and a
one-output row `w = [0.10, 0.10]` on a crude grid where rounding pushes column 0 up by 0.04. RTN would round
column 1 independently and eat both errors. GPTQ quantizes column 0, sees the +0.04 output-side overshoot,
and because column 1 is 0.9-correlated pushes a compensating adjustment onto `w_1 ≈ 0.10 − 0.9·0.04 ≈ 0.064`,
which rounds to a grid point that *cancels* most of column 0's output error along the `[1,1]` direction `H`
weights heavily. The net `tr((W−Ŵ)H(W−Ŵ)ᵀ)` comes out markedly smaller than RTN's. On an *uncorrelated* `H =
I` the off-diagonal is zero, the push does nothing, and GPTQ degenerates to RTN — the correct behavior: no
correlation, no free cancellation. That degeneracy is reassuring, because it means GPTQ can only help or tie
under this objective, never hurt.

Should I be greedier still — int3, int2? Two forces say stop at 4. The metadata grows as a *fraction* of the
shrinking payload: at G=32 the `4 B/group` is `0.125 B/weight`, 25% on int4's `0.5 B` but 33% on int3 and
50% on int2, so the effective bytes stop halving cleanly and I'd need coarser groups, fighting the resolution
I need even more desperately at 8 or 4 levels. And the quality cliff: my SNR estimate degrades ~2× per bit,
and GPTQ's error-feedback has a floor — it redistributes error but can't create resolution that isn't there —
so below 4 bits the "minimal" loss I'm betting on becomes a real harness-visible regression. So 4 bits is the
aggressive-but-safe stopping point for the *representation* lever: the last width where grouping-plus-GPTQ
can plausibly hold quality and the savings still beat the metadata. Past this, the byte-per-weight lever is
nearly exhausted and I'll need a different axis. The two ideas divide labor cleanly — grouping sets the *grid*
finely enough that 4 bits has real local resolution, GPTQ chooses *which way to round onto that grid* so the
surviving error lands where the inputs least excite. That's how 4 bits goes from "noticeably degraded" to
"minimal" loss. The calibration is offline and one-time — accumulate `H` per linear over a wikitext set, run
the sweep per layer — producing a static checkpoint at no decode-time cost.

The runtime linear stores the int4-packed weights and per-group scales-and-zeros and calls
`torch.ops.aten._weight_int4pack_mm`, which reads the 4-bit-packed weights from HBM, dequants on the fly with
the group qparams, and does the matmul — again the win is the *read*: 4-bit storage streamed across the
bottleneck, widened inside the kernel. The `padding` handles the kernel's shape constraint (`in_features` a
multiple of the group size and tiling): a linear whose `in_features` isn't a clean multiple gets its input
zero-padded to the next valid size, and the padded columns contribute nothing, so it's correctness-preserving
with a little wasted read on odd-shaped layers. The weights aren't a naive int4 array either — they're
pre-packed into the tiled layout `_weight_int4pack_mm` expects (`inner_k_tiles=8`) so the kernel reads
contiguous coalesced blocks rather than scatter-gathering, which is what lets the int4 path approach its
byte-count advantage. And it composes with the compiled decode graph as int8 did: `WeightOnlyInt4Linear` is a
plain module with fixed-shape `weight` and `scales_and_zeros` buffers, so it captures into the same CUDA
graph and keeps the scheduling win stacked underneath.

The prediction, against 155.58 tok/s at 1069 GB/s. Size the int4 model: ~6.47B linear params at `0.625
B/weight` is `~4.0 GB`, plus the ~0.5 GB bf16 tail, so ~4.5 GB streamed per token versus the int8 model's
~6.87 GB — a byte ratio `6.87/4.5 ≈ 1.53×`. Real bandwidth inefficiency at 4-bit packing pulls the realized
gain below that, so I expect roughly `~1.3×`, with achieved GB/s dropping again (same signature: fewer
bytes/token, lower absolute GB/s, and the un-quantized tail is now a bigger *fraction* of the shrinking
token). The risk is entirely on quality, and it's realer than int8: 16 levels is genuinely coarse, my SNR
estimate says several times worse even with grouping, and if G=32 + GPTQ don't recover the output the harness
will show it. My bet is that H-weighted error-feedback plus G=32 grouping holds the degradation to *minimal*
— small but nonzero, distinct from int8's no-observable-loss, and acceptable for the throughput. If
hellaswag/winogrande drop more than marginally, the diagnosis is either grouping too coarse (smaller G, more
metadata) or calibration too thin (`H` from too few tokens, error-feedback aimed at the wrong directions),
and I'd revisit those knobs. The change is the group quantizer, the GPTQ `faster_quant` runner, and the
int4-packed runtime linear; full scaffold code in the answer.
