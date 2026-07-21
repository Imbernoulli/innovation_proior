SmoothQuant got me across the line into activation quantization: OPT-175B W8A8 back to 66.8% zero-shot,
a tenth of a point off the 66.9% FP16, by migrating the activation outliers into the weights offline.
The linear layers run as plain INT8 GEMMs. But INT8 is the comfortable case. The real prize is W4A4 —
4-bit weights *and* 4-bit activations, and 4-bit KV-cache — because that is what halves the memory and
doubles the integer throughput again over INT8, and it is the KV-cache quantization that unlocks long
contexts and large batches where the cache, not the weights, becomes the memory bottleneck. So I try
SmoothQuant's recipe at 4-bit activations, and it does not hold up. On Llama-2-7B, SmoothQuant-style
W4A4 lands at a WikiText perplexity around 83 — far from the FP16 5.47, an essentially unusable model.
The migration helped, but I predicted this ceiling at the last rung and here it is: the migrated
outliers are *smaller*, not *gone*. At INT8 the residual spikes fit inside the 256-level grid; at INT4,
with only sixteen levels, even the tamed 3× outlier blows the per-tensor activation scale (Δ set by 3×
means ordinary activations round to ~2 levels), and the bulk of the signal rounds to mush again. Offline
rescaling is the wrong instrument for 4-bit activations.

Let me look harder at *why* the outliers are so destructive, because every method so far has been
fighting the outliers' *magnitude* — clipping them, rescaling them, splitting the difficulty — and at
4-bit that is not enough. The deeper fact is geometric, and I only see it when I stop thinking about
sizes and start thinking about *directions*. The activation outliers live in a few specific, fixed
coordinate directions of the hidden state — the same handful of feature channels, over and over. The
quantization grid is axis-aligned: it puts a scalar step Δ on each coordinate independently, so the
representable points form a regular lattice whose axes are the standard basis. The problem is a collision
between a *signal* whose energy is concentrated along a few coordinate axes and a *quantizer* whose
resolution is allocated per coordinate axis. Every method so far has accepted that the outlier lives on
axis 137 (say) and tried to make axis 137 cheaper to represent. But the axes are not sacred. What if I
could *rotate the coordinate frame* so that the energy piled into a few axis-aligned spikes is spread
evenly across all the coordinates?

Before I chase the rotation idea, let me make sure I am not overlooking a cheaper fix for W4A4, because
rotation is a big structural change. One option is to push SmoothQuant harder — a larger α to migrate more
outlier mass onto the weights. But I already know the ceiling: at α → 1 the activations flatten but the
weights overload, and at 4-bit the weights are *not* the free operand they were at 8-bit (16 levels, not
256), so I would just be trading a broken-activation model for a broken-weight one. The split has no sweet
spot at 4-bit because both sides are tight. A second option is finer activation grouping — but the useful
granularity is per-*channel* along C_in, and that is the contraction axis the INT8/INT4 tensor core
forbids as an epilogue, the exact wall I hit at the last rung; group-along-C_in activation scales cannot
run on the fast kernel. A third is a per-channel affine grid with a learned clip — but clipping a 3×
residual outlier at 4-bit still leaves the ordinary signal with ~5 levels, not enough. All three are
magnitude-domain fixes, and the magnitude domain is exhausted: at 16 levels there is no rescaling that
gives the bulk enough resolution while a spike, however tamed, still sits in the same axis-aligned grid.
That exhaustion is what forces me out of the magnitude domain and into the *geometry*.

This is the move, and I want to quantify it before I trust it. Take a hidden vector with a spike: one
coordinate at magnitude M = 100, the rest near 1, in dimension n = 4096. Multiply by a random orthogonal
matrix Q. A single basis vector e_i, rotated, becomes a column of Q — a random unit vector uniformly
distributed on the sphere in n dimensions, so each of its coordinates has magnitude on the order of
1/√n. The spike of size M therefore gets smeared into a contribution of about M/√n in *every*
coordinate: with n = 4096, √n = 64, so the 100× spike becomes 100/64 ≈ 1.56 in each coordinate — no
longer an outlier at all, just slightly above the bulk. The outlier has been *dissolved* into the bulk,
not clipped and not migrated, but smeared out so that no single coordinate is special. After the
rotation the per-coordinate distribution is near-Gaussian with no spikes, exactly the distribution a
per-tensor grid handles well. This is a statement about kurtosis: the pre-rotation activation has heavy
tails (high kurtosis, energy in a few coordinates), and a random rotation drives any distribution toward
Gaussian (kurtosis toward 3) by the same averaging that gives the central limit theorem. So rotation
does not shrink the outlier's energy — energy is conserved under an orthogonal map — it *redistributes*
it so the grid can spend resolution uniformly and waste none.

The objection is obvious and it is the whole reason this has not been the standard move: I cannot just
rotate the activations, because rotating the activations changes the function the network computes,
unless I undo the rotation somewhere. If I insert Q before a linear and do nothing else, the output is
wrong. So I need rotations that are *free* — that change the internal representation but leave the model's
output exactly unchanged. This is where I get to exploit a structural property of the Transformer I have
not used yet: **computational invariance**. The residual stream is acted on only by linear maps that read
from it and write to it, and an orthogonal Q can be pushed *through* a linear map by absorbing it into the
weights: a reader's input side takes Qᵀ, a writer's output side takes Q, and the two foldings meet in the
stream as inverses, so end to end the computation is identical — I have changed the *basis* the hidden
state is stored in, nothing else.

The one place this could break is normalization, and it is worth checking rather than assuming. LLaMA
uses RMSNorm, which divides the hidden vector by its own root-mean-square norm. An orthogonal Q preserves
norm exactly (‖Qh‖ = ‖h‖ because QᵀQ = I), so RMSNorm(Qh) = Qh/‖Qh‖ = Q·(h/‖h‖) = Q·RMSNorm(h) — the
rotation commutes cleanly through the normalization. There is a subtlety: RMSNorm is usually followed by
a per-channel learned scale γ, and diag(γ) does *not* commute with Q. The fix is to first *fuse* that
per-channel γ into the adjacent linear's weights, leaving the normalization a pure normalization (no
per-channel scale) through which Q passes freely. So the recipe has a preprocessing step — absorb the
RMSNorm scales into the following linears — and then the residual-stream rotation is exact.

Let me trace the invariance through one attention block by hand to be sure the two foldings really cancel
and I have the transposes in the right places, using the row-vector convention PyTorch actually uses: a
linear computes y = x Wᵀ for a row-vector activation x and stored weight W of shape [out, in]. I decide
the stream shall be stored rotated as x' = x Q. A *reader* (W_q, W_k, W_v) must still produce its
original output x Wᵀ; since x = x' Qᵀ, that output is x' Qᵀ Wᵀ = x' (W Q)ᵀ, so the folded reader weight is
W ← W Q. A *writer* (W_o) produces a contribution c = a W_oᵀ from its input a, and to land in the rotated
stream it must emit c Q = a W_oᵀ Q = a (Qᵀ W_o)ᵀ, so the folded writer weight is W ← Qᵀ W. Those are
the folded reader weight W ← W Q and writer weight W ← Qᵀ W (which are `rotate_attention_inputs` and
`rotate_attention_output` in the answer). Going around the block leaves the stream consistently in the
x' = xQ frame, the residual add is between two vectors both in that frame, and after the head un-rotates
(x' Qᵀ = x) the logits are identical to FP16 because QᵀQ = I — any single sign or transpose error would
move the FP16 output, so the FP16-match is the check.

The value/output path needs its own treatment because the outliers there
live in the per-head dimension, not the residual dimension, so a second Hadamard at head-dim granularity
is applied online between V and O; and the keys and queries get a Hadamard before the KV-cache is
quantized, so the cached K and V are stored in an outlier-free frame at 4 bits. These online Hadamards
are the ones whose cost the WHT keeps negligible — and I should size that cost to be sure. A dense
per-token rotation on the head dimension d_h = 128 would be 128² ≈ 16k multiply-adds per token per head;
the Walsh–Hadamard butterfly does it in 128·log₂128 = 128·7 ≈ 900 add/subtracts, an ~18× reduction, and
against the thousands of FLOPs the attention itself spends per token it is in the noise. So the online
rotations are genuinely close to free, which is what makes it acceptable to leave them un-fused.

The KV-cache quantization is not a bonus — it is a first-class reason to want this at all, and worth a
sizing argument. In long-context or large-batch serving the cache dominates memory: for Llama-2-7B with
32 layers, 32 heads, head-dim 128, the K and V cache is 2 · 32 · 32 · 128 = 524k values *per token*, so a
4096-token context in FP16 is 524k · 4096 · 2 bytes ≈ 4.3 GB — comparable to the entire 4-bit weight
footprint, and it grows linearly with context and batch while the weights are fixed. Quantizing the cache
to 4 bits cuts that 4× to ~1.1 GB. But the cache holds keys and values, which have the same axis-aligned
outlier pathology as any activation, so naive 4-bit KV would break exactly as naive W4A4 does — which is
why the K/Q Hadamard is part of the pipeline, not an afterthought. The rotation makes the *cache*
quantizable by the same geometric argument that makes the activations quantizable.

Concretely, I fuse the norms, then pick an orthogonal Q and bake it into the weights: every matrix that
reads the residual stream gets its input rotated (W ← W Q), every matrix that writes to it gets its
output counter-rotated (W ← Qᵀ W), the embeddings get rotated (so the stream starts in the rotated frame)
and the language-model head un-rotated (so logits come out in the original frame). Nothing about the
output changes; everything about the *distribution* of the activations changes. The activations entering
each quantized linear are now in the rotated frame, outlier-free, and quantize cleanly to 4 bits.

Which orthogonal matrix should Q be? A random orthogonal matrix — sample a Gaussian matrix, take its QR
factorization — already works: it spreads the spikes exactly as the 1/√n argument says. But I can do
better on the *cost* axis with a **Hadamard** matrix. A Hadamard matrix has all entries ±1/√n, so
applying it is a sequence of additions and subtractions — a Walsh–Hadamard transform — computable in
O(n log n) via the butterfly recursion rather than the O(n²) of a dense orthogonal matmul. That matters
because not every rotation in the pipeline can be fused offline. The residual-stream rotation folds into
weights once and is free at runtime. But some rotations have to be applied *online* — inside the
attention value/output path, and on the keys and queries to make the KV-cache quantizable — and for those
an O(n²) dense rotation per token would be a real inference tax, whereas an O(n log n) Hadamard is
nearly free. A *plain* Hadamard is risky: its rows are a fixed, structured ±1 pattern, so a spike sitting exactly
where a Hadamard row is constant-sign could be concentrated rather than spread. Composing with a random
±1 sign diagonal randomizes which coordinates get which sign before the butterfly, breaking any such
alignment in expectation while staying exactly orthogonal and O(n log n). So Q is a randomized Hadamard on
the hidden dimension, fused into the weights where possible and run as a cheap WHT kernel where not (the
`rotate_model` pass is in the answer).

With the outliers dissolved, the activations quantize to 4 bits on ordinary per-tensor (or per-token)
scales — no special outlier path, no channels kept in higher precision, every matmul genuinely in 4-bit.
And I do not throw away the weight machinery I already trust: after rotating, I still run **GPTQ** on the
rotated weights to compensate the 4-bit weight rounding. Not AWQ, for a specific reason — AWQ scales
salient weight channels identified from activation magnitude, but the rotation has just *destroyed* the
axis-aligned channel structure it keys on; in the rotated frame every coordinate is statistically the
same, so there are no salient channels to scale. GPTQ does not care about channel identity — it minimizes
‖ŴX̂ − WX̂‖² in whatever basis — so it is exactly the right tool in a washed-out frame. The two stack
rather than fight: rotation prepares the operands, GPTQ optimizes the weight grid on them, and the online
K/Q Hadamard makes the KV-cache quantizable to 4 bits — the full W4A4KV4 pipeline, rotation-based
quantization.

The qualitative break from everything before is that there are *no higher-precision channels anywhere*.
SmoothQuant left the outliers smaller-but-present and leaned on the 8-bit grid's headroom; a
mixed-precision scheme would keep them in FP16. Here every coordinate is the same near-Gaussian, so every
matmul — weights, activations, KV — runs at 4 bits on a plain per-tensor/per-token grid with no
exceptions. That is what makes it *true* W4A4KV4 rather than "4-bit with an FP16 side channel."

Let me also reason about the *scale dependence*, because it is a falsifiable prediction the mechanism
makes. The outlier-spreading is a 1/√n effect: the residual dimension n grows with model size (4096 at
7B, 5120 at 13B, 8192 at 70B), so the smearing is *more* effective in wider models — a spike of the same
relative size dissolves into a smaller per-coordinate contribution (100/√8192 ≈ 1.1 at 70B versus
100/√4096 ≈ 1.56 at 7B). And the near-Gaussianization improves with dimension too, by the same
central-limit averaging. So the rotation argument should get *cleaner* as the model gets wider, which
predicts the W4A4 gap to FP16 should *shrink* with scale — the opposite of the usual expectation that
bigger models are harder to quantize.

The bar is SmoothQuant pushed to W4A4 on Llama-2-7B: ≈83 perplexity, broken. My claim is that *removing*
the outliers by rotating them out of existence finally makes 4-bit activations quantize as cleanly as
4-bit weights, and with GPTQ on the rotated weights the whole W4A4KV4 model stays within a small margin
of FP16 — Llama-2-7B down from ~83 to about 6.1 against FP16 5.47, and, by the 1/√n argument, the gap
*shrinking* with width (13B ~5.40 vs 4.88, 70B ~3.79 vs 3.32, within half a perplexity point at the
largest size). Two risks I can bound: the un-fused online Hadamards, cheap by the O(n log n) WHT; and the
assumption that a *random* rotation is good enough — a Hadamard spreads the mass on average but is not
*chosen* for this model, and different draws give measurably different realized error. That last point is
the loose thread: if rotating away the outliers is the right idea, the next question is whether the
rotation should be *learned* rather than left to chance.
