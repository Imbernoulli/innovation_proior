Weight-only quantization is in good shape now. AWQ at INT4-g128 on Llama-2-7B sits at 5.60 perplexity,
a tenth off FP16, and even INT3-g128 is at 6.24. I have squeezed the weights about as hard as the
batch-1 generation use case demands. But I have been quietly ignoring half the matmul this whole time.
Every method so far — RTN, GPTQ, AWQ — quantizes W and leaves the activations X in FP16. That is fine
when the bottleneck is reading weights from DRAM, which is the memory-bound, batch-1 regime. It is *not*
fine for the other regime: large-batch serving, where you are throughput-bound and you want the multiply
itself to run on the hardware's integer tensor cores. Those INT8 GEMM units need *both* operands in
INT8. Weight-only quantization buys me nothing there, because the arithmetic still happens in FP16 — the
weights get dequantized on the way in and the matmul runs at FP16 rates.

The regime is different in a way that changes which quantization helps. In large-batch prefill or
serving, a single weight matrix is reused across hundreds of token-vectors, so arithmetic intensity
climbs past the roofline ridge and the kernel is compute-bound — the lever is now the *rate of the
multiply*, and an INT8 tensor core runs at roughly 2× the FP16 pipe. To claim that the matmul's two
inputs must both be INT8 so the accumulation happens in the integer datapath, and weight-only cannot get
there because X is still FP16. So the target shifts to **W8A8** — INT8 weights *and* activations — with
the linears running as dense INT8 GEMMs.

The weights I already know how to quantize to 8 bits; from the floor's SQNR arithmetic, 8-bit RTN on
weights is ~41 dB, essentially lossless, and I do not need any of the GPTQ/AWQ machinery for that — the
grid is fine enough. The whole problem is the activations, and it is a brutal one. When I just apply RTN
to the activations and run W8A8, accuracy on a large model collapses — on OPT-175B the zero-shot average
falls to about 35.5%, which is chance-level; the model is destroyed. I need to understand *why* before I
can fix it, because this is the same outlier failure I flagged back at the RTN floor and identified again
as the persistent activation channels AWQ read for saliency — and now I have to actually defeat it rather
than route around it.

Here is the structure of the difficulty, and I want to quantify it because the numbers dictate the
approach. LLM activations have a small number of *persistent channels* — the same input-feature columns
across nearly all tokens — whose magnitudes are on the order of 100× larger than everything else. A
per-tensor activation quantizer sets its single step Δ from the global maximum, which those outlier
channels dominate. Run the arithmetic at 8 bits (levels −127…127): if the outlier channel peaks at 100
and ordinary activations sit near 1, then Δ = 100/127 = 0.79, so an ordinary activation of magnitude 1
rounds to round(1/0.79) = round(1.27) = 1 — a single level, when it deserved ~127 of them. The ordinary
signal, which is the overwhelming majority of the information, is being crushed into one or two levels
near zero while the grid spends its whole range describing the handful of outliers. The information in
the bulk of the activations is annihilated, and the model goes to chance. And the problem is not
scale-invariant — it sharpens with size: small models have fairly uniform channels that naive per-tensor
INT8 survives, but as width and depth grow a few channels pull away by larger factors until the ratio is
the ~100× that destroys per-tensor quantization at the hundred-billion scale. So OPT-175B is the right
stress test — the giant model where INT8 throughput matters most and naive quantization fails worst — and
its 35.5% (chance ~33% on this suite) is a *destroyed*, not merely degraded, model: rescue the worst case
and the rest follow.

The natural fix everyone reaches for is *per-channel* activation quantization: a separate Δ per input
channel, so the outlier channels get their own coarse grid and the quiet ones keep a fine one. And it
works — simulated per-channel activation quantization recovers FP16 accuracy, because the ordinary
channels now round against ~1, not against 100.

But there is a wall, and it is a hardware wall, not a statistics one — this is the crux of the whole
rung. Look at where the activation scaling axis lives in the matmul Y = X W, with X of shape [T tokens ×
C_in] and W of shape [C_in × C_out]. A per-channel activation scale is indexed by C_in — the
*contraction* axis of the matmul, the axis being summed over. An INT8 GEMM computes Y_tk = Σ_c X_tc W_ck
in the integer accumulator and can only apply dequantization scales as an *epilogue*, along the output
dimensions T and C_out, *after* the sum. It physically cannot apply a different scale per element of the
contraction axis c, because by the time the scale would be applied that axis has already been summed
away — the value in the accumulator is Σ_c X_tc W_ck, and no single post-hoc scalar per (t,k) can undo a
different per-c scaling that happened inside the sum. So the granularity that fixes activation outliers
(per-channel, indexed by C_in) is exactly the granularity the fast kernel forbids, and the granularity
the kernel allows (per-token, indexed by T) does nothing about outliers, because the difficulty is
organized by channel, not by token — every token has the same outlier channels, so a per-token scale
sees the same 100× spike in every row and cannot separate it out. The field is stuck between a quantizer
that works but will not run and a quantizer that runs but does not work.

The two escapes people reach for both fail for the same reason. A *mixed-precision outlier decomposition*
— keep the outlier channels in FP16, run the rest through the INT8 GEMM, add the outlier contribution
back — preserves accuracy but is the ragged-layout disease on the activation side and worse, since the
outlier set shifts with the input: a dynamic split, a separate FP16 matmul, a scatter-add, all eating the
throughput win. Fully *dynamic per-channel* activation quantization is exactly the contraction-axis
granularity the tensor core forbids. Both try to handle outliers *at runtime*, where the hardware allows
only per-token/per-tensor scales. The only way out is to make the activations un-outlier-y *before* they
reach the kernel.

So let me stop trying to scale activations *at runtime* and ask whether I can rebalance the difficulty
*offline*. The outliers are a property of which channels are large — a fixed, persistent set — and I can
see those channels from calibration data ahead of time. This is exactly the equivalence-transform lever
AWQ used to protect weight channels, but I am going to point it the other way. There, I scaled salient
weight channels *up* to protect them and divided activations *down* to compensate, and the goal was
better weight rounding. Here the weights are already easy (8-bit is lossless) and the *activations* are
the problem, so I want to divide the activation outliers *down* and let the weights absorb the magnitude
by scaling *up*. For Y = X W, insert a per-input-channel factor diag(s) that shrinks the activations and
grows the weights by the same amount:

  Y = (X · diag(s)⁻¹) (diag(s) · W) = X̂ Ŵ.

This is an exact identity before quantization — the same computational-equivalence move, just aimed at
the operand that is now the bottleneck. Now the activations X̂ have had their outlier channels divided
down — the 100× spikes are tamed toward the bulk — and the weights Ŵ have absorbed that magnitude. The
activations become *easy* to quantize (no more channel outliers blowing up Δ) and the weights become
*harder*, but weights were nearly free to quantize to begin with, so this is a good trade. Crucially, the
per-channel factor is along C_in for *both* tensors, and — this is the whole point — for both it can be
folded away offline: diag(s)⁻¹ folds into the preceding LayerNorm (or previous linear) and diag(s) folds
into this linear's weights, once, before serving. So at runtime there is no per-channel scaling kernel at
all — the correction that had to live on the forbidden contraction axis has been *baked into the
parameters*, and the served matmul is a plain INT8 GEMM with only the allowed per-tensor/per-token
scales. I have moved the per-channel correction off the contraction axis by paying for it offline.

The only real decision is how much difficulty to migrate — the choice of s — and I want to derive it
rather than tune it blindly. If I push s all the way to flatten the activations completely, I dump all
the difficulty onto the weights and overload *them*; if I push the other way I am back to broken
activations. So I want s to *split* the difficulty between the two operands. The clean way to express
that: for each channel j, take

  s_j = max(|X_j|)^α / max(|W_j|)^(1−α),  with α = 0.5 by default.

At α = 0.5, s_j = √(max|X_j|/max|W_j|), so the smoothed activation max max|X_j|/s_j and the smoothed
weight max max|W_j|·s_j both become √(max|X_j|·max|W_j|) — the same value, the geometric mean; the burden
is shared exactly evenly. With numbers: an outlier channel at max|X_j| = 100, max|W_j| = 0.1 gets s_j =
√1000 = 31.6, and both smoothed maxima become √10 = 3.16. The activation outlier fell 31× into the range
of ordinary channels while the weight max only rose from 0.1 to 3.16, which the 8-bit weight grid handles
with ease. At α = 1 the activations flatten but the weights overload; at α = 0 the reverse. Models with
heavier outliers (GLM-130B) want α ≈ 0.75 to push more difficulty onto the still-easy weights: when the
activation maxima tower over the weight maxima, nudging α up lowers the smoothed activation max faster
than it raises the weight max (the activation max being the larger number under the exponents α and 1−α),
so more of the imbalance lands where it can be afforded. The right α is a quick validation-set grid
search. The activation maxima come from a calibration pass; the weight maxima are exact. Once α and those
statistics are fixed, s is closed-form — no gradients, no per-weight
reconstruction, no mixed-precision outlier path. It is worth appreciating how cheap this is next to the
weight-only methods: GPTQ needed a d×d Hessian and a Cholesky inverse per layer, and AWQ needed ~20
forward passes per layer to search α against output MSE. SmoothQuant needs only a *single* statistic per
input channel — the running max of |X| over the calibration pass — plus the exact weight max, and one
scalar α shared across the whole model (or grid-searched over a handful of values on a validation set).
There is no per-layer optimization at all; the migration is a closed-form reparametrization applied
uniformly. That lightness is not incidental — it is what makes the method deployable on a 175B model
where even one Hessian per layer would be a serious compute and memory burden.

The trade is favorable because the two operands are not symmetric. Before smoothing, the weight
per-channel maxima are already fairly uniform — trained weights lack the 100× persistent-channel
pathology activations have — so multiplying a channel by s = 31.6 raises its weight max to ~3.16 while the
spread of weight maxima widens only modestly, and at 8 bits a per-channel grid absorbs a max of 3.16
trivially (Δ = 3.16/127 = 0.025). The migration converts a *fatal* activation problem into a *negligible*
weight one, which is why the trade wins and why α = 0.5 is a safe default rather than a knife-edge: even
an even split leaves both sides inside 8-bit's comfort zone. (The `smooth_ln_fcs` fold is in the answer.)

I should be careful about *which* linears share a smoothing factor, because the fold has to stay exact.
The factor diag(s)⁻¹ is folded into a LayerNorm, and every linear that reads that LayerNorm's output must
receive the matching diag(s) — so the group is "one LayerNorm and all the linears it feeds": the
attention input LayerNorm with Q, K, V together, and the FFN input LayerNorm with the first MLP
projection. They must share one s because they share one input channel axis; picking s per-linear would
break the single fold into the shared LayerNorm. That is a genuine constraint on the recipe, not a
detail — it is why smoothing is organized around LayerNorm-fed groups rather than individual matrices.
If the producer is a linear instead of a LayerNorm, the reciprocal folds into that producer's output
weights and bias instead.

With the outliers migrated, I get to *choose* the runtime granularities the tensor core actually allows,
and the choice matters. On the activation side the kernel permits a per-token scale (indexed by T, an
output axis of the GEMM), so I use per-token dynamic INT8 for activations — each token-vector gets its
own Δ computed cheaply at runtime — which costs nothing on the contraction axis and mops up any
token-to-token magnitude variation the offline smoothing did not touch. On the weight side I use
per-channel (per-output-channel) INT8, also an allowed epilogue axis. Both are legal because both are
indexed by output dimensions, and together they are as fine-grained as the hardware will let me be
*without* touching the forbidden C_in axis — which is precisely the axis I already neutralized offline.
So the division of labor is clean: the offline smoothing kills the *channel* (C_in) structure of the
difficulty, and the allowed per-token/per-channel runtime scales handle the residual *token* and
*output-channel* structure. Nothing is left on the axis the kernel cannot reach.

Now, why does this cross the finish line at 8-bit but only there? Once smoothed, the activations have no
channel spikes, so their per-tensor Δ is set by ordinary magnitudes and the 256-level INT8 grid gives
every activation ample resolution; the weights, now peaking around a few units instead of a fraction,
are still comfortably within 8-bit's ~41 dB headroom. Both operands quantize cleanly, the layers run as
plain INT8 GEMMs, and I expect the model to come back to near-FP16. But I can already see the ceiling. I
did not *remove* the outliers, I *shrank* them: the 100× spike became a 3.16× one relative to the bulk.
At 8 bits, 3.16× fits inside the grid with room to spare. At 4 bits, though, there are only 16 levels,
and a per-tensor Δ set by a residual 3.16× outlier gives ordinary activations round(1/(3.16/7)) =
round(2.2) = 2 levels — coarse again, and the bulk starts collapsing. So the same per-tensor scheme that
survives 8-bit activations will *not* survive 4-bit activations: the migrated outliers are *smaller*,
not *gone*, and at 16 levels smaller is not enough. I flag that now because it is the wall the next rung
must break.

The bar is the *naive* W8A8: OPT-175B at 35.5% zero-shot, chance level, against FP16's 66.9%. I use
zero-shot accuracy rather than perplexity here deliberately — the breakthrough is whether INT8 *activation*
quantization works at scale at all, and on a 175B model that is how it is reported; being the coarser
metric, restoring it to near-FP16 is a claim the model is genuinely functional. My bet is that migrating
the outliers into the weights offline lets both operands quantize to INT8 on hardware-friendly scales and
brings the OPT-175B zero-shot average back to within a fraction of a point of 66.9% — essentially
lossless. The risk is the split, which is why α is grid-searched per model. If it holds, I have crossed
from weight-only into true weight-and-activation quantization. But this is INT8: at W4A4 the residual
outliers reappear inside the 16-level grid, and a sharper instrument — one that *removes* the outliers
rather than shrinking them — will be needed.
