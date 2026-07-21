GPTQ closed most of the weight-only gap: 3-bit per-channel LLaMA-7B went from RTN's 25.54 down to 8.07,
against FP16's 5.68. That is a real win, and it makes me want to push harder on the low-bit weight-only
regime, because that is the one that actually speeds up batch-1 generation — where the bottleneck is
reading weights from DRAM, FLOPs are nearly free, and quantizing weights to 3–4 bits raises the
arithmetic intensity several-fold. But two things about GPTQ bother me when I try to go lower still, and
they are worth stating precisely rather than as unease.

First, it leans on a full second-order reconstruction: a Hessian H = 2XXᵀ per layer, a Cholesky inverse,
column-by-column error feedback. That is heavy machinery, and because it fits the quantized weights to
the calibration set's input statistics — H *is* that calibration set's second-moment matrix — I worry it
can overfit to the calibration distribution. It is a regression of the weights against a particular few
hundred sequences, and I already flagged that seam at the previous rung. Second, and more fundamentally,
GPTQ treats every weight as equally deserving of accuracy. Its objective min‖WX − ŴX‖² weights every
output direction the same; it has no notion that *some weights matter far more than others*. I want to
interrogate that, because if a tiny fraction of weights carries most of the model's quality, then
spending the bit budget uniformly is wasteful, and protecting exactly that fraction might beat
reconstructing everything — and do it without any Hessian at all.

So let me run the diagnostic honestly, as a real experiment I can reason about the outcome of. Suppose I
keep a small fraction of weight channels — say 0.1% to 1% — in full FP16 and quantize all the rest to
INT3. This is a mixed-precision probe: it asks "if I could protect a few channels perfectly, which ones
would recover the most quality?" The first candidate selection is the obvious one: keep the channels
with the largest *weight* magnitude, on the theory that big weights carry big signal. I reason through
what that should do, and then I look: keeping the top-magnitude weight channels in FP16 *barely helps* —
the INT3 model with its 1% largest-weight channels protected is still far from FP16. That is a surprise,
and it is the most informative result on this rung, so I want to understand why rather than move on. The
magnitude of a weight is not the same as its *importance to the output*. A weight w contributes wx to the
layer output, and its importance is governed by the product — by how large the activation x flowing
through that channel tends to be — not by w alone. A large weight sitting on an input channel that is
almost always near zero contributes almost nothing; a modest weight on a persistently large activation
channel dominates the output. So I try the other selection: keep in FP16 the input channels whose
*activations* have the largest average magnitude over the calibration set. And this recovers almost all
of the FP16 accuracy from an otherwise-INT3 model. The saliency is *activation-defined*, not
weight-defined. In hindsight it is obvious — importance lives in the product wx, and the weights are
already roughly uniform in scale while the activations have the heavy, persistent channels — but I had to
run both selections to see it, and it completely reframes the problem: the question is not how to round
weights better, it is which *input channels* deserve resolution, and the answer is written in the
activation statistics.

Now I have the right notion of saliency, but the obvious implementation is a dead end, and it is worth
being concrete about why. Keeping 1% of channels in FP16 and the rest in INT3 means storing a
*mixed-precision* tensor: a ragged layout with a handful of FP16 columns interleaved among INT3 columns.
On a GPU that is a disaster — the whole point of quantization was a clean, regular, hardware-friendly
weight matrix that a single kernel streams and a single scaled-integer GEMM consumes; a mixed FP16/INT
layout means predication, gather, two datapaths, and it throws away most of the bandwidth win I was
chasing in the first place. The 1% of FP16 channels would also, at INT3, cost disproportionately in
storage: at ~3 bits per quantized weight, replacing 1% of them with 16-bit values adds 0.01·(16−3) =
0.13 bits/weight, small, but the irregular *layout* cost dwarfs the storage cost. So I need the
*protection* the FP16 channels gave — their effective extra resolution — without the irregular layout at
all.

Two tempting alternatives to the plain activation-magnitude proxy fail. I could use a second-order
saliency — the Hessian diagonal [H]_jj = 2·(XXᵀ)_jj is, up to scale, the mean squared activation of
channel j — but that reintroduces exactly the machinery and calibration-regression risk I am escaping,
to refine a signal a one-pass mean|X| already captures. Or I could give salient channels a finer group
size (g32 on them, g128 elsewhere), but that is the ragged-layout problem again: mixed metadata, a
non-uniform kernel. The scaling transform is strictly better — it delivers the resolution through a
scalar per channel that folds entirely away.

Here is the move. I do not have to store the salient channels in higher precision; I can *scale them up*
before quantizing so that, on the integer grid, they get more effective resolution, and then divide the
corresponding activations down by the same factor so the layer's function is unchanged. For a linear
layer this is an exact equivalence transform:

  W X = (W · diag(s)) · (diag(s)⁻¹ X).

Multiply a weight channel by s and divide its input channel by s; the product WX is identical before
any quantization. The payoff is what happens *after* quantization, and I should derive it rather than
assert it, because the whole method turns on this ratio. With a round-to-nearest quantizer Q(w) =
Δ·round(w/Δ) and step Δ = max(|w|)/(2^(N−1)−1), the quantization error on a single weight is on the order
of Δ/2 — half a step — regardless of the weight's own value. So the *unscaled* salient channel
contributes an output error of about (Δ/2)·x. Now scale that channel's weight by s > 1 and its input by
1/s: the contribution becomes Q(ws)·(x/s), and its error is about (Δ'/2)·(x/s), where Δ' is the step of
the *scaled* channel's group. The ratio of scaled-to-unscaled error is therefore

  Err(Q(ws)(x/s)) / Err(Q(w)x) = (Δ'/Δ) · (1/s).

If s is moderate enough that the scaled-up salient weight does not become the new group maximum, then
the group's step is unchanged, Δ' ≈ Δ, and the salient channel's quantization error shrinks by about
1/s. That is the entire mechanism: I have given the important channel effectively finer resolution —
the same protection the FP16 slot gave it — using nothing but a per-channel scale that folds away at
inference. No mixed precision, no ragged layout; the stored tensor is still a plain group-wise INT3/INT4
matrix, one clean datapath.

Let me put numbers on it to be sure the mechanism is real and not a sign error. Suppose a salient
channel has weight magnitude 0.2 sitting in a g128 group whose max magnitude is 1.0, so at INT4 (levels
0…15, but read symmetric-equivalently 2^(N−1)−1 = 7 on each side for the scale) the step is Δ = 1.0/7 =
0.143, and this channel's weight rounds with error up to Δ/2 = 0.071 — which is 36% of the weight's own
value of 0.2, a large relative error precisely because the channel is small relative to its group max.
Now scale it by s = 2: the weight becomes 0.4, still below the group max of 1.0 (so Δ' = Δ = 0.143
unchanged), and its input is divided by 2. The channel's output-error contribution was (0.071)·x; it is
now (Δ'/2)·(x/2) = (0.071)·(x/2), exactly half. The ratio (Δ'/Δ)·(1/s) = 1·(1/2) = 0.5 checks out — the
protection is real and it came for free because the scaled weight stayed under the group max. Push to s =
6, though, and the weight becomes 1.2, now *above* the old group max: Δ' = 1.2/7 = 0.171, and the ratio is
(0.171/0.143)·(1/6) = 1.2·0.167 = 0.20 for this channel — better still for it — but every one of the
other 127 weights in the group now rounds against 0.171 instead of 0.143, a 20% coarser step across the
whole group. That is the trade laid bare, and it is why s cannot just be pushed to infinity.

So there is a sweet spot in s: past the point where the scaled salient weight becomes the group max,
every gain to the protected channel comes out of its 127 group-mates, and eventually that damage
overwhelms the protection. The sweet spot depends on the joint distribution of weights and activations in
each layer, which I have no clean closed-form model of. So the principled move is to *parametrize* the
scale to track activation magnitude and *search* the one free knob: s = s_X^α, with s_X the
per-input-channel average activation magnitude and α a single exponent, searched over a small grid that
directly minimizes the layer's output MSE against the FP16 reference.

One choice inside s_X deserves a second's thought: should the per-channel activation statistic be the
*mean* absolute magnitude over calibration tokens, or the *max*? Max would key the protection to the
single largest activation ever seen in a channel, which is exactly the volatile, outlier-driven quantity
I do not trust to generalize from a few hundred sequences; a channel that spikes once should not
dominate the layer's scale allocation. The mean absolute value is the stable, expectation-like statistic
— it says "this channel is persistently large", which is the property that makes a channel's weights
persistently important to the output. So s_X = mean|X| per channel, and it happens to be the same
persistent-channel structure I first met as the villain at the RTN floor: the handful of input features
that are large across nearly all tokens. Here I am only *reading* those channels to decide where to
spend weight resolution — I am not yet trying to quantize them — but it is the same fixed set of
coordinates, and I file away that the activation side of the problem is going to come back.

The α knob spans a well-posed range. At α = 0, s_X^0 = 1 everywhere: no scaling, plain group RTN. As α
grows, salient channels get scaled up harder relative to the quiet ones — protection rising — until the
scaled salient weights start becoming group maxima and the Δ' penalty bites. So output MSE as a function
of α is a bowl, and a grid of ~20 values from 0 to 1 finds its bottom. One normalization keeps the search
honest: after forming s = s_X^α I divide by √(max(s)·min(s)), pinning the scales geometrically around 1
so the transform redistributes resolution among channels without a common factor s_X^α inflating every
group's max in lockstep — otherwise the MSE curve would mix relative protection (what I want) with a
global scale the per-group Δ already handles. The quantizer the search wraps is the plain group-wise
affine grid from the floor, with a per-128 scale and zero-point; nothing here is second-order. The
intelligence is entirely in the *scales* the search chose — AWQ moves the cleverness out of the rounding
algorithm and into a one-parameter reweighting of the layer. (The statistic, search, and grid are in the
answer.)

Now compare the cost of this against GPTQ, because "lighter" is a real claim I should be able to defend.
GPTQ needs the d×d Hessian, its Cholesky inverse (O(d³) per layer), and column-wise error feedback. AWQ
needs the per-channel activation mean (one pass over calibration, O(tokens·d)), and then ~20 forward
passes through the layer to score the α grid — no Hessian, no inverse, no second-order anything. It is
not only cheaper in FLOPs; it is *structurally* safer against the overfitting worry, because it never
regresses individual weights against the calibration set. It fits exactly one scalar α per layer against
the output MSE, and one scalar cannot memorize a few hundred sequences the way a per-weight second-order
correction can. So I get robustness to the calibration distribution as a side effect of using a far
smaller hypothesis class.

Then the chosen scale is *folded into the surrounding ops*, and the fold is exact, not approximate. The
current linear reads x = LayerNorm(h) and I want it to see diag(s)⁻¹·x while its weights carry diag(s).
Since LayerNorm's output is an elementwise affine map γ⊙x̂ + β, dividing γ (and β) by s elementwise
produces exactly diag(s)⁻¹·(γ⊙x̂ + β), and multiplying the next linear's weight columns by s restores WX
identically — the only approximation left in the whole method is the quantization itself, which the scale
is chosen to shrink. So after folding, the served model is byte-for-byte a normal group-wise INT3/INT4
weight-only model with FP16 activations; an inference kernel cannot even tell this was applied. The
activation statistic decided which channels deserved resolution; the equivalence transform delivered it
without an irregular layout — protect by scaling, not by storing.

The bar and the bet, and I have to be careful about which column I compare in. AWQ is naturally a
group-wise method — the whole error-ratio argument is about group maxima — so the honest comparison is
at the *same* grouping, and g128 numbers are not interchangeable with the per-channel numbers of the
earlier rungs. So I pin AWQ against RTN-*g128*, not against the 25.54 or 8.07 per-channel figures: on
Llama-2-7B, RTN at INT3-g128 is 6.66 and at INT4-g128 is 5.73, both against an FP16 of ≈5.47. My claim is
that one searched per-channel scale per layer — pure forward passes, no second-order machinery, no risk
of regressing onto the calibration set — closes most of the remaining gap to FP16 by spending resolution
where the activations say it matters. On the magnitudes: at INT3-g128 the RTN gap to FP16 is 6.66 − 5.47 ≈ 1.19, and salient-channel
protection attacks its dominant part (the few channels whose large activations make their rounding count
most), so a good fraction of the gap should close — INT3-g128 down toward ~6.2. At INT4-g128 the grid is
already fine (16 levels, RTN only 0.26 above FP16), so the same mechanism shaves it to within about a
tenth of FP16, into the ~5.6 range. The risk is the group-max coupling — over-scale and I trade one
channel's accuracy for its group-mates', which is why α is searched against true output MSE. If it holds,
a lighter, calibration-robust weight-only method matches or beats the heavy second-order one, and the
weight-only frontier is in good shape at both 3 and 4 bits. The next pressure cannot come from weights at
all — it has to come from the side every method so far has left in FP16: the *activations*, whose
persistent outlier channels I have so far only identified, not quantized.
