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

There is a tempting alternative to the pure activation-magnitude proxy that I should rule out before I
build on it. GPTQ already gave me a per-channel importance signal — the Hessian diagonal [H]_jj = 2·(XXᵀ)_jj
is, up to scale, the mean squared activation of input channel j, so I could weight or select channels by
a second-order saliency instead of by mean|X|. But that reintroduces exactly the machinery and the
calibration-regression risk I am trying to escape, and the diagnostic already told me what I need: the
importance axis is the activation channel, and the *mean absolute* activation is a stable enough proxy
that keeping the top-scoring channels recovers nearly all of FP16. Spending a Hessian to refine a signal
that a one-pass mean already captures well would trade away the lightness that is half the point of this
rung. A second alternative is to protect salient channels by giving *them* a finer group size — say g32
on the salient columns, g128 elsewhere — but that is the ragged-layout problem again in a different
costume: mixed group sizes mean mixed metadata and a non-uniform kernel. The scaling transform is
strictly better because it delivers the resolution through a scalar per channel that folds entirely
away. So I reject both and commit to searched activation-magnitude scaling.

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

The catch is the same Δ' in that ratio, and it is where the method could quietly break, so I want to see
the failure explicitly. Δ' is the max magnitude of the *group* the scaled channel sits in, divided by
the level count. If I scale a salient channel by, say, s = 4 and that pushes its weight from being 40% of
the group max to being the new group max at 1.6× the old one, then Δ' = 1.6·Δ, and the error ratio for
the protected channel is (1.6)·(1/4) = 0.4 — still an improvement, but now *every other weight in that
128-wide group* rounds against a step 1.6× coarser, so I have bought one channel's accuracy by degrading
its 127 group-mates. Push s harder and the group-mate damage overwhelms the protection. So there is a
sweet spot in s, and it is not something I should try to derive in closed form — it depends on the joint
distribution of weights and activations in each layer, which I do not have a clean model of. The
principled move is to *parametrize* the scale so it tracks activation magnitude and then *search* the one
free knob. I set s = s_X^α, where s_X is the per-input-channel average activation magnitude over the
calibration set and α is a single exponent, and I search α over a small grid, choosing the value that
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

```python
@torch.no_grad()
def get_act_scale(x):                       # per-input-channel mean |activation|, the stable statistic
    return x.abs().view(-1, x.shape[-1]).mean(0)
```

Let me think about what the α knob spans, because that tells me the search is well-posed. At α = 0, s_X^0
= 1 for every channel: no scaling, back to plain group RTN. As α grows, the salient (large-s_X) channels
get scaled up more and more aggressively relative to the quiet ones, so protection increases — until α is
large enough that the scaled salient weights start becoming group maxima and the Δ' penalty bites. So the
output MSE as a function of α should be a bowl: high at α = 0 (no protection), falling as protection
turns on, then rising again as over-scaling coarsens the groups. A grid of ~20 values of α from 0 to 1 is
plenty to find the bottom of that bowl. There is one more normalization I want, so that changing α does
not secretly rescale the whole layer: after forming s = s_X^α I divide by √(max(s)·min(s)), which
recenters the scales around 1 geometrically, so the transform redistributes resolution among channels
without inflating or shrinking the overall weight magnitude — the group maxima move only because of the
*relative* reweighting, which is exactly what I want to isolate. Without this normalization the search
over α would be confounded: a larger α would not only reweight channels but also multiply every weight by
a common factor s_X^α evaluated at the mean channel, which shifts every group's max in lockstep and
changes Δ globally, so the output-MSE curve I am minimizing would mix the effect I care about (relative
protection) with an effect I do not (a global scale that the per-group Δ already handles). Dividing by the
geometric midpoint √(max·min) pins the scales symmetrically around 1 on a log axis, so the search sees
only the redistribution and the bowl in α is clean.

```python
x_scale = get_act_scale(x)                 # per-input-channel mean |activation| over calibration
best_loss, best_scales = float("inf"), None
for grid in range(n_grid):                 # search the exponent α
    alpha = grid / n_grid
    scales = x_scale.pow(alpha).clamp(min=1e-4)
    scales = scales / (scales.max() * scales.min()).sqrt()
    for fc in linears:                     # W · diag(s), quantize, divide back by diag(s)
        fc.weight.mul_(scales)
        fc.weight.data = pseudo_quantize_tensor(fc.weight.data, n_bit, q_group_size) / scales
    loss = (org_out - module(x)).float().pow(2).mean().item()   # output MSE vs FP16
    if loss < best_loss:
        best_loss, best_scales = loss, scales
```

The quantizer the search wraps is a plain group-wise affine grid — the same round-clamp-dequantize as
the floor, but with a per-128 scale and zero-point so each group tracks its own min and max:

```python
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=128):
    org_shape = w.shape
    w = w.reshape(-1, q_group_size)                         # group-wise affine grid
    max_val, min_val = w.amax(dim=1, keepdim=True), w.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int  # Δ per 128-wide group
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    w = (torch.clamp(torch.round(w / scales) + zeros, 0, max_int) - zeros) * scales
    return w.reshape(org_shape)
```

Nothing here is second-order; the intelligence is entirely in the *scales* the search chose before this
runs, and this routine just executes the grid. That is the point — AWQ moves the cleverness out of the
rounding algorithm and into a one-parameter reweighting of the layer.

Now compare the cost of this against GPTQ, because "lighter" is a real claim I should be able to defend.
GPTQ needs the d×d Hessian, its Cholesky inverse (O(d³) per layer), and column-wise error feedback. AWQ
needs the per-channel activation mean (one pass over calibration, O(tokens·d)), and then ~20 forward
passes through the layer to score the α grid — no Hessian, no inverse, no second-order anything. It is
not only cheaper in FLOPs; it is *structurally* safer against the overfitting worry, because it never
regresses individual weights against the calibration set. It fits exactly one scalar α per layer against
the output MSE, and one scalar cannot memorize a few hundred sequences the way a per-weight second-order
correction can. So I get robustness to the calibration distribution as a side effect of using a far
smaller hypothesis class.

Then the chosen scale is *folded into the surrounding ops* — divided into the preceding LayerNorm or
previous linear's output, multiplied into the current linear's weights — so the network function is
exactly preserved and there is zero runtime overhead. I should convince myself the fold is genuinely
exact and not just approximately so. The current linear reads x = LayerNorm(h), and I want it to instead
see diag(s)⁻¹·x while the linear's weights carry diag(s). Since LayerNorm's output is an elementwise
affine map γ⊙x̂ + β of the normalized input, dividing γ (and β) by s elementwise produces exactly
diag(s)⁻¹·(γ⊙x̂ + β) — the per-channel division commutes through the elementwise scale-and-shift with no
residual. Then multiplying the next linear's weight columns by s restores WX identically. There is no
approximation anywhere in the fold: the only approximation in the whole method is the quantization
itself, and the scale is chosen to make *that* smaller. That is what "zero runtime overhead" means
concretely — after folding, the served model is byte-for-byte a normal group-wise INT weight-only model;
an inference kernel cannot even tell AWQ was applied. The stored model is an ordinary group-wise INT3 or
INT4 weight-only model with FP16 activations. The activation statistic decided which channels deserved
resolution; the equivalence transform delivered it without an irregular layout. This is **activation-
aware weight quantization**: protect by scaling, not by storing.

```python
@torch.no_grad()
def scale_ln_fcs(ln, linears, scales):     # fold diag(s)^-1 into LayerNorm, diag(s) into next linears
    ln.weight.div_(scales)
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(scales)
    for fc in linears:
        fc.weight.mul_(scales.view(1, -1))
```

The bar and the bet, and I have to be careful about which column I compare in. AWQ is naturally a
group-wise method — the whole error-ratio argument is about group maxima — so the honest comparison is
at the *same* grouping, and g128 numbers are not interchangeable with the per-channel numbers of the
earlier rungs. So I pin AWQ against RTN-*g128*, not against the 25.54 or 8.07 per-channel figures: on
Llama-2-7B, RTN at INT3-g128 is 6.66 and at INT4-g128 is 5.73, both against an FP16 of ≈5.47. My claim is
that one searched per-channel scale per layer — pure forward passes, no second-order machinery, no risk
of regressing onto the calibration set — closes most of the remaining gap to FP16 by spending resolution
where the activations say it matters. Reasoning about the magnitudes: at INT3-g128 the RTN gap to FP16 is
6.66 − 5.47 ≈ 1.19, and the salient-channel protection attacks the dominant part of that (the few
channels whose large activations make their rounding error count most), so I expect roughly a third of
the gap to close — INT3-g128 down toward ~6.2. At INT4-g128 the grid is already fine (16 levels, RTN
only 0.26 above FP16), so there is less to recover, but the same mechanism should shave it to within
about a tenth of FP16 — into the ~5.6 range. The risk is the group-max coupling I derived: over-scale and
I trade one channel's accuracy for its group-mates', which is exactly why α is searched against true
output MSE rather than set by a formula. If it holds, I have a lighter, calibration-robust weight-only
method that matches or beats the heavy second-order one — and the whole weight-only frontier, at both 3
and 4 bits, is in good shape. Which means the next pressure cannot come from the weights at all. It has
to come from the side every method so far — RTN, GPTQ, and now AWQ — has left untouched in FP16: the
*activations*, whose persistent outlier channels I have so far only had to identify, not quantize.
