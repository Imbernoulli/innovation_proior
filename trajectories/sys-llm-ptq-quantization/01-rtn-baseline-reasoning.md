I have a fixed pretrained LLM and a single instruction: make it smaller and faster to serve without
retraining it. The weights are FP16; the obvious thing to do is store them in fewer bits. Before I
reach for anything clever I want to know exactly how far the simplest possible scheme gets me, because
whatever it leaves on the table is the problem the rest of the work has to solve, and I do not want to
build machinery to fix a problem that rounding alone already handles.

The simplest scheme is uniform integer quantization by round-to-nearest. A weight matrix W is a grid
of floats; I pick a step size Δ, divide every entry by Δ, round to the nearest integer, clamp into the
representable range, and store the integers plus Δ. At inference I multiply back by Δ to recover an
approximation Ŵ ≈ W. The only real decision is the step size, and the only honest way to set it from
the data alone is to look at the magnitudes: take Δ = max(|W|) / (2^(N−1) − 1) for an N-bit symmetric
grid, so the largest-magnitude weight lands at the edge of the range and everything else falls inside.
There is no calibration here, no Hessian, no search — Δ comes straight from the tensor's own min/max.
That is the whole appeal: it costs nothing, it is exact in the sense that it never overflows the grid,
and it is trivially parallel.

Let me first make the payoff concrete, because it is the reason I am doing any of this. A 7B model in
FP16 is 7×10⁹ weights × 2 bytes ≈ 14 GB just to hold the parameters. At 4 bits that falls to 7×10⁹ ×
0.5 bytes ≈ 3.5 GB, a clean 4×; at 3 bits it is roughly 2.6 GB, closer to 5.3×. Those numbers matter
for two different reasons and I should keep them separate. The first is that the whole model now fits in
a much smaller memory budget — a 4× reduction is the difference between spilling to a second accelerator
and not. The second, and the one that actually governs speed, is that batch-1 autoregressive generation
is *memory-bandwidth-bound*: for a single token the arithmetic per weight is one multiply-add, so the
kernel spends its time reading the weight matrices out of DRAM, not computing. If I cut the bytes read
per weight by 4×, the arithmetic intensity rises and the token latency drops by something close to that
factor. So low-bit weight-only quantization is not a memory trick that costs speed — for this workload
it *is* the speed. That is why I care about pushing bits down rather than settling at a safe 8.

The bandwidth claim is a roofline fact: at ~2 FLOPs per weight element read against an accelerator's
hundreds-of-FLOPs-per-byte ridge point, the batch-1 kernel sits far out on the bandwidth-bound side, so
its runtime is essentially (bytes of weights)/(bandwidth) and quartering the bytes moves it almost
linearly — weight-only quantization buys speed even though it touches no FLOPs. The picture inverts only
for large-batch serving, where many tokens reuse each weight, arithmetic intensity climbs, and I would
instead need the *matmul* itself in low precision — a fork I note now because the later, harder regime
lives on it.

Now the first real design question: *granularity*. One Δ for an entire weight matrix (per-tensor) is
crude, because different output channels of a linear layer have very different scales, and a single Δ
forces the quiet channels to round against the loud channels' range. Concretely, if one output row of a
4096×4096 projection has entries up to 0.8 and another row peaks at 0.05, a shared per-tensor Δ is set
by the 0.8 row; the 0.05 row then has an effective step 16× too coarse for it, and at low bit-width its
weights collapse toward a couple of levels near zero. The cheap fix that costs almost nothing is
**per-channel** scaling: one Δ per output channel (per row of the weight). The storage overhead is a
single FP16 scale per row — for a 4096-row matrix that is 4096 extra scalars against 4096×4096 ≈ 1.7×10⁷
weights, i.e. 0.024% overhead, negligible — and it lets every channel round against its own range. The
finer-still option is **group-wise**: split each row into contiguous groups of, say, 128 weights and
give each group its own Δ (and, for an affine grid, its own zero-point). Group-128 — written g128 — now
tracks the local magnitude within each block, so an outlier weight only coarsens its own 128-wide
neighborhood instead of the whole row.

I should price g128 honestly, because it is not free and the accounting is exactly what makes cross-
comparisons dishonest if I am sloppy. One FP16 scale per 128 weights is 16 bits / 128 weights = 0.125
bits per weight of overhead; add a zero-point for the affine grid and it is 0.25. So a nominal "4-bit
g128" model is really carrying about 4.125–4.25 effective bits per weight, while "4-bit per-channel"
carries essentially 4.00. That difference is small in bits but it buys disproportionately better
rounding, because it is spent precisely where it helps — localizing Δ. The consequence I have to burn
into every comparison from here on: a g128 number is always going to look better than a per-channel
number at the same nominal bit-width, and if I ever line them up in the same column I will fool myself.
I will keep both settings but only ever compare within a matched grouping.

Now the real question: how low can the bit-width go before this falls apart? Let me quantify the
rounding noise rather than hand-wave it. Round-to-nearest with step Δ produces an error uniformly
distributed on [−Δ/2, Δ/2], with variance Δ²/12 and RMS Δ/√12 ≈ 0.289 Δ. For an N-bit symmetric grid the
positive levels number 2^(N−1) − 1, so Δ = M / (2^(N−1) − 1) where M is the channel's max magnitude. The
honest figure of merit is not the absolute error but the error *relative to a typical weight*, so let me
model a channel's weights as roughly Gaussian with standard deviation σ. For a channel of ~4096 weights
the max is about σ√(2 ln 4096) ≈ σ√(2·8.32) ≈ 4.08 σ, so M ≈ 4σ. Then at **8 bits** Δ = 4σ/127 = 0.031σ,
RMS noise ≈ 0.009σ, a signal-to-quantization-noise ratio of (σ/0.009σ)² ≈ 1.2×10⁴, about 41 dB — the
weights are essentially untouched. At **4 bits** Δ = 4σ/7 = 0.571σ, RMS noise ≈ 0.165σ, SQNR ≈ 37, about
16 dB — modest but survivable, especially with g128 keeping M local. At **3 bits** Δ = 4σ/3 = 1.33σ, RMS
noise ≈ 0.385σ, SQNR ≈ 6.7, about 8 dB — the rounding noise is now nearly 40% of the signal's own
standard deviation. And this is per weight, in a network stacked 32 layers deep; the perturbations
accumulate and shift the output distribution far enough that the softmax over the vocabulary is
materially wrong. So the arithmetic already tells me where the cliff is: 8-bit is free, 4-bit is a
manageable dent (worse per-channel than g128), and 3-bit per-channel is where an entire channel's worth
of weights is trying to live on a grid with only 2³ = 8 levels total, and the SQNR falls into single
digits.

Let me trace the mapping by hand on a tiny channel to be sure I have the grid geometry right, because
the symmetric convention has an asymmetry that is easy to get wrong. Take five weights w = [0.80, 0.31,
−0.12, 0.05, −0.62] at 3-bit symmetric. The max magnitude is M = 0.80, and 2^(N−1) − 1 = 3, so Δ =
0.80/3 = 0.267. Dividing and rounding gives w/Δ = [3.00, 1.16, −0.45, 0.19, −2.32] → round →
[3, 1, 0, 0, −2], then clamp to [−4, 3]. Dequantizing back, Ŵ = Δ·q = [0.80, 0.267, 0, 0, −0.533]. Read
the errors: the 0.80 weight is exact (it defined the scale), the 0.31 became 0.267 (error 0.043), the
−0.12 and 0.05 both collapsed to **0**, and −0.62 landed at −0.533 (error 0.087). Two of five weights
were annihilated and the two mid-sized ones each moved several percent — precisely the "bulk crushed
into the innermost levels" story, seen at the scale of five numbers. Notice also the asymmetry the code
encodes: the clamp is to [−(maxq+1), maxq] = [−4, 3], so the grid has 8 levels but one *more* on the
negative side; a symmetric-max scale with an asymmetric level count wastes a sliver of range but keeps
zero exactly representable, which matters because an exactly-zero weight must stay zero (a huge fraction
of the useful signal is small, and I do not want rounding to nudge near-zero weights off zero
systematically). This confirms the mapping does what I claimed and that the failure at 3-bit is the grid
being coarse relative to the *typical* weight, not any bug in the rounding.

The mechanism is that Δ is pinned by the largest weight while the trained distribution is heavy in the
middle: a weight at 10% of the channel max lands at round(0.1·4σ/1.33σ) = round(0.30) = 0, annihilated,
and one at 20% rounds to a single level. So the *bulk* of every channel is crushed into the innermost
two or three levels while the grid wastes resolution out at the tails — a sharp break, not a graceful
one.

One more piece of the weight-only story I want to reason through before I move on: *why* per-weight
rounding noise that looks small in SQNR terms turns into a 4× perplexity blow-up rather than a mild
degradation. The reason is depth. Each of the 32 blocks applies several quantized linear maps, and the
residual stream carries the accumulated perturbation forward. If each layer injects an output
perturbation whose relative size is ε and these are roughly independent, the norm of the accumulated
error across L layers grows like √L · ε in the benign case where the residual path keeps them from
compounding multiplicatively — but the attention and normalization are *not* linear, so a shifted
hidden state changes which tokens attend to which, and the errors correlate and amplify rather than
average out. At 8-bit, ε ≈ 0.009 and even a pessimistic ε√L over 32 layers is ~0.05 — invisible. At
3-bit per-channel, ε ≈ 0.385, and now even the benign √32 · 0.385 ≈ 2.2 says the hidden state has moved
by more than twice its own scale by the output, long before the nonlinear amplification is counted. A
model whose final-layer hidden state is off by 2× produces a next-token distribution that is nearly
unrelated to the true one, and perplexity — the exponential of the average negative log-likelihood —
punishes that sharply. That is the quantitative bridge from a per-weight rounding noise that looks small
in SQNR terms to a multiple-fold perplexity blow-up, and it tells me the fix cannot be *smaller* noise
per weight from a cleverer scale alone; it has to be noise that is *shaped* so that its effect on the
layer output cancels rather than accumulates.

There is a second, worse failure I should name now even though I am not attacking it yet, because it
sets up everything that comes later. Everything above is *weight-only*: I quantize W and leave the
activations X in FP16. If I also try to quantize the activations to low bits — which is what I would
need to run the matmul itself on integer tensor cores rather than just save weight memory — RTN does not
merely degrade, it detonates. Transformer activations are not heavy-tailed-but-bounded the way weights
are; they have a few *persistent channels* whose values are on the order of 100× everything else. Let me
run that through the same arithmetic to see how bad it is. Suppose the ordinary activations sit around
magnitude 1 and one channel spikes to 100. A single per-tensor activation grid at 4 bits sets M = 100,
so Δ = 100/7 = 14.3. An ordinary activation of magnitude 1 then rounds to round(1/14.3) = round(0.07) =
0 — it is quantized to *nothing*. Even a fairly strong ordinary activation at magnitude 5 rounds to
round(0.35) = 0. So essentially the entire ordinary signal, which is where the model's information
lives, is annihilated to preserve the single outlier channel. That is not a bug in my arithmetic; it is
the structural fact that the rest of this ladder exists to defeat, and it is why I expect W4A4 with plain
RTN to produce perplexity on the order of thousands — a fully broken model.

Before I commit, let me consider the alternatives I could reach for instead of plain magnitude-max RTN,
and rule them out on their own terms rather than by taste. One is a *clipped* / percentile scale: set Δ
from, say, the 99.9th percentile magnitude rather than the true max, sacrificing the extreme weights so
the bulk rounds finer. That genuinely helps the weight case a little — it trades a handful of clipped
outliers for better resolution on the majority — but it introduces a hyperparameter (which percentile?)
that I would have to tune per layer, which violates the "free, calibration-free floor" I am trying to
establish here; and on the activation side it does nothing about the fundamental problem, since the
outlier channels are not a few stray values I can clip but a persistent, structured feature I would be
throwing away. A second alternative is a nonuniform grid — say a logarithmic or k-means codebook fit to
each channel — which would match the heavy-middle weight distribution far better than a uniform grid.
But a nonuniform codebook cannot be multiplied through as a single scale Δ; dequantization becomes a
lookup, the matmul can no longer be a clean scaled-integer GEMM, and I lose exactly the hardware-friendly
regularity that was the entire point. So both tempting refinements fail against the constraint that the
floor must be free, uniform, and require no per-layer tuning. Plain magnitude-max RTN is the honest
baseline; its job is to fail informatively, not to be good.

The one lever I am *deliberately* leaving on the table is the calibration set — the few hundred
sequences of activations the harness can collect by running the model forward. RTN never looks at it: Δ
comes only from the weight tensor's own statistics, so RTN literally cannot know that a small weight
multiplied against a large, frequent activation matters more than a large weight against a dead channel.
That blindness is not an oversight, it is the definition of the floor — I want the simplest possible
scheme so that the *distance* between it and anything using calibration is a clean measurement of what
calibration is worth. When I read a large gap at 3-bit per-channel, I will know the gap is exactly the
value of the information RTN threw away, and that framing is what makes the next step well-posed rather
than a shot in the dark.

There is a small but real refinement I should fold into the g128 path specifically: an *affine* grid.
The symmetric scheme above forces the grid to be centered at zero, which is right for a whole channel
whose distribution is roughly symmetric. But a 128-wide group can easily be lopsided — its 128 weights
might run from −0.05 to +0.40, so a symmetric scale would set M = 0.40 and waste the entire negative half
of the grid on a range that holds almost nothing. The affine fix is to store both a scale and a
zero-point: Δ = (max − min)/(2^N − 1) using the full unsigned level count, and z = round(−min/Δ), so the
integer grid runs 0…2^N − 1 and dequantizes as Δ·(q − z). On the lopsided group this recovers the
negative-half resolution the symmetric grid threw away, and it is the natural pairing for g128 because
the extra zero-point is only 0.125 bits/weight more — I have already accepted per-group scale overhead,
so a per-group zero-point is the same order of cost for a real gain. I will keep the symmetric path for
whole-channel weights and use the affine path for groups; both are still plain round-clamp-dequantize
with no calibration.

The two quality axes are not equally sensitive, and that matters for reading the results. WikiText-2
perplexity is the exponential-of-loss quantity my depth argument says will blow up; zero-shot accuracy
averaged over a fixed suite is coarser, because the argmax over a few answer strings survives
perturbations that a full next-token distribution does not. So perplexity is the leading indicator — it
screams at 3-bit per-channel while accuracy sags more gently, and only a detonated model (W4A4) gets even
the argmax wrong. I keep that ordering so I do not later mistake a modest accuracy dip for a healthy
model.

So RTN is the right floor — free, calibration-free, genuinely fine at 8 bits — and the code is just the
magnitude-based step size and a clamp-and-round, per-channel or per-group (the full quantizer is in the
answer). The predictions follow from the SQNR arithmetic: excellent at 8 bits (≈41 dB, untouched),
acceptable at 4-bit g128 (within a few tenths of FP16), a several-fold blow-up at 3-bit *per-channel*
where the SQNR collapses toward 8 dB, and — the moment activations join at 4 bits — detonation to
perplexity in the thousands from the 14.3-vs-1 outlier arithmetic. Those numbers split the rest of the
ladder into two well-posed gaps. The weight-only cliff at 3-bit per-channel is a *structured* loss whose
errors are correlated through the layer output, so they should be *compensable* — and the lever sits in
the calibration activations RTN refuses to use. The activation detonation is a different beast, energy
concentrated in a few fixed coordinates, and it will need its own instrument.
