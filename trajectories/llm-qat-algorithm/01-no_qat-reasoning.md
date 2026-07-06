The training-side algorithm is the whole point, but it bolts onto a fixed format — per-group symmetric
round-to-nearest — and the floor I have to start from is that format with *nothing* trained on top of
it: pure post-training quantization, the irreversible rounding applied once to the pretrained weights
and nothing else. Before I can reason about what training buys, I need to know exactly what the
rounding does on its own, because that is the error every later rung is trying to claw back. So the
first rung is deliberately the do-nothing control: zero training steps, materialize the integer model,
measure. Its value is precisely that it isolates the damage of the format from the contribution of any
algorithm. Everything I build afterward will be measured as a distance from this floor, so I want the
floor pinned down not just as a recipe but as a quantity I can predict.

Let me nail the irreversible step, since it is the substrate the whole ladder stands on. An integer
code `q` is, by itself, just an index; it only becomes a number because I attach a rule `q ↦ r`. The
first thing forced on me is the *shape* of that rule. A layer is a matrix multiply — sums of products
of weights and activations — and I want to do that accumulation on the codes and only convert back at
the end, so the products and sums of the codes must track the products and sums of the reals. The only
`g` for which "multiply the reals" reduces to "do integer work on the codes plus a cheap fix-up" is an
*affine* one: `r = A·q + B`. Anything nonlinear and `g(q1)·g(q2)` stops being expressible through
`q1·q2` plus corrections, and the whole reason for integers evaporates. So the map is affine — forced,
not chosen.

Now look at `B`. Convolutions and the attention/MLP plumbing pad and zero things; in the quantized
world that padding is some code, and whatever real value the code denotes is what actually gets added.
If real 0 is *not* exactly representable, every padded position contributes the same small nonzero
value — the same error, every time, which is a *bias*, the one kind of error that compounds across depth
rather than averaging out. The distinction is worth making quantitative, because it is why I single this
one error out. A zero-mean error of size `ε` injected independently at each of 24 layers accumulates in
the residual stream like a random walk, growing as `sqrt(24)·ε ≈ 5ε`; a *biased* error of the same size
`ε`, pointing the same way every layer, accumulates coherently as `24·ε`, nearly five times larger, and
it does not average out no matter how deep the model is or how much data I feed it. So a representable
real zero is not a nicety — it is the difference between an error that stays bounded across depth and one
that marches. That is enough to force the constraint. So I need real 0 to land exactly on a code. Reparameterize the same affine
map as `r = S·(q − Z)`: set `r = 0` at `q = Z` and `S` cancels, leaving `Z = −Z`, i.e. the constraint
is satisfied by construction for any `S`. So the correspondence is `r = S(q − Z)`: a positive real step
size `S` and an integer zero-point `Z` (the code that maps to real 0). But a trained weight tensor is
roughly zero-centered and symmetric, so the natural grid is centered at zero and `Z = 0` falls out,
collapsing the map to `r = S·q`. Let me count what that buys in the kernel, because it is the reason the
format earns its keep. A layer `y = Wx` costs `out·in` multiply-accumulates. In the general affine scheme
each product is `(q_w − Z_w)(q_x − Z_x) = q_w q_x − Z_w q_x − Z_x q_w + Z_w Z_x`; the core term `q_w q_x`
is the cheap integer accumulation I want, but the `Z_w q_x` term forces an extra reduction over the input
of `q_x` for every output — a whole second pass' worth of arithmetic — just to cancel the weight
zero-point. Setting `Z_w = 0` deletes that term outright, so the `O(out·in)` accumulation runs on small
integers and only a single per-output scalar `S1·S2/S3` survives in floating point. `Z = 0` is not
cosmetic; it removes an entire correction pass. That is the entire reason integer quantization is worth
doing, and it is why the format is what it is.

That leaves a single knob, the scale `S`. With no appetite for tuning the range against data — this is
the *no-calibration* control — the honest choice is the max-magnitude one: in the signed code range
`qmin = −2^{B−1}`, `qmax = 2^{B−1} − 1`, tie the scale to the positive endpoint, `S = max|w| / qmax`,
the smallest step that covers the weights without clipping. I should name the trade-off this resolves,
because it is the crux of the whole ladder: a *smaller* `S` would clip the range tighter than `max|w|`,
shrinking the rounding error on the bulk at the cost of clipping the few weights beyond the tighter
range to the rail — trading rounding error for clipping error, a balance that requires looking at the
data and optimizing. The max-abs scale is exactly the choice that needs *no* data: the smallest `S` that
covers both rails with no clipping. The extra negative code is simply outside the observed no-clip range.
And round-to-nearest, because for a one-shot conversion it minimizes each element's error,
`|error| ≤ S/2`, with residual variance about `S²/12` under the uniform-within-a-cell model. I
deliberately do *not* reach for stochastic rounding here even though it is pointwise unbiased: its
unbiasedness only earns its keep when the *same* value is rounded over and over, so a tiny per-step bias
accumulates linearly while the extra variance averages down. PTQ rounds each weight exactly once. There
is no accumulation to fight, so stochastic rounding would buy variance for nothing.

Before I settle on round-to-nearest with an affine grid I should at least name what I am giving up, so I
know it is the format and not an oversight. A non-uniform codebook — pick the four (or eight, or sixteen)
real values per group that best fit the weight histogram, k-means style — would clearly reconstruct the
weights with less error than a uniform grid, because it can put codes where the mass is. But it breaks
the one property the whole scheme is built to preserve: with a codebook, `r = LUT[q]` is not affine in
`q`, so the accumulation can no longer run as an integer matmul with a per-output rescale; every product
would need a lookup and the `O(out·in)` integer core evaporates, which was the entire justification for
quantizing at all. So the uniform affine grid is not the most accurate map — it is the most accurate map
*that keeps the matmul integer*, and that constraint is exactly why the substrate is what it is. The
freedom I have is in `S`, not in the shape of the grid.

The one real refinement, then, is to make the scale per-group rather than per-tensor. A single
tensor-wide `S` is wrecked by the two known one-shot failure modes, and both are worth a number so I know
the refinement is load-bearing and not decorative. Take the cross-channel range spread first. Suppose one
output channel has `max|w| = 1.0` and another, narrow, channel has `max|w| = 0.01` — a hundredfold spread,
which is entirely ordinary in a trained transformer. A single tensor scale is set by the largest,
`S = 1.0/qmax`; at INT4 that is `S = 0.143`. Now every weight in the narrow channel is at most `0.01`, so
`w/S ≤ 0.07`, which rounds to `0` — the *entire* narrow channel is annihilated to zero, zero codes used,
because its whole magnitude fits inside half of one tensor-wide cell. Per-group rescues it: the narrow
channel's own groups have `max|w_g| ≈ 0.01`, so their scale is `≈ 0.00143` and the weights are resolved
across the codes as they should be. The second mode is the mirror image: a lone outlier weight inflates
`max|w|` for the whole tensor and coarsens everyone else, so one stray large value in one channel drags
the grid coarser for all the millions of well-behaved weights that share the tensor scale. Partitioning
each row into contiguous blocks of `group_size = 128` columns, each with its own `max|w|` and its own
`S`, localizes both: an outlier only coarsens the 128 weights in its own group, and a narrow channel's
group gets a scale matched to its own magnitude. Let me price the overhead exactly, because "negligible"
should be a number. With `in = 2048` there are 16 groups per row, so I store one fp16 scale per 128
int-`B` weights; that is `16/128 = 0.125` extra bits per weight, so the *effective* bit-width is `B +
0.125`. At INT2 that is `2.125` bits — the grouping buys a per-128 localization of both failure modes for
a one-sixteenth-bit surcharge. Choosing 128 rather than a finer 64 (0.25 extra bits) or a coarser 256
(0.0625) is the usual knee: 128 is fine enough to isolate a single outlier to a small block yet coarse
enough that the scale storage stays a rounding error on the budget. So the irreversible step is fixed:
per-group symmetric round-to-nearest, `S_g = max|w_g| / qmax`, with a `1e-12` floor on `max|w_g|` so an
all-zero group does not divide by zero — and that floor is benign, since an all-zero group quantizes to
all zeros regardless of `S`, so the arbitrary large `S` the floor produces changes nothing.

There is a second inefficiency baked into the symmetric-signed grid that I should account for honestly,
because it gets worse exactly where I can least afford it. The signed range is asymmetric by one code:
`qmin = −2^{B−1}`, `qmax = 2^{B−1}−1`, so there is one extra code on the negative side. But the max-abs
scale ties `S` to the *positive* endpoint, `S = max|w_g|/qmax`, and the weights live in `[−max|w_g|,
max|w_g|]`, so the most negative reachable reconstruction point is `−qmax·S = −max|w_g|`, never
`−2^{B−1}·S`. That most-negative code is structurally unreachable — one code per group is simply wasted.
At INT4 that is 1 of 16 codes, a 6% loss of an already-fine grid; at INT2 it is 1 of 4, a full 25% of
the codebook thrown away, which is why the effective INT2 grid is three levels and not four. The waste
scales the wrong way — it bites hardest at the bit-width that can least tolerate it — and there is
nothing a *blind* scheme can do about it, because using that fourth code would mean shifting the grid
off-center, which needs an asymmetric zero-point I already argued away for the sake of the clean integer
kernel. It is one more thing only a data-aware, trained method could reclaim, and one more reason to
expect the INT2 floor to be brutal.

Now I want to turn the `S/2` floor from a bound into a number, because a control is only useful if I can
predict its failure and read the later rungs against that prediction. Take a group and model its weights
as roughly zero-centered with per-group standard deviation `σ`. The max of 128 magnitudes of such a
distribution sits around three-and-a-third `σ` — `max|w_g| ≈ sqrt(2 ln 256)·σ ≈ 3.3σ` — so the step is
`S ≈ 3.3σ / qmax`. Under the uniform-in-cell model the per-weight quantization error has RMS `S/sqrt(12)
≈ 0.95σ / qmax`. Divide by the signal RMS `σ` and the *relative* perturbation to a typical weight is
just `0.95 / qmax`, independent of `σ`, a clean function of bit-width alone. At INT4, `qmax = 7`: about
`0.14`, a 14% relative jitter on each weight. At INT3, `qmax = 3`: about `0.32`. At INT2, `qmax = 1`:
about `0.95` — the rounding noise is essentially as large as the weight it perturbs. That last number is
the whole story of why two bits is a different regime and not merely a harder version of four: at INT4 I
am adding a small nudge to each weight; at INT2 I am replacing each weight with something almost
uncorrelated with it.

Before I trust that `0.95/qmax` relative-error law, let me check it against a limit I already know.
As `B` grows the grid should vanish into the reals: `qmax = 2^{B−1}−1` grows exponentially, so
`0.95/qmax` falls to zero and the reconstruction converges to the fp weights, degradation `→ 0` — the
format degrades gracefully into full precision, which is the correct behavior and a sign the algebra is
not backwards. Run it the other way and it also lands where it must: at INT8, `qmax = 127`, relative
error `≈ 0.75%`, which is the sub-one-percent weight jitter the old 8-bit inference work relied on to
ship models with negligible loss — so the same formula reproduces the regime that is known to work
before it predicts the regime that fails. The three bit-widths I am handed sit on the steep part of that
curve: `0.14, 0.32, 0.95` for `qmax = 7, 3, 1`, each roughly doubling the last, so I should expect the
*degradation* to grow faster than linearly across INT4 → INT3 → INT2, not in even steps — the gaps
between the three should widen, not stay constant.

I should be honest that the uniform-in-cell model is *optimistic* at two bits, which only sharpens the
conclusion. With the no-clip max-abs scale the in-range codes a weight can actually land on are
`{qmin,…,qmax}` intersected with `[−qmax, qmax]`, i.e. the reconstruction points span `−qmax·S` to
`qmax·S` in steps of `S`. At INT4 that is 15 usable levels; at INT3, 7; at INT2 the weights live in
`[−max|w_g|, max|w_g|] = [−S, S]`, so the only reachable reconstruction points are `{−S, 0, +S}` — three
levels, the `−2` code never touched. Three levels for 128 weights is not "coarse quantization," it is a
sign-and-zero sketch. The uniform-cell variance formula assumes many cells; at three levels it
understates the damage, so `0.95` relative error is a floor on the floor.

Let me trace one concrete group at INT2 so the arithmetic is not abstract. Say a group's weights are
`[0.90, −0.70, 0.20, −0.15, 0.05, …]` with the outlier `0.90` setting the range. Then `S = 0.90/1 =
0.90`. Dividing through: `0.90/S = 1.00 → 1`, `−0.70/S = −0.78 → −1`, `0.20/S = 0.22 → 0`, `−0.15/S =
−0.17 → 0`, `0.05/S = 0.06 → 0`. Multiply back by `S`: the group becomes `[0.90, −0.90, 0, 0, 0, …]`.
The outlier survives, one mid-weight is dragged from `−0.70` to `−0.90` (a 29% error), and every weight
below half the group max is annihilated to zero. This is what "three levels" means in practice: the
grid keeps the sign of the two largest magnitudes and erases the rest. There is no `S` that fixes this
without looking at the data, because the outlier `0.90` is what forced `S = 0.90` in the first place —
the max-abs rule is structurally hostage to the largest weight in each group.

That is the wall I will be staring at for the rest of the ladder. This conversion is *blind*: it looks
only at the weights, never at the data or the loss, and once it rounds, the error is frozen. Grouping
localizes the outlier and range-spread damage but does nothing to the underlying coarseness — within a
group of 128 weights snapped onto three levels, most weights are still thrown away. There is no
mechanism in this scheme to *repair* the error: I round once and freeze. And crucially, the two levers
that could help — moving the weights to configurations that round cleanly, and moving the *levels* to
sit where the weights are — both require the loss, gradients, and optimization that this control
forbids. That is exactly why every later rung exists. This rung's only job is to measure how far down
the hole starts.

So this rung's edit is the one that makes the harness do nothing. The scaffold default fakes-quant in
the forward and would, if I left it, run a 500-step finetune through the straight-through estimator —
but that is already the next rung's idea, not this one. The point of the PTQ control is *no training at
all*, so I set `num_steps = 0` and `learning_rate = 0`, which turns the AdamW loop into a no-op and
leaves the pretrained fp32 weights untouched. Then the fixed post-step does the only real work: it calls
the no-grad `quantize_dequantize_weight` once on every wrapped linear, snapping the fp32 weights to their
per-group integer grid, and evaluates. Because training is off, the wrapper's forward does not need to
fake-quant inside any loop — by eval time the weight has already been replaced by its dequantized
version, so the wrapper just calls a plain linear on the already-quantized weight. This is the one place
the wrapper diverges from the scaffold default: the default's `forward` applies `fake_quantize_weight`
every call (it expects to be trained through), but the PTQ control's `forward` is the bare `F.linear` on
`self.linear.weight`. The differentiable `fake_quantize_weight` is still present and correct — it just is
never exercised, because zero gradients are ever taken. And I keep the deliberate exclusions, each of which is an isolation, not a
concession. Activations stay full precision because this is a weight-only study: quantizing activations
adds a *second*, input-dependent error source — the scale would have to be recomputed per token from the
running activation range, a moving target that mixes into the weight error and makes it impossible to say
which of the two is responsible for a perplexity change. Holding activations at fp32 keeps the causal
attribution clean: whatever moves across the ladder is the weight grid and the training against it,
nothing else. The LM head (`embed_out` for this GPTNeoX backbone) is restored to a plain Linear for a
sharper reason — it maps the 2048-dim residual into the ~50k-way vocabulary logits, and a coarse grid
there corrupts every token's score directly, with no downstream layer to absorb the error; the
accuracy-per-bit there is far worse than in an interior projection, so it is left full precision by
construction. The full scaffold module is in the answer.

What must this floor do, and why run it? The relative-error model already draws the shape, and I want to
commit to it falsifiably against `wikitext2_ppl` and `degradation`. At INT4 the grid is fine enough — 15
in-range levels, ~14% relative jitter — that even blind RTN should leave perplexity only modestly above
the `fp16_ppl = 13.2033` baseline; this is the regime the older 8-bit-style inference work lived in, and
the damage is real but survivable, so I expect a small positive `degradation`, a few points of PPL at
most. At INT3 — 7 levels, ~32% relative jitter — the floor climbs sharply and I expect a clear
degradation, perplexity well into the tens, comfortably above baseline but still a recognizable model.
At INT2 — three levels, ~95% relative jitter, the rounding noise as large as the weights — I expect the
model to be effectively destroyed: every weight smashed onto that grid is a perturbation so large that
the composed function bears little resemblance to the trained one, and perplexity should blow up by
orders of magnitude, not points. The extreme version of the prediction is worth stating because it is
checkable: if each layer's output is roughly decorrelated from the trained one, the next-token
distribution should degrade toward — and possibly past — the uninformative uniform baseline, which for
this backbone's ~50k vocabulary sits near perplexity 50000; a five-digit INT2 perplexity would confirm
the composition has collapsed rather than merely degraded. That collapse is the entire motivation for
the ladder, and it is worth being explicit about what the measurement will let me *do*, since a control's
whole worth is in the comparisons it enables. Every later rung shares this rung's format, evaluation
harness, and QDQ exactly; the only thing that changes is what happens to the weights before the round. So
any rung's improvement over this floor is, by construction, attributable to that one difference and
nothing about the format — the format's contribution is now a fixed, measured constant I can subtract.
The diagnosis is already pointed: this is not a learning problem and not a tuning problem —
it is a *no-repair* problem. The grid is fixed and the rounding is irreversible, so the only leverage is
to move the weights before I round, which means turning training back on. The very next rung does exactly
that, with the cheapest possible mechanism for sending a gradient through the round: the straight-through
estimator the scaffold default already carries.
