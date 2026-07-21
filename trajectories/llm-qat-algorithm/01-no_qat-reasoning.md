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
value — a *bias*, the one kind of error that compounds across depth rather than averaging out. That is
why I single this error out. A zero-mean error of size `ε` injected independently at each of 24 layers
accumulates in the residual stream like a random walk, growing as `sqrt(24)·ε ≈ 5ε`; a *biased* error
of the same size pointing the same way every layer accumulates coherently as `24·ε`, nearly five times
larger, and it does not average out no matter how deep the model is or how much data I feed it. So a
representable real zero is the difference between an error that stays bounded across depth and one that
marches. Reparameterize the affine map as `r = S·(q − Z)`: set `r = 0` at `q = Z` and the constraint
is satisfied for any `S`. A trained weight tensor is roughly zero-centered and symmetric, so the
natural grid is centered at zero, `Z = 0` falls out, and the map collapses to `r = S·q`.

That collapse is where the format earns its keep in the kernel, so it is worth counting. In the general
affine scheme each product is `(q_w − Z_w)(q_x − Z_x) = q_w q_x − Z_w q_x − Z_x q_w + Z_w Z_x`; the
core `q_w q_x` is the cheap integer accumulation, but the `Z_w q_x` term forces an extra reduction over
the input of `q_x` for every output — a whole second pass' worth of arithmetic — just to cancel the
weight zero-point. Setting `Z_w = 0` deletes that term outright, so the `O(out·in)` accumulation runs
on small integers and only a single per-output scalar `S1·S2/S3` survives in floating point. `Z = 0` is
not cosmetic; it removes an entire correction pass, and that is the reason integer quantization is
worth doing at all.

That leaves a single knob, the scale `S`. With no appetite for tuning the range against data — this is
the *no-calibration* control — the honest choice is the max-magnitude one: in the signed code range
`qmin = −2^{B−1}`, `qmax = 2^{B−1} − 1`, tie the scale to the positive endpoint, `S = max|w| / qmax`,
the smallest step that covers the weights without clipping. This is the crux the whole ladder turns on:
a *smaller* `S` would round the bulk more finely at the cost of clipping the few weights beyond the
tighter range to the rail — trading rounding error for clipping error, a balance that requires looking
at the data. The max-abs scale is exactly the choice that needs *no* data. And round-to-nearest,
because for a one-shot conversion it minimizes each element's error, `|error| ≤ S/2`, with residual
variance about `S²/12` under the uniform-within-a-cell model. I deliberately do *not* reach for
stochastic rounding here even though it is pointwise unbiased: its unbiasedness only earns its keep when
the *same* value is rounded over and over, so a tiny per-step bias accumulates linearly while the extra
variance averages down. PTQ rounds each weight exactly once — no accumulation to fight — so stochastic
rounding would buy variance for nothing.

I should name what the uniform affine grid gives up, so I know it is the format and not an oversight. A
non-uniform codebook — pick the four (or eight, or sixteen) real values per group that best fit the
weight histogram, k-means style — would reconstruct the weights with less error, because it puts codes
where the mass is. But `r = LUT[q]` is not affine in `q`, so the accumulation can no longer run as an
integer matmul with a per-output rescale; every product would need a lookup and the `O(out·in)` integer
core evaporates. So the uniform affine grid is the most accurate map *that keeps the matmul integer*,
and that constraint is exactly why the substrate is what it is. My freedom is in `S`, not in the shape
of the grid.

The one real refinement is to make the scale per-group rather than per-tensor, and both failure modes a
single tensor-wide `S` invites are worth a number so I know the refinement is load-bearing. Take
cross-channel range spread first. One output channel has `max|w| = 1.0`, another narrow channel has
`max|w| = 0.01` — a hundredfold spread, entirely ordinary in a trained transformer. A single tensor
scale is set by the largest, `S = 1.0/qmax`; at INT4 that is `S = 0.143`, so every weight in the narrow
channel has `w/S ≤ 0.07` and rounds to `0` — the *entire* narrow channel annihilated, because its whole
magnitude fits inside half of one tensor-wide cell. The second mode is the mirror image: a lone outlier
inflates `max|w|` for the whole tensor and coarsens the grid for all the millions of well-behaved
weights sharing the scale. Partitioning each row into contiguous blocks of `group_size = 128` columns,
each with its own `max|w|` and `S`, localizes both: an outlier only coarsens the 128 weights in its own
group, and a narrow channel's group gets a scale matched to its own magnitude. The overhead is a number
too: with `in = 2048` there are 16 groups per row, one fp16 scale per 128 int-`B` weights, i.e.
`16/128 = 0.125` extra bits per weight, so the effective width is `B + 0.125` — at INT2 that is `2.125`
bits. Choosing 128 over a finer 64 (0.25 extra bits) or a coarser 256 (0.0625) is the usual knee: fine
enough to isolate a single outlier to a small block, coarse enough that the scale storage stays a
rounding error on the budget. So the irreversible step is fixed: per-group symmetric round-to-nearest,
`S_g = max|w_g| / qmax`, with a `1e-12` floor on `max|w_g|` so an all-zero group does not divide by zero
— a benign floor, since an all-zero group quantizes to all zeros regardless of `S`.

There is a second inefficiency baked into the symmetric-signed grid, and it gets worse exactly where I
can least afford it. The signed range has one extra code on the negative side (`qmin = −2^{B−1}`,
`qmax = 2^{B−1}−1`), but the max-abs scale ties `S` to the *positive* endpoint, so the most negative
reachable reconstruction point is `−qmax·S = −max|w_g|`, never `−2^{B−1}·S`. That most-negative code is
structurally unreachable — one code per group wasted. At INT4 that is 1 of 16, a 6% loss; at INT2 it is
1 of 4, a full 25% of the codebook thrown away, which is why the effective INT2 grid is three levels
and not four. The waste bites hardest at the bit-width that can least tolerate it, and there is nothing
a *blind* scheme can do about it — using that fourth code means shifting the grid off-center, which
needs the asymmetric zero-point I already argued away for the clean integer kernel. One more thing only
a data-aware, trained method could reclaim, and one more reason to expect the INT2 floor to be brutal.

Now to turn the `S/2` floor from a bound into a number, so I can predict this control's failure and read
the later rungs against the prediction. Model a group's weights as zero-centered with per-group standard
deviation `σ`. The max of 128 magnitudes sits around `max|w_g| ≈ sqrt(2 ln 256)·σ ≈ 3.3σ`, so the step
is `S ≈ 3.3σ / qmax`. Under the uniform-in-cell model the per-weight error has RMS `S/sqrt(12) ≈ 0.95σ /
qmax`; divide by signal RMS `σ` and the *relative* perturbation to a typical weight is `0.95 / qmax`,
independent of `σ` — a clean function of bit-width alone. At INT4, `qmax = 7`: about `0.14`, a 14%
relative jitter. At INT3, `qmax = 3`: about `0.32`. At INT2, `qmax = 1`: about `0.95` — the rounding
noise is essentially as large as the weight it perturbs. That last number is the whole story of why two
bits is a different regime and not a harder version of four: at INT4 I add a small nudge to each weight;
at INT2 I replace each weight with something almost uncorrelated with it. As `B` grows the law also
lands where it must — `0.95/qmax → 0`, degradation `→ 0`, the format degrading gracefully into full
precision — and at INT8 (`qmax = 127`) it gives `≈ 0.75%` relative error, the sub-one-percent jitter the
old 8-bit inference work relied on to ship models with negligible loss. The three bit-widths I am handed
sit on the steep part of that curve, `0.14 → 0.32 → 0.95` roughly doubling each step, so I should expect
degradation to grow *faster* than linearly across INT4 → INT3 → INT2, the gaps widening rather than
staying constant.

The uniform-in-cell model is *optimistic* at two bits, which only sharpens that. With the no-clip
max-abs scale the reachable reconstruction points at INT2 are just `{−S, 0, +S}` — three levels, the
`−2` code never touched. Three levels for 128 weights is not "coarse quantization," it is a
sign-and-zero sketch, and the uniform-cell variance formula assumes many cells, so at three levels it
*understates* the damage: `0.95` relative error is a floor on the floor. One concrete group makes it
unabstract. Say the weights are `[0.90, −0.70, 0.20, −0.15, 0.05, …]` with the outlier `0.90` setting
the range, so `S = 0.90`. Dividing through: `0.90 → 1`, `−0.70/S = −0.78 → −1`, `0.20/S = 0.22 → 0`,
`−0.15 → 0`, `0.05 → 0`. Multiplying back: the group becomes `[0.90, −0.90, 0, 0, 0, …]`. The outlier
survives, one mid-weight is dragged from `−0.70` to `−0.90` (a 29% error), and every weight below half
the group max is annihilated to zero. There is no `S` that fixes this without looking at the data,
because the outlier `0.90` is what forced `S = 0.90` in the first place — the max-abs rule is
structurally hostage to the largest weight in each group.

That is the wall I will be staring at for the rest of the ladder. This conversion is *blind*: it looks
only at the weights, never at the data or the loss, and once it rounds, the error is frozen. Grouping
localizes the outlier and range-spread damage but does nothing to the underlying coarseness. There is no
mechanism here to *repair* the error — this is not a learning problem or a tuning problem, it is a
*no-repair* problem — and the two levers that could help, moving the weights to configurations that
round cleanly and moving the *levels* to sit where the weights are, both require the loss, gradients,
and optimization this control forbids. That is exactly why every later rung exists. This rung's only job
is to measure how far down the hole starts.

So this rung's edit makes the harness do nothing. The scaffold default fake-quants in the forward and
would, if left alone, run a 500-step straight-through finetune — but that is already the next rung's
idea. The point of the PTQ control is *no training at all*, so I set `num_steps = 0` and
`learning_rate = 0`, turning the AdamW loop into a no-op and leaving the pretrained fp32 weights
untouched. The fixed post-step then does the only real work: it calls the no-grad
`quantize_dequantize_weight` once on every wrapped linear, snapping the fp32 weights to their per-group
integer grid, and evaluates. Because training is off, the wrapper's forward does not need to fake-quant
inside any loop — by eval the weight has already been replaced by its dequantized version — so the
wrapper is the bare `F.linear` on `self.linear.weight`. This is the one place the wrapper diverges from
the scaffold default, whose `forward` applies `fake_quantize_weight` every call. The differentiable
`fake_quantize_weight` is still present and correct; it is simply never exercised, since zero gradients
are taken. I keep the deliberate exclusions, each an isolation rather than a concession. Activations stay
full precision because this is a weight-only study: quantizing them adds a second, input-dependent error
source whose scale would have to be recomputed per token, mixing into the weight error and destroying
the clean causal attribution — whatever moves across the ladder should be the weight grid and nothing
else. The LM head (`embed_out` for this GPTNeoX backbone) is restored to a plain Linear for a sharper
reason: it maps the residual into ~50k-way logits, and a coarse grid there corrupts every token's score
directly, with no downstream layer to absorb the error, so its accuracy-per-bit is far worse than an
interior projection's. The full scaffold module is in the answer.

So what will this floor do, falsifiably, against `wikitext2_ppl` and `degradation`? At INT4 the grid is
fine — 15 in-range levels, ~14% jitter, the regime the old 8-bit inference work lived in — so blind RTN
should leave perplexity only modestly above the `fp16_ppl = 13.2033` baseline, a few points at most. At
INT3 — 7 levels, ~32% jitter — the floor climbs sharply, perplexity well into the tens, comfortably
above baseline but still a recognizable model. At INT2 — three levels, ~95% jitter — I expect the model
effectively destroyed, perplexity up by orders of magnitude, not points: if each layer's output is
roughly decorrelated from the trained one, the next-token distribution should degrade toward or past the
uninformative uniform baseline, which for this backbone's ~50k vocabulary sits near perplexity 50000, so
a five-digit INT2 perplexity would confirm the composition has collapsed rather than merely degraded.
That collapse is the entire motivation for the ladder. And the measurement's worth is in the comparisons
it unlocks: every later rung shares this rung's format, harness, and QDQ exactly, changing only what
happens to the weights before the round, so any improvement over this floor is attributable to that one
difference — the format's contribution is now a fixed, measured constant I can subtract. The very next
rung turns training back on, with the cheapest mechanism for sending a gradient through the round: the
straight-through estimator the scaffold default already carries.
