The z-loss did what label smoothing could not, and the numbers say so, so let me read them carefully
before deciding the next move. `val_loss` came down to 2.2926 from smoothing's 2.3377 — a drop of 0.0451
nats, which in perplexity is `exp(2.3377) = 10.36` falling to `exp(2.2926) = 9.90`, about a 4.4% relative
improvement on the very objective I am graded on. WikiText-2 recovered from 47.13 to 44.09 (down 3.04, a
6.4% relative move) and LAMBADA from 71.80 to 68.25 (down 3.55, 4.9%), so all three likelihood metrics
moved *together* in the same direction — that coherence is the signature of a real modeling gain rather
than a lucky wobble on one corpus. The downstream accuracies barely moved: arc_easy 54.04 → 54.55
(+0.51), hellaswag 33.63 → 33.85 (+0.22), piqa 63.71 → 63.00 (−0.71), winogrande 51.78 → 51.14 (−0.64) —
two up, two down, all inside what a single seed's noise can produce, net essentially flat. And the run
cost 20532 seconds against smoothing's 20526, six seconds more over five and a half hours, confirming
that the extra `logsumexp` per step was free as predicted. So the story is clean: the handle that
mattered on this short bfloat16 run was the absolute logit *level*, not the gap, and a small
squared-log-partition penalty that supplies the restoring force cross-entropy structurally lacks buys a
real likelihood gain with no downstream cost and no train/eval mismatch. The level fix was the right
diagnosis.

But read what z-loss actually *guarantees*, because that is where its ceiling is and where the next rung
lives, and I can put a number on the gap in the guarantee. The penalty `λ·(log Z)²` is a *soft average*
force: its gradient `2λ·log Z·p_j` pulls the level down only in proportion to how far the mean
log-partition has drifted, and that mean is taken across every valid position in the loss call — a
micro-batch of 64 sequences of length on the order of 512 packs on the order of `3·10⁴` positions into
one average. Now suppose a single one of those positions has an excited coordinate that drives its
`log Z` to, say, 20 while all the others sit near 0. Its contribution to the mean penalty is
`20² / 3·10⁴ ≈ 0.013`, and multiplied by `λ = 1e-4` that is about `1.3·10⁻⁶` added to the loss —
utterly invisible. The restoring gradient it feels is `2λ·log Z·p_j / N ≈ 2·10⁻⁴·20·p_j / 3·10⁴`, of
order `10⁻⁷·p_j`, essentially nothing. So the averaging dilutes any single spiked position by a factor of
tens of thousands: z-loss holds the level down *on average* and is structurally blind to an individual
coordinate that climbs into the dangerous regime at one position. The level is held; the individual values
are not bounded. That is the precise residual failure the 2.2926 leaves on the table, and it is what I go
after now.

So let me restate the loss-layer problem one more time, but now aimed at the values rather than the level.
For one position with logits `z` and target `y`, cross-entropy is `-z_y + log Σ_j exp(z_j)`, and I have
already established it has no finite minimizer — it decreases monotonically as the gap
`z_y - max_{j≠y} z_j` widens and reaches its infimum only at `+∞`. The objective is, literally, an
instruction to make the logits diverge. z-loss leaned on that orbit from one side (push the whole level
down); but the cleaner statement of what I want is structural: I want the logits that feed the exponential
to be *bounded*, every one of them, not just small on average. And whatever I do had better not scramble
which token the model prefers — softmax cares only about logit differences and the argmax sets the
prediction, so I want to tame the logits' *scale* without touching their *order*.

Why does scale matter at all, when softmax is shift-invariant? The same bfloat16 arithmetic that drove
the z-loss story, only now I want a per-value cure. bfloat16 keeps seven mantissa bits to float32's
twenty-three, about `2^16` coarser rounding, and the absolute roundoff grows with the magnitude of the
number because a fixed mantissa width spans each binade `[2^k, 2^{k+1})` with spacing `2^{k-7}`. The
logits go straight into an exponential, and `exp` turns an additive perturbation of its input into a
multiplicative one — `δ` in a logit becomes a factor `e^δ` on the unnormalized probability — so a roundoff
error that is negligible on a small logit becomes a real distortion of the probability once the logit is
large. The same concrete case as before makes it vivid: ten logits at 128 and one at 128.5. The value 128
sits in `[2^7, 2^8)` where the spacing is `2^{7-7} = 1`, so 128.5 is exactly halfway between the only two
representable neighbors, 128 and 129, and rounds away to 128; the target probability lurches from
`exp(0.5)/(exp(0.5)+10) = 0.142` to `1/11 = 0.091`, a 36% swing manufactured purely by roundoff. z-loss
attacks this by keeping the average level small, which helps, but a single coordinate that the penalty's
average did not happen to press on can still climb into the unfaithful regime. I want a guarantee on every
value.

Before I reach for anything fancy, let me reconsider the two handles I have already spent and confirm
neither is the value-bound I now want. The target attack — label smoothing — removed the incentive for
*one specific* gap to run away but put no ceiling on absolute magnitude; a model can sit at logits
`(1000, 990, 990, …)` with exactly the smoothed-optimal gaps while being numerically enormous, and the
2.3377 already showed it pays for distorting the evaluated objective. The level attack — z-loss — pushes
magnitudes down on average via a coefficient mixed into every step, but the dilution arithmetic I just did
shows it enforces no actual bound on any single logit and perturbs the loss value rather than transforming
the model's forward map. So I have done the target and the average level; the one thing left, and the one
that gives a *per-value* guarantee, is to transform the logit values themselves before the softmax. Bound
them structurally.

The crudest version of "bound them" is to clamp: `z ← clamp(z, -s, s)` before the softmax. That caps the
magnitude and kills the roundoff blow-up. But think about the gradient, because I am backpropagating
through this for thirteen thousand steps. `clamp` has derivative exactly 1 inside `(-s, s)` and exactly 0
outside. The moment a logit exceeds `s`, its gradient through the clamp is *exactly* zero — it gets no
learning signal through this loss-layer map, a coordinate the loss would otherwise move back down is stuck
behind a dead derivative — and there is a kink at `±s` where the function is non-differentiable and Adam's
per-coordinate statistics see a discontinuous slope. This is the same failure shape I avoided with z-loss:
hard constraints backfire where a smooth penalty stabilizes without the quality loss. So write down what I
actually want as three requirements and let them force the form. A transformation `g` applied to the
logits before the softmax that (1) bounds their range so they cannot run away or blow up the low-precision
exp, (2) is smooth with a nonzero gradient everywhere so no coordinate goes dead, and (3) is strictly
monotone so it never reorders the logits and the model's preferred token is preserved. Monotone and
bounded and smooth on the whole real line — that is a squashing function, an S-curve.

The canonical one is the hyperbolic tangent: `tanh: ℝ → (-1, 1)`, smooth, odd, strictly increasing,
`tanh(0) = 0`, derivative `1 - tanh²` which is 1 at the origin and decays smoothly to 0. Let me check that
it really keeps gradient alive where clamp kills it, with actual numbers. At an input three cap-scales out,
the tanh cap `s·tanh(z/s)` has slope `1 - tanh²(3) = 1 - 0.995² = 0.0099` — about 1% of full gradient,
small but nonzero; at two cap-scales out it is `1 - tanh²(2) = 1 - 0.964² = 0.071`, still 7% alive. So an
outlier logit stays connected to the task gradient at every finite input, and can always be pulled back
down, exactly where `clamp` would have severed it at a hard zero. That is requirement (2) verified rather
than asserted.

But `tanh` alone lands in `(-1, 1)`, far too tight — I do not want to crush every logit into a unit
interval, that destroys the useful spread. I want to cap at some scale `s` — say fifteen or thirty — not at
1. So scale the output: `s·tanh(·)`. And the inside matters. The naive `s·tanh(z)` has slope at the origin
`d/dz[s·tanh(z)] = s·(1 - tanh²(z)) = s` at `z = 0`, so small logits get *amplified* by `s` — that is
backwards; I want the map to be the *identity* on healthy logits and bend only the outliers. The fix is to
divide by `s` inside as well: `g(z) = s·tanh(z/s)`, whose slope `1 - tanh²(z/s)` is exactly 1 at `z = 0`
and decays to 0 for large `|z|`. For small `z`, `tanh(z/s) ≈ z/s` so `g(z) ≈ z` — identity on the normal
operating range — and for large `z`, `g(z) → ±s` — a smooth asymptotic cap. It guarantees `g(z) ∈ (-s, s)`
for every real `z` — the per-value bound z-loss could not give — while the smoothness keeps every
coordinate's gradient alive and the monotonicity keeps the argmax fixed.

Now I have a clean method — `z̃ = s·tanh(z/s)`, then cross-entropy on `z̃` — but I am running this transform
on a `(B·T, V)` tensor with `V` in the tens of thousands every step, with the cross-entropy right after,
so the cost and the intermediate tensors matter, and there is an exact reparameterization that folds more
cleanly. `tanh` is exp-based — `tanh(u) = 1 - 2/(e^{2u}+1)` — essentially a sigmoid in disguise, and the
identity `tanh(u) = 2σ(2u) - 1` is exact (verify:
`2σ(2u) - 1 = 2/(1+e^{-2u}) - 1 = (1 - e^{-2u})/(1 + e^{-2u}) = tanh(u)`). So a symmetric soft cap can be
rewritten entirely in sigmoid terms. Take the `s = 15` variant: `g(z) = 15·tanh(z/15) = 15·(2σ(2z/15) - 1)
= 30·σ(z/7.5) - 15`. Let me check the reparameterization at two points so I trust it and did not fumble a
constant: at `z = 7.5`, `15·tanh(0.5) = 15·0.4621 = 6.93` and `30·σ(1) - 15 = 30·0.7311 - 15 = 6.93`; at
`z = 15`, `15·tanh(1) = 15·0.7616 = 11.42` and `30·σ(2) - 15 = 30·0.8808 - 15 = 11.42`. They agree —
`15·tanh(z/15)` and `30·σ(z/7.5) - 15` are the *same function*, exactly, both with range `(-15, +15)`. Why
bother? Two reasons. First, the `-15` additive constant is invisible to the softmax —
`softmax(z + c·1) = softmax(z)` for any scalar `c`, the constant cancels in the normalization — so I can
drop it, and the whole thing collapses to a single sigmoid times a scale, `30·σ(z/7.5)`, which is
cross-entropy-equivalent to the symmetric 15-cap. Second, the sigmoid form fuses cleanly into a single
cap-plus-cross-entropy pass over the vocabulary without materializing a separate capped-logit tensor.

Pin the gradient in the sigmoid parameterization to confirm the shape survives. With `z̃ = A·σ(u)`,
`u = (z + B)/C` — `A` the output scale, `C` the input scale, `B` a shift — the derivative is
`dz̃/dz = (A/C)·σ(u)(1 - σ(u)) = (1/C)·z̃·(1 - z̃/A)`. A bell again: zero in both tails
(`σ(1-σ) → 0` as `u → ±∞`), peaked at `u = 0` with height `A/(4C)`. Same everywhere-positive,
vanishing-at-the-extremes profile as the tanh slope — outliers stay softly connected to the task gradient,
the central region keeps a smooth usable slope, nothing dies, no kink. `A` sets the cap height
(range `(0, A)`), `C` the steepness, `B` moves the inflection to `z = -B`.

So why shift `B` off zero and re-tune `A` away from the clean `(A, B, C) = (30, 0, 7.5)` that came straight
from the `s = 15` tanh? Because once I have accepted that the cap is a bounded, smooth, strictly-monotone
squash and that a global additive output constant washes out of the softmax while the *shape* of the
saturation matters, the exact constants are knobs for this particular regime — a short nanoGPT run, this
model size, this vocabulary, bfloat16. The load-bearing constraints are the qualitative three: bounded
range (so the exp is safe), strict monotonicity (so the token ranking is preserved), smooth positive
gradient for every finite input (so training stays healthy). The concrete loss-layer constants this task
uses are `A = 23, B = 5, C = 7.5`. Let me actually read off what that curve does, coordinate by
coordinate, because the constants only mean something once I see the map. Its range is `(0, 23)`; the
inflection is at `z = -5`, where `σ(0) = 0.5` gives output `11.5`, the midpoint of the range; the maximum
slope is `A/(4C) = 23/30 = 0.767`. Tabulating the map on a spread of raw logits: `z = -15 → 4.8`,
`z = -10 → 7.8`, `z = -5 → 11.5`, `z = 0 → 15.2`, `z = 5 → 18.2`, `z = 10 → 20.3`, `z = 15 → 21.5`. So a
healthy band of raw logits in `[-10, 10]` lands in capped `[7.8, 20.3]`, a spread of about 12.5, with the
positive (winning-token) logits getting the top of the range and the strongly-negative ones compressed
toward the floor. This is *not* pointwise identical to `15·tanh(z/15)`, nor that curve plus a
softmax-invisible constant — the shifted midpoint and smaller height make it a tuned, asymmetric sigmoid
member of the same bounded-monotone family, the modded-nanoGPT form `23·σ((z+5)/7.5)`. It is still strictly
increasing (argmax preserved), bounded by 23, and smooth (gradient everywhere).

The shift `B = 5` deserves a mechanistic reading rather than a shrug at "it is a tuned knob," because it
puts the inflection — the point of maximum slope, where the map is most nearly the identity and the
gradient is fattest — at raw logit `z = -5`, not at `0`, and that asymmetry is doing real work. Think about
how logits are actually distributed across a 50k vocabulary under a peaked softmax: for any given context
a handful of tokens are plausible and carry logits near the top, while the *overwhelming majority* of the
vocabulary is implausible and sits well below the winner, at negative logits. So the bulk of the
coordinates the cap must handle every single step live at negative `z`, and centering the high-gradient
region of the bell at `z = -5` lays the fattest part of the slope over that bulk — the many not-quite-zero
tail tokens whose relative ordering the model is still busy learning — while the few strongly positive
winning logits sit up in the gentle saturating shoulder where they are safely capped and only lightly
nudged. Read the tabulation that way and it lines up: the identity-like high-slope stretch covers roughly
`z ∈ [-10, 0]`, exactly the band where the informative tail lives, and heavy saturation sets in for
`z ≳ 5`, exactly where the winners are and where I *want* the cap to bite. An inflection at `0` would
instead spend the fattest gradient on logits near zero and start saturating the mildly-negative tail too
early, throttling the coordinates that carry the most tokens. The asymmetry is not arbitrary; it aligns
the map's most faithful region with the part of the logit distribution that is most populous and least
settled.

Two things I should actually verify about `A = 23` rather than take on faith: that the cap lands the exp in
the faithful bfloat16 regime, and that a range of 23 is wide enough to express the distributions the model
needs. For the first: the capped logits live in `(0, 23)`, so they sit in the binade `[16, 32) = [2^4, 2^5)`
at the top end, where the bfloat16 spacing is `2^{4-7} = 2^{-3} = 0.125`. The very same 0.5 gap that was
annihilated up at 128 — where the spacing was 1.0 — is now four full ulps wide and resolved cleanly. So
capping at 23 does not merely make the numbers smaller; it moves them into the part of the bfloat16 range
where the differences that carry the model's predictions actually survive the rounding. That is the whole
point, checked with the resolution arithmetic. For the second: the largest log-odds the softmax can now
express between any two tokens is the maximum capped gap, `A = 23`, so `p_y/p_k` is bounded by
`e^{23} ≈ 9.7·10⁹`. A model that wants to say the true token is ten billion times more likely than another
can still do so; only pathological certainty beyond that is forbidden — and that pathological certainty is
exactly the runaway I have been trying to prevent since rung one. So `A = 23` is comfortably large enough
for real language distributions and just small enough to keep the exp honest; the two requirements meet in
that neighborhood.

There is a clean way to see that this per-value cap *contains* the level fix z-loss gave me rather than
merely replacing it, and it is worth spelling out because it tells me not to stack the two. With every
capped logit bounded above by `A = 23`, the log-partition obeys the same log-sum-exp sandwich I used in
the z-loss rung: `log Z ≤ max_k z̃_k + log V ≤ 23 + log(50{,}257) = 23 + 10.8 = 33.8`, a hard ceiling that
holds at *every* position with no penalty term and no coefficient to tune. z-loss added
`λ·(log Z)²` to *discourage* a large `log Z` on average, and the dilution arithmetic showed that force
goes soft against a single spiked position; the soft cap instead makes an unbounded `log Z` *structurally
impossible*, position by position, because you cannot make the sum-exp of a set of numbers run away once
every number in the set is boxed below 23. So the value bound does not sit *beside* the level fix — it
dominates it: bounding every logit bounds their log-sum-exp for free, and it does so per position rather
than on the batch mean. That is the precise sense in which this is the more complete intervention, and it
is why I drop the explicit z-loss term rather than keep both. Stacking would be spending a coefficient to
regularize a quantity the cap already constrains structurally — regularizing `log Z` toward 0 on top of a
map that has already made `log Z` unable to exceed 33.8 — which buys nothing and adds a knob I would then
have to defend. One structural bound in the forward map is cleaner than a structural bound plus a soft
penalty chasing the same thing.

I have to make sure this is a faithful modeling loss, not a sneaky distortion, because the contract
forbids lowering the reported loss by distorting the distribution — exactly the line that cost label
smoothing its number. A temperature scaling `z ↦ z/τ` is a *linear* squeeze that uniformly flattens or
sharpens every gap and can mechanically move the loss without the model improving: at `τ < 1` it sharpens
every prediction and lowers cross-entropy on confident tokens for free, which would be a cheat. Is the
soft cap secretly doing that? Its maximum slope is `0.767`, strictly *below* 1, so locally the cap
*flattens* gaps — in its near-linear region it behaves like a temperature of `1/0.767 = 1.30`, softer than
plain logits, which if anything *raises* cross-entropy on a confident token rather than lowering it. So the
cap cannot be manufacturing a lower loss by sharpening; whatever likelihood it buys has to be earned
through the numerics and the healthier optimization, not through a distributional trick. And here is why it
can win where smoothing lost: the cap is a monotone, invertible map that is part of the model's forward
pass, so the model can *compensate* for the compression by learning larger raw logits — to realize a capped
gap of `g`, it produces raw logits `g^{-1}(g)`, and since `g` compresses, `g^{-1}` expands — whereas
smoothing moved the *target itself* off the data, a bias no amount of training can route around. The cross-
entropy is then computed honestly on the capped logits: it is the true negative log-likelihood of the model
whose output head is "linear layer followed by the soft cap." The cap is part of the model's forward map,
applied in both training and the eval forward pass, so the reported `val_loss` is the honest cross-entropy
of the capped-output model — no train/eval split needed, unlike smoothing.

The compression does carry a cost I should quantify honestly rather than wave past, because it attenuates
the gradient the task sends back to the raw logits. By the chain rule the gradient a raw logit `z`
actually receives is `(∂CE/∂z̃)·(dz̃/dz)`, and `dz̃/dz = (1/C)·z̃(1 - z̃/A)` peaks at `A/(4C) = 23/30 =
0.767` at the inflection and is *smaller* everywhere else, falling toward zero in the saturated tails. Put
numbers on the region that matters: a winning logit sitting out near the cap at `z̃ = 20` receives
`dz̃/dz = (1/7.5)·20·(1 - 20/23) = (1/7.5)·20·0.130 = 0.348`, under half the peak, and a very saturated one
at `z̃ = 22` gets `(1/7.5)·22·(1 - 22/23) = (1/7.5)·22·0.043 = 0.127`. So the winning token's logit — the
one the model most wants to drive — is exactly the one whose gradient is most throttled. Does that stall
learning? It does not, and the reason is the whole point of choosing a monotone map that lives *inside*
the forward pass. The model routes around the throttle by producing a *larger raw logit*: to realize a
given capped value it climbs further up the near-flat part of the curve, and the attenuated gradient is
just the price of holding a bounded, safe output while it does so. This is the crucial structural
difference from smoothing's failure. Smoothing moved the *target* off the data — a bias baked into the
objective that no amount of training can route around, which is why it cost the 2.3377. The cap leaves the
target the data and reshapes only the *forward map*, a monotone invertible reshaping the model can fully
compensate for by relearning its logit scale; so at convergence the cap costs expressiveness only in the
hardest-saturated corner, not everywhere, and it does not distort *what* is being modeled, only *how* the
head parameterizes it. The attenuated gradient slows the approach to the answer; it does not move the
answer.

One numerics detail: cast the logits to float32 before the sigmoid so the saturating exp inside it runs in
high precision — the whole point was numerical safety, so the exp-bearing part is done in the higher
precision — and keep `ignore_index = -1` for the packed-boundary positions. The full scaffold function is
in the answer.

So the delta from step 2: where z-loss held the *average* level down with a soft penalty that the dilution
arithmetic showed bounds no single logit, I now pass the logits through a smooth, bounded, strictly-
monotone sigmoid soft cap `23·σ((z+5)/7.5)` *inside the model's forward map* and compute plain cross-
entropy on the capped logits — a structural per-value bound rather than an average nudge, with the gradient
alive everywhere and the argmax untouched. Reading z-loss's shape, here is what I expect, falsifiably. The
primary `val_loss` should move *below* z-loss's 2.2926 — because a hard per-value bound is strictly safer
numerically than an average penalty, so the softmax is faithful everywhere and the optimizer trains on
undistorted gradients; I would expect a further step down off z-loss's 2.2926, of the same rough order
z-loss bought over smoothing, though how much of that improvement is real versus seed noise is
exactly what the `val_loss` table will decide and I will not pretend to the digit. The perplexities should
follow z-loss's recovery further — below its 44.09 / 68.25 — if the value-bound is a
real gain. The downstream accuracies are the genuine open question and where the risk lives: bounding
logits structurally could pinch a little expressiveness on tasks where confident margins help, so I would
watch hellaswag in particular for a possible small trade against the `val_loss` gain rather than assume a
uniform sweep. If `val_loss` improves but a downstream metric like hellaswag slips, that is the soft cap's
signature trade — a faithful but slightly squashed output head — and the honest summary of this ladder
would be that the value-bound is the strongest single loss-layer move available here, with z-loss the right
fallback when downstream margin matters more than the last hundredth of validation loss.
