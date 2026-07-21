The z-loss did what label smoothing could not, and the numbers say so. `val_loss` came down to 2.2926 from
smoothing's 2.3377 — a drop of 0.0451 nats, `exp(2.3377) = 10.36` falling to `exp(2.2926) = 9.90`, about a
4.4% relative improvement on the objective I am graded on. WikiText-2 recovered from 47.13 to 44.09 (6.4%)
and LAMBADA from 71.80 to 68.25 (4.9%), all three likelihood metrics moving *together* — the signature of a
real modeling gain rather than a lucky wobble on one corpus. Downstream barely moved: arc_easy 54.04 →
54.55, hellaswag 33.63 → 33.85, piqa 63.71 → 63.00, winogrande 51.78 → 51.14 — two up, two down, net flat
within single-seed noise. And at 20532 seconds against smoothing's 20526 the extra `logsumexp` was free as
predicted. So the level, not the gap, was the handle that mattered on this short bfloat16 run.

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

The cleaner statement of what I want is structural: the logits that feed the exponential should be
*bounded*, every one of them, not just small on average — and without scrambling which token the model
prefers, since softmax reads only differences and the argmax sets the prediction. So tame the logits'
*scale* without touching their *order*. Scale matters despite shift-invariance for the same bfloat16
reason as the z-loss rung — absolute roundoff grows with magnitude, and `exp` turns it into a relative
error on the softmax weight, the ten-logits-at-128-plus-one-at-128.5 case that manufactured a 36% swing —
but z-loss only pressed the average down, so a single coordinate it did not happen to press on can still
climb into the unfaithful regime. The target attack (smoothing) put no ceiling on absolute magnitude at
all and paid for distorting the objective; the level attack (z-loss) bounds no single logit. Target and
average level are spent; what is left, and the one thing that gives a *per-value* guarantee, is to
transform the logit values themselves before the softmax. Bound them structurally.

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
derivative `1 - tanh²`, which is 1 at the origin and decays smoothly to 0. Where `clamp` severs an outlier
at a hard zero, `tanh` keeps it connected: three cap-scales out the slope is `1 - tanh²(3) = 0.0099`, about
1% of full gradient, and two out it is `0.071` — small but never zero, so an outlier can always be pulled
back down. Requirement (2), met.

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

Now I have a clean method — `z̃ = s·tanh(z/s)`, then cross-entropy on `z̃` — but there is an exact
reparameterization into sigmoid form that folds more cleanly into the cross-entropy pass. The identity
`tanh(u) = 2σ(2u) - 1` is exact, so the `s = 15` variant becomes
`g(z) = 15·tanh(z/15) = 30·σ(z/7.5) - 15`, the same function on range `(-15, +15)`. The `-15` additive
constant is invisible to the softmax (`softmax(z + c·1) = softmax(z)`), so I can drop it, collapsing the
symmetric 15-cap to a single sigmoid times a scale, `30·σ(z/7.5)`, which fuses into one
cap-plus-cross-entropy pass without a separate capped-logit tensor.

In the general sigmoid parameterization `z̃ = A·σ(u)`, `u = (z + B)/C`, the derivative is
`dz̃/dz = (1/C)·z̃·(1 - z̃/A)` — a bell, zero in both tails, peaked at `u = 0` with height `A/(4C)`: the
same everywhere-positive, vanishing-at-the-extremes profile as the tanh slope, no dead coordinate, no kink.
`A` sets the cap height (range `(0, A)`), `C` the steepness, `B` moves the inflection to `z = -B`.

So why shift `B` off zero and re-tune `A` away from the clean `(A, B, C) = (30, 0, 7.5)` the `s = 15` tanh
handed me? The load-bearing constraints are only the qualitative three — bounded range, strict
monotonicity, smooth positive gradient everywhere; within those, the exact constants are knobs for this
regime (this model size, vocabulary, bfloat16). The values I take are `A = 23, B = 5, C = 7.5`, and the
constants mean something only once I read off the curve. Its range is `(0, 23)`; the
inflection is at `z = -5`, where `σ(0) = 0.5` gives output `11.5`, the midpoint of the range; the maximum
slope is `A/(4C) = 23/30 = 0.767`. Tabulating the map on a spread of raw logits: `z = -15 → 4.8`,
`z = -10 → 7.8`, `z = -5 → 11.5`, `z = 0 → 15.2`, `z = 5 → 18.2`, `z = 10 → 20.3`, `z = 15 → 21.5`. So a
healthy band of raw logits in `[-10, 10]` lands in capped `[7.8, 20.3]`, a spread of about 12.5, with the
positive (winning-token) logits getting the top of the range and the strongly-negative ones compressed
toward the floor. This is *not* pointwise identical to `15·tanh(z/15)`, nor that curve plus a
softmax-invisible constant — the shifted midpoint and smaller height make it a tuned, asymmetric sigmoid
member of the same bounded-monotone family, the modded-nanoGPT form `23·σ((z+5)/7.5)`. It is still strictly
increasing (argmax preserved), bounded by 23, and smooth (gradient everywhere).

The shift `B = 5` is doing real work: it puts the inflection — the fattest-gradient, most nearly identity
point — at raw logit `z = -5` rather than `0`. Under a peaked softmax over 50k tokens, a handful of
plausible tokens sit at the top and the overwhelming majority sit at negative logits, so the bulk of the
coordinates the cap handles every step live at negative `z`. Centering the high-gradient region there lays
the fattest slope over the many not-quite-zero tail tokens whose ordering the model is still learning
(`z ∈ [-10, 0]`), while the few winning logits sit up in the saturating shoulder where they are safely
capped (`z ≳ 5`). An inflection at `0` would throttle the mildly-negative tail too early. The asymmetry
aligns the map's most faithful region with the most populous, least settled part of the distribution.

Two things about `A = 23` are worth checking. It lands the exp in the faithful bfloat16 regime: capped
logits in `(0, 23)` sit at worst in the binade `[16, 32)`, where the spacing is `2^{4-7} = 0.125`, so the
same 0.5 gap that was annihilated at 128 (spacing 1.0) is now four full ulps wide and resolved cleanly. And
a range of 23 is wide enough: the largest log-odds it can express between two tokens is `A = 23`, bounding
`p_y/p_k` by `e^{23} ≈ 9.7·10⁹` — the model can still say the true token is ten billion times more likely,
and only certainty beyond that (the runaway I have chased since rung one) is forbidden. Large enough for
real language, small enough to keep the exp honest.

This per-value cap *contains* the level fix rather than sitting beside it, which is why I drop the explicit
z-loss term instead of stacking both. With every capped logit bounded above by `A = 23`, the same
log-sum-exp sandwich gives `log Z ≤ 23 + log V = 33.8` at *every* position, with no penalty term and no
coefficient — where z-loss only discouraged a large `log Z` on average and went soft against a single
spiked position, the cap makes an unbounded `log Z` structurally impossible position by position. Stacking
a `λ·(log Z)²` penalty on top would regularize a quantity the map already constrains; it buys nothing and
adds a knob to defend.

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

The compression does attenuate the gradient the task sends to the raw logits, and it is worth pricing. The
raw logit receives `(∂CE/∂z̃)·(dz̃/dz)`, and `dz̃/dz` peaks at `0.767` and falls off in the saturated tails:
a winning logit near the cap at `z̃ = 20` receives `(1/7.5)·20·(1 - 20/23) = 0.348`, under half the peak,
and one at `z̃ = 22` gets `0.127`. So the winning token's logit — the one the model most wants to drive — is
exactly the one most throttled. That slows the approach but does not stall it: the model climbs further up
the near-flat curve to realize the same capped value, the attenuated gradient being the price of a bounded,
safe output. The cost is confined to the hardest-saturated corner, not spread across the whole objective
the way smoothing's off-data bias was.

One numerics detail: cast the logits to float32 before the sigmoid so the saturating exp inside it runs in
high precision — the whole point was numerical safety, so the exp-bearing part is done in the higher
precision — and keep `ignore_index = -1` for the packed-boundary positions. The full scaffold function is
in the answer.

So where z-loss held the *average* level down with a penalty that bounds no single logit, I pass the logits
through a smooth, bounded, strictly-monotone sigmoid cap `23·σ((z+5)/7.5)` inside the forward map and take
plain cross-entropy on the capped logits — a per-value bound, gradient alive everywhere, argmax untouched.
What I expect, falsifiably: `val_loss` should step *below* z-loss's 2.2926, since a per-value bound is
strictly safer than an average one and the softmax stays faithful everywhere — of the same rough order
z-loss bought over smoothing, though how much is real versus seed noise the table will decide. Perplexities
should follow below 44.09 / 68.25. The open risk is downstream: bounding the logits structurally could
pinch expressiveness on tasks that lean on confident margins, so a small accuracy trade against the
`val_loss` gain would be the cap's signature, not a uniform sweep. If it materializes, the honest reading
is that the value-bound is the strongest single loss-layer move here while z-loss remains the fallback when
downstream margin matters more than the last hundredth of validation loss.
