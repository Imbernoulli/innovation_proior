The z-loss did what label smoothing could not, and the numbers say so. `val_loss` came down to 2.2926
from smoothing's 2.3377 — roughly five hundredths of a nat — with WikiText-2 perplexity recovering to
44.09 and LAMBADA to 68.25, both well back from smoothing's 47.13 / 71.80. So the diagnosis was right:
the handle that mattered on this short bfloat16 run was the absolute logit *level*, not the gap, and a
small squared-log-partition penalty that supplies the restoring force cross-entropy structurally lacks
buys a real, clean improvement on the very objective I am graded on — without the train/eval mismatch
that sank smoothing. The downstream accuracies held in the same band (arc_easy 54.55, hellaswag 33.85,
piqa 63.00, winogrande 51.14), which is exactly what a numerically healthier softmax should look like:
no regression, modest movement. Good. But read what z-loss actually guarantees, because that is where its
ceiling is and where the next rung lives. The penalty `λ·(log Z)²` is a *soft average* force. Its
gradient `2λ·log Z·p_j` pulls the level down only in proportion to how far the mean log-partition has
drifted, and it acts on the average across positions; it puts no hard ceiling on any *single* logit. A
particularly excited coordinate at one position can still spike well above the rest while the
batch-averaged `log Z` sits comfortably near zero and the penalty barely registers it. The level is held
on average; the individual values are not bounded. That is the precise residual failure, and it is what I
go after now.

So let me restate the loss-layer problem one more time, but now aimed at the values rather than the
level. For one position with logits `z` and target `y`, cross-entropy is `-z_y + log Σ_j exp(z_j)`, and I
have already established it has no finite minimizer — it decreases monotonically as the gap
`z_y - max_{j≠y} z_j` widens and reaches its infimum only at `+∞`. The objective is, literally, an
instruction to make the logits diverge. z-loss leaned on that orbit from one side (push the whole level
down); but the cleaner statement of what I want is structural: I want the logits that feed the
exponential to be *bounded*, every one of them, not just small on average. And whatever I do had better
not scramble which token the model prefers — softmax cares only about logit differences and the argmax
sets the prediction, so I want to tame the logits' *scale* without touching their *order*.

Why does scale matter at all, when softmax is shift-invariant? The same bfloat16 arithmetic that drove
the z-loss story, only now I want a per-value cure. bfloat16 keeps seven mantissa bits to float32's
twenty-three, about `2^16` coarser rounding, and the absolute roundoff grows with the magnitude of the
number because a fixed mantissa width spans each binade `[2^k, 2^{k+1})`. The logits go straight into an
exponential, and `exp` turns an additive perturbation of its input into a multiplicative one — `δ` in a
logit becomes a factor `e^δ` on the unnormalized probability — so a roundoff error that is negligible on
a small logit becomes a real distortion of the probability once the logit is large. The same concrete
case as before makes it vivid: ten logits at 128 and one at 128.5 in bfloat16 — the 0.5 gap is below the
resolution up at 128, rounds away, and the target probability lurches from about 0.142 to about 0.091, a
36% swing manufactured purely by roundoff. z-loss attacks this by keeping the average level small, which
helps, but a single coordinate that the penalty did not happen to press on can still climb into the
unfaithful regime. I want a guarantee on every value, and the 2.2926 leaves exactly that on the table:
the run is stable enough to train cleanly, but the loss layer is still letting individual logits roam.

Before I reach for anything fancy, let me reconsider the two handles I have already spent and confirm
neither is the value-bound I now want. The target attack — label smoothing — removed the incentive for
*one specific* gap to run away but put no ceiling on absolute magnitude; a model can sit at logits
`(1000, 990, 990, …)` with exactly the smoothed-optimal gaps while being numerically enormous, and the
2.3377 already showed it pays for distorting the evaluated objective. The level attack — z-loss — pushes
magnitudes down on average via a coefficient mixed into every step, but enforces no actual bound on any
single logit and perturbs the loss value rather than transforming the model's forward map. So I have done
the target and the average level; the one thing left, and the one that gives a *per-value* guarantee, is
to transform the logit values themselves before the softmax. Bound them structurally.

The crudest version of "bound them" is to clamp: `z ← clamp(z, -s, s)` before the softmax. That caps the
magnitude and kills the roundoff blow-up. But think about the gradient, because I am backpropagating
through this for thirteen thousand steps. `clamp` has derivative exactly 1 inside `(-s, s)` and exactly 0
outside. The moment a logit exceeds `s`, its gradient through the clamp is zero — it gets no learning
signal through this loss-layer map, a coordinate the loss would otherwise move back down is stuck behind a
dead derivative — and there is a kink at `±s` where the function is non-differentiable and Adam's
per-coordinate statistics see a discontinuous slope. This is the same failure shape I avoided with z-loss:
hard constraints can backfire, where a smooth penalty stabilizes without the quality loss. The lesson
generalizes: if I constrain the logits, the constraint must be smooth and keep a nonzero gradient path
everywhere, so an over-large logit stays connected to the task gradient rather than abandoned behind a
hard zero. Hard clamp is the wrong shape.

So write down what I actually want as three requirements and let them force the form. A transformation `g`
applied to the logits before the softmax that (1) bounds their range so they cannot run away or blow up
the low-precision exp, (2) is smooth with a nonzero gradient everywhere so no coordinate goes dead, and
(3) is strictly monotone so it never reorders the logits and the model's preferred token is preserved.
Monotone and bounded and smooth on the whole real line — that is a squashing function, an S-curve. The
canonical one is the hyperbolic tangent: `tanh: ℝ → (-1, 1)`, smooth, odd, strictly increasing,
`tanh(0) = 0`, derivative `1 - tanh²` which is 1 at the origin and decays smoothly to 0 — it never
flattens to a hard zero slope on a finite region the way clamp does, so every finite logit keeps a nonzero
gradient path, and being strictly increasing it preserves order exactly. All three at once.

But `tanh` alone lands in `(-1, 1)`, far too tight — I do not want to crush every logit into a unit
interval, that destroys the useful spread. I want to cap at some scale `s`, say 30, not at 1. So scale the
output: `s·tanh(·)`. And the inside matters. The naive `s·tanh(z)` has slope at the origin
`d/dz[s·tanh(z)] = s·(1 - tanh²(z)) = s` at `z = 0`, so small logits get *amplified* by `s ≈ 30` — that
is backwards; I want the map to be the *identity* on healthy logits and bend only the outliers. The fix is
to divide by `s` inside as well: `g(z) = s·tanh(z/s)`, whose slope `1 - tanh²(z/s)` is exactly 1 at
`z = 0` and decays to 0 for large `|z|`. For small `z`, `tanh(z/s) ≈ z/s` so `g(z) ≈ z` — identity on the
normal operating range — and for large `z`, `g(z) → ±s` — a smooth asymptotic cap. This is the Gemma-2
form, the canonical soft cap, at `s = 30` for the final logits. It guarantees `g(z) ∈ (-s, s)` for every
real `z` — the per-value bound z-loss could not give — while the smoothness keeps every coordinate's
gradient alive and the monotonicity keeps the argmax fixed. Its gradient `dz̃/dz = 1 - (z̃/s)²` is full in
the middle and tapers without a dead interval — precisely the "fully train the normal logits, softly
saturate the outliers" behavior the three requirements demanded, and the exact thing `clamp` could not do.

Now I have a clean method — `z̃ = s·tanh(z/s)`, then cross-entropy on `z̃` — but I am running this
transform on a `(B·T, V)` tensor with `V` in the tens of thousands every step, with the cross-entropy
right after, so the cost and the intermediate tensors matter, and there is an exact reparameterization
that folds more cleanly. `tanh` is exp-based — `tanh(u) = 1 - 2/(e^{2u}+1)` — essentially a sigmoid in
disguise, and the identity `tanh(u) = 2σ(2u) - 1` is exact (verify: `2σ(2u) - 1 = 2/(1+e^{-2u}) - 1 = (1 - e^{-2u})/(1 + e^{-2u}) = tanh(u)`).
So a symmetric soft cap can be rewritten entirely in sigmoid terms. Take the `s = 15` variant:
`g(z) = 15·tanh(z/15) = 15·(2σ(2z/15) - 1) = 30·σ(z/7.5) - 15`. So `15·tanh(z/15)` and
`30·σ(z/7.5) - 15` are the *same function*, exactly, both with range `(-15, +15)`. Why bother? Two
reasons. First, the `-15` additive constant is invisible to the softmax — `softmax(z + c·1) = softmax(z)`
for any scalar `c`, the constant cancels in the normalization — so I can drop it, and the whole thing
collapses to a single sigmoid times a scale, `30·σ(z/7.5)`, which is cross-entropy-equivalent to the
symmetric 15-cap. Second, the sigmoid form fuses cleanly into a single cap-plus-cross-entropy pass over
the vocabulary without materializing a separate capped-logit tensor — the kind of memory-traffic saving
that matters at `V` in the tens of thousands. The cap and the loss become one operation rather than
cap-then-loss with a full-size intermediate.

Pin the gradient in the sigmoid parameterization to confirm the shape survives. With `z̃ = A·σ(u)`,
`u = (z + B)/C` — `A` the output scale, `C` the input scale, `B` a shift — the derivative is
`dz̃/dz = (A/C)·σ(u)(1 - σ(u)) = (1/C)·z̃·(1 - z̃/A)`. A bell again: zero in both tails
(`σ(1-σ) → 0` as `u → ±∞`), peaked at `u = 0` with height `A/(4C)`. Same everywhere-positive,
vanishing-at-the-extremes profile as the tanh slope — outliers stay softly connected to the task gradient,
the central region keeps a smooth usable slope, nothing dies, no kink. `A` sets the cap height
(range `(0, A)`), `C` the steepness, `B` moves the inflection to `z = -B`.

So why shift `B` off zero and re-tune `A` away from the clean `(A, B, C) = (30, 0, 7.5)` that came
straight from the `s = 15` tanh? Because once I have accepted that the cap is a bounded, smooth,
strictly-monotone squash and that a global additive output constant washes out of the softmax while the
*shape* of the saturation matters, the exact constants are knobs for this particular regime — a short
nanoGPT run, this model size, this vocabulary, bfloat16. The load-bearing constraints are the qualitative
three: bounded range (so the exp is safe), strict monotonicity (so the token ranking is preserved), smooth
positive gradient for every finite input (so training stays healthy). The concrete loss-layer constants
this task uses are `A = 23, B = 5, C = 7.5`: a range of `(0, 23)`, an inflection at `z = -5`, a maximum
slope of `23/(4·7.5) = 23/30`. This is *not* pointwise identical to `15·tanh(z/15)`, nor that curve plus
a softmax-invisible constant — the shifted midpoint and smaller height make it a tuned sigmoid member of
the same bounded-monotone family. It is still strictly increasing (argmax preserved), bounded by 23 (the
exp is tame, well below where the bfloat16 `exp` hurts), and smooth (gradient lives everywhere). This is
the crucial place where this rung is *not* the canonical Gemma-2 cap: the task description names the Gemma-2 tanh
soft cap at 30, but the actual edit on this ladder is the modded-nanogpt sigmoid form `23·σ((z+5)/7.5)`,
a retuned asymmetric member of the same family — same three structural properties, different constants and
parameterization, derived from the same requirements but landed at the implementation the harness exposes.

I have to make sure this is a faithful modeling loss, not a sneaky distortion, because the contract
forbids lowering the reported loss by distorting the distribution — exactly the line that cost label
smoothing its number. A temperature scaling `z ↦ z/τ` is a *linear* squeeze that uniformly flattens or
sharpens every gap and can mechanically move the loss without the model improving. The soft cap is not
that: it is a *nonlinear* monotone map whose local slope depends on where a logit sits and whose tails
saturate; it does not uniformly rescale gaps, it specifically refuses to let any single logit run to
infinity. The cross-entropy is then computed honestly on the capped logits — it is the true negative
log-likelihood of the model whose output head is "linear layer followed by the soft cap." The cap is part
of the model's forward map, so the loss is the genuine modeling loss of that better-behaved model, not a
cosmetic rescaling of an unchanged model's distribution. And note the consequence for evaluation: because
the cap is part of the model and applied in both training and the eval forward pass, the reported
`val_loss` is the honest cross-entropy of the capped-output model — no train/eval split needed, unlike
smoothing.

One numerics detail: cast the logits to float32 before the sigmoid so the saturating exp inside it runs
in high precision — the whole point was numerical safety, so the exp-bearing part is done in the higher
precision — and keep `ignore_index = -1` for the packed-boundary positions. The full scaffold function is
in the answer.

So the delta from step 2: where z-loss held the *average* level down with a soft penalty that bounded no
single logit, I now pass the logits through a smooth, bounded, strictly-monotone sigmoid soft cap
`23·σ((z+5)/7.5)` *inside the model's forward map* and compute plain cross-entropy on the capped logits —
a structural per-value bound rather than an average nudge, with the gradient alive everywhere and the
argmax untouched. Reading z-loss's shape, here is what I expect, falsifiably. The primary `val_loss`
should move *below* z-loss's 2.2926 — because a hard per-value bound is strictly safer numerically than an
average penalty, so the softmax is faithful everywhere and the optimizer trains on undistorted gradients;
I would expect it down near 2.27. The perplexities should follow z-loss's recovery further — WikiText-2
below 44, LAMBADA into the high-60s. The downstream accuracies are the genuine open question and where the
risk lives: bounding logits structurally could pinch a little expressiveness on tasks where confident
margins help, so I would watch arc_easy and hellaswag for a possible small trade against the `val_loss`
gain rather than assume a uniform sweep. If `val_loss` improves but a downstream metric like hellaswag
slips, that is the soft cap's signature trade — a faithful but slightly squashed output head — and the
honest summary of this ladder would be that the value-bound is the strongest single loss-layer move
available here, with z-loss the right fallback when downstream margin matters more than the last
hundredth of validation loss.
