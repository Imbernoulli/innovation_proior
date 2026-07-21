OPD closed the progression where its construction pointed, and the residual it left is a specific,
fixable instability rather than a vague "try harder." `baseline:opd` reads GSM8K 0.4852, MATH-500
0.310, AMC 0.0688. Read against the whole trend: on GSM8K the counts are dagger 595, rs_kd 599, taid
618, gkd 622, opd 640 out of 1319 — every move toward mode-seeking added problems, and OPD's `+18`
over gkd clears the `±1.4%` single-standard-error bar, a real gain. On MATH-500 the counts are 145,
146, 140, 156, 155 out of 500: gkd peaked at 156, opd came in at 155 — one lower, well inside the
`±2%` band on its own but *directionally* the mode-collapse tax I flagged, and the tell is that it is
the first metric to stop rising when GSM8K jumped hardest. So the picture is unambiguous: the
divergence direction is the binding axis, more mode-seeking lifts GSM8K monotonically, but *unbounded*
reverse KL stops paying on the hard set because it overshoots into collapse. The gap to close is to
keep OPD's mode-seeking commitment, which won GSM8K by the largest single-step jump so far, while stopping
it from collapsing on MATH-500 — and the fix has to come out of the mechanism or it is guesswork.

There is a corroborating signal in the train-loss column, and for once it points the same way as the
mechanism. OPD's final train loss came in at 0.558 — back up near dagger's 0.507, an order of
magnitude above gkd's 0.077. Not a cross-loss quality measure, but *within* the reverse-KL world
diagnostic: reverse KL is unbounded, its per-token value scales like `log(p_S/p_T)`, and on the
teacher-surprised on-policy tokens where `p_T → 0` that value is large, so a converged OPD averaging a
high number is exactly what an objective dominated by a few blowing-up tokens looks like — the same
tokens I am about to finger for the MATH-500 collapse. Whatever I land next, if it truly bounds the
gradient, should also sit *lower* than 0.558 at convergence, a second cheap prediction the train-loss
column will let me check.

Write OPD's gradient properly. The per-token loss is `L = Σ_v p_S(v)·(log p_S(v) − log p_T(v))`, the
student softmax `p_S = softmax(z)` carries the parameters, `p_T` is fixed. With the softmax Jacobian
the entropy leg cancels and what survives is `∂L/∂z_k = p_S(k)·(log r_k − KL(p_S ‖ p_T))`, ratio
`r_k = p_S(k)/p_T(k)`; in the sampled-sequence form the per-token multiplier is `log r + 1`. Either
way the coefficient is a *logarithm of the ratio*, and it runs away when `p_T → 0` — when the teacher
assigns near-zero probability to what the student produced, `r_k → ∞` and `log r_k → +∞`. And the
on-policy loop guarantees this is the *typical* case, not an edge case: the batches are the student's
own generations, sequences the 7.6B teacher did not write, so on a self-generated math chain the
teacher is frequently surprised. It is worst on the hard set: a 0.5B student wandering through a
competition-level derivation produces more off-distribution tokens than on a short GSM8K sum, so the
teacher-surprise events that detonate the gradient are densest where MATH-500 lives. OPD's coefficient
`log(p_S/p_T)` detonates on its own training data precisely where the student is exploring, throwing
huge noisy steps that yank it toward whatever narrow region the teacher does endorse — the mechanism
of the mode collapse I measured. I cannot fix it by changing the data, because the data is the point.

Name the blow-up exactly, because the fix follows from it and not from taste: a *raw probability sits
in a denominator and goes to zero* — the teacher `p_T` in the ratio `p_S/p_T`, and nothing bounds a
raw distribution from below on an off-distribution token. The cheaper patches are all worse. Clamping
`log r` to a ceiling, or clipping the per-token loss, discards the *magnitude* of every large-ratio
token identically, so a mildly-surprised and a catastrophically-surprised token get the same truncated
push and the gradient is no longer the gradient of any coherent objective — I would optimize a clipped
surrogate whose fixed point I cannot name. An additive `ε` floor `p_T ← max(p_T, ε)` is a constant
unrelated to the student, so it over-corrects on confident tokens and under-corrects on exploratory
ones. What I want is a floor that (a) is a proper distribution, so the loss stays a real divergence
with a nameable minimum, and (b) scales with where the student actually is. So the structural cure is
to not let the raw teacher be the denominator: replace it with a *mixture* `p̃ = (1−α)·p_T + α·p_S`.
On a token where `p_T ≈ 0` the mixture still carries the `α·p_S` leg, so `p̃ ≥ α·p_S`, the ratio
`p_S/p̃ ≤ 1/α` is *bounded*, and the log cannot diverge. This is reverse KL skewed toward a mixture:
`KL(p_S ‖ (1−α)·p_T + α·p_S)`. At `α = 0` it is exactly OPD's reverse KL, and it is a proper
divergence for every `α`, so unlike the clip it has a nameable minimum: the student that equals the
teacher makes `p̃ = p_T` and drives the divergence to zero, the same target OPD had.

Check the skewed gradient actually does what I claim. Differentiate `KL(p_S ‖ p̃)` where the `α·p_S`
leg *also* depends on the parameters (the subtlety a lazy derivation treating `p̃` as fixed would
miss): carrying the dependence through both the outer `p_S` weight and the `α·p_S` inside `p̃`, the
reverse-KL product rule gives a per-token coefficient `g(r) = log r + 1 − α·r` with `r = p_S/p̃`,
versus OPD's `log r + 1`. Maximize `g` over the admissible range `r ∈ (0, 1/α]`: `g'(r) = 1/r − α > 0`
for all `r < 1/α`, so `g` is increasing and attains its maximum at the boundary `r = 1/α`, where
`g(1/α) = log(1/α) + 1 − α·(1/α) = log(1/α)`. At `α = 0.1` the ceiling is `log 10 ≈ 2.30`. Two things
bought that at once: `r ≤ 1/α` caps the `log r` term, and the new `−α·r` term *subtracts* and grows
with `r`, actively pulling the coefficient back down exactly where it would spike (contributing
`−α·(1/α) = −1` at the extreme). So skewed reverse KL has a bounded gradient on precisely the
teacher-surprised on-policy tokens that made OPD's MATH-500 collapse, while keeping reverse KL's
mode-seeking weighting untouched — the force is still `p_S(k)·g`, still weighted by the student's own
mass, still zero-forcing, still committing to the teacher's modes. It removes the detonation without
touching the property that won GSM8K.

And the minimum stays where OPD's was: at `p_S = p_T` the mixture is `p̃ = p_T`, so the loss is
`KL(p_T ‖ p_T) = 0`, and the gradient vanishes there too — at `p_S = p_T` every `r_k = 1`, so
`g(r_k) = 1 − α` is constant across `k`, and the softmax gradient's mean-subtraction sends a constant
coefficient to zero. So the skew tames the *approach* to the minimum without moving the minimum
itself; I am still distilling the teacher, not an `α`-blurred surrogate with a shifted optimum.

Put a real token through both to feel the size of the cure. At a teacher-surprised position where the
student settled on `p_S = 0.30` that the teacher finds nearly impossible, `p_T = 0.001`: OPD's
coefficient is `log(0.30/0.001) + 1 = log 300 + 1 = 6.70`, and there are many such tokens on a
self-generated competition chain, so the step is dominated by the noisiest positions. The skew:
`p̃ = 0.9·0.001 + 0.1·0.30 = 0.0309`, `r = 9.71` just under the `1/α = 10` ceiling,
`g(r) = log 9.71 + 1 − 0.971 = 2.30` — better than half of OPD's `6.70`, on the exact token doing the
damage. And the cap is *soft* in the right places: at a mildly surprised token `p_T = 0.05`, OPD gives
`2.79` and the skew `1.99`, so it barely touches the well-behaved tokens and clamps hardest on the
catastrophic ones — a smooth compression, not a flat clip that discards magnitude uniformly.

Now the opposite failure, because a floor too high stops me distilling the teacher at all. Push
`α → 1` and `p̃ → p_S`, so the loss becomes `KL(p_S ‖ p_S) = 0` — distilling the student into itself,
teacher signal gone. So `α` cannot be large. There is a second pressure and it points the other way:
the loss is estimated on mini-batches, and the skewed estimator's L2 error carries inverse-`α` terms —
the unprotected raw denominator is what injects variance, so a larger `α` floors it and tightens the
estimate. But the raw "the gradient got smaller" benefit of large `α` is illusory under an Adam-style
optimizer, which normalizes each coordinate by a running estimate of its gradient scale and divides a
*uniformly* smaller coefficient right back out; what survives is the *shape* of the estimation error
in the units Adam steps in, the L2 norm normalized by the gradient scale. In those units the
inverse-`α` pieces want `α` large while the `1/(1−α)` teacher-fidelity pieces want `α` small — a
convex trade-off with an interior optimum. Price the neighbors: at `α = 0.05` the ceiling rises to
`log 20 ≈ 3.0`, barely tamer than raw OPD, with twice the estimation variance; at `α = 0.2` the
ceiling drops to `log 5 ≈ 1.6` but the target is `0.8·p_T + 0.2·p_S`, a visibly teacher–student blend.
So `0.05` under-tames and over-varies while `0.2` over-dilutes; `α = 0.1` — ceiling `log 10`, target
90% teacher, ratio bounded at `1/α = 10` — is the point where the tail is bounded hard enough to stop
the detonation and the target is still overwhelmingly the teacher.

This also shows skewing is not the same move as the generalized JSD already on the ladder, so I am not
re-running the earlier interior-divergence experiment with a different number. The symmetric JSD is
structurally a *sum of two skewed KLs whose skew parameters are tied to one `β`*, so the single `β`
that sets the skew of the forward leg forces the *complementary* skew on the reverse leg — I cannot
make *both* legs mildly skewed at once: a mild `α = 0.1` on the reverse term would force `0.9` on the
forward term, and balancing at `β = 0.5` gives two *unskewed* KLs (the earlier symmetric point), not a
mildly-skewed reverse one. A single freely-chosen-`α` skewed reverse KL reaches that mild interior
optimum the estimation analysis wants; the JSD's coupled parameter structurally cannot get there.
Different knob, and the one the OPD failure mode actually calls for — I could not have reached this
`α = 0.1` reverse-skew point by any setting of the JSD's `β`.

Now the edit surface, because I only fill the loss body — no replay buffer, no scheduler, no `lmbda`
change. The full streamlined recipe this loss comes from pairs the skewed KL with two data-side
mechanisms — an adaptive self-generation probability that ramps up on-policy sampling guided by
validation loss, and an off-policy replay buffer with a decaying replay ratio to amortize generation
cost — both needing framework-level access to the data pipeline this single-loss surface does not
expose. So I drop them and land only the *loss*: the skewed reverse KL at `α = 0.1`, computed on
whatever batch the trainer's static `lmbda` mixing produced, exactly as every prior loss consumed it.
The synergy is real — the bounded gradient is what makes replaying slightly-stale samples safe — but
not mine to build here; the per-token loss stands on its own.

The arithmetic mirrors OPD's reverse-KL body with the denominator swapped for the mixture, and two
implementation details are load-bearing. Divide both logit tensors by the shared temperature, softmax
to `p_T` and `p_S` in float32 (the mixture floor and the log live or die on the small-probability
tokens that are the whole point; bf16 would lose exactly that precision), form `p̃ = (1−α)·p_T +
α·p_S` in probability space, take its log, and accumulate the per-token reverse KL `Σ_v p_S·(log p_S −
log p̃)`. First, keep *both* legs: unlike a forward KL the student-entropy leg `Σ_v p_S·log p_S`
depends on the parameters and carries the `+1` normalization gradient — the `log r + 1 − α·r`
coefficient only comes out with it present; drop it and I would compute a cross-entropy whose gradient
is a *different*, un-normalized object and the bound analysis would not apply. Second, any `±inf`
logit positions (the crop can leave a few) must be masked to contribute exactly zero rather than
propagate a `nan` through the `p·log p` product, so I guard the products with an `isinf` mask before
summing. Then sum over the vocabulary, mask to the completion tokens (`labels ≠ −100`), and average
per token for `batchmean`. The direction is pinned by construction — the student is the outer weight,
the mixture the comparison — so this is genuinely `KL(p_S ‖ p̃)`, the skewed *reverse* direction, not
its forward cousin. On cost it sits at OPD's: two softmaxes, one mixture, one log, a two-leg
vocabulary sum at the same `[B, T, V]` width, far under the JSD point's mixture-plus-two-KLs. The
float32 softmaxes are the one deliberate expense, and the bounded-gradient cure is essentially free
over the endpoint it fixes. The full body is in the answer.

What this has to clear, stated against the measured results — and because this is the endpoint there
is no feedback of its own, only the bar OPD set: GSM8K 0.4852, MATH-500 0.310. The claim is asymmetric
and follows directly from the mechanism. On GSM8K the skew keeps OPD's mode-seeking weighting and
leaves the target 90% teacher, so it should *hold* OPD's level — if anything the bounded gradient
should let it train a touch cleaner on the tokens where OPD threw wild steps, so I expect it to match
or modestly exceed 0.4852. The real test is MATH-500: the entire reason to skew is that OPD's
unbounded gradient overshot into mode collapse there, and the floor `p̃ ≥ α·p_S` is designed to stop
exactly that detonation on the teacher-surprised on-policy tokens I argued are densest on the hard set,
so the vindicating result is MATH-500 recovering *above* OPD's 0.310 while GSM8K holds. The ways I
could be wrong are each diagnostic: if MATH-500 stays pinned at OPD's level, the collapse is driven by
the reverse *direction* itself, not the gradient blow-up, and I would have to soften the direction, not
the denominator; if GSM8K *drops* below OPD, the `α` is too large and is diluting the mode-seeking
commitment, and I would push it toward the small end of the convex interior. The clean win — GSM8K ≥
0.4852 and MATH-500 > 0.310 — is the one the bounded-gradient mechanism predicts, and it is what I
would validate first. AMC stays noise.
