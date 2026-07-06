OPD closed the ladder where its construction pointed, and the residual it left is a specific,
fixable instability rather than a vague "try harder." `baseline:opd` reads GSM8K 0.4852, MATH-500
0.310, AMC 0.0688. Let me read those against the whole ladder before I move, because the shape of the
five-rung trend is the argument for what to do next. On GSM8K the counts are dagger 595, rs_kd 599,
taid 618, gkd 622, opd 640 out of 1319 — every move toward mode-seeking added problems, and OPD's
640 is `+18` over gkd's 622, a `+0.0136` jump that clears the `±1.4%` single-standard-error bar on
GSM8K, so it is a real gain and not noise. Full commitment to the teacher's preferred continuation
pays most on short, near-deterministic arithmetic chains, exactly as the reverse-KL endpoint
predicted. On MATH-500 the counts are 145, 146, 140, 156, 155 out of 500: gkd peaked at 156 (0.312),
and opd came in at 155 (0.310) — one problem lower, `−0.002`, well inside the `±2%` MATH-500 noise
band on its own but *directionally* the mode-collapse tax I flagged, and the tell is that it is the
first metric to stop rising when GSM8K jumped hardest. So the picture across the whole ladder is now
unambiguous. The divergence direction is the binding axis; more mode-seeking lifts the headline GSM8K
metric monotonically; but *unbounded* reverse KL stops paying on the hard set — GSM8K `+18` while
MATH-500 goes `−1` — because it overshoots into collapse. That is the precise gap I want to close:
keep OPD's mode-seeking commitment, which won GSM8K by the largest single-rung margin on the ladder,
but stop it from collapsing on the hard set. And the way OPD overshoots tells me the mechanism, so let
me look at *why* unbounded reverse KL is brittle, in the gradient, because the fix has to come out of
the mechanism or it is guesswork.

There is a corroborating signal in the train-loss column, and for once it points the same way as the
mechanism rather than being the incomparable-across-losses number I have kept warning myself about.
OPD's final training loss came in at 0.558 — back up near dagger's 0.507 and an order of magnitude above
gkd's 0.077. I will not read it across losses as a quality measure, but *within* the reverse-KL world it
is diagnostic: reverse KL is unbounded, its per-token value scales like `log(p_S/p_T)`, and on the
teacher-surprised on-policy tokens where `p_T → 0` that value is large, so a converged OPD averaging a
high number is exactly what an objective dominated by a few blowing-up tokens looks like — the same
tokens I am about to finger for the MATH-500 collapse, seen now in the loss magnitude itself. Whatever I
land next should, if it truly bounds the gradient, also sit *lower* than 0.558 at convergence, because
bounding the coefficient bounds the per-token loss too; that is a second, cheap prediction the
train-loss column will let me check alongside the accuracies.

OPD trains reverse KL `KL(p_S ‖ p_T)` on the student's own on-policy rollouts. Write its gradient
properly. The per-token loss is `L = Σ_v p_S(v)·(log p_S(v) − log p_T(v))`, the student softmax
`p_S = softmax(z)` carries the parameters, and `p_T` is fixed. Differentiating with the softmax
Jacobian `∂p_S(v)/∂z_k = p_S(v)(δ_{vk} − p_S(k))`, the entropy leg `Σ_v p_S(v)·∂log p_S(v)/∂z_k`
cancels to zero, and what survives is `∂L/∂z_k = p_S(k)·(log r_k − KL(p_S ‖ p_T))` with the ratio
`r_k = p_S(k)/p_T(k)`. So the force on logit `k` is the student's own mass `p_S(k)` times a
coefficient `log r_k` measured against the mean; in the sampled-sequence form of the same object the
per-token multiplier is `log r + 1`. Either way the coefficient is a *logarithm of the ratio*, and now
read where it runs away: it blows up when `p_T → 0` — when the teacher assigns near-zero probability
to what the student produced, `r_k → ∞` and `log r_k → +∞`. Here is the thing the on-policy loop
guarantees: the batches are the *student's own* generations, sequences the 7.6B teacher did not write,
so on a self-generated math chain the teacher is frequently surprised — `p_T ≈ 0` is not a rare edge
case here, it is the *typical* case on exactly the data OPD trains on. And it is worst on precisely
the hard set: a 0.5B student wandering through a competition-level derivation produces more
off-distribution tokens than it does on a short GSM8K sum, so the teacher-surprise events that
detonate the gradient are densest where MATH-500 lives. So OPD's gradient coefficient `log(p_S/p_T)`
detonates on its own training data precisely where the student is exploring, throwing huge, noisy
steps that yank it toward whatever narrow region the teacher does endorse — which is the mechanism of
the mode collapse I measured on MATH-500. The unbounded reverse KL is not just *theoretically*
brittle; its instability lives in the same on-policy generations that make OPD strong on GSM8K, which
is why I cannot fix it by changing the data — the data is the point.

Name the source of the blow-up exactly, because the fix follows from it and not from taste. The
coefficient explodes because a *raw probability sits in a denominator and goes to zero* — the teacher
probability `p_T` in the ratio `p_S/p_T`. A raw distribution can be arbitrarily close to zero on an
off-distribution token; nothing bounds it from below. Before I reach for the structural cure I should
be honest about the cheaper patches and why they are worse, because "clip the gradient" is what a
practitioner tries first. I could clamp `log r` to some ceiling, or clip the per-token loss, or add a
floor `p_T ← max(p_T, ε)` before the log. Each removes the `nan` but each also throws away signal
blindly: a hard clip discards the *magnitude* of every large-ratio token identically, so a token where
the teacher is mildly surprised and one where it is catastrophically surprised get the same truncated
push, and the gradient is no longer the gradient of any coherent objective — I would be optimizing a
clipped surrogate whose fixed point I cannot name. An additive `ε` floor on `p_T` is closer but it is
a constant floor unrelated to the student, so it over-corrects on confident tokens and under-corrects
on the exploratory ones. What I want is a floor that (a) is a proper distribution so the loss stays a
real divergence with a nameable minimum, and (b) scales with where the student actually is. So the
structural cure is to not let the raw teacher distribution be the denominator at all: replace it with
a *mixture* that always contains a sliver of the student, so the denominator is floored away from zero
*by the student's own mass*. If the comparison distribution is `p̃ = (1−α)·p_T + α·p_S` for a small
`α`, then on a token where `p_T ≈ 0` the mixture still carries the `α·p_S` leg, so `p̃ ≥ α·p_S`, the
ratio `p_S/p̃ ≤ 1/α` is *bounded*, and the log can no longer diverge. This is a reverse KL skewed
toward a mixture rather than the raw teacher — take the reverse KL of the student against `p̃`:
`KL(p_S ‖ (1−α)·p_T + α·p_S)`. At `α = 0` it is exactly OPD's reverse KL; as `α` grows the floor rises
and the gradient tames. It is a proper divergence for every `α`, so unlike the clip it has a minimum I
can name: the student that equals the teacher makes `p̃ = p_T` and drives the divergence to zero, the
same target OPD had.

Let me check the skewed gradient actually does what I claim, because if the bound is real this is a
clean win over OPD with no other moving part. Differentiate `KL(p_S ‖ p̃)` where `p̃ = (1−α)·p_T +
α·p_S` and the `α·p_S` leg *also* depends on the parameters — this is the subtlety, since a lazy
derivation that treats `p̃` as fixed would miss a term. Carrying the parameter dependence through both
the outer `p_S` weight and the `α·p_S` inside `p̃`, the reverse-KL product rule still gives the
`log r + 1` structure, but with `r = p_S/p̃`, and the extra `θ`-dependence in the mixture adds a term,
leaving a per-token coefficient `g(r) = log r + 1 − α·r`. Compare it to OPD's `log r + 1`. Now do the
one calculation that decides whether the bound is real: maximize `g` over the admissible range
`r ∈ (0, 1/α]`. Its derivative is `g'(r) = 1/r − α`, which is positive for all `r < 1/α`, so `g` is
increasing and attains its maximum at the boundary `r = 1/α`, where `g(1/α) = log(1/α) + 1 − α·(1/α) =
log(1/α) + 1 − 1 = log(1/α)`. At `α = 0.1` that ceiling is `log 10 ≈ 2.30`. So the skewed coefficient
is bounded above by `≈ 2.3` on the worst possible token — the teacher-surprised `p_T → 0` token that
sent OPD's coefficient to `+∞`. Two things bought that at once: first `r ≤ 1/α` caps the `log r` term,
and second the new `−α·r` term *subtracts* and grows with `r`, actively pulling the coefficient back
down exactly where it would otherwise spike, contributing `−α·(1/α) = −1` at the extreme. So skewed
reverse KL has a bounded, well-behaved gradient on precisely the teacher-surprised on-policy tokens
that made OPD's MATH-500 collapse — and it keeps reverse KL's mode-seeking weighting untouched (the
force is still `p_S(k)·g`, still weighted by the student's own mass, still zero-forcing, still
committing to the teacher's modes, which is what won GSM8K). It removes the detonation without
touching the property that made OPD strong. That is the whole design in one gradient.

One check I owe on the other end of the range, because a bounded gradient is worthless if it does not
vanish at the right place: does the skewed loss still have its minimum where the student equals the
teacher? At `p_S = p_T` the mixture is `p̃ = (1−α)·p_T + α·p_T = p_T`, so the loss is `KL(p_T ‖ p_T) = 0`
— the value bottoms out at the teacher, the same target OPD had. And the gradient vanishes there too,
which the coefficient makes visible: at `p_S = p_T` the ratio `r_k = p_S(k)/p̃(k) = 1` for *every*
token, so `g(r_k) = log 1 + 1 − α = 1 − α` is the same constant across `k`. The softmax gradient of any
reverse-style loss carries a mean-subtraction — the force on logit `k` is `p_S(k)·(g(r_k) − Σ_j p_S(j)·
g(r_j))` — and a constant coefficient subtracts to zero: `(1−α) − (1−α) = 0`. So the skew leaves the
fixed point exactly where OPD's was, student = teacher, with zero gradient; the `α·p_S` floor tames the
*approach* to that minimum without moving the minimum itself. That is the reassurance I need that I am
still distilling the teacher and not some `α`-blurred surrogate with a shifted optimum.

Put a real token through both to feel the size of the cure. Take a teacher-surprised on-policy
position where the student has settled on a continuation with `p_S = 0.30` that the teacher finds
nearly impossible, `p_T = 0.001`. OPD's coefficient there is `log(0.30/0.001) + 1 = log 300 + 1 =
6.70` — a huge force from a single token, and there are many such tokens on a self-generated
competition chain, so the step is dominated by the noisiest positions. Now the skewed mixture:
`p̃ = 0.9·0.001 + 0.1·0.30 = 0.0309`, so `r = 0.30/0.0309 = 9.71` (just under the `1/α = 10` ceiling),
and `g(r) = log 9.71 + 1 − 0.1·9.71 = 2.27 + 1 − 0.97 = 2.30` — capped at `log 10`, better than
half of OPD's `6.70`, on the exact token that was doing the damage. And crucially the cap is *soft* in
the right places: at a *mildly* surprised token, `p_T = 0.05`, OPD gives `log 6 + 1 = 2.79` and the
skew gives `1.99`, so the skew barely touches the well-behaved tokens and clamps hardest exactly on the
catastrophic ones. That is the shape I want — not a flat clip that discards magnitude uniformly, but a
smooth compression that leaves ordinary distillation gradients almost intact and only tames the tail.

Now I should worry about the opposite failure, because a floor that is too high stops me distilling
the teacher at all, and I do not want to trade collapse for mush. Push `α → 1` and `p̃ → p_S`, so the
loss becomes `KL(p_S ‖ p_S) = 0` — I would be distilling the student into itself and the teacher
signal vanishes. So `α` cannot be large. There is a second consideration pulling the same way and a
third pulling against it, and I want to actually reason about the trade rather than pick a round
number. The loss is estimated on mini-batches, so a high-variance estimator gives a noisy objective;
the empirical L2 error of the skewed estimator, under mild assumptions, carries inverse-`α` terms — the
unprotected raw denominator is what injects estimation variance, and moving `α` away from zero floors
it, so a *larger* `α` both bounds the gradient and tightens the mini-batch estimate. That argues `α`
not too small. But pushing `α` up trades against the `1/(1−α)` pieces (the teacher-fidelity side) and,
more subtly, against how the optimizer actually moves: an Adam-style optimizer normalizes each
coordinate by a running estimate of its gradient scale, so a *uniformly* smaller gradient coefficient
is divided right back out — the raw "the gradient got smaller" benefit of large `α` is largely
compensated away, and what survives is the *shape* of the estimation error in the units Adam steps in,
the L2 norm normalized by the gradient scale. Looking at the error in those units, the inverse-`α`
pieces want `α` large while the inverse-`(1−α)` pieces want `α` small, giving a convex trade-off with
an interior optimum: small enough that the target is still mostly the teacher (so I am distilling the
teacher, not a teacher–student blend), large enough to floor the denominator and cut the variance. A
mild value, `α = 0.1`, sits in that interior — the comparison distribution is `p̃ = 0.9·p_T + 0.1·p_S`,
90% faithful to the teacher, with just enough student mixed in to bound the ratio at `1/α = 10` and cap
the worst-case coefficient at `log 10`. So `α = 0.1`, and it is an interior optimum I can point at, not
a default.

Let me make the interior concrete by pricing the two neighbors I could have picked, because "interior
optimum" is only convincing if the endpoints are visibly worse. Halve it to `α = 0.05`: the ratio
ceiling rises to `1/α = 20` and the worst-case coefficient to `log 20 ≈ 3.0`, so the gradient is barely
a third more tamed than raw OPD and much less than `α = 0.1`'s `log 10 ≈ 2.3`, while the inverse-`α`
estimation variance is twice as large — I have half-fixed the detonation and paid for it in noise.
Double it to `α = 0.2`: the ceiling drops to `log 5 ≈ 1.6`, tighter still, but the comparison
distribution is now `0.8·p_T + 0.2·p_S`, only 80% teacher — I have started distilling a visibly
teacher–student blend rather than the teacher, and the `1/(1−α)` fidelity term is what climbs. So `0.05`
under-tames and over-varies while `0.2` over-dilutes; `α = 0.1` — ceiling `log 10`, target 90% teacher —
is the point where the tail is bounded hard enough to stop the detonation and the target is still
overwhelmingly the teacher. The convex trade-off is not a slogan: it is `log(1/α)` falling while the
`1/(1−α)`-flavored dilution rises, and `0.1` sits near where they cross for this pair.

This also tells me why skewing is not the same move as the generalized JSD I already have on the
ladder, so I am not just re-running the earlier interior-divergence rung with a different number — I
want to be sure the knob is genuinely new. The symmetric JSD is, structurally, a *sum of two skewed
KLs whose skew parameters are tied to one `β`* — schematically `β·(skewed forward)^β + (1−β)·(skewed
reverse)^{1−β}` — so the single `β` that sets the skew of the forward leg forces the *complementary*
skew on the reverse leg. I cannot make *both* legs mildly skewed at once: if I want a mild `α = 0.1`
skew on the reverse term, the JSD coupling forces `0.9` on the forward term, slamming that leg into
the high-skew regime, and if I balance them at `β = 0.5` I get two *unskewed* KLs (the earlier rung's
symmetric point), not a mildly-skewed reverse one. The estimation-vs-`α` analysis says I want a *mild*
skew on the one term I am actually using; a single freely-chosen-`α` skewed reverse KL reaches that
mild interior optimum, and the JSD's coupled parameter structurally cannot get there. The earlier
`β = 0.5` rung balanced two unskewed KL directions; this skews *one* mode-seeking direction by a small,
freely tuned amount. Different knob, and the one the OPD failure mode actually calls for — I could not
have reached this `α = 0.1` reverse-skew point by any setting of the JSD's `β`.

Now make it concrete in this task's edit surface, because I only get to fill the `compute_distill_loss`
body and nothing else — no replay buffer, no scheduler, no `lmbda` change. The full streamlined recipe
this loss comes from pairs the skewed KL with two data-side mechanisms — an adaptive self-generation
probability that ramps up on-policy sampling guided by validation loss, and an off-policy replay buffer
with a decaying replay ratio to amortize generation cost — and those need framework-level access to the
data pipeline that this single-loss surface does not expose. So I drop them and land only the *loss*:
the skewed reverse KL at `α = 0.1`, computed on whatever batch the trainer's static `lmbda` mixing
produced, exactly as every prior rung consumed that mixing. The synergy is real — the skewed loss's
fast, stable early movement is what would let an off-policy buffer work, and its bounded gradient is
what makes replaying slightly-stale samples safe — but it is not mine to build here; the per-token loss
stands on its own, and the bounded-gradient property is a loss-level fact that does not depend on the
data machinery around it.

The arithmetic mirrors OPD's reverse-KL body with the denominator swapped for the mixture, and two
implementation details are load-bearing. Divide both logit tensors by the shared temperature, softmax
to get `p_T` and `p_S` (in float32, because the mixture and the log will otherwise lose precision on
the small-probability tokens that are the whole point), form the mixture `p̃ = (1−α)·p_T + α·p_S` in
probability space and take its log, then accumulate the per-token reverse KL `Σ_v p_S·(log p_S −
log p̃)`. First detail: I keep *both* legs, because unlike a forward KL the student-entropy leg
`Σ_v p_S·log p_S` depends on the parameters and carries the `+1` normalization gradient I derived — the
`log r + 1 − α·r` coefficient only comes out with the entropy leg present; drop it and I would be
computing a cross-entropy whose gradient is a *different*, un-normalized object, and the bound analysis
would no longer apply. Second detail: any `±inf` logit positions (the trainer's crop can leave a few)
must be masked to contribute exactly zero rather than propagate a `nan` through the `p·log p` product,
so I guard the products with an `isinf` mask before summing. Then sum over the vocabulary to a
per-token value, mask to the completion tokens (`labels ≠ −100`), and average per token for
`batchmean`. The direction is pinned by construction: the student is the outer weight and the mixture
is the comparison, so this is genuinely `KL(p_S ‖ p̃)`, the skewed *reverse* direction, not its forward
cousin — swap them and I would get the mass-covering objective the ladder spent five rungs climbing
away from. The full scaffold body is in the answer.

On cost, this sits right at OPD's: two softmaxes, one elementwise mixture, one log, and a two-leg
vocabulary sum, all at the same `[B, T, V]` width — a hair more arithmetic than the pure reverse KL for
the mixture and its log, no extra memory, and far under the JSD rung's mixture-plus-two-KLs. The float32
softmaxes are the one deliberate expense: the mixture floor and the log live or die on the
small-probability tokens, and doing them in bf16 would lose exactly the precision the whole method is
built to exploit. So the bounded-gradient cure is essentially free over the endpoint it fixes.

What this has to clear, stated against the measured ladder, with no numbers invented — and because
this is the endpoint there is no feedback of its own to lean on, only the bar OPD set. The bar is OPD,
the strongest baseline: GSM8K 0.4852, MATH-500 0.310. The falsifiable claim is asymmetric and it
follows directly from the mechanism. On GSM8K the skewed reverse KL keeps OPD's mode-seeking weighting,
so it should *hold* OPD's level — I would not expect the skew to cost the GSM8K commitment that reverse
KL bought, since `α = 0.1` leaves the target 90% teacher and the direction unchanged; if anything the
bounded gradient should let it train a touch cleaner on the tokens where OPD was throwing wild steps,
so I expect it to match or modestly exceed 0.4852. The real test is MATH-500: the entire reason to skew
is that OPD's unbounded gradient overshot into mode collapse there (155/500, `−1` under gkd's peak),
and the floor `p̃ ≥ α·p_S` is designed to stop exactly that detonation on teacher-surprised on-policy
tokens, which I argued are densest on the hard set. So the prediction that would vindicate this rung is
MATH-500 recovering *above* OPD's 0.310 — back to or past gkd's 0.312 — while GSM8K holds OPD's lead.
The three ways I could be wrong are each diagnostic. If MATH-500 stays pinned at OPD's level, the
collapse is not driven by the gradient blow-up but by the reverse *direction* itself, and the skew is
treating the wrong cause — I would then have to soften the direction, not the denominator. If GSM8K
*drops* below OPD, the `α` is too large and is diluting the mode-seeking commitment, and I would push
`α` back toward the small end of the convex interior. And if *both* move up, the bounded-gradient story
is exactly right and the endpoint is the clean synthesis of the whole ladder: OPD's commitment with
GKD's stability. The clean win — GSM8K ≥ 0.4852 and MATH-500 > 0.310 — is the one the bounded-gradient
mechanism predicts, and it is what I would validate first.
