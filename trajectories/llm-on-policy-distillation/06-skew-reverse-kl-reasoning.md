OPD closed the ladder where its construction pointed, and the residual it left is a specific,
fixable instability rather than a vague "try harder." `baseline:opd` reads GSM8K 0.4852, MATH-500
0.310, AMC 0.0688. The GSM8K prediction landed: 0.4852 is the best on the ladder, clearing GKD's
0.4716 by the larger margin I expected once the divergence went all the way to reverse KL — full
commitment to the teacher's preferred continuation pays most on short, near-deterministic arithmetic
chains. And MATH-500 came in at 0.310, a hair *under* GKD's 0.312 — exactly the mode-collapse risk I
flagged: pure reverse KL overshot half-JSD's balanced compromise and the student committed to slightly
too few of the teacher's reasoning modes on the hard set. So the picture across the whole ladder is
now unambiguous. The divergence direction is the binding axis; more mode-seeking lifts the headline
GSM8K metric; but *unbounded* reverse KL pays a small MATH-500 tax because it overshoots into
collapse. That is the precise gap I want to close: keep OPD's mode-seeking commitment, which won
GSM8K, but stop it from collapsing on the hard set — and the way OPD overshoots tells me the
mechanism, so let me look at *why* unbounded reverse KL is brittle, in the gradient.

OPD trains reverse KL `KL(p_S ‖ p_T)` on the student's own on-policy rollouts. Write its gradient. The
KL is `Σ_v p_S·(log p_S − log p_T)`, the parameters sit in both the outer weight `p_S` and the inner
`log p_S`, and collapsing onto the sampled sequence gives a per-token coefficient of the form
`log r + 1` where `r = p_S/p_T` is the student-to-teacher ratio. Now read where that coefficient runs
away. It blows up when `p_T → 0` — when the teacher assigns near-zero probability to what the student
produced. And here is the thing the on-policy loop guarantees: the batches are the *student's own*
generations, sequences the 7.6B teacher did not write, so on a self-generated math chain the teacher is
frequently surprised — `p_T ≈ 0` is not a rare edge case here, it is the *typical* case on exactly the
data OPD trains on. So OPD's gradient coefficient `log(p_S/p_T)` detonates on its own training data
precisely where the student is exploring, throwing huge, noisy steps that yank it toward whatever
narrow region the teacher does endorse — which is the mechanism of the mode collapse I measured on
MATH-500. The unbounded reverse KL is not just *theoretically* brittle; its instability lives in the
same on-policy generations that make OPD strong on GSM8K. That is the lever.

Name the source of the blow-up exactly, because the fix follows from it. The coefficient explodes
because a *raw probability sits in a denominator and goes to zero* — the teacher probability `p_T` in
the ratio `p_S/p_T`. A raw distribution can be arbitrarily close to zero on an off-distribution token.
So the structural cure is to not let the raw teacher distribution be the denominator: replace it with a
*mixture* that always contains a sliver of the student, so the denominator is floored away from zero. If
the comparison distribution is `p̃ = (1−α)·p_T + α·p_S` for a small `α`, then on a token where `p_T ≈ 0`
the mixture still carries the `α·p_S` leg, so `p̃ ≥ α·p_S`, the ratio `p_S/p̃ ≤ 1/α` is *bounded*, and
the log can no longer diverge. This is a reverse KL skewed toward a mixture rather than the raw teacher —
take the reverse KL of the student against `p̃`: `KL(p_S ‖ (1−α)·p_T + α·p_S)`. At `α = 0` it is exactly
OPD's reverse KL; as `α` grows the floor rises and the gradient tames.

Let me check the skewed gradient actually does what I claim, because if the bound is real this is a
clean win over OPD with no other moving part. Differentiate `KL(p_S ‖ p̃)` where `p̃ = (1−α)·p_T +
α·p_S` and the `α·p_S` leg also depends on the parameters. The reverse-KL product rule still gives the
`log r + 1` structure, but `r` is now `p_S/p̃`, and the `θ`-dependence in the `α·p_S` leg of `p̃` adds a
term, leaving a coefficient `log r + 1 − α·r` with `r = p_S/p̃`. Compare to OPD's `log r + 1`. Two
things help at once. First, `r = p_S/p̃ ≤ 1/α` is bounded, so the `log r` term cannot run to infinity
the way `log(p_S/p_T)` did when the teacher vanished. Second, the new `−α·r` term *subtracts* and grows
with `r`, actively pulling the coefficient back down exactly where it would otherwise spike. So skewed
reverse KL has a bounded, well-behaved gradient on precisely the teacher-surprised on-policy tokens that
made OPD's MATH-500 collapse — it keeps reverse KL's mode-seeking weighting (still weighted by `p_S`,
still zero-forcing, still committing to the teacher's modes, which is what won GSM8K) but removes the
detonation that overshot into collapse.

Now I should worry about the opposite failure, because a floor that is too high stops me distilling the
teacher at all. There is a second consideration too: the loss is estimated on mini-batches, and a
high-variance estimator gives a noisy objective. So how big should `α` be? The empirical L2 error of the
skewed estimator, under mild assumptions, carries inverse-`α` terms — moving `α` away from zero shrinks
the estimation-error contribution that comes from the unprotected raw denominator, so a *larger* `α`
both bounds the gradient and tightens the mini-batch estimate. That argues for `α` not too small. But
pushing `α` toward one is wrong, for a reason specific to how the optimizer moves: an Adam-style
optimizer normalizes by a running estimate of the gradient scale, so a uniformly smaller gradient
coefficient is divided back out — the gradient-shrinkage "benefit" of large `α` is largely compensated
away. Looking at the estimation error in the units the optimizer actually steps in — the L2 norm
normalized by the gradient scale — the inverse-`α` pieces want `α` large while inverse-`(1−α)` pieces
want `α` small, giving a convex trade-off with an interior optimum: small enough that the target is
still mostly the teacher (so I am distilling the teacher, not a teacher-student blend), large enough to
floor the denominator and cut the variance. A mild value, `α = 0.1`, sits in that interior — the
comparison distribution is `p̃ = 0.9·p_T + 0.1·p_S`, 90% faithful to the teacher, with just enough
student mixed in to bound the ratio. So `α = 0.1`.

This also tells me why skewing is not the same move as GKD's generalized JSD, which I already have on the
ladder, so I am not just re-running rung four with a different number. JSD is a *sum of two skewed KLs
whose skew parameters are tied to one `β`* — schematically `β·SKL^β + (1−β)·SRKL^{1−β}` — so the single
`β` that sets the skew of the forward leg forces the complementary skew on the reverse leg. I cannot
make *both* legs mildly skewed at once: if I want a mild `α = 0.1` skew on the reverse term, JSD forces
`0.9` on the forward term, slamming that leg into the high-skew regime. The estimation-vs-`α` analysis
says I want a *mild* skew on the one term I am actually using; a single freely-chosen-`α` skewed reverse
KL reaches that mild interior optimum, and JSD's coupled parameter structurally cannot. GKD's `β = 0.5`
balanced two unskewed KL directions; this skews *one* mode-seeking direction by a small, freely tuned
amount. Different knob, and the one the OPD failure mode actually calls for.

Now make it concrete in this task's edit surface, because I only get to fill the `compute_distill_loss`
body and nothing else — no replay buffer, no SGO scheduler, no `lmbda` change. The full streamlined
recipe this loss comes from pairs the skewed KL with two data-side mechanisms — an adaptive SGO
probability that ramps up self-generation guided by validation loss, and an off-policy replay buffer
with a decaying replay ratio to amortize generation cost — and those need framework-level access to the
data pipeline that this single-loss surface does not expose. So I drop them and land only the *loss*: the
skewed reverse KL at `α = 0.1`, computed on whatever batch the trainer's static `lmbda` mixing produced,
exactly as every prior rung consumed that mixing. The synergy is real (the skewed loss's fast, stable
early movement is what would let an off-policy buffer work) but it is not mine to build here; the
per-token loss stands on its own.

The arithmetic mirrors OPD's reverse-KL body with the denominator swapped for the mixture. Divide both
logit tensors by the shared temperature, softmax to get `p_T` and `p_S`, form the mixture `p̃ =
(1−α)·p_T + α·p_S` in probability space and take its log, then accumulate the per-token reverse KL
`Σ_v p_S·(log p_S − log p̃)` — keeping *both* legs, because unlike a forward KL the student-entropy leg
`Σ p_S·log p_S` depends on the parameters and carries the `+1` normalization gradient I derived, so I
must not drop it. Guard any `±inf` logit positions so they contribute zero rather than `nan`, sum over
the vocabulary to a per-token value, mask to the completion tokens (`labels ≠ −100`), and average per
token for `batchmean`. The direction is pinned by construction: the student is the outer weight and the
mixture is the comparison, so this is genuinely `KL(p_S ‖ p̃)`, the skewed *reverse* direction, not its
forward cousin. The full scaffold body is in the answer.

What this has to clear, stated against the measured ladder, with no numbers invented. The bar is OPD,
the strongest baseline: GSM8K 0.4852, MATH-500 0.310. The falsifiable claim is asymmetric and it
follows directly from the mechanism. On GSM8K the skewed reverse KL keeps OPD's mode-seeking weighting,
so it should *hold* OPD's level — I would not expect the skew to cost the GSM8K commitment that reverse
KL bought, since `α = 0.1` leaves the target 90% teacher and the direction unchanged; if anything the
bounded gradient should let it train a touch cleaner, so I expect it to match or modestly exceed 0.4852.
The real test is MATH-500: the entire reason to skew is that OPD's unbounded gradient overshot into
mode collapse there (0.310, under GKD), and the floor `p̃ ≥ α·p_S` is designed to stop exactly that
detonation on teacher-surprised on-policy tokens. So the prediction that would vindicate this rung is
MATH-500 recovering *above* OPD's 0.310 — back to or past GKD's 0.312 — while GSM8K holds OPD's lead.
If instead MATH-500 stays pinned at OPD's level, the diagnosis would be that the collapse is not driven
by the gradient blow-up but by the reverse direction itself, and the skew is treating the wrong cause;
and if GSM8K *drops* below OPD, the `α` is too large and is diluting the mode-seeking commitment. The
clean win — GSM8K ≥ 0.4852 and MATH-500 > 0.310 — is the one the bounded-gradient story predicts, and it
is what I would validate first.
