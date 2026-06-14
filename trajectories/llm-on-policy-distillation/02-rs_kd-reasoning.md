The hard-target floor landed where its construction said it would: `baseline:dagger` reads GSM8K
0.4511, MATH-500 0.290, AMC 0.0813. It functions — 595 of 1319 grade-school problems solved — but it
is the bottom of the pile on the two reliable metrics, and the MATH-500 number is the tell. On the
harder set, where the chains are long and the teacher's per-token distribution is genuinely
multi-modal, collapsing that distribution to its argmax and demanding one-hot cross-entropy left the
student with the least signal of any loss I will try: 0.290 is a hair above the eventual sparse-soft
result and below everything that uses the teacher's mass faithfully. And the AMC 0.0813 being the
*highest* AMC of any baseline is exactly the noise I warned about — three solved problems out of 40
on an avg@8 protocol, not a real signal. So the diagnosis is clean: the hard target threw away the
teacher's soft distribution, and on the problems that need it most that cost showed. The obvious next
move is to stop discarding the distribution — give the student soft per-token targets that preserve
the relative mass on the alternatives. The question is *how much* of the distribution I can afford to
carry, because that is where the next wall is.

Here is the thing that actually blocks me, and it is not modeling, it is the cost of touching the
whole distribution at every token. The faithful soft target is the teacher's full vocabulary
distribution, and the Qwen vocabulary is ~152k entries. Materializing a forward KL over the full
152k-wide distribution at every one of the millions of completion tokens in a 2000-step run is
expensive in compute and, more pointedly, the natural way to make soft distillation *cheap* — cache
the teacher's distribution once and reuse it — is hopeless at full width, because a full-width soft
target per token is enormous. The dagger floor avoided this entirely (one argmax id per token) but
paid for it in lost signal. So I want the dagger result's cheapness with the soft target's signal,
which means a *sparse* summary of the teacher's distribution per token — a handful of numbers — that
nevertheless reproduces what full-distribution distillation would have done.

The obvious sparse summary, and the one everybody reaches for, is top-`K`: keep the `K` highest-
probability teacher tokens and throw the rest away. It feels right and I can even justify it locally.
If I am forced to keep only `K` entries and renormalize them, which `K` minimize the `L1` error to
the true distribution? Writing the kept-and-renormalized vector and summing the error, the `L1` error
comes out as `2(1 − a)` where `a` is the kept mass, so to minimize it I keep the `K` *largest*
probabilities — top-`K` is `L1`-optimal. The intuition checks out, and yet the known failure of
top-`K` caches is exactly what `L1` does not predict: as `K` shrinks the student becomes
*over-confident* and miscalibrated, and you need far more kept tokens than the budget wants. So
"least `L1` error per token" is the wrong yardstick; something about the truncation poisons the
target in a way reconstruction error cannot see. Let me find it in the gradient, because the gradient
is what trains the model.

The forward-KL logit gradient against a target `t` is `(Σ_i t_i) p_j − t_j` (I keep the leading sum
un-simplified, because the moment I truncate, that sum stops being one). With a top-`K` target the
kept mass is `a = Σ_{i∈K} t_i < 1`, so the gradient is `a·p_j − t_j`. For a token *outside* the kept
set, `t_j = 0`, so the gradient is `a·p_j`, which only vanishes at `p_j = 0` — the loss actively
drives every non-kept token's probability to zero, including rare ground-truth math tokens that
happened to fall outside the top-`K`. For a token *inside* the set, the gradient `a·p_j − t_j`
vanishes at `p_j = t_j / a`, the teacher probability scaled *up* by `1/a > 1`. So the student is
trained to be over-confident on the head, inflated by `1/a`, and driven to zero on the tail. That is
the miscalibration, derived — and it gets worse the smaller `K` is, exactly the observed pathology.
This is the same disease the hard target had in a different costume: dagger over-sharpened by
collapsing to one token; naive top-`K` over-sharpens by sub-normalizing the head. Truncation, in both
cases, manufactures over-confidence.

So I cannot just truncate. But I am committed to keeping only a handful of tokens for cost. The cure
is to stop pretending the kept mass is the whole story and explicitly carry the leftover as one extra
"tail" bucket, on *both* sides. Hold out the top-`K` teacher log-probs, and append a single bucket
holding the residual mass `log(1 − Σ_{i∈K} p_T(i))`; build the student's `K+1`-vector on the *same*
`K` support, with its own residual `log(1 − Σ_{i∈K} p_S(i))`. Now both sides are proper `(K+1)`-
element distributions, and I take the KL over those `K+1` buckets. Let me check the gradient this
restores. The tail term's derivative enters only through the student residual, and working it out, on
the kept tokens the tail bucket contributes exactly the `(1 − a)·p_j` that, added to the top-`K`
gradient `a·p_j − t_j`, recombines to `p_j − t_j` — the *full-distribution* gradient. The `1/a`
up-scaling is gone; the head is supervised faithfully. On the tail tokens the combined gradient is a
residual correction proportional to the student's own tail mass rather than a hard push to zero. So
the tail bucket cures the over-confidence the same way carrying the leftover always does: it makes
the target a genuine probability vector again, so the leading `Σ t` factor is one and there is no
inflation.

Two implementation details decide whether this is numerically real or a `nan` factory, and they are
where the task's actual loss body lives. First, the residual must be computed in log space without
ever exponentiating the head: `log(1 − Σ exp(logp_topk))` is `log(−expm1(logsumexp(logp_topk)))`,
and the inner `logsumexp` must be clamped strictly below zero (a tiny negative ceiling) so that the
`1 − Σ` inside the log can never hit zero or go negative from floating-point drift — otherwise
`log(0)` blows up on exactly the confident positions where the head mass rounds to one. Second, the
`top-K + tail` decomposition is exact in expectation but numerical error can push the `(K+1)`-vector
slightly off a unit sum, so I renormalize both sides in log space (subtract their own logsumexp)
before the divergence, keeping the KL well-defined. Then the per-token loss is `KL(p_T ‖ p_S)` over
the `K+1` support — forward KL, the mass-covering direction, which is the right one when the goal is
to reproduce the teacher's *whole* distribution including its tail. I pin `K = 128`: large enough
that the head plus tail bucket captures almost all the teacher's mass on these peaked math
distributions, small enough to keep memory tame against the 152k vocabulary, and close to the
open-source RS-KD default.

I should be explicit about how this realization differs from the unbiased-sampling story that names
the method, because the name "random sampling KD" suggests drawing tokens from the teacher and
counting, and that is *not* what this task's loss does. The sampling view says: the cure for top-`K`'s
bias is to treat the summary as an importance-sampling estimate whose proposal must have support
everywhere the teacher does — sample token ids from the teacher, cache the empirical `counts/N`, and
get an *unbiased* target whose gradient equals full distillation in expectation, tail included. That
is the cleanest derivation of *why* truncation is biased (it is a proposal that is zero on the entire
tail, violating the support condition for unbiasedness). But the loss I actually fill here is the
*deterministic top-`K` + explicit tail bucket* realization — pick the `K` largest, lump the rest into
one residual bucket on both sides, KL over `K+1`. That is the `_add_tail_bucket` trick, and it
addresses the *same* root cause (the sub-stochastic target and its `1/a` over-scaling) by a different
mechanism: instead of an unbiased random support, a fixed top-`K` support with the leftover mass
carried so the target is normalized. It is biased relative to the pure sampling estimator — the tail
is one undifferentiated bucket, with no per-token tail signal — but it removes the over-confidence
that broke naive top-`K`, and it is fully deterministic and cache-friendly, which is what the cost
constraint demanded. I note the gap so I do not over-claim: this loss recovers the *head* gradient
faithfully and gets the *total* tail mass right, but it does not reconstruct the teacher's tail
*distribution* the way the sampling estimator does in expectation. For peaked math-teacher
distributions, where almost all mass sits in the top tokens, that is a small price; on flatter
distributions it would matter more.

Now the falsifiable expectations against `dagger`. RS-KD here uses real soft targets on the head —
the teacher's relative mass on its top 128 tokens, faithfully, with no `1/a` inflation — so it should
beat the hard target wherever the soft structure carries information. But the tail is collapsed and
the divergence is the mass-covering forward KL, which under the 0.5B/7.6B capacity gap spreads the
student's limited mass to cover the teacher's support rather than committing to a mode. So I expect a
*modest* lift over dagger, not a dramatic one: GSM8K should edge up from 0.4511 (dagger barely had to
use soft structure on short arithmetic chains, so the head-faithful target helps only a little), and
MATH-500 should rise from dagger's 0.290 toward the low-0.29s as the soft head replaces the spurious
one-hot sharpening — but it will *not* reach the reverse-KL methods, precisely because forward KL
keeps it mass-covering exactly where a small student should be mode-seeking, and the lumped tail
denies it the rare-token signal that long competition chains need. If RS-KD lands just above dagger on
GSM8K and roughly level on MATH-500, that is the signal that head-faithful soft targets help but the
*divergence direction* and the *fixed-target* framing are the next things to attack: the ladder
should move next to a loss that adapts the target to the student's reach, and then to one that flips
the divergence to mode-seeking. AMC will stay noise; I will not read it.
