The hard-target floor landed where its construction said it would: `baseline:dagger` reads GSM8K
0.4511, MATH-500 0.290, AMC 0.0813. Let me read those numbers with the noise floor I fixed at the
last rung, because the whole point of establishing a floor is to measure the next climb against it
honestly. GSM8K 0.4511 is 595 of 1319; its standard error near one-half is `sqrt(0.4511·0.5489/1319)
≈ 0.0137`, so about `±18` problems — anything under a percentage point of movement on GSM8K is inside
the noise. MATH-500 0.290 is 145 of 500, standard error `sqrt(0.29·0.71/500) ≈ 0.020`, about `±10`
problems. AMC 0.0813 is three solved problems out of 40 on avg@8, where one problem is `2.5%`, so it
is pure noise — and its being the *highest* AMC I will see for a while is exactly the small-sample
artifact I promised to ignore, not a signal that the crudest loss is somehow best at competition
problems. So the readable story is: dagger functions (595 grade-school problems solved) but sits at
the bottom on the two reliable metrics, and the MATH-500 number is the tell — on the harder set,
where chains are long and the teacher's per-token distribution is genuinely multi-modal, collapsing
that distribution to its argmax and demanding one-hot cross-entropy left the student with the least
signal of any loss I will try. The diagnosis is clean: the hard target threw away the teacher's soft
distribution, and on the problems that need it most that cost showed. The obvious next move is to stop
discarding the distribution — give the student soft per-token targets that preserve the relative mass
on the alternatives. The question is *how much* of the distribution I can afford to carry, because
that is where the next wall is.

Here is the thing that actually blocks me, and it is not modeling, it is the cost of touching the
whole distribution at every token. The faithful soft target is the teacher's full vocabulary
distribution, and the Qwen vocabulary after the crop is `V = 151936` entries. Materializing a forward
KL over the full ~152k-wide distribution at every completion token is not the expensive part by
itself — the scaffold default already does exactly that — the expensive part is that the natural way
to make soft distillation *cheap* is to cache the teacher's soft target once and reuse it across the
run, and a full-width soft target is hopeless to cache. Price it: the run processes on the order of
`32·2000 = 64000` sequences, each a few hundred completion tokens, so caching a full-width float
target is `~64000 × few-hundred × 151936` floats — astronomically large, tens of terabytes, plainly
off the table. The dagger floor avoided this entirely (one argmax id per token, kilobytes) but paid
for it in lost signal. So what I want is the dagger result's cheapness with the soft target's signal,
which means a *sparse* summary of the teacher's distribution per token — a handful of numbers — that
nevertheless reproduces what full-distribution distillation would have done.

The obvious sparse summary, and the one everybody reaches for, is top-`K`: keep the `K` highest-
probability teacher tokens and throw the rest away. It feels right and I can even justify it locally.
If I am forced to keep only `K` entries and renormalize them, which `K` minimize the `L1` error to
the true distribution? Let the kept set be `S` with mass `a = Σ_{i∈S} p_i`; the reconstruction puts
`p_i/a` on kept tokens and `0` on the rest, so the `L1` error is `Σ_{i∈S} |p_i/a − p_i| + Σ_{i∉S} p_i
= Σ_{i∈S} p_i·(1/a − 1) + (1 − a) = a·(1−a)/a + (1−a) = 2(1 − a)`. Minimizing `2(1−a)` means
maximizing the kept mass `a`, which is achieved by keeping the `K` *largest* probabilities — top-`K`
is `L1`-optimal. Let me sanity-check the algebra on a tiny case so I trust it: `p = (0.5, 0.3, 0.15,
0.05)`, `K = 2` keeps `{0.5, 0.3}`, `a = 0.8`, renormalized `(0.625, 0.375)`; the `L1` error is
`|0.625−0.5| + |0.375−0.3| + 0.15 + 0.05 = 0.125 + 0.075 + 0.2 = 0.4`, which is exactly `2(1−0.8) =
0.4`. So the intuition checks out — and yet the known failure of top-`K` caches is precisely what
`L1` does not predict: as `K` shrinks the student becomes *over-confident* and miscalibrated, and you
need far more kept tokens than the budget wants. So "least `L1` error per token" is the wrong
yardstick; something about the truncation poisons the target in a way reconstruction error cannot
see. Let me find it in the gradient, because the gradient is what trains the model.

The forward-KL logit gradient against a target `t` is `(Σ_i t_i) p_j − t_j` (I keep the leading sum
un-simplified, because the moment I truncate, that sum stops being one). With a top-`K` target the
kept mass is `a = Σ_{i∈K} t_i < 1`, so the gradient is `a·p_j − t_j`. For a token *outside* the kept
set, `t_j = 0`, so the gradient is `a·p_j`, which only vanishes at `p_j = 0` — the loss actively
drives every non-kept token's probability to zero, including rare ground-truth math tokens that
happened to fall outside the top-`K`. For a token *inside* the set, the gradient `a·p_j − t_j`
vanishes at `p_j = t_j / a`, the teacher probability scaled *up* by `1/a > 1`. Let me put a number on
the inflation so I know whether it is a rounding effect or a real distortion: if the kept mass is
`a = 0.8` and a head token carries teacher probability `t_j = 0.3`, the student's stationary point is
`0.3/0.8 = 0.375` — a `25%` over-statement of that token's probability, and it worsens as `a` falls,
which is exactly what "you need more kept tokens than the budget wants" means in practice. This is the
miscalibration, derived, and it gets worse the smaller `K` is, matching the observed pathology. It is
the same disease the hard target had in a different costume: dagger over-sharpened by collapsing to
one token; naive top-`K` over-sharpens by sub-normalizing the head. Truncation, in both cases,
manufactures over-confidence — and having just watched dagger pay for over-confidence on MATH-500, I
am not going to walk into the same wall by a different door.

So I cannot just truncate. But I am committed to keeping only a handful of tokens for cost. The cure
is to stop pretending the kept mass is the whole story and explicitly carry the leftover as one extra
"tail" bucket, on *both* sides. Hold out the top-`K` teacher log-probs, and append a single bucket
holding the residual mass `log(1 − Σ_{i∈K} p_T(i))`; build the student's `K+1`-vector on the *same*
`K` support, with its own residual `log(1 − Σ_{i∈K} p_S(i))`. Now both sides are proper `(K+1)`-
element distributions, and I take the KL over those `K+1` buckets. Let me check the gradient this
restores. The tail term's derivative enters only through the student residual, and working it out, on
the kept tokens the tail bucket contributes exactly the `(1 − a)·p_j` that, added to the top-`K`
gradient `a·p_j − t_j`, recombines to `a·p_j − t_j + (1−a)·p_j = p_j − t_j` — the *full-distribution*
gradient, the one whose stationary point is `p_j = t_j` with no `1/a` factor anywhere. Re-run the
number from before: the head token with `t_j = 0.3` now has its fixed point at `p_j = 0.3`, not
`0.375` — the `25%` inflation is gone. On the tail tokens the combined gradient is a residual
correction proportional to the student's own tail mass rather than a hard push to zero, so the rare
math tokens outside the top-`K` are no longer driven to extinction; they are held at whatever share of
the `(1−a)` residual the student assigns them. So the tail bucket cures the over-confidence the same
way carrying the leftover always does: it makes the target a genuine probability vector again, so the
leading `Σ t` factor is one and there is no inflation.

Before I commit to this exact construction I should walk the couple of alternatives that could carry
the same idea, because the tail bucket is a specific choice and I want to know it is the right one and
not just the first. The cleanest theoretical object is the *unbiased sampling* estimator: instead of
keeping the deterministic top-`K`, draw token ids from the teacher and cache the empirical `counts/N`,
which gives a target whose gradient equals full distillation in expectation, tail included, because a
sampling proposal has support everywhere the teacher does. That is genuinely the cleanest derivation
of *why* truncation is biased — a proposal that is identically zero on the entire tail violates the
support condition unbiasedness needs — but it buys that unbiasedness with sampling variance and a
non-deterministic, harder-to-cache target, and on peaked math distributions where almost all mass
sits in a few head tokens the variance reduction from more samples is exactly where the effort goes.
The other alternative is to keep more head tokens and skip the tail bucket, but the gradient
derivation just showed that no finite `K` removes the `1/a` inflation — it only shrinks `1−a`, so
"keep more tokens" is fighting the symptom with budget rather than fixing the sub-stochastic target.
The tail bucket fixes the target with a single extra number on each side, which is why I take it over
both: it recovers the head gradient exactly and gets the *total* tail mass right, deterministically
and cache-friendly, which is what the cost constraint demanded.

Two implementation details decide whether this is numerically real or a `nan` factory, and they are
where the task's actual loss body lives. First, the residual must be computed in log space without
ever exponentiating the head: `log(1 − Σ exp(logp_topk))` is `log(−expm1(logsumexp(logp_topk)))`,
and the inner `logsumexp` must be clamped strictly below zero (a tiny negative ceiling) so that the
`1 − Σ` inside the log can never hit zero or go negative from floating-point drift. Let me trace the
two regimes so I know the clamp is load-bearing and not decorative. On a moderately peaked position
the head mass might be `a = 0.999`; then `logsumexp(logp_topk) = log(0.999) ≈ −1.0e-3`, `expm1(−1.0e-3)
≈ −1.0e-3`, `−expm1 = 1.0e-3`, and the tail bucket is `log(1.0e-3) ≈ −6.9` — finite, sensible. But on
a razor-peaked position where the top-`K` mass rounds to exactly `1.0` in bf16, `logsumexp` computes
as `0.0`, `expm1(0) = 0`, and `log(0) = −inf`, which propagates to a `nan` in the KL on precisely the
*confident* positions where the head rounds to one — the worst place to blow up. The clamp at a
ceiling like `−1e-7` forces `logsumexp ≤ −1e-7`, so `−expm1(−1e-7) ≈ 1e-7` and the tail bucket is
`log(1e-7) ≈ −16.1`, finite. So the clamp is exactly what keeps the confident positions from
detonating. Second, the `top-K + tail` decomposition is exact in expectation but numerical error can
push the `(K+1)`-vector slightly off a unit sum, so I renormalize both sides in log space (subtract
their own logsumexp) before the divergence, keeping the KL well-defined. Then the per-token loss is
`KL(p_T ‖ p_S)` over the `K+1` support — forward KL, the mass-covering direction, which is the right
one when the goal is to reproduce the teacher's *whole* distribution including its tail. I pin
`K = 128`: large enough that the head plus tail bucket captures almost all the teacher's mass on these
peaked math distributions, small enough to keep memory tame against the 152k vocabulary — a `128+1`
vs `151936` support is a `~1180×` reduction in target width, the whole reason the sparse summary was
worth deriving — and close to the open-source RS-KD default.

Two limits confirm the `K` knob is a genuine dial and not a hack. Push `K → V`: the top-`K` captures
everything, the residual mass `1 − a → 0` on both sides, the tail bucket vanishes, and the `K+1` KL
collapses to the exact full-vocabulary forward KL it was meant to approximate — so at maximum budget I
recover the faithful object, which is the correct behavior for an approximation. Pull `K → 1`: I am
left with a two-bucket KL, the teacher's top token against everything-else, a coarse but still
*normalized* soft target — and, crucially, *not* the dagger floor, which was one-hot cross-entropy
with no tail at all. So the tail-bucketed top-`K` family is an honest interpolation between the full
forward KL (`K = V`) and a binary head/tail split (`K = 1`), with the hard target sitting off it
entirely — it was the `τ → 0` sharpening of the teacher, a different axis. `K = 128` sits near the
full-KL end of that dial, which is where I want it: faithful head, negligible residual on peaked math
distributions, and a memory cost roughly `1180×` under the full width.

Let me put the `K = 128` choice on firmer ground than "near the open-source default," by asking what
mass the head actually captures on these distributions and where the captured-mass curve flattens. The
teacher is a 7.6B math-tuned model scoring curated solution tokens, so its next-token distribution is
sharply peaked — a top token often carrying 0.5 or more and the mass decaying fast below it. On such a
distribution the top-32 already captures the overwhelming bulk, but the residual it leaves — a few
percent — is exactly what my single tail bucket has to lump, and a few percent lumped is where the flat
on-policy positions bleed signal. Doubling to top-128 pushes the captured head mass well above 0.99 on
the peaked positions, shrinking the lumped residual roughly fourfold, and doubling again to top-512
would recover only the last sliver of an already-negligible tail while quadrupling the target width and
the `topk`/`gather` footprint. So the captured-mass gain per token is steeply diminishing past ~128
while the memory cost is linear in `K`, which is the classic sign the knob should sit at the elbow:
`K = 128` is where the head is faithful enough that the residual is small and the width is still
`~1180×` under the full vocabulary. If MATH-500 later demanded more tail fidelity I would find it out by
raising `K` and watching the lumped residual shrink, not by guessing — but the diminishing-returns shape
says I should not expect much there, which is itself part of why I predict only a modest lift.

Unlike the hard target, this loss *does* use the temperature the signature passes, and it must,
because a soft target is a genuine distribution and temperature is how I control its sharpness. I
divide both logit tensors by the shared `temperature` before the softmaxes; at the default `0.9 < 1`
that sharpens both sides symmetrically. Sharpening *symmetrically* is the point — dividing teacher and
student by the same constant changes the scale on which the KL measures their mismatch without tilting
it toward either distribution, and it does not even change which tokens land in the top-`K`, since a
positive rescale of the logits preserves their order. So the temperature here is a shared knob on the
head-vs-tail weighting, not a lever that could quietly bias the target the way it would if I applied
it to only one side. I keep it shared so the divergence measures a behavioral gap and not a sharpness
mismatch I introduced myself.

Two practical notes before I run it, because they decide whether the cost argument that motivated the
whole construction actually holds and how I will read the result. The `torch.topk` over the 152k
vocabulary at every position is `O(V)` work to surface 128 entries, negligible against the matmul that
produced the logits in the first place, so the sparse summary costs essentially nothing in FLOPs — its
entire value is in the `K+1` vs `V` memory and cache footprint, not in compute, which is exactly the
axis the caching argument was about. And when the run reports RS-KD's final training loss I must not
compare it to dagger's `0.507` as though lower were better: dagger's number is a per-token cross-
entropy against a one-hot, RS-KD's is a forward KL over `K+1` buckets, and the two measure different
objectives on different supports. Only the downstream accuracies are commensurable across losses; the
train-loss column is a within-loss convergence diagnostic, not a cross-loss ranking, and I will read
it that way for the rest of the ladder.

I should be explicit about how this realization differs from the unbiased-sampling story that names
the method, because the name "random sampling KD" suggests drawing tokens from the teacher and
counting, and that is *not* what this task's loss does. The sampling view says: the cure for top-`K`'s
bias is to treat the summary as an importance-sampling estimate whose proposal must have support
everywhere the teacher does — sample token ids from the teacher, cache the empirical `counts/N`, and
get an *unbiased* target whose gradient equals full distillation in expectation, tail included. But
the loss I actually fill here is the *deterministic top-`K` + explicit tail bucket* realization —
pick the `K` largest, lump the rest into one residual bucket on both sides, KL over `K+1`. That is the
`_add_tail_bucket` trick, and it addresses the *same* root cause (the sub-stochastic target and its
`1/a` over-scaling) by a different mechanism: instead of an unbiased random support, a fixed top-`K`
support with the leftover mass carried so the target is normalized. It is biased relative to the pure
sampling estimator — the tail is one undifferentiated bucket, with no per-token tail signal — but it
removes the over-confidence that broke naive top-`K`, and it is fully deterministic and cache-
friendly, which is what the cost constraint demanded. I note the gap so I do not over-claim: this loss
recovers the *head* gradient faithfully and gets the *total* tail mass right, but it does not
reconstruct the teacher's tail *distribution* the way the sampling estimator does in expectation. For
peaked math-teacher distributions, where almost all mass sits in the top tokens, that is a small
price; on flatter distributions it would matter more.

One more mechanism decides where this loss will fall short, and it is the same on-policy interaction
that shadowed the floor. Half the batches are the student's own rollouts, and on a student-generated
prefix the 7.6B teacher is scoring a context it would not have written, so its next-token distribution
there is flatter and more hedged than on the curated dataset half. A flatter teacher distribution puts
*more* mass outside the top-128, which is exactly the mass my single tail bucket lumps into one
undifferentiated number. So the top-`K` + tail approximation is *tightest* on the peaked dataset
positions and *loosest* on the flat on-policy ones — and the flat on-policy positions are
disproportionately the hard, long, off-distribution completions where MATH-500 lives. The construction
that makes RS-KD cheap is therefore least faithful exactly where dagger already struggled, which is a
second reason I expect the MATH-500 lift to be small: I am carrying the head faithfully, but on the
positions that matter most the head is a smaller fraction of the story, and the part I lump is the
part that grows.

Now the falsifiable expectations against `dagger`. RS-KD here uses real soft targets on the head —
the teacher's relative mass on its top 128 tokens, faithfully, with no `1/a` inflation — so it should
beat the hard target wherever the soft structure carries information. But two things temper how much I
expect. First, the divergence is still the mass-covering forward KL, which under the 0.5B/7.6B
capacity gap spreads the student's limited mass to cover the teacher's support rather than committing
to a mode. Second, dagger already did well on GSM8K precisely because short arithmetic chains have
unambiguous, peaked teacher argmaxes — exactly the positions where the soft head adds *least*, because
a peaked distribution's top token already carries most of the mass. So I expect a *modest* lift over
dagger, not a dramatic one, and I can bound my expectation with the noise floor: GSM8K should edge up
from 0.4511, but a move smaller than the `±18`-problem standard error would sit inside the noise, so I
am looking for something on the order of a handful of problems, not a jump. MATH-500 should rise from
dagger's 0.290 toward the low-0.29s as the soft head replaces the spurious one-hot sharpening — but it
will *not* reach a mode-seeking loss, precisely because forward KL keeps it mass-covering exactly
where a small student should be committing, and the lumped tail denies it the rare-token signal that
long competition chains need. If RS-KD lands just above dagger on GSM8K and roughly level on MATH-500,
that is the signal that head-faithful soft targets help but the *divergence direction* and the
*fixed-target* framing are the next things to attack: the ladder should move next to a loss that
adapts the target to the student's reach, and then to one that flips the divergence toward mode-
seeking. If instead RS-KD *matched* dagger to within noise on both metrics, that would be the more
interesting falsification — it would say the soft head carries almost nothing for this pair and the
bottleneck was never "hard vs soft" at all, which would send me straight to the target and the
direction. AMC will stay noise; I will not read it.
