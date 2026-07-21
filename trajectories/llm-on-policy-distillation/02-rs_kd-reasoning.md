The hard-target floor landed where its construction said: `baseline:dagger` reads GSM8K 0.4511,
MATH-500 0.290, AMC 0.0813. Read against the noise floor — GSM8K 595/1319 has a standard error near
one-half of about `±18` problems, MATH-500 145/500 about `±10`, AMC three solved of 40 is pure noise
— the readable story is that dagger functions (595 grade-school problems solved) but sits at the
bottom on both reliable metrics, and MATH-500 is the tell: on the harder set, where chains are long
and the teacher's per-token distribution is genuinely multi-modal, collapsing it to the argmax and
demanding one-hot cross-entropy left the student the least signal of any loss I will try. AMC being
the *highest* I will see for a while is the small-sample artifact I promised to ignore, not a signal
that the crudest loss is best at competition problems. The diagnosis is clean: the hard target threw
away the teacher's soft distribution, and on the problems that need it most that cost showed. The next
move is to stop discarding the distribution — give the student soft per-token targets that preserve
the relative mass on the alternatives. The question is *how much* of the distribution I can afford to
carry, because that is the next wall.

What blocks me is not modeling, it is the cost of touching the whole distribution at every token. The
faithful soft target is the teacher's full vocabulary distribution, `V = 151936` after the crop.
Materializing a full-width forward KL per token is not itself the expensive part — the scaffold
default already does exactly that — the expensive part is that the natural way to make soft
distillation *cheap* is to cache the teacher's soft target once and reuse it across the run, and a
full-width target is hopeless to cache: `~64000` sequences × a few-hundred tokens × `151936` floats
is tens of terabytes, off the table. The dagger floor avoided this entirely (one argmax id per token,
kilobytes) but paid for it in lost signal. So I want dagger's cheapness with the soft target's
signal: a *sparse* summary of the teacher's distribution per token — a handful of numbers — that
nevertheless reproduces what full-distribution distillation would have done.

The obvious sparse summary is top-`K`: keep the `K` highest-probability teacher tokens and throw the
rest away. It even has a local justification — if I must keep only `K` entries and renormalize, which
`K` minimize the `L1` error to the true distribution? With kept mass `a = Σ_{i∈S} p_i` the
reconstruction puts `p_i/a` on kept tokens and `0` on the rest, so the `L1` error is
`Σ_{i∈S} p_i·(1/a − 1) + (1 − a) = 2(1 − a)`, minimized by maximizing `a`, i.e. keeping the `K`
largest — top-`K` is `L1`-optimal. And yet the known failure of top-`K` caches is exactly what `L1`
does not predict: as `K` shrinks the student becomes *over-confident* and miscalibrated, and you need
far more kept tokens than the budget wants. So "least `L1` error" is the wrong yardstick; the poison
is in the gradient, which is what trains the model.

The forward-KL logit gradient against a target `t` is `(Σ_i t_i)·p_j − t_j` (I keep the leading sum
un-simplified, because the moment I truncate it stops being one). With a top-`K` target the kept mass
is `a = Σ_{i∈K} t_i < 1`, so the gradient is `a·p_j − t_j`. For a token *outside* the kept set
`t_j = 0`, so the gradient `a·p_j` vanishes only at `p_j = 0` — the loss actively drives every
non-kept token to zero, including rare ground-truth math tokens that fell outside the top-`K`. For a
token *inside* the set the gradient vanishes at `p_j = t_j/a`, the teacher probability scaled *up* by
`1/a > 1`: at `a = 0.8` a head token with `t_j = 0.3` sits at `0.375`, a `25%` over-statement,
worsening as `a` falls — exactly what "you need more kept tokens than the budget wants" means. This
is the same over-confidence disease dagger had in a different costume: dagger over-sharpened by
collapsing to one token; naive top-`K` over-sharpens by sub-normalizing the head. Having just watched
dagger pay for over-confidence on MATH-500, I will not walk into it again by a different door.

So I cannot just truncate, but I am committed to keeping only a handful of tokens for cost. The cure
is to stop pretending the kept mass is the whole story and carry the leftover as one extra "tail"
bucket on *both* sides: hold the top-`K` teacher log-probs, append a bucket holding the residual
`log(1 − Σ_{i∈K} p_T)`, build the student's `K+1`-vector on the *same* `K` support with its own
residual `log(1 − Σ_{i∈K} p_S)`, and take the KL over the `K+1` buckets. Now the tail term
contributes exactly `(1 − a)·p_j` on the kept tokens, which added to the top-`K` gradient `a·p_j −
t_j` recombines to `p_j − t_j` — the *full-distribution* gradient, fixed point at `p_j = t_j` with no
`1/a` factor. The head token with `t_j = 0.3` now bottoms out at `0.3`, not `0.375`; the inflation is
gone. On the tail tokens the combined gradient is a residual correction proportional to the student's
own tail mass rather than a hard push to zero, so rare math tokens outside the top-`K` are no longer
driven to extinction. The tail bucket makes the target a genuine probability vector again, so the
leading `Σ t` factor is one.

The cleanest theoretical alternative is the *unbiased sampling* estimator that names the method:
instead of the deterministic top-`K`, draw token ids from the teacher and cache the empirical
`counts/N`, whose gradient equals full distillation in expectation, tail included, because a sampling
proposal has support everywhere the teacher does. That is genuinely the cleanest derivation of *why*
truncation is biased — a proposal identically zero on the tail violates the support condition
unbiasedness needs — but it buys that with sampling variance and a non-deterministic, harder-to-cache
target, and on peaked math distributions where almost all mass sits in a few head tokens the variance
reduction is where the effort goes. The other alternative, keep more head tokens and skip the tail
bucket, only shrinks `1 − a`; no finite `K` removes the `1/a` inflation, so it fights the symptom
with budget. The tail bucket fixes the target with a single extra number on each side — recovering
the *head* gradient exactly and getting the *total* tail mass right, deterministically and
cache-friendly, which is what the cost constraint demanded. It is biased relative to the pure
sampling estimator (the tail is one undifferentiated bucket, no per-token tail signal), so it
recovers the head gradient faithfully and the total tail mass but not the teacher's tail
*distribution*; for peaked math-teacher distributions that is a small price, on flatter ones it would
matter more.

Two implementation details decide whether this is numerically real or a `nan` factory. First, the
residual must be computed in log space without exponentiating the head:
`log(1 − Σ exp(logp_topk)) = log(−expm1(logsumexp(logp_topk)))`, with the inner `logsumexp` clamped
strictly below zero. On a razor-peaked position where the top-`K` mass rounds to exactly `1.0` in
bf16, `logsumexp` computes as `0.0`, `expm1(0) = 0`, `log(0) = −inf`, which propagates to a `nan` in
the KL on precisely the *confident* positions — the worst place to blow up. A ceiling like `−1e-7`
forces `logsumexp ≤ −1e-7`, so the tail bucket is `log(1e-7) ≈ −16.1`, finite. Second, the
`top-K + tail` decomposition is exact in expectation but numerical error can push the `K+1`-vector
off a unit sum, so I renormalize both sides in log space (subtract their own logsumexp) before the
divergence. Then the per-token loss is `KL(p_T ‖ p_S)` over the `K+1` support — forward KL, the
mass-covering direction, the right one when the goal is to reproduce the teacher's *whole*
distribution including its tail.

I pin `K = 128`. The two limits show it is a genuine dial, not a hack: `K → V` captures everything,
the residual vanishes on both sides, and the `K+1` KL collapses to the exact full-vocabulary forward
KL it approximates; `K → 1` leaves a two-bucket head-vs-tail KL, coarse but still *normalized* — and
crucially *not* the dagger floor, which was one-hot cross-entropy with no tail. So the tail-bucketed
top-`K` family is an honest interpolation between full forward KL and a binary split, with the hard
target off it entirely. On where to sit: the teacher is a 7.6B math model scoring curated tokens, so
its next-token distribution is sharply peaked; top-32 already captures the bulk, but the few-percent
residual it leaves is exactly what the single tail bucket must lump, and top-128 pushes captured head
mass above 0.99 on peaked positions, shrinking the lumped residual roughly fourfold, while top-512
would recover only a sliver at quadruple the width. Steeply diminishing captured mass past ~128
against linear memory cost is the elbow, and `128+1` vs `151936` is a `~1180×` reduction — the whole
reason the sparse summary was worth deriving. If MATH-500 later demanded more tail fidelity I would
find it by raising `K` and watching the residual shrink, but the diminishing-returns shape says not to
expect much there, which is part of why I predict only a modest lift.

Unlike the hard target this loss *uses* the temperature the signature passes, and it must, because a
soft target is a genuine distribution and temperature controls its sharpness. I divide both logit
tensors by the shared `temperature` before the softmaxes; at the default `0.9 < 1` that sharpens both
sides symmetrically, changing the scale on which the KL measures the mismatch without tilting toward
either distribution, and it does not even change which tokens land in the top-`K`. I keep it shared
so the divergence measures a behavioral gap and not a sharpness mismatch I introduced.

Two practical notes. The `torch.topk` over the 152k vocabulary is `O(V)` work to surface 128 entries,
negligible against the matmul that produced the logits — the sparse summary's entire value is the
`K+1` vs `V` memory footprint, not FLOPs. And when the run reports RS-KD's train loss I must not
compare it to dagger's `0.507`: dagger's is a per-token cross-entropy against a one-hot, RS-KD's a
forward KL over `K+1` buckets — different objectives on different supports. Only downstream accuracies
are commensurable across losses; the train-loss column is a within-loss convergence diagnostic, and I
will read it that way for the rest of this progression.

One mechanism decides where this falls short, the same on-policy interaction that shadowed the floor.
On a student-generated prefix the 7.6B teacher is scoring a context it would not have written, so its
distribution there is flatter and more hedged, putting *more* mass outside the top-128 — exactly the
mass my single tail bucket lumps into one undifferentiated number. So the approximation is *tightest*
on the peaked dataset positions and *loosest* on the flat on-policy ones, which are disproportionately
the hard, long, off-distribution completions where MATH-500 lives. The construction that makes RS-KD
cheap is least faithful exactly where dagger already struggled.

So the expectation against dagger. RS-KD uses real soft targets on the head — the teacher's relative
mass on its top 128 tokens, faithfully, with no `1/a` inflation — so it should beat the hard target
wherever the soft structure carries information. But two things temper how much. The divergence is
still mass-covering forward KL, which under the capacity gap spreads the student's limited mass to
cover the teacher's support rather than committing to a mode; and dagger already did well on GSM8K
precisely because short arithmetic chains have peaked argmaxes — where the soft head adds least. So I
expect a *modest* lift over dagger on GSM8K, a handful of problems that may sit inside or barely clear
the noise band, and roughly level on MATH-500 — it will *not* reach a mode-seeking loss, because
forward KL keeps it mass-covering exactly where a small student should be committing, and the lumped
tail denies it the rare-token signal long chains need. If RS-KD lands just above dagger on GSM8K and
roughly level on MATH-500, that says head-faithful soft targets help but the *divergence direction*
and the *fixed-target* framing are the next things to attack: adapt the target to the student's
reach, then flip the divergence toward mode-seeking. If instead it *matched* dagger to within noise
on both, the more interesting falsification, that would say the soft head carries almost nothing for
this pair and the bottleneck was never "hard vs soft" at all. AMC stays noise.
