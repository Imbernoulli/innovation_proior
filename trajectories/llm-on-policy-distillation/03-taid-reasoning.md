RS-KD did what I predicted, and the size of the move is the diagnosis. `baseline:rs_kd` reads GSM8K
0.4541, MATH-500 0.292, AMC 0.0469. Differenced against dagger in problems: GSM8K 595 → 599, four
problems against a `±18`-problem standard error, a quarter of one SE — indistinguishable from zero.
MATH-500 145 → 146, a single problem against `±10`. AMC fell three to two, the small-sample churn I
promised to ignore. So replacing the hard one-hot with the teacher's true relative mass on its top
128 tokens, `1/a` over-confidence cured by the tail bucket, was *correct* in principle but bought
essentially nothing measurable. The gain lives entirely inside the single-seed noise floor.

One more number, and reading it correctly is the difference between a real lead and a trap. RS-KD's
final train loss was 0.342 against dagger's 0.507 — a large drop. But those measure *different*
objectives on different supports (a forward KL over `K+1` buckets versus a cross-entropy against a
one-hot), so the drop is not commensurable and is not evidence about accuracy. And in fact the
accuracies are flat while the train loss moved a lot, which read together is itself the diagnosis:
the loss got much better at fitting its own target and downstream accuracy did not budge, so the
*representation* of the target was never the bottleneck. I improved how faithfully the student
matches the teacher and the student got no better at math. So the bottleneck is not "hard vs soft
head," and not "how well I summarize the distribution." Two things are still wrong that RS-KD touched
neither of: the divergence is mass-covering forward KL, so a 0.5B student is told to put probability
everywhere the 7.6B teacher does, smearing its limited capacity; and — the one I attack now — the
*target is fixed at the teacher* from step zero. RS-KD's target, dagger's, the scaffold default's all
ask the small student to match the full far-away teacher immediately, and under a large capacity gap
that target may be out of reach for the entire run. The near-zero RS-KD gain is exactly consistent: I
improved the *representation* of the target without changing that it is too far.

There is a documented, ugly phenomenon behind this: hold a fixed small student and distill it from a
progressively larger teacher, and the student gets *worse* as the teacher grows — a better teacher
producing a worse student — because the teacher distribution drifts too far from anything the student
can represent, and a fixed objective against a fixed too-far target is hard to optimize no matter how
the two KL directions are balanced. My pair sits squarely in that regime: a 0.5B base student against
a 7.6B math-tuned instruct teacher, roughly a `15×` capacity gap plus an instruction-tuning gap on
top. The on-policy loop already adapts the *data* to the student (it trains on the student's own
rollouts), the right instinct applied to the state distribution. I want the same instinct applied to
the *target itself*: start it somewhere the student can reach and migrate it toward the teacher as the
student improves. A moving target, moving cheaply from logits alone, with no second frozen network and
no extra sampling, because the edit surface is one loss body.

Before building it, the cheaper things I could try instead. Annealing the fixed teacher's temperature
— start it very soft, sharpen over training — changes the target's *sharpness*, not its *identity*; a
softened teacher is still centered on reasoning the student cannot represent, so it smooths the
approach without moving the destination closer. Training longer is ruled out by the RS-KD result
itself: fitting the fixed target *better* did not raise accuracy, so more optimization against it is
more of what already did not work. Both leave the destination fixed at the unreachable teacher. The
only move that attacks reachability directly is to move the destination.

The most reachable target at step zero is the student's own current distribution — fitting it is
trivial, the loss is essentially already minimized — and the place I want to end is the teacher. So I
want a target that is the student at `t=0` and the teacher at `t=1` and slides continuously between
them, with `t` ramping over training: an intermediate teacher, at every moment only a little ahead of
where the student currently is — close enough to fit, far enough to pull.

How to blend the two distributions into `p_t`? Not in probability space, and a concrete example shows
why. Take three tokens with student logits `(2, 0, −2)`, `p_S = (0.867, 0.117, 0.016)` peaked on the
first, and a mirror teacher `p_T = (0.016, 0.117, 0.867)` peaked on the third. At `t = 0.5` the
probability mixture `0.5(p_S + p_T) = (0.441, 0.117, 0.441)` is bimodal — two humps, one on each
parent's mode, a valley between — which literally tells the student "be confident about token one AND
token three at once," an incoherent instruction. Mix in *logit* space instead: `0.5(student + teacher)
= (0, 0, 0)`, softmax `(0.333, 0.333, 0.333)`, a single smooth spread. The reason is that a convex
combination in logit space is `exp((1−t)·s + t·τ) ∝ p_S^{1−t}·p_T^t`, the normalized *geometric*
blend, which preserves the relative confidences within each and slides the single mode from the
student's toward the teacher's as `t` grows, rather than superimposing their peaks. So I define the
intermediate teacher at the logit level: `target_logits = (1−t)·student_logits + t·teacher_logits`,
softmax to `p_t`, and minimize forward KL of `p_t` against the student.

I keep *forward* KL here rather than flipping the direction while I am at it, deliberately, for
experimental hygiene. Two things could be wrong with the fixed-target losses — the target is too far
(reachability) and forward KL is mass-covering (direction) — and I want to attack them one at a time
so the leaderboard tells me which one moved. Changing both the target schedule and the divergence at
once would make any jump unattributable. So I hold the divergence at forward KL — the same
direction the earlier losses used — and change *only* the target from fixed-teacher to
moving-intermediate-teacher; whatever this moves is cleanly the reachability axis, and whatever it
fails to move points at the direction axis for later.

One subtlety decides whether this is coherent. `p_t` contains the student's own logits in its `(1−t)`
leg, and I use `p_t` as the *target* in a KL whose other argument is also the student. If gradients
flow through the student copy inside `p_t`, the KL can be driven down two ways — by moving `p_S`
toward `p_t` (the update I want) or by dragging `p_t` down onto `p_S` (a degenerate "move the
goalposts toward me" channel that lowers the loss while teaching nothing) — and the optimizer will
happily do the second. So I **detach** the student logits inside `p_t`, treat that leg as a constant
for this step, and let gradient flow only through the student in the KL's other argument. This detach
is load-bearing and easy to get wrong; without it the loss is silently degenerate.

The dynamics: at small `t`, `p_t` is almost the student's own (detached) distribution, so this is
essentially self-distillation, letting the student consolidate its own modes early rather than
drowning under a teacher it cannot represent — precisely the regime RS-KD never entered, which is why
its gain was flat. As `t` grows the target tilts toward the teacher and the student takes on richer
structure, always from a target only modestly beyond its current reach; late in training it is
ordinary forward-KL distillation but from a student walked up to the teacher's neighborhood instead
of dropped in cold. And on the on-policy half the early intermediate teacher sits on exactly the
student's own visited prefixes, so early on this is a gentle self-consolidation on the trajectory
distribution the student actually generates — the moving target and the on-policy states reinforce
rather than fight.

The self-distillation observation should make me nervous, not comfortable: repeatedly distilling a
model into itself drives collapse, because the recursion only contracts the signal with nothing
replenishing it. The saving grace is the `t·teacher` leg — a constant slice of teacher signal that
does not depend on the student and so does not shrink, flooring the signal norm at a fraction
proportional to `t` rather than letting it bleed to zero. The constant teacher injection is what makes
the moving target safe, and it requires the schedule to keep `t` rising and not stall in the
self-distillation-like region near zero.

Now the schedule, where this is *deliberately simpler* than the method's full form. The full TAID
keys `t`'s speed to a momentum-EMA of the student's smoothed relative loss progress — an adaptive
schedule that races ahead when learning is easy and slows when it is hard, needing scalar state (a
running momentum, a previous-loss buffer) carried across steps. The edit surface is a *stateless* loss
body called once per step with `step` and `total_steps` and nothing persistent; there is no clean
place to hold the EMA without smuggling in module state the contract does not provide. So I drop the
adaptive momentum and use the plain **linear** schedule, the non-adaptive variant the method's
non-collapse argument is written for: `t = t_start + (1 − t_start)·min(1, step/total_steps)`. On
`total_steps = 2000`: step 0 gives `t_start`, step 1000 its midpoint, step 2000 gives 1.0. I set
`t_start = 0.4`, not near zero: at step zero the target is `0.6·student + 0.4·teacher`, which skips
the trivially-easy earliest phase where the target is almost exactly the student and there is nothing
to learn, while leaving a real but reachable gap. Starting much lower — an earlier mistaken value was
0.10 — makes the step-zero target `0.9·student + 0.1·teacher`, near-zero distillation signal for the
whole first stretch: another way to reproduce RS-KD's flat result, by handing the student a target it
has already fit. So `t_start = 0.4` is the load-bearing constant, and `t_end = 1.0` so the final
target is the genuine teacher. If `total_steps` is unavailable I fall back to a fixed `0.7`, the
schedule's midpoint.

There is a hard reason `t_start` cannot be zero, which also verifies the detach. At `t = 0` the target
logits are `1·student.detach() + 0·teacher`, so `p_t = p_S.detach()` exactly, and the loss is
`KL(p_S.detach() ‖ p_S) = 0` with zero gradient — not merely weak signal but identically none. That
confirms the detach makes `p_t` a genuine external target (an un-detached version would not be
identically zero here, precisely because the student would be chasing a moving copy of itself), and it
is the reason the schedule must start strictly above zero. `t_start = 0.4` sits comfortably away from
that degenerate point.

A cost note, different from RS-KD's. The logit interpolation touches the full `[B, T, V]` tensors —
two `log_softmax` and a weighted sum over the whole 152k vocabulary — so TAID pays the full-vocabulary
cost that RS-KD's sparse top-`K` summary was built to avoid. That is the scaffold default's cost, no
new memory concern. And it is not carelessness: a moving target is *inherently un-cacheable*, because
its student leg changes every step, so there is no fixed per-token soft target to precompute and
reuse. The sparsity-and-cache strategy was only ever available for a *fixed* target; once the target
must move to fix reachability, the full-vocabulary per-step cost is unavoidable and affordable.

For the body: forward KL of the detached intermediate teacher against the student, `KL(p_t ‖ p_S)`,
whose first term `Σ p_t·log p_t` is the negative entropy of the detached `p_t` — a constant for this
step — so minimizing the KL is gradient-equivalent to the cross-entropy `−Σ p_t·log p_S`. I take the
KL form with the direction pinned so the intermediate teacher is the *target* and the student the
input, mask to completion tokens (`labels != -100`), reduce per token, temperature on both logit
tensors before the interpolation, consistent with the rest of the losses. The full body is in the
answer.

Expectations against the fixed-target losses below. The moving target should *finally* break the
student out of the dagger/RS-KD cluster on the reliable metrics, because for the first time it is not
asked to match the far-away teacher from step zero — it consolidates early and is walked up. On GSM8K
a real break means clearing the `±18`-problem band, since dagger's 595 and rs_kd's 599 sit inside it.
The honest worry is MATH-500: the target is still reached via *forward* KL, mass-covering, so even a
well-scheduled walk-up ends at a mass-covering match to a teacher the student cannot fully represent,
and on the hardest set that ceiling may bite — I would not be surprised if MATH-500 comes in flat-to-
down even as GSM8K jumps, because the curriculum fixes *reachability*, not *direction*. If that split
appears — GSM8K up, MATH-500 flat-to-down — it is the precise signal that the next move must attack
the divergence direction: stop being mass-covering, let the student be mode-seeking. If instead *both*
metrics jump together, that falsifies my claim that direction is a separate axis and says the
reachability fix was the whole story. AMC stays noise.
