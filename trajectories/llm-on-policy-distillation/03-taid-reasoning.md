RS-KD did what I predicted, and the size of the move is the diagnosis. `baseline:rs_kd` reads GSM8K
0.4541, MATH-500 0.292, AMC 0.0469. Against `dagger` (0.4511 / 0.290 / 0.0813) the head-faithful soft
target bought almost nothing: +0.003 on GSM8K, +0.002 on MATH-500, and AMC actually fell (to two
solved problems — still noise). So replacing the hard one-hot with the teacher's true relative mass
on its top 128 tokens, with the `1/a` over-confidence cured by the tail bucket, was *correct* and
*near-free*, and it confirms the soft head carries real information — but the lift is within the noise
floor of a single seed. That tells me the bottleneck is not "hard vs soft head." Two things are still
wrong, and RS-KD touched neither. First, the divergence is forward KL — mass-covering — so a 0.5B
student is still being told to put probability everywhere the 7.6B teacher does, smearing its limited
capacity. Second, and this is the one I want to attack now, the *target is fixed at the teacher* from
step zero. RS-KD's target, dagger's target, the scaffold default's target — all of them ask the small
student to match the full far-away teacher immediately. Under a large capacity gap that target may
simply be out of reach for the entire run, and no amount of cleverness in *summarizing* a fixed,
unreachable target fixes that. The near-zero RS-KD gain is consistent with exactly this: I improved
the *representation* of the target without changing that the target is too far.

Let me sharpen the discomfort, because the precision is the lead. The problem is not which divergence
or how I sparsify it; it is that all of these distill toward a target the student cannot get to.
There is a documented, ugly phenomenon behind this: hold a fixed small student and distill it from a
progressively larger teacher, and the student gets *worse* as the teacher grows — a better teacher
producing a worse student — because the teacher distribution drifts too far from anything the student
can represent, and a fixed objective against a fixed too-far target is hard to optimize no matter how
the two KL directions are balanced. The on-policy loop I am sitting on already adapts the *data* to
the student (it trains on the student's own rollouts), which is the right instinct applied to the
state distribution. I want the same instinct applied to the *target itself*: start the target
somewhere the student can reach, and migrate it toward the teacher as the student improves. A moving
target, not a fixed one — and moving cheaply, from logits alone, with no second frozen network and no
extra sampling, because the edit surface is one loss body and nothing else.

What would "somewhere the student can reach" mean at step zero? The most reachable target is the
student's own current distribution — fitting that is trivial, the loss is essentially already
minimized. And the place I want to end is the teacher. So I want a target that is the student at the
start and the teacher at the end and slides continuously between them. Introduce a time knob
`t ∈ [0,1]`: a target distribution `p_t` that is the student at `t=0` and the teacher at `t=1`, with
`t` ramping over training. At every moment the target is only a little ahead of where the student
currently is — close enough to fit, far enough to pull. An intermediate teacher.

How do I blend the two distributions into `p_t`? The lazy option is a convex combination of
probabilities, `t·p_T + (1−t)·p_S`, like the JSD mixture I will reach for at the next rung. But I do
not want to mix in probability space here: a probability mixture of two peaked distributions is
bimodal — it literally puts a bump on each — and softmax outputs are not naturally averaged that way;
the mixture can wash out or double the structure I am trying to hand the student smoothly. The
additive geometry lives in the *logits*: softmax is `exp(logit)/Z`, so a convex combination *in logit
space* is a normalized geometric blend of the two distributions, a smooth tempering that preserves the
*relative* confidences within each. That is the object I want — a target whose internal ranking
degrades gracefully from student-like to teacher-like rather than a two-humped average. So define the
intermediate teacher at the logit level: `target_logits = (1−t)·student_logits + t·teacher_logits`,
softmax it to get `p_t`, and minimize forward KL of `p_t` against the student.

One subtlety decides whether this is coherent. `p_t` contains the student's own logits in its `(1−t)`
leg, and I am using `p_t` as the *target* in a KL whose other argument is also the student. If
gradients flow through the student copy inside `p_t`, I am differentiating the target with respect to
the very parameters I am updating — the student chases a target made partly of itself, a degenerate
"move the goalposts toward me" channel that does nothing useful. The target must be a target. So I
**detach** the student logits inside `p_t` — treat that `(1−t)·student_logits` as a constant for this
step — and let gradient flow only through the student in the KL's other argument. This detach is
load-bearing and easy to get wrong; without it the loss is silently degenerate.

Let me sanity-check the dynamics before I trust it, threading it against what I have already seen. At
small `t`, `p_t` is almost the student's own (detached) distribution, so I am distilling the student
into a slightly-perturbed copy of itself — essentially self-distillation, which lets the student
consolidate its own modes early rather than drowning under a teacher it cannot represent. This is
precisely the regime RS-KD never entered: RS-KD demanded teacher-matching from step zero, which is why
its gain was flat. As `t` grows the target tilts toward the teacher and the student is asked to take
on the teacher's richer structure, but always from a target only modestly beyond its current reach;
late in training the target is essentially the teacher and I am doing ordinary forward-KL
distillation, but now from a student walked up to the teacher's neighborhood instead of dropped into
it cold. That graded approach is the medicine for the capacity gap that the fixed-target losses below
me lacked.

The self-distillation observation should make me nervous, though, not comfortable, because
self-distillation has a known dark side: repeatedly distilling a model into itself drives the solution
to collapse after enough rounds — the recursion only contracts the signal, with nothing replenishing
it. If early TAID *is* self-distillation, am I building a method that collapses by construction? The
saving grace is the `t·teacher` leg: the target is not built only from the (shrinking) student
prediction, it carries a constant slice of the teacher signal that does not depend on the student and
so does not shrink. In the regression proxy the interpolated update injects a fixed teacher term every
step, which floors the signal norm at a fraction proportional to `t` rather than letting it bleed to
zero, guaranteeing the student cannot collapse in the late phase exactly where pure self-distillation
is doomed. The constant teacher injection is what makes the moving target safe; I do not need to
re-derive the full bound here, only to know that the teacher leg is the thing that prevents the
collapse, and that it requires the schedule to keep `t` rising and not stall in the self-distillation-
like region near zero.

Now the schedule, and here is the place this task's loss is *deliberately simpler* than the method's
full form, which I should be explicit about so I land the literal edit. The full TAID keys `t`'s speed
to the student's smoothed relative loss progress — a momentum-EMA of the fractional loss drop, passed
through a sigmoid and a `(1−t)` diminishing factor, floored by a linear schedule and capped at the
teacher — so the target races ahead when learning is easy and slows when it is hard. That adaptive
machinery needs to carry scalar state (a running momentum, a previous-loss buffer) across steps. The
edit surface here is a *stateless* loss body called once per step with `step` and `total_steps` and
nothing persistent; there is no clean place to hold the EMA or the previous loss across calls without
smuggling in module state the contract does not provide. So I drop the adaptive momentum and use the
plain **linear** schedule, which is the non-adaptive variant the method compares against and the one
its non-collapse proof is written for: `t = t_start + (1 − t_start)·min(1, step/total_steps)`,
ramping linearly from `t_start` to 1.0 over the run. If `total_steps` is unavailable I fall back to a
fixed mid-value. I set `t_start = 0.4`, not near zero: at step zero the target is then ~60% student /
40% teacher, which skips the trivially-easy earliest phase where the target is almost exactly the
student and there is nothing to learn, while leaving the student a real but reachable gap to close.
Starting much lower (the earlier mistaken value was 0.10) would make the step-zero target ~90%
student, giving near-zero distillation signal — a different way to reproduce RS-KD's flat result. So
`t_start = 0.4` is the load-bearing constant, and `t_end = 1.0` so the final target is the genuine
teacher.

One implementation reality folds in. Forward KL of the detached intermediate teacher against the
student, `KL(p_t ‖ p_S) = Σ p_t·(log p_t − log p_S)`, has a first term `Σ p_t·log p_t` that is the
negative entropy of the *detached* `p_t` — a constant for this step — so as far as the student's
gradient is concerned it drops, and minimizing the KL is gradient-equivalent to minimizing the cross-
entropy `−Σ p_t·log p_S`. Either form gives the same student gradient; I take the KL form with the
direction pinned so that the intermediate teacher is the *target* (the distribution the KL is "from")
and the student is the input being pushed — `KL(target ‖ student)` — which in the framework's
log-target convention means the student log-probs are the input and the target log-probs are the
target. Mask to completion tokens (`labels != -100`), reduce per token. Temperature divides both logit
tensors before the interpolation, consistent with the rest of the ladder. The full scaffold body is in
the answer.

Falsifiable expectations against the fixed-target rungs below me. TAID's moving target should *finally*
break out of the dagger/RS-KD cluster on the reliable metrics, because for the first time the student
is not asked to match the far-away teacher from step zero — it consolidates early and is walked up.
On GSM8K I expect it to clear both, into the high-0.46s, because the linear walk-up should let the
student fit teacher-like structure it could not reach cold. The honest worry is MATH-500: the target
is still reached via *forward* KL, mass-covering, so even a well-scheduled walk-up ends at a mass-
covering match to a teacher the student cannot fully represent, and on the hardest set that ceiling may
bite — I would not be surprised if MATH-500 comes in at or even slightly below the dagger/RS-KD level
(low-0.28s) even as GSM8K jumps, because the curriculum fixes *reachability* but not *direction*. If
that split appears — GSM8K up, MATH-500 flat-to-down — it is the precise signal that the moving target
solved the capacity-gap reachability problem and the *next* rung must attack the divergence direction:
stop being mass-covering, let the student be mode-seeking. AMC stays noise; I will not read it.
