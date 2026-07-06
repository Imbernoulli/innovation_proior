RS-KD did what I predicted, and the size of the move is the diagnosis. `baseline:rs_kd` reads GSM8K
0.4541, MATH-500 0.292, AMC 0.0469. Let me difference it against `dagger` (0.4511 / 0.290 / 0.0813)
in problems, not decimals, because that is the only way to tell signal from noise on these sample
sizes. GSM8K went 595 → 599, a gain of *four* problems out of 1319; the standard error there is about
`±18` problems, so four is a quarter of one standard error — statistically indistinguishable from
zero. MATH-500 went 145 → 146, a *single* problem, against a `±10`-problem standard error — again
noise. AMC actually fell, three solved to two, which is exactly the small-sample churn I promised to
ignore. So replacing the hard one-hot with the teacher's true relative mass on its top 128 tokens,
with the `1/a` over-confidence cured by the tail bucket, was *correct* — the soft head carries real
information in principle — but for this pair it bought essentially nothing measurable. The gain lives
entirely inside the single-seed noise floor.

There is one more number to read, and reading it correctly is the difference between a real lead and a
trap. RS-KD's final training loss was 0.342 against dagger's 0.507 — a large drop. It would be easy to
call that "RS-KD learned the target much better," but I set the rule last rung and it applies here:
those two train losses measure *different objectives* on different supports (a forward KL over `K+1`
buckets versus a cross-entropy against a one-hot), so the drop is not commensurable and is not
evidence of anything about accuracy. And in fact the accuracies are flat while the train loss moved a
lot — which, read together, is itself the diagnosis: the loss got much better at fitting its own
target and the downstream accuracy did not budge, so the *representation* of the target was never the
bottleneck. I improved how faithfully the student matches the teacher and the student got no better at
math. That tells me the bottleneck is not "hard vs soft head," and it is not "how well I summarize the
distribution." Two things are still wrong, and RS-KD touched neither. First, the divergence is forward
KL — mass-covering — so a 0.5B student is still being told to put probability everywhere the 7.6B
teacher does, smearing its limited capacity. Second, and this is the one I want to attack now, the
*target is fixed at the teacher* from step zero. RS-KD's target, dagger's target, the scaffold
default's target — all of them ask the small student to match the full far-away teacher immediately.
Under a large capacity gap that target may simply be out of reach for the entire run, and no amount of
cleverness in *summarizing* a fixed, unreachable target fixes that. The near-zero RS-KD gain is
consistent with exactly this: I improved the *representation* of the target without changing that the
target is too far.

Let me sharpen the discomfort, because the precision is the lead. The problem is not which divergence
or how I sparsify it; it is that all of these distill toward a target the student cannot get to. There
is a documented, ugly phenomenon behind this: hold a fixed small student and distill it from a
progressively larger teacher, and the student gets *worse* as the teacher grows — a better teacher
producing a worse student — because the teacher distribution drifts too far from anything the student
can represent, and a fixed objective against a fixed too-far target is hard to optimize no matter how
the two KL directions are balanced. My pair sits squarely in that regime: a 0.5B base student against
a 7.6B math-tuned instruct teacher, roughly a `15×` capacity gap plus an instruction-tuning gap on top.
The on-policy loop I am sitting on already adapts the *data* to the student (it trains on the student's
own rollouts), which is the right instinct applied to the state distribution. I want the same instinct
applied to the *target itself*: start the target somewhere the student can reach, and migrate it toward
the teacher as the student improves. A moving target, not a fixed one — and moving cheaply, from logits
alone, with no second frozen network and no extra sampling, because the edit surface is one loss body
and nothing else.

Before I build the moving target, let me be honest about the cheaper things I could try instead, so I
know I am not skipping a simpler fix. I could keep the fixed teacher and just anneal its temperature —
start it very soft and sharpen it over training — but sharpening changes the target's *sharpness*, not
its *identity*; the teacher's modes sit where they sit, and a softened teacher is still centered on
reasoning the student cannot represent, so this smooths the approach without moving the destination
closer. I could keep the fixed target and simply train longer, but the budget is `~6.4` passes and,
more to the point, the RS-KD result just told me that fitting the fixed target *better* does not raise
accuracy — the target is too far, not badly summarized, so more optimization against it is more of a
thing that already did not work. Both alternatives leave the destination fixed at the unreachable
teacher. The only move that attacks reachability directly is to move the destination, which is what a
schedule of intermediate targets does.

What would "somewhere the student can reach" mean at step zero? The most reachable target is the
student's own current distribution — fitting that is trivial, the loss is essentially already
minimized. And the place I want to end is the teacher. So I want a target that is the student at the
start and the teacher at the end and slides continuously between them. Introduce a time knob
`t ∈ [0,1]`: a target distribution `p_t` that is the student at `t=0` and the teacher at `t=1`, with
`t` ramping over training. At every moment the target is only a little ahead of where the student
currently is — close enough to fit, far enough to pull. An intermediate teacher.

How do I blend the two distributions into `p_t`? The lazy option is a convex combination of
*probabilities*, `t·p_T + (1−t)·p_S` — an ordinary mixture distribution. But I do not want to mix in
probability space here, and a concrete example shows why. Take three tokens with student logits
`(2, 0, −2)`, so `p_S = softmax(2,0,−2) = (0.867, 0.117, 0.016)`, peaked on the first token, and a
teacher that is the mirror image, `p_T = (0.016, 0.117, 0.867)`, peaked on the third. At `t = 0.5` the
probability mixture is `0.5(p_S + p_T) = (0.441, 0.117, 0.441)` — two humps, one on each parent's mode,
a valley in between. That is a *bimodal* target: it literally tells the student "be confident about
token one AND confident about token three at once," an incoherent instruction that puts a bump on each
peak rather than a single intermediate belief. Now mix in *logit* space instead: `0.5(student + teacher
logits) = (0, 0, 0)`, softmax `(0.333, 0.333, 0.333)` — a single smooth spread, no double hump. The
reason is that softmax is `exp(logit)/Z`, so a convex combination in logit space is
`exp((1−t)·s + t·τ) ∝ exp(s)^{1−t}·exp(τ)^{t} ∝ p_S^{1−t}·p_T^{t}` — the normalized *geometric* blend
of the two distributions, a smooth tempering that preserves the *relative* confidences within each
rather than superimposing their peaks. In the asymmetric general case this geometric blend slides the
single mode from the student's toward the teacher's as `t` grows, which is exactly the "target only a
little ahead" behavior I want, whereas the arithmetic mixture always shows both humps at once. So I
define the intermediate teacher at the logit level: `target_logits = (1−t)·student_logits +
t·teacher_logits`, softmax it to get `p_t`, and minimize forward KL of `p_t` against the student.

One deliberate choice I am making by keeping *forward* KL for the intermediate teacher, rather than
flipping the direction while I am here, is experimental hygiene. Two things could be wrong with the
fixed-target losses — the target is too far (reachability), and forward KL is mass-covering
(direction) — and I want to attack them one at a time so the leaderboard can tell me which one moved.
If I changed both the target schedule *and* the divergence direction on the same rung and the metrics
jumped, I could not attribute the jump. So I hold the divergence at forward KL — the same direction
dagger, RS-KD, and the scaffold default all used — and change *only* the target from fixed-teacher to
moving-intermediate-teacher. Whatever TAID moves is then cleanly the reachability axis, and whatever it
fails to move points at the direction axis for later. That control is worth more here than reaching for
a second fix I have not yet isolated the need for.

One subtlety decides whether this is coherent. `p_t` contains the student's own logits in its `(1−t)`
leg, and I am using `p_t` as the *target* in a KL whose other argument is also the student. If
gradients flow through the student copy inside `p_t`, I am differentiating the target with respect to
the very parameters I am updating — the student chases a target made partly of itself. Trace what that
does: the KL `KL(p_t ‖ p_S)` can be driven down two ways, by moving `p_S` toward `p_t` (the update I
want) or by moving `p_t` toward `p_S` (dragging the target down onto the student), and if the student
leg of `p_t` carries gradient the optimizer will happily do the second — a degenerate "move the
goalposts toward me" channel that lowers the loss while teaching nothing. The target must be a target.
So I **detach** the student logits inside `p_t` — treat that `(1−t)·student_logits` as a constant for
this step — and let gradient flow only through the student in the KL's other argument. Then the only
way to reduce the loss is to move `p_S` toward the (frozen-this-step) `p_t`, which is the pull I want.
This detach is load-bearing and easy to get wrong; without it the loss is silently degenerate.

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

There is a synergy with the on-policy loop worth naming, because it is the second time the loop's
data axis and my loss axis line up. Half the batches are the student's own rollouts. Early in
training, when `t` is small, the intermediate teacher on an on-policy batch is the geometric blend of
the student and the teacher evaluated at the student's *own* visited prefix — a target only slightly
tilted toward the teacher, sitting on exactly the states the student generates at inference. So early
TAID on on-policy data is a gentle self-consolidation on the student's own trajectory distribution,
nudged a little toward the teacher, which is about the safest thing I could ask a small model to do:
tighten up what you already do on the states you actually visit, then follow the teacher up from there.
The moving target and the on-policy states reinforce rather than fight — the schedule keeps the target
reachable in *belief* while the loop keeps the states realistic in *data*.

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
ramping linearly from `t_start` to 1.0 over the run. Let me lay the schedule out on the actual budget
so `t_start` is a considered number and not a guess. With `total_steps = 2000`: at step 0, `t = t_start`;
at step 1000, `t = t_start + (1−t_start)·0.5`; at step 2000, `t = 1.0`. I set `t_start = 0.4`, not near
zero. At step zero the target logits are then `0.6·student + 0.4·teacher` — about 60% student, 40%
teacher — which skips the trivially-easy earliest phase where the target is almost exactly the student
and there is nothing to learn, while leaving the student a real but reachable gap to close; at step
1000 `t = 0.7` (70% teacher), and it reaches the genuine teacher at the end. Starting much lower — the
earlier mistaken value was 0.10 — would make the step-zero target `0.9·student + 0.1·teacher`, roughly
90% student, giving near-zero distillation signal for the whole first stretch: a different way to
reproduce RS-KD's flat result, this time by handing the student a target it has already fit. So
`t_start = 0.4` is the load-bearing constant, chosen to keep the step-zero gap non-trivial, and
`t_end = 1.0` so the final target is the genuine teacher. If `total_steps` is unavailable I fall back
to a fixed mid-value `0.7`, which corresponds to the schedule's midpoint rather than either endpoint.

There is a mathematical reason `t_start` cannot be zero, beyond "near-zero signal," and it is worth
stating because it also verifies the detach is doing what I think. At `t = 0` the target logits are
`1·student_logits.detach() + 0·teacher_logits`, so `p_t = p_S.detach()` exactly, and the loss is
`KL(p_S.detach() ‖ p_S)` evaluated at the current parameters — which is `KL(p ‖ p) = 0`, with zero
gradient, because the target *is* the student's current distribution frozen as a constant. So `t = 0`
is not merely weak signal, it is *identically no signal*: the loss and its gradient are both zero.
That is the cleanest confirmation that the detach makes `p_t` a genuine external target (an un-detached
version would not be identically zero here, precisely because the student would be chasing a moving
copy of itself), and it is the hard reason the schedule must start strictly above zero. `t_start = 0.4`
sits comfortably away from that degenerate point.

A cost note, because it is different from RS-KD's and I should not carry the wrong mental model. The
logit interpolation touches the full `[B, T, V]` tensors — two full `log_softmax` and a weighted sum
over the whole 152k vocabulary — so TAID pays the full-vocabulary cost that RS-KD's sparse top-`K`
summary was built to avoid. That is the same cost as the scaffold default's full forward KL, so it
adds no new memory concern beyond what the loop already tolerated. But I want to be clear that I did
*not* just throw away RS-KD's caching win by carelessness: a moving target is *inherently
un-cacheable*. Its student leg changes every step as the student moves, so there is no fixed
per-token soft target to precompute and reuse — the sparsity-and-cache strategy that made RS-KD cheap
was only ever available for a *fixed* target. Once I decided the target must move to fix reachability,
the full-vocabulary per-step cost is unavoidable, and it is affordable because it is exactly the
scaffold's cost.

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
is not asked to match the far-away teacher from step zero — it consolidates early and is walked up. On
GSM8K I expect it to clear both — dagger's 595 and rs_kd's 599 sit inside a `±18`-problem band, so a
real break means clearing that band, into the high-0.46s — because the linear walk-up should let the
student fit teacher-like structure it could not reach cold. The honest worry is MATH-500: the target
is still reached via *forward* KL, mass-covering, so even a well-scheduled walk-up ends at a
mass-covering match to a teacher the student cannot fully represent, and on the hardest set that ceiling
may bite — I would not be surprised if MATH-500 comes in at or even slightly below the dagger/RS-KD
level (low-0.28s) even as GSM8K jumps, because the curriculum fixes *reachability* but not *direction*.
If that split appears — GSM8K up, MATH-500 flat-to-down — it is the precise signal that the moving
target solved the capacity-gap reachability problem and the *next* rung must attack the divergence
direction: stop being mass-covering, let the student be mode-seeking. If instead *both* metrics jump
together, that would falsify my claim that direction is a separate axis and say the reachability fix
was the whole story. AMC stays noise; I will not read it.
