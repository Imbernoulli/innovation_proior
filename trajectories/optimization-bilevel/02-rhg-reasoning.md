The truncated unroll just told me something important by *not* failing. T-RHG, back-propagating
through only the last `K = 100` of `T = 500` inner steps, landed at 84.61 mean linear test accuracy,
84.79 on the hidden MLP, with linear f1 89.06 and a high cleaner recall of 0.964 against a middling
precision of 0.828. On the toy it sat at exactly 260.7 mean convergence_steps at full success and 0.030
residual — the shared projected step, as I expected, since the toy collapses every method to the same
update. Before I decide what to do next I want to read those numbers as mechanism, not just record them.

Start with the variance, because it is the cleanest signal about whether the contraction I bet on
actually held. The three linear seeds were 84.82, 84.10, 84.92 — a range of 0.82 and a standard
deviation of about 0.37, driven almost entirely by the single seed-123 dip; the other two sit within
0.10 of each other. A truncated estimator whose bias were on a knife's edge would show up as jitter that
swamps 0.37 points; instead the estimate is quiet and reproducible across inits, which is exactly what
`(1 - lr_inner*mu)^K` being small looks like — the dropped deep terms are contributing a stable
near-nothing, not a seed-dependent something. Next the cleaner profile, which I can cross-check against
the reported f1 to be sure I am reading the table correctly. With precision `0.828` and recall `0.964`,
the harmonic mean is `2 * 0.828 * 0.964 / (0.828 + 0.964) = 1.596 / 1.792 ≈ 0.891`, which reproduces the
reported linear f1 of `89.06` to the digit. So the numbers are internally consistent, and the shape they
describe is unambiguous: recall far above precision means the cleaner keeps almost every genuinely-clean
example (few clean points thrown away) but also lets a fair number of corrupted ones through (precision
only 0.83). That is the fingerprint of an *under-solved* inner classifier — a lightly-trained weighted
model does not commit hard enough to push suspect examples' weights down, so it over-keeps. The recall
is high not because the method is discriminating well but because it is discriminating *softly*.

One more comparison in the step-1 table is quietly diagnostic: the MLP came in at 84.79 and the linear
model at 84.61, essentially tied, with the MLP a hair *above* despite carrying `238510` parameters
against the linear model's `7850` — thirty times the capacity. If the ~85% number were a *capacity*
ceiling, the far larger MLP should have pulled clearly ahead; it did not. So the limit is not how
expressive `y` is, it is something both models hit regardless of size — which points straight at the
shared bottleneck, the crude inner solve. A capacity wall would separate the two architectures; a
solve wall flattens them together, and flattened together is what I see. That is my first real evidence
that the story of the next move is "solve the inner problem better," not "make `y` bigger."

So the truncation bias I was worried about did *not* show up as a deficit: the numbers are squarely
where the contraction argument predicted, which is either because `K = 100` is already deep enough that
the dropped 400 transitions were geometrically negligible, or because the inner problem is well-enough
conditioned that even a short adjoint is a good descent direction. Those two readings are not the same
claim, and they matter for everything downstream, so I want to settle which it is. The natural way to do
that — and to ask whether the unrolling family can do better at all by spending the memory I saved — is
to remove the truncation entirely and back-propagate through the *whole* inner trajectory. That is full
reverse-mode hyper-gradient, RHG, and it is the `K = T` special case of exactly the method I just ran.

Let me be careful about what "the same method, more of it" buys, because the result of step 1 is a
strong prior and I do not want to spend memory blindly. The exact hypergradient of the `T`-step inner
run is the full trajectory sum `d_x F = grad_x f + sum_{t=0}^{T} B_t A_{t+1} ... A_T grad_y f`, with
`A_t = I - lr_inner * grad_yy g` the state Jacobian of the inner gradient-descent map and
`B_t = -lr_inner * grad_xy g` its direct dependence on `x`. Truncation kept only the last `K` of those
`T + 1` terms; full RHG keeps all of them. The terms it adds back are precisely the deep ones, indexed
`t <= T - K`, each carrying a product of more than `K` contraction matrices `A_{t+1} ... A_T`, so each is
bounded in size by `(1 - lr_inner*mu)^{T-t} <= (1 - lr_inner*mu)^{K}`. In other words, the extra terms
full RHG sums are *exactly* the ones whose magnitude T-RHG's bias bound said were negligible, and whose
negligibility the low seed-variance already circumstantially confirmed. If the inner map genuinely
contracts, adding them should move the hypergradient — and therefore the final accuracy — almost not at
all. The 84.61/84.79 result from step 1 is my baseline expectation for what RHG will reproduce.

I can put a number on how little I expect to move. The relative weight of everything RHG adds over T-RHG
is at most `(1 - lr_inner*mu)^{100}`; even at a modest curvature-step product of `0.05` that factor is
`about 0.006`, and at `0.03` it is `about 0.05`. So unless the inner problem is genuinely ill-conditioned
(`lr_inner*mu` down around a hundredth, where the factor climbs toward `0.37`), the deep terms should
shift the descent direction by at most a few percent, which on an 85%-accuracy plateau is fractions of a
point — well inside the 0.37-point seed noise I just measured. Concretely I predict RHG within about a
third of a point of T-RHG on both models, and I would be genuinely surprised by a multi-point jump.

So why run it, if I expect it to match? Two reasons, and both are about *learning the geometry* rather
than chasing points. First, full RHG is the exact gradient of the procedure I actually ran, with no
truncation bias at all; if it lands meaningfully above T-RHG, that gap is the price truncation was
charging me, and it would tell me `K = 100` was too aggressive and the inner problem less well
conditioned than I hoped — a direct, falsifiable read on `lr_inner*mu`. If it lands at the same place,
that confirms the truncation was free and the contraction assumption holds, which is the more useful
thing to know because it licenses the cheap variant going forward. Second, full RHG is the canonical
reference point the whole unrolling lineage is measured against; pinning its number fixes the ceiling of
the "differentiate the inner run" family on this task, so when a penalty reformulation later jumps ahead
I know the jump is about the *method class*, not about truncation.

Let me re-derive the adjoint so I am sure full unrolling is just the truncated sweep with `K = T`, and
so I land the right hyperparameters. The Lagrangian view makes this clean: minimize `f(x, y_T)` subject
to the `T` constraints `y_t = Phi_t(y_{t-1}, x)`, attach a row multiplier `alpha_t` to each, and form
`L = f(x, y_T) + sum_t alpha_t . (y_t - Phi_t(y_{t-1}, x))`. Stationarity in `y_T` gives
`alpha_T = grad_y f`; stationarity in each interior `y_{t-1}` gives the backward recursion
`alpha_{t-1} = alpha_t . (d Phi_t / d y_{t-1}) = alpha_t . A_t`; and stationarity collected over `x`
gives the gradient `d_x F = grad_x f + sum_t alpha_t . B_t`. The multiplier `alpha_t` is the adjoint
state — the sensitivity of the final validation loss to a perturbation of the inner iterate at time `t`
— and the recursion carries it from the end of training to the beginning. Every operation is a
transposed-Jacobian-vector product: `alpha . A` is the gradient of "alpha dotted with the next state"
with respect to the current state, `alpha . B` the same with respect to `x`. Neither forms a matrix.
Setting `K = T` means the backward loop runs from `T` down to `1` over the *entire* stored trajectory
rather than the last hundred states; the recursion, the accumulation, the per-step cost are identical —
only the window length and therefore the memory change. This is why I can say with confidence that RHG
and T-RHG are the same algorithm: the step-1 code and the step-2 code differ in one integer.

The harness makes that literally true. `run_rhg_family` reads `K` from the hyperparameters and runs the
forward inner loop keeping the suffix of `K + 1` states, then calls the fixed `hg.reverse(...,
[fp_map] * K, ...)` adjoint sweep. To go from truncated to full I set `K = T = 500`; the helper then
stores all 500 transitions and back-propagates through every one. Everything else — `T = 500`,
`lr = 0.001` outer, `lr_inner = 0.1` linear and `0.4` MLP, `outer_itr = 100`, `eval_interval = 1` — I
keep identical to step 1, because the only variable I am changing is whether the deep trajectory terms
are included. And exactly as in step 1, this rung ignores the `outer_grad / inner_grad / inner_val`
callables the scaffold exposes; the adjoint needs the inner step built with `create_graph=True` as a
graph node, which the helper constructs internally inside its own `fp_map`. Those exposed first-order
callables are for the penalty rungs that the background lists, not for unrolling.

I should be explicit about the cost I am paying for the exactness, because it is the structural weakness
this whole family inherits and the reason the next rung will look elsewhere. Full RHG stores the entire
inner trajectory, `O(T * dim(y))` memory. Put the numbers back on it: for the MLP that is `500 *
238510 ≈ 1.19e8` floats, about `477 MB` of checkpoints per inner run, against T-RHG's `95 MB` at
`K = 100` — a fivefold memory increase — and in compute, `500` forward steps plus `500` backward TJVPs
per outer step, `1000` gradient-scale evaluations against T-RHG's `600`, so about `5/3` the work,
`100000` inner-gradient evaluations across the run. I am spending all of that here only to *measure*
whether it was worth spending, and the step-1 result already strongly suggests it was not, because
T-RHG at one fifth the memory matched the contraction prediction. That is a defensible thing to do once,
as a diagnostic; it is not a thing I would do routinely.

There is a deeper reconciliation that tells me the ceiling of this family is set by the inner solve, not
by the differentiation, and it is the whole reason I do not expect RHG to break 85%. In the converged
limit the full backward sum is the Neumann series of the inverse Hessian, so RHG with `K = T` is
implicit differentiation computed by summation: its hypergradient is `-grad_y f (grad_yy g)^{-1}
grad_xy g` evaluated *at whatever `y_T` the inner run reached*. That means RHG's gradient is only ever as
good as the inner optimizer's convergence. And here the inner optimizer is 500 gradient-descent steps at
a fixed `lr_inner`, run fresh from a random init every single outer step — never warm-started, never
given more than 500 steps. The inner optimality gap after `T` steps contracts like
`(1 - lr_inner*mu)^T (g(y_0) - v)`, which is only small if `lr_inner*mu*T` is large; if `mu` is small
(the flat MLP landscape), 500 steps from scratch leaves the inner problem genuinely under-solved, and
RHG then differentiates a classifier that has not really found its minimum. So I expect the unrolling
family, full or truncated, to be capped well below the penalty reformulations — around 85% test accuracy
— not because the differentiation is wrong but because the inner problem is barely solved before I
differentiate it. The 85% is the signature of a crude inner solve, and no amount of extra backward depth
can fix a forward pass that stopped short of the minimum.

I can put numbers on that forward gap the same way I did the backward truncation, and the comparison
exposes a distinction I had been blurring. The inner optimality gap after `T` steps contracts like
`(1 - lr_inner*mu)^T` from the *initial random point*. If the well-scaled linear model has
`lr_inner*mu ≈ 0.02`, then `(0.98)^{500} ≈ e^{-10} ≈ 5e-5`: 500 steps genuinely nail the linear inner
minimum, and RHG differentiates a converged classifier. But if the flat MLP landscape has `lr_inner*mu`
down around `0.003`, then `(0.997)^{500} ≈ e^{-1.5} ≈ 0.22` — after all 500 steps the inner gap is still
a fifth of where it started, and RHG is differentiating a classifier that never reached its minimum.
Here is the subtlety: the backward truncation quality depends on the *local* curvature in the converged
tail, where the contraction argument lives and where T-RHG's clean numbers say the geometry is benign;
the forward solve quality depends on the curvature *along the whole descent from a random init*, through
whatever flat plateaus the MLP loss has early on. Those are different `mu`s, and they decouple. So T-RHG
looking healthy (good local contraction near the end, truncation free) tells me nothing reassuring about
whether the forward pass ever climbed out of the early flat region to reach the true `y*`. That is
exactly why the ceiling can be real even when the truncation is free: "the last hundred Jacobians
contract" and "500 steps from scratch found the minimizer" are separate facts, and the 85% plateau is
the second one failing while the first holds.

The outer loop compounds it. With `lr = 0.001` and only `outer_itr = 100` steps, `x` itself moves very
little over the run, so even a perfect hypergradient would leave the *outer* problem lightly converged;
both levels are under-solved, and RHG's exactness only sharpens a direction that is being taken too few,
too-small steps along. This is the structural cul-de-sac of the differentiate-the-inner-run family: its
budget is dominated by re-solving the inner problem from scratch, which leaves nothing for solving it
*well* or for taking many outer steps.

The toy, as in step 1, is decoupled from all of this. `run_rhg_family` in toy mode dispatches to the
same `_toy_pbgd_step` as the value-gap method: `S(x) = {-x}`, `v(x) = 0`, nothing to unroll, one
projected penalized step on `f + gamma * grad g` with `gams = (10.0,)`, `alpha0 = 0.1`. So I expect RHG's
toy line to be *identical* to T-RHG's — and I mean identical to the last digit, `260.712` mean steps,
`0.030088` residual, success `1.0`, not merely "within noise," because the toy step never reads `K` at
all: it is the same `_toy_pbgd_step` with the same `gams`, `alpha0`, and the same 1000 seeded inits, so
the two runs execute bit-for-bit the same arithmetic. That is a sharper falsifiable claim than the
hyper-cleaning one: if the toy tables differ even in a low digit, then the `K` knob is leaking into the
toy code path where it has no business, and I have a wiring bug rather than a property of unrolling. The
toy simply cannot separate these two rungs; only hyper-cleaning can, and only through the deep trajectory
terms.

There is a detail in the step-1 toy table worth reading before I move on, because it is the kind of thing
that looks like noise and is not. The mean convergence was 260.7 but the *median* was 302.3 — the median
sits well *above* the mean, which means the step-count distribution over the 1000 inits is left-skewed:
a substantial block of inits converge much faster than the typical ~302 steps, dragging the average down,
while the bulk clusters near 302. That is consistent with the projected penalized step on `f + 10 g`
having two regimes across the domain `[0, 3]` — inits that start near the `y = -x` valley snap to
tolerance in a handful of steps, inits that start far from it grind through the full few-hundred-step
descent — and it is a property of the shared toy landscape, not of unrolling. So I expect RHG to
reproduce not just the mean but the whole shape: median 302.3, mean 260.7, to the digit, because it runs
the identical toy step. If the mean matched but the median drifted, something in the seeding or the
tolerance check would have changed, which the single-integer `K` edit cannot touch.

The memory and runtime give me a second, independent wiring check on hyper-cleaning. If the helper is
truly storing the whole trajectory at `K = 500`, RHG's inner run must hold about `477 MB` of MLP
checkpoints against T-RHG's `95 MB`, and its per-outer-step work must be about `5/3` of T-RHG's from the
extra 400 backward TJVPs, so I expect RHG's runtime-to-best to sit modestly above T-RHG's. If instead
RHG runs at the same speed and memory as the truncated version, the helper is silently ignoring the
larger `K` and I am not actually running full unrolling — a different failure than a bad hyperparameter,
and one the runtime column would expose before the accuracy column could mislead me.

So here is what I expect, stated against the step-1 numbers so it is falsifiable. On the toy, RHG should
reproduce T-RHG exactly: 260.7 mean steps, success 1.0, residual 0.030. On hyper-cleaning, RHG should
land *at or just above* T-RHG: linear test accuracy around 84.6 (T-RHG was 84.61) and MLP around 84.8
(T-RHG was 84.79), with linear f1 near 89.4 and the same high-recall, middling-precision cleaner profile
(recall around 0.97, precision around 0.83) — and I can sanity-check that f1 prediction against those
two: `2 * 0.83 * 0.97 / (0.83 + 0.97) ≈ 1.610 / 1.80 ≈ 0.894`, so recall-0.97/precision-0.83 and
f1-89.4 are the same prediction stated two ways, and if the measured f1 drifts far from that the
precision-recall pair I am imagining is wrong. If anything I expect the deep terms to buy a *sliver* of
precision — full unrolling includes the early trajectory's influence on which examples got down-weighted,
so RHG's cleaner precision might tick from T-RHG's 0.828 up toward 0.83, lifting f1 from 89.06 toward 89.4
— but a sliver is the ceiling on what an exact hypergradient of a crude solve can add, and a sliver on an
85% plateau is scientifically a null result, which is the useful thing to establish. If RHG matches T-RHG within the ~0.37-point seed noise, the
truncation was free — the contraction held and `K = 100` was already deep enough — and the unrolling
family's ceiling on this task is ~85% accuracy, set by the 500-step inner solve. If RHG instead jumps
several points above T-RHG, then the deep terms mattered, `lr_inner*mu` is smaller than I estimated, and
the inner problem is poorly conditioned. Either way, the gap to the penalty reformulations — which I have
not yet measured but expect to be large — will be the real story, because it is the gap between
"differentiate a crude inner solve" and "fold lower-level optimality into the objective and descend it
directly" without ever resetting an inner loop. That is where the next rung has to go if this ceiling is
real.

I can even see the budget arithmetic that would justify the switch. This rung spends about `100000`
inner-gradient evaluations, but they are structured wastefully: `100` outer steps, each burning `500`
forward steps to re-solve the inner problem from a random init that it then throws away, plus `500`
backward steps to differentiate the throwaway. The same `100000` evaluations spent as *one persistent
coupled trajectory* over `(x, y)` — never resetting `y`, letting it accumulate progress across outer
steps — would be on the order of `100000` genuine coupled updates instead of `100` outer moves, three
orders of magnitude more actual descent on the joint variables. That reallocation is the entire promise
of the penalty class the background points to: stop paying to re-solve and re-differentiate a fresh
inner problem every outer step, and instead pay once, continuously, into a single objective whose
descent drives `y` toward lower-optimality and `x` toward the outer minimum together. Whether that
promise pays off is the next rung's question; this rung's job was to fix the unrolling ceiling as the
number it has to beat. The full scaffold edit is in the answer.
