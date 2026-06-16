The truncated unroll just told me something important by *not* failing. T-RHG, back-propagating
through only the last `K = 100` of `T = 500` inner steps, landed at 84.61 mean linear test accuracy
(84.10 / 84.82 / 84.92 across the three seeds), 84.79 on the hidden MLP, with linear f1 89.06 and a
high cleaner recall of 0.964 against a middling precision of 0.828. On the toy it sat at exactly 260.7
mean convergence_steps at full success and 0.030 residual — the shared projected step, as I expected,
since the toy collapses every method to the same update. So the truncation bias I was worried about did
*not* show up as a deficit: the numbers are squarely where the contraction argument predicted, which is
either because `K = 100` is already deep enough that the dropped 400 transitions were geometrically
negligible, or because the inner problem is well-enough conditioned that even a short adjoint is a good
descent direction. The natural way to settle which it is — and to ask whether the unrolling family can
do better at all by spending the memory I saved — is to remove the truncation entirely and
back-propagate through the *whole* inner trajectory. That is full reverse-mode hyper-gradient, RHG, and
it is the `K = T` special case of exactly the method I just ran.

Let me be careful about what "the same method, more of it" buys, because the result of step 1 is a
strong prior. The exact hypergradient of the `T`-step inner run is the full trajectory sum
`d_x F = grad_x f + sum_{t=0}^{T} B_t A_{t+1} ... A_T grad_y f`, with `A_t = I - lr_inner * grad_yy g`
the state Jacobian of the inner gradient-descent map and `B_t = -lr_inner * grad_xy g` its direct
dependence on `x`. Truncation kept only the last `K` of those `T + 1` terms; full RHG keeps all of them.
The terms it adds back are precisely the deep ones, indexed `t <= T - K`, each carrying a product of
more than `K` contraction matrices `A_{t+1} ... A_T`, so each is bounded in size by `(1 -
lr_inner*alpha)^{T-t} <= (1 - lr_inner*alpha)^{K}`. In other words, the extra terms full RHG sums are
*exactly* the ones whose magnitude T-RHG's bias bound said were negligible. If the inner map genuinely
contracts, adding them should move the hypergradient — and therefore the final accuracy — almost not at
all. The 84.6/84.8 result from step 1 is my baseline expectation for what RHG will reproduce.

So why run it, if I expect it to match? Two reasons, and both are about *learning the geometry* rather
than chasing points. First, full RHG is the exact gradient of the procedure I actually ran, with no
truncation bias at all; if it lands meaningfully above T-RHG, that gap is the price truncation was
charging me, and it would tell me `K = 100` was too aggressive and the inner problem less well
conditioned than I hoped. If it lands at the same place, that confirms the truncation was free and the
contraction assumption holds — which is the more useful thing to know, because it licenses the cheap
variant going forward. Second, full RHG is the canonical reference point the whole unrolling lineage is
measured against; pinning its number fixes the ceiling of the "differentiate the inner run" family on
this task, so when the penalty methods jump ahead I know the jump is about the *method class*, not about
truncation.

Let me re-derive the adjoint so I am sure full unrolling is just the truncated sweep with `K = T`, and
so I land the right hyperparameters. The Lagrangian view makes this clean: minimize `f(x, y_T)` subject
to the `T` constraints `y_t = Phi_t(y_{t-1}, x)`, attach a row multiplier `alpha_t` to each, and the
stationarity conditions give `alpha_T = grad_y f`, the backward recursion `alpha_{t-1} = alpha_t . A_t`,
and the gradient `d_x F = grad_x f + sum_t alpha_t . B_t`. The multiplier `alpha_t` is the adjoint
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
graph node, which the helper constructs internally inside its own `fp_map`. The exposed first-order
callables are for the penalty rungs that come later, not for unrolling.

I should be honest about the cost I am paying for the exactness, because it is the structural weakness
this whole family inherits and the reason the next rung will look elsewhere. Full RHG stores the entire
inner trajectory, `O(T * dim(y))` memory; for the MLP that is 500 copies of the parameter vector, which
is why the truncated variant existed in the first place. I am spending that memory here only to measure
whether it was worth spending — and the step-1 result already strongly suggests it was not, because
T-RHG at one fifth the memory matched the contraction prediction. There is a deeper reconciliation that
tells me the ceiling of this family is set by the inner solve, not by the differentiation: in the
converged limit the full backward sum is the Neumann series of the inverse Hessian, so RHG with `K = T`
is implicit differentiation computed by summation. That means RHG's hypergradient is only as good as the
inner optimizer's convergence — and here the inner optimizer is just 500 gradient-descent steps at a
fixed `lr_inner`, which is a *crude* inner solve compared to the tens of thousands of coupled steps the
penalty methods will take. So I expect the unrolling family, full or truncated, to be capped well below
the penalty methods not because the differentiation is wrong but because the inner problem is barely
solved before I differentiate it.

The toy, as in step 1, is decoupled from all of this. `run_rhg_family` in toy mode dispatches to the
same `_toy_pbgd_step` as the value-gap method: `S(x) = {-x}`, `v(x) = 0`, nothing to unroll, one
projected penalized step on `f + gamma * grad g` with `gams = (10.0,)`, `alpha0 = 0.1`. So I expect RHG's
toy line to be *identical* to T-RHG's — the same 260.7 mean convergence_steps, the same full success
rate, the same 0.030 residual — and any difference there would again be a wiring bug, not a property of
unrolling. The toy simply cannot separate these two rungs; only hyper-cleaning can, and only through the
deep trajectory terms.

So here is what I expect, stated against the step-1 numbers so it is falsifiable. On the toy, RHG should
reproduce T-RHG exactly: 260.7 mean steps, success 1.0, residual 0.030. On hyper-cleaning, RHG should
land *at or just above* T-RHG: linear test accuracy around 84.6 (T-RHG was 84.61) and MLP around 84.8
(T-RHG was 84.79), with linear f1 near 89.5 and the same high-recall, middling-precision cleaner
profile (recall ~0.97, precision ~0.83). If RHG matches T-RHG within seed noise, the truncation was free
— the contraction held and `K = 100` was already deep enough — and the unrolling family's ceiling on
this task is ~85% accuracy, set by the 500-step inner solve. If RHG instead jumps several points above
T-RHG, then the deep terms mattered and the inner problem is poorly conditioned. Either way, the gap to
the penalty methods — which I have not yet measured but expect to be large — will be the real story,
because it is the gap between "differentiate a crude inner solve" and "fold lower-level optimality into
the objective and descend it directly." The full scaffold edit is in the answer.
