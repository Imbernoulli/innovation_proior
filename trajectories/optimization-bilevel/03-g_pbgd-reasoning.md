Full RHG confirmed the thing I suspected at the end of step 1: it did not separate from the truncated
unroll. RHG landed at 84.63 mean linear test accuracy (84.16 / 84.82 / 84.92) and 84.79 on the hidden
MLP, against T-RHG's 84.61 and 84.79 — identical within seed noise, with the same high-recall (0.969),
middling-precision (0.832) cleaner profile and linear f1 of 89.55 versus T-RHG's 89.06. So the deep
trajectory terms I spent five times the memory to include moved nothing. The contraction held, the
truncation was free, and — more diagnostically — the whole unrolling family is *capped* at about 85%
accuracy on this task. The toy line was exactly the shared 260.7 steps, as predicted, because the toy
decouples. The cap is the message. RHG is, in the converged limit, implicit differentiation summed as a
Neumann series, so its hypergradient is only ever as good as the inner solve underneath it — and the
inner solve here is 500 gradient-descent steps at a fixed `lr_inner`, run fresh from a random init every
single outer step. That is a *crude* inner optimizer, and it is differentiated before it has really
converged. The 85% ceiling is the price of solving the inner problem badly, not of differentiating it
imperfectly. So the move that could actually break the ceiling is to stop chasing the inner solution at
all and instead fold lower-level optimality directly into one joint objective over `(x, y)` that I
descend with coupled first-order steps, never resetting, never unrolling. That is the penalty family,
and the cheapest member of it — the one with an *exactly computable* penalty gradient and no inner loop
— is the squared lower-level gradient norm, G-PBGD.

Let me derive why the gradient-norm penalty is the natural next thing and exactly where it is fragile,
because the fragility is what the toy numbers will expose. The bilevel constraint is `y in S(x) =
argmin_y g(x, y)`. Two scalar penalties measure it: the value gap `g(x, y) - v(x)` with
`v(x) = min_y g`, and the squared lower-level gradient norm `||grad_y g(x, y)||^2`. The gradient-norm
one is irresistible here because its gradient is exactly computable with first-order autodiff — no inner
solve, no value tracking. Differentiating `||grad_y g||^2 = grad_y g . grad_y g` gives `2 grad_yy g .
grad_y g` in `y` and `2 grad_xy g . grad_y g` in `x`: both are Hessian-vector products, the Hessian of
`g` times the already-computed vector `grad_y g`. I never materialize the Hessian; I compute `grad_y g`
keeping the graph (`create_graph=True`), form the scalar `f + (gamma/2)||grad_y g||^2`, and one more
backward hands me exactly `gamma grad_yy g . grad_y g` and `gamma grad_xy g . grad_y g` — the `1/2`
cancels the `2` from differentiating the square. Compare to the value gap, whose gradient needs
`grad v(x)`, i.e. knowledge of a lower solution `y*(x)` — exactly the object the unrolling rungs were
straining to approximate. So the gradient-norm penalty is the *simplest possible* penalty member: plain
coupled gradient descent on `F_gamma = f + (gamma/2)||grad_y g||^2`, the penalty gradient available
exactly, no inner optimizer anywhere. After watching the unrolling family bottleneck on its inner solve,
"no inner solve at all" is precisely the structural escape I want.

But I have to stress-test it, because the penalty method's reputation rests on "`gamma` large enough
implies penalized solution approximates bilevel solution," and the lower-level non-convexity can break
that. The cleanest counterexample lives in one dimension: `min f = sin^2(y - 2pi/3)` subject to `y in
argmin g = y^2 + 2 sin^2 y`. The only lower solution is `y* = 0`, so the only bilevel solution is
`y = 0`. The penalty `(grad_y g)^2 = (2y + 2 sin 2y)^2` is a legitimate optimality metric, zero exactly
on the argmin. Now look at `y = 2pi/3`: the derivative of `(y + sin 2y)^2` is `2(y + sin 2y)(1 + 2 cos
2y)`, and `1 + 2 cos(4pi/3) = 1 + 2(-1/2) = 0`, so the penalty gradient vanishes there for *every*
`gamma`, and `f` is flat there too. So `y = 2pi/3` is a stationary point of `f + gamma(grad_y g)^2` at
any penalty strength, and it is not a solution. Coordinate descent on the penalty can park at junk. The
decisive structural fact is that at `y = 2pi/3`, `grad_yy g = 2 + 4 cos 2y = 0` — the lower-level
Hessian degenerates — and that degenerate factor kills the penalty gradient `2 grad_yy g . grad_y g`
even though `grad_y g` itself is nonzero. The gradient-norm penalty can stall wherever `grad_yy g` goes
singular. This is the gradient-norm penalty's signature weakness, and it is exactly why the *value-gap*
penalty (the next rung) is the more robust sibling: the value gap's local stationarity is `grad_y f +
gamma grad_y g = 0`, controlling `||grad_y g|| <= L/gamma` directly with no `grad_yy g` in the way.

The framework that makes this precise is the squared-distance bound. Write the constraint as
`d^2_{S(x)}(y) = 0`. A penalty `p` is a `rho`-squared-distance bound if `p >= 0`, `rho p >=
d^2_{S(x)}(y)`, and `p = 0` exactly on `S(x)`. Under the lower level being `(1/mu)`-PL plus smooth, the
error bound `g - v >= (1/mu) d^2` makes the value gap a `mu`-bound with one PL chain; the gradient norm
needs the *stronger* `(1/sqrt(mu))`-PL to chain twice (`||grad_y g||^2 >= (1/mu)(g - v) >= (1/mu^2)
d^2`), so it is one PL step weaker a surrogate. With an SDB and `L`-Lipschitz `f`, the calmness
inequality `min_{z>=0}(-Lz + (gamma/rho)z^2) = -L^2 rho/(4 gamma)` shows globals of `f + gamma p` have
penalty residual `O(1/gamma)`, and locals `O(1/gamma^2)` — but the local bound for the gradient-norm
penalty needs the singular values of `grad_yy g` bounded below by some `sigma > 0` (from
`||grad_yy g . grad_y g|| >= sigma ||grad_y g||`), the exact condition `y = 2pi/3` violates. So the
gradient-norm penalty is principled *when the lower Hessian does not degenerate*, and `gamma =
Theta(delta^{-1/2})` is the tight scaling for a target residual `delta` — a finite `gamma`, not
`gamma -> infinity`.

Now the practical knobs, which fall out of the analysis and which I have to wire into this task's
helper, not invent. `F_gamma` is `L_gamma`-smooth with `L_gamma` growing linearly in `gamma`, so a
stable step needs `alpha <~ 1/L_gamma`; a huge `gamma` from step one is a stiff landscape that forces a
tiny step before `y` reaches the lower valley. So I ramp `gamma` from `0` to a finite cap over a fixed
number of steps — start as essentially pure validation-loss descent, then phase in the lower-level
penalty. And once `gamma > 1` the penalty term `gamma grad p` dominates the gradient, so I rescale the
whole joint step by `min(1/gamma, 1)`, keeping the effective penalty step order-one rather than
order-`gamma`. The scaffold's `run_g_pbgd` already implements exactly this: it forms `gxy =
(sigmoid(x)*CE_train).mean()`, takes `dgdy = autograd.grad(gxy, params, create_graph=True)`, sets
`objective = min(1/(gamma+eps),1) * (fy + 0.5*gamma*||dgdy||^2)`, backprops, and steps both `x_opt`
(lr `lrx`) and `y_opt` (lr `lry`) with `gamma` annealed linearly from `gamma_init` to `gamma_max` over
`gamma_argmax_step`. So my edit is to point `algorithm` at `run_g_pbgd` and set the schedule. Critically
— and this is where this task's G-PBGD departs from a generic gradient-norm penalty — the helper does
*not* use the exposed `inner_grad/outer_grad` callables either; it builds the penalty gradient directly
from the model and the `sigmoid(x)` weights with a retained graph, so the HVP is exact.

The hyperparameters I take are the aggressive penalty schedule this task's reference uses: linear
`lrx=0.3, lry=0.5, gamma_max=37.0, gamma_argmax_step=5000, outer_itr=40000`; MLP `lrx=0.5, lry=0.5,
gamma_max=37.0, gamma_argmax_step=30000, outer_itr=50000`. The large `gamma_max=37` is the
`Theta(delta^{-1/2})` regime pushed hard so the gradient-norm penalty pins `y` tightly onto the
lower-stationarity manifold; the `min(1/gamma,1)` rescale is what keeps that stable. There is no
`lr_inner`, no `T`, no `K` — there is no inner loop at all, which is the whole point of escaping the
unrolling ceiling.

The toy is where the gradient-norm penalty's fragility becomes *measurable*, and I should predict it
honestly against the unrolling rungs' toy numbers. Unlike RHG and T-RHG, whose toy mode dispatched to
the value-gap projected step, `run_g_pbgd` in toy mode dispatches to `_toy_pbgd_step` with method
`"g_pbgd"`, which uses `toy_gpbgd_penalty_grad` — the gradient of `||grad_y g||^2` — instead of
`grad g`. So G-PBGD descends a *different*, stiffer toy objective: the gradient-norm penalty's landscape
near `S(x) = {-x}` is sharper and its curvature factor `grad_yy g` modulates the step, so I expect it to
need *more* steps to hit the stationarity tolerance than the value-gap-style rungs did. Concretely I
expect the toy convergence_steps to rise from the shared 260.7 to roughly 300, with a larger residual
(~0.06-0.10 rather than 0.030) — the penalty pins the gradient norm small but at a coarser residual.
This is not a regression I am introducing by mistake; it is the gradient-norm penalty's geometry showing
through, the same `grad_yy g`-sensitivity that made `y = 2pi/3` spurious in the counterexample.

So my falsifiable expectations against the prior numbers. On hyper-cleaning I expect G-PBGD to *break
the unrolling ceiling decisively*: the penalty descends one coupled objective for tens of thousands of
steps without ever resetting an inner solve, so I expect MLP test accuracy to jump from RHG's 84.79 into
the low 90s (I expect roughly 92), and linear accuracy from 84.63 to about 90. But I expect a cost on
the cleaner f1, because the aggressive gradient-norm penalty drives precision up at the expense of
recall — I expect linear f1 to actually *drop* below the unrolling rungs (from ~89.5 toward ~80), with
cleaner recall falling to ~0.78, because the penalty is tuned for accuracy, not for the precision-recall
balance the unroll happened to strike. And on the toy I expect a *worse* convergence count than the
unrolling rungs — ~300 steps, residual ~0.08 — because of the stiffer gradient-norm landscape. If
G-PBGD instead matched the unrolling toy numbers, the toy dispatch would not be using the gradient-norm
penalty; if its MLP accuracy did not break 90, the penalty escape would have failed and the bottleneck
would not have been the inner solve after all. The full scaffold edit is in the answer.
