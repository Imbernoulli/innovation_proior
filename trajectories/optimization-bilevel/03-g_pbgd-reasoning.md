Full RHG confirmed the thing I suspected at the end of step 1: it did not separate from the truncated
unroll. RHG landed at 84.63 mean linear test accuracy (84.16 / 84.82 / 84.92) and 84.79 on the hidden
MLP, against T-RHG's 84.61 and 84.79. Let me read that difference as a quantity rather than a shrug: the
linear gap is `84.63 - 84.61 = 0.02` points, against a per-seed standard deviation of about `0.37` that
I measured on the T-RHG seeds; the MLP number is identical to the digit. So spending five times the
memory to add back the deep trajectory terms moved the accuracy by roughly a twentieth of one seed's
worth of noise. The deep terms I was worried about truncating are, measurably, worth nothing. The
contraction held, the truncation was free — and the more diagnostic reading is that the whole unrolling
family is *capped* at about 85% accuracy on this task, independent of how much backward depth I buy. The
toy line was exactly the shared 260.7 steps in both rungs, as predicted, because the toy decouples. The
cap is the message.

RHG is, in the converged limit, implicit differentiation summed as a Neumann series, so its hypergradient
is only ever as good as the inner solve underneath it — and the inner solve here is 500 gradient-descent
steps at a fixed `lr_inner`, run fresh from a random init every single outer step. That is a *crude*
inner optimizer, and it is differentiated before it has really converged; the fact that the 30-times-larger
MLP tied the tiny linear model at ~85% already told me the wall is the solve, not the capacity. The 85%
ceiling is the price of solving the inner problem badly, not of differentiating it imperfectly. So the
move that could actually break the ceiling is to stop chasing the inner solution at all and instead fold
lower-level optimality directly into one joint objective over `(x, y)` that I descend with coupled
first-order steps, never resetting, never unrolling — reallocating the same evaluation budget from
re-solving a throwaway inner problem 100 times into one long coupled trajectory. That is the penalty
family, and the cheapest member of it — the one with an *exactly computable* penalty gradient and no
inner loop at all — is the squared lower-level gradient norm, G-PBGD.

Before I commit I want to be sure the penalty escape is genuinely better than the other two things I
could do with what the last two rungs taught me, because "the inner solve is the bottleneck" admits more
than one fix. Option one: keep unrolling but solve the inner problem harder — push `T` up, or warm-start
the inner loop so it does not restart from a random init each outer step. This directly attacks the
diagnosed cause, but it does not change the *structure*: I still store `O(K * dim(y))` states, still
differentiate through a solve, and warm-starting only helps if I also stop treating each outer step as an
independent inner problem — at which point I am halfway to a coupled method anyway and have kept the
memory overhead for nothing. Option two: compute the hypergradient by implicit differentiation directly,
solving `(grad_yy g) v = grad_y f` with a few conjugate-gradient or Neumann steps instead of a full
unroll. But the background already flags implicit differentiation as needing second-order information, and
worse, it *still* needs a lower minimizer `y*(x)` to evaluate the Hessian at — the same crude-solve
bottleneck, just relocated from the unroll into the linear solve. Neither option escapes the ceiling;
both keep paying to approximate `y*(x)` well. Option three is the penalty reformulation, which is
categorically different because it never needs `y*(x)` at a fixed `x` at all — it lets `y` and `x` descend
together toward joint stationarity of `F_gamma`. Within the penalty family there are again two members,
and the ordering is clear: the gradient norm has an *exact* penalty gradient computable with no inner
machinery, whereas the value gap needs `grad v(x)` and hence some estimate of a lower solution. So the
disciplined first penalty rung is the one that adds *zero* new approximation — the gradient norm — and I
save the value-gap sibling for when I have measured whether the gradient norm's known fragility actually
bites here.

The budget confirms the escape is affordable, not a compute blowout. Each G-PBGD step is one forward pass
to form `gxy` and `fy` plus one backward through `min(1/gamma,1)*(fy + 0.5*gamma*||grad_y g||^2)` — about
two gradient-scale evaluations, since the HVP rides inside a single extra backward. So `40000`–`50000`
coupled steps cost on the order of `80000`–`100000` gradient evaluations, the *same* order as RHG's
`100000`. The difference is entirely in how the budget is spent: RHG buys `100` outer moves on `x`, each
preceded by a throwaway 500-step re-solve; G-PBGD buys `40000`–`50000` genuine coupled moves on `x`, each
carrying `y` forward with it. Same compute, roughly four hundred times more outer progress — that is the
reallocation, made quantitative.

Let me derive why the gradient-norm penalty is the natural next thing and exactly where it is fragile,
because the fragility is what the toy numbers will expose. The bilevel constraint is `y in S(x) =
argmin_y g(x, y)`. Two scalar penalties measure it: the value gap `g(x, y) - v(x)` with
`v(x) = min_y g`, and the squared lower-level gradient norm `||grad_y g(x, y)||^2`. The gradient-norm
one is irresistible here because its gradient is exactly computable with first-order autodiff — no inner
solve, no value tracking. Differentiating `||grad_y g||^2 = grad_y g . grad_y g` gives `2 grad_yy g .
grad_y g` in `y` and `2 grad_xy g . grad_y g` in `x`: both are Hessian-vector products, the Hessian of
`g` times the already-computed vector `grad_y g`. I never materialize the Hessian; I compute `grad_y g`
keeping the graph (`create_graph=True`), form the scalar `f + (gamma/2)||grad_y g||^2`, and one more
backward hands me exactly `gamma grad_yy g . grad_y g` and `gamma grad_xy g . grad_y g` — and I should
check the constant, because it is easy to be off by a factor. Differentiating `(gamma/2)||grad_y g||^2`
in `y` gives `(gamma/2) . 2 . grad_yy g . grad_y g = gamma grad_yy g . grad_y g`: the `1/2` cancels the
`2` from the square exactly, so the objective's own backward pass produces the HVP with no stray factor.
Compare the value gap, whose gradient needs `grad v(x)`, i.e. knowledge of a lower solution `y*(x)` —
exactly the object the unrolling rungs were straining to approximate. So the gradient-norm penalty is the
*simplest possible* penalty member: plain coupled gradient descent on `F_gamma = f + (gamma/2)||grad_y
g||^2`, the penalty gradient available exactly, no inner optimizer anywhere. After watching the unrolling
family bottleneck on its inner solve, "no inner solve at all" is precisely the structural escape I want.

But I have to stress-test it, because the penalty method's reputation rests on "`gamma` large enough
implies penalized solution approximates bilevel solution," and lower-level non-convexity can break that.
The cleanest counterexample lives in one dimension: `min f = sin^2(y - 2pi/3)` subject to `y in
argmin g = y^2 + 2 sin^2 y`. Let me actually work it. The inner objective `g = y^2 + 2 sin^2 y` has
`grad_y g = 2y + 2 sin 2y = 2(y + sin 2y)`, which vanishes at `y = 0` (there `2 sin 0 = 0`), and `g` is a
nonnegative bowl there, so the only lower solution is `y* = 0` and the only bilevel solution is `y = 0`.
The penalty `(grad_y g)^2 = 4(y + sin 2y)^2` is a legitimate optimality metric, zero exactly on the
argmin. Now look at `y = 2pi/3`. Its penalty gradient is `d/dy[4(y + sin 2y)^2] = 8(y + sin 2y)(1 + 2 cos
2y)`, and `1 + 2 cos(4pi/3) = 1 + 2(-1/2) = 0`, so the penalty gradient vanishes there for *every*
`gamma`. And `f = sin^2(y - 2pi/3)` has `grad_y f = sin(2(y - 2pi/3))`, which is `sin 0 = 0` at
`y = 2pi/3`, so `f` is flat there too. So `y = 2pi/3` is a stationary point of `f + gamma (grad_y g)^2` at
*any* penalty strength, and it is not a solution — coordinate descent on the penalty can park at junk no
matter how hard I crank `gamma`. The decisive structural fact is *why* the penalty gradient died: at
`y = 2pi/3`, `grad_yy g = 2 + 4 cos 2y = 2 + 4(-1/2) = 0` — the lower-level Hessian degenerates — and
that degenerate factor kills the penalty gradient `2 grad_yy g . grad_y g` even though `grad_y g = 2(2pi/3
+ sin 4pi/3) = 2(2pi/3 - sqrt3/2) ≈ 2.46` is emphatically nonzero. The gradient-norm penalty can stall
wherever `grad_yy g` goes singular. That is its signature weakness, and it is exactly why the value-gap
penalty is the more robust sibling: the value gap's local stationarity is `grad_y f + gamma grad_y g = 0`,
which controls `||grad_y g|| <= L/gamma` directly, with no `grad_yy g` anywhere in the way.

The framework that makes this precise is the squared-distance bound, and it is worth grinding through
because it turns "large enough `gamma`" into an actual finite number. Write the constraint as
`d^2_{S(x)}(y) = 0`. A penalty `p` is a `rho`-squared-distance bound if `p >= 0`, `rho p >=
d^2_{S(x)}(y)`, and `p = 0` exactly on `S(x)`. Under the lower level being `(1/mu)`-PL plus smooth, the
error bound `g - v >= (1/mu) d^2` makes the value gap a `mu`-bound with one PL chain. The gradient norm
needs the *stronger* `(1/sqrt(mu))`-PL to chain twice: `||grad_y g||^2 >= (1/mu)(g - v) >= (1/mu^2) d^2`,
two PL applications stacked, so it is one PL step weaker a surrogate than the value gap — the same fact
the counterexample showed geometrically now shows up as an extra assumption. With an SDB and
`L`-Lipschitz `f`, the calmness inequality bounds how far below `f`'s own values the penalized objective
can dip: `min_{z >= 0} (-L z + (gamma/rho) z^2)`, minimized at `z* = L rho/(2 gamma)` with value
`-L z* + (gamma/rho) z*^2 = -L^2 rho/(2 gamma) + L^2 rho/(4 gamma) = -L^2 rho/(4 gamma)`. So globals of
`f + gamma p` have penalty residual `O(1/gamma)`, and a second-order version gives locals `O(1/gamma^2)`
— but the *local* bound for the gradient-norm penalty needs the singular values of `grad_yy g` bounded
below by some `sigma > 0`, from `||grad_yy g . grad_y g|| >= sigma ||grad_y g||`, the exact condition
`y = 2pi/3` violates. So the gradient-norm penalty is principled *when the lower Hessian does not
degenerate*, and `gamma = Theta(delta^{-1/2})` is the tight scaling for a target residual `delta` — a
finite `gamma`, not `gamma -> infinity`.

I want to connect that abstract degeneracy to *this* problem, because it is not a pathology I have to
hunt for — the linear hyper-cleaner has it built in. The inner objective is a `sigmoid(x)`-weighted
softmax cross-entropy on the linear map `784 -> 10`. Its Hessian in the weights is rank-deficient in real
directions: wherever a batch of examples is nearly linearly separable, the winning logits run off toward
saturation and the softmax curvature in those directions collapses toward zero, and wherever training
features are collinear, whole subspaces of weight space leave the loss unchanged. Those are precisely the
`grad_yy g ≈ 0` directions the counterexample warned about, and the gradient-norm penalty's step
`grad_yy g . grad_y g` is small exactly there — so on those examples the penalty barely pushes, and the
cleaner's decision about them is driven by `f` and noise rather than by lower-stationarity. That is my
mechanistic reason to expect the *cleaner's* quality, not its raw accuracy, to be where the gradient-norm
penalty pays for its fragility.

Now the practical knobs, which fall out of the analysis and which I have to wire into this task's helper,
not invent. `F_gamma` is `L_gamma`-smooth with `L_gamma` growing linearly in `gamma` (the penalty
contributes `gamma` times the smoothness of `||grad_y g||^2`), so a stable step needs `alpha <~
1/L_gamma`; a huge `gamma` from step one is a stiff landscape that forces a tiny step before `y` reaches
the lower valley. So I ramp `gamma` from `0` to a finite cap over a fixed number of steps — start as
essentially pure validation-loss descent, then phase in the lower-level penalty. And once `gamma > 1` the
penalty term `gamma grad p` dominates the gradient, so I rescale the whole joint step by `min(1/gamma,
1)`, keeping the effective penalty step order-one rather than order-`gamma`; at the target `gamma = 37`
that rescale is `1/37 ≈ 0.027`, so the raw joint gradient is shrunk to about `2.7%` of itself, which is
exactly what keeps a landscape `37` times stiffer from blowing up the step. The scaffold's `run_g_pbgd`
already implements exactly this: it forms `gxy = (sigmoid(x)*CE_train).mean()`, takes `dgdy =
autograd.grad(gxy, params, create_graph=True)`, sets `objective = min(1/(gamma+eps),1) * (fy +
0.5*gamma*||dgdy||^2)`, backprops, and steps both `x_opt` (lr `lrx`) and `y_opt` (lr `lry`) with `gamma`
annealed linearly from `gamma_init` to `gamma_max` over `gamma_argmax_step`. So my edit is to point
`algorithm` at `run_g_pbgd` and set the schedule. Critically — and this is where this task's G-PBGD
departs from a generic gradient-norm penalty — the helper does *not* use the exposed
`inner_grad/outer_grad` callables either; it builds the penalty gradient directly from the model and the
`sigmoid(x)` weights with a retained graph, so the HVP is exact.

The hyperparameters I take are the aggressive penalty schedule this task's reference uses: linear
`lrx=0.3, lry=0.5, gamma_max=37.0, gamma_argmax_step=5000, outer_itr=40000`; MLP `lrx=0.5, lry=0.5,
gamma_max=37.0, gamma_argmax_step=30000, outer_itr=50000`. Read the schedule as arithmetic. The large
`gamma_max = 37` is the `Theta(delta^{-1/2})` regime pushed hard: by the calmness bound a global
residual scales like `1/gamma ≈ 1/37 ≈ 0.027` and a local one like `1/gamma^2 ≈ 1/1369 ≈ 7.3e-4`, so
`37` is chosen to pin `y` onto the lower-stationarity manifold to a residual on the order of `1e-3`.
The `min(1/gamma,1)` rescale is what makes that strength survivable. The ramp length differs because the
landscapes differ: the linear model reaches `gamma_max` in `5000` of its `40000` steps (an eighth of the
run at full strength is enough for a convex inner loss), while the MLP, whose sigmoid hidden layer makes
`L_gamma` larger and the loss stiffer, spreads the ramp over `30000` of `50000` steps so the step size
never has to collapse faster than `y` can follow. And there is no `lr_inner`, no `T`, no `K` — there is
no inner loop at all, which is the whole point of escaping the unrolling ceiling: the entire `40000`–
`50000`-step budget goes into one persistent coupled trajectory rather than into re-solving 100 throwaway
inner problems.

There is a second reason to ramp `gamma` rather than start it at `37`, beyond keeping the step below
`1/L_gamma`, and it ties directly back to the fragility. Annealing `gamma` from `0` is a homotopy: at
`gamma = 0` the objective is pure validation loss `f`, an easy and smooth (if under-determined) landscape
in `(x, y)`; as `gamma` grows it continuously deforms toward the constrained bilevel problem, dragging
`y` onto the lower-stationarity manifold a little more at each step. Following that path keeps `(x, y)`
inside the basin that tracks the true solution, whereas dropping the iterate straight into the
`gamma = 37` landscape at a random init risks landing near exactly the degenerate `grad_yy g ≈ 0`
directions where the counterexample's spurious stationary points live — precisely the traps the linear
cross-entropy's flat directions create. So the ramp is doing double duty: it protects the step size, and
it steers around the gradient-norm penalty's own bad basins by approaching them slowly. That said, it
cannot repair the fragility, only avoid triggering it at initialization; wherever the penalty gradient
genuinely vanishes at a non-solution, no schedule saves it, which is why I still expect the cleaner to pay.

The toy is where the gradient-norm penalty's fragility becomes *measurable*, and I should predict it
honestly against the unrolling rungs' toy numbers. Unlike RHG and T-RHG, whose toy mode dispatched to the
value-gap projected step, `run_g_pbgd` in toy mode dispatches to `_toy_pbgd_step` with method `"g_pbgd"`,
which uses `toy_gpbgd_penalty_grad` — the gradient of `||grad_y g||^2` — instead of `grad g`. So G-PBGD
descends a *different*, stiffer toy objective, and I can work out exactly how stiff. On the toy, with
`u = x + y`, `grad_y g = 2u + x sin 2u`, so `grad_yy g = 2 + 2x cos 2u`. The gradient-norm penalty is
`(grad_y g)^2 = (2u + x sin 2u)^2`, whose `y`-derivative is `2(2u + x sin 2u)(2 + 2x cos 2u) = 2 grad_y g
. grad_yy g`. Near the solution valley `u = 0` (`y = -x`), `grad_y g ≈ 2u` and `grad_yy g ≈ 2 + 2x`, so
the penalty gradient behaves like `2 . (2u) . (2 + 2x) = 8u(1 + x)` — proportional to `(1 + x)`, whereas
the value-gap-style step there descends `grad g ≈ 2u` with no `(1 + x)` amplification. So over the domain
`x in [0, 3]` the gradient-norm toy is up to four times stiffer near the valley, and stiffer landscapes at
a fixed step `alpha0 = 0.1` take more iterations to settle under the tolerance and settle at a coarser
residual. Concretely I expect the toy convergence_steps to rise from the shared 260.7 to roughly 300, with
a larger residual (on the order of 0.06–0.10 rather than 0.030) — the penalty pins the
gradient norm small but at a coarser residual. This is not a regression I am introducing by mistake; it
is the gradient-norm penalty's geometry showing through, the same `grad_yy g`-sensitivity that made
`y = 2pi/3` spurious in the counterexample.

So my falsifiable expectations against the prior numbers. On hyper-cleaning I expect G-PBGD to *break the
unrolling ceiling decisively*: the penalty descends one coupled objective for tens of thousands of steps
without ever resetting an inner solve, so I expect MLP test accuracy to jump from RHG's 84.79 into the
low 90s — a gain of several points, decisively clear of the unrolling plateau — and linear accuracy from
84.63 up toward about 90. But I expect a cost on the cleaner f1, because the aggressive gradient-norm penalty drives
precision up at the expense of recall — I expect linear f1 to actually *drop* below the unrolling rungs
(from RHG's ~89.5 toward ~80), with cleaner recall falling to about 0.78 even as cleaner *precision* rises — I expect precision to move
the other way, up toward ~0.85 from the unrolling family's ~0.83, because a hard penalty discards
aggressively and the survivors are cleaner, so the f1 drop is a recall story, not a precision one — because
the penalty is tuned for accuracy, not for the precision-recall balance the soft, under-solved unroll
happened to strike; and
I expect that f1 to be *unstable* across seeds, because it is hostage to the degenerate `grad_yy g`
directions I identified in the linear cross-entropy, which differ init to init. If G-PBGD instead matched
the unrolling toy numbers, the toy dispatch would not be using the gradient-norm penalty; if its MLP
accuracy did not break 90, the penalty escape would have failed and the bottleneck would not have been
the inner solve after all; and if its cleaner f1 came out as high and stable as the unrolling family's,
then the lower Hessian is better conditioned than the counterexample fears and the robustness argument
for the value-gap sibling loses its motivation. The full scaffold edit is in the answer.
