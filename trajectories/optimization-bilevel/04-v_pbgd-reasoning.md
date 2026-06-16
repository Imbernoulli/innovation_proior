G-PBGD broke the unrolling ceiling exactly where I expected and broke something else exactly where I
warned it would. On the hidden MLP it jumped to 92.38 test accuracy — the best number so far, eight
points above RHG's 84.79 — confirming that the bottleneck was never the differentiation but the crude
500-step inner solve, and that descending one coupled penalized objective for tens of thousands of steps
escapes it. Linear accuracy rose to 89.84, also well above the unrolling rungs. But look at what the
gradient-norm penalty cost. The cleaner f1 on the linear model *collapsed* to 80.63 mean (76.67 / 81.69
/ 83.52 across seeds — enormous variance, a 7-point spread) against the unrolling family's stable ~89.5,
and the cause is visible in the cleaner profile: precision held at 0.839 but recall fell to 0.776, so
the aggressive `gamma_max = 37` penalty drove the cleaner to throw away too many genuinely clean
examples to keep its lower-stationarity small. And the toy regressed exactly as predicted: 303.7 mean
convergence_steps (up from the shared 260.7) at a residual of 0.081 (up from 0.030), because the toy
dispatched to the gradient-norm penalty's stiffer landscape. So G-PBGD bought MLP accuracy with a
fragile, high-variance cleaner and a worse toy convergence. The diagnosis is the one I rehearsed in the
counterexample: the gradient-norm penalty's local stationarity only controls `grad_yy g . grad_y g`, so
its behavior is hostage to the lower-level curvature `grad_yy g` — which is what makes both its f1
unstable and its toy residual coarse. The fix is the *robust* member of the penalty family, whose local
stationarity controls `grad_y g` directly with no curvature factor in the way: the value-gap penalty,
V-PBGD.

Let me derive why the value gap removes exactly the two failures I just measured. The penalty is
`p = g(x, y) - v(x)` with `v(x) = min_y g(x, y)`, and the single structural difference from the gradient
norm is where the Hessian appears in the local relation. At a local solution of `f + gamma p`,
stationarity in `y` is `grad_y f + gamma grad_y g = 0`, because `v` carries no `y`-dependence at all. So
`||grad_y g|| <= L/gamma` *directly* — no `grad_yy g`, no singular-value condition. Under the lower level
being `(1/mu)`-PL, the error bound `g - v >= (1/mu) d^2` then gives `p <= mu L^2/gamma^2 = O(1/gamma^2)`
with one PL chain, so the value gap is a `mu`-squared-distance bound under *weaker* assumptions than the
gradient norm needed (`1/sqrt(mu)`-PL). The contrast is the whole story of step 3's failure: the
gradient-norm penalty's `||grad_yy g . grad_y g|| <= L/(2 gamma)` only bounds `||grad_y g||` after
dividing by the smallest singular value of `grad_yy g`, which vanishes wherever the lower Hessian
degenerates — and the linear hyper-cleaner's cross-entropy has exactly such flat directions, which is
why its f1 swung 7 points across seeds. The value gap has no such division, so it does not get whipsawed
by lower-level curvature; I expect its cleaner f1 to be both *higher and far more stable* than G-PBGD's.

I want to be sure the value gap is a faithful penalty at the global level too, not just a convenient
local fix, because the f1 instability G-PBGD showed was a *variance* problem and I need the value gap to
be principled across the whole landscape. The squared-distance-bound argument gives it cleanly. Take any
feasible `(x, y)`, project `y` to `y_x in S(x)` so `d = d_{S(x)}(y) = ||y_x - y||`, and use
`L`-Lipschitz `f`: `f(x, y) - f(x, y_x) >= -L d`. Add `gamma p` and the bound `p >= d^2/mu`: the whole
expression is at least `-L d + (gamma/mu) d^2`, a scalar quadratic in `d >= 0` whose minimum over `d` is
`-L^2 mu/(4 gamma)`. Setting `gamma* = L^2 mu/(4 eps_1)` floors it at `-eps_1`, so any bilevel global
solution (where `p = 0`) is an `eps_1`-global-min of the penalized problem, and conversely an
`eps_2`-global-min of the penalized problem with `gamma > gamma*` has `p <= (eps_1 + eps_2)/(gamma -
gamma*)` — penalty residual shrinking like `O(1/gamma)` globally and `O(1/gamma^2)` locally. The lever
is the quadratic `(gamma/mu) d^2`, present only because `p` *dominates* `d^2`, overpowering the linear
Lipschitz slack `-L d` once `gamma` clears the threshold. And the `gamma = Theta(delta^{-1/2})` scaling
is genuinely tight, not loose analysis: on the trivial instance `f = y, g = y^2` the penalized minimizer
is `y = -1/(2 gamma)` with gap `1/(4 gamma^2)`, so forcing the gap below `delta` requires `gamma >=
1/(2 sqrt(delta))`. A *finite* `gamma` suffices for any finite accuracy — which is exactly why this
task's reference can use `gamma_max = 0.1-0.2` rather than the huge value the gradient-norm penalty
needed, and the gentleness is what protects the cleaner's recall.

The one subtlety the value gap introduces is computing `grad v(x)`, and the PL Danskin lemma dissolves
it. Naively `grad v` needs a lower solution `y*(x)` and looks like the implicit machinery I have been
avoiding. But `v(x) = g(x, y*(x))`, and the chain rule gives `grad v = grad_x g(x, y*) + (dy*/dx)^T
grad_y g(x, y*)`; at an unconstrained lower minimum `grad_y g(x, y*) = 0`, so the scary implicit term
multiplies by zero and `grad v(x) = grad_x g(x, y*)` for *any* `y* in S(x)` — no Hessian, no
implicit-function theorem, no strong convexity. So I only need an *approximate* lower minimizer, and I
estimate it with a short inner loop, warm-started so the inner gap stays tied to outer progress. The
warm start matters for a reason the analysis pins down: inner gradient descent contracts the value gap
geometrically under PL, `g(x, omega_{t+1}) - v <= (1 - beta/(2 mu))(g(x, omega_t) - v)`, but that bound
carries the *initial* gap `g(x_k, omega_1) - v(x_k)`, and `x_k` drifts across outer iterations — a cold
start at an unrelated point could let the initial gap grow without bound, so no fixed inner-step count
would control the estimation error. Warm-starting `omega_1 = y_k` ties the initial inner gap to the
outer step size through PL (`g(x_k, y_k) - v <= (1/mu)||grad_y g||^2`, and `grad_y g` is controlled by
the outer move), so a logarithmically short inner loop keeps the gradient-estimation error
`gamma^2 L_g^2 d^2_{S(x)}(y_hat)` summable and the outer projected descent telescopes to a `1/K`
stationarity rate. This is why a *single* warm-started inner step (`inner_itr = 1`) is enough here:
because the inner network persists and is never reset, each outer iteration only has to close the small
incremental gap that the last outer move opened. The
value-gap direction is then `h = grad g(x, y) - (grad_x g(x, y_hat), 0)` — the subtraction lives only in
the `x`-coordinates, because `v` depends only on `x`, so in `y` the value-gap gradient is just
`grad_y g(x, y)`.

This is where this task's V-PBGD is concrete and where I must match the harness, not the generic method.
The scaffold's `run_v_pbgd` keeps a *persistent auxiliary* inner network `net_inner`, initialized from
the main model and updated across outer iterations by `inner_itr` SGD steps on the `sigmoid(x)`-weighted
training loss at frozen weights `sigx = sigmoid(x).detach()`. It then forms `fy = CE(val, clean)`,
`gxy = (sigmoid(x)*CE_train).mean()` through the *main* model, and `vx = (sigmoid(x)*CE_inner).mean()`
through `net_inner` with the inner outputs *detached* — so the gradient through `vx` flows only via
`sigmoid(x)`, giving exactly `grad_x g(x, y_hat)` and nothing through `y_hat`. The objective is
`min(1/(gamma+eps),1) * (fy + gamma*(gxy - vx))`, backpropped to step both `x_opt` and `y_opt`, with
`gamma` ramped linearly from 0 to `gamma_max`. So the value-gap subtraction `gxy - vx` is realized by
the detach pattern, and the `min(1/gamma,1)` rescale is the same stabilizer as G-PBGD. My edit is to
point `algorithm` at `run_v_pbgd` and set the schedule. Note this rung *does* run a real inner loop
(`inner_itr` SGD steps on `net_inner`) — unlike G-PBGD which had none — but it is a single warm-started
step per outer iteration, not the 500-step-from-scratch unroll the RHG family paid for; that is the
distinction between approximating `grad v` cheaply and differentiating a full inner solve. And like every
prior rung, it ignores the exposed `inner_grad/outer_grad/inner_val` callables; the helper builds the
gradients directly from the two networks and `sigmoid(x)`.

The hyperparameters are the gentle schedule this task's reference uses, and the contrast with G-PBGD's
`gamma_max = 37` is itself diagnostic: linear `lrx=lry=0.1, lr_inner=0.01, inner_itr=1, gamma_max=0.2,
gamma_argmax_step=30000, outer_itr=40000`; MLP `lrx=0.1, lry=0.01, lr_inner=0.01, inner_itr=1,
gamma_max=0.1, gamma_argmax_step=10000, outer_itr=80000`. The tiny `gamma_max` (0.1-0.2 versus 37) is
possible *because* the value gap is the more faithful surrogate — it does not need to be slammed to a
huge strength to pin `y`, since its `O(1/gamma^2)` local residual is achieved at modest `gamma`, and a
gentle penalty is exactly what keeps the cleaner's recall from collapsing the way G-PBGD's did. The
longer MLP horizon (80k outer steps) lets the gentle penalty converge fully.

The toy, finally, returns to the well-behaved regime, and this is a clean falsifiable prediction against
step 3. `run_v_pbgd` in toy mode dispatches to `_toy_pbgd_step("v_pbgd")`, which uses `grad g` (the
value-gap gradient, since `S(x) = {-x}` makes `v(x) = 0` and `p = g`), *not* the gradient-norm penalty.
So V-PBGD descends the same smooth landscape the unrolling rungs did, and I expect its toy line to snap
back to the shared 260.7 mean convergence_steps at 0.030 residual and full success — recovering the ~43
steps and the residual that G-PBGD lost to its stiffer gradient-norm toy objective. If the toy did not
return to 260.7, the toy dispatch would not be using the value gap.

So my falsifiable expectations against the measured numbers, since this is meant to be the strongest
rung. On the toy I expect V-PBGD to *recover* the unrolling family's 260.7 steps and 0.030 residual,
beating G-PBGD's 303.7 / 0.081 — fixing the toy regression. On hyper-cleaning I expect it to keep the
penalty family's accuracy win over the unrolling family while *fixing the f1 collapse*: linear accuracy
around 90 (G-PBGD was 89.84, so comparable or slightly better, and above the unrolling 84.6), but with
linear f1 back up around 91.7 — eleven points above G-PBGD's fragile 80.63 — and a stable cleaner recall
near 0.95 rather than G-PBGD's 0.78. On the hidden MLP I expect ~91.5, which would sit just below
G-PBGD's 92.38: the gentle value-gap penalty trades roughly a point of raw MLP accuracy for a far more
faithful, lower-variance cleaner and a recovered toy convergence. That is the trade I am willing to
make, and it is why the value gap is the robust default: it wins the toy primary metric, the linear
primary metric, and both f1 scores decisively, losing only the single hidden MLP accuracy by about a
point. If the value gap instead matched G-PBGD's broken f1, the detach pattern would not be realizing the
`gxy - vx` value gap and I would have wired the wrong subtraction. The full scaffold edit is in the
answer.
