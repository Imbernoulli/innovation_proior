G-PBGD broke the unrolling ceiling exactly where I expected and broke something else exactly where I
warned it would. On the hidden MLP it jumped to 92.38 test accuracy — the best number so far, about eight
points above RHG's 84.79 — confirming that the bottleneck was never the differentiation but the crude
500-step inner solve, and that descending one coupled penalized objective for tens of thousands of steps
escapes it. Linear accuracy rose to 89.84, also well above the unrolling rungs. But now I have to read
what the gradient-norm penalty cost, and I want to do it with the numbers rather than a wave of the hand.

The cleaner f1 on the linear model *collapsed* to 80.63 mean, and the collapse has a specific shape.
Across seeds it was 83.52 / 76.67 / 81.69 — a range of 6.85 points — against the unrolling family's
stable ~89.5 with a seed range of only 0.33. So the penalty f1 is not just lower, it is about twenty-one
times more variable init to init. And when I decompose the drop into precision and recall the cause is
unambiguous: G-PBGD's cleaner precision is 0.839, essentially unchanged from RHG's 0.832 (a move of
+0.007), while its recall *crashed* from RHG's 0.969 to 0.776, a drop of 0.193. The f1 collapse is
entirely a recall failure — I can check it closes: `2 * 0.839 * 0.776 / (0.839 + 0.776) ≈ 1.302 / 1.615
≈ 0.806`, reproducing the 80.63. So the aggressive `gamma_max = 37` penalty drove the cleaner to throw
away too many genuinely clean examples to keep its lower-stationarity small: it is over-discarding, the
opposite failure from the unroll's soft over-keeping.

There is one more number that tells me *where* this fragility lives. On the MLP the same penalty did *not*
wreck the cleaner: recall only slipped from 0.979 to 0.928, precision rose from 0.822 to 0.889, and the
net MLP f1 went *up*, 90.82 versus RHG's 89.34. So the gradient-norm penalty is a clean win on the MLP
cleaner and a disaster on the linear one — its own f1 splitting ten points across models, 90.82 versus
80.63, for the *same* penalty at the *same* `gamma = 37`. Nothing changed between the two runs except the
inner architecture, so that gap can only come from the inner *curvature* differing — exactly what the
counterexample predicted. The *linear* softmax cross-entropy is riddled with flat `grad_yy g ≈ 0`
directions — near-separable batches whose logits saturate, collinear pixel features whose weight subspaces
are unidentified — where the penalty step `grad_yy g . grad_y g` vanishes; the MLP's sigmoid-warped loss
keeps enough curvature that it does not. And the toy regressed as G-PBGD anticipated: 303.7 mean
convergence_steps at 0.081 residual, up from the shared 260.7 / 0.030, because it dispatched to the
gradient-norm penalty's stiffer landscape. A method whose cleaner quality is that sensitive to the
lower-level Hessian is the wrong default: its local stationarity only controls `grad_yy g . grad_y g`,
hostage to `grad_yy g`. The fix is the *robust* member of the penalty family, whose local stationarity
controls `grad_y g` directly with no curvature factor in the way: the value-gap penalty, V-PBGD.

Before I commit to the value gap I should check that it beats the two cheaper repairs I could try on
G-PBGD itself, because switching penalties is not free. The first repair is to just turn `gamma` down: if
`gamma = 37` over-discards, why not run the gradient-norm penalty at a gentle `gamma`? The SDB accounting
forbids it. The gradient norm is an `O(1/gamma^2)` bound only after *two* PL chains, so to pin `y` to a
given lower-stationarity residual it needs a substantially *larger* `gamma` than the value gap does; drop
`gamma` and the penalty stops pinning `y` onto the lower manifold, and the accuracy win — the whole reason
I left the unrolling family — evaporates. The gradient norm is caught in a vise: large `gamma` crashes the
linear recall through the degenerate `grad_yy g`, small `gamma` gives up the pinning, and there is no
middle setting that does both, because the two failures are controlled by the same knob pulling opposite
ways. The second repair is to go back to unrolling but warm-start it, so the inner solve is no longer
crude. That would lift the 85% ceiling, but it reintroduces exactly what I escaped: `O(K * dim(y))`
trajectory memory, a differentiate-through-a-solve structure, and the need to actually converge the inner
problem before differentiating. The value gap gets the warm-started-inner benefit — a persistent, cheaply
updated `y_hat` — *without* differentiating through it, because the Danskin lemma lets me use `y_hat` only
to evaluate `grad_x g`, never to backpropagate through the inner steps. So the value gap is the option that
keeps the penalty family's escape *and* fixes the fragility *and* avoids the unroll's memory, which none of
the cheaper repairs manage. That is why it is worth the switch.

Let me derive why the value gap removes exactly the two failures I just measured. The penalty is
`p = g(x, y) - v(x)` with `v(x) = min_y g(x, y)`, and the single structural difference from the gradient
norm is where the Hessian appears in the local relation. At a local solution of `f + gamma p`,
stationarity in `y` is `grad_y f + gamma grad_y g = 0`, because `v` carries no `y`-dependence at all. So
`||grad_y g|| <= L/gamma` *directly* — no `grad_yy g`, no singular-value condition, no division by a
curvature that can vanish. Under the lower level being `(1/mu)`-PL, the error bound `g - v >= (1/mu) d^2`
then gives `p <= mu L^2/gamma^2 = O(1/gamma^2)` with one PL chain, so the value gap is a `mu`-squared-
distance bound under *weaker* assumptions than the gradient norm needed (`1/sqrt(mu)`-PL to chain twice).
The contrast is the whole story of step 3's failure: the gradient-norm penalty's `||grad_yy g . grad_y
g|| <= L/(2 gamma)` only bounds `||grad_y g||` after dividing by the smallest singular value of
`grad_yy g`, which vanishes wherever the lower Hessian degenerates — and the linear hyper-cleaner's
cross-entropy has exactly such flat directions, which is why its f1 swung 6.85 points across seeds while
the better-conditioned MLP's did not. The value gap has no such division, so it does not get whipsawed by
lower-level curvature; I expect its cleaner f1 to be both *higher and far more stable* than G-PBGD's, and
in particular to fix the *linear* case that was the whole problem.

The value gap is faithful at the global level too, not just a convenient local fix. For any feasible
`(x, y)`, project `y` to `y_x in S(x)` with `d = ||y_x - y||`; `L`-Lipschitz `f` gives
`f(x, y) - f(x, y_x) >= -L d`, and adding `gamma p` with `p >= d^2/mu` floors the penalized objective at
`-L d + (gamma/mu) d^2 >= -L^2 mu/(4 gamma)`. So any bilevel global solution is an `eps_1`-global-min of
the penalized problem for `gamma = L^2 mu/(4 eps_1)`, penalty residual shrinking `O(1/gamma)` globally and
`O(1/gamma^2)` locally — the lever being that `p` dominates `d^2` and overpowers the linear Lipschitz
slack once `gamma` clears threshold. The `gamma = Theta(delta^{-1/2})` scaling is tight, not loose: on
`f = y, g = y^2` the penalized `y + gamma y^2` minimizes at `y = -1/(2 gamma)` with gap `1/(4 gamma^2)`,
so a gap below `delta` needs exactly `gamma >= 1/(2 sqrt(delta))`. A *finite* `gamma` suffices — and
because the value gap is the more faithful surrogate (one PL chain, not two), the `gamma` it needs is far
smaller than the gradient norm's, which is exactly why I can afford a gentle penalty here, and gentleness
is what protects the cleaner's recall from the over-discarding that `gamma = 37` caused.

The one subtlety the value gap introduces is computing `grad v(x)`, and the PL Danskin lemma dissolves it.
Naively `grad v` needs a lower solution `y*(x)` and looks like the implicit machinery I have been avoiding.
But `v(x) = g(x, y*(x))`, and the chain rule gives `grad v = grad_x g(x, y*) + (dy*/dx)^T grad_y g(x, y*)`;
at an unconstrained lower minimum `grad_y g(x, y*) = 0`, so the scary implicit term multiplies by zero and
`grad v(x) = grad_x g(x, y*)` for *any* `y* in S(x)` — no Hessian, no implicit-function theorem, no strong
convexity, only that `y*` be a stationary point of the inner problem. So I only need an *approximate* lower
minimizer, and I estimate it with a short inner loop, warm-started so the inner gap stays tied to outer
progress. The warm start matters for a reason the analysis pins down: inner gradient descent contracts the
value gap geometrically under PL, `g(x, omega_{t+1}) - v <= (1 - beta/(2 mu))(g(x, omega_t) - v)`, but that
bound carries the *initial* gap `g(x_k, omega_1) - v(x_k)`, and `x_k` drifts across outer iterations — a
cold start at an unrelated point could let the initial gap grow without bound, so no fixed inner-step count
would control the estimation error. Warm-starting `omega_1 = y_k` ties the initial inner gap to the outer
step size through PL (`g(x_k, y_k) - v <= (1/mu)||grad_y g||^2`, and `grad_y g` is controlled by the outer
move), so a logarithmically short inner loop keeps the gradient-estimation error `gamma^2 L_g^2
d^2_{S(x)}(y_hat)` summable and the outer projected descent telescopes to a `1/K` stationarity rate. This
is why a *single* warm-started inner step (`inner_itr = 1`) is enough here: because the inner network
persists and is never reset, each outer iteration only has to close the small incremental gap that the last
outer move opened — the exact opposite of RHG, which paid for 500 inner steps *from scratch* every outer
step precisely because it warm-started nothing. The value-gap direction is then `h = grad g(x, y) -
(grad_x g(x, y_hat), 0)` — the subtraction lives only in the `x`-coordinates, because `v` depends only on
`x`, so in `y` the value-gap gradient is just `grad_y g(x, y)`.

This is where this task's V-PBGD is concrete and where I must match the harness, not the generic method.
The scaffold's `run_v_pbgd` keeps a *persistent auxiliary* inner network `net_inner`, initialized from the
main model and updated across outer iterations by `inner_itr` SGD steps on the `sigmoid(x)`-weighted
training loss at frozen weights `sigx = sigmoid(x).detach()`. It then forms `fy = CE(val, clean)`,
`gxy = (sigmoid(x)*CE_train).mean()` through the *main* model, and `vx = (sigmoid(x)*CE_inner).mean()`
through `net_inner` with the inner outputs *detached* — so the gradient through `vx` flows only via
`sigmoid(x)`, giving exactly `grad_x g(x, y_hat)` and nothing through `y_hat`, which is the Danskin lemma
realized as a detach. The objective is `min(1/(gamma+eps),1) * (fy + gamma*(gxy - vx))`, backpropped to
step both `x_opt` and `y_opt`, with `gamma` ramped linearly from 0 to `gamma_max`. So the value-gap
subtraction `gxy - vx` is realized by the detach pattern, and the `min(1/gamma,1)` rescale is the same
stabilizer as G-PBGD. My edit is to point `algorithm` at `run_v_pbgd` and set the schedule. Note this rung
*does* run a real inner loop (`inner_itr` SGD steps on `net_inner`) — unlike G-PBGD which had none — but it
is a single warm-started step per outer iteration, not the 500-step-from-scratch unroll the RHG family paid
for; that is the distinction between approximating `grad v` cheaply and differentiating a full inner solve.
And like every prior rung, it ignores the exposed `inner_grad/outer_grad/inner_val` callables; the helper
builds the gradients directly from the two networks and `sigmoid(x)`.

The detach orientation is easy to get backwards, so I pin it on the lower manifold where `y = y_hat = y*`
and the penalty gradient must vanish. In `y` the value-gap gradient is `grad_y g(x, y*) = 0`
(inner-stationary). In `x` it is `grad_x[gxy - vx] = grad_x g(x, y) - grad_x g(x, y_hat)`, where `vx`
contributes only through `sigmoid(x)` because the inner outputs are detached — so it carries exactly
`grad_x g(x, y_hat)`, and with `y = y_hat` the two cancel. Both coordinates vanish, confirming the
subtraction is oriented right. Had I let the gradient flow through `y_hat` instead, the `x`-cancellation
would fail and the penalty would push even on the manifold — a wiring error the linear-f1 number would
expose.

The hyperparameters are the gentle schedule this task's reference uses, and the contrast with G-PBGD's
`gamma_max = 37` is itself diagnostic: linear `lrx=lry=0.1, lr_inner=0.01, inner_itr=1, gamma_max=0.2,
gamma_argmax_step=30000, outer_itr=40000`; MLP `lrx=0.1, lry=0.01, lr_inner=0.01, inner_itr=1,
gamma_max=0.1, gamma_argmax_step=10000, outer_itr=80000`. The tiny `gamma_max` — 0.1 to 0.2 against
G-PBGD's 37, roughly two orders of magnitude smaller — is possible *because* the value gap is the more
faithful surrogate: its `O(1/gamma^2)` local residual is reached at modest strength, so it does not have
to be slammed to a huge `gamma` to pin `y`, and a gentle penalty is exactly what keeps the cleaner's recall
from collapsing the way G-PBGD's did at `gamma = 37`. The longer MLP horizon (80k outer steps, double the
linear run) lets the gentle penalty converge fully on the stiffer network, and the smaller MLP `lry = 0.01`
keeps the main model's step conservative where the loss is least well-scaled. The ratio `lrx / lr_inner =
0.1 / 0.01 = 10` on the linear run encodes the telescoping condition directly: the outer variable `x`
moves ten times faster than the inner `lr_inner` scale, but the single warm-started inner step closes a
`(1 - lr_inner*mu)`-fraction of whatever gap the outer move opened, so as long as `lrx` is not so large
that one outer step opens a gap the one inner step cannot chase, the estimation error stays summable and
the outer descent telescopes — which is exactly the regime a modest `lrx = 0.1` against a persistent,
never-reset `net_inner` sits in.

Two consequences of the gentle `gamma` are worth stating because they are checkable. First, the
`min(1/(gamma+eps), 1)` rescale that did heavy lifting for G-PBGD — shrinking its step to `1/37 ≈ 0.027`
— is here essentially a *no-op*: with `gamma_max` at most `0.2`, `1/(gamma+eps) > 1` for the entire run,
so `min(...) = 1` always and the joint step is never rescaled. The stabilizer that a `gamma = 37`
landscape required is simply not needed when `gamma` never exceeds one; the value gap's faithfulness
removed the very stiffness the rescale existed to tame. Second, the cost per outer step is tiny: one warm-
started SGD step on `net_inner`, one forward for `fy / gxy / vx`, one backward — about three gradient-scale
evaluations, so the linear run spends roughly `3 * 40000 = 120000` and the MLP roughly `3 * 80000 =
240000` evaluations, the same order as the whole RHG run but purchasing `40000`–`80000` genuine coupled
outer steps instead of `100`. Against RHG's `~600` evaluations *per outer step*, V-PBGD spends `~3` — two
hundred times cheaper per step — which is exactly what makes the long persistent trajectory affordable and
is the arithmetic behind escaping the crude-solve ceiling without an inner unroll.

The toy, finally, returns to the well-behaved regime, and this is a clean falsifiable prediction against
step 3. `run_v_pbgd` in toy mode dispatches to `_toy_pbgd_step("v_pbgd")`, which uses `grad g` (the
value-gap gradient, since `S(x) = {-x}` makes `v(x) = 0` and `p = g`), *not* the gradient-norm penalty. So
V-PBGD descends the same smooth landscape the unrolling rungs did — no extra `grad_yy g` curvature factor
in the step — and I expect its toy line to snap back to the shared 260.7 mean convergence_steps at 0.030
residual and full success, recovering the ~43 steps and the residual that G-PBGD lost to its stiffer
gradient-norm toy objective. If the toy did not return to 260.7, the toy dispatch would not be using the
value gap.

So my falsifiable expectations. On the toy I expect V-PBGD to *recover* the unrolling family's 260.7 steps
and 0.030 residual exactly — it dispatches to the same `grad g` step, so this is a wiring prediction, not
an estimate — reversing G-PBGD's toy regression. On hyper-cleaning I expect it to keep the penalty
family's accuracy win (comparable to G-PBGD, well above the unrolling 84.6) while *fixing the linear f1
collapse*: linear f1 back up to at or above the unrolling family's ~89.5 and, crucially, *stable* across
seeds rather than swinging six points, with cleaner recall recovered to the unrolling family's high level,
because the value gap does not divide by the degenerate `grad_yy g` that crashed G-PBGD's linear recall.
That would make the value gap the best cleaner on the board, not just the least broken, and should close
G-PBGD's ten-point cross-model f1 gap, landing linear and MLP f1 close together instead of split. On the
hidden MLP I expect to give up a little against G-PBGD, precisely because the MLP was the case where the
gradient-norm penalty already *worked* (its MLP f1 rose, recall barely moved), so the value gap's
robustness buys little there and its gentler `gamma` trades a bit of raw MLP accuracy for the far more
faithful, lower-variance cleaner and the recovered toy convergence. That is the trade I am willing to
make: it should win the toy primary, the linear primary, and both f1 scores, losing only the single hidden
MLP accuracy by a little. If instead V-PBGD matched G-PBGD's broken linear f1, the detach pattern would
not be realizing the `gxy - vx` value gap and I would have wired the wrong subtraction; if its toy did not
snap back to 260.7, the toy dispatch would be on the wrong penalty. The full scaffold edit is in the
answer.
