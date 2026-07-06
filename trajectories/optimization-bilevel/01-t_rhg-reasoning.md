I am standing at the editable region of `custom_strategy.py` with one function to fill,
`algorithm(state, hparams, grad_fns)`, and two instances to satisfy at once: the toy convergence
problem and the MNIST data hyper-cleaning problem. Before I reach for anything clever I want the
cheapest update that is unambiguously *correct* as a bilevel solver, so I can read its measured
behavior and know what the harder rungs have to beat. The most conservative thing I can write down is
the method that makes the fewest assumptions about the inner objective: differentiate the actual inner
training run rather than assume it is strongly convex or invert any Hessian. That is the reverse-mode
hypergradient, and its truncated form is the natural starting rung because it is the same algorithm at
a smaller, cheaper memory footprint.

Let me be precise about what the hypergradient even is here, because the whole rung is an exercise in
computing one derivative without forming a matrix I cannot afford. In hyper-cleaning the upper variable
`x` is a vector of per-example logits, one per training point, and the lower variable `y` is the
classifier's parameters. Concretely: the split hands me 5000 training examples, so `x` lives in
`R^5000`, one weight `sigmoid(x_i) in (0, 1)` per corrupted point. The inner variable `y` is either the
linear map `784 -> 10`, which is `784*10 + 10 = 7850` parameters, or the MLP `784 -> 300 -> 10`, which
is `784*300 + 300 + 300*10 + 10 = 238510` parameters. Keep those three numbers in view — `dim(x)=5000`,
`dim(y)=7850` or `238510` — because every affordability decision in this rung is decided by their ratio.
The inner problem trains `y` to minimize the `sigmoid(x)`-weighted cross-entropy on the corrupted
training set; the outer problem wants the clean validation cross-entropy small at whatever `y` the inner
optimizer produces. So the object I actually want to descend is `F(x) = f(x, y_T(x))`, where `y_T(x)`
is not a formula but the endpoint of `T` steps of inner gradient descent. Its total derivative is
`d_x F = grad_x f + (d y_T/d x)^T grad_y f`. The two partials are one backward pass each; the entire
difficulty is the middle factor `d y_T/d x`, the sensitivity of the inner solution to the per-example
weights, because `y_T` is the tail of a long chain of inner updates, each of which depended on `x`.

There is one structural simplification I should bank before I do any work, because it changes what the
hypergradient even contains. The outer objective `f` is the clean-validation cross-entropy evaluated at
`y`; it does not see the per-example weights `x` at all except through the trained classifier. So
`grad_x f = 0` identically, and the whole hypergradient collapses to `d_x F = (d y_T/d x)^T grad_y f`.
Every bit of descent signal on the weights flows through the sensitivity of the inner solve — there is
no direct channel. That is worth remembering: if I get `d y_T/d x` even roughly right, I get the entire
gradient roughly right, and if I get it wrong there is nothing else to fall back on.

So I make the chain explicit, because that is the only way to see which arithmetic order is affordable.
The inner optimizer is a dynamical system: `y_{t+1} = Phi_{t+1}(y_t, x)`, with `y_T` the final state.
For the gradient-descent inner step the map is `Phi_{t+1}(y_t, x) = y_t - lr_inner * grad_y g(y_t, x)`.
Then `y_T` is a composition of `T` such maps, and differentiating the recursion totally with respect to
`x` and collecting terms gives the exact hypergradient as a sum over the whole trajectory,
`d_x F = grad_x f + sum_t B_t A_{t+1} A_{t+2} ... A_T grad_y f`, where `A_t = d Phi_t/d y_{t-1} =
I - lr_inner * grad_yy g(y_{t-1}, x)` is how step `t` reacts to a perturbation of the *state*, and
`B_t = d Phi_t/d x = -lr_inner * grad_xy g(y_{t-1}, x)` is how it reacts to a perturbation of `x`
directly. Each term in that sum is the influence of the per-example weights *at one moment of the inner
run* on the final validation loss, propagated forward to the end through every subsequent state
Jacobian.

Now the cost question, which is the entire reason there is a fork in the road. Those `A_t` are
parameter-by-parameter matrices — for the MLP `y` they are `238510 x 238510`, tens of billions of
entries, unstorable — so I can never materialize them, let alone multiply chains of them. The trick is
that I never need the matrices, only their action on a vector, and the order in which I contract the
product decides the price. There are exactly two orders. If I push the *row* `grad_y f` leftward through
the chain of `A_t`, I only ever carry a single adjoint vector of the size of `y`, and each `alpha . A_t`
and `alpha . B_t` is one transposed-Jacobian-vector product that autograd gives me for the price of one
inner step. This is the adjoint recursion: set `alpha_T = grad_y f`, sweep `t` down from `T`, accumulate
`h += alpha_t . B_t` and propagate `alpha_{t-1} = alpha_t . A_t`. It is exactly back-propagation through
time over the optimizer's own steps, costing `O(T)` time with no penalty in the number of
hyperparameters. The complementary *forward* contraction would carry a sensitivity matrix of size
`dim(y) x dim(x)` from the start of the trajectory forward. Put the numbers on it: for the MLP that
matrix is `238510 x 5000 ≈ 1.19e9` entries, about `4.8 GB` at single precision, and I would have to
propagate it through every inner step and effectively run the inner recursion `dim(x) = 5000` times to
fill its columns. Reverse mode replaces all of that with one vector of length `dim(y)` and one backward
sweep. With `x` this wide, reverse mode is not merely preferable, it is the only direction that fits in
memory at all.

Before I trust the recursion I should hand-trace it on the smallest case that has any depth, `T = 2`,
because a sign or an ordering error here silently corrupts every hypergradient. With two steps
`y_2 = Phi_2(Phi_1(y_0, x), x)`, direct differentiation gives `d y_2/dx = B_2 + A_2 B_1` (the direct
dependence of step 2, plus the direct dependence of step 1 pushed forward through step 2's state
Jacobian). Now run the adjoint by hand: initialize `alpha_2 = grad_y f`, accumulate `h = alpha_2 . B_2`,
propagate `alpha_1 = alpha_2 . A_2`, then accumulate `h += alpha_1 . B_1 = alpha_2 . A_2 . B_1`. The
total is `h = grad_y f . (B_2 + A_2 B_1) = grad_y f . (d y_2/dx)`, exactly the middle factor contracted
against `grad_y f`. So the sweep reproduces the total derivative with the state Jacobians appearing in
the correct order (`A_2` multiplies the *earlier* step's `B_1`, never the later one), which is the
ordering the reverse pass has to get right and the one place a naive forward-then-transpose would slip.
That trace is my assurance that when I hand the helper `K = 100`, the hundred TJVPs it does are summing
the last hundred of these terms in the right sequence.

But the reverse contraction has one cost I have to confront, and it is what makes the *truncated*
variant the rung I actually fill rather than full unrolling. To evaluate `A_t` and `B_t` on the way
back I need `y_{t-1}`, the state I was at when I took step `t`. The backward sweep visits states in
reverse, so I would have to hold the entire trajectory `y_0, ..., y_T` in memory: `O(T * dim(y))`. Make
that concrete too. For the MLP over `T = 500` steps that is `500 * 238510 ≈ 1.19e8` floats, about
`477 MB` just for the checkpoints of one inner run — and I redo that inner run every outer iteration.
For the linear model it is a comfortable `500 * 7850 ≈ 3.9e6` floats, `16 MB`, so the wall is entirely
an MLP problem, which is exactly the hidden, hardest target. The way out that keeps the estimator
exact-in-spirit and optimizer-agnostic — no fragile reversal of the dynamics, no information-buffer bit
tricks — is to notice where the gradient signal actually lives. When the last inner iterates sit in a
region where `g(., x)` is well-conditioned (locally strongly convex with smoothness, `lr_inner` below
the smoothness reciprocal), each `A_t` is a contraction, `||A_t|| <= 1 - lr_inner * mu < 1`, with `mu`
the local curvature floor. The term at depth `T - t` from the end carries the factor `A_{t+1} ... A_T`,
whose size is bounded by `(1 - lr_inner * mu)^{T - t}` — it decays geometrically the further back I
go. So the deep terms contribute almost nothing. I run the inner optimizer forward for the full `T`
steps so the final weights are properly converged, but in the backward pass I only walk back `K`
transitions and stop. The truncated estimate `h_{T-K} = grad_x f + sum_{t=T-K+1}^{T} B_t A_{t+1} ... A_T
grad_y f` keeps memory at `O(K * dim(y))` — for the MLP at `K = 100`, `100 * 238510 ≈ 2.4e7` floats,
about `95 MB`, a clean `5x` reduction off the `477 MB` wall.

I want to be honest about how large the truncation bias actually is, because it is the whole gamble of
this rung and I can bound it on paper. The dropped terms are the ones at depth `d = T - t >= K`, and
each is bounded by `r^d` with `r = 1 - lr_inner * mu`. Their geometric tail sums to `r^K / (1 - r)`,
while the full sum is bounded by `1 / (1 - r)`; the ratio is exactly `r^K`. So the *relative* truncation
error is `(1 - lr_inner * mu)^K`, no worse. For that to be under a percent I need `(1 - lr_inner*mu)^K <
0.01`, i.e. `K * lr_inner * mu > ln(100) ≈ 4.6`, i.e. with `K = 100` a curvature-step product
`lr_inner * mu > 0.046`. Read against my step sizes: the linear model at `lr_inner = 0.1` needs the
inner PL constant `mu > 0.46`, and the MLP at `lr_inner = 0.4` needs `mu > 0.115`. Those are not
guaranteed — a badly conditioned weighted cross-entropy could have a much smaller `mu` — so I cannot
*prove* from here that `K = 100` is deep enough. What I can say is that if the contraction is even mildly
favorable the bias is negligible, and that the clean way to find out is the very next rung, which removes
the truncation and compares. For now I treat `K = 100` as a bet that the inner problem is well enough
conditioned, and I flag the hyper-cleaning accuracy gap between this rung and the full unroll as the
falsifiable test of that bet.

There is a satisfying consistency check that tells me this truncation is not a crude hack but a
principled approximation with a name. In the limit where the inner iterates have converged, the per-step
Jacobians settle to constants `A_inf = I - lr_inner * grad_yy g(y*)` and `B_inf = -lr_inner *
grad_xy g(y*)`, and the full backward accumulation becomes `grad_y f . (sum_{k=0}^{inf} A_inf^k) .
B_inf`. Now `sum_{k=0}^{inf} A_inf^k` is the Neumann series of `(I - A_inf)^{-1}`, and `I - A_inf =
lr_inner * grad_yy g`, so the infinite backward sum equals `grad_y f . (lr_inner * grad_yy g)^{-1} .
(-lr_inner * grad_xy g)` — the two `lr_inner` factors cancel exactly, leaving `-grad_y f .
(grad_yy g)^{-1} . grad_xy g`, the implicit-function hypergradient with no step-size dependence at all.
That cancellation is a real check that I derived the maps right: if `lr_inner` had survived in the
converged limit, one of `A_t` or `B_t` would carry the wrong power of the step size. And the truncation
lands cleanly on this picture through the finite identity `(I - A)(I + A + ... + A^{K-1}) = I - A^K`: the
`K`-step backward sum is the `K`-term partial Neumann series, so its relative error against the true
inverse is `||A_inf^K|| ≈ (1 - lr_inner*mu)^K` — the same bound I got by counting dropped terms, arrived
at a second way. So `K` is literally the order of the inverse-Hessian approximation, and this rung is,
quietly, implicit differentiation computed by summing instead of solving. That is the cleanest possible
"do the honest thing cheaply" baseline.

A one-line scalar instance makes the order-of-approximation claim tangible. Suppose in the converged
region `grad_yy g` acts like a scalar curvature `a` and `lr_inner * a = 0.05`, so `A_inf = 0.95`. The
true inverse factor is `sum_{k>=0} 0.95^k = 1/0.05 = 20`. The `K = 100` partial sum is
`(1 - 0.95^{100})/0.05`, and `0.95^{100} ≈ e^{-100*0.0513} ≈ e^{-5.13} ≈ 0.0059`, so the partial sum is
`20 * (1 - 0.0059) ≈ 19.88` — a `0.6%` shortfall, matching `r^K` exactly. Push the curvature down to
`lr_inner * a = 0.01`, `A_inf = 0.99`: now the true factor is `100`, but `0.99^{100} ≈ e^{-1} ≈ 0.37`,
so the `K = 100` sum recovers only `63` of it — a `37%` error. That is the honest picture in two numbers:
truncation at `K = 100` is nearly free when the inner curvature-step product is a few hundredths, and
badly biased when it is a hundredth, which is precisely the `lr_inner * mu ≳ 0.046` threshold restated.
I cannot read `mu` off the problem from here, so which regime the weighted cross-entropy lands in is a
measurement, not a derivation.

Now I have to map this onto *this task's* harness, and the mapping decides everything about what the
code can and cannot do. The scaffold already implements the entire reverse-mode machinery for me:
`run_rhg_family(state, hparams, grad_fns)` dispatches on `state["task"]`, and in hyper-cleaning mode it
runs the forward inner loop of `T` steps keeping the last `K + 1` states, calls the fixed
`hg.reverse(...)` adjoint sweep over `[fp_map] * K` to set `x.grad`, and steps the upper optimizer.
The truncation knob is literally the hyperparameter `K`. So filling this rung is not re-deriving
back-propagation; it is choosing `K`, `T`, `lr`, `lr_inner` and pointing `algorithm` at the helper. The
crucial harness fact I must respect: although `grad_fns` also exposes `outer_grad`, `inner_grad`, and
`inner_val`, the reverse-mode family does *not* consume them — it builds its own differentiable inner
map with `create_graph=True` inside the helper, because the adjoint needs the inner step to be a graph
node, which the pre-detached `inner_grad` callable is not designed to give. The exposed first-order
callables are there for the penalty reformulations that the background lists; the unrolling rung ignores
them and calls the helper.

A second harness fact pins down what this rung does in *toy* mode, and it is a place where the
task's implementation departs from the generic reverse-mode story. In toy mode `run_rhg_family`
dispatches to the very same `_toy_pbgd_step` as the value-gap method — it takes one projected penalized
first-order step on `f + gamma * grad g`, not a differentiated inner unroll. The reason is structural,
and I can see it directly from the problem definition: the toy has `S(x) = {-x}`, and I can check that
`-x` really is the inner minimizer. Writing `u = x + y`, the inner objective is `g = u^2 + x sin^2 u`,
so `grad_y g = 2u + x sin(2u)`, which vanishes at `u = 0`, i.e. `y = -x`, and `g(x, -x) = 0` is the
floor of a nonnegative-near-`u=0` bowl for `x in [0, 3]`. So `v(x) = min_y g = 0` on the whole domain,
the value gap `g - v = g` has nothing to unroll, and the harness collapses every method to the same
projected step there. My `TOY_HPARAMS` for this rung are therefore the shared `gams=(10.0,),
alpha0=0.1`, and I should *expect identical toy numbers* to the other value-gap-style rungs — whatever
the shared projected step produces, this rung inherits it exactly. The step itself is transparent: it
descends `f + 10 * g` with a projected step of size `0.1` onto `[0, 3]`, so it simultaneously pulls `y`
toward `-x` (through the `g` term) and pushes `x` down the reduced outer landscape `f(x, -x)`, and the
convergence count is however many such steps it takes to bring the stationarity residual under the fixed
tolerance from 1000 random inits. The only place the truncated method can distinguish itself is
hyper-cleaning, and only through `K`.

That leaves the choice of `K`, which is the whole identity of this rung versus full unrolling. I take
`T = 500` inner steps, `K = 100` (truncate to the last fifth of the trajectory), `lr = 0.001` outer,
`lr_inner = 0.1` linear and `0.4` MLP, with `outer_itr = 100` and `eval_interval = 1`. The contraction
argument says the bias from dropping the first 400 transitions is bounded by `(1 - lr_inner*mu)^{100}`
— negligible if the inner problem is even mildly well-conditioned, per the `lr_inner*mu > 0.046`
threshold above — while the memory drops by `5x` relative to the full unroll. The non-interference
structure of hyper-cleaning sharpens this: since `grad_x f = 0`, the truncated adjoint is the *entire*
hypergradient estimate, and for the right problem class it is essentially as good a descent direction as
the full one even at small `K`. So I expect `K = 100` to cost me very little accuracy against `K = 500`
— but I have flagged that this is a bet on the conditioning, not a proof.

The rest of the budget follows from counting gradient evaluations, and the count is itself a diagnosis
of this family's ceiling. Each outer step runs `T = 500` forward inner gradient steps plus `K = 100`
backward transposed-Jacobian-vector products, roughly `600` gradient-scale evaluations, and I take
`outer_itr = 100` of them, so the whole run spends on the order of `60000` inner-gradient evaluations —
and critically, the inner classifier is trained from a fresh random init on every one of those 100 outer
steps, never warm-started. That is a *lot* of compute buying a *shallow* inner solve: 500 plain
gradient-descent steps at a fixed `lr_inner`, repeated from scratch, is nowhere near the convergence a
persistent inner optimizer run for tens of thousands of coupled steps would reach. The two inner step
sizes reflect the two landscapes: the linear weighted cross-entropy is convex and well-scaled, so
`lr_inner = 0.1` is a safe step below its smoothness reciprocal, whereas the MLP with its sigmoid hidden
layer has a flatter, more poorly-scaled loss where a timid step barely moves in 500 iterations, so I push
`lr_inner = 0.4` to give the inner run a chance to actually descend within the horizon. Even so, 500
steps from a random init on the MLP is optimistic, and I read the `outer_itr = 100` cap as the honest
admission that each outer step is expensive enough that I cannot afford many — which means the outer
descent itself is only lightly converged too. Both facts point the same way: this rung's number will be
capped by how badly the inner (and outer) problems are solved, not by any imperfection in the
differentiation. The `reg = 0.0` in both dicts is deliberate too: no ridge on the per-example logits `x`,
so nothing stops the hypergradient from pushing a weight's `x_i` toward the sigmoid rails (`sigmoid(x_i)`
toward 0 or 1) — the cleaner is free to fully drop or fully keep an example, which is what I want a
data-cleaner to do, and it means any softness in the recovered split comes from the crude solve, not from
a regularizer holding the weights back.

What do I expect to *see*, falsifiably, since this is the first rung and there are no prior numbers to
diagnose? On the toy I expect the shared projected step to converge cleanly from all 1000 inits: full
`success_rate`, on the order of a couple hundred steps to hit the tolerance, and a small residual at
the few-percent level — and I will treat any deviation as a bug in how I wired the helper, not a
property of truncation, since the toy cannot separate unrolling depth at all. On hyper-cleaning I expect
this unrolling step to land only in the mid-80s of test accuracy, because differentiating a single
fixed-`lr_inner` 500-step inner run does not actually solve the inner problem well — the ceiling here is
the crudeness of that inner solve, not the quality of the differentiation, and I would not be surprised
if a reformulation that drove lower-level optimality harder later cleared this floor by several points. I also expect
a lopsided cleaner profile: high recall because a lightly-solved inner classifier over-keeps examples
rather than aggressively discarding them, against only middling precision. And I expect `K = 100` to
track full unrolling closely; if the truncated rung later turns out to sit far below the full one, the
contraction assumption is failing, `lr_inner*mu` is below my threshold, and the geometry is telling me
the inner problem is not well-conditioned. This rung's job is to set that floor; the next rung removes
the truncation and asks whether the missing 400 transitions were worth their memory. The full scaffold
edit is in the answer.
