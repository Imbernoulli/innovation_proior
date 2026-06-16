I am standing at the editable region of `custom_strategy.py` with one function to fill,
`algorithm(state, hparams, grad_fns)`, and two instances to satisfy at once: the toy convergence
problem and the MNIST data hyper-cleaning problem. Before I reach for anything clever I want the
cheapest update that is unambiguously *correct* as a bilevel solver, so I can read its measured
behavior and know what the harder rungs have to beat. The most conservative thing I can write down is
the method that makes the fewest assumptions about the inner objective: differentiate the actual inner
training run. That is the reverse-mode hypergradient, and its truncated form is the natural starting
rung because it is the same algorithm at a smaller, cheaper memory footprint.

Let me be precise about what the hypergradient even is here, because the whole rung is an exercise in
computing one derivative without forming a matrix I cannot afford. In hyper-cleaning the upper variable
`x` is a vector of per-example logits, one per training point, and the lower variable `y` is the
classifier's parameters. The inner problem trains `y` to minimize the `sigmoid(x)`-weighted
cross-entropy on the corrupted training set; the outer problem wants the clean validation
cross-entropy small at whatever `y` the inner optimizer produces. So the object I actually want to
descend is `F(x) = f(x, y_T(x))`, where `y_T(x)` is not a formula but the endpoint of `T` steps of
inner gradient descent. Its total derivative is `d_x F = grad_x f + (d y_T/d x)^T grad_y f`. The two
partials are one backward pass each; the entire difficulty is the middle factor `d y_T/d x`, the
sensitivity of the inner solution to the per-example weights, because `y_T` is the tail of a long chain
of inner updates, each of which depended on `x`.

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
parameter-by-parameter matrices — for the MLP `y` they are hundreds of thousands of dimensions square —
so I can never materialize them, let alone multiply chains of them. The trick is that I never need the
matrices, only their action on a vector, and the order in which I contract the product decides the
price. If I push the *row* `grad_y f` leftward through the chain of `A_t`, I only ever carry a single
adjoint vector of the size of `y`, and each `alpha . A_{t}` and `alpha . B_t` is one
transposed-Jacobian-vector product that autograd gives me for the price of one inner step. This is the
adjoint recursion: set `alpha_T = grad_y f`, sweep `t` down from `T`, accumulate `h += alpha_t . B_t`
and propagate `alpha_{t-1} = alpha_t . A_t`. It is exactly back-propagation through time over the
optimizer's own steps, costing `O(T)` time with no penalty in the number of hyperparameters — which
matters here because `x` has one coordinate per training example, thousands of them. The complementary
forward contraction would carry a sensitivity matrix of size `dim(y) x dim(x)` and run `dim(x)` times
slower; with `x` this wide that is hopeless, so reverse mode is the only sane direction.

But the reverse contraction has one cost I have to confront, and it is what makes the *truncated*
variant the rung I actually fill rather than full unrolling. To evaluate `A_t` and `B_t` on the way
back I need `y_{t-1}`, the state I was at when I took step `t`. The backward sweep visits states in
reverse, so I would have to hold the entire trajectory `y_0, ..., y_T` in memory: `O(T * dim(y))`. For
the MLP over a long horizon that product is the wall everyone hits. The way out that keeps the
estimator exact-in-spirit and optimizer-agnostic — no fragile reversal of the dynamics, no
information-buffer bit tricks — is to notice where the gradient signal actually lives. When the last
inner iterates sit in a region where `g(., x)` is well-conditioned (locally strongly convex with
smoothness, `lr_inner` below the smoothness reciprocal), each `A_t` is a contraction, `||A_t|| <= 1 -
lr_inner * alpha < 1`. The term at depth `T - t` from the end carries the factor `A_{t+1} ... A_T`,
whose size is bounded by `(1 - lr_inner * alpha)^{T - t}` — it decays geometrically the further back I
go. So the deep terms contribute almost nothing. I run the inner optimizer forward for the full `T`
steps so the final weights are properly converged, but in the backward pass I only walk back `K`
transitions and stop. The truncated estimate `h_{T-K} = grad_x f + sum_{t=T-K+1}^{T} B_t A_{t+1} ... A_T
grad_y f` has bias bounded by something on the order of `(1 - lr_inner * alpha)^K` — geometrically
small in `K` — at memory `O(K * dim(y))` instead of `O(T * dim(y))`. Full reverse-mode unrolling is the
`K = T` special case; truncation is the same algorithm at a fraction of the memory, which is exactly the
property that makes it a sound, cheap first rung.

There is a satisfying consistency check that tells me this truncation is not a crude hack. In the limit
where the inner iterates have converged, the per-step Jacobians settle to constants `A_inf = I -
lr_inner * grad_yy g(y*)` and `B_inf = -lr_inner * grad_xy g(y*)`, and the full backward accumulation
becomes `grad_y f . (sum_k A_inf^k) . B_inf`. But `sum_k A_inf^k` is the Neumann series of `(I -
A_inf)^{-1} = (lr_inner * grad_yy g)^{-1}`, so the infinite backward sum equals the implicit-function
hypergradient `-grad_y f . (grad_yy g)^{-1} . grad_xy g`. The `K`-step truncation is precisely keeping
the first `K` terms of that Neumann series — `K` is the order of the inverse-Hessian approximation. So
this rung is, quietly, an implicit-differentiation method computed by summing instead of solving, and
truncating it trades a controllable bias for memory. That is the cleanest possible "do the honest thing
cheaply" baseline.

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
callables are there for the penalty methods; the unrolling rung ignores them and calls the helper.

A second harness fact pins down what this rung does in *toy* mode, and it is a place where the
task's implementation departs from the generic reverse-mode story. In toy mode `run_rhg_family`
dispatches to the very same `_toy_pbgd_step` as the value-gap method — it takes one projected penalized
first-order step on `f + gamma * grad g`, not a differentiated inner unroll. The reason is structural:
the toy has `S(x) = {-x}`, so `v(x) = 0` and there is nothing to unroll; the "inner solve" is trivial,
and the harness collapses every method to the same projected step there. So my `TOY_HPARAMS` for this
rung are the same `gams=(10.0,), alpha0=0.1` as every other rung, and I should *expect identical toy
numbers* to the value-gap and full-unroll rungs. The only place the truncated method can distinguish
itself is hyper-cleaning, and only through `K`.

That leaves the choice of `K`, which is the whole identity of this rung versus full unrolling. I take
`T = 500` inner steps, `K = 100` (truncate to the last fifth of the trajectory), `lr = 0.001` outer,
`lr_inner = 0.1` linear and `0.4` MLP, with `outer_itr = 100` and `eval_interval = 1`. The contraction
argument says the bias from dropping the first 400 transitions is bounded by `(1 - lr_inner*alpha)^{100}`
— negligible if the inner problem is even mildly well-conditioned — while the memory drops by `5x`
relative to the full unroll. The non-interference structure of hyper-cleaning sharpens this: the
validation loss does not depend on `x` except through `y`, so `grad_x f = 0` and the truncated adjoint
is, for the right problem class, essentially as good a descent direction as the full one even at small
`K`. So I expect `K = 100` to cost me very little accuracy against `K = 500`.

What do I expect to *see*, falsifiably, since this is the first rung and there are no prior numbers to
diagnose? On the toy I expect the shared projected step: roughly 260 mean `convergence_steps` at full
`success_rate` and a residual near 0.03 — and I will treat any deviation as a bug in how I wired the
helper, not a property of truncation. On hyper-cleaning I expect the unrolling family to land
*meaningfully below* the penalty methods, because differentiating only a 500-step inner run with a
fixed `lr_inner` does not solve the inner problem nearly as well as 40k–80k coupled penalty steps do —
I expect linear and MLP test accuracy in the mid-80s rather than the 90s, with a high cleaner recall
(the unroll over-keeps examples) but only middling precision. And I expect `K = 100` to track full
unrolling closely: if the truncated rung were far below the full one, the contraction assumption would
be failing and the geometry would be telling me the inner problem is not well-conditioned. This rung's
job is to set that floor; the next rung removes the truncation and asks whether the missing 400
transitions were worth their memory. The full scaffold edit is in the answer.
