The scaffold hands me vanilla gradient descent at `lr = 0.01`, and before I touch anything I want to be
honest about what the contest actually is, because it is not the contest the loss surface suggests. The
diagonal-net `w = u² − v²` is, as a function class, just a linear predictor — `w` can be anything in
`R^d`. So if all I cared about were driving the training loss to zero, the parameterization would be a
distraction and any reasonable step size would do. But the harness does not score training loss; it
scores the smallest sample size at which the *test* MSE drops below one, on a `k`-sparse ground truth
in a large ambient `d`. With `n < d` the training problem is underdetermined: there is an affine
subspace of `w` that fit the data exactly, and the test error is decided by *which* of those zero-loss
solutions I land on. That choice is not made by the loss — it is made by the optimizer's dynamics on
the non-convex `(u, v)` surface. This is the whole game: the diagonal parameterization plus
gradient-based dynamics has an *implicit bias*, and the bias I induce sets the sample complexity.

So the right question for the first rung is not "what is a fast optimizer" but "what kind of bias does
my update rule impose, and does it point toward sparsity?" The starting facts pin one end of this. The
initialization `alpha/sqrt(2d)` with `alpha = 1e-3` is deliberately tiny and sets `u = v`, so
`w_hat = u² − v² = 0` — I begin at the origin in predictor space, with essentially no signal in either
parameter vector. From the origin, gradient descent on this parameterization grows coordinates
*multiplicatively*: the gradient of the loss w.r.t. `u_i` carries a factor of `u_i` itself (chain rule
through the square), so a coordinate's growth rate is proportional to its current magnitude. Tiny
coordinates stay tiny until the data's correlation structure singles them out; the coordinates that
correspond to the true support, which the residual keeps pushing on, escape the flat region near zero
first, while the off-support coordinates linger. That escape-in-order behaviour is the engine of the
sparse bias — it is why a near-zero start matters and why the loss alone does not decide the answer.
Vanilla GD with a sensible step size already rides this. So my first rung should *probe the geometry*,
not just reproduce GD: the task hint is explicit that adaptive methods reshape the implicit bias, and I
do not yet know whether reshaping helps or hurts here. The most principled adaptive method to start
with is the one with a clean derivation and a provable affinity for sparse data — AdaGrad — so I will
make it the first thing I measure, and let the number tell me whether per-coordinate adaptation is a
friend or an enemy of the diagonal-net's sparsity.

Let me derive what AdaGrad actually does and why anyone would expect it to help on sparse problems,
because the expectation is exactly what I want to test. The canonical setting is online subgradient
descent: I want regret `R(T) = Σ_t f_t(x_t) − inf_x Σ_t f_t(x)` sublinear in `T`. Plain projected
gradient descent with `η_t = η/√t` gives a bound of order `D · sqrt(Σ_t ||g_t||²₂)`, on the order of
`√(dT)`, and that isotropic rate is minimax-tight — I cannot beat it by being cleverer with the *same*
single global step size. The looseness is elsewhere. Reading gradient descent as mirror descent —
linearize the loss and stay close in a Bregman divergence `B_ψ` of a strongly convex `ψ` — the regret
for any fixed `ψ` has the shape `(1/η) B_ψ(x*, x_1) + (η/2) Σ_t ||g_t||²_{ψ*}`, where the data term is a
sum of *dual* norms set entirely by `ψ`. Everyone fixes `ψ` in advance, blind to the gradients. On
sparse, heavy-tailed data the Euclidean `ψ` pays the full ambient dimension even though gradient mass
lives on a few coordinates. The leverage is to let `ψ` be chosen *from the data*. Restrict `ψ` to a
Mahalanobis form `½⟨x, H x⟩`; then `||g||²_{ψ*} = ⟨g, H⁻¹ g⟩`, and the data-dependent regret becomes
`Σ_t ⟨g_t, H_t⁻¹ g_t⟩`.

Solve the hindsight problem first: with all gradients in hand, which fixed diagonal `H = diag(s)`,
subject to a trace budget, minimizes `Σ_t Σ_i g_{t,i}²/s_i`? Lagrange gives `s_i ∝ sqrt(Σ_t g_{t,i}²)`
— the optimal per-coordinate denominator is the *accumulated ℓ2 norm of that coordinate's gradients*,
the square root of the running sum of squared gradients, and the optimal value is governed by
`Σ_i ||g_{1:T,i}||₂`, which on sparse data is tiny because rarely-firing coordinates contribute almost
nothing. The online version sets it incrementally — `s_{t,i} = sqrt(Σ_{τ≤t} g_{τ,i}²)` — and a doubling
argument shows the causal version costs only a factor of `√2`, yielding regret of order
`Σ_i ||g_{1:T,i}||₂`, exponentially smaller in `d` than `√(dT)` on sparse heavy-tailed features. The
update that drops out is `x_{t+1,i} = x_{t,i} − lr · g_{t,i} / (sqrt(Σ_{τ≤t} g_{τ,i}²) + eps)`: a single
per-coordinate accumulator, a square root, and a divide; the `eps` only floors the denominator so a
coordinate with no accumulated gradient mass is still invertible.

Now I have to translate that promise into *this* task, and here is where I have to be careful, because
the regret story was built for sparse-*feature* online learning and my problem is sparse-*solution*
full-batch recovery — they are not the same, and the diagonal-net non-convexity is precisely what the
regret analysis ignores. Three things change under the harness. First, there is no per-example
subgradient stream: the harness computes a **full-batch** gradient on the whole training set every
step, and the only stochasticity is the fresh Rademacher label noise. So AdaGrad's accumulator here is
not averaging over a sparse-feature stream; it is accumulating the squared full-batch gradient of each
of `u` and `v` — which, on the diagonal-net, carries the factor of `u_i` (resp. `v_i`). Second, my
parameters are `u` and `v`, not the predictor `w`, so the per-coordinate rescaling acts on the
*square-root coordinates*, and what it reshapes is the multiplicative escape dynamics that drive the
sparse bias — not an abstract feature geometry. Third, and most consequential: the per-coordinate
denominator does exactly what AdaGrad always does — it shrinks the step on coordinates that have
accumulated large gradient mass and enlarges it on coordinates that have stayed quiet. On the support
coordinates, which the residual hammers, the accumulated `Σ g²` grows fast, so AdaGrad *damps* the very
coordinates whose rapid multiplicative escape is the mechanism of recovery; meanwhile the off-support
coordinates, which see only the label-noise-driven gradient, keep a small denominator and so get a
*relatively larger* effective step. That is the opposite of what the sparse bias wants. The harness
also exposes only the three functions and gives me `delta` but no `sigma`/`noise_scale` argument, so I
cannot adapt the floor to the noise level analytically — `eps` is a fixed `1e-6` floor, well below any
healthy gradient RMS, and that is all the noise-handling the interface affords.

I will keep the configuration to the literal canonical AdaGrad — `lr = 0.01`, `eps = 1e-6`, the two
accumulators `state_sum_u`, `state_sum_v` initialised to zero, and the divide-by-`sqrt(accum)+eps`
update applied identically to `u` and `v`. I deliberately do not tune `lr` per setting or add momentum;
the point of the first rung is a clean read on what *coordinate-wise adaptivity alone* does to the
sparse bias, against the vanilla-GD default, with everything else held at the textbook value. The
running-sum-of-squares denominator is exactly `diag(G_t)`'s square root from the regret derivation; the
state is two `float64` vectors of length `d`; the step is three elementwise ops per parameter vector.
The full scaffold module is in the answer.

Now the falsifiable expectation, since there is no prior rung to reflect on. If the sparse-recovery
intuition from the regret bound carried over, AdaGrad's per-coordinate rate would let rare (off-support
should-be-zero) coordinates and frequent (support) coordinates move at appropriately different rates,
and I would recover from *fewer* samples than vanilla GD. But the mechanism analysis above predicts the
reverse: by damping the support coordinates' multiplicative escape and inflating the off-support steps,
AdaGrad should *blur* the saddle-to-saddle ordering that makes the diagonal-net sparse, and recovery
should need *more* samples. The decaying effective rate (the denominator only grows) compounds this —
once a support coordinate's accumulator is large, it crawls, and on the high-dimensional settings the
crawl can stall before the plateau rule fires. Concretely I expect AdaGrad to be the *weakest* rung:
the smallest `n*` it can recover from should be markedly larger than what a plain step achieves, and
the gap should *widen* with the ambient dimension — worst on `d10000_k50`, where the dilution of the
sparse bias and the rate decay both bite hardest. If instead AdaGrad recovers at small `n` across the
board, my mechanism story is wrong and adaptivity is helping the geometry; the next rung will be chosen
to confirm or kill exactly that. Either way, this rung's job is to convert the hint's "this can help or
hurt" into a measured number, and my bet — to be tested — is that on the diagonal-net, coordinate-wise
adaptivity hurts the sparse implicit bias, and the very next thing to try is to *strip the adaptivity
out* and let plain gradient descent ride the multiplicative escape unimpeded.
