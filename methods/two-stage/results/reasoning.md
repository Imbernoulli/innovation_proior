Let me start from what actually hurts. I have Gaussian inputs `z ~ N(0, I_d)` in high
dimension, and a target that only cares about a few directions: `y = g*(<w1*,z>, ...,
<wr*,z>)`, with the `wk*` orthonormal, spanning some `r`-dimensional subspace `V*` with
`r` tiny next to `d`. The inputs are isotropic — there is no structure in `z` at all; all
the structure is in which directions the label reads off. A kernel method is hopeless here
in the way that matters: it fixes its features up front, so it never learns *which*
directions are `V*`, and it pays the full ambient dimension. A two-layer net could in
principle do better, because its first-layer rows `w_i` can rotate to point into `V*`, and
once they do the second layer only has to fit an `r`-dimensional function. So the entire
game is: get the first-layer rows to align with `V*`, fast, with few samples. Let me make
"align" concrete — for a neuron `i` with row `w_i`, the quantity I care about is how much
of it lies in `V*`: `||Pi* w_i|| / ||w_i||`, where `Pi*` projects onto `V*`. At a random
start that ratio is tiny; I want it `Theta(1)`.

How tiny is it at the start? Draw `w_i^0` uniformly on the sphere `S^{d-1}`. Its projection
onto a fixed `r`-dimensional subspace has squared length about `r/d`, so
`||Pi* w_i^0|| ~ sqrt(r/d)`, i.e. `O(1/sqrt(d))` up to logs. That number is going to haunt
everything. A random neuron is almost entirely orthogonal to the subspace I want it to find.

The naive thing is to just run SGD and wait. But I know how that goes for a single-index
target with information exponent `ell` (the smallest Hermite degree appearing in the
target): one-pass SGD needs about `d^{ell-1}` samples and steps, because at the random
start the gradient's correlation with the unknown direction is `O(1/sqrt(d))^{something}`,
the dynamics sit near a saddle, and crawling away from it is slow and sequential. The rate
blows up with `ell`, and the joint motion of `W` and `a` is tangled. I want something I can
actually *analyze*, and that is faster.

Two instincts. First: if the trouble is that `W` and `a` move together, freeze one of them.
Second: if the trouble is that each gradient step barely correlates with `V*`, take a *big*
step — but I need to know how big, and that requires understanding the gradient.

Let me freeze the second layer `a` and look at the gradient on `W` alone. Loss is squared
error, `(1/2) (y - f_hat(z))^2`. The gradient of that with respect to a row `w_i`, with the
`1/sqrt(p)` prefactor in the network, is

  `g_i = (a_i / sqrt(p)) * (1/n) sum_nu z^nu sigma'(<w_i, z^nu>) (f*(z^nu) - f_hat(z^nu))`.

There's a nuisance: the `f_hat` term, the network's own output. If I could make the network
output exactly zero at initialization, the residual would just be `f*` and the gradient
would be a clean correlation between the activation derivative and the labels. I can arrange
that: pair the neurons, `w_i^0 = w_{p-i+1}^0` and `a_i^0 = -a_{p-i+1}^0`, so the two members
of each pair cancel and `f_hat(.; W^0, a^0) = 0`. Then at the first step

  `g_i = (a_i / sqrt(p)) * (1/n) sum_nu z^nu sigma'(<w_i^0, z^nu>) f*(z^nu)`.

Now I want its expectation, to see what direction the step pushes in. The term
`E[z sigma'(<w,z>) f*(z)]` — there's a standard way to handle `z` times a function under a
Gaussian: integration by parts, i.e. Stein's lemma, `E[z h(z)] = E[grad_z h(z)]`. With
`h(z) = sigma'(<w,z>) f*(z)`,

  `E[z sigma'(<w,z>) f*(z)] = w E[sigma''(<w,z>) f*(z)] + E[sigma'(<w,z>) grad_z f*(z)]`.

So the expected gradient has a piece along `w` itself (a rescaling, not alignment) and a
piece `E[sigma'(<w,z>) grad_z f*(z)]` that can point into new directions — that second piece
is the one that can rotate `w` toward `V*`. Let me expand both in Hermite tensors. Write
`sigma`'s Hermite coefficients as `(c_k)`. The `k`-th Hermite coefficient of
`z -> sigma''(<w,z>)` is `c_{k+2} w^{⊗k}`, and of `z -> sigma'(<w,z>)` is `c_{k+1} w^{⊗k}`.
Pairing against the target's Hermite tensors `C_k* = C_k(f*)` via `<f,g>_gamma = sum_k
<C_k(f), C_k(g)>` gives

  `E[z sigma'(<w,z>) f*(z)] = sum_k c_{k+2} <w^{⊗k}, C_k*> w + sum_k c_{k+1} C_{k+1}* x_{1..k} w^{⊗k}`,

where `x_{1..k}` contracts the first `k` modes of the `(k+1)`-tensor against `w^{⊗k}`,
leaving a vector. The first sum is along `w` again; the second sum is the part that can move
into `V*`.

Now plug in `w = w_i^0`, a random sphere point, and remember `||Pi* w_i^0|| ~ 1/sqrt(d)`.
Each `<C_k*, (w_i^0)^{⊗k}>` involves contracting the target tensor — whose singular vectors
all live in `V*` (because `C_k(f*) = C_k(g*) . (W*,...,W*)`, so they're built out of the
teacher directions) — against `k` copies of a vector that has only an `O(1/sqrt(d))` foot in
`V*`. So that scalar is `O((1/sqrt(d))^k)`: each Hermite order costs another factor of
`1/sqrt(d)`. The contraction term `C_{k+1}* x_{1..k} (w_i^0)^{⊗k}` is likewise
`O((1/sqrt(d))^k)`. So the *lowest* surviving order dominates, and which order is lowest is
set by the first nonzero Hermite coefficient of the target — the leap index `ell`. Truncate:

  `E[g_i] ≈ a_i * C_ell* x_{1..(ell-1)} (w_i^0)^{⊗(ell-1)}`,  with magnitude `~ d^{-(ell-1)/2}`.

(Everything above order `ell` is smaller by more powers of `1/sqrt(d)`; the `w`-aligned sum
starts at `k = ell` too and is comparably small.) This is the crux. The useful part of the
gradient — the part that points into `V*` — has size only `d^{-(ell-1)/2}`. A vanilla step
`w <- w - eta g` with `eta = O(1)` moves the row by `O(d^{-(ell-1)/2})` into `V*`, which is
nothing: the alignment ratio stays near its `O(1/d)` starting value. That's the lazy regime,
and it's why a small step keeps you in kernel-land.

So make the step *giant* on purpose: choose `eta` to exactly cancel the smallness. I want
the post-step projection `Pi* w_i^1 = Pi* w_i^0 + eta Pi* g_i` to be `Theta(1)`. Since
`Pi* g_i ~ d^{-(ell-1)/2}` (it's the leap term, which lives in `V*`), I need

  `eta ~ d^{(ell-1)/2}`,

and accounting for the `1/sqrt(p)` and the `a_i ~ 1/sqrt(p)` factors that sit in front,
`eta = p d^{(ell-1)/2}` does it. More usefully, in terms of batch size: this is the scaling
`eta = O(p sqrt(n/d))`, an aggressive, `p`-sized learning rate. The point isn't the exact
constant — it's that the step has to be large enough to overcome `ell-1` powers of
`1/sqrt(d)`, and a kernel-regime `O(1)` step simply can't.

I should sanity-check that a giant step doesn't just blow up the row's norm and wash out the
alignment. The norm update is `||w_i^1||^2 = 1 + 2 eta <w_i^0, g_i> + eta^2 ||g_i||^2`. The
cross term `<w_i^0, g_i>` is small (the gradient is nearly orthogonal to the starting row),
so the norm is dominated by `eta^2 ||g_i||^2`. Both numerator (the `V*` part) and the bulk
of `g_i` get scaled by `eta`, so the *ratio* `||Pi* w_i^1||^2 / ||w_i^1||^2` settles to a
positive `Theta(1)` random variable — which is exactly the right object: the alignment, not
the raw norm, becomes order one. Good. And there's a hard floor: if I don't have enough
data, the empirical leap term is buried in sampling noise. Below `n = Theta(d^ell)` the
estimate of `C_ell* x ...` is dominated by fluctuations, and one can show
`||Pi* w_i^1||^2 / ||w_i^1||^2` stays `O(polylog(d)/d^{(1∧delta)/2})` — i.e. *nothing* is
learned. So `n = Theta(d^ell)` is the threshold for one giant step to see leap order `ell`.

Now a wall. Stare at what the leap term actually is: `C_ell* x_{1..(ell-1)} (w_i^0)^{⊗(ell-1)}`
lives in `V_ell*`, the span of the higher-order-SVD singular vectors of the single tensor
`C_ell*`. For the common case `ell = 1`, `C_1*` is just a vector — the first Hermite
coefficient of the target — and `V_ell*` is *one-dimensional*. So a single giant step, no
matter how huge `n` is, aligns every neuron with the *same one direction*: the linear part
of the target. If `r > 1`, I've recovered a single-neuron approximation and the other `r-1`
teacher directions are invisible. One step is structurally limited to one direction (more
precisely, to the directions present at the leap order). That's a real limitation, not a
tuning issue.

Two ways out. One: brute force — push the batch to `n = O(d^2)` and *remove* the dominant
`C_1*` direction first, by subtracting a plug-in estimate of the low-degree Hermite
components from the labels (`y <- y - sum_{m<k} c_hat_m H_m(z)`, `c_hat_m` estimated from
the batch), so the next-order tensor `C_2*` becomes visible and the step can specialize to
several directions at once. That works but it's tied to one step, needs the bigger batch,
and the plug-in estimate is only reliable for `n = omega(d polylog d)`. Two — and this is
the cleaner idea — *take more steps*.

Why would more steps help when the first step only saw `V_ell*`? Because after step one the
rows already have a foot in the learned directions, and that changes what the *next*
gradient can see. Use a fresh batch each step (so each gradient is independent of the
current `W` — that's what keeps the per-step Hermite analysis honest, and it's the
large-batch online idealization). After learning a subspace `U_t*`, condition the target on
its `U_t*`-component: define `f*_{U,x}(x_perp) = f*(x + x_perp)` for `x in U_t*`. The
leading new direction the next step picks up is the first Hermite coefficient of this
*conditioned* function, `mu_{U_t*, x}(f*) = E_{x_perp}[grad_{x_perp} f*_{U,x}]`. So the
recovered subspace grows as

  `U_0* = {0}`,   `U_{t+1}* = U_t* ⊕ span{ mu_{U_t*, x}(f*) : x in U_t* }`.

Each already-learned direction is a *ladder*: the network conditions on it, and any
direction "linearly connected" to it (appearing in a monomial together with it, so that
conditioning exposes a nonzero first Hermite coefficient) gets learned at the next step.
Let me check this resolves the `ell=1` stall. Take `f*(z) = z1 + z2 + z1^2 - z2^2`. The
first Hermite coefficient points along `(e1+e2)/sqrt(2) =: v*`, so `U_1* = span(v*)` — one
direction, as the single-step analysis said. Rewrite in the basis `(v*, v_perp)`:
`f* = sqrt(2) <v*,z> + 2 <v*,z><v_perp,z>`. Conditioning on `x = lambda v*`, the gradient in
`v_perp` is `2 lambda v_perp` — nonzero! So `U_2* = V*`: the second step climbs to the full
subspace with only `n = O(d)`. Contrast `f*(z) = z1 + z2 + z1^2 + z2^2`: in the rotated
basis `f* = sqrt(2)<v*,z> + <v*,z>^2 + <v_perp,z>^2`, and conditioning on any `x in U_1*`
gives `mu = 0` — `v_perp` is never learned. So multiple steps recover exactly the
*staircase-connected* part of `V*` with `n = O(d)` and a handful of steps, while "hard"
directions (zero first Hermite coefficient, not connected) still need the `n = O(d^2)`
brute force. This is a genuine speedup over one-pass SGD's `d^{ell-1}`: the large-batch
gradient already contains an `O(1)` fraction along each new connected direction, so each is
learned in *one* step instead of after a long saddle escape, and I never need the
coordinate-wise projections that batch-one analyses lean on. So the recipe for stage one is
clear: iterate giant gradient steps on the first layer, fresh batch each step, frozen second
layer — and keep going long enough to climb the staircase.

One more practical point about stage one. To keep the alignment ratio the clean object and
the neuron norms controlled across many steps, it's natural to keep each row on the unit
sphere — initialize rows on `S^{d-1}` and re-normalize after each update. That doesn't change
the *direction* the gradient rotates toward; it just stops the giant steps from inflating
`||w_i||` and muddying the `||Pi* w_i||/||w_i||` story. And it matches the picture exactly:
what I care about is the row's orientation relative to `V*`, so normalizing the orientation
out is the honest thing.

Now stage one has rotated the first layer into (the staircase-reachable part of) `V*`. The
rows are good features; they are not yet a predictor. I froze `a` the whole time, so I still
have to fit the readout. Here's where the second instinct pays off again: with `W` frozen,
fitting `a` is *convex*. The features are the conjugate-kernel map `phi(z) = sigma(W z) in
R^p`, fixed, and the predictor `f_hat = (1/sqrt(p)) a^T sigma(W z)` is *linear* in `a`. So
fitting `a` to least squares with an `L2` penalty,

  `a_hat = argmin_a (1/n) sum_nu (y^nu - (1/sqrt(p)) a^T sigma(W z^nu))^2 + lambda ||a||^2`,

is ordinary ridge regression — closed form, no backprop through the second layer at all.
That's the whole reason for splitting the two layers: stage one is the only nonconvex,
feature-learning part, and stage two is a one-shot linear solve. Let me write the solution.
With the feature matrix `X = sigma(W Z) in R^{n x p}` and labels `Y`, the ridge normal
equations give `a_hat = (X^T X + lambda I_p)^{-1} X^T Y`. But `p` here can be larger than
`n` — overparametrized networks are the interesting case — and then inverting a `p x p`
matrix is wasteful and ill-posed. Use the push-through identity
`(X^T X + lambda I_p)^{-1} X^T = X^T (X X^T + lambda I_n)^{-1}` to solve in the smaller
space when `n < p`:

  `a_hat = X^T (X X^T + lambda I_n)^{-1} Y`   if `n < p`,
  `a_hat = (X^T X + lambda I_p)^{-1} X^T Y`   if `n >= p`.

Same minimizer either way; just take the cheaper, better-conditioned branch. `lambda > 0`
keeps the Gram matrix invertible and regularizes the readout. And I should fit `a` on a
*fresh* batch, independent of the data that trained `W`: after the giant steps `W` is no
longer independent of the stage-one data, so reusing it would correlate the features with
the fit and complicate (and bias) the readout — resampling makes the features a fixed map
"pretrained" on separate data, which is both cleaner to reason about and the safer thing to
do.

What should I expect this two-stage predictor to achieve, and what not? On the directions
the first layer actually learned, the frozen features behave like a finite-dimensional
random-features model in an `r`-dimensional space — with enough neurons it can fit *any*
function of the learned subspace. Orthogonal to the learned subspace, though, the features
are still random high-dimensional features, so the same degree-`k` kernel barrier applies
there. Made precise: with bounded width `p`, no choice of `a` can beat the best predictor
that depends only on the learned subspace `U` —
`E[(f*(z) - f_hat(z))^2] >= E[Var(f*(z) | P_U z)] - o(1)` — and that lower bound is exactly
the variance of `f*` left unexplained by `U`. So everything rides on how much of `V*` stage
one recovered: a single step gets `V_ell*` (one direction if `ell=1`), multiple staircase
steps get the connected part, and only those parts are then learnable by the readout. The
two stages are not independent knobs — stage one decides the ceiling, stage two reaches it.

Let me assemble it. Initialize the first-layer rows on the unit sphere and the second layer
in `+/-` pairs so the network starts at zero output. Stage one: for `T` steps, draw a fresh
Gaussian batch, compute the first-layer gradient `g_i = (a_i/sqrt(p)) (1/n) sum_nu z^nu
sigma'(<w_i,z^nu>)(f*(z^nu) - f_hat(z^nu))`, take a giant step `W <- W + eta G` with `eta ~
p sqrt(n/d)`, and re-normalize rows to the sphere. Stage two: freeze `W`, draw a fresh
batch, form `X = sigma(W Z)`, and solve ridge for `a` with the `n<p`/`n>=p` branch. The
predictor is `(1/sqrt(p)) a_hat^T sigma(W z)`.

```python
import numpy as np

def sigma(x):                 # ReLU student activation
    return np.maximum(x, 0.0)

def sigma_prime(x):
    return (x > 0).astype(float)

def net(Z, W, a, p):          # two-layer net with 1/sqrt(p) prefactor
    return (1.0 / np.sqrt(p)) * sigma(Z @ W.T) @ a

def sample_data(n, d, teacher, link):
    Z = np.random.randn(n, d)
    Y = link(Z @ teacher.T)               # y = g*(U z), multi-index target
    return Z, Y

def ridge_estimator(X, y, lam):
    # min_a ||y - X a||^2 + lam ||a||^2 ; take the cheaper normal-equation branch
    m, p = X.shape
    if m >= p:
        return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
    else:                                  # push-through identity for n < p
        return X.T @ np.linalg.solve(X @ X.T + lam * np.eye(m), y)

def two_stage(d, p, r, teacher, link, n, T, eta_scale=10.0, lam=1.0):
    # --- init: rows on the sphere, second layer in +/- pairs so output starts at 0
    W = np.random.randn(p, d); W /= np.linalg.norm(W, axis=1, keepdims=True)
    a = np.sign(np.random.randn(p)) / np.sqrt(p)        # frozen during stage 1

    # --- Stage 1: giant gradient steps on the first layer, fresh batch each step
    eta = eta_scale * p * np.sqrt(n / d)                # eta ~ p sqrt(n/d): cancels d^{-(ell-1)/2}
    for t in range(T):
        Z, Y = sample_data(n, d, teacher, link)
        resid = Y - net(Z, W, a, p)                     # residual; =Y at t=0 by symmetric init
        # g_i = (a_i/sqrt p) * (1/n) sum_nu z^nu sigma'(<w_i,z^nu>) resid_nu
        G = (1.0 / n) * Z.T @ ((1.0 / np.sqrt(p)) *
              np.outer(resid, a) * sigma_prime(Z @ W.T))   # shape (d, p)
        W = W + eta * G.T                               # giant step into V*
        W /= np.linalg.norm(W, axis=1, keepdims=True)   # keep rows on the sphere

    # --- Stage 2: freeze W, ridge-fit the readout on a fresh batch (CK features)
    Z2, Y2 = sample_data(n, d, teacher, link)
    X = sigma(Z2 @ W.T)                                 # conjugate-kernel features
    a_hat = ridge_estimator(X / np.sqrt(p), Y2, lam)    # closed-form readout
    predict = lambda Zte: (1.0 / np.sqrt(p)) * sigma(Zte @ W.T) @ a_hat
    return W, a_hat, predict
```

The causal chain, end to end: the structure is in `V*` but a random row touches it only at
`O(1/sqrt(d))`; Stein's lemma + the Hermite expansion show the first gradient's useful part
is the leap-index term of size `d^{-(ell-1)/2}`, so freezing the second layer (to get a clean
correlation gradient and a zero-output start) and taking a giant step `eta ~ p sqrt(n/d)`
rotates the rows into `V_ell*` once `n = Theta(d^ell)`; one step only reaches the leap-order
directions (a single one when `ell=1`), so iterating giant steps on fresh batches climbs the
conditioning staircase to recover the connected part of `V*` with `n = O(d)`; and with the
features fixed, the readout is a convex ridge solve on the conjugate-kernel features, fit on
fresh data — yielding a predictor whose reach is exactly the subspace stage one managed to
recover.
