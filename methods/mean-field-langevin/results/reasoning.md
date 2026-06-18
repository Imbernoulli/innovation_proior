Let me start from the thing that actually stings. I want to fit `y = g(Ux) + xi` with a two-layer net, where `U` picks out a handful of directions out of `d` and `g` is some unknown scalar function of those `k = O(1)` projections. Counting alone says this should be cheap: `g` lives on `k` directions, so there are about `kd = O(d)` numbers in the model, and on isotropic data the information floor for estimating `O(d)` parameters is `n ~ d`. Yet when I train the net by plain SGD, the sample size I actually need is nowhere near `d`. For a single relevant direction it's been pinned down sharply: the cost is graded by the information exponent `s` of the link — the index of the first nonzero Hermite coefficient of `g`, equivalently the degree of the first nonvanishing term when you expand the population correlation about the uninformative equator. You get `n ~ d` only for `s = 1`, `n ~ d log d` for `s = 2`, and `n ~ d^{s-1}` for `s >= 3`. For several directions the analogue is the leap complexity, and SGD on a two-layer net takes about `d^{max(Leap, 2)}` steps, crawling saddle-to-saddle. The benchmark link I care about is a sum of third Hermite polynomials, `He_3(z) = z^3 - 3z`, information exponent exactly `3`. So I'm staring at `n ~ d^2` when the floor is `n ~ d`, and I want to understand *why* the gradient method pays this and whether a *standard* training procedure can avoid it.

Why does SGD pay it? Picture one neuron's weight `w` and the overlap `<w, u>` with a true direction `u`. At a random start in `d` dimensions that overlap is `O(1/sqrt(d))`, tiny. The drift that pulls `w` toward `u` comes from the gradient of the population correlation, and near the equator that correlation behaves like `overlap^s` — so its derivative, the actual pull, scales like `overlap^{s-1}`. For `s >= 3` that's `(1/sqrt(d))^{s-1}`, vanishingly weak. The iterate just diffuses in the search phase for an enormous number of steps before the signal is strong enough to take over. The whole `d^{s-1}` wall is the cost of escaping that flat, near-equatorial region one weak gradient step at a time. The difficulty is not that the answer is hard to *represent* — a two-layer net approximates any such `g` fine — it's that the loss landscape around a random init is almost flat in the directions I need, and a single gradient trajectory feels almost nothing there.

So the enemy is the flatness-at-init of a non-convex landscape that one descending trajectory has to climb out of. What would make that go away? If the optimization problem were *convex*, "flat at the start" wouldn't trap me — there'd be no spurious basins, and I could in principle move steadily to the global optimum. The trouble is the loss in the weights `W` is manifestly non-convex. But there's a lift I keep coming back to. The width-`m` network `yhat_m(x;W) = (1/m) sum_j Psi(x; w_j)` and the norm penalty `(1/m) sum_j ||w_j||^2` don't care about the order of the neurons — they depend on `W` only through the empirical measure `mu = (1/m) sum_j delta_{w_j}` over neuron weights. So write everything as a functional of `mu`:

```
yhat(x; mu) = integral Psi(x; w) dmu(w),    R(mu) = integral ||w||^2 dmu(w).
```

Now look at the data-fit term as a functional of `mu`. The prediction is *linear* in `mu`, and the per-example loss `rho(yhat - y)` is convex in its argument, so the population/empirical risk `J(mu) = E[rho(yhat(x;mu) - y)]` is a *convex* functional of `mu`. The norm penalty is linear in `mu`, hence convex too. The non-convexity I was fighting in `W`-space was an artifact of pinning the measure to `m` atoms; lift to the space of measures and the objective is convex. That reframes the problem completely. Instead of one weight vector descending a bumpy landscape, I have a *population* of neurons — a distribution — flowing down a convex functional. There are no spurious local minima as a measure.

How do I actually move a measure downhill? Each neuron should follow the negative gradient of the objective with respect to *its own* position, which is the gradient of the first variation of `J` at the current `mu`. The first variation `J'[mu]` is the function whose integral against `nu - mu` gives the directional derivative of `J(mu)` in the direction `nu - mu` — the measure-space analogue of a gradient. So the natural flow is each neuron `w_j` drifting by `-grad_w J'[mu](w_j)`. As `m -> infinity` the empirical measure of the neurons under this drift converges (propagation of chaos) to a deterministic measure `mu_t` solving a Wasserstein gradient flow of `J`. Good — that's the convex descent in measure space made concrete as an interacting particle system.

But convexity of `J` in `mu` alone doesn't hand me a *rate*, and worse, it doesn't even guarantee I land on the global optimum from a bad start: a pure gradient flow of a convex functional over measures can stall, and the minimizing measure can be degenerate (e.g. collapse onto a few atoms) in ways that make the flow's convergence delicate. I need to regularize the *measure itself*, not just penalize neuron norms, and I want the regularizer to do two jobs: keep the minimizer a nice spread-out density, and give me a strong-convexity-like handle for a rate. The canonical object that does both is entropy. Add a relative-entropy term against a base measure `tau`:

```
F_beta(mu) = J(mu) + (1/beta) H(mu | tau),   H(mu | tau) = integral ln(dmu/dtau) dmu,
```

with `beta` an inverse temperature controlling how much entropy I pay. Now `F_beta` is *strictly* convex in `mu` (entropy is strictly convex), it has a unique minimizer `mu*_beta`, and crucially the entropy term is what couples to noise in the dynamics — let me see how.

The gradient flow of `F_beta` in Wasserstein space is no longer the bare drift; the entropy contributes a Laplacian (diffusion) term. Writing it out, the density evolves by a nonlinear Fokker–Planck equation

```
d_t mu_t = div(mu_t grad J'[mu_t]) + beta^{-1} Laplacian(mu_t),
```

and the corresponding particle SDE is

```
dw_t = - grad J'[mu_t](w_t) dt + sqrt(2 beta^{-1}) dB_t,
```

with `B_t` Brownian motion. So the entropy regularizer is *exactly* an injected Gaussian noise of scale `sqrt(2/beta)` on each neuron. That's the mean-field Langevin dynamics: the convex measure-space descent of `J`, plus diffusion. And now I can see why this should beat single-trajectory SGD on the flatness problem. The noise term is doing the saddle-escape that SGD's gradient had to do unaided. Near the flat equator where the drift is `O(overlap^{s-1})` and almost nothing, the diffusion keeps the measure exploring; it's the temperature, not the feeble gradient, that gets the population off the flat region and toward the directions that matter. The cost of the high information exponent — the long, weak search phase — is no longer paid by waiting for a tiny gradient to accumulate.

Now I need the rate, because "convex + noise converges" is not a guarantee I can size `n`, `m`, and the iteration count from. The right way to read this dynamics is as Langevin sampling of a Gibbs measure. If I freeze the interaction and look at the *linearized* potential at the current `mu`, the stationary law of `dw = -grad J'[mu](w) dt + sqrt(2/beta) dB` is the Gibbs distribution

```
p_mu  proportional to  exp(-beta * J'[mu](w)) dtau(w),
```

the proximal Gibbs distribution of `mu`. This is the object that ties the whole thing together. The gap `F_beta(mu) - F_beta(mu*_beta)` is controlled by how far `mu` is from its own proximal Gibbs `p_mu` — `p_mu` "fills the duality gap." So if `p_mu` is a *nice* distribution to sample from, the flow converges fast. The precise sense of "nice" for Langevin convergence is a log-Sobolev inequality: `p_mu` satisfies an LSI with constant `C_LSI` if

```
H(mu | p_mu) <= (C_LSI / 2) I(mu | p_mu)   for all mu,
```

where `I(mu | nu) = integral || grad ln(dmu/dnu) ||^2 dmu` is the relative Fisher information. An LSI is the measure-space stand-in for strong convexity of the potential — it's the gradient-domination condition that turns a flow into an exponentially contracting one. And under a *uniform-in-time* LSI on the proximal Gibbs measures along the trajectory, the free energy decays exactly geometrically,

```
F_beta(mu_t) - F_beta(mu*_beta) <= exp(-2t / (beta C_LSI)) (F_beta(mu_0) - F_beta(mu*_beta)).
```

There's the rate. The time constant is `beta C_LSI`. So convergence speed is governed entirely by two numbers I now have to understand: `beta` (how cold I run) and `C_LSI` (how well-conditioned the proximal Gibbs is).

Before I worry about `C_LSI`, let me connect the entropy regularizer back to something I can actually *implement*, because right now I have a continuous SDE and an abstract base measure `tau`, and I want the everyday two-layer training rule that realizes it. Take `tau` to be the Lebesgue measure and ask: what is the relationship between the entropy term `(1/beta) H(mu | Leb)` and the ordinary `L2` weight penalty I already know how to add? Combine the norm penalty and the entropy:

```
(lambda/2) integral ||w||^2 dmu + (1/beta) H(mu | Leb)
  = (1/beta) [ integral (beta lambda / 2) ||w||^2 dmu + integral ln(dmu/dLeb) dmu ]
  = (1/beta) integral ln( dmu / exp(-(beta lambda/2)||w||^2) ) dmu  + const
  = (1/beta) H(mu | gamma) + const,   gamma = N(0, (lambda beta)^{-1} I).
```

So an `L2` penalty of strength `lambda` *together with* entropy at temperature `1/beta` is precisely a relative entropy to a Gaussian base measure `gamma`. The weight decay is the relative-entropy regularizer — it's not an add-on, it's the KL term in disguise, with the base measure being the Gaussian whose width is set by `lambda beta`. That's the bridge I wanted: the two knobs of the abstract free energy, `lambda` and `beta`, are nothing more exotic than weight decay and the Langevin noise scale. Concretely the per-neuron drift `-grad J'[mu]` splits into the data-fit gradient plus `-lambda w` (the gradient of the norm penalty), i.e. a weight-decay drift; and the diffusion is the injected `sqrt(2/beta)` Gaussian noise. The whole "mean-field Langevin" objective is: ordinary regularized risk, trained with ordinary weight decay, plus Gaussian noise on the parameters. Nothing about the *algorithm* needs the measure-space machinery — that machinery is what tells me it converges globally.

Let me discretize to get the actual update. Euler–Maruyama on `dw = -grad J'[mu] dt + sqrt(2 beta^{-1}) dB` with step `eta`: the drift contributes `-eta grad J'[mu]` and the Brownian increment over time `eta` contributes `sqrt(2 eta beta^{-1})` times a standard Gaussian. For the finite-width interacting system there's a bookkeeping factor I have to be careful with. The gradient of the *averaged* objective `Jhat_lambda(W) = (1/n) sum_i rho(yhat_m(x_i;W) - y_i) + (lambda/2)(1/m) sum_j ||w_j||^2` with respect to one neuron `w_j` is `O(1/m)`, because `yhat_m` divides by `m` and the penalty divides by `m`. But the measure-space drift `-grad J'[mu](w_j)` is `O(1)` per neuron. To make the discrete particle step track the SDE I have to undo that `1/m`: scale the learning rate on the gradient by `m`. So the update is

```
w_j^{l+1} = w_j^l - m eta grad_{w_j} Jhat_lambda(W) + sqrt(2 eta beta^{-1}) xi_j^l,   xi_j^l ~ N(0, I) iid.
```

Expand the gradient: `m eta grad_{w_j} Jhat_lambda` `= eta * (per-neuron data gradient) + eta lambda w_j`, the second piece being exactly the weight-decay step. So in plain words the update is: subtract `eta` times the data gradient, subtract `eta lambda` times the weight (decay), add `sqrt(2 eta beta^{-1})` Gaussian noise. That is the mean-field Langevin algorithm — and it is just noisy weight-decayed gradient descent. Let me hold onto the noise constant precisely, because it's easy to get wrong: the Euler–Maruyama discretization of a diffusion with coefficient `sqrt(2 beta^{-1})` puts `sqrt(2 eta beta^{-1})` in front of the standard Gaussian, not `sqrt(eta/beta)` and not `2 sqrt(eta beta^{-1})`. If I store the noise level as `noise_std = 1/sqrt(beta)`, the per-step noise multiplier is `sqrt(2 eta) * noise_std`, which is `sqrt(2 eta / beta)`. I'll keep it in that form.

Now back to the two numbers in the rate. Sample complexity first, because that's the prize — I want to know whether this escapes the `d^{s-1}` wall. The quantity that turns out to control how many samples I need is an *effective dimension* of the problem,

```
d_eff = tr(Sigma) / || Sigma^{1/2} U^T ||_F^2 = c_x^2 / r_x^2,
```

with `c_x = tr(Sigma)^{1/2}` (the overall input scale) and `r_x = || Sigma^{1/2} U^T ||_F` (how much of the input's variance actually lands in the relevant subspace). On isotropic data `Sigma = I_d`, `tr(Sigma) = d` and `|| U^T ||_F^2 = k`, so `d_eff = d/k ~ d`. But if the covariance concentrates variance in (or aligned with) the relevant directions — a spiked or fast-decaying spectrum aligned with `U` — then `d_eff` can be far below `d`, even `polylog(d)`. The claim I'm chasing is that the sample complexity is `n ~ d_eff` up to log factors, *independent of the information/leap exponent of `g`*. Let me sanity-check the mechanism for why the exponent drops out. In single-trajectory SGD the exponent appeared because one weight had to climb out of the equator under an `overlap^{s-1}` drift. Here the measure-space objective is convex and the *minimizer* `mu*_beta` already places mass in the right directions regardless of how the link's correlation expands — the difficulty of *finding* that minimizer is pushed entirely into `C_LSI` (the compute), and the difficulty of *generalizing* from `n` samples to the population is what `d_eff` measures. The statistical question "is the empirical minimizer close to the population minimizer" is governed by the effective number of directions the data illuminates, `d_eff`, not by how high-order `g` is. So the exponent that strangled SGD's *statistics* is gone; what it costs me instead shows up in the *computation*, through `C_LSI`.

How do I set `lambda` and `beta` to make the entropy regularizer help rather than dominate? The entropy term is `(1/beta) H`, and for it not to swamp the data fit I need `beta` large enough — specifically `beta >~ d_eff`. I can see the scale from the minimizer: with the Gaussian base measure `gamma`, the relative entropy of a useful minimizer against `gamma` is of order `lambda beta / r_x^2`, and to learn with any non-trivial accuracy I need that to be `~ c_x^2/r_x^2 = d_eff`. So I take `lambda = lambdatilde r_x^2` for a small `lambdatilde`, and `beta = Theta(d_eff / lambdatilde)`. That's the "cold enough to learn" condition, and it forces `beta` large.

And here's the wall. The rate's time constant is `beta C_LSI`, and I just argued I need `beta` large. What is `C_LSI`? Certify it with Holley–Stroock: the proximal Gibbs potential `beta J'[mu]` is a bounded perturbation of a strongly convex quadratic (the `(beta lambda/2)||w||^2` from the Gaussian base measure), and a bounded perturbation of a strongly log-concave measure keeps an LSI, with the constant inflated by `exp` of the perturbation's range. The data-fit part of the first variation is bounded by the activation bound `iota` times the loss Lipschitz constant `C_rho`, so

```
C_LSI <= (1 / (beta lambda)) exp(4 C_rho iota beta).
```

The `exp(beta)` is fatal. With `beta ~ d_eff`, `C_LSI ~ exp(d_eff)`, so the number of iterations `~ beta C_LSI ~ exp(d_eff)` and the width needed for propagation of chaos blows up the same way. This is not just a loose bound — in Euclidean space an exponential dependence of the LSI constant on the inverse temperature is genuinely unavoidable in the worst case. So the lift buys me *statistical* optimality (`n ~ d_eff`, exponent-free) at the price of *computational* cost exponential in `d_eff`. Hold on though — exponential in `d_eff`, not in `d`. When the covariance is structured so that `d_eff = polylog(d)`, `exp(d_eff)` is quasipolynomial in `d`. So the lift is already a real win whenever the data has low effective dimension; the exponential is only in the worst case `d_eff = d`.

I'd like to do better than "only when `d_eff` is tiny," so let me look hard at where the `exp(beta)` came from. It came from Holley–Stroock paying `exp` of the perturbation range of `beta J'` over *all of Euclidean space* — the weights are unconstrained, the Gaussian base measure decays, but the perturbation can still range widely, and an LSI on `R^d` is fragile to that. What if I *constrain the weights to a compact set* where the geometry itself supplies curvature? Put the neurons on the unit sphere `S^{d-1}`. Two things change. First, on a compact manifold I no longer need the `L2` penalty at all to keep the measure tame — the manifold is compact, so I can drop `lambda` and regularize only by entropy against the uniform measure `tau` on the sphere, `F_beta(mu) = J(mu) + beta^{-1} H(mu | tau)`. Second, and this is the point, the sphere has *positive curvature*, and the Bakry–Émery criterion turns a lower bound on Ricci curvature directly into an LSI constant — no `exp(beta)`. For `S^{d-1}` the Ricci curvature is `(d-2) g` (the metric), so the curvature-dimension condition `Ric >= rho d * g` holds with an absolute `rho`, and the proximal Gibbs LSI constant comes out

```
C_LSI <= (rho d - beta C_rho K)^{-1},
```

where `K` measures the size of the activation's second derivative averaged over data — `K ~ ||Sigma||`, a constant. For `beta` up to order `d`, this is `~ 1/d`: *polynomial* in `d`, with no exponential in `beta` at all. The same potential range that cost `exp(beta)` in Euclidean space costs nothing on the curved compact manifold, because the curvature itself does the confining that the Gaussian base measure could only do weakly. So constraining the neurons to the sphere, and replacing the drift's `-lambda w` decay with a *projection back onto the sphere* after each noisy step (a retraction of the Riemannian Langevin update), turns the iteration count from `exp(d_eff)` into `poly(d)` in the regime where the teacher's neurons are spread across the sphere. That's the route to actually-fast training, and it's why I want spherical initialization and renormalization in the implementation rather than free Euclidean weights.

Let me also nail down the architecture details that the convergence theory quietly demands, because they're not arbitrary. I freeze the second-layer weights at fixed `+-1` values, half the neurons at `+1` and half at `-1`. Why fix them? Because the whole convex-measure lift is a statement about the distribution of *first-layer* weights — `yhat = integral Psi(x;w) dmu(w)` with the measure over `w`, the neuron directions. If I also trained the second layer I'd have signed coefficients to track and the clean "measure over neurons with unit weights" picture would need the harder signed-measure machinery. Fixing the magnitudes at `1/m` keeps the predictor a plain average `(1/m) sum_j Psi`, exactly the empirical-measure object. But a single non-negative activation like ReLU with all-`+1` second layer can only represent non-negative functions, and `g` is signed. The fix is the signed-neuron construction: half the neurons enter with `+1` and half with `-1` (or, equivalently, an activation `Psi(x;w) = phi(<om1, x>) - phi(<om2, x>)` built from two half-weight blocks), so differences of non-negative bumps can represent a signed target. That's why the second layer is `+1/m` on one half and `-1/m` on the other. The activation itself, for the *theory*, is a smoothed ReLU `phi_{kappa,iota}` that is `C^2`, bounded by `iota`, with bounded first and second derivatives — boundedness is what Holley–Stroock needs to certify the LSI (the perturbation has finite range), and `C^2`-smoothness is what the discrete-time analysis needs for the Euler–Maruyama error to be controlled; both recover plain ReLU as `kappa, iota -> infinity`. In code I can just use ReLU, but I should remember the smoothing is the technical reason the bound holds. And a bias coordinate: append a constant `1` to `x` so each neuron has a learnable threshold — without it the bumps `phi(<w, x>)` are all centered at the origin and can't place features at different offsets along a direction.

One more thing I want to be honest about: the data regime. The convergence statement is a population/large-fresh-sample statement — the drift `-grad J'[mu]` is the *population* gradient, and the finite-width finite-sample iteration only tracks it when the gradient is computed on enough fresh data that the empirical gradient is close to the population one. So the natural implementation draws from a large training pool to approximate population gradients, rather than overfitting a tiny fixed batch; that's the practical reflection of the `n ~ d_eff` sample-complexity statement and of the propagation-of-chaos requirement that the empirical measure track the mean-field limit.

Now the finite, runnable guarantee, because everything above was continuous-time and infinite-width. Discretizing in time (step `eta`), in particles (width `m`), and using stochastic gradients introduces error, but the contraction survives as a one-step decay of the finite-width free energy: there are constants such that

```
F^m(mu_{l+1}) - F(mu*) <= exp(-eta / (2 beta C_LSI)) (F^m(mu_l) - F(mu*)) + eta A,
```

with the additive `A = O(eta + 1/m + ...)` collecting the discretization and finite-width error. Unrolling, the optimality gap decays geometrically at rate `exp(-eta/(2 beta C_LSI))` down to a floor of order `A` set by step size and width — so to reach accuracy `epsilon` I take `eta` small enough that the floor is below `epsilon`, `m` large enough that `1/m` is below `epsilon`, and the iteration count `l ~ (beta C_LSI / eta) log(1/epsilon)`. The `beta C_LSI` is the same controlling product as in continuous time: poly in `d` on the sphere, exp in `d_eff` in free Euclidean space. This is the bound that lets me actually size the run.

Let me assemble the recipe and write the update into the harness. Initialize the first-layer rows on the unit sphere (this is the compact-manifold setting that gives the good LSI, and it also keeps neuron norms `O(1)`); append a bias coordinate; fix the second layer at `+-1/m` in two signed halves; draw from a large fresh pool to approximate population gradients; and iterate noisy weight-decayed gradient descent with the Euler–Maruyama noise scale `sqrt(2 eta / beta)`, renormalizing weights to the sphere after each step in the Riemannian variant. With `phi` = ReLU and `phiprime` its indicator derivative, the per-batch first-layer gradient of the squared loss is `(1/n) [ phiprime(W x^T) ⊙ a ] (residual ⊙ x)` summed appropriately, and one step is:

```python
import math
import torch


def predict(X, W, a, phi):
    """yhat = (1/?) sum_j Psi(x; w_j); with fixed second layer a, this is phi(X W^T) @ a."""
    return phi(X @ W.T) @ a


def first_layer_gradient(X, y, W, a, phi, phiprime):
    """Gradient of the per-batch squared loss w.r.t. the first-layer weights W."""
    n = X.shape[0]
    residual = (predict(X, W, a, phi) - y).reshape(1, -1)          # [1, n]
    backprop = phiprime(W @ X.T) * a.reshape(-1, 1)                # [m, n]
    return (backprop * residual) @ X / n                          # [m, d]


def init_first_layer_on_sphere(m, d):
    """Neurons are the particles of the measure mu; start them uniform on the sphere
    (compact-manifold setting -> polynomial LSI; keeps ||w_j|| = 1)."""
    W = torch.randn(m, d)
    return W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)


def mfla_step(W, X, y, a, phi, phiprime, lr, weight_decay, beta, project_to_sphere=True):
    """One mean-field Langevin update: noisy weight-decayed gradient descent.

    drift  = -(data gradient) - lambda * w          # data fit + weight decay (= KL drift)
    noise  = sqrt(2 * lr / beta) * standard Gaussian # Euler-Maruyama of the Langevin diffusion
    The m prefactor on the data gradient rescales the O(1/m) per-neuron gradient to the
    O(1) measure-space drift, matching the interacting-particle SDE.
    """
    m = W.shape[0]
    g = first_layer_gradient(X, y, W, a, phi, phiprime)           # [m, d]
    noise = math.sqrt(2.0 * lr / beta) * torch.randn_like(W)
    W = W - lr * m * g - lr * weight_decay * W + noise
    if project_to_sphere:                                         # Riemannian retraction
        W = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return W


def train_mfla(X, y, W, a, phi, phiprime, n_iters, lr=0.1, weight_decay=0.01,
               beta=1.0 / 0.001, project_to_sphere=True):
    for _ in range(n_iters):
        W = mfla_step(W, X, y, a, phi, phiprime, lr, weight_decay, beta, project_to_sphere)
    return W
```

Let me retrace the chain. Plain SGD on a two-layer net pays a sample complexity graded by the link's information/leap exponent — `d^{s-1}` for exponent `s >= 3` — because one weight trajectory has to escape a near-flat equator under an `overlap^{s-1}` drift, and that's the wall I want past while the information floor is only `n ~ d`. Lifting the network to a measure over neurons makes the data-fit functional convex, killing the spurious-basin problem in principle, but a bare convex flow gives no rate and no guarantee of landing on the global optimum from a flat start. Adding entropy regularization fixes both: the free energy `J(mu) + beta^{-1} H(mu | tau)` is strictly convex with a unique minimizer, and its Wasserstein gradient flow is exactly Langevin dynamics — the entropy term *is* an injected Gaussian noise of scale `sqrt(2/beta)`, and that noise does the saddle-escape the weak gradient couldn't, so the exponent drops out of the statistics. The convergence rate is `exp(-2t/(beta C_LSI))`, controlled by the proximal Gibbs measure `p_mu ∝ exp(-beta J'[mu])` satisfying a log-Sobolev inequality. The whole abstract objective reduces to a familiar training rule because `L2` weight decay plus entropy equals relative entropy to a Gaussian base measure — so the algorithm is just noisy weight-decayed gradient descent, with the noise scale fixed by Euler–Maruyama at `sqrt(2 eta / beta)` and the gradient scaled by width `m`. The sample complexity comes out `n ~ d_eff`, the effective dimension `tr(Sigma)/||Sigma^{1/2}U^T||_F^2`, independent of the exponent. The computational cost lives in `C_LSI`: exponential in `beta ~ d_eff` in free Euclidean space (Holley–Stroock's `exp(beta)`), hence quasipolynomial only when `d_eff = polylog(d)` — but constraining the neurons to the positively-curved sphere makes the Bakry–Émery LSI constant polynomial in `d`, which is why the neurons are initialized and kept on the sphere and the `L2` decay is replaced by a projection there. Architecturally the second layer is frozen at signed `+-1/m` so the predictor is a plain average over a measure of first-layer neurons while still representing signed targets, the activation is a bounded smooth ReLU so Holley–Stroock and the discrete-time error bound apply, and a bias coordinate gives each neuron a threshold. The finite-time finite-width version contracts one step at a time down to an `O(eta + 1/m)` floor, which is what sizes a real run.
