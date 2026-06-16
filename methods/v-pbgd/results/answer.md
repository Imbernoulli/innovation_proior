# V-PBGD, distilled

V-PBGD (value-function penalty-based bilevel gradient descent) solves bilevel problems whose
lower level is non-convex (only Polyak-Lojasiewicz, not strongly convex, possibly with a
non-singleton solution set) using only first-order information, with a finite-time convergence
rate. It penalizes the lower-level *value gap* `g(x, y) - v(x)` and descends the penalized
single-level objective; the only subtlety — computing `grad v(x)` — is removed by a Danskin/PL
lemma that makes it a plain first-order gradient of `g` at an approximate inner minimizer.

## Problem it solves

```
min_{x,y}  f(x, y)    s.t.   x in C,   y in S(x) := argmin_{y in R^{d_y}} g(x, y),
```

with `g(x, .)` satisfying the `1/mu`-PL inequality (and `L_g`-smoothness), `f(x, .)` `L`-Lipschitz,
`f, g` Lipschitz-smooth, `C` closed convex. No strong convexity, no Hessian inverse, no
differentiation through the inner trajectory.

## Key idea

1. **Squared-distance bound.** Lower-level optimality `y in S(x)` is `d^2_{S(x)}(y) = 0`. Since
   `d^2` is uncomputable, penalize a smooth `p(x, y)` that is a `rho`-squared-distance bound:
   `p >= 0`, `rho*p >= d^2_{S(x)}(y)`, and `p = 0` iff `d_{S(x)}(y) = 0`. Under PL, the value
   gap `p = g(x,y) - v(x)` qualifies because PL plus smoothness implies quadratic growth
   `g - v >= (1/mu) d^2`. The gradient-norm square `p = ||grad_y g||^2` also qualifies, but only
   under the stronger `1/sqrt(mu)` PL constant required for that case.

2. **Use the value gap, not the gradient norm.** A local solution of the value-gap-penalized
   problem has `||grad_y g|| <= L/gamma` (from `grad_y f + gamma grad_y g = 0`), so PL gives
   `p = g - v <= mu L^2/gamma^2 = O(1/gamma^2)` — *no condition on* `grad_yy g`. The gradient-norm
   penalty instead needs the singular values of `grad_yy g` bounded below, because its stationarity
   only controls `grad_yy g grad_y g`. In the diagnostic toy example `y = 2*pi/3` is spurious exactly
   because that Hessian factor vanishes while the lower-level gradient norm is still nonzero.

3. **First-order `grad v` (Danskin / PL).** `grad p = grad(g - v)` needs `grad v`. Under PL +
   smoothness (Nouiehed et al. 2019, Lemma A.5), `grad v(x) = grad_x g(x, y*)` for *any*
   `y* in S(x)` (the implicit term `(dy*/dx)^T grad_y g(x, y*)` vanishes since `grad_y g = 0` at an
   unconstrained minimum), and `v` is `(L_g + L_g^2 mu)`-smooth. So approximate `y*` by `T_k` inner
   GD steps, warm-started at `y_k`, and set `grad v(x_k) ~ grad_x g(x_k, y_hat_k)`.

The update direction is `h_k = grad g(x_k, y_k) - (grad_x g(x_k, y_hat_k), 0)` (subtract only in
`x`), and the outer step is projected gradient descent on `F_gamma = f + gamma p`.

## Faithful reformulation (why it is legitimate)

- **Global relation.** With the calmness inequality `min_{z>=0} (-Lz + (gamma*/rho) z^2) =
  -L^2 rho/(4 gamma*)` and `gamma* = L^2 rho/(4 eps_1)`: any bilevel global solution is an
  `eps_1`-global-min of `UP_{gamma p}`, and conversely an `eps_2`-global-min of `UP_{gamma p}`
  (`gamma > gamma*`) has `p(x_gamma, y_gamma) <= (eps_1 + eps_2)/(gamma - gamma*)`, hence small
  `d^2` to `S(x)`.
- **Local relation (value gap).** A local solution of `UP_{gamma p}` has `p <= mu L^2/gamma^2`, so
  with `gamma >= L sqrt(3 mu/delta)` it is a local solution of the `delta`-approximate bilevel
  problem `UP_delta`.
- **Tightness.** `f = y, g = y^2`: solution of `UP_{gamma p}` is `-1/(2 gamma)`, gap
  `1/(4 gamma^2)`, so `gamma = Omega(delta^{-0.5})` is required and tight.
- **Stationary relation (no constraint qualification).** The value-gap stationarity equations give
  the KKT multiplier candidate `w = gamma(y - y*)`. Be careful with scaling: from
  `||grad_y f + gamma grad_y g|| <= eta`, feasibility is bounded by `(L + eta)/gamma`. Thus the
  local/global accuracy scaling `gamma = Omega(delta^{-0.5})` gives an immediate
  `O(sqrt(delta))` feasibility bound when `eta = O(delta)`; an `O(eps)` KKT residual requires
  choosing `gamma` and the penalized stationarity tolerance so that `(L + eta)/gamma = O(eps)`
  (equivalently, reparameterize the theorem's tolerance).

## Algorithm (V-PBGD, unconstrained lower level)

```
Input: (x_1, y_1) in C x R^{d_y}; step sizes alpha, beta; penalty gamma; inner steps T_k.
for k = 1, ..., K:
    omega_1 = y_k                                            # warm start
    for t = 1, ..., T_k:                                     # inner GD on g(x_k, .)
        omega_{t+1} = omega_t - beta * grad_y g(x_k, omega_t)
    y_hat_k = omega_{T_k + 1}                                # approximate y*(x_k) in S(x_k)
    h_k = grad g(x_k, y_k) - (grad_x g(x_k, y_hat_k), 0)     # estimate of grad p = grad(g - v)
    (x_{k+1}, y_{k+1}) = Proj_{C x R^{d_y}}( (x_k, y_k) - alpha * ( grad f(x_k, y_k) + gamma * h_k ) )
```

Constants: `alpha in (0, (L_f + gamma(2 L_g + L_g^2 mu))^{-1}]`, `beta in (0, L_g^{-1}]`,
`gamma >= L sqrt(3 mu / delta)`, `T_k = Omega(log(alpha k))`. The step `alpha ~ 1/gamma` because
`F_gamma` is `L_gamma = L_f + gamma(2 L_g + L_g^2 mu)`-smooth.

## Convergence

With `G_gamma(z) = (1/alpha)(z - Proj_Z(z - alpha grad F_gamma(z)))` the projected-gradient mapping
and `C_f = inf f`:

```
(1/K) sum_{k=1}^K ||G_gamma(x_k, y_k)||^2  <=  18 (F_gamma(x_1, y_1) - C_f) / (alpha K)  +  10 L^2 L_g^2 / K.
```

Complexity `O~(gamma eps^{-1})` to an `eps`-stationary point of `UP_{gamma p}`; with
`delta = eps`, `gamma = O(eps^{-0.5})`, total `O~(eps^{-1.5})`. Proof skeleton: inner GD is
linearly convergent under PL (`g(x, omega_{t+1}) - v <= (1 - beta/(2 mu))(g(x, omega_t) - v)`),
the error bound gives `d^2_{S(x)}(y_hat) <= mu (1 - beta/(2mu))^{T} (g(x, omega_1) - v)`; the
warm start bounds `g(x_k, omega_1) - v(x_k) <= (2/(mu gamma^2 alpha^2))||y_{k+1} - y_k||^2 +
2 L^2/(mu gamma^2)`; `T_k = Omega(log k)` drives the gradient-estimation error
`||grad F_gamma - hat grad F_gamma||^2 = gamma^2 ||grad v - grad_x g(.,y_hat)||^2 <= gamma^2 L_g^2
d^2_{S(x)}(y_hat)` below `O(1/k^2)`; descent of the `L_gamma`-smooth `F_gamma` via the projection
optimality `Proj_Z(z - alpha q) = argmin_{z' in Z} <q, z'> + (1/(2 alpha))||z - z'||^2` plus Young;
telescope with `sum 1/k^2 <= 1`.

## Variants

- **Constrained lower level (`U(x) = U` compact convex).** Inner = *projected* GD; `grad v` via a
  generalized Danskin proposition (under quadratic growth + error bound, or + convexity); inner
  linear convergence via the proximal-PL condition. Rate `(1/K) sum ||G_gamma||^2 <=
  8(F_gamma(z_1) - C_f)/(alpha K) + 3 L_g^2 mu C_g / K`; complexity `O~(eps^{-1.5})`
  (quadratic-growth + convex) or `O~(eps^{-2})` (quadratic-growth + error-bound).
- **Stochastic (V-PBSGD).** Inner SGD with `beta_t = 1/(L_g sqrt t)`; pick `y_hat` from the
  step-size-weighted distribution `P(i = t) ~ beta_t`; outer minibatch size `M`. Rate
  `O(1/(alpha K)) + O(gamma^2 c^2 / M) + O(gamma^2 ln T / sqrt T)`.
- **Nonsmooth exact penalty (PBPL).** `p = ||grad_y g||` (norm, not square). For `gamma > L sqrt(mu)`
  the penalized global solutions equal the bilevel global solutions *exactly* — constant
  `gamma = O(1)`, no `eps` gap. Solve the smooth-plus-norm composite by the Prox-linear method
  (linearize `f` and `grad_y g`, so the subproblem uses the Jacobian of `grad_y g`, i.e. Hessian
  blocks of `g`; keep `gamma||.||`, add `(1/(2 lambda))||z - z^k||^2`), giving
  `min_k ||G_lambda||^2 <= 2 lambda^{-1}(F(z^0) - C_f + sum delta_k)/K = O(1/K)` and complexity
  `O~(eps^{-1})`. Caveat: the subproblem rarely has a closed form.

## Code grounding

Toy verification (here `S(x) = {-x}` so `v(x) = 0`, value gap `p = g`, inner loop collapses).
The snippet below uses the derivative of the displayed `f`. The released author file
`V-PBGD/toy/toy.py` differs in the first `df` denominator, using
`1 + np.exp(2-4*x)**2` by Python precedence rather than `(1 + np.exp(2-4*x))**2`; that local
implementation detail should not be silently treated as the mathematical derivative.

```python
import numpy as np

def g(x, y):
    return (x + y) ** 2 + x * np.sin(x + y) ** 2

def dg(x, y):                                       # gradient of g in (x, y)
    return 2 * np.array([x + y + 0.5 * np.sin(x + y) ** 2 + x * np.sin(x + y) * np.cos(x + y),
                         x + y + x * np.sin(x + y) * np.cos(x + y)])

def f(x, y):
    return np.cos(4 * y + 2) / (1 + np.exp(2 - 4 * x)) + 0.5 * np.log((4 * x - 2) ** 2 + 1)

def df(x, y):
    return np.array([4 * np.exp(2 - 4 * x) * np.cos(4 * y + 2) / (1 + np.exp(2 - 4 * x)) ** 2
                     + (16 * x - 8) / ((4 * x - 2) ** 2 + 1),
                     -4 * np.sin(4 * y + 2) / (1 + np.exp(2 - 4 * x))])

def box(x, xlim):                                   # projection onto C = [0, 3]
    return min(max(x, min(xlim)), max(xlim))

def solve_vpbgd(x, y, alpha, gam, xlim, eps=1e-5):
    dF = df(x, y) + gam * dg(x, y)                  # grad F_gamma = grad f + gamma*grad(g - v)
    x_, y_ = box(x - alpha * dF[0], xlim), y - alpha * dF[1]
    pg = (1 / alpha) * np.array([x - x_, y - y_])   # projected-gradient mapping G_gamma
    k = 0
    while np.linalg.norm(pg) > eps:                 # stop at ||G_gamma|| <= eps
        x, y = x_, y_
        dF = df(x, y) + gam * dg(x, y)
        x_, y_ = box(x - alpha * dF[0], xlim), y - alpha * dF[1]
        pg = (1 / alpha) * np.array([x - x_, y - y_])
        k += 1
    return x, y, k

if __name__ == '__main__':
    xlim, gam, alpha0, N = [0., 3.], 10.0, 0.1, 1000
    for n in range(N):
        x0, y0 = np.random.uniform(0, 3.5), np.random.uniform(-5, 8.5)
        x, y, k = solve_vpbgd(x0, y0, alpha0 / gam, gam, xlim)   # alpha = alpha0 / gamma
        print('run', n, 'steps', k, '(x,y)=({:.3f},{:.3f})'.format(x, y))
```

Data hyper-cleaning (real inner loop; canonical code keeps a persistent `net_inner` auxiliary
network initialized from `net`, then updates it across outer iterations). In `vx`, the auxiliary
model outputs and auxiliary regularizer are detached, but `sigmoid(x)` is not; therefore the value
gap contributes `grad g(x,y) - (grad_x g(x, y_hat), 0)`:

```python
import copy
import torch
import torch.nn.functional as F

def sq_norm(params):
    return sum(torch.linalg.norm(w) ** 2 for w in params)

def run_vpbgd_hyperclean(net, x, tr, val, lrx, lry, lr_inner, inner_itr,
                         gamma_init, gamma_max, gamma_argmax_step, outer_itr, reg=0.0):
    # net: classifier (lower variable y); x: per-example logits (upper variable), weight=sigmoid(x)
    y_opt = torch.optim.SGD(net.parameters(), lr=lry)
    x_opt = torch.optim.SGD([x], lr=lrx)
    gam = gamma_init
    step_gam = (gamma_max - gamma_init) / gamma_argmax_step          # linear gamma ramp

    net_inner = copy.deepcopy(net); net_inner.train()               # persistent auxiliary inner net
    opt_inner = torch.optim.SGD(net_inner.parameters(), lr=lr_inner)

    for k in range(outer_itr):
        net.train()
        with torch.no_grad():
            sigx = torch.sigmoid(x)                                  # frozen weights for inner loop
        for _ in range(inner_itr):                                   # inner GD on g(x_k, .) -> y_hat
            opt_inner.zero_grad()
            ce = F.cross_entropy(net_inner(tr.data), tr.dirty_target, reduction='none')
            inner_loss = (sigx * ce).mean() + reg * sq_norm(net_inner.parameters())
            inner_loss.backward()
            opt_inner.step()

        y_opt.zero_grad(); x_opt.zero_grad()
        fy = F.cross_entropy(net(val.data), val.clean_target)                          # f(x, y)
        ce_tr = F.cross_entropy(net(tr.data), tr.dirty_target, reduction='none')
        gxy = (torch.sigmoid(x) * ce_tr).mean() + reg * sq_norm(net.parameters())      # g(x, y)
        ce_in = F.cross_entropy(net_inner(tr.data), tr.dirty_target, reduction='none').detach()
        vx = (torch.sigmoid(x) * ce_in).mean() + reg * sq_norm(net_inner.parameters()).detach()
        # v(x) ~ g(x, y_hat): y_hat side detached, x side live through sigmoid(x)

        lr_decay = min(1.0 / (gam + 1e-8), 1.0)                      # realizes alpha ~ 1/gamma
        loss = lr_decay * (fy + gam * (gxy - vx))                    # F_gamma = f + gamma*(g - v)
        loss.backward()                            # -> grad f + gamma*(grad g - (grad_x g(.,y_hat),0))
        x_opt.step(); y_opt.step()
        gam = min(gamma_max, gam + step_gam)
```

Reference hyper-parameters: linear model `lrx=lry=0.1, lr_inner=0.01, inner_itr=1,
gamma_max=0.2, gamma_argmax_step=30000, outer_itr=40000`; MLP (`784->300->10`, sigmoid hidden)
`lrx=0.1, lry=0.01, lr_inner=0.01, inner_itr=1, gamma_max=0.1, gamma_argmax_step=10000,
outer_itr=80000`. Toy: `gamma=10, alpha0=0.1, 1000` random inits over `x in [0, 3]`.
