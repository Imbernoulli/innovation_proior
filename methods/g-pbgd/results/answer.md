# G-PBGD, distilled

G-PBGD is the gradient-norm-penalty special case of penalty-based bilevel gradient descent. It
solves a bilevel problem `min_{x,y} f(x,y)` s.t. `x in C`, `y in argmin_y g(x,y)` by replacing the
lower-level argmin constraint with a penalty on the squared lower-level gradient norm and running
one joint projected-gradient step on `(x,y)`. With the convention used in the canonical
implementation,

```
F_gamma(x, y) = f(x, y) + (gamma / 2) ||nabla_y g(x, y)||^2.
```

The factor `1/2` is only a normalization — it rescales the penalty constant so that differentiating
the penalty yields a clean Hessian-vector product:

```
nabla_x F_gamma = nabla_x f + gamma * nabla_xy g * nabla_y g
nabla_y F_gamma = nabla_y f + gamma * nabla_yy g * nabla_y g.
```

So the update needs Hessian-vector products of `g`, but no Hessian inverse, no value-function
estimate, and no unrolled lower trajectory. G-PBGD is the simplest computational member of the
penalty family; the value-gap variant (V-PBGD) is the more robust, fully first-order sibling that
estimates a lower solution via an inner loop and carries the main finite-time analysis.

## Problem it solves

First-order, scalable bilevel optimization with a finite-accuracy guarantee when the lower problem
`g(x,.)` is **not strongly convex** (so the implicit-gradient method has no Hessian to invert and
the solution set `S(x)` may be non-singleton), without the memory-grows-with-unroll-length cost of
iterative differentiation, and with an upper-level constraint `x in C`.

## Key idea

Abstract the lower-level feasibility `y in S(x)` as `d^2_{S(x)}(y) = 0` and penalize a smooth
surrogate `p` that upper-bounds that squared distance.

- **Squared-distance bound (SDB).** `p` is a `rho`-SDB if `p >= 0`, `rho p >= d^2_{S(x)}(y)`, and
  `p = 0` iff `d_{S(x)}(y) = 0`. This is the minimal property that makes the penalty principled.
- **The penalty relation.** If `p` is a `rho`-SDB and `f(x,.)` is `L`-Lipschitz, then `min f + gamma p`
  approximates the bilevel problem: the objective gap is `<= L sqrt(rho epsilon)` for a residual
  budget `p <= epsilon`; **global** solutions of `F_gamma` have `p = O(1/gamma)`; **local** solutions
  have `p = O(1/gamma^2)`, but only under extra local structure — convexity/star-convexity of `p`
  toward `S(x)`, or, for the gradient-norm penalty, a lower-bound `sigma_min(nabla_yy g) >= sigma > 0`
  at non-solution points.
- **PL certifies the penalty.** If `g(x,.)` satisfies the Polyak-Lojasiewicz inequality, the
  gradient-norm penalty `p = ||nabla_y g||^2` is a `mu`-SDB under the stronger `(1/sqrt(mu))`-PL
  condition (PL chained twice: `||nabla_y g||^2 >= (1/sqrt(mu))(g-v) >= (1/mu) d^2`); the value-gap
  penalty `p = g-v` is a `mu`-SDB under `(1/mu)`-PL (one chain). PL holds for over-parameterized nets
  and softmax-policy RL returns — the non-strongly-convex regime the implicit gradient cannot reach.
- **Why the naive penalty can fail.** Without the SDB / sigma condition, a local minimum of
  `f + gamma p` need not be a bilevel solution. Canonical instance:
  `min sin^2(y - 2*pi/3)` s.t. `y in argmin (y^2 + 2 sin^2 y)`; the only solution is `y = 0`, yet for
  every `gamma > 0` the penalty `f + gamma (y + sin 2y)^2` has a spurious stationary point at
  `y = 2*pi/3`, where `g'' = 0` (so `sigma = 0`, voiding the local bound) while `g'` is nonzero. The
  value-gap penalty excludes it without the sigma condition; the gradient-norm penalty can stall
  there — this is why it is the simpler but more fragile variant.
- **Finite, tight `gamma`.** To reach lower-level accuracy `delta` one needs `gamma = Theta(delta^{-0.5})`
  — sufficient and tight (witness `min y` s.t. `y in argmin y^2`, where the penalized solution is
  `-1/(2 gamma)`, residual `1/(4 gamma^2)`). A **finite** `gamma` suffices; no `gamma -> infinity`.

## Correct guarantee boundary (no unconditional KKT recovery)

The multiplier identity `w = gamma nabla_y g` makes the two penalty-stationarity equations
(`||nabla_x f + gamma nabla_xy g nabla_y g|| <= delta`, `||nabla_y f + gamma nabla_yy g nabla_y g|| <= delta`)
coincide with the **stationarity** blocks of the KKT conditions for the reformulation `nabla_y g = 0`,
with no constraint qualification (no CRCQ/LICQ) needed there. But that is only the stationarity
half. The **feasibility** block `||nabla_y g|| <= O(delta)` is *not* free from
`||nabla_yy g nabla_y g||` alone (the diagnostic above is exactly a point with small stationarity
residual but `nabla_y g != 0`). It follows only with `gamma = Omega(delta^{-0.5})`, which propagates
`||nabla_y f|| <= L`, `||nabla_yy g|| <= L_g` into `||nabla_y g|| = O((L+delta)/gamma) = O(delta^{0.5})`.
So a `delta`-stationary point of `F_gamma` is an `O(delta)`-stationary point of the bilevel problem
**when the residual is controlled** — a conditional guarantee, with the constraint-qualification-free
property holding on the stationarity blocks.

## Defaults and why

- **gamma ramped `0 -> gamma_max` over a fixed number of steps.** `F_gamma` is `L_gamma`-smooth with
  `L_gamma ~ L_f + gamma(2 L_g + L_g^2 mu)` growing linearly in `gamma`; a stable step needs
  `alpha <~ 1/L_gamma`, so a large `gamma` from the start forces a tiny step before `y` reaches the
  lower valley. Start as pure `f`-descent (`gamma_init = 0`) and phase in the stationarity penalty.
- **Step rescale `min(1/gamma, 1)`** (toy: `alpha = alpha0/gamma`). The penalty contributes
  `gamma nabla p`, which is `O(gamma)`; once `gamma > 1`, scale the whole joint step by `1/gamma` so
  the penalty step is `O(1)`, keeping the update inside the `alpha <= 1/L_gamma` stable regime.
- **The `1/2` in `(gamma/2)||nabla_y g||^2`.** Cancels the factor 2 from differentiating the square,
  so the penalty gradient is exactly `gamma nabla_yy g nabla_y g` (clean HVP).
- **Separate block step sizes `lrx, lry`.** `x` (per-example weights) and `y` (network parameters)
  have very different scale/curvature — one shared rate is wrong.
- **Projection on `x` only** (or, in data hyper-cleaning, the `sigmoid(x)` parametrization that keeps
  example weights in `(0,1)`); the inner variable `y` is unconstrained here.

## Final algorithm

```
input: f, g, upper constraint C (or feasible parametrization), step sizes (alpha_x, alpha_y),
       penalty cap gamma_max, warm-up length R, iterations K
gamma <- 0,  step_gamma <- gamma_max / R
for k = 1 .. K:
    q     <- nabla_y g(x, y)                 # retain graph for the HVPs
    loss  <- f(x, y) + (gamma/2) * ||q||^2
    scale <- min(1/gamma, 1)                 # implemented with epsilon at gamma = 0
    take one gradient step on scale * loss   # = scale*(nabla f + gamma nabla p), nabla p exact via HVP
    project x onto C (or keep via parametrization)
    gamma <- min(gamma_max, gamma + step_gamma)   # finite cap, no gamma -> infinity
```

## Working code

Filling the joint-update slot of the bilevel harness — data hyper-cleaning instance, grounded in the
canonical implementation (`G-PBGD/data_hyper_clean_gpbgd.py`). Upper variable `x` are cleaner logits
with weights `sigmoid(x) in (0,1)`; `y` is the classifier:

```python
import torch
import torch.nn.functional as F


def loss_F(tensors):
    """Squared L2 norm summed over a list of parameters / gradients."""
    return sum(torch.linalg.norm(w) ** 2 for w in tensors)


def g_pbgd_step(x, net, x_opt, y_opt, tr, val, gam, reg):
    """One joint G-PBGD update of (x = data-cleaner logits, y = net parameters)."""
    x_opt.zero_grad()
    y_opt.zero_grad()

    # upper objective f: clean validation loss
    fy = F.cross_entropy(net(val.data), val.clean_target)

    # lower objective g: sigmoid-weighted (corrupted) training loss (+ optional ridge)
    ce_tr = F.cross_entropy(net(tr.data), tr.dirty_target, reduction="none")
    gxy = (torch.sigmoid(x) * ce_tr).mean() + reg * loss_F(net.parameters())

    # lower-level gradient with graph retained -> a 2nd backward yields the HVP
    dgdy = torch.autograd.grad(gxy, net.parameters(), create_graph=True)

    # penalty p = ||grad_y g||^2; the 1/2 makes d(gam/2 * p) = gam * Hessian-vector.
    # once gamma > 1, rescale the whole joint step by 1/gamma so the penalty term
    # (gam * grad p) contributes an O(1) step rather than an O(gamma) one.
    lr_decay = min(1.0 / (gam + 1e-8), 1.0)
    loss = lr_decay * (fy + gam / 2.0 * loss_F(dgdy))

    loss.backward()        # grads on x (via sigmoid(x)->g->dgdy) and on net params y
    x_opt.step()           # joint GD step on (x, y) with separate block step sizes
    y_opt.step()


def train(x, net, hparams, tr, val):
    y_opt = torch.optim.SGD(net.parameters(), lr=hparams["lry"])
    x_opt = torch.optim.SGD([x], lr=hparams["lrx"])
    gam = hparams["gamma_init"]                                   # start at 0: pure f descent
    step_gam = (hparams["gamma_max"] - hparams["gamma_init"]) / hparams["gamma_argmax_step"]
    for k in range(hparams["outer_itr"]):
        g_pbgd_step(x, net, x_opt, y_opt, tr, val, gam, hparams["reg"])
        gam = min(hparams["gamma_max"], gam + step_gam)           # finite cap
    return x, net


# canonical hyper-cleaning knobs (linear classifier):
HYPERCLEAN_LINEAR = dict(lrx=0.3, lry=0.5, gamma_init=0.0, gamma_max=37.0,
                         gamma_argmax_step=5_000, outer_itr=40_000, reg=0.0)
# 2-layer MLP (784 -> 300 -> 10, sigmoid hidden):
HYPERCLEAN_MLP = dict(lrx=0.5, lry=0.5, gamma_init=0.0, gamma_max=37.0,
                      gamma_argmax_step=30_000, outer_itr=50_000, reg=0.0)
```

Toy / numerical-verification instance — the same joint projected-GD step with closed-form gradients
and an explicit box projection of `x` onto `[0,3]`:

```python
import numpy as np


def box(v, lo, hi):
    return min(max(v, lo), hi)


def g_pbgd_toy_step(x, y, alpha, gam, df, dpenalty, xlim=(0.0, 3.0)):
    """Joint projected-GD step on F_gamma = f + gamma * ||grad_y g||^2 (grad p exact)."""
    dF = df(x, y) + gam * dpenalty(x, y)          # d/d(x,y) [ f + gamma * penalty ]
    x_new = box(x - alpha * dF[0], *xlim)         # project upper variable onto C = [0,3]
    y_new = y - alpha * dF[1]                      # y unconstrained
    return x_new, y_new


def solve_toy(x0, y0, gam, df, dpenalty, alpha0=0.1, eps=1e-5):
    alpha = alpha0 / gam                           # alpha = alpha0/gamma (the 1/gamma rescale)
    x, y, k = x0, y0, 0
    while True:
        x_new, y_new = g_pbgd_toy_step(x, y, alpha, gam, df, dpenalty)
        pg = np.linalg.norm([(x - x_new) / alpha, (y - y_new) / alpha])  # projected-grad norm
        x, y, k = x_new, y_new, k + 1
        if pg <= eps:
            return x, y, k


# canonical toy knobs: gamma = 10, alpha0 = 0.1, x projected to [0, 3], 1000 random starts.
TOY_HPARAMS = dict(gams=(10.0,), alpha0=0.1)
```
