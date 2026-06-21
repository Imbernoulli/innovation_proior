# Context: first-order bilevel optimization without lower-level strong convexity (circa 2021-2022)

## Research question

A bilevel problem couples an upper objective to a lower one through an argmin:

```
minimize    f(x, y)
subject to  x in C,   y in S(x) := argmin_{u in U(x)} g(x, u).
```

`f` is the upper- (outer-) level objective, `g` the lower- (inner-) level objective. The pair
`(x, y)` is only admissible when `y` is an optimal solution of the lower problem for that `x`.
This pattern is everywhere in machine learning: hyperparameter optimization, data cleaning /
importance weighting, meta-learning, and adversarial / poisoning setups all instantiate it,
with `x` the hyper-variable (weights, importance scores, a shared representation) and `y` the
model trained for that `x`.

The defining feature is the moving feasible set `S(x)`: for every change in `x`, admissibility of
`y` is re-defined by a lower optimization problem, and `S(x)` drifts as `x` moves. The setting of
interest is a **first-order**, **scalable** method, usable when `y` is the parameters of a deep
network with millions of entries; a lower problem that is **not strongly convex**, so `S(x)` may be
a flat valley or a non-singleton set; an **upper-level constraint** `x in C` (a box, a norm ball);
and a **finite-accuracy** relation to the original bilevel problem. The broad question is how to
drive `y` toward a lower-level solution while descending `f` in `x`, using only first-order
information about `f` and `g`.

## Background

By this point gradient-based bilevel optimization is a well-developed area, with theory and
practical methods concentrated on the strongly-convex lower-level case.

**The strongly-convex lower level and the implicit gradient.** When `g(x, .)` is strongly
convex, `S(x) = {y*(x)}` is a singleton, and the problem reduces to minimizing the implicitly
defined function `f(x, y*(x))`. Its gradient comes from the lower-level stationarity identity
`nabla_y g(x, y*(x)) = 0`: differentiating it in `x` gives
`nabla_xy g + nabla_yy g (dy*/dx) = 0`, so `dy*/dx = -(nabla_yy g)^{-1} nabla_xy g^T`, and the
**hypergradient** is

```
df/dx = nabla_x f - nabla_xy g (nabla_yy g)^{-1} nabla_y f      evaluated at (x, y*(x)).
```

Ghadimi & Wang (2018) gave the first finite-time analysis for this implicit-gradient (IG) route;
Chen et al. (2021) sharpened it to match (stochastic) gradient descent rates. The construction
relies on `nabla_yy g` being invertible everywhere, i.e. on strong convexity.

**Iterative differentiation (ITD / RHG).** Another line replaces the exact solution `y*(x)` by the
output of an explicit inner algorithm — typically `T` steps of gradient descent
`omega_{t+1} = omega_t - beta nabla_y g(x, omega_t)` — and differentiates through the whole unrolled
trajectory. Reverse-mode hypergradient (RHG; Franceschi et al. 2017; Maclaurin et al. 2015)
backpropagates through all `T` inner steps; truncated RHG (T-RHG; Shaban et al. 2019)
backpropagates through only the last `K` steps. This needs no Hessian inverse and tolerates a
non-strongly-convex inner problem; the reverse pass stores or recomputes the inner iterates.

**Penalizing a lower-level optimality metric.** A single-level route adds a penalty that
measures how far `y` is from solving the lower problem and descends the sum. Two metrics are
natural: the squared **lower-level gradient norm** `||nabla_y g(x, y)||^2`, which vanishes exactly
at stationary points of `g(x, .)`; and the **function-value gap** `g(x, y) - v(x)`, with
`v(x) := min_y g(x, y)`, which vanishes exactly on `S(x)`. The classical penalty method minimizes
`f(x, y) + (gamma/2) ||nabla_y g(x, y)||^2` jointly over `(x, y)`, driving `gamma` up so the
penalty forces lower-level optimality. The gradient-norm penalty has an exactly-available gradient:
`nabla_y ||nabla_y g||^2 = 2 nabla_yy g nabla_y g` and
`nabla_x ||nabla_y g||^2 = 2 nabla_xy g nabla_y g` are Hessian-vector products, computable by one
extra backward through `nabla_y g` — no Hessian inverse, no inner unroll, no second variable to
track.

**A known cautionary instance for the penalty.** Consider the one-dimensional instance

```
minimize f(y) = sin^2(y - 2*pi/3)
subject to y in argmin g(y),   g(y) = y^2 + 2 sin^2 y.
```

The lower objective `g = y^2 + 2 sin^2 y` is non-convex but satisfies the Polyak-Lojasiewicz
inequality (it is the canonical `y^2 + c*sin^2 y` invex-but-non-convex PL example), and
`g'(y) = 2y + 2 sin 2y` has only the zero `y = 0`, so the only lower solution — hence the only
bilevel solution — is `y = 0`. Here `(g'(y))^2` is zero exactly on `argmin g`, so it is a valid
optimality metric. Yet for any penalty constant `gamma > 0`, the penalized objective
`f + gamma (y + sin 2y)^2` is stationary at `y = 2*pi/3`: there `f'(2*pi/3) = 0`, and the penalty
derivative vanishes because `g''(2*pi/3) = 2 + 4 cos(4*pi/3) = 0`, while `g'(2*pi/3)` is nonzero.
That point is not lower-feasible, yet gradient descent settles there.

**The Polyak-Lojasiewicz toolbox.** The PL inequality is a condition weaker than strong convexity
for non-convex lower problems: `g(x, .)` is `(1/mu)`-PL if
`||nabla_y g(x, y)||^2 >= (1/mu)(g(x, y) - v(x))` for all `y`. Karimi, Nutini & Schmidt (2016)
showed that for a Lipschitz-smooth `g`, PL is equivalent to an error bound and implies **quadratic
growth**: `g(x, y) - v(x) >= (1/mu) d^2_{S(x)}(y)`, where `d_{S(x)}(y)` is the distance from `y` to
the solution set. PL permits non-unique, non-convex solution sets, yet still forces every
stationary point to be a global minimizer. It holds for over-parameterized neural networks
(Liu et al. 2022) and for the softmax-policy discounted return in reinforcement learning
(Mei et al. 2020).

## Baselines

These are the prior methods a new bilevel method would be measured against and would react to.

**Implicit gradient / approximate inversion (Ghadimi & Wang 2018; Pedregosa 2016; Chen et al.
2021).** Compute `df/dx = nabla_x f - nabla_xy g (nabla_yy g)^{-1} nabla_y f` at the (assumed
unique) lower solution, using a linear solve or a truncated Neumann series for the inverse-Hessian
vector product, and descend in `x`. Finite-time rates matching (stochastic) GD are known. Applies
under lower-level strong convexity and a singleton `S(x)`.

**Reverse-mode hypergradient (RHG; Franceschi et al. 2017; Maclaurin et al. 2015) and truncated
RHG (T-RHG; Shaban et al. 2019).** Run `T` inner GD steps and reverse-differentiate through the
trajectory to get an approximate hypergradient; T-RHG truncates the reverse pass to the last `K`
steps. Tolerates non-strong-convexity of `g`. Memory and compute scale with the unroll length `T`
(the iterates are stored or recomputed).

**Classical / inversion-free penalty (Bertsekas; Mehra & Hamm 2021).** Reformulate the bilevel
problem as the single-level constrained problem `min_{x,y} f(x,y)` s.t. `nabla_y g(x,y) = 0` (valid
when the lower solution is unique), then apply the penalty method: minimize
`f(x, y) + (gamma_k/2)(Psi(||h(x,y)||) + ||nabla_y g(x, y)||^2)` alternately over `y` and `x`, with
`Psi` an exterior penalty for the upper constraints. Mehra & Hamm prove that as the penalty
constants `gamma_k -> infinity`, limit points satisfy the KKT conditions of the reformulation, under
a constraint qualification (LICQ) at the optimum. The method is inversion-free with
linear-in-dimension time and constant space.

**Value-function / log-barrier penalty (Liu et al. 2021).** Penalize the function-value gap
`g(x, y) - v(x)` via an interior-point / log-barrier scheme; handles non-convex lower levels.
Convergence is established asymptotically.

**First-order KKT methods (BOME; Ye et al. 2022).** Descend a combination of the upper gradient and
the value-gap gradient and prove convergence to a KKT point of the bilevel problem, under the
constant-rank constraint qualification (CRCQ) and uniform bounds on
`||nabla g||, ||nabla f||, |f|, |g|`.

## Evaluation settings

The yardsticks already in use at the time.

- **Non-convex toy bilevel for numerical verification.** A low-dimensional instance whose bilevel
  solution can be computed in closed form, run from many random starts to check that the method
  lands on the true solution, and swept over the penalty constant `gamma` to read off the empirical
  iteration-count and lower-level-accuracy dependence. A representative instance:
  `min f(x,y) = cos(4y+2)/(1+e^{2-4x}) + 0.5 ln((4x-2)^2 + 1)` s.t. `x in [0,3]`,
  `y in argmin g(x,y) = (x+y)^2 + x sin^2(x+y)`, whose lower solution is `S(x) = {-x}`, so the
  effective single-level objective is `f(x, -x)` on `x in [0,3]`. Protocol: project `x` to the box
  `[0,3]`; sample on the order of 1000 random initial `(x, y)`; iterate to a small
  projected-gradient norm; report iteration count and final lower-level residual versus `gamma`.
- **Data hyper-cleaning on MNIST** (Franceschi et al. 2017; Shaban et al. 2019). MNIST is split
  into 5000 train / 5000 validation / 10000 test; a fraction (50%) of the *training* labels are
  corrupted to uniformly random labels. The upper variable `x in R^N` assigns each training example
  an importance weight `omega^i(x) = sigmoid(x_i) in (0,1)`; the lower problem trains a classifier
  `y` on the weighted training loss; the upper objective is the clean validation loss. Models: a
  linear classifier `784 -> 10` and a 2-layer MLP `784 -> 300 -> 10` with a sigmoid hidden layer.
  Metrics: clean test accuracy, F1 of the recovered cleaner (does it down-weight the corrupted
  examples), and runtime / GPU memory to reach best accuracy.

## Code framework

The method plugs into the standard joint first-order bilevel training harness already used for the
baselines. The substrate that exists before the method is: a data/loss pipeline that exposes the
upper objective `f(x, y)` and the lower objective `g(x, y)`; ordinary optimizers (e.g. SGD) for the
two blocks `x` and `y`; autodiff that can produce `nabla_y g` with a retained graph (so a second
backward through it is possible); and a projection `Proj_C` of the upper variable onto its feasible
set (or a parametrization that keeps `x` feasible). The open slot is the per-iteration update rule
in `joint_step`: how `y` is driven toward a lower-level solution while `f` is descended in `x`. The
scaffold below carries exactly that one hole.

```python
import torch


def joint_step(state, hparams, grad_fns):
    """One joint update of the bilevel pair (x, y).

    state    : current iterate, e.g. {'x': Tensor, 'y': [...], 'gamma': float, 'k': int}
    hparams  : scalar knobs (block step sizes, penalty schedule, total iterations)
    grad_fns : callables exposing first-order (and graph-retaining) information --
               f(x, y) and its gradient, g(x, y) and nabla_y g(x, y) with a retained graph,
               and proj, the projection of x onto the upper-level feasible set C.

    Returns the updated state after one outer (or method-equivalent) step.
    """
    x, y = state['x'], state['y']

    # --- the contribution we will design ---------------------------------------
    # Given first-order access to f and g (and a retained graph for nabla_y g so a
    # second backward is available), form the joint update on (x, y) that drives
    # y toward a lower-level solution while descending f.
    # TODO: the update rule we will design.
    # ---------------------------------------------------------------------------
    pass


# existing joint training loop the update plugs into
def train(state, hparams, grad_fns):
    for k in range(hparams['outer_itr']):
        state = joint_step(state, hparams, grad_fns)     # apply the joint update rule
        state['x'] = grad_fns['proj'](state['x'])        # keep x in the feasible set C
    return state
```

The outer loop supplies first-order information about `f` and `g` each step and re-projects `x`;
`joint_step` is where the joint update rule will live.
