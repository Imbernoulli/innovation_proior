# Context: first-order bilevel optimization without lower-level strong convexity (circa 2021-2022)

## Research question

A bilevel problem couples an outer objective to the solution set of an inner one:

```
min_{x,y}  f(x, y)    s.t.   x in C,   y in S(x) := argmin_{y in U(x)} g(x, y).
```

Here `f` is the upper-level objective, `g` the lower-level objective, `C` the upper-level
constraint set, and `U(x)` the lower-level feasible set. The difficulty is the coupling: the
admissible `y` are exactly the minimizers of `g(x, .)`, so as `x` moves, the whole set `S(x)`
moves with it. These problems sit under hyper-parameter optimization, meta-learning,
adversarial training, image reconstruction, and reinforcement-learning policy design, and they
are notoriously hard to solve.

The most mature theory handles the easy case. When `g(x, .)` is *strongly convex* and
unconstrained, `S(x)` is a single point `y*(x)`, the problem collapses to minimizing the implicit
single-level objective `f(x, y*(x))`, and its gradient is available in closed form. The question
is how to handle bilevel problems when the lower level is non-convex and possibly constrained,
using only first-order information.

## Background

By this time gradient-based bilevel optimization is a workhorse of machine learning, but the
theory is sharply split by what is assumed about the lower level.

**The implicit-gradient picture (strongly convex lower level).** When `g(x, .)` is strongly
convex and unconstrained, `S(x) = {y*(x)}` is a singleton and `y*(x)` is differentiable. The
implicit function theorem applied to the stationarity condition `grad_y g(x, y*(x)) = 0` gives
the *hypergradient* of the reduced objective `f(x, y*(x))`:

```
d/dx [ f(x, y*(x)) ] = grad_x f - grad_xy g(x, y*) [grad_yy g(x, y*)]^{-1} grad_y f.
```

This is exact and well understood, with finite-time complexity results once strong convexity is
in place.

**A diagnostic phenomenon: naive penalization gets stuck.** A tempting escape is to convert the
bilevel constraint into a penalty. Lower-level optimality `y in S(x)` is equivalent to some
optimality metric vanishing — e.g. `grad_y g(x, y) = 0`. So one could add a penalty term and
solve a single-level problem `min_{x,y} f(x, y) + gamma * (optimality metric)` by ordinary
gradient descent. The trouble is concrete and instructive. Take the one-variable instance

```
min_{x,y}  sin^2(y - 2*pi/3)    s.t.   y in argmin_{y in R} ( y^2 + 2 sin^2 y ),
```

whose only bilevel solution is `y* = 0`. Here `grad_y g = 2y + 2 sin(2y)`, which vanishes
exactly on `S(x)`, so `||grad_y g||^2 = 4(y + sin 2y)^2` is a legitimate optimality metric.
Penalize it: `min f(x, y) + 4*gamma*(y + sin 2y)^2`. For *every* `gamma > 0`, the point
`y = 2*pi/3` is a local minimizer of the penalized problem, yet it is neither a global nor even
a local solution of the original bilevel problem. Gradient descent on the naive penalty
converges to junk. This observed failure — penalize the gradient norm, descend, land at a
spurious point — is the empirical fact that any penalty-based method has to explain and avoid.

**The Polyak-Lojasiewicz toolkit.** A function `g(x, .)` satisfies the `1/mu`-PL inequality if

```
||grad_y g(x, y)||^2 >= (1/mu) ( g(x, y) - v(x) ),    v(x) := min_y g(x, y),
```

for all `y`. PL is strictly weaker than strong convexity — it permits non-convex landscapes and
flat valleys of minimizers (non-singleton `S(x)`) — yet it is enough to make gradient descent on
`g(x, .)` converge *linearly*, and it shows up in over-parameterized networks and in
policy-gradient objectives. Karimi, Nutini & Schmidt (2016) established the chain that makes PL
usable here: PL implies the *quadratic-growth* property `g(x, y) - v(x) >= (1/mu) d^2_{S(x)}(y)`
(where `d_{S(x)}(y)` is the distance from `y` to the minimizer set), and PL is equivalent to an
*error bound* `||grad_y g|| >= c * d_{S(x)}(y)`. So under PL, the function value gap, the
gradient norm, and the distance to the solution set are all mutually controlled.

**The value function and its gradient.** The lower-level value function `v(x) = min_y g(x, y)`
is central but awkward: it need not be smooth, and even when it is, `grad v(x)` is *not* in
general `grad_x g(x, y)` at an arbitrary `y`. The classical envelope/Danskin result says that at
the *optimizer* the implicit term drops out — `grad v(x) = grad_x g(x, y*(x))` — because the
chain-rule term `(dy*/dx)^T grad_y g(x, y*)` vanishes when `grad_y g(x, y*) = 0` at an
unconstrained lower-level minimum. Nouiehed et al. (2019, Lemma A.5) sharpened this for the
PL setting *without* requiring a unique minimizer: under PL and `L_g`-smoothness,
`grad v(x) = grad_x g(x, y*)` for *any* `y* in S(x)`, and `v` is `(L_g + L_g^2 mu)`-smooth.

**The calmness / exact-penalty lineage.** Treating `y in S(x)` as the constraint
`g(x, y) - v(x) <= 0` and penalizing it is the value-function reformulation (Outrata 1990;
Ye & Zhu 1995). Classical exact-penalty results for it (Jane Ye 1997) rely on a *calmness*
condition paired with 2-Holder continuity, which is hard to verify in practice. The open
question is what generic, checkable conditions on a penalty term make the penalized problem
faithfully represent the bilevel problem.

## Baselines

**Approximate implicit differentiation (AID/BSA; Ghadimi & Wang 2018; Pedregosa 2016;
Chen et al. 2021).** Use the hypergradient formula above, approximating the
Hessian-inverse-vector product (e.g. by a truncated Neumann series or conjugate gradient). Under
lower-level strong convexity this gives finite-time convergence, eventually almost matching plain
SGD.

**Iterative differentiation / reverse-hypergradient (RHG; Franceschi et al. 2017; truncated
T-RHG, Shaban et al. 2019).** Replace `S(x)` by the output of `T` steps of (projected) inner
gradient descent, `y_T(x)`, and differentiate the validation loss through the entire unrolled
trajectory — reverse-mode backprop through the optimizer iterates (or forward-mode). No explicit
Hessian inverse.

**Gradient-norm penalty (Mehra & Hamm 2021).** Penalize `p(x, y) = ||grad_y g(x, y)||^2` and
descend the single-level objective `f + gamma * p`. The penalty is exactly differentiable.
Its gradient is `grad p = 2 grad_yy g * grad_y g`, which requires second-order information.

**Log-barrier value-function method (Liu et al. 2021, BVFSM).** Reformulate via the
value-function constraint `g(x, y) - v(x) <= 0` and handle it with a log-barrier plus
regularization, yielding a sequence of single-level problems.

**Dynamic-barrier first-order method (BOME; Ye et al. 2022).** Also work from the value gap
`q(x, y) = g(x, y) - g*(x)`, estimating `g*(x)` by running inner gradient descent and taking the
last iterate, and combine `grad f` with `grad q` through a dynamic-barrier rule — a genuinely
first-order update with a *non-asymptotic* rate to a KKT point. The analysis assumes the
constant-rank constraint qualification (CRCQ) plus uniform boundedness of
`||grad g||, ||grad f||, |f|, |g|`.

## Evaluation settings

The natural yardsticks already in use:

- **A non-convex toy bilevel problem** with one upper and one lower variable, for verifying
  solution quality and the predicted complexity scaling. A standard instance has
  `f(x, y) = cos(4y + 2)/(1 + e^{2 - 4x}) + (1/2) ln((4x - 2)^2 + 1)` over `x in [0, 3]`, with a
  non-convex lower level such as `g(x, y) = (x + y)^2 + x sin^2(x + y)`, for which the lower-level
  solution map is the single line `S(x) = {-x}` so the bilevel objective reduces to a known
  one-dimensional curve. The protocol: run from many (e.g. 1000) random initial points
  `(x_1, y_1)` and inspect where the last iterates land; sweep the penalty constant `gamma` and
  measure the number of steps to a fixed stationarity tolerance `||G(x_K, y_K)||^2 <= 1e-4` and
  the resulting lower-level gap, to read off how iteration count and accuracy scale with `gamma`.

- **Data hyper-cleaning on MNIST** (Franceschi et al. 2017; Shaban et al. 2019). A fraction of
  training labels (e.g. 50%) is corrupted to uniformly random labels. The upper variable `x` is a
  per-training-example weight (through a sigmoid, `omega^i(x) = sigmoid(x_i)`); the lower variable
  `y` is a classifier trained on the weighted, corrupted training set; the upper objective is the
  classifier's loss on a *clean* validation set. The yardsticks: test accuracy on a clean test
  set, the F1 score of the learned cleaner (how well the weights separate clean from corrupted
  examples), and — to assess scalability — GPU memory and wall-clock runtime to best accuracy.
  Standard split: 5000 train / 5000 validation / 10000 test. Models: a linear classifier
  (`784 -> 10`) and a two-layer MLP (`784 -> 300 -> 10` with a sigmoid hidden layer). Reported
  over multiple independent runs with confidence margins.

## Code framework

The method plugs into a generic bilevel training harness that already exists. The pieces present
*before* the method are: a way to evaluate the upper and lower objectives and their gradients,
an inner optimizer that can take steps on `g(x, .)`, an outer optimizer that updates `x` (and
`y`) with a projection onto `C`, and an outer loop that schedules everything.

```python
import torch
import torch.nn.functional as F


def project_C(z):
    """Project the upper variable onto the feasible set C (identity if unconstrained)."""
    # ... existing projection for the problem at hand ...
    return z


def solve_bilevel(f, g, init_xy, alpha, K):
    """Generic outer loop. f, g expose values and gradients of the upper/lower objectives.
    The job is to drive (x, y) toward the bilevel solution using first-order info only."""
    x, y = init_xy
    for k in range(K):
        # --- form a first-order update direction for (x, y)
        #     that respects y in argmin_y g(x, y), without second-order oracles.
        #     The scaffold only fixes where the direction is consumed:
        #       x = project_C(x - alpha * dx); y = y - alpha * dy
        pass
    return x, y


def run_hyperclean(net, net_inner, x, data_tr, data_val, opt_x, opt_y, opt_inner, K):
    for k in range(K):
        net.train()
        # inner optimizer can take steps on the lower-level (weighted training) loss
        # outer optimizers opt_x, opt_y can update x and y from an assembled objective
        # assemble the first-order objective / direction in the open slot above and step
        pass
```

The outer loop hands over the objectives, the inner optimizer, and the projection; the update
direction encodes the bilevel constraint `y in S(x)`.
