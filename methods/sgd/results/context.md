# Context: minimizing an expected loss you can only see through noise

## Research question

We have a parameter vector `w` and a scalar objective that is an average over data,

```
F(w) = E_z[ Q(z, w) ],
```

the *expected risk* — the loss we would incur on a fresh example drawn from the data
distribution. In practice we cannot evaluate `F` or its gradient: the distribution of `z` is
unknown, and even when it is replaced by a finite sample of `n` examples, the empirical
average

```
R_n(w) = (1/n) sum_{i=1}^{n} Q(z_i, w)
```

is expensive to differentiate when `n` is large, because one gradient of `R_n` costs `n`
per-example gradients. So we are in a regime with two pressures at once. First, every quantity
we want — the value of the objective, its gradient — is only ever available *through noise*:
the best we can cheaply form is a gradient on a random handful of examples (or a single one),
which is a noisy, unbiased draw of the true gradient. Second, `n` is so large that we cannot
afford to touch all `n` examples per update; the binding resource is computing time, not the
number of examples on disk.

The precise goal is a method that drives `w` toward a minimizer of the *expected* risk `F`
while (1) paying only a small, `n`-independent cost per update; (2) using only first-order
information (no Hessian, no matrix the size of `w`-by-`w`); (3) tolerating gradients that are
genuinely noisy single samples rather than exact averages; and (4) coming with a guarantee
that the iterates actually approach the optimum despite that noise, and a principled rule for
how to set whatever step-size knob it exposes. The tension is that the cheapest possible
gradient — one example — is also the noisiest, so the open problem is whether descent can be
made to work at all when each step is taken along a direction we only trust *in expectation*.

## Background

The field state rests on three older bodies of work.

**Deterministic gradient descent (Cauchy).** For a smooth `F`, the method of steepest descent
takes `w_{t+1} = w_t - gamma * grad F(w_t)`. Under a Lipschitz-gradient assumption (the
gradient does not change faster than a constant `L`: `||grad F(w) - grad F(w')|| <= L ||w -
w'||`) one has the quadratic upper bound

```
F(w') <= F(w) + grad F(w)^T (w' - w) + (L/2) ||w' - w||^2,
```

so a small enough step strictly decreases `F`, and for strongly convex `F` (curvature bounded
below by `c > 0`) the error contracts geometrically — *linear* convergence, residual `~
exp(-t)`. The catch in the data setting: each step needs the full-batch gradient `(1/n) sum_i
grad Q(z_i, w)`, which costs `n` per-example gradients.

**Least squares and regression (Gauss, Legendre).** The oldest fitting principle: given pairs
`(x_i, y_i)`, choose parameters minimizing `sum_i (y_i - [beta_0 + beta_1 x_i])^2`. This both
assumes a *form* for how the response depends on `x` (here, linear) and processes the whole
sample at once. More abstractly, for a random response `Y` associated with an input `x`, the
conditional expectation `M(x) = E[Y | x]` is the *regression of Y on x*; least squares is the
special case where `M` is assumed linear and fit in closed form. The limitation that matters
later: it presumes a known functional form for `M` and is a batch computation.

**Stochastic approximation: root-finding from noisy measurements (Robbins; Kiefer &
Wolfowitz).** A different tradition starts from a function `M(x)` that is *unknown* and can
only be sampled through a noisy random response `Y(x)` with `M(x) = E[Y(x)]`, and asks for the
root `theta` of `M(x) = alpha`. One is not allowed to evaluate `M` directly — only to observe
one noisy `Y` at a chosen level `x`. The body of theory here studies recursions of the form

```
theta_n = theta_{n-1} - gamma_n [ h(theta_{n-1}) + noise_n ],
```

where `h` has the sought root and the noise is zero-mean, and asks: under what conditions on
the *step sizes* `gamma_n` do the iterates converge to the root despite the per-step noise?
The load-bearing diagnostic fact from this literature — the thing one must know before
designing anything — is a clean decomposition of the iterate error. For the simplest instance,
estimating the mean `x` of i.i.d. draws via `theta_n = theta_{n-1} - gamma_n(theta_{n-1} -
x_n)`, one can unroll the recursion to

```
theta_n - x =  prod_{k=1}^{n}(1 - gamma_k) * (theta_0 - x)                 [deterministic error]
            +  sum_{i=1}^{n} prod_{k=i+1}^{n}(1 - gamma_k) * gamma_i (x_i - x)   [random error],
```

and, taking second moments with `E[(x_i - x)^2] = sigma^2`,

```
E||theta_n - x||^2 = prod_{k=1}^{n}(1 - gamma_k)^2 ||theta_0 - x||^2
                   + sum_{i=1}^{n} gamma_i^2 prod_{k=i+1}^{n}(1 - gamma_k)^2 * sigma^2.
```

The two terms pull in opposite directions: the first (the memory of the starting point) only
vanishes if the step sizes do not decay too fast, while the second (the accumulated
measurement noise) only vanishes if they decay fast enough. The whole question is whether a
single step-size schedule can kill both at once. Two further pre-existing facts: the noisy
response can be averaged over a small group of `r` observations at the same level before
stepping, which scales the per-step noise variance by `1/r`; and a sibling construction
(Kiefer & Wolfowitz, 1952) handles the case where one cannot even observe a noisy *derivative*
— only noisy function *values* `Y(x +/- c_n)` — by forming a finite-difference estimate `(Y(x
+ c_n) - Y(x - c_n)) / (2 c_n)` and feeding that into the same kind of recursion, at the cost
of a bias that must be driven to zero as `c_n -> 0`.

The prevailing wisdom this sets up: descent and curvature theory is for *exact* gradients;
fitting is *batch* and assumes a model form; and noisy root-finding has step-size conditions
but had been developed for one-dimensional estimation problems, not framed as a way to
*optimize* a high-dimensional learning objective.

## Baselines

**Full-batch (deterministic) gradient descent.** `w_{t+1} = w_t - gamma (1/n) sum_i grad
Q(z_i, w_t)`. Core idea: follow the exact average gradient downhill; for strongly convex,
Lipschitz-smooth `F` it converges *linearly* (residual `~ rho^t`, `rho < 1`), and a
second-order variant scaling by an approximate inverse Hessian converges quadratically near
the optimum. **Gap:** the cost is `n` per-example gradients *per step*, and the iteration count
to a target accuracy is essentially independent of `n`; so the total work scales with `n`, and
when `n` is enormous the method spends an entire pass over the data to take a single step. It
optimizes the *empirical* risk `R_n`, not directly the expected risk `F`, so once `n` examples
are fixed it cannot benefit from more data without re-paying the per-step cost.

**Second-order / Newton-type steps.** Replace the scalar `gamma` by a matrix `Gamma_t`
approaching the inverse Hessian: `w_{t+1} = w_t - Gamma_t (1/n) sum_i grad Q`. Core idea: warp
the step by curvature for very fast (quadratic) local convergence — one iteration solves an
exact quadratic. **Gap:** still batch (cost `n` per step plus the curvature estimate), and it
forms/uses a `w`-by-`w` matrix, which is infeasible to store or invert when `w` is high
dimensional.

**Batch least-squares / closed-form fitting.** Solve `min_w sum_i (y_i - <w, x_i>)^2` (or its
regularized form) directly. Core idea: exact minimizer of a fixed empirical objective. **Gap:**
needs the whole sample at once, assumes a specific (here linear) model form, and yields a
single estimate rather than a procedure that improves online as data streams in.

**One-dimensional noisy root-finding recursions (the stochastic-approximation prior art).**
Recursions `theta_n = theta_{n-1} - gamma_n[h(theta_{n-1}) + noise_n]` with decaying `gamma_n`
were known to converge for scalar estimation under step-size conditions, and the
mean-estimation instance reduces to a running average. **Gap:** these were posed and analyzed
as *estimation* procedures for a scalar quantity from noisy scalar measurements; what they
delivered was a convergence theory and a feel for which step-size schedules work, not a
high-dimensional optimization method, and the conditions were stated for the abstract recursion
rather than connected to minimizing a learning objective.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **Convex empirical-risk minimization** — linear models with a convex per-example loss and an
  `L2` penalty, `R_n(w) = (lambda/2)||w||^2 + (1/n) sum_i loss(y_i <w, x_i>)`: logistic
  regression (log-loss), linear SVM (hinge loss), and the like. The regularizer makes the
  objective strongly convex, the setting in which convergence rates are cleanest. Metric:
  training objective (and held-out error) versus *amount of computation* — wall-clock or number
  of examples processed — not just versus iteration count, because per-iteration cost differs
  across methods.
- **Sparse high-dimensional regression / classification** — feature vectors in very high
  dimension with few nonzeros per example (bag-of-words text, `Lasso`-style `L1`-penalized
  least squares), where an update whose cost scales with the number of nonzero features is the
  point. A standard device is to write each weight as a difference of two nonnegative variables,
  `w = u - v`, and a stationary linear target is a `k`-sparse ground-truth vector to be
  recovered from `n` noisy examples; the quantity of interest is how small `n` can be.
- **Mean / quantile estimation and regression-function root-finding** — the one-dimensional
  testbeds where an iterate's mean-square error to the true value can be tracked exactly, used
  to study how the choice of step-size schedule controls the bias and variance terms.
- Protocol: identical initialization across methods; the step-size knob searched over a grid;
  for the noisy methods, several random seeds, since the iterate is itself a random variable.

## Code framework

The method plugs into a fixed optimizer harness for a diagonal-net sparse-recovery task. The
model, data generation, stopping rule, and gradient computation already exist: PyTorch autograd
computes full-batch MSE gradients with fresh Rademacher label noise at each training step. What
is not settled is the update rule applied to the two parameter vectors after those gradients are
available. The editable surface is therefore only a configuration function, an optimizer-state
initializer, and a `step()` function that maps `(u, v, grad_u, grad_v, state, hyperparameters)` to
the next `(u, v, state)` tuple.

```python
from typing import Any
import torch


def get_hyperparameters(
    dim: int,
    sparsity: int,
    delta: float,
) -> dict[str, Any]:
    """Return optimizer hyperparameters for this problem setting."""
    # TODO: the configuration for the update rule.
    pass


def init_state(
    u: torch.Tensor,
    v: torch.Tensor,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Initialize optimizer state from the model parameters u, v."""
    # TODO: whatever state the update rule needs.
    pass


def step(
    u: torch.Tensor,
    v: torch.Tensor,
    grad_u: torch.Tensor,
    grad_v: torch.Tensor,
    state: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Perform one optimizer step."""
    # TODO: given grad_u, grad_v, state, and hyperparameters,
    #       produce u_new, v_new, and the updated state.
    pass


# fixed training loop excerpt the update rule plugs into
def train(model, X_train, y_train, get_hyperparameters, init_state, step, delta):
    hparams = get_hyperparameters(model.dim, model.sparsity, delta)
    state = init_state(model.u, model.v, hparams)
    while not converged(model):
        model.zero_grad()
        noise = delta * (2 * torch.randint(0, 2, y_train.shape) - 1).float()
        y_noisy = y_train + noise                  # fresh label noise each step
        loss = 0.5 * torch.mean((model(X_train) - y_noisy) ** 2)
        loss.backward()                            # autograd fills grad_u, grad_v
        with torch.no_grad():
            u_new, v_new, state = step(model.u, model.v,
                                       model.u.grad, model.v.grad,
                                       state, hparams)
            model.u.data.copy_(u_new)
            model.v.data.copy_(v_new)
```

The loop hands `step()` one gradient tensor per parameter and writes back exactly the tensors it
returns; the single empty slot is the update rule itself.
