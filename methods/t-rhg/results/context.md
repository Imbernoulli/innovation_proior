## Research question

Many learning problems are naturally *nested*. An outer objective `f(w, λ)` is to be minimized over a
hyperparameter `λ ∈ R^N`, but it depends on `λ` only through the solution of an inner problem
`min_w g(w, λ)` over a parameter `w ∈ R^M`. Writing `ŵ*(λ)` for the inner solution, the upper-level
objective is `F(λ) = E_S[ f_S(ŵ*_S(λ), λ) ]`, a stochastic expectation over a context `S` (a task, a
minibatch). Crucially, `ŵ*(λ)` is almost never available in closed form: in practice it is the output of
a *prespecified iterative algorithm* — say `T` steps of gradient descent on `g` — run for a finite number
of steps. So `ŵ*(λ)` is genuinely defined as "where `T` steps of the inner optimizer land," not as an exact
argmin. This is deliberate: it lets `λ` account for, and even control, the imperfections and the dynamics
of the inner optimizer (its step size, its initialization).

To optimize `λ` by a first-order method we need the total derivative

```
d_λ f = ∇_λ f + ∇_λ ŵ*(λ) · ∇_{ŵ*} f      (the "hypergradient")
```

where `∇_λ f ∈ R^N` and `∇_{ŵ*} f ∈ R^M` are cheap (a stochastic first-order oracle gives them), but the
matrix-vector product `∇_λ ŵ*(λ) · ∇_{ŵ*} f` is the hard object: it couples the entire inner trajectory to
`λ`. The question is how to compute or usefully approximate this hypergradient when both `λ` and `w` are
high-dimensional — thousands of hyperparameters and thousands of parameters at once.

## Background

By this time gradient-based hyperparameter optimization and meta-learning have been recast as bilevel
optimization (Domke 2012; Maclaurin et al. 2015; Franceschi et al. 2017). The prevailing wisdom and the
load-bearing facts:

- **Black-box hyperparameter search does not scale in dimension.** Grid search and random search
  (Bergstra & Bengio 2012) and Bayesian optimization (Snoek et al. 2012) work for a handful of
  hyperparameters. For high-dimensional `λ` one needs the gradient `d_λ F`.

- **The inner optimizer is a dynamical system, and the hypergradient is its sensitivity.** Treat the inner
  iteration as `w_{t+1} = Ξ_{t+1}(w_t, λ)`, `w_0 = Ξ_0(λ)`, `ŵ* = w_T`. For gradient descent
  `Ξ_{t+1}(w_t, λ) = w_t - γ ∇_w g(w_t, λ)`. Unrolling the chain rule through this graph gives the exact
  hypergradient as a sum over the whole trajectory:

  ```
  d_λ f = ∇_λ f + Σ_{t=0}^{T} B_t A_{t+1} A_{t+2} ⋯ A_T ∇_{ŵ*} f
  ```

  with `A_{t+1} = ∇_{w_t} Ξ_{t+1}(w_t, λ)` and `B_{t+1} = ∇_λ Ξ_{t+1}(w_t, λ)`, `B_0 = d_λ Ξ_0(λ)`. For the
  GD map, `A_t = I - γ ∇²_{w} g(w_{t-1}, λ)` and `B_t = -γ ∇_{λ,w} g(w_{t-1}, λ)`. Each term in the sum is a
  product of Jacobians stretching from step `t` all the way to step `T`.

- **A contraction fact about the inner map.** If `g` is `α`-strongly convex and `β`-smooth in `w` and the
  step size obeys `γ ≤ 1/β`, then `0 ≼ I - γ ∇²_w g ≼ (1 - γα) I`, so `‖A_t‖ ≤ 1 - γα < 1`. Gradient
  descent on such a `g` converges linearly, `‖w_t - w*‖ ≤ ‖w_0 - w*‖ · (1 - γα)^t`. This is a standard
  property (Hazan et al. 2016).

- **Implicit differentiation gives a trajectory-free formula.** If the inner problem is solved
  *exactly* to a unique `w*(λ)`, the implicit function theorem (Larsen et al. 1996; Bengio 2000; Domke 2012)
  yields

  ```
  d_λ f = ∇_λ f - ∇_{λ,w} g · (∇_{w,w} g)^{-1} · ∇_{ŵ*} f
  ```

  with all derivatives at `(w*(λ), λ)`. This needs no trajectory in memory; it assumes the inner
  problem was solved to optimality and requires applying the inverse Hessian `(∇_{w,w} g)^{-1}`.

## Baselines

These are the prior methods for estimating the hypergradient.

**Reverse-mode differentiation / Reverse-HG (Maclaurin et al. 2015; Franceschi et al. 2017).** Compute the
trajectory sum by back-propagation through the unrolled inner optimization. Initialize
`α_T = ∇_{ŵ*} f`, `h_T = ∇_λ f`, then sweep backward:

```
h_{t-1} = h_t + B_t α_t,        α_{t-1} = A_t α_t,        d_λ f = h_{-1}.
```

This is structurally identical to back-propagation through time. Franceschi et al. (2017) derive it cleanly
from a Lagrangian: introduce multipliers `α_t` for each constraint `w_t = Ξ_t(w_{t-1}, λ)`, and the
stationarity conditions reproduce exactly this backward recursion. Cost: time `O(cT)` (one inner-step cost
`c` per backward step) and space `O(MT)` — every intermediate iterate `w_t` must be kept to evaluate
`A_t` and `B_t` on the way back.

**Forward-mode differentiation / Forward-HG (Franceschi et al. 2017).** Propagate the sensitivity matrix
`Z_t = ∇_λ w_t` forward alongside the inner iteration: `Z_0 = B_0`, `Z_{t+1} = Z_t A_{t+1} + B_{t+1}`, and
`d_λ f = Z_T ∇_{ŵ*} f + ∇_λ f`. No trajectory needs to be stored (each `w_t` can be overwritten). Cost: space
`O(MN)` (the matrix `Z_t`), time `O(cNT)` — `N` times slower than reverse-mode, since the cost scales with the
number of hyperparameters.

**Reversible-dynamics reverse-mode (Maclaurin et al. 2015).** Avoid storing the trajectory by *reconstructing*
it: run the backward pass while exactly reversing the inner SGD-with-momentum dynamics, recovering each `w_t`
on the fly. With exact arithmetic this gives `O(M)` storage. Finite precision introduces drift, so an
"information buffer" stores the discarded bits (about `log₂(1/γ)` per step) to maintain accuracy. The approach
is specific to the SGD-with-momentum update.

**Checkpointing (Hascoet & Araya-Polo 2006).** Store the inner state only every `√T` steps and recompute the
intervening segments forward during the backward pass. Space drops to `O(M√T)` with doubled computation time.

**Implicit differentiation with conjugate gradient (Domke 2012; Pedregosa 2016; Gould et al. 2016).** Use the
IFT formula above and approximate the inverse-Hessian-vector product `(∇_{w,w} g)^{-1} ∇_{ŵ*} f` by `K` steps
of conjugate gradient (only Hessian-vector products needed, never the full Hessian). Space `O(M)`. This
presumes the inner problem has reached its exact minimizer `w*`.

**One-step heuristics (Luketina et al. 2016; Finn et al. 2017; Baydin et al. 2018).** In learning-rate
adaptation and first-order MAML, practitioners back-propagate through the single most recent inner step
and use that as the hyperparameter gradient — cheap, `O(M)` memory. Promising empirical results have been
reported.

## Evaluation settings

The natural yardsticks already in use for gradient-based bilevel optimization:

- **A small deterministic toy bilevel problem** with `λ, w ∈ R²`: an outer objective with many saddle points
  and an inner quadratic `½ (w - λ)^T G (w - λ)` with `G = diag(1, ½)` (so `1`-smooth, `½`-strongly convex),
  with `ŵ*` defined as `T = 100` GD steps at `γ = 0.1` from a fixed start. Used to *visualize* the geometry of
  the approximate gradient — its angle against the true gradient, its norm, convergence trajectories — because
  here the exact hypergradient can also be computed for comparison. Metrics: number of outer steps to converge,
  residual, gradient norm on a log scale.

- **Data hyper-cleaning on MNIST (LeCun et al. 1998).** Train a classifier on a training split whose labels
  are corrupted at rate `½`, with a separate clean validation split and a held-out test split (e.g.
  5000 / 5000 / 10000). The inner objective is a per-example **weighted** cross-entropy plus an `L2`
  regularizer, `g(w, λ) = Σ_i σ(λ_i) · CE(w; x_i, y_i) + reg · ‖w‖²`, with one weight `λ_i ∈ R` per training
  example (`|λ| = 5000`). The outer objective `f` is the cross-entropy on the clean validation set, which does
  **not** depend on `λ` directly (`∇_λ f = 0`). Optimizing `λ` should down-weight the corrupted examples.
  Models: a linear classifier (`784 → 10`) and a 2-layer MLP (`784 → 300 → 10`, sigmoid hidden). Metrics: test
  accuracy; an F1 score for identifying corrupted examples (threshold on `σ(λ_i)`); cleaner precision/recall.

- **Multi-task / task-interaction (Evgeniou et al. 2005) and one-shot classification on Omniglot (Lake et al.
  2015; cf. Finn et al. 2017).** Bilevel formulations where the inner problem learns task-specific models with a
  hyperparameter coupling matrix, or a meta-learned initialization/regularization for few-shot tasks; the latter
  is genuinely stochastic and has `∇_λ f ≠ 0`. Metrics: validation/test accuracy across tasks, run time per
  outer iteration, memory.

Protocol throughout: a fixed inner horizon `T` of gradient descent at step size `γ`; the upper-level variable
updated by a first-order method (SGD/Adam) with a decaying outer step size; comparisons read against the exact
trajectory hypergradient where it is computable.

## Code framework

The estimator plugs into a standard bilevel training harness. The harness is settled; what is *not* settled is
how to turn the inner optimization run and the cheap first-order oracles `∇_{ŵ*} f`, `∇_λ f` into a usable
`d_λ f`. So the substrate is only the generic machinery that already exists: the inner gradient-descent map `Ξ`
(one differentiable step), the inner forward loop, the outer loss, and the outer optimizer. The empty slot is the
hypergradient routine.

```python
import torch


def inner_step(params, hparams, lr_inner, inner_loss_fn):
    """One differentiable inner GD step: Ξ(w, λ) = w - γ ∇_w g(w, λ).
    create_graph=True so the step stays in the autograd graph if needed downstream."""
    loss = inner_loss_fn(params, hparams)                       # g(w, λ)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return [p - lr_inner * g for p, g in zip(params, grads)]


def hypergradient(trajectory, hparams, inner_step_fn, outer_loss_fn):
    """Turn the inner trajectory + the cheap first-order oracles ∇_{ŵ*}f and ∇_λ f
    into an estimate of d_λ f = ∇_λ f + ∇_λ ŵ* · ∇_{ŵ*} f, and write it into
    hparams[i].grad. How to combine the available objects is the open question."""
    # TODO: the hypergradient estimator we will design.
    pass


def bilevel_train(hparams, fresh_inner_params, inner_loss_fn, outer_loss_fn,
                  lr_inner, T, outer_opt):
    for outer_iter in range(NUM_OUTER):
        params = fresh_inner_params()          # re-initialize the inner problem
        trajectory = [params]
        w = params
        for t in range(T):                     # inner forward: T GD steps on g
            w = inner_step(w, hparams, lr_inner, inner_loss_fn)
            trajectory.append(w)
        outer_opt.zero_grad()
        hypergradient(trajectory, hparams, inner_step, outer_loss_fn)   # fills hparams.grad
        outer_opt.step()                       # one outer update on λ
```

The forward loop supplies the inner trajectory; `hypergradient` is where the estimator will live.
