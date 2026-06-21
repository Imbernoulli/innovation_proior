## Research question

Many learning problems are nested. There is an inner objective `J(w, lambda)` that we minimize
over model weights `w` (the regularized training loss of a neural network, say), and an outer
objective `E` that we want small at the *result* of that inner minimization — typically the
error on a held-out validation set. The outer objective depends on a vector of
*hyperparameters* `lambda`: regularization coefficients, per-example weights, learning rates,
momentum factors, the entries of a task-interaction matrix, even whole synthetic training sets.
The goal is to choose `lambda` so that the model trained under it generalizes well:

```
min_lambda  f(lambda) = E(w_T(lambda)),   where  w_T(lambda) ≈ argmin_w J(w, lambda).
```

A central concern is the dimensionality of `lambda`. Grid and random search, and the Bayesian /
sequential-model-based optimization methods built on top of them, treat `f` as a black box and
explore the hyperparameter space by repeatedly *retraining* and evaluating, each query costing a
full inner optimization. These methods are applied with tens to a few hundred hyperparameters.
Modern problems can pose thousands or millions: one weight per training example to down-weight
mislabeled data, one regularization strength per parameter, a full learning-rate schedule, a
dense task-relatedness matrix. For those, an approach is to obtain the *gradient* of `f` with
respect to `lambda` and descend it. The question is: how do you compute `df/dlambda` when
`w_T(lambda)` is not a closed-form function of `lambda` but the output of an iterative optimizer,
and when `lambda` may be enormous?

## Background

The field state is that black-box hyperparameter search dominates practice. Grid search is the
default; random search (Bergstra & Bengio 2012) is a strict improvement because it does not
waste queries on dimensions that do not matter. Bayesian optimization (Snoek et al. 2012;
Swersky et al. 2013; Snoek et al. 2015) and sequential model-based optimization with random
forests (Hutter et al. 2011) or tree-Parzen estimators (Bergstra et al. 2011, 2013) build a
surrogate of the response surface and query it intelligently, reaching a few hundred
hyperparameters (Bergstra et al. 2013). All of these share one structural ceiling: the
information each completed training run yields about `lambda` is essentially *one scalar* (the
validation loss), so the cost of search grows with the dimension of `lambda`.

The alternative line treats the training procedure as differentiable. Two classical facts about
computing derivatives sit underneath everything here. First, **algorithmic differentiation**
(Griewank & Walther 2008; Baydin et al. 2015) gives two dual ways to differentiate any
composition of smooth maps. For a differentiable `F: R^n → R^p` evaluated in time `c`: a
Jacobian-times-vector product `J_F r` costs `O(c)` in *forward mode*, and a
vector-times-Jacobian product `q^T J_F` costs `O(c)` in *reverse mode*. The whole Jacobian
costs `O(n·c)` in forward mode and `O(p·c)` in reverse mode — so the cheap direction depends on
the shape `n` vs `p`. Crucially, neither product ever needs the Jacobian *matrix* to be
materialized. Second, this exact forward/reverse duality already appeared in the recurrent
neural network literature: gradients of an RNN over time can be computed forward, while it runs,
by real-time recurrent learning (RTRL; Williams & Zipser 1989), propagating a derivative matrix
alongside the hidden state; or backward, after it runs, by back-propagation through time (BPTT;
Werbos 1990) — see Pearlmutter (1995) for the survey of the two. The Lagrangian derivation of
back-propagation itself (LeCun 1988) frames a layered/iterated computation as a constrained
optimization whose multipliers carry the gradient information backward.

A learning procedure by stochastic gradient descent (or momentum, RMSProp, Adam) is precisely
such an iterated computation. Write its state as `s_t` (weights, plus accessory variables like
velocities), evolving as

```
s_t = Phi_t(s_{t-1}, lambda),   t = 1, ..., T,
```

where `Phi_t` is the smooth map performed by the `t`-th optimizer step on minibatch `t`, and
`lambda` enters both directly (through the objective, e.g. a regularization weight or a learning
rate) and indirectly (through `s_{t-1}`). For gradient descent with momentum, for instance,
`s_t = (v_t, w_t)` with `v_t = mu·v_{t-1} + ∇J_t(w_{t-1})` and `w_t = w_{t-1} − eta·(...)`, so
`lambda = (mu, eta)`. The response function `f(lambda) = E(s_T(lambda))` is a known, smooth
composition of `T` known smooth maps — exactly the object algorithmic differentiation was built
for. A useful sanity-check identity follows from one application of the chain rule: defining the
per-step Jacobians

```
A_t = ∂Phi_t/∂s_{t-1}  (d×d),     B_t = ∂Phi_t/∂lambda  (d×m),
```

the total derivative `Z_t = ds_t/dlambda` obeys the recursion `Z_t = A_t Z_{t-1} + B_t` with
`Z_0 = 0`, and unrolling it gives the hypergradient in closed (if unwieldy) form,

```
∇f(lambda) = ∇E(s_T) · Z_T = ∇E(s_T) · Σ_{t=1}^T ( Π_{s=t+1}^T A_s ) B_t.
```

This expression is the shared ground: every gradient-based method below is, in the end, some way
of evaluating it. The difficulty is purely computational — `A_t` is `d×d` with `d` in the
millions, the product runs over all `T` steps, and how you organize the arithmetic decides
whether it costs `O(T)` or `O(T·m)` time and `O(1)` or `O(T)` memory in the size of the model.

A second, separate observation about the design space concerns what happens when the inner
problem is solved *exactly*. If `w*(lambda) = argmin_w J(w, lambda)` is an isolated minimizer
and `J` is twice differentiable and strongly convex in `w` there, the implicit function theorem
applied to the stationarity condition `∇_w J(w*, lambda) = 0` gives the hypergradient without
any reference to the trajectory:

```
∇f(lambda) = ∇_lambda E − ∇_{lambda,w}J · (∇_{w,w}J)^{-1} · ∇_w E,
```

all derivatives at `(w*(lambda), lambda)`. This is exact at the optimum but presupposes one has
reached it, and it requires working with the inverse of the `d×d` training Hessian.

## Baselines

These are the prior gradient-based hyperparameter methods a new approach would be measured
against and would react to.

**Differentiating a fixed number of optimization steps (Domke 2012).** Rather than wait for the
inner problem to be solved, run the iterative optimizer (gradient descent, momentum) for a fixed
number of steps and compute the gradient of the validation error of that fixed-length procedure
by a back-optimization algorithm. Core idea: accept that the minimizer is only approximate, and
differentiate the *approximation you actually computed*.

**Differentiating an exactly reversed training run (Maclaurin, Duvenaud & Adams 2015).** Treat a
full training run as a chain of elementary operations and propagate the derivative of the
validation loss backward through all of them in a single sweep; by the standard fact about
reverse-mode differentiation this costs only about twice a forward run, *independent of the
number of hyperparameters*, which is what makes thousands of hyperparameters (per-pixel data
preprocessing, per-parameter `L2` penalties, full learning-rate schedules) tractable. The
obstacle they confront head-on is memory: a naive backward sweep needs every intermediate weight
vector `w_1, ..., w_T`, and for a large network the weight vector alone is on the order of a
gigabyte and is updated tens of thousands of times, so the trajectory cannot be stored. Their
solution is to avoid storing it by running the dynamics *backward*: for SGD with momentum,
`w_{t-1}` can be recovered from `w_t` and the velocity, so the trajectory is regenerated on the
fly during the reverse pass. The catch is finite precision — the momentum decay `gamma < 1`
multiplies the velocity each step and shifts bits off the bottom, so exact reversal is
impossible without bookkeeping; they recover exactness by storing the discarded low-order bits
in an "information buffer" via an integer divide/multiply trick, cutting memory by a factor of
~200 at `gamma = 0.9`.

**Implicit differentiation with an approximate gradient (Pedregosa 2016).** Use the
implicit-function-theorem expression for `∇f` above, but make it practical in two ways: solve
the inner problem only to a tolerance `epsilon` (so hyperparameters can be updated before the
weights have fully converged), and solve the resulting linear system with the training Hessian
approximately, by an iterative solver that needs only Hessian-vector products rather than the
explicit inverse. With a schedule of decreasing `epsilon`, the inexact hypergradient steps
converge. This sidesteps storing any trajectory — the memory is `O(d)`.

**Black-box search (random / Bayesian / SMBO).** Random search (Bergstra & Bengio 2012),
Bayesian optimization (Snoek et al. 2012), and SMBO (Hutter et al. 2011; Bergstra et al. 2011)
need no derivatives and make no smoothness assumptions, which is their strength.

## Evaluation settings

The natural yardsticks for a high-dimensional gradient-based hyperparameter method, all
pre-existing:

- **Data hyper-cleaning on MNIST.** Take a balanced subset of MNIST, split into a training set,
  a clean validation set, and a test set; corrupt the labels of a fraction of the *training*
  examples (e.g. half). Attach one hyperparameter to each training example weighting its
  contribution to the training loss, so `lambda` has one coordinate per training point (thousands
  of them). A linear softmax classifier and a small 2-layer MLP (`784 → 300 → 10`, sigmoid
  hidden layer) are the inner models. The hope is that the validation-driven outer optimization
  drives the weights of mislabeled examples toward zero. Metrics: test accuracy of the resulting
  cleaned model, and the `F1` score / precision / recall of the example-weights as a *detector*
  of which training labels were corrupted.
- **A small numerical bilevel problem.** A low-dimensional coupled inner/outer pair on a scalar
  upper variable `x` projected to an interval and a scalar lower variable `y`, run from many
  random initializations, used to read off convergence behavior (steps to a stationarity
  tolerance, residual, success rate, runtime) cleanly, without confounding from a large model.
- **Learning task interactions (multitask).** Learn a symmetric non-negative task-interaction
  matrix `C` (and a scalar `rho`) entering a multitask regularizer `Σ_{j,k} C_{j,k}‖w_j−w_k‖² +
  rho Σ_k ‖w_k‖²`, on CIFAR-10 / CIFAR-100 features — tens to thousands of hyperparameters under
  symmetry / non-negativity / `L1`-budget constraints.
- **Large-scale optimizer-hyperparameter tuning.** A deep multitask network (millions of
  parameters; tens of millions of state variables) where the hyperparameters include the
  learning rate, momentum, and an auxiliary-task weight — a setting where storing any trajectory
  is out of the question, used to stress the memory axis.
- Protocol: the outer (hyper-)optimization itself is run with an off-the-shelf optimizer (Adam)
  on the computed hypergradient, with constraints on `lambda` enforced by projection; the inner
  optimizer is plain (stochastic) gradient descent run for `T` steps.

## Code framework

A generic bilevel / hyperparameter-optimization harness is already available. The inner model,
its weighted training loss, the validation loss, and the inner optimizer (one differentiable
gradient-descent step) are all in hand; the outer loop computes a hypergradient and takes a
projected outer step on `lambda`. What is *not* settled — the one empty slot — is how to turn the
inner trajectory that was run and the validation loss into the gradient of the validation loss
with respect to `lambda`. The `hypergradient` stub is the one slot to be filled; everything around
it already exists.

```python
import torch
import torch.nn.functional as F


def inner_update(params, hparams, lr_inner):
    """One differentiable gradient-descent step on the lambda-weighted training loss.
    Returns the new parameter list as a function of `params` and `hparams` in the
    autograd graph (so the step itself can be differentiated)."""
    loss = weighted_train_loss(params, hparams)          # depends on hparams (e.g. per-example weights)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return [p - lr_inner * g for p, g in zip(params, grads)]


def outer_loss(params, hparams):
    """Validation objective evaluated at the inner iterate (E in min E(w_T(lambda)))."""
    return val_loss(params)


def hypergradient(params_history, hparams, update_maps, outer_loss):
    """Gradient of the validation objective w.r.t. the hyperparameters `hparams`,
    given the inner-optimization trajectory `params_history` (first to last) and the
    per-step update maps that produced it.

    params_history : list of inner iterates [w_0, w_1, ..., w_T] (or a suffix of it)
    hparams        : the outer variables lambda, each requires_grad=True
    update_maps    : the inner update maps applied along the trajectory
    outer_loss     : (params, hparams) -> validation scalar
    """
    # TODO: the hypergradient computation we will design — turn the trajectory and the
    #       validation loss into d/d lambda of the validation loss, without forming any
    #       d-by-d matrix, and decide what (if anything) of the trajectory must be kept.
    pass


def hyperopt_loop(hparams, T, lr_inner, hyper_lr, num_outer_steps, project):
    hyper_opt = torch.optim.Adam(hparams, lr=hyper_lr)
    for _ in range(num_outer_steps):
        params = fresh_params()                          # re-init the inner model
        history = [params]
        for _ in range(T):                               # run the inner optimizer for T steps
            params = inner_update(params, hparams, lr_inner)
            history.append(params)

        hyper_opt.zero_grad()
        hypergradient(history, hparams, [inner_update] * (len(history) - 1), outer_loss)
        hyper_opt.step()                                 # descend the hypergradient
        with torch.no_grad():
            for h in hparams:
                h.copy_(project(h))                      # enforce constraints on lambda
    return hparams
```

The outer loop runs the inner optimizer for `T` steps and hands the trajectory to
`hypergradient`; that function is where the rule for computing `df/dlambda` will live.
