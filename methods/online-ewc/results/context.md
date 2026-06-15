## Research question

A network is shown a sequence of learning contexts — task 1, then task 2, then task 3, and so on — and
after each it must keep performing the earlier ones. Plain stochastic gradient descent on the new context
overwrites exactly the weights that mattered to the old contexts: this is *catastrophic forgetting*, and it
is the defining obstacle of continual learning. The regularisation family attacks it by adding a penalty
`R(theta)` to the per-step loss that discourages moving the weights that were important to past contexts,
while leaving the unimportant ones free. Two design choices fully specify such a method: the *importance
estimator* (how much each parameter mattered to each past context, computed once a context finishes) and the
*penalty form* (how those importances are turned into a per-step penalty during later training).

The hard constraint that sharpens the problem is **lifelong scale at fixed model capacity**. A method meant
for a genuine lifelong setting must have a per-step cost and a memory footprint that do **not** grow with the
number of contexts seen so far — an agent that lives through hundreds or thousands of contexts cannot store a
per-context anchor and a per-context importance map, nor pay an O(number-of-contexts) cost on every gradient
step. At the same time the model has a *fixed* number of parameters, so the protective constraints
accumulated from old contexts compete for the same finite capacity that new contexts need. A solution has to
retain old contexts, still leave room to learn new ones, and do both with memory and compute that stay constant
in the number of contexts. The prior art below achieves retention but trades away scalability and long-run
plasticity; closing that gap is the problem.

## Background

By this time the dominant principled story for the importance estimator is the **probabilistic / Bayesian**
view of sequential learning. Treat the weights `theta` as random; after observing the data of contexts
`1..k`, what we know about `theta` is the posterior `p(theta | T_{1:k})`. Because the contexts' data are
conditionally independent given `theta`, this posterior factorises and, crucially, can be built up
*recursively*:

```
p(theta | T_{1:k})  ∝  p(theta | T_{1:k-1}) · p(T_k | theta).
```

That is, the posterior after all `k` contexts equals the posterior after the first `k-1` (used as a prior)
times the likelihood of the `k`-th. The recursion is order-independent and makes approximate posterior
updating the natural lens for continual learning. Catastrophic forgetting, in this language, is what happens
when you collapse the posterior to a single point estimate and then move it freely: all the information about
which directions the old contexts pinned down is thrown away.

The posterior over a deep net's weights is intractable, so the standard tool is **Laplace's method**
(MacKay): approximate `-log p(theta | ·)` near its mode `theta*` by its second-order Taylor expansion. The
first-order term vanishes at the mode, leaving a quadratic whose curvature is the Hessian of the negative log
posterior. Near a well-fit optimum the data part of that Hessian is well-approximated by the **diagonal
Fisher information** `F`, which has three properties that make it the estimator of choice (Pascanu & Bengio
2013): (a) it equals the expected curvature of the negative log-likelihood near a minimum, so it is the right
local quadratic; (b) it is computable from *first-order* gradients alone, hence affordable on millions of
parameters; (c) it is positive semidefinite, so the induced penalty is a valid convex bowl. The diagonal of
`F` is `F_ii = E_x E_{y ~ p_theta(·|x)} [ (∂ log p_theta(y|x)/∂theta_i)^2 ]` — the model's *own* expected
squared score — and in practice it is estimated by running examples through the trained model and averaging
squared gradients of the log-likelihood. (A cheaper "empirical Fisher" variant uses the observed label
instead of an expectation over the model's predicted label distribution.) The full `F` is a `d×d` object and
infeasible to form for a large net, so a *diagonal* approximation (off-diagonals set to zero) is the
working compromise: it gives a per-parameter importance, one scalar per weight.

Two facts about the design space are load-bearing here. First, the *local* nature of Fisher: `F`
is computed at one optimum and only describes the loss surface in a small neighbourhood of it; a quadratic
penalty built from `F` at `theta*` is only a faithful proxy for the old context's loss while `theta` stays
near `theta*`. Second, a quadratic penalty `(1/2) F (theta - a)^2` is parameterised by two things, a
stiffness `F` and an anchor point `a`; both have to be chosen and stored for a penalty to be evaluable.
Quadratic algebra can sometimes combine penalty terms, but whether such a combination is meaningful depends
on which posterior object the quadratic is supposed to approximate.

It is also a documented phenomenon that on **fixed capacity** the accumulation of protective constraints
eventually backfires: once enough contexts have pinned down enough weights, the network is so constrained
that new contexts can no longer be learned — the regularisers "over-constrain" the parameters. So
indefinitely piling up undamped constraints is not only a memory problem; past a point it is a *learning*
problem.

## Baselines

These are the prior methods a new continual-regularisation method would be measured against and would react to.

**Elastic Weight Consolidation (EWC) — Kirkpatrick et al., PNAS 2017.** The foundational
regularisation-based method and the direct Bayesian instantiation of the story above. After finishing a
context `t`, store its optimum `theta*_t` and its diagonal Fisher `F_t`, then while learning later contexts
add a quadratic penalty pulling each weight back toward `theta*_t` with per-weight stiffness `F_{t,i}`:

```
L(theta) = L_new(theta) + Σ_t (lambda/2) Σ_i F_{t,i} (theta_i - theta*_{t,i})^2.
```

Each term is "a spring anchoring the weights to the old solution", stiff exactly where that context was
sensitive. It is derived as a diagonal-Laplace approximation of the sequential posterior, and it works:
important weights are slowed, unimportant ones stay free, and shared structure across contexts is reused.
**Gap:** the implementation keeps **a separate penalty term, with its own anchor
`theta*_t` and its own Fisher `F_t`, for every past context**, so both the storage and the per-step cost of
evaluating the penalty grow *linearly* in the number of contexts. In a long-lived setting that linear growth
is the binding constraint, and on fixed capacity the growing stack of springs also over-constrains the net so
later contexts learn poorly.

**The quadratic-penalty / recursive-Laplace analysis — Huszár 2017.** A close reading of EWC's own
derivation. It shows that the literal multi-context extension of EWC's two-context Laplace argument
has a consistency problem: later optima are already found under earlier penalties, so treating every stored
anchor as an independent constraint changes the effective weighting of old data and can bias learning toward
earlier contexts. **Gap:** this is a diagnosis of EWC's multi-anchor bookkeeping, not a complete lifelong
regulariser. It identifies that the existing stack is not the only Bayesian-looking object available and that
early-context weighting must be handled carefully, but it does not by itself decide how a fixed-capacity
learner should trade old constraints against new ones, nor how to revise a previous context's contribution
when contexts **recur** without keeping task-specific factors.

**Synaptic Intelligence (SI) — Zenke, Poole & Ganguli 2017.** A different importance estimator that
sidesteps the post-hoc Fisher computation. It accumulates importance *online along the training trajectory*:
as the weights move during a context, each parameter's running contribution to reducing the loss is tracked,
`omega_i ≈ Σ (-grad_i · Δtheta_i)`, then normalised by that parameter's total drift over the context,
`(theta_i - theta*_i)^2 + xi`, and used as the stiffness in the *same* quadratic-penalty form as EWC. SI
already maintains a **single** accumulated importance per parameter (so its penalty cost is constant in the
number of contexts) and a single anchor at the latest optimum. **Gap:** its importance
is a path integral that depends on the optimiser's actual trajectory and step sizes rather than on a local
curvature of the loss, so it lacks EWC's clean Bayesian-posterior interpretation; and on a capacity-limited
network its accumulated importance only ever grows, so old constraints can dominate later learning.

**Background tool — stochastic expectation propagation (Li, Hernández-Lobato & Turner, NeurIPS 2015).**
Expectation propagation maintains one approximating factor per data factor and, to update one, divides it out
of the product ("cavity"), refines, and multiplies back. This gives a clean remove/refit/add view of approximate
Bayesian updating, but a literal per-factor implementation is linear in the number of factors. Stochastic EP is
a related approximate-inference recipe for tying many factors into an aggregate approximation with fractional
cavity updates in natural-parameter space, developed for generic probabilistic models rather than for neural
continual-learning regularisers.

## Evaluation settings

The natural yardsticks already in use for continual-learning regularisation:

- **Split-MNIST (task-incremental).** MNIST digits split into 5 binary tasks (2 classes each), presented in
  sequence; a task label is available at test time so only the relevant output head is scored.
- **Permuted-MNIST (domain-incremental).** A sequence of contexts (commonly 10), each applying a different
  fixed random permutation to the input pixels of MNIST; the label set is shared, the input distribution
  shifts per context. This is the canonical forgetting stress test from the EWC literature.
- **Split-CIFAR100 (task-incremental).** CIFAR-100 split into 10 ten-way tasks learned in sequence, a harder
  vision setting.
- **Protocol.** Train on each context for a fixed budget with a standard stochastic optimiser, then freeze
  that context's data and move on; the importance estimator is invoked once per context boundary, the penalty
  is added at every training step of later contexts. The diagonal Fisher is typically estimated from a capped
  number of single-example passes over the just-finished context's data for speed. Sequential supervised
  learning of many handwritten alphabets (Omniglot) is the same kind of scalability test with many contexts.
- **Primary metric.** Average accuracy across all contexts after the whole sequence has been trained (higher
  is better) — it rewards retaining old contexts *and* learning new ones, the two objectives in tension.

## Code framework

The training harness already owns the data pipeline, the base network, the task loss, the optimiser, and the
outer loop that moves from one context to the next. The editable surface is deliberately small: one hook is
called after a context finishes and may return a per-parameter update for whatever state the harness stores;
the other hook is called at every training step and returns a scalar penalty to add to the task loss. The
meaning of that state, how the post-context hook should update it, and how the per-step hook should turn it
into a penalty are still blank.

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def estimate_importance(model, dataset, prev_params, device):
    """Called once, after training on a context finishes.

    Returns a dict {param_name: update_tensor} (same shapes as the params).
    May run forward/backward passes over `dataset`.
    `model.param_list` yields generators of (name, param) over regularised params.
    """
    # TODO: design the post-context per-parameter update.
    pass


def compute_regularization_loss(model, state_dict, prev_params_dict):
    """Called at every training step. Must be cheap.

    `state_dict` is the harness-owned per-parameter state;
    `prev_params_dict` are stored parameter snapshots from context boundaries.
    Returns a scalar tensor: the penalty added to the task loss.
    """
    # TODO: design the per-step penalty from the stored state and snapshots.
    pass


# existing per-context continual-learning training loop the hooks plug into
def train_continual(model, loss_fn, contexts, optimizer, device):
    state = {}
    prev_params = {}
    for dataset in contexts:                       # contexts arrive one at a time
        for inputs, targets in DataLoader(dataset, batch_size=128, shuffle=True):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, targets.to(device))
            loss = loss + compute_regularization_loss(model, state, prev_params)
            loss.backward()
            optimizer.step()

        update = estimate_importance(model, dataset, prev_params, device)
        state = apply_state_update(state, update)  # provided by the harness
        prev_params = {n.replace('.', '__'): p.detach().clone()
                       for gp in model.param_list for n, p in gp()}


def apply_state_update(state, update):
    """Harness-owned bookkeeping for the hook output."""
    # TODO: fixed by the surrounding benchmark harness.
    pass
```

The outer loop adds whatever `compute_regularization_loss` returns to the task loss every step, and after
each context calls `estimate_importance` once.
