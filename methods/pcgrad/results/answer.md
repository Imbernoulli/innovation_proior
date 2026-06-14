# PCGrad (Projecting Conflicting Gradients), distilled

PCGrad is a model-agnostic "gradient surgery" step for multi-task learning. Whenever two
tasks' gradients conflict — negative inner product / cosine — it projects each task's gradient
onto the *normal plane* of the other, removing only the conflicting component while leaving
cooperative components intact. It operates purely on the gradients of the shared parameters, so
it drops in front of any base optimizer (SGD with momentum, Adam) and composes with any
architecture or loss-weighting scheme.

## Problem it solves

Joint training of a shared network on several tasks by descending the summed loss
`L = sum_i L_i` (step along `g = sum_i g_i`) often underperforms training the tasks
separately. The failure is geometric: when per-task gradients conflict, the summed gradient
cancels in the overlap, and when one gradient dominates in magnitude, it swamps the others —
and under high curvature a first-order step overestimates the gain on the dominating task and
underestimates the damage to the dominated one. PCGrad combines the per-task gradients so this
destructive interference is removed.

## The tragic triad

Conflict alone is harmless (averaging two opposing gradients still descends the sum). Harm
requires three conditions co-occurring:
- **(a) conflicting gradients**: `cos phi_ij < 0`, i.e. `g_i . g_j < 0`.
- **(b) dominating gradients**: large magnitude difference, measured by gradient magnitude
  similarity `Phi(g_i, g_j) = 2||g_i|| ||g_j|| / (||g_i||^2 + ||g_j||^2)` (= 1 when equal,
  -> 0 as they diverge).
- **(c) high curvature**: multi-task curvature
  `H(L; theta, theta') = integral_0^1 grad L(theta)^T grad^2 L(theta + a(theta'-theta)) grad L(theta) da`
  large, so the linear model mis-estimates the trade.

## Key idea (the update)

For task `i`, iterate over the other tasks `j` (sampled in **random order**); if the current
(running) `g_i^PC` conflicts with `g_j`, replace it by its projection onto `g_j`'s normal plane:

```
if g_i^PC . g_j < 0:
    g_i^PC  <-  g_i^PC - (g_i^PC . g_j / ||g_j||^2) g_j
```

If `g_i^PC . g_j >= 0`, leave it unchanged (preserve positive transfer). The applied update is
`Delta theta = sum_i g_i^PC`. Two tasks, both conflicting, give the closed form
`g^PC = g - (g_1 . g_2 / ||g_1||^2) g_1 - (g_1 . g_2 / ||g_2||^2) g_2`, equivalently
`g^PC = (1 - cos phi_12 / R) g_1 + (1 - cos phi_12 * R) g_2` with `R = ||g_1|| / ||g_2||` — when
`cos phi_12 < 0` both coefficients exceed 1, so each task's own direction is amplified.

## Algorithm

```
Require: model parameters theta, batch of tasks {T_k}
  g_k  <-  grad_theta L_k(theta)        for all k
  g_k^PC  <-  g_k                        for all k
  for T_i in batch:
      for T_j sampled uniformly from batch \ {T_i}, in random order:
          if g_i^PC . g_j < 0:
              g_i^PC  <-  g_i^PC - (g_i^PC . g_j / ||g_j||^2) g_j
  return  Delta theta = sum_i g_i^PC      # pass to any base optimizer (SGD-momentum, Adam)
```

## Why these choices

- **Direction, not magnitude.** Reweighting the losses (uncertainty weighting, GradNorm) cannot
  cancel a conflicting *direction* — `w_1 g_1 + w_2 g_2` still conflicts for any nonnegative
  weights. The cancellation is geometric, so the fix is geometric.
- **Project onto the normal plane (subtract the projection).** Removes exactly the component of
  `g_i` along `g_j` (the only part the inner product `g_i . g_j` sees), the minimal change that
  makes the new `g_i` orthogonal to `g_j`; keeps the orthogonal (non-conflicting) part. Closed
  form, no QP — unlike MGDA's min-norm QP or GEM's constrained QP.
- **Conditional on `cos < 0`.** When gradients cooperate, the shared component is helping;
  projecting it away would destroy positive transfer (the failure of unconditional cosine
  regularization).
- **Symmetric, simultaneous.** Every task's gradient is de-conflicted, since no task is
  privileged when learning all at once (unlike A-GEM, which projects only the current task in a
  sequential setting).
- **Random task order.** With more than two tasks the sequential projections are order-dependent
  (the running `g_i^PC` changes as it is projected against successive tasks); random shuffling
  makes PCGrad symmetric w.r.t. task order in expectation.

## Theory

**Theorem 1 (convex convergence).** `L_1, L_2` convex differentiable, `grad L` `L`-Lipschitz,
step `t <= 1/L`. For `cos phi_12 < 0` the quadratic upper bound plus the step condition gives
`L(theta^+) <= L(theta) - (1/2) t (1 - cos^2 phi_12) ||g||^2`. So PCGrad converges to either the
optimum `L(theta^*)` (where `g = 0`) or a point with `cos phi_12 = -1` (where the factor and
`g^PC` vanish). The latter requires exactly anti-parallel gradients, which stochastic minibatch
gradients avoid in practice. (Non-convex: `min_k ||g_k||^2 <= 2(L(theta_0) - L^*)/(K(1-alpha^2)t)`
if `cos phi_{12,k} >= alpha > -1`. With `n` tasks, the analogous descent statement needs
`cos(g, g^PC) >= 1/2` and `||g^PC|| <= ||g||`.)

**Theorem 2 (single-step improvement, non-convex).** `grad L` `L`-Lipschitz, curvature
lower-bounded `H(L; theta, theta^MT) >= ell ||g||^2`, `ell <= L`. With curvature bounding
measure `xi(g_1, g_2) = (1 - cos^2 phi_12) ||g_1 - g_2||^2 / ||g_1 + g_2||^2`, then
`L(theta^PCGrad) <= L(theta^MT)` under the sufficient conditions
- **(a)** `cos phi_12 <= -Phi(g_1, g_2)` (conflict + magnitude difference, fused via `Phi`),
- **(b)** `ell >= xi(g_1, g_2) L` (high curvature),
- **(c)** `t >= 2/(ell - xi(g_1, g_2) L)` (step large enough for curvature to bite; the
  denominator must be positive for a finite threshold).

These match the tragic-triad conditions. At `t = 2/(ell - xi L)`, the difference lower bound is
`t (1 - cos^2 phi_12)(||g_1||^2 + ||g_2||^2) >= 0`. The same algebra gives
the full sufficient-and-necessary characterization by setting
`A = (1/2)||g_1+g_2||^2 ell - (1/2)(1-cos^2 phi_12)||g_1-g_2||^2 L` and
`B = (||g_1||^2+||g_2||^2)cos^2 phi_12 + 2||g_1||||g_2||cos phi_12`: either
`-Phi <= cos phi_12 < 0`, `ell <= xi L`, and `0 < t <= B/A`, or
`cos phi_12 <= -Phi`, `ell >= xi L`, and `t >= B/A`. The displayed
`2/(ell - xi L)` condition is the cleaner sufficient large-step form for the second case, with
the step-size condition itself forcing a positive curvature margin.

**Theorem (heavy-ball momentum).** Using `g^PC = (1 - cos phi_12/R) g_1 + (1 - cos phi_12 * R) g_2`
and the mean-value form `g^PC = H_k (theta_k - theta^*)`, with `L_i`-smooth `mu_i`-strongly-convex
tasks the eigenvalues of `H_k` lie in `[mu_k, L_k]`; the standard heavy-ball tuning
`alpha_k = 4/(sqrt(L_k) + sqrt(mu_k))`,
`beta_k = max{|1 - sqrt(alpha_k mu_k)|, |1 - sqrt(alpha_k L_k)|}^2` makes the companion matrix's
spectral norm `(sqrt(kappa_k) - 1)/(sqrt(kappa_k) + 1) < 1` (`kappa_k = L_k/mu_k`), so PCGrad
with momentum converges linearly unless `cos phi_12 = -1`.

## Working code

A wrapper around any base optimizer, de-conflicting per-task gradients before the optimizer steps:

```python
import copy
import numpy as np
import random
import torch


class PCGrad:
    """Project Conflicting Gradients. Wraps a base optimizer; de-conflicts per-task
    gradients of the shared parameters, then steps the base optimizer as usual."""

    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        """objectives: list of per-task scalar losses [L_1, ..., L_n]."""
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

    def _project_conflicting(self, grads, has_grads):
        shared = torch.stack(has_grads).prod(0).bool()    # params touched by every task
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)                         # random task order
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)             # conflict test: sign of inner product
                if g_i_g_j < 0:                           # cos(g_i, g_j) < 0
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)   # project onto g_j's normal plane
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction == 'mean':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')
        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)               # one backward per task
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:                        # task may not touch every param (multi-head)
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def _flatten_grad(self, grads, shapes):
        return torch.cat([g.flatten() for g in grads])

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
```

Two-task specialization (e.g. a fine head and a coarse head sharing a backbone), where the loop
reduces to a single symmetric projection:

```python
import torch


def pcgrad_two_task(g0, g1):
    """g0, g1: flattened gradients of the shared parameters for the two tasks.
    Returns the combined update Delta theta = g0^PC + g1^PC."""
    dot = torch.dot(g0, g1)
    if dot < 0:                                           # conflicting
        g0_proj = g0 - dot / (g1.norm() ** 2) * g1        # remove g0's component along g1
        g1_proj = g1 - dot / (g0.norm() ** 2) * g0        # remove g1's component along g0 (original g0)
        g0, g1 = g0_proj, g1_proj
    return g0 + g1
```

Usage: `opt = PCGrad(torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9))`; per step,
`opt.zero_grad(); opt.pc_backward([loss_fine, loss_coarse]); opt.step()`.
