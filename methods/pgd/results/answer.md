# Projected Gradient Descent (L_inf Attack)

PGD — Projected Gradient Descent — is the canonical strong white-box evasion attack. It solves
the inner maximization `max_{||delta||_inf <= eps} L(theta, x + delta, y)` by iterated
steepest ascent under the L_inf metric: start by adding a uniform perturbation from the eps-box
and clipping to the valid image range, repeatedly step `alpha` along the sign of the
input-gradient, and after every step project back onto the intersection of the L_inf ball and
`[0,1]`. It is the multi-step, random-start generalization of the fast gradient sign method,
and is the practical first-order adversary target for this threat model.

## Problem it solves

White-box, L_inf-bounded evasion: given full access to a classifier `f_theta` (parameters and
input-gradients), a clean input `x in [0,1]^d` with label `y`, and a budget `eps`, find
`x_adv` with `||x_adv - x||_inf <= eps` and `x_adv in [0,1]^d` that the model misclassifies.
Equivalently, maximize the loss over the eps-box as reliably as a first-order method can. The
inner loss surface is non-concave in `x`, so the goal is the strongest local maximizer, not a
closed-form solution.

## Key idea

1. **Steepest ascent under L_inf is `sign(grad)`.** Maximizing the linearized loss `g^T delta`
   over `||delta||_inf <= eps` decouples across coordinates; each is pushed to `eps*sign(g_i)`,
   giving the optimum `delta = eps*sign(g)` with gain `eps*||g||_1`. (Under L_2 the optimum
   would instead be `eps*g/||g||_2`; the constraint norm sets the step shape.) One full-budget
   such step is FGSM, but it trusts a single linearization across the whole budget.

2. **Iterate to follow the curved surface.** Take many small steps `alpha*sign(grad)`,
   re-evaluating the gradient at each iterate, instead of one big leap. Extra steps keep
   correcting stale-gradient error and generally strengthen the search up to diminishing returns.

3. **Project after each step.** The feasible set is the L_inf ball `||x'-x||_inf <= eps`
   intersected with `[0,1]^d`. The Euclidean projection onto an L_inf box is exactly
   coordinate-wise clipping (each 1-D squared distance to an interval is minimized at the
   nearest endpoint). With the image box included, coordinate `i` is projected onto
   `[max(x_i-eps,0), min(x_i+eps,1)]`; because these are same-axis boxes, this is implemented
   by clipping `delta = x' - x` to `[-eps, eps]`, then clipping `x + delta` to `[0,1]`.

4. **Random start (with optional restarts).** The gradient exactly at `x` is corrupted by sharp local
   curvature artifacts that mask the true ascent direction. Sample `u_i ~ Uniform(-eps, eps)`,
   set `x_0 = clip(x + u, 0, 1)`, and then run the projected iteration. If extra attack budget
   is available, repeat from several starts and keep the highest-loss result.

The landscape supports this choice: from many random starts the iterate's loss rises and plateaus,
and the plateau values concentrate tightly (no extreme outliers, even over very many restarts),
while the maxima themselves are distinct and nearly orthogonal — many local maxima of nearly
equal loss. This does not prove that no isolated better maximum exists, but it supports using
random-start projected iteration as the strongest practical first-order search target.

## Final algorithm

```
u_i   <- Uniform(-eps, eps) for each coordinate
x_0   <- clip(x + u, 0, 1)                                      # random-start perturbation
for t = 0 .. steps-1:
    g_t   <- grad_x L(theta, x_t, y)                            # white-box input-gradient
    x'    <- x_t + alpha * sign(g_t)                            # L_inf steepest-ascent step
    delta <- clip(x' - x, -eps, eps)                            # project onto L_inf eps-ball
    x_{t+1} <- clip(x + delta, 0, 1)                            # project onto [0,1]
return x_steps
```

Equivalently, `x_{t+1} = Pi_C( x_t + alpha*sign(∇_x L(theta, x_t, y)) )`, where
`C = {z: ||z-x||_inf <= eps, z in [0,1]^d}` — projected gradient *descent* on the negative loss.

## Hyperparameters and why

- **Step direction `sign(grad)`** — L_inf-steepest ascent (not raw gradient, which is L_2).
- **Step size `alpha`** — small enough not to overshoot the box every step (projection then
  wastes the step at the boundary), large enough that `steps*alpha` exceeds the box diameter
  `2*eps` so the iterate can cross the ball from any start and move along the boundary. A common
  rule is `alpha = 2.5*eps/steps` (total reach `2.5*eps`); with many steps a larger `alpha`
  such as `eps/4` is also fine (e.g. `steps=40` gives reach `10*eps`).
- **`steps`** — more steps generally strengthen the search up to diminishing returns; tens of
  steps in practice.
- **Random start magnitude `eps`** — sample each perturbation coordinate uniformly from
  `[-eps, eps]`, then clip to `[0,1]`; optional restarts sample basins.
- **Loss** — cross-entropy is the default; a logit-margin loss
  `max_{j != y} z_j - z_y` is the drop-in when the attack should optimize the decision
  boundary margin directly, optionally with a confidence offset. The iteration is unchanged
  otherwise.

## Relation to prior methods

- **FGSM** = one step of size `eps` from `x_0 = x`: `x + eps*sign(∇_x L)`. The single-step,
  no-restart, full-budget special case.
- **BIM / iterative FGSM** = the deterministic-start (`x_0 = x`, no random init), single-
  trajectory case of PGD; same sign step with the same coordinate clips.
- **R+FGSM** = a single FGSM step after one random pre-step; PGD folds the random start into
  the full iteration instead of a one-shot.

## Why this is the principled target (Danskin)

For robust training the defender solves `min_theta E[ max_{delta in S} L(theta, x+delta, y) ]`.
Let `phi(theta) = max_{delta in S} g(theta, delta)`, `g(theta,delta) = L(theta, x+delta, y)`,
`S` compact. Danskin's theorem: if each `g(.,delta)` is differentiable and `∇_theta g` is
continuous, `phi` is locally Lipschitz, directionally differentiable, with
`phi'(theta; d) = sup_{delta in S*(theta)} d^T ∇_theta g(theta, delta)`; if the maximizer is
unique, `∇phi(theta) = ∇_theta g(theta, delta*)`.

Descent corollary, with the sign explicit: in the unique-maximizer case, let
`h = ∇_theta g(theta, delta*)`. The direction `+h` gives
`phi'(theta; h) = ||h||_2^2 > 0`, so it is ascent for the max-function. The descent direction is
`d = -h`, which satisfies `phi'(theta; -h) = -h^T h = -||h||_2^2 < 0` whenever `h != 0`. If
several exact maximizers are active, the directional derivative is the supremum over their
gradients; the negative of one arbitrary active gradient is not guaranteed to be descent unless
the active gradients align. The nonsmooth safe version is to let `p` be the nonzero
minimum-norm element of the convex hull of the active gradients. Projection geometry gives
`p^T a >= ||p||_2^2` for every active gradient `a`, hence
`phi'(theta; -p) = sup_a (-p)^T a <= -||p||_2^2 < 0`. In the usual tie-free exact-max case, the
loss-gradient at the inner maximizer is the gradient robust training steps against.
ReLU/max-pool nondifferentiability and approximate local maximizers are practical caveats;
restricting to a neighborhood where the local max is global recovers the same conclusion
locally.

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_attack(
    model: nn.Module,
    images: torch.Tensor,   # (N, C, H, W) in [0, 1]
    labels: torch.Tensor,   # (N,)
    eps: float,             # L_inf budget
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    _ = n_classes
    model.eval()
    steps = 40
    alpha = eps / 4.0                       # steps*alpha = 10*eps >> 2*eps diameter

    x = images.clone().detach().to(device)       # ball is centered at the clean image
    labels = labels.clone().detach().to(device)

    # random-start perturbation, projected into [0,1]
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), labels)        # ascend this loss (pluggable)
        grad = torch.autograd.grad(loss, x_adv)[0]          # white-box input-gradient

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()             # L_inf steepest-ascent step
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)   # project onto L_inf ball
            x_adv = torch.clamp(x + delta, 0.0, 1.0)            # project onto the box intersection
        x_adv = x_adv.detach()

    return x_adv
```
