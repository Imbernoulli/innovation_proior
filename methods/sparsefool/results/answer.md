# SparseFool, distilled

SparseFool is a fast, geometry-inspired sparse (`L_0`) adversarial attack. It exploits the low
mean curvature of deep decision boundaries: near a data point the boundary is modeled by a single
affine hyperplane, and the sparse perturbation is computed as the box-constrained `L_1` projection
onto that hyperplane, solved one coordinate at a time so that pixel-validity is respected
natively. An outer loop relinearizes the boundary at each iterate to track its curvature.

## Problem it solves

Compute an adversarial perturbation that flips a classifier's label while changing as few spatial
pixels as possible (`min ||r||_0`) and keeping the image in the valid box `[l, u]`. Exact `L_0`
minimization is NP-hard; existing sparse attacks (JSMA, one-pixel, greedy local search) are slow,
scale poorly, and produce high-magnitude pixels that leave the valid range.

## Key idea

1. **Relax `L_0` to `L_1`.** Minimizing `||r||_1` under linear constraints is the convex surrogate
   for cardinality and recovers the sparsest solution under standard conditions. The `L_1`-dual
   `L_infinity` makes the projection onto a hyperplane *one-hot* — all mass on the single
   coordinate of largest `|w_j|` — which is sparse by construction.
2. **Linearize the decision boundary, not the classifier.** Trained deep classifiers have low
   mean curvature near data, so the boundary is locally well approximated by a hyperplane through
   the boundary point `x_B = x + r_adv` (found by `L_2`-DeepFool) with oriented normal
   `w = grad f_adv(x_B) - grad f_true(x_B)`, the gradient of `f_adv - f_true`.
3. **Solve the box-constrained `L_1` projection coordinate by coordinate.** Put the required `L_1`
   mass on the largest-`|w_j|` free coordinate, clip into the box, and *retire* any coordinate
   that saturates — then re-spend the residual gap on the next coordinate. Validity is inside the
   optimization, not a post-hoc clip. (Naively ignoring the box, as a plain `L_1` projection
   does, fools ~100% before clipping but only ~13% after, because sparse perturbations are
   high-magnitude per pixel.)
4. **Outer loop for curvature.** A single linearization fails when the sparse step leaves the
   locally-flat neighborhood, so SparseFool relinearizes the boundary at each new iterate until
   the label flips.

## The single knob

`lambda >= 1` aims at the point `x^(i) + lambda (x_B^(i) - x^(i))` — past the estimated boundary
— to absorb the curvature the linear model underestimates. `lambda` near 1: sparsest, lower
fooling rate, more iterations. Larger `lambda`: faster (even one-step) crossing, less sparse. It
is the only parameter (typical: `lambda = 1` for MNIST, `lambda = 3` for CIFAR-10). The final
crossing uses a DeepFool-style overshoot `1 + eta`, `eta = 0.02`.

## Final algorithm

```
LinearSolver(x_0, w, x_B, l, u):                  # box-constrained L1 projection
  S = {}; i = 0; x^(0) = x_0
  while w^T(x^(i) - x_B) has not changed sign and free coords remain:
    d = argmax_{j not in S} |w_j|
    r_d = |w^T(x^(i) - x_B)| / |w_d| * sign(w_d)   # all L1 mass on coordinate d
    x^(i+1) = clip(x^(i) + r, l, u)                # native validity
    S = S union {d}; i += 1
  return x^(i)

SparseFool(x_0, f, l, u, lambda>=1):
  x^(0) = x_0; i = 0
  while k(x^(i)) == k(x^(0)):
    r_adv = L2-DeepFool(x^(i))                     # x_B - x^(i)
    x_B = x^(i) + r_adv                            # boundary estimate
    w^(i) = grad f_{k(x_B)}(x_B) - grad f_{k(x^(i))}(x_B)
    target_B = x^(i) + lambda * r_adv              # same as x^(i)+lambda(x_B-x^(i))
    x^(i+1) = LinearSolver(x^(i), w^(i), target_B, l, u)
    i += 1
  return r = x^(i) - x^(0)
```

`L_2`-DeepFool, as the sub-routine, iterates: linearize the classifier, project to the closest
competing class with `r_i = (|f'_l| / ||w'_l||_2^2) w'_l` where `w'_k = grad f_k - grad f_true`,
`f'_k = f_k - f_true`, `l = argmin_k |f'_k| / ||w'_k||_2`, and overshoot by `1 + eta` to cross.
For the `L_p` extension, Holder duality gives `q = p / (p - 1)`; the closest-class denominator is
`||w'_k||_q`, and the `p = 1` limit uses `q = infinity`, giving the one-coordinate projection above.

## Working code

Faithful to the canonical implementation (an internal `L_2`-DeepFool plus a coordinate-greedy
box-constrained `L_1` solver), exposed through the harness's `run_attack` slot.

```python
import numpy as np
import torch
import torch.nn as nn


def _deepfool_l2(model, x, num_classes, overshoot=0.02, max_iter=50):
    """L2-DeepFool: returns the oriented unit normal and unscaled boundary point x_B."""
    x = x.clone().detach()
    logits = model(x).flatten()
    order = logits.argsort(descending=True)[:num_classes]
    label = order[0].item()

    pert_x = x.clone()
    r_tot = torch.zeros_like(x)
    k_i = label
    it = 0
    while k_i == label and it < max_iter:
        xv = pert_x.clone().detach().requires_grad_(True)
        fs = model(xv).flatten()
        fs[order[0]].backward(retain_graph=True)
        grad_true = xv.grad.clone()

        best_pert, best_w = torch.tensor(float("inf")), None
        for k in range(1, num_classes):                  # closest competing class in L2
            xv.grad.zero_()
            fs[order[k]].backward(retain_graph=True)
            w_k = xv.grad.clone() - grad_true
            f_k = (fs[order[k]] - fs[order[0]]).detach()
            pert_k = f_k.abs() / w_k.norm()
            if pert_k < best_pert:
                best_pert, best_w = pert_k, w_k

        r_i = torch.clamp(best_pert, min=1e-4) * best_w / best_w.norm()
        r_tot = r_tot + r_i
        pert_x = pert_x + r_i
        check = x + (1 + overshoot) * r_tot
        k_i = model(check).argmax().item()
        it += 1

    xv = pert_x.clone().detach().requires_grad_(True)    # normal of (f_adv - f_true) at landing
    fs = model(xv).flatten()
    (fs[k_i] - fs[label]).backward()
    w = xv.grad.clone()
    w = w / w.norm()
    return w, x + r_tot


def _linear_solver(x_0, normal, x_B, lb, ub):
    """min ||r||_1 s.t. w^T(x_0 + r - x_B) = 0, lb <= x_0 + r <= ub; greedy by |w_j|."""
    shape = x_0.size()
    w = normal.clone().detach()
    plane_normal = w.view(-1)
    plane_point = x_B.clone().detach().view(-1)
    x_i = x_0.clone().detach()

    gap = torch.dot(plane_normal, x_0.view(-1) - plane_point)
    sign_true = gap.sign().item()
    beta = 0.001 * sign_true                              # small overshoot to cross the plane
    current_sign = sign_true

    while current_sign == sign_true and w.nonzero().size(0) > 0:
        gap = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta
        pert = gap.abs() / w.abs().max()
        mask = torch.zeros_like(w)
        mask[np.unravel_index(torch.argmax(w.abs()).cpu(), shape)] = 1.0   # d = argmax|w_j|
        r_i = torch.clamp(pert, min=1e-4) * mask * w.sign()
        x_i = torch.max(torch.min(x_i + r_i, ub), lb)    # clip into the box
        current_sign = torch.dot(plane_normal, x_i.view(-1) - plane_point).sign().item()
        w[r_i != 0] = 0                                  # retire the used coordinate
    return x_i


def sparsefool(model, x_0, lb, ub, lam=3.0, max_iter=20, overshoot=0.02, num_classes=10):
    """SparseFool: outer relinearization + box-aware sparse L1 projection. lam is the only knob."""
    pred = model(x_0).argmax().item()
    x_i = x_0.clone().detach()
    fool_im = x_i.clone()
    fool_label = pred
    loops = 0
    while fool_label == pred and loops < max_iter:
        w, x_B = _deepfool_l2(model, x_i, num_classes)
        x_B = x_i + lam * (x_B - x_i)                     # x_B-x_i is r_adv; aim lam past it
        x_i = _linear_solver(x_i, w, x_B, lb, ub)
        fool_im = torch.max(torch.min(x_0 + (1 + overshoot) * (x_i - x_0), ub), lb)
        fool_label = model(fool_im).argmax().item()
        loops += 1
    return fool_im


def run_attack(model, images, labels, pixels, device, n_classes):
    model.eval()
    adv = []
    for i in range(len(images)):
        x = images[i : i + 1].to(device)
        lb = torch.zeros_like(x)
        ub = torch.ones_like(x)
        adv.append(sparsefool(model, x, lb, ub, lam=3.0, max_iter=20,
                              overshoot=0.02, num_classes=n_classes))
    return torch.cat(adv, dim=0)
```

## Why each piece

- **`L_1` over `L_0`:** convex, tractable, and its `L_infinity` dual makes the hyperplane
  projection one-hot, so sparsity comes for free.
- **Linearize the boundary, not `f`:** the boundary's low mean curvature near data makes one
  affine hyperplane a faithful local model and gives a clean box-constrained linear program;
  linearizing `f` (as `L_1`-DeepFool does) is accurate only right at the boundary and ignores the
  box.
- **Coordinate-greedy projection with retirement:** the box-aware linear solve used by the canonical
  implementation; saturated coordinates are dead, so re-spend the residual gap elsewhere;
  this is what makes validity native instead of a destructive end-clip.
- **Outer relinearization:** flatness is only local, so re-fit the hyperplane at each iterate.
- **`lambda`, overshoot, `min` clamps:** aim past the boundary and cross it decisively rather than
  asymptotically approaching a curved surface.
