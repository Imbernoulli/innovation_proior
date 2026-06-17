I start from the constraint I cannot avoid: I want an adversarial image that changes as few spatial pixels as possible and still remains a valid image. In symbols, I want to minimize `||r||_0` subject to `k(x + r) != k(x)` and `l <= x + r <= u`. The `L_0` term is exactly the sparsity objective, but it makes the problem combinatorial. I cannot search all subsets of coordinates in an ImageNet-scale input, and the known `L_0` problem is NP-hard, so I need a tractable surrogate before I can design an attack.

The natural surrogate is `L_1`. Compressed sensing gives me the right instinct: under linear constraints, `L_1` is the convex relaxation that can recover sparse solutions under appropriate conditions. That means the real bottleneck is not the norm alone. I need to turn the label-change condition into a linear equality or inequality, then solve the resulting `L_1` problem while respecting the box.

DeepFool is the first tool to examine because it already converts a nonlinear classifier into local hyperplane projections. For a binary affine classifier `f(x) = w^T x + b`, the closest `L_2` boundary point is obtained by the orthogonal projection `r = -f(x) w / ||w||_2^2`. In the multiclass one-vs-all version, at the current iterate I form `w'_k = grad f_k - grad f_y` and `f'_k = f_k - f_y` for each competitor `k`, choose `l = argmin_k |f'_k| / ||w'_k||_2`, and step by `r_i = (|f'_l| / ||w'_l||_2^2) w'_l`. That is the dense `L_2` move I can use to find a nearby boundary.

The same projection calculation has a useful `L_p` form. Holder duality gives the dual exponent `q = p / (p - 1)`, so for `p > 1` the closest-class test uses `||w'_k||_q` and the update is `(|f'_l| / ||w'_l||_q^q) |w'_l|^{q-1} sign(w'_l)`. If I take the limiting `p = 1` case, then `q = infinity`. The distance denominator is `||w'||_infinity`, and the optimizer puts all of the correction on a coordinate `d` where `|w'_d|` is maximal: `r_d = |f'_l| / |w'_d| * sign(w'_d)`. This is the first hint that an `L_1` projection naturally produces sparse, almost one-hot moves.

I try that route and immediately run into the box. The plain `L_1` version can fool nearly all of the tested VGG-16 ImageNet examples while changing about `0.037%` of pixels, but when I clip the adversarial images to `[0, 255]`, the fooling rate drops to about `13%`. This failure is not incidental. Sparse perturbations concentrate large magnitude on a few coordinates, exactly the coordinates most likely to exceed the valid range. An end clip removes the high-magnitude evidence the attack relies on.

Clipping inside the iteration does not repair the underlying projection either. The `L_1` projection assumes that the selected coordinate can move freely until it closes the hyperplane gap. If the coordinate hits `l` or `u`, it can no longer contribute in that direction, and the linear solve that selected it is no longer solving the constrained problem. I need the box inside the optimization, not after it.

I then change the object I linearize. DeepFool linearizes the classifier values, but sparse `L_1` steps can move far along one coordinate, so the classifier linearization becomes stale quickly. What I need is a linear model of the decision boundary itself. The geometric observation that makes this plausible is that deep decision boundaries near natural images have low mean curvature. Near `x`, a boundary point found by a small `L_2` adversarial perturbation should sit on a surface that is locally close to a hyperplane.

So I let `r_adv` be the `L_2`-DeepFool perturbation from the current point, and I write `x_B = x + r_adv`. This identity matters later: `x_B - x = r_adv`. If `a = k(x_B)` is the adversarial class reached near the boundary and `y = k(x)` is the original class, the normal to the local boundary between those two classes is the gradient of the logit difference, `w = grad f_a(x_B) - grad f_y(x_B)`. With that orientation, points on the original side have negative local gap for `f_a - f_y`, and the boundary model is `w^T(z - x_B) = 0`.

The relaxed sparse problem now has the form `min_r ||r||_1` subject to `w^T(x + r - x_B) = 0` and `l <= x + r <= u`. Without the box, the `L_1` solution is one-hot. I choose `d = argmax_j |w_j|` and spend all mass on coordinate `d`. With the oriented normal above, the current gap `g = w^T(x - x_B)` is negative on the true-class side, so the update `r_d = |g| / |w_d| * sign(w_d)` gives `w_d r_d = |g|` and closes the gap. This is the sign convention used by the algorithm. If I used an arbitrary normal orientation I would have to write the generic `-g / w_d`, but here `w = grad f_adv - grad f_true` fixes the side and makes the `sign(w_d)` formula correct.

The box-constrained version becomes a greedy projection with retirement. I keep a set `S` of coordinates that can no longer help. At each inner step I choose the largest remaining `|w_j|`, compute the same one-coordinate correction from the current residual gap, update the image, and immediately project it into `[l, u]`. If the new point crosses the hyperplane, I am done. If not, the chosen coordinate must have saturated at a box boundary, so I add it to `S` and solve the residual gap with the next best coordinate. Each coordinate is used at most once in this inner solve, and every intermediate point is valid by construction.

The implementation uses a sign-crossing test instead of asking floating point arithmetic for exact equality. It stores `sign_true = sign(w^T(x - x_B))`, adds a small `beta = 0.001 * sign_true` to aim just beyond the plane, floors each scalar perturbation at `1e-4`, and loops while the current sign matches the initial sign. This matches the LTS4 and torchattacks coordinate solver: select `argmax(abs(coord_vec))`, set `r_i = clamp(pert, min=1e-4) * mask * sign(coord_vec)`, clamp the image, then zero out the used coordinate in `coord_vec`.

A single boundary hyperplane is still not enough. Low curvature is local, and the sparse projection can move the input away from the neighborhood where I estimated the plane. If I fit one hyperplane once and trust it globally, the projected point can miss the real decision boundary. The fix is to make the construction iterative: at the current iterate, run `L_2`-DeepFool again, rebuild `x_B` and `w`, run the box-aware `L_1` linear solver, and stop only when the current image changes label relative to the original label.

I also need to aim slightly past the estimated boundary. The boundary point from DeepFool is close to the surface, and curvature can make the exact local target too conservative. I therefore replace the target point by `x + lambda (x_B - x)`, with `lambda >= 1`. Since `x_B - x = r_adv`, this is identical to using `x + lambda r_adv`. The official LTS4 code folds this identity into the DeepFool subroutine by multiplying `r_tot` by `lambda_fac` before returning the boundary point used by the linear solver. The torchattacks version does the same at the caller by taking the DeepFool adversarial image and setting `adv_image = image + lam * (adv_image - image)`.

The trade-off from `lambda` is exactly the one I expect. Values close to `1` aim near the boundary and tend to preserve sparsity, but they may lower the fooling rate and require more outer iterations. Larger values push farther across the estimated boundary, often reducing iterations and improving success, but they can spend more coordinates. This is why `lambda` is the real control parameter, with typical values such as `1` for MNIST and `3` for CIFAR-10 or the common implementation default.

The normal computation has to stay tied to the adversarial-vs-true boundary. The mathematical update is `grad f_{k(x_B)}(x_B) - grad f_{k(x)}(x_B)`. When I translate this to the LTS4 implementation, DeepFool computes `(fs[0, k_i] - fs[0, label]).backward()` at the landing point and normalizes that gradient before scaling `r_tot` by `lambda`. In torchattacks, after the lambda-scaled point is formed, the implementation computes `cost = fs[pre] - fs[label]` and differentiates that cost. Both paths keep the same oriented boundary normal: adversarial logit minus original-label logit.

Finally, after the outer loop obtains a candidate sparse point, I return the DeepFool-style overshot image `x_0 + (1 + epsilon)(x_i - x_0)` and clamp it to the valid range. This final overshoot is separate from `lambda`: `lambda` moves the local hyperplane target during each solve, while `epsilon` nudges the accumulated perturbation across the actual classifier boundary before returning the adversarial image.

So let me put the whole construction into the code I would actually run, filling the empty slot: an inner `_linear_solver` that does the box-aware coordinate-greedy `L_1` projection, a `_deepfool_l2` subroutine that hands back the boundary point and its oriented normal, and an outer loop that relinearizes and applies `lambda` and the final overshoot.

```python
import numpy as np
import torch
import torch.nn as nn


def _deepfool_l2(model, x, num_classes, overshoot=0.02, max_iter=50):
    """L2-DeepFool: returns the oriented unit boundary normal and the boundary point x_B.
    Iteratively linearize the classifier, project to the closest competing class in L2,
    overshoot to cross, then read off the normal of (f_adv - f_true) at the landing point."""
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
            pert_k = f_k.abs() / w_k.norm()              # |f'_k| / ||w'_k||_2
            if pert_k < best_pert:
                best_pert, best_w = pert_k, w_k

        r_i = torch.clamp(best_pert, min=1e-4) * best_w / best_w.norm()   # L2 projection step
        r_tot = r_tot + r_i
        pert_x = pert_x + r_i
        check = x + (1 + overshoot) * r_tot              # have we crossed?
        k_i = model(check).argmax().item()
        it += 1

    xv = pert_x.clone().detach().requires_grad_(True)    # normal of (f_adv - f_true) at landing
    fs = model(xv).flatten()
    (fs[k_i] - fs[label]).backward()
    w = xv.grad.clone()
    w = w / w.norm()
    return w, x + r_tot                                  # x + r_tot is x_B; x_B - x = r_adv


def _linear_solver(x_0, normal, x_B, lb, ub):
    """min ||r||_1 s.t. w^T(x_0 + r - x_B) = 0, lb <= x_0 + r <= ub; greedy by |w_j|."""
    shape = x_0.size()
    w = normal.clone().detach()
    plane_normal = w.view(-1)
    plane_point = x_B.clone().detach().view(-1)
    x_i = x_0.clone().detach()

    gap = torch.dot(plane_normal, x_0.view(-1) - plane_point)   # signed gap to the plane
    sign_true = gap.sign().item()
    beta = 0.001 * sign_true                              # tiny overshoot to cross, not kiss
    current_sign = sign_true

    while current_sign == sign_true and w.nonzero().size(0) > 0:
        gap = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta
        pert = gap.abs() / w.abs().max()                 # L1 mass needed on the max coord
        mask = torch.zeros_like(w)
        mask[np.unravel_index(torch.argmax(w.abs()).cpu(), shape)] = 1.0   # d = argmax|w_j|
        r_i = torch.clamp(pert, min=1e-4) * mask * w.sign()    # put it all on coordinate d
        x_i = torch.max(torch.min(x_i + r_i, ub), lb)    # clip into the box (native validity)
        current_sign = torch.dot(plane_normal, x_i.view(-1) - plane_point).sign().item()
        w[r_i != 0] = 0                                  # retire d: used / saturated
    return x_i


def sparsefool(model, x_0, lb, ub, lam=3.0, max_iter=20, overshoot=0.02, num_classes=10):
    """SparseFool: outer loop relinearizes the locally-flat boundary, inner solver does the
    box-aware sparse L1 projection. lam (>= 1) aims past the boundary; it is the only knob."""
    pred = model(x_0).argmax().item()
    x_i = x_0.clone().detach()
    fool_im = x_i.clone()
    fool_label = pred
    loops = 0
    while fool_label == pred and loops < max_iter:
        w, x_B = _deepfool_l2(model, x_i, num_classes)   # boundary point + oriented normal at x_i
        x_B = x_i + lam * (x_B - x_i)                     # x_B - x_i is r_adv; aim lam past it
        x_i = _linear_solver(x_i, w, x_B, lb, ub)        # one sparse, box-valid step
        fool_im = torch.max(torch.min(x_0 + (1 + overshoot) * (x_i - x_0), ub), lb)  # cross + clamp
        fool_label = model(fool_im).argmax().item()
        loops += 1
    return fool_im
```

Let me trace the chain back so nothing is decorative. I wanted minimal-`L_0` perturbations, but `L_0` is NP-hard, so I relaxed to the convex `L_1` surrogate, which recovers the sparse solution under standard conditions. The `L_1`-dual being `L_infinity` means projecting onto a hyperplane spends all mass on one coordinate — sparse for free — which is exactly `L_1`-DeepFool. But `L_1`-DeepFool linearizes the classifier and ignores the box, and because sparse perturbations concentrate large magnitude on few pixels, clipping to the valid range collapses the fooling rate from near-100% to 13%: validity has to be inside the optimization. The escape was to linearize the *boundary* instead of the classifier, justified by the low mean curvature of deep decision boundaries near data — one affine hyperplane through the `L_2`-DeepFool point `x_B` with oriented normal `w = grad f_adv - grad f_true`. The `L_1` problem under that hyperplane plus the box is solved by a coordinate-greedy projection that retires any coordinate that saturates against the box, so validity is native and sparsity is preserved. A single linearization fails because the flatness is only local and a sparse step leaves the neighborhood, so the outer loop relinearizes at each iterate until the label flips. The one parameter `lambda` aims past the estimated boundary to absorb the curvature, trading sparsity against fooling rate and complexity. The result is fast, scales to high-dimensional images, and returns valid, sparse adversarial examples.
