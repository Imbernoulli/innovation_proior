I want an adversarial image that changes as few spatial pixels as possible while remaining a valid image: minimize $\|r\|_0$ subject to $k(x+r)\neq k(x)$ and $l \le x+r \le u$. This sparse regime is a genuinely different and very physical threat from the usual dense $L_2$/$L_\infty$ attacks — a few raindrops glinting on a STOP sign, a sticker on a lens, a handful of bright flowers in a crop field — a small set of locations changed a lot rather than a faint film over the whole frame. A usable method has to actually minimize the number of perturbed pixels, run fast and scale to ImageNet-sized inputs, and return an image whose pixels stay inside $[l,u]$. None of the available options meets all three at once. Exact $L_0$ minimization is combinatorial and NP-hard, so it is out of reach directly. The existing sparse attacks each fail in a specific way: JSMA scores pixels by a Jacobian saliency map and greedily perturbs them, but it searches over pairs of candidate pixels per step so its cost explodes with input dimension and it is impractical on ImageNet; the one-pixel attack uses differential evolution and needs an enormous number of model queries; greedy local search is likewise black-box, query-heavy, and scales poorly. All three also produce high-magnitude pixels that are perceptible and routinely leave the valid range.

The natural escape from $L_0$ is the convex $L_1$ relaxation: under linear constraints, minimizing $\|r\|_1$ recovers the sparsest solution under standard restricted-isometry conditions. So the plan is to turn the label-change condition into a linear constraint and solve the resulting box-constrained $L_1$ problem. DeepFool already converts a nonlinear classifier into local hyperplane projections, and it has a useful $L_p$ form: by Hölder duality the dual exponent is $q = p/(p-1)$, so the closest-class test uses $\|w'_k\|_q$ and the step puts mass according to that norm. In the limiting case $p=1$ we get $q=\infty$: the distance denominator becomes $\|w'\|_\infty$ and the optimizer places all of the correction on the single coordinate of largest $|w'_j|$. So an $L_1$ projection is naturally sparse — almost one-hot — for free. But trying this route directly hits a wall that turns out to be fundamental, not incidental. Plain $L_1$-DeepFool fools nearly 100% of tested VGG-16 ImageNet examples while changing about $0.037\%$ of pixels, yet clipping the resulting images to $[0,255]$ collapses the fooling rate to about $13\%$. Sparse perturbations concentrate large magnitude on a few coordinates — exactly the coordinates most likely to exceed the dynamic range — so an end-clip removes the very evidence the attack relies on. Folding the clip into the iteration does not repair it either, because the $L_1$ projection assumes the selected coordinate can move freely until it closes the hyperplane gap; once that coordinate saturates against $l$ or $u$ the linear solve that selected it is no longer solving the constrained problem. Validity has to live inside the optimization, not after it.

I propose SparseFool. The first design choice is to change the object I linearize. DeepFool linearizes the classifier values, but a sparse $L_1$ step can move far along one coordinate, so the classifier linearization goes stale almost immediately. What I need instead is a linear model of the decision boundary itself, and this is justified by a geometric fact about trained deep classifiers: near a natural data point the decision boundary has low mean curvature, so it is well approximated by a single hyperplane. I find a point on that boundary with $L_2$-DeepFool: let $r_\text{adv}$ be its perturbation from the current iterate and set $x_B = x + r_\text{adv}$, so that $x_B - x = r_\text{adv}$. If $a = k(x_B)$ is the adversarial class reached at the boundary and $y = k(x)$ is the original class, the normal to the local boundary between them is the gradient of the logit difference, $w = \nabla f_a(x_B) - \nabla f_y(x_B)$, and the boundary model is $w^\top(z - x_B) = 0$. The orientation matters: with $w = \nabla f_\text{adv} - \nabla f_\text{true}$ the true-class side has a negative local gap, which fixes the sign convention used in the solver.

The relaxed sparse problem is then
$$\min_r \|r\|_1 \quad \text{s.t.}\quad w^\top(x + r - x_B) = 0,\quad l \le x+r \le u.$$
Without the box, the $L_1$ solution is one-hot: choose $d = \arg\max_j |w_j|$ and spend all mass there. Writing the residual gap as $g = w^\top(x - x_B)$, which is negative on the true-class side, the update $r_d = |g|/|w_d|\cdot \mathrm{sign}(w_d)$ gives $w_d r_d = |g|$ and exactly closes the gap; the $\mathrm{sign}(w_d)$ form is correct precisely because the oriented normal pins down the side (an arbitrary orientation would need the generic $-g/w_d$). To respect the box I make this a greedy projection with retirement. I keep a set $S$ of coordinates that can no longer help. At each inner step I take the largest remaining $|w_j|$, compute the one-coordinate correction from the current residual gap, update the image, and immediately clip it into $[l,u]$. If the new point crosses the hyperplane, I am done; if not, the chosen coordinate must have saturated at a box boundary, so I retire it into $S$ and re-spend the residual gap on the next-best coordinate. Each coordinate is used at most once and every intermediate point is valid by construction — this is what makes validity native rather than a destructive end-clip, and it is the direct fix for the $100\% \to 13\%$ collapse. In the implementation I avoid asking floating point for exact hyperplane equality by using a sign-crossing test: store $\mathrm{sign\_true} = \mathrm{sign}(w^\top(x - x_B))$, add a tiny $\beta = 0.001\cdot\mathrm{sign\_true}$ to aim just beyond the plane rather than kiss it, floor each scalar perturbation at $10^{-4}$, and loop while the current sign still matches the initial sign.

A single hyperplane is not enough, because the low-curvature flatness is only *local* and a sparse step can move the input out of the neighborhood where I estimated the plane. So the construction is iterative: at the current iterate I rerun $L_2$-DeepFool, rebuild $x_B$ and $w$, run the box-aware $L_1$ solver, and stop only when the image's label flips relative to the original. I also need to aim slightly past the estimated boundary, since the DeepFool point sits close to a surface whose curvature makes the exact local target too conservative. I therefore replace the target with $x + \lambda(x_B - x) = x + \lambda r_\text{adv}$, $\lambda \ge 1$. This $\lambda$ is the real control knob and the only parameter: values near $1$ aim close to the boundary and keep the perturbation sparsest but lower the fooling rate and need more outer iterations, while larger $\lambda$ push farther across, often crossing in fewer (even one) iterations and improving success at the cost of spending more coordinates — typical settings are $\lambda = 1$ for MNIST and $\lambda = 3$ for CIFAR-10. Finally, after the outer loop yields a candidate, I return the DeepFool-style overshot image $x_0 + (1+\eta)(x_i - x_0)$ with $\eta = 0.02$ and clamp it; this $\eta$ overshoot is separate from $\lambda$ — $\lambda$ moves the local hyperplane target during each solve, while $\eta$ nudges the accumulated perturbation decisively across the actual classifier boundary rather than asymptotically approaching a curved surface. The $L_2$-DeepFool sub-routine itself iterates the dense closest-class projection: form $w'_k = \nabla f_k - \nabla f_\text{true}$ and $f'_k = f_k - f_\text{true}$, pick $l = \arg\min_k |f'_k|/\|w'_k\|_2$, step $r_i = (|f'_l|/\|w'_l\|_2^2)\, w'_l$, and overshoot by $1+\eta$ to cross.

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
