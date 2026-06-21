The problem is to make adversarial robustness evaluation trustworthy. A classifier's robust accuracy is only as reliable as the attack used to measure it, and the standard projected-gradient-descent attack on cross-entropy has two structural weaknesses. First, its step size is a fixed hyperparameter that must be tuned per model: too large and it oscillates without refining the perturbation, too small and it cannot explore the ball, and in practice the loss plateaus long before the iteration budget is exhausted while the optimizer cannot tell whether it is still improving. Second, cross-entropy is scale-variant: multiplying all logits by the same positive constant leaves every decision unchanged, but it sharpens the softmax and drives the true-class probability toward one, which makes the input gradient vanish in finite precision. A defender can therefore blind a CE-PGD evaluation without making the model any more robust. Existing alternatives such as the CW margin are shift-invariant but still scale-fragile, while black-box attacks are slow and minimum-norm attacks are expensive for many classes. None of the standard tools is autonomous, parameter-free, and complete.

The method I propose is AutoAttack. It is not a single attack but a parameter-free robustness-evaluation protocol built around two new components. The first is Auto-PGD, or APGD, a budget-aware variant of PGD whose step size is controlled by the optimization trajectory rather than by a user knob. The second is the Difference-of-Logits-Ratio loss, or DLR, whose invariances match those of the argmax decision. These two pieces are combined with two complementary existing attacks, targeted FAB and Square Attack, into a small ensemble whose members fail for different reasons; a point is counted as robust only if none of the attacks finds a valid adversarial perturbation.

APGD starts from an aggressive step size eta equal to twice the perturbation budget eps. In the L-infinity case, a signed step of that size can cross the entire feasible box before projection, so the first iterate lands on an informative boundary location. After each projected step, APGD keeps track of the best objective value and the corresponding point seen so far, because the raw iterates are not monotone. At fixed checkpoints during the budget it checks two progress conditions. Condition one counts how many recent steps actually increased the objective; if fewer than a fraction rho of them did, the step size is too large and is halved. Condition two catches stalled progress: if the step size was not reduced at the previous checkpoint and the best objective has not improved, the step size is halved anyway. Whenever the step size is reduced, the current iterate is restarted from the best point found so far, so smaller steps refine a promising neighborhood rather than continuing from a poor one. A heavy-ball momentum term blends the fresh projected step with the previous displacement, which stabilizes the deliberately large early steps. The only free parameter is the iteration budget; the standard version fixes it at one hundred iterations.

The DLR loss is designed to be invariant to the same transformations as the classifier decision. The argmax is unchanged if a constant is added to all logits or if all logits are multiplied by a positive scalar, so the loss should be both shift-invariant and scale-invariant. The natural building block is a ratio of logit differences, because differences cancel additive shifts and a ratio cancels multiplicative scales. The untargeted DLR loss is defined as the negative of the correct-class margin divided by a robust scale estimate: minus the quantity z_y minus the largest logit among the other classes, divided by the gap between the first and third sorted logits. While the sample is correctly classified the numerator is non-negative and no larger than the denominator, so the loss lies between minus one and zero; after a wrong class overtakes the true class the numerator becomes negative and the loss becomes positive. Using the first-versus-third logit gap in the denominator avoids a zero-over-zero collapse when the top two classes are tied at the decision boundary. For targeted attacks the numerator is z_y minus z_t, pushing a specific wrong class above the true class, and the denominator uses the third and fourth sorted logits so it remains active.

The full standard ensemble runs four attacks sequentially on the shrinking set of points that are still robust. APGD-CE uses the cross-entropy loss and is kept untargeted because reducing true-class confidence helps against randomized defenses. Targeted APGD-DLR attacks each surviving point toward the nine highest-scoring wrong classes using the scale-invariant DLR loss. Targeted FAB linearizes the decision boundary and minimizes the perturbation norm, which is effective when gradients are informative but non-smooth. Finally, Square Attack is a gradient-free score-based random search that serves as a backstop against gradient masking. A point survives only if none of these attacks flips its prediction within the budget.

```python
import torch
import torch.nn.functional as F


def dlr_loss(z, y):
    """Difference-of-Logits-Ratio loss (untargeted)."""
    z_sorted, _ = z.sort(dim=1)
    u = torch.arange(z.shape[0], device=z.device)
    top_is_y = (z_sorted[:, -1] == z[u, y]).float()
    runner_up = z_sorted[:, -2] * top_is_y + z_sorted[:, -1] * (1.0 - top_is_y)
    return -(z[u, y] - runner_up) / (z_sorted[:, -1] - z_sorted[:, -3] + 1e-12)


def dlr_loss_targeted(z, y, y_target):
    """Targeted Difference-of-Logits-Ratio loss."""
    z_sorted, _ = z.sort(dim=1)
    u = torch.arange(z.shape[0], device=z.device)
    return -(z[u, y] - z[u, y_target]) / (
        z_sorted[:, -1] - 0.5 * (z_sorted[:, -3] + z_sorted[:, -4]) + 1e-12)


class APGD:
    """Auto-PGD: budget-aware, step-size-free projected gradient ascent."""

    def __init__(self, model, eps, norm="Linf", n_iter=100, loss="ce",
                 rho=0.75, n_target_classes=9, device="cuda"):
        self.model = model
        self.eps = eps
        self.norm = norm
        self.n_iter = n_iter
        self.loss = loss
        self.thr_decr = rho
        self.n_target_classes = n_target_classes
        self.device = device
        self.n_iter_2 = max(int(0.22 * n_iter), 1)
        self.n_iter_min = max(int(0.06 * n_iter), 1)
        self.size_decr = max(int(0.03 * n_iter), 1)

    def _grad_dir(self, g):
        if self.norm == "Linf":
            return torch.sign(g)
        norm = (g ** 2).flatten(1).sum(-1).sqrt()
        return g / (norm.view(-1, *([1] * (g.dim() - 1))) + 1e-12)

    def _project(self, x_adv, x):
        if self.norm == "Linf":
            x_adv = torch.min(torch.max(x_adv, x - self.eps), x + self.eps)
        else:
            d = x_adv - x
            n = (d ** 2).flatten(1).sum(-1, keepdim=True).sqrt()
            factor = torch.clamp(self.eps / (n + 1e-12), max=1.0)
            x_adv = x + d * factor.view(-1, *([1] * (x.dim() - 1)))
        return x_adv.clamp(0.0, 1.0)

    def _loss_fn(self, z, y, y_target=None):
        if self.loss == "ce":
            return F.cross_entropy(z, y, reduction="none")
        if self.loss == "dlr":
            return dlr_loss(z, y)
        return dlr_loss_targeted(z, y, y_target)

    def _check_oscillation(self, loss_steps, j, k):
        t = torch.zeros(loss_steps.shape[1], device=self.device)
        for c in range(k):
            t += (loss_steps[j - c] > loss_steps[j - c - 1]).float()
        return (t <= k * self.thr_decr).float()

    def _single_run(self, x, y, y_target=None):
        if self.norm == "Linf":
            x_adv = x + self.eps * (2 * torch.rand_like(x) - 1)
        else:
            t = torch.randn_like(x)
            n = (t ** 2).flatten(1).sum(-1, keepdim=True).sqrt()
            x_adv = x + self.eps * t / (n.view(-1, *([1] * (x.dim() - 1))) + 1e-12)
        x_adv = x_adv.clamp(0.0, 1.0)

        loss_steps = torch.zeros(self.n_iter, x.shape[0], device=self.device)
        x_adv.requires_grad_(True)
        z = self.model(x_adv)
        loss_indiv = self._loss_fn(z, y, y_target)
        grad = torch.autograd.grad(loss_indiv.sum(), [x_adv])[0].detach()

        x_best = x_adv.detach().clone()
        x_best_adv = x_adv.detach().clone()
        loss_best = loss_indiv.detach().clone()
        grad_best = grad.clone()
        acc = z.detach().max(1)[1] == y

        step_size = 2.0 * self.eps * torch.ones(
            [x.shape[0], *([1] * (x.dim() - 1))], device=self.device)
        x_adv = x_adv.detach()
        x_adv_old = x_adv.clone()
        k, counter = self.n_iter_2, 0
        loss_best_last = loss_best.clone()
        reduced_last = torch.ones_like(loss_best)

        for i in range(self.n_iter):
            with torch.no_grad():
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                a = 0.75 if i > 0 else 1.0
                z_step = self._project(x_adv + step_size * self._grad_dir(grad), x)
                x_adv = self._project(
                    x_adv + a * (z_step - x_adv) + (1 - a) * grad2, x)

            x_adv.requires_grad_(True)
            z = self.model(x_adv)
            loss_indiv = self._loss_fn(z, y, y_target)
            grad = torch.autograd.grad(loss_indiv.sum(), [x_adv])[0].detach()
            x_adv = x_adv.detach()

            with torch.no_grad():
                pred = z.detach().max(1)[1] == y
                acc = torch.min(acc, pred)
                newly = (pred == 0)
                x_best_adv[newly] = x_adv[newly]

                y1 = loss_indiv.detach()
                loss_steps[i] = y1
                improved = y1 > loss_best
                x_best[improved] = x_adv[improved].clone()
                grad_best[improved] = grad[improved].clone()
                loss_best[improved] = y1[improved]

                counter += 1
                if counter == k:
                    osc = self._check_oscillation(loss_steps, i, k)
                    no_impr = (1.0 - reduced_last) * (loss_best_last >= loss_best).float()
                    halve = torch.max(osc, no_impr)
                    reduced_last = halve.clone()
                    loss_best_last = loss_best.clone()
                    sel = halve > 0
                    step_size[sel] /= 2.0
                    x_adv[sel] = x_best[sel].clone()
                    grad[sel] = grad_best[sel].clone()
                    k = max(k - self.size_decr, self.n_iter_min)
                    counter = 0

        return x_best_adv, acc

    def perturb(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        if self.loss != "dlr-targeted":
            adv, _ = self._single_run(x, y)
            return adv
        adv = x.clone()
        acc = self.model(x).max(1)[1] == y
        for target_class in range(2, self.n_target_classes + 2):
            idx = acc.nonzero().squeeze(1)
            if idx.numel() == 0:
                break
            xt, yt = x[idx], y[idx]
            order = self.model(xt).sort(dim=1)[1]
            y_target = order[:, -target_class]
            adv_curr, acc_curr = self._single_run(xt, yt, y_target)
            broken = (acc_curr == 0).nonzero().squeeze(1)
            acc[idx[broken]] = False
            adv[idx[broken]] = adv_curr[broken]
        return adv


def auto_attack(model, images, labels, eps, device, n_classes,
                norm="Linf", FAB=None, Square=None):
    """Standard AutoAttack: worst case over APGD-CE, APGD-t, FAB-t, and Square."""
    model.eval()
    x, y = images.to(device), labels.to(device)
    robust = (model(x).max(1)[1] == y)
    x_adv = x.clone()

    apgd_ce = APGD(model, eps, norm, n_iter=100, loss="ce", device=device)
    apgd_t = APGD(model, eps, norm, n_iter=100, loss="dlr-targeted",
                  n_target_classes=min(9, n_classes - 1), device=device)
    members = [apgd_ce.perturb, apgd_t.perturb]
    if FAB is not None:
        members.append(lambda xx, yy: FAB(model, eps, norm).perturb(xx, yy))
    if Square is not None:
        members.append(lambda xx, yy: Square(model, eps, norm).perturb(xx, yy))

    for attack in members:
        idx = robust.nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        adv_curr = attack(x[idx], y[idx])
        with torch.no_grad():
            broken = model(adv_curr).max(1)[1] != y[idx]
        x_adv[idx[broken]] = adv_curr[broken]
        robust[idx[broken]] = False
    return x_adv.clamp(0.0, 1.0)
```
