I start from the evaluation failure, not from an attack recipe. I have a classifier with logits `z = g(x)` and decision `argmax_k z_k`; I have a correctly classified point `x_orig` with label `y`; I have a feasible set `S = {x : ||x - x_orig||_p <= eps} ∩ [0,1]^d`. I need a point in `S` whose predicted class is not `y`. The standard way to search is projected gradient ascent on some scalar loss `L(g(x), y)`, because for `l_infinity` and `l_2` I can project back onto `S`. So the reliability of the whole robustness number rests on two choices: what loss I climb, and how I choose the projected ascent steps. If either choice can be defeated without changing the decision boundary, the evaluation can report robustness that is not real.

I look at the optimizer first. Ordinary PGD fixes a step size `eta`, usually starts randomly in the ball, and repeats something like `x^{k+1} = P_S(x^k + eta sign(grad f(x^k)))` for the `l_infinity` case. The direction is not the suspicious part: `sign(grad)` is the steepest direction for an `l_infinity` linearization, and the normalized gradient is the analogous `l_2` direction. The suspicious part is that `eta` is a user-chosen constant. If I sweep fixed steps over a long run, the best-so-far loss typically jumps early and then plateaus; tiny steps avoid some overshoot but do not reach strong losses in the available budget. That means the iteration count by itself is not a reliable measure of attack strength. A fixed step does not know whether it is still improving, and it cannot be both large enough for early exploration and small enough for late local refinement.

Then I look at the loss. Cross-entropy is `CE(x,y) = -log p_y = -z_y + log(sum_j exp(z_j))`, where `p_i = exp(z_i)/sum_j exp(z_j)`. Its input gradient is `grad_x CE(x,y) = (-1 + p_y) grad_x z_y + sum_{i != y} p_i grad_x z_i`. If the true-class probability is close to one, all the coefficients in that expression are close to zero, and in finite precision they can become exactly zero. A decision-preserving logit rescaling can create exactly this condition: if `h = alpha g` with `alpha > 0`, the argmax is unchanged for every input, but the softmax gets sharper as `alpha` grows. Conversely, if an already overconfident model is divided by a large scale factor, the gradient can reappear even though the classifier decisions still match. So cross-entropy has a scale degree of freedom that the decision rule does not have, and that mismatch is enough to blind a CE-PGD evaluation.

I need a loss with the same invariances as the decision. Adding a constant to every logit and multiplying every logit by a positive scalar both leave `argmax` unchanged, so the loss should be shift-invariant and scale-invariant. A margin loss such as `-z_y + max_{i != y} z_i` has the right decision sign: it is positive exactly when some wrong class beats `y`. It is also shift-invariant because it is a difference of logits. But it is not scale-invariant; multiplying all logits by `alpha` multiplies the margin by `alpha`. I therefore need a ratio of logit differences, so the additive shift cancels inside each difference and the multiplicative scale cancels between numerator and denominator.

I sort the logits in decreasing order, `z_{pi_1} >= z_{pi_2} >= z_{pi_3} >= ...`. The numerator should be the raw correct-class margin `z_y - max_{i != y} z_i`, with a leading minus sign because I am maximizing the loss and I want the loss to become positive after misclassification. The tempting denominator `z_{pi_1} - z_{pi_2}` is unsafe because the attack explicitly tries to close the top-two gap; at the decision boundary it can create a `0/0` shape. The gap `z_{pi_1} - z_{pi_3}` is the right normalizer because it keeps the top-two competition out of the denominator while still measuring the logit scale. That gives `DLR(x,y) = - (z_y - max_{i != y} z_i) / (z_{pi_1} - z_{pi_3})`.

I check the sign and range before trusting it. While the input is correctly classified, `pi_1 = y` and `max_{i != y} z_i = z_{pi_2}`, so `DLR = - (z_y - z_{pi_2})/(z_y - z_{pi_3})`. The numerator is nonnegative and no larger than the denominator, so the loss lies in `[-1, 0]`. When a wrong class overtakes `y`, the raw margin `z_y - max_{i != y} z_i` turns negative and the leading minus sign makes the DLR loss positive. This is the sign pattern I want: bounded and comparable before success, positive exactly after the decision flips.

For targeted attacks I want class `t` specifically to beat the true class, so the numerator becomes `z_y - z_t`, again with the leading minus sign. I keep the ratio structure but use a denominator that remains nonconstant for the targeted objective: `z_{pi_1} - (z_{pi_3} + z_{pi_4})/2`. The targeted loss is therefore `Targeted-DLR(x,y,t) = - (z_y - z_t)/(z_{pi_1} - (z_{pi_3} + z_{pi_4})/2)`. It still uses only differences of logits, still cancels positive rescalings, and now maximizing it directly pushes `z_t` above `z_y`.

With the loss fixed, I return to the step-size problem. I want a large first step for exploration and smaller later steps for refinement, but I do not want a human to tune the transition. I start from `eta^(0) = 2 eps`. In `l_infinity`, a signed step of size `2 eps` can cross the whole feasible box before projection, so the projection sends the point to an informative boundary location. For `l_2`, I use the normalized gradient and the same `2 eps` scale. I keep the best point and best objective value seen so far, because the raw iterate is not monotone.

I put checkpoints into the iteration budget and only decide about halving at those checkpoints. Let `p_0 = 0`, `p_1 = 0.22`, and `p_{j+1} = p_j + max{p_j - p_{j-1} - 0.03, 0.06}`, with `w_j = ceil(p_j N)`. The first interval is long enough to explore; later intervals shrink by `0.03 N` until they hit a `0.06 N` floor, so the schedule checks progress more often as the search becomes local.

At checkpoint `w_j`, I use two tests. Condition 1 counts how many steps in `[w_{j-1}, w_j)` actually increase the objective: `sum_i 1[f(x^{i+1}) > f(x^i)]`. If this count falls below `rho (w_j - w_{j-1})`, with `rho = 0.75`, the current step is too aggressive too often and I halve it. Condition 2 catches the case where the success count does not expose a cycle: if the step size was not reduced at the previous checkpoint and the best objective has not improved since then, I halve anyway. Whenever I halve, I restart the current iterate from `x_max`, the best point so far, because a smaller step should refine the best neighborhood found by the larger step.

I also keep a momentum term because the early steps are deliberately large. I first compute the raw projected step `z^{k+1} = P_S(x^k + eta^k grad-dir(x^k))`. Then I blend that proposed move with the previous displacement and project again: `x^{k+1} = P_S(x^k + alpha (z^{k+1} - x^k) + (1 - alpha)(x^k - x^{k-1}))`, with `alpha = 0.75`. At the first update I use the plain projected step because there is no previous displacement. This gives me a projected ascent routine whose step size is controlled by observed progress rather than by per-model tuning.

Now I have an adaptive PGD core and a scale-invariant loss, but I do not assume one gradient attack is enough. If a defense makes gradients discontinuous, flat, noisy, or otherwise uninformative, even a better loss can fail. I therefore want a small ensemble whose members fail for different reasons. I keep an untargeted cross-entropy APGD run because reducing true-class confidence is useful, especially when outputs are stochastic. I add targeted DLR-APGD because it is the scale-invariant gradient attack and because targeting the highest-scoring wrong classes gives a sharper objective. I add targeted FAB because it searches by linearizing decision boundaries and minimizing perturbation norm rather than by maximizing a fixed-ball loss. I add Square Attack because it is score-based random search and uses no gradients at all.

I run the ensemble sequentially on a shrinking set. I mark initially correct points as still robust, run APGD-CE, store any valid adversarial examples it finds, and remove those points. I then run targeted DLR-APGD on the survivors, then targeted FAB, then Square Attack. The final robust accuracy is the worst case over members: a point survives only if none of the attacks finds a valid in-ball misclassification. The standard settings are fixed: APGD-CE as a single 100-iteration run, targeted DLR-APGD as a single 100-iteration run over 9 target classes, targeted FAB as a single run over 9 target classes, and Square Attack with 5000 queries. The result is an autonomous robustness evaluation built from two new pieces, the adaptive PGD schedule and the DLR loss, plus complementary existing attacks chosen for diversity rather than for another tunable knob.

Let me write it down as real code — the two new pieces, APGD with the DLR loss, plus the ensemble that wires APGD together with FAB and Square.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def dlr_loss(z, y):
    # DLR(x,y) = -(z_y - max_{i!=y} z_i) / (z_pi1 - z_pi3)
    # numerator = decision margin (CW); denom = (1st - 3rd) logit gap, stays > 0
    z_sorted, _ = z.sort(dim=1)
    u = torch.arange(z.shape[0])
    top_is_y = (z_sorted[:, -1] == z[u, y]).float()
    # max_{i!=y}: if argmax is y use 2nd largest, else the largest
    runner_up = z_sorted[:, -2] * top_is_y + z_sorted[:, -1] * (1.0 - top_is_y)
    return -(z[u, y] - runner_up) / (z_sorted[:, -1] - z_sorted[:, -3] + 1e-12)


def dlr_loss_targeted(z, y, y_target):
    # Targeted-DLR(x,y,t) = -(z_y - z_t) / (z_pi1 - (z_pi3 + z_pi4)/2)
    u = torch.arange(z.shape[0])
    z_sorted, _ = z.sort(dim=1)
    return -(z[u, y] - z[u, y_target]) / (
        z_sorted[:, -1] - 0.5 * (z_sorted[:, -3] + z_sorted[:, -4]) + 1e-12)


class APGD:
    """Auto-PGD: budget-aware, step-size-free PGD. Only free parameter is n_iter."""

    def __init__(self, model, eps, norm="Linf", n_iter=100, loss="ce",
                 rho=0.75, n_target_classes=9, device="cuda"):
        self.model, self.eps, self.norm = model, eps, norm
        self.n_iter, self.loss, self.thr_decr = n_iter, loss, rho
        self.n_target_classes = n_target_classes
        self.device = device
        # checkpoints: w_0=0, period_1 = 0.22 N, shrink by 0.03 N, floor 0.06 N
        self.n_iter_2 = max(int(0.22 * n_iter), 1)    # length of first window
        self.n_iter_min = max(int(0.06 * n_iter), 1)  # floor on window length
        self.size_decr = max(int(0.03 * n_iter), 1)   # window shrink per checkpoint

    def _grad_dir(self, g):
        if self.norm == "Linf":
            return torch.sign(g)                       # Linf steepest ascent
        t = (g ** 2).flatten(1).sum(-1).sqrt()         # L2: normalized gradient
        return g / (t.view(-1, *([1] * (g.dim() - 1))) + 1e-12)

    def _project(self, x_adv, x):
        if self.norm == "Linf":
            x_adv = torch.min(torch.max(x_adv, x - self.eps), x + self.eps)
        else:  # L2
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
        # Condition 1: count steps that increased the loss over the last-k window;
        # halve if successes <= rho * k  (too few successful steps -> overshooting)
        t = torch.zeros(loss_steps.shape[1], device=self.device)
        for c in range(k):
            t += (loss_steps[j - c] > loss_steps[j - c - 1]).float()
        return (t <= k * self.thr_decr).float()

    def _single_run(self, x, y, y_target=None):
        # random start inside the ball, then project
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

        x_best, x_best_adv = x_adv.detach().clone(), x_adv.detach().clone()
        loss_best = loss_indiv.detach().clone()
        acc = z.detach().max(1)[1] == y                 # still correctly classified

        # eta^0 = 2*eps : one signed step crosses the whole ball (full exploration)
        step_size = 2.0 * self.eps * torch.ones(
            [x.shape[0], *([1] * (x.dim() - 1))], device=self.device)
        x_adv = x_adv.detach()
        x_adv_old = x_adv.clone()
        grad_best = grad.clone()

        k = self.n_iter_2 + 0
        counter = 0
        loss_best_last, reduced_last = loss_best.clone(), torch.ones_like(loss_best)

        for i in range(self.n_iter):
            with torch.no_grad():
                grad2 = x_adv - x_adv_old            # previous step (momentum term)
                x_adv_old = x_adv.clone()
                a = 0.75 if i > 0 else 1.0           # alpha; plain step at i=0

                # raw projected gradient step
                z_step = self._project(x_adv + step_size * self._grad_dir(grad), x)
                # blend fresh step (a) with carried previous step (1-a), reproject
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
                newly = (pred == 0)                  # broken this iteration
                x_best_adv[newly] = x_adv[newly]     # store adversarial examples

                y1 = loss_indiv.detach()
                loss_steps[i] = y1
                improved = y1 > loss_best             # track best-so-far (x_max,f_max)
                x_best[improved] = x_adv[improved].clone()
                grad_best[improved] = grad[improved].clone()
                loss_best[improved] = y1[improved]

                counter += 1
                if counter == k:                     # at a checkpoint w_j
                    osc = self._check_oscillation(loss_steps, i, k)        # Cond 1
                    no_impr = (1.0 - reduced_last) * (loss_best_last >= loss_best).float()
                    halve = torch.max(osc, no_impr)  # Cond 2: not reduced & f_max flat
                    reduced_last = halve.clone()
                    loss_best_last = loss_best.clone()

                    sel = halve > 0
                    step_size[sel] /= 2.0            # halve eta
                    x_adv[sel] = x_best[sel].clone() # restart from best point
                    grad[sel] = grad_best[sel].clone()

                    k = max(k - self.size_decr, self.n_iter_min)  # next window shorter
                    counter = 0

        return x_best_adv, acc

    def perturb(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        if self.loss != "dlr-targeted":
            adv, acc = self._single_run(x, y)
            return adv
        # targeted: attack toward the n highest-scoring wrong classes
        adv = x.clone()
        acc = self.model(x).max(1)[1] == y
        for target_class in range(2, self.n_target_classes + 2):
            idx = acc.nonzero().squeeze(1)
            if idx.numel() == 0:
                break
            xt, yt = x[idx], y[idx]
            order = self.model(xt).sort(dim=1)[1]
            y_target = order[:, -target_class]       # the target-th highest wrong class
            adv_curr, acc_curr = self._single_run(xt, yt, y_target)
            broken = (acc_curr == 0).nonzero().squeeze(1)
            acc[idx[broken]] = False
            adv[idx[broken]] = adv_curr[broken]
        return adv


def auto_attack(model, images, labels, eps, device, n_classes,
                norm="Linf", FAB=None, Square=None):
    """AutoAttack (standard): worst-case over apgd-ce, apgd-t, fab-t, square,
    run sequentially on the shrinking set of still-robust points."""
    model.eval()
    x, y = images.to(device), labels.to(device)
    robust = (model(x).max(1)[1] == y)               # only attack initially-correct pts
    x_adv = x.clone()

    apgd_ce = APGD(model, eps, norm, n_iter=100, loss="ce", device=device)
    apgd_t = APGD(model, eps, norm, n_iter=100, loss="dlr-targeted",
                  n_target_classes=min(9, n_classes - 1), device=device)
    members = [apgd_ce.perturb, apgd_t.perturb]
    if FAB is not None:                              # targeted FAB, boundary-minimization
        members.append(lambda xx, yy: FAB(model, eps, norm).perturb(xx, yy))
    if Square is not None:                           # gradient-free black-box backstop
        members.append(lambda xx, yy: Square(model, eps, norm).perturb(xx, yy))

    for attack in members:
        idx = robust.nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        adv_curr = attack(x[idx], y[idx])            # only the still-robust points
        with torch.no_grad():
            broken = model(adv_curr).max(1)[1] != y[idx]
        x_adv[idx[broken]] = adv_curr[broken]
        robust[idx[broken]] = False                  # worst-case over attacks
    return x_adv.clamp(0.0, 1.0)
```

The causal chain: I wanted an evaluation that does not lie and needs no per-model tuning. Two diagnosed lies — a fixed step size that wastes the budget and can't be both coarse and fine, and a cross-entropy loss whose extra scale degree of freedom lets a decision-identical model mask its gradient. The loss is fixed by matching its symmetry group to the decision's, giving the shift-and-scale-invariant DLR as a ratio of logit-differences with a denominator that never collapses. The step size is fixed by making it a function of the trajectory: start at `2eps`, halve at checkpoints when too few steps succeed (Condition 1) or progress has stalled (Condition 2), restart from the best point, with a heavy-ball term to stabilize the large early steps — APGD, whose only knob is the budget. And because any single gradient attack still has a failure mode some defense will sit on, reliability comes from a diverse, parameter-free ensemble — APGD-CE, targeted-DLR-APGD, targeted FAB, gradient-free Square — applied worst-case over the shrinking set of still-robust points.
