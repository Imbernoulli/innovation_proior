I start from the evaluation failure, not from an attack recipe. I have a classifier with logits `z = g(x)` and decision `argmax_k z_k`; I have a correctly classified point `x_orig` with label `y`; I have a feasible set `S = {x : ||x - x_orig||_p <= eps} ∩ [0,1]^d`. I need a point in `S` whose predicted class is not `y`. The standard way to search is projected gradient ascent on some scalar loss `L(g(x), y)`, because for `l_infinity` and `l_2` I can project back onto `S`. So the reliability of the whole robustness number rests on two choices: what loss I climb, and how I choose the projected ascent steps. If either choice can be defeated without changing the decision boundary, the evaluation can report robustness that is not real. That is the lens I want to use on both choices: does anything an attack relies on survive a transformation that leaves every decision untouched?

I look at the optimizer first. Ordinary PGD fixes a step size `eta`, usually starts randomly in the ball, and repeats something like `x^{k+1} = P_S(x^k + eta sign(grad f(x^k)))` for the `l_infinity` case. The direction is not the suspicious part: `sign(grad)` is the steepest direction for an `l_infinity` linearization, and the normalized gradient is the analogous `l_2` direction. The suspicious part is that `eta` is a user-chosen constant. If I sweep fixed steps over a long run, the best-so-far loss typically jumps early and then plateaus; tiny steps avoid some overshoot but do not reach strong losses in the available budget. That means the iteration count by itself is not a reliable measure of attack strength. A fixed step does not know whether it is still improving, and it cannot be both large enough for early exploration and small enough for late local refinement.

Then I look at the loss. Cross-entropy is `CE(x,y) = -log p_y = -z_y + log(sum_j exp(z_j))`, where `p_i = exp(z_i)/sum_j exp(z_j)`. Its input gradient is `grad_x CE(x,y) = (-1 + p_y) grad_x z_y + sum_{i != y} p_i grad_x z_i`. If the true-class probability is close to one, all the coefficients in that expression — `(-1 + p_y)` and each `p_i` for `i != y` — are close to zero. I want to know how fast that happens and whether finite precision turns "small" into "exactly zero," because a defense could engineer it. Take a concrete five-class logit vector `z = (4, 2, 1, 0.5, 0)` with `y = 0`. At its natural scale `p_y = 0.811` and the largest gradient coefficient has magnitude `0.189` — a healthy gradient. Now form the decision-identical classifier `h = alpha g`, which leaves `argmax` unchanged for every input but sharpens the softmax. Computing the coefficients of `grad_x CE` for `h`:

```
alpha=   1  p_y=0.811        max|coeff|=1.89e-01
alpha=  10  p_y≈1.0          max|coeff|=2.06e-09
alpha= 100  p_y≈1.0          max|coeff|=1.38e-87
alpha=1000  p_y≈1.0          max|coeff|=0.00e+00
```

So a positive rescaling that no decision can detect drives the entire CE input-gradient toward zero, and by `alpha = 1000` it is exactly zero even in double precision. In single precision the collapse comes much sooner: at `alpha = 10`, `p_y` rounds to the representable value `1.0f`, so `(-1 + p_y)` evaluates to exactly `0.0f` and `sign(grad)` is zero everywhere. A CE-PGD evaluation run against `h` would report the point as robust while `g` and `h` make identical decisions. Cross-entropy has a scale degree of freedom that the decision rule does not have, and that single mismatch is enough to blind the evaluation.

That diagnosis tells me what the loss must look like. The decision `argmax_k z_k` is unchanged by two operations: adding a constant to every logit, and multiplying every logit by a positive scalar. A trustworthy loss should share both invariances. A margin loss `-z_y + max_{i != y} z_i` has the right decision sign — positive exactly when some wrong class beats `y` — and it is shift-invariant because it is a difference of logits. But multiplying all logits by `alpha` multiplies the margin by `alpha`, so it inherits exactly the scale weakness I am trying to remove. To cancel the scale I need a ratio: a logit-difference over a logit-difference, so the additive shift cancels inside each difference and the multiplicative `alpha` cancels between numerator and denominator.

The numerator should be the raw correct-class margin `z_y - max_{i != y} z_i`, with a leading minus sign so that maximizing the loss drives this margin negative and the loss positive after misclassification. For the denominator I sort the logits in decreasing order, `z_{pi_1} >= z_{pi_2} >= z_{pi_3} >= ...`, and I need a logit-gap that does not collapse exactly where the attack is working. The obvious candidate is the top-two gap `z_{pi_1} - z_{pi_2}`, but the attack's whole job is to close that gap, so I should check what happens to the ratio along a path to the boundary. Walk class-1's logit up toward class-0's while `y = 0` stays the label, with `z = (2, c1, 0.5, 0)`:

```
c1     num=z_y-max_{i!=y}   denom(1-2)   denom(1-3)   ratio with (1-3)
1.000      +1.000            1.0000        1.5000        +0.6667
1.900      +0.100            0.1000        1.5000        +0.0667
1.990      +0.010            0.0100        1.5000        +0.0067
2.000      +0.000            0.0000        1.5000        +0.0000
2.001      -0.001            0.0010        1.5010        -0.0007
```

At the boundary `c1 = 2.0` both the numerator and `z_{pi_1} - z_{pi_2}` hit zero together, so `z_{pi_1} - z_{pi_2}` as the denominator gives a `0/0` shape exactly at the moment of misclassification — the worst place to lose definition. The `z_{pi_1} - z_{pi_3}` gap, in contrast, stays pinned at `1.5` straight through the crossing, so it keeps the top-two competition out of the denominator while still measuring the logit scale. That settles the choice: `DLR(x,y) = - (z_y - max_{i != y} z_i) / (z_{pi_1} - z_{pi_3})`.

Now I check the invariances and the range I actually get, rather than assume them. On `z = (4, 2, 1, 0.5, 0)`, `y = 0`, the DLR value is `-2/3`. Adding `7` to every logit leaves it at `-2/3`; multiplying by `10`, by `100`, by `0.01` all leave it at `-2/3` — the shift and the positive scale cancel as intended. For the range: while the input is correctly classified, `pi_1 = y` and `max_{i != y} z_i = z_{pi_2}`, so `DLR = -(z_y - z_{pi_2})/(z_y - z_{pi_3})`; the numerator is nonnegative and no larger than the denominator, so the value should sit in `[-1, 0]`. Sampling 100000 random correctly-classified logit vectors, the observed DLR ranges from `-0.99999` to `-2.4e-05`, consistent with the `[-1, 0]` bound and never escaping it. And when I push class 1 to overtake class 0 (`z = (1, 3, 1, 0.5, 0)`, `y = 0`), DLR returns `+1.0`: positive exactly when the decision flips, as the leading minus sign was meant to arrange. Bounded and comparable before success, positive after the decision flips — that is the behavior the construction promised, now seen rather than asserted.

For targeted attacks I want a chosen class `t` specifically to beat the true class, so the numerator becomes `z_y - z_t`, again with the leading minus sign. The denominator needs the same two invariances but must also not become proportional to the numerator, or the ratio would be a constant with no gradient. The top-two gap is unsafe here too, so I anchor the denominator to logits below the `y`/`t` competition: `z_{pi_1} - (z_{pi_3} + z_{pi_4})/2`. Checking that it stays a real (nonconstant) function as `z_y` and `z_t` trade off at fixed sum, with `z = (z_y, z_t, 0.5, 0.2, 0)`:

```
z_y   z_t   num=z_y-z_t   denom   ratio
3.0   1.0     +2.00       2.650   +0.7547
2.5   1.5     +1.00       2.150   +0.4651
2.0   2.0     +0.00       1.650   +0.0000
1.5   2.5     -1.00       2.150   -0.4651
```

The denominator varies smoothly (`2.65 -> 2.15 -> 1.65 -> 2.15`) instead of tracking the numerator, so the targeted loss keeps a nonzero gradient throughout and flips sign exactly when `z_t` overtakes `z_y`. The targeted loss is therefore `Targeted-DLR(x,y,t) = - (z_y - z_t)/(z_{pi_1} - (z_{pi_3} + z_{pi_4})/2)`. It still uses only differences of logits, still cancels positive rescalings, and now maximizing it directly pushes `z_t` above `z_y`.

With the loss fixed, I return to the step-size problem. I want a large first step for exploration and smaller later steps for refinement, but I do not want a human to tune the transition. I start from `eta^(0) = 2 eps`. In `l_infinity`, a signed step of size `2 eps` can cross the whole feasible box before projection, so the projection sends the point to an informative boundary location. For `l_2`, I use the normalized gradient and the same `2 eps` scale. I keep the best point and best objective value seen so far, because the raw iterate is not monotone.

I put checkpoints into the iteration budget and only decide about halving at those checkpoints. Let `p_0 = 0`, `p_1 = 0.22`, and `p_{j+1} = p_j + max{p_j - p_{j-1} - 0.03, 0.06}`, with `w_j = ceil(p_j N)`. I should see what this actually produces before relying on the intuition that windows shrink. For `N = 100`:

```
p = [0.0, 0.22, 0.41, 0.57, 0.70, 0.80, 0.87, 0.93, 0.99]
w = [0, 22, 41, 58, 70, 80, 87, 93, 99]
window lengths = [22, 19, 17, 12, 10, 7, 6, 6]
```

The first window is `22 = 0.22 N` iterations, long enough to explore; later windows shrink by about `0.03 N = 3` each, and they bottom out at `6 = 0.06 N` rather than collapsing to length one. So the schedule checks progress more and more often as the search becomes local, but never faster than every six steps — which matches `n_iter_2 = 22`, `size_decr = 3`, `n_iter_min = 6` in the code.

At checkpoint `w_j`, I use two tests. Condition 1 counts how many steps in `[w_{j-1}, w_j)` actually increase the objective: `sum_i 1[f(x^{i+1}) > f(x^i)]`. If this count falls below `rho (w_j - w_{j-1})`, with `rho = 0.75`, the current step is too aggressive too often and I halve it. Condition 2 catches the case where the success count does not expose a cycle: if the step size was not reduced at the previous checkpoint and the best objective has not improved since then, I halve anyway. Whenever I halve, I restart the current iterate from `x_max`, the best point so far, because a smaller step should refine the best neighborhood found by the larger step.

I also keep a momentum term because the early steps are deliberately large. I first compute the raw projected step `z^{k+1} = P_S(x^k + eta^k grad-dir(x^k))`. Then I blend that proposed move with the previous displacement and project again: `x^{k+1} = P_S(x^k + alpha (z^{k+1} - x^k) + (1 - alpha)(x^k - x^{k-1}))`, with `alpha = 0.75`. At the first update I use the plain projected step because there is no previous displacement. This gives me a projected ascent routine whose step size is controlled by observed progress rather than by per-model tuning.

Now I have an adaptive PGD core and a scale-invariant loss, but I do not assume one gradient attack is enough. The scale-invariance check fixes the specific gradient-masking mechanism I diagnosed, but a defense can break gradients in other ways: discontinuous outputs, flat regions, injected noise, or just an unlucky local maximum that one loss happens to sit in. A loss that is correct by construction does not protect me against an optimizer that gets stuck for an unrelated reason. So I want a small ensemble whose members fail for different reasons. I keep an untargeted cross-entropy APGD run because reducing true-class confidence is useful, especially when outputs are stochastic. I add targeted DLR-APGD because it is the scale-invariant gradient attack and because targeting the highest-scoring wrong classes gives a sharper objective. I add targeted FAB because it searches by linearizing decision boundaries and minimizing perturbation norm rather than by maximizing a fixed-ball loss. I add Square Attack because it is score-based random search and uses no gradients at all — so a defense that masks gradients perfectly still has to survive a query-only attacker.

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

The causal chain: I wanted an evaluation that does not lie and needs no per-model tuning. Two diagnosed lies — a fixed step size that wastes the budget and can't be both coarse and fine, and a cross-entropy loss whose extra scale degree of freedom lets a decision-identical model mask its gradient, which I watched collapse to exactly zero under a logit rescaling. The loss is fixed by matching its symmetry group to the decision's, giving the shift-and-scale-invariant DLR as a ratio of logit-differences with a denominator that I checked stays away from `0/0` at the boundary; its `[-1,0]` range and its sign-flip at misclassification came out of direct computation rather than hope. The step size is fixed by making it a function of the trajectory: start at `2eps`, halve at checkpoints when too few steps succeed (Condition 1) or progress has stalled (Condition 2), restart from the best point, with a heavy-ball term to stabilize the large early steps — APGD, whose only knob is the budget, and whose checkpoint windows I traced out to confirm they shrink toward a floor rather than collapse. And because any single gradient attack still has a failure mode some defense will sit on, reliability comes from a diverse, parameter-free ensemble — APGD-CE, targeted-DLR-APGD, targeted FAB, gradient-free Square — applied worst-case over the shrinking set of still-robust points.
