# AutoAttack, distilled

AutoAttack is a parameter-free, autonomous protocol for evaluating adversarial robustness under
an `l_p` (`l_infinity` or `l_2`) budget `eps`. It contributes two pieces — **Auto-PGD (APGD)**, a
budget-aware, step-size-free variant of PGD, and the **Difference-of-Logits-Ratio (DLR) loss**,
which is invariant to both shifts and rescalings of the logits — and combines APGD (on cross-
entropy and on the targeted DLR loss) with two existing complementary attacks, **FAB** (white-box,
minimum-norm) and **Square Attack** (gradient-free black-box), into a diverse ensemble with fixed
attack settings in the standard version.

## Problem it solves

Robustness evaluations overestimate robustness, because the standard attack — PGD with a fixed,
user-chosen step size on the cross-entropy loss — has two structural failure modes: a fixed step
size cannot be both coarse (to explore) and fine (to converge), wastes the iteration budget once
the loss plateaus, and never reacts to its own progress; and cross-entropy is scale-variant, so a
defense can replace `g` by the decision-identical `h = alpha·g` with large `alpha`, drive the
true-class probability to ~1, and make the input-gradient vanish (exactly zero in finite
precision) — gradient masking with no real increase in robustness. A single attack also has a
single failure mode that some defense will be built on.

## Key idea

- **Match the loss's symmetry to the decision's.** The decision `argmax_k z_k` is invariant to
  adding a constant to all logits and to multiplying all logits by `alpha>0`. Build a loss with
  exactly those invariances as a *ratio of logit-differences*:

  `DLR(x,y) = - (z_y - max_{i != y} z_i) / (z_{pi_1} - z_{pi_3})`,

  where `pi` orders the logits in decreasing order. The raw margin `z_y - max_{i != y} z_i` is
  positive while class `y` still wins and negative after misclassification; the leading minus sign
  makes the DLR loss positive exactly when `argmax_i z_i != y`. The ratio cancels the scale
  `alpha`, and the denominator uses the 1st-vs-3rd logit gap so the top-two collision does not
  create a `0/0` objective. For a correctly classified point `DLR in [-1, 0]`. Targeted form:
  `Targeted-DLR(x,y,t) = - (z_y - z_t) / (z_{pi_1} - (z_{pi_3} + z_{pi_4})/2)`,
  whose denominator uses the 3rd and 4th sorted logits to avoid a constant targeted loss while
  preserving both invariances.

- **Make the step size a function of the trajectory (APGD).** Start at `eta^(0) = 2·eps` (one
  signed step crosses the whole `l_infinity` ball — full exploration). At budget-fraction
  checkpoints, halve `eta` and restart from the best point so far if either too few steps
  succeeded (Condition 1) or progress has stalled (Condition 2). A heavy-ball momentum term
  stabilizes the large early steps. APGD exposes the iteration budget `N`; the standard ensemble
  fixes it at `N = 100`.

- **Buy reliability with diversity.** No single gradient attack is enough, so combine three
  mechanisms — gradient ascent in a fixed ball (APGD on CE and on targeted DLR),
  boundary-minimization (targeted FAB), and gradient-free random search (Square Attack) — and
  take the worst case per point over the ensemble. APGD-CE is kept untargeted (minimizing
  true-class confidence helps against randomized defenses); DLR-APGD and FAB use their targeted
  forms (more effective, and FAB-targeted is independent of the number of classes).

## APGD algorithm

Maximize `f` over `S = {x : ||x - x_orig||_p <= eps} ∩ [0,1]^d`. Direction
`grad-dir = sign(grad f)` for `l_infinity`, `grad f / ||grad f||_2` for `l_2`. With `alpha = 0.75`:

```
eta^(0) = 2*eps
x^(1) = P_S(x^(0) + eta^(0) * grad-dir(x^(0)))            # plain first step
f_max, x_max = best of {f(x^(0)), f(x^(1))}, argmax
for k = 1 .. N-1:
    z^(k+1) = P_S(x^(k) + eta^(k) * grad-dir(x^(k)))      # raw projected step
    x^(k+1) = P_S(x^(k) + alpha*(z^(k+1)-x^(k))
                       + (1-alpha)*(x^(k)-x^(k-1)))        # + heavy-ball momentum
    update (x_max, f_max) if f(x^(k+1)) > f_max
    if k is a checkpoint w_j:
        # Condition 1: sum_{i=w_{j-1}}^{w_j-1} 1[f(x^(i+1))>f(x^(i))] < rho*(w_j - w_{j-1})
        #              implemented here as <= rho*window
        # Condition 2: eta unchanged since w_{j-1} AND f_max unchanged since w_{j-1}
        if Condition 1 or Condition 2:
            eta <- eta / 2
            x^(k+1) <- x_max                               # restart from best point
return x_max  (and any in-ball point that flipped the prediction)
```

Checkpoints in budget fractions: `p_0 = 0`, `p_1 = 0.22`,
`p_{j+1} = p_j + max{p_j - p_{j-1} - 0.03, 0.06}`, `w_j = ceil(p_j * N)`. Fixed constants:
`eta^(0) = 2*eps`, `alpha = 0.75`, `rho = 0.75`. (For `l_2`, `eta^(0)` is set to `2*eps` as in
the `l_infinity` case.) Implementation detail: the success-count condition is read off as
"of the last `k` steps, how many increased the loss," with the window `k` shrinking by `0.03·N`
each checkpoint down to a floor of `0.06·N`.

## AutoAttack ensemble (standard version)

`attacks_to_run = [apgd-ce, apgd-t, fab-t, square]`: APGD-CE with no random restarts and
`n_iter = 100`; targeted-DLR-APGD with `n_iter = 100` and 9 target classes (the 9 highest-scoring
wrong classes); targeted FAB with `n_iter = 100`, 9 target classes, and 1 restart; Square Attack
with one run of 5000 queries. Applied sequentially to the shrinking set of still-robust points; a
point is broken if
*any* member finds a valid in-ball perturbation that flips the prediction (worst case per point).
All hyperparameters are fixed across datasets, models, and norms.

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def dlr_loss(z, y):
    # -(z_y - max_{i!=y} z_i) / (z_pi1 - z_pi3); shift- and scale-invariant
    z_sorted, _ = z.sort(dim=1)
    u = torch.arange(z.shape[0])
    top_is_y = (z_sorted[:, -1] == z[u, y]).float()
    runner_up = z_sorted[:, -2] * top_is_y + z_sorted[:, -1] * (1.0 - top_is_y)
    return -(z[u, y] - runner_up) / (z_sorted[:, -1] - z_sorted[:, -3] + 1e-12)


def dlr_loss_targeted(z, y, y_target):
    # -(z_y - z_t) / (z_pi1 - (z_pi3 + z_pi4)/2)
    u = torch.arange(z.shape[0])
    z_sorted, _ = z.sort(dim=1)
    return -(z[u, y] - z[u, y_target]) / (
        z_sorted[:, -1] - 0.5 * (z_sorted[:, -3] + z_sorted[:, -4]) + 1e-12)


class APGD:
    """Auto-PGD: budget-aware, step-size-free PGD. Free parameter: n_iter."""

    def __init__(self, model, eps, norm="Linf", n_iter=100, loss="ce",
                 rho=0.75, n_target_classes=9, device="cuda"):
        self.model, self.eps, self.norm = model, eps, norm
        self.n_iter, self.loss, self.thr_decr = n_iter, loss, rho
        self.n_target_classes, self.device = n_target_classes, device
        self.n_iter_2 = max(int(0.22 * n_iter), 1)    # first window length
        self.n_iter_min = max(int(0.06 * n_iter), 1)  # window floor
        self.size_decr = max(int(0.03 * n_iter), 1)   # window shrink per checkpoint

    def _grad_dir(self, g):
        if self.norm == "Linf":
            return torch.sign(g)
        t = (g ** 2).flatten(1).sum(-1).sqrt()
        return g / (t.view(-1, *([1] * (g.dim() - 1))) + 1e-12)

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
        # Condition 1: halve when successes <= rho*k
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

        step_size = 2.0 * self.eps * torch.ones(           # eta^0 = 2*eps
            [x.shape[0], *([1] * (x.dim() - 1))], device=self.device)
        x_adv = x_adv.detach()
        x_adv_old = x_adv.clone()
        k, counter = self.n_iter_2, 0
        loss_best_last, reduced_last = loss_best.clone(), torch.ones_like(loss_best)

        for i in range(self.n_iter):
            with torch.no_grad():
                grad2 = x_adv - x_adv_old               # previous step
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
                    osc = self._check_oscillation(loss_steps, i, k)             # Cond 1
                    no_impr = (1.0 - reduced_last) * (loss_best_last >= loss_best).float()
                    halve = torch.max(osc, no_impr)                            # Cond 2
                    reduced_last = halve.clone()
                    loss_best_last = loss_best.clone()
                    sel = halve > 0
                    step_size[sel] /= 2.0
                    x_adv[sel] = x_best[sel].clone()       # restart from best point
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
            y_target = self.model(xt).sort(dim=1)[1][:, -target_class]
            adv_curr, acc_curr = self._single_run(xt, yt, y_target)
            broken = (acc_curr == 0).nonzero().squeeze(1)
            acc[idx[broken]] = False
            adv[idx[broken]] = adv_curr[broken]
        return adv


def auto_attack(model, images, labels, eps, device, n_classes,
                norm="Linf", FAB=None, Square=None):
    """AutoAttack (standard): worst case over apgd-ce, apgd-t, fab-t, square,
    applied sequentially to the shrinking set of still-robust points."""
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

The two new pieces (APGD and the DLR loss) are the contribution; FAB and Square Attack are
existing attacks plugged in for diversity. In the canonical library the ensemble is invoked as a
single call (`AutoAttack(model, norm="Linf", eps=eps, version="standard")`), which wires
`apgd-ce`, `apgd-t`, `fab-t`, and `square` exactly as above.
