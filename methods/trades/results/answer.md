# TRADES: TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization

## Problem

Train a classifier `sign(f(x))` that is accurate not only on clean inputs but on the worst
input inside every `l_inf` ball `B(x, eps)`, while giving an explicit, controllable trade-off
against clean accuracy. The natural worst-case (robust) error
`R_adv(f) = E 1{ exists x' in B(x,eps) s.t. f(x') y <= 0 }` is NP-hard and non-differentiable,
and minimizing it bluntly tends to sacrifice clean accuracy with no dial to control how much.

## Key idea

Decompose the robust error exactly into two additive pieces:

```
R_adv(f) = R_nat(f) + R_bdy(f),
R_nat(f) = E 1{ f(x) y <= 0 }                                   (natural / clean error)
R_bdy(f) = E 1{ x in B(DB(f), eps), f(x) y > 0 }               (boundary error)
```

where `DB(f) = {x : f(x) = 0}` and `B(DB(f), eps) = { x : exists x' in B(x,eps), f(x)f(x') <= 0 }`.
A robust mistake is either an already-wrong clean point (natural error) or a correctly-classified
point sitting within `eps` of the decision surface that a perturbation tips over (boundary error).
These two events are disjoint and exhaustive, so the split is an equality.

Bound each term by a differentiable surrogate and weight them. For a standard nonnegative,
non-increasing classification-calibrated margin loss `phi` with `phi(0) >= 1`, `psi`-transform
from Bartlett, Jordan & McAuliffe (2006), and any `lambda > 0`:

```
R_adv(f) - R_nat^*  <=  psi^{-1}( R_phi(f) - R_phi^* )  +  E max_{x' in B(x,eps)} phi( f(x) f(x') / lambda ).
```

The first term is the calibrated natural surrogate (accuracy); the second is a robustness
**regularizer that measures the disagreement between the clean prediction `f(x)` and the
perturbed prediction `f(x')`** — not between `f(x')` and the label `y` (this is the decisive
difference from PGD adversarial training). `lambda` trades the two: large `lambda` emphasizes the
natural surrogate, while small `lambda` makes sign disagreement inside the perturbation ball
costly. The bound is tight: a two-atom construction saturates it up to arbitrary slack, so there
is no uniformly sharper differentiable surrogate bound of this form.

## Algorithm (multi-class, practical form)

Replace `phi(f(x)y)` with multi-class cross-entropy `L`, replace the boundary surrogate with the
KL divergence between clean and perturbed output distributions, drop the explicit `psi^{-1}` as a
training heuristic, tune the trade-off with `beta = 1/lambda`, and solve the inner maximization
with PGD:

```
min_f  E { CE(f(x), y)  +  beta * max_{x' in B(x,eps)} KL( p(x) || p(x') ) },   p = softmax(f).
```

Per minibatch:
1. Freeze the network (eval mode). Initialize `x' = x + 0.001 * N(0, I)`
   (a tiny nudge: `x` is the global minimum of `KL(p(x)||p(x'))` with zero gradient there).
2. For `K` steps: `x' <- clamp( Pi_box( x' + eta_1 * sign( grad_{x'} KL(p(x)||p(x')) ) ), [0,1] )`,
   where `Pi_box` projects into the `eps`-box around `x` (`l_inf` steepest ascent = sign step).
3. Unfreeze (train mode). Minimize `CE(f(x), y) + beta * KL(p(x) || p(x'))` with one SGD step.

Standard training settings: `beta = 6` for the robustness-leaning setting (`beta` in `[1, 10]`
is the useful range), Gaussian initialization scale `0.001`. CIFAR-10: `eps = 0.031`, `eta_1 = 0.007`,
`K = 10`, `eta_2 = 0.1`, batch `128`, WRN-34-10. MNIST: `eps = 0.3`, `eta_1 = 0.01`, `K = 40`,
`eta_2 = 0.01`, batch `128`. Because the regularizer is label-free, it extends directly to
semi-supervised training.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def trades_loss(model, x_natural, y, optimizer,
                step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0):
    """Clean CE + beta * KL(p(x) || p(x')) boundary regularizer (l_inf)."""
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)

    # inner maximization: x' that most disagrees with the clean prediction.
    # start off-center: x is the global minimum of KL(p(x) || p(x')).
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())     # l_inf sign step
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon),
                          x_natural + epsilon)                              # project into eps-box
        x_adv = torch.clamp(x_adv, 0.0, 1.0)                               # project into [0,1]

    model.train()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    optimizer.zero_grad()

    # outer minimization: natural CE + beta * boundary KL.
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(x_adv), dim=1),
        F.softmax(model(x_natural), dim=1))                                # KL(p_nat || p_adv)
    loss = loss_natural + beta * loss_robust
    return loss
```

Drop into a standard SGD training loop: call `loss = trades_loss(...)`, then
`loss.backward(); optimizer.step()`. The single hyperparameter beyond a normal trainer is
`beta = 1/lambda`, the robustness/accuracy knob.
