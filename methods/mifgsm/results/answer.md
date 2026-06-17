# MI-FGSM, distilled

MI-FGSM (Momentum Iterative Fast Gradient Sign Method) is a white-box `L_inf` evasion attack
that adds momentum to the iterative sign-gradient attack. At each step it normalizes the
current input gradient, accumulates it into a velocity vector, and takes a sign-step along the
velocity. The accumulated direction stabilizes the trajectory and carries it through poor local
maxima, so the attack stays a strong white-box adversary (like the iterative attack) while
producing perturbations that transfer to black-box models (like the one-step attack) — alleviating
the usual attack-ability-vs-transferability trade-off.

## Problem it solves

Craft `x*` near `x` that the model misclassifies, under `||x* - x||_inf <= eps`, white-box.
One-step FGSM transfers but underfits the model (low white-box success at usable `eps`);
iterative I-FGSM reaches high white-box success but overfits the surrogate and transfers
poorly, worse as iterations grow. MI-FGSM wants both at once.

## Key idea

Read the attack as gradient ascent on the loss `J(x*, y)` and cure I-FGSM's greedy zig-zag
into surrogate-specific local maxima with **momentum**:

- Accumulate a velocity from the **L1-normalized** per-step gradient, so each iteration
  contributes a unit-magnitude direction and `mu` is a clean past-vs-present weight
  (raw-gradient magnitudes vary across iterations and would otherwise let one step dominate).
- Step with the **sign** of the accumulated velocity — the `L_inf`-optimal direction — which
  also bounds each per-step coordinate move by `alpha = eps / T`.
- The averaged direction cancels the components that flip sign step to step (the surrogate's
  idiosyncratic "holes" / sharp local maxima) and reinforces the component shared across
  models, which is the part that transfers. White-box strength is retained because it is still
  iterative ascent on the true loss surface.

`mu = 0` recovers I-FGSM because `sign(grad / ||grad||_1) = sign(grad)`. `T = 1`
(with `g_0 = 0` and `alpha = eps`) recovers FGSM. At `mu = 1`, `g_t` is the equal-weight
sum of all past normalized gradients.

## Final algorithm (non-targeted, `L_inf`)

```
Input:  model f with loss J; clean x, label y; budget eps; iterations T; decay mu
Output: x* with ||x* - x||_inf <= eps

alpha = eps / T
g_0   = 0;   x*_0 = x
for t = 0 .. T-1:
    grad      = grad_x J(x*_t, y)
    g_{t+1}   = mu * g_t + grad / ||grad||_1          # accumulate L1-normalized gradient
    x*_{t+1}  = x*_t + alpha * sign(g_{t+1})          # L_inf-optimal sign step
    x*_{t+1}  = clip into the eps-ball of x, then into the valid pixel range
return x*_T
```

Variants (same recipe, swap the budget geometry / objective):
- **L2 bound (MI-FGM):** `x*_{t+1} = x*_t + alpha * g_{t+1} / ||g_{t+1}||_2`.
- **Targeted (label `y*`):** accumulate `grad_x J(x*, y*)` and step `x*_{t+1} = x*_t - alpha *
  sign(g_{t+1})` for `L_inf`, or subtract `alpha * g_{t+1} / ||g_{t+1}||_2` for `L2`
  (minimize the target-class loss).
- **Ensemble of K models:** fuse logits `l(x) = sum_k w_k * l_k(x)` (`w_k >= 0`, `sum w_k = 1`),
  take cross-entropy on the fused logits, and accumulate that gradient. The logit-fused loss keeps
  pre-softmax detail while forcing one perturbation to satisfy several models at once.

## Defaults and why

- `T = 10` iterations; `alpha = eps / T` so aligned nonzero sign-steps fill the budget while
  each step's `L_inf` size is bounded by `alpha`.
- `mu = 1.0`: maximal undiscounted accumulation of the consensus direction without raw-gradient
  scale blow-up, because each newly added gradient is normalized before accumulation.
- L1 normalization per iteration: gradient magnitudes vary across steps; normalizing keeps the
  velocity an average of *directions* with each step voting equally. The specific norm is less
  important than the per-step scale normalization; the accumulated sign should reflect persistent
  direction, not whichever raw gradient happened to have the largest magnitude.

## Working code

Filling the `run_attack` slot of the evaluation harness (faithful to the canonical
`torchattacks` MIFGSM update order; the step size below uses the `alpha = eps / T`
schedule from the method derivation):

```python
import torch
import torch.nn as nn


def run_attack(
    model: nn.Module,
    images: torch.Tensor,    # (N, C, H, W), values in [0, 1]
    labels: torch.Tensor,    # (N,)
    eps: float,              # L_inf budget
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    model.eval()
    steps = 10
    alpha = eps / steps      # alpha = eps / T
    decay = 1.0              # mu

    x = images.detach().to(device)
    labels = labels.detach().to(device)
    adv = x.clone().detach()
    momentum = torch.zeros_like(x)                   # g_0 = 0
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        adv.requires_grad = True
        outputs = model(adv)
        cost = loss_fn(outputs, labels)              # maximize J(x*, y)
        grad = torch.autograd.grad(
            cost, adv, retain_graph=False, create_graph=False
        )[0]

        # mean-abs is proportional to the per-sample L1 norm in the formula; the fixed
        # factor preserves the same accumulated sign direction for fixed-size images.
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * decay               # g_{t+1} = normalized grad + mu * g_t
        momentum = grad

        adv = adv.detach() + alpha * grad.sign()              # L_inf-optimal step
        delta = torch.clamp(adv - x, min=-eps, max=eps)        # project into eps-ball
        adv = torch.clamp(x + delta, min=0, max=1).detach()    # keep in [0, 1]

    return adv
```

Targeted / L2 differences are exactly the swaps above: use the target-class loss with a negative
step, or replace the accumulated `grad.sign()` step by `grad / ||grad||_2`.
