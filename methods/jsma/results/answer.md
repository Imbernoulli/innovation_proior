# JSMA — the Jacobian-based Saliency Map Attack

## Problem

Given a frozen, differentiable feedforward classifier `F` and a clean input `X` (image with
features in `[0,1]`) correctly labeled `Y`, craft `X*` that the network labels as a chosen
target `Y* != Y` while modifying **as few input features as possible** — an `L0`-constrained,
targeted, white-box, test-time attack. Unlike `L_inf`/`L2` attacks (FGSM, L-BFGS), the
perturbation is sparse: a few pixels, each allowed to move arbitrarily far within `[0,1]`.

## Key idea

Differentiate the network's **outputs** (logits), not its loss, with respect to the **input**.
This Jacobian — the *forward derivative* `∂F_j/∂x_i` — is a per-feature, per-class, *signed*
sensitivity map. From it, build an **adversarial saliency map** scoring each feature by how much
increasing it both raises the target logit and lowers all others. Greedily saturate the
highest-scoring feature(s), recompute, and repeat until the label flips or the feature budget is
spent. Greedy saturation is what keeps the support (the `L0` count) small.

## The method

**Forward derivative (Jacobian of the outputs).** For an acyclic feedforward `F` with
differentiable activations,
```
∂F_j/∂x_i = f'_{n+1,j}(W_{n+1,j}·H_n + b_{n+1,j}) · ( W_{n+1,j} · ∂H_n/∂x_i ),
∂H_k/∂x_i|_p = f'_{k,p}(W_{k,p}·H_{k-1} + b_{k,p}) · ( W_{k,p} · ∂H_{k-1}/∂x_i ),   k = 2..n,
```
with base case `∂H_0/∂x_i = e_i`. In practice: one backward pass per output neuron seeds row
`j` of the `N×M` Jacobian.

**Adversarial saliency map (increasing version).** For target `t`,
```
S(X,t)[i] = 0,                                            if ∂F_t/∂x_i < 0  or  Σ_{j≠t} ∂F_j/∂x_i > 0
          = (∂F_t/∂x_i) · | Σ_{j≠t} ∂F_j/∂x_i |,          otherwise.
```
Gate out any feature that lowers the target or helps the rest; among the rest, score = (target
sensitivity) × (magnitude of the negative others-sensitivity). Compute on the **logits**, not
the softmax, so the two conditions carry independent information.

For decreasing features, the signs flip because the input step is negative: gate on
`∂F_t/∂x_i < 0` and `Σ_{j≠t} ∂F_j/∂x_i > 0`, then score
`|∂F_t/∂x_i| · (Σ_{j≠t} ∂F_j/∂x_i)`.

**Pairs, not singletons.** Few single features pass both sign gates, so search over **pairs**
`(p1,p2)`, summing each derivative over the pair before the product:
```
argmax_{(p1,p2)} ( Σ_{i∈{p1,p2}} ∂F_t/∂x_i ) · | Σ_{i∈{p1,p2}} Σ_{j≠t} ∂F_j/∂x_i |,
   left factor > 0, right factor < 0.
```
One feature can compensate the other's sign flaw. Larger groups cost too much (`O(M^g)`).

**Greedy loop.** Each iteration: recompute the Jacobian at the current `X*` (the network is
non-linear, so sensitivities shift), pick the best pair, push both by `θ = +1` (saturate to
the maximum — minimizes the number of distinct features needed), clamp to `[0,1]`, and drop
saturated features from the search domain `Γ`. Halt when `argmax_j F_j(X*) = t`, the iteration
cap is reached, or `Γ` is empty. With a fractional feature budget `Υ`, the original percentage
form is `max_iter = floor(num_features·Υ/2)`; in the torchattacks-style count form below, use
`ceil(pixels/2)`, matching `ceil(num_features·gamma/2)` when `pixels = num_features·gamma`.
(`÷2` because two features are committed per iteration; `increasing` `θ>0` is more reliable
than decreasing, which removes input information.)

## Working code (faithful to torchattacks.JSMA)

```python
import numpy as np
import torch
import torch.nn as nn


def compute_jacobian(model: nn.Module, x: torch.Tensor, n_classes: int,
                     device: torch.device) -> torch.Tensor:
    """Forward derivative: row c is dF_c/dx over all input features (one backward pass per class)."""
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)                                  # pre-softmax outputs
    num_features = int(np.prod(x.shape[1:]))           # C*H*W
    jac = torch.zeros(n_classes, num_features, device=device)
    for c in range(n_classes):
        if x.grad is not None:
            x.grad.zero_()
        logits[0, c].backward(retain_graph=True)
        jac[c] = x.grad.reshape(-1).clone()
    return jac


@torch.no_grad()
def saliency_pair(jac, target, increasing, search_domain, num_features):
    """argmax over feature pairs of the adversarial saliency score (Eqs. above)."""
    all_sum = jac.sum(dim=0, keepdim=True)
    target_grad = jac[target]
    others_grad = (all_sum - target_grad).squeeze(0)

    out = (search_domain == 0).float()                 # features outside the search domain
    increase_coef = (2.0 if increasing else -2.0) * out
    target_grad = target_grad - increase_coef * target_grad.abs().max()
    others_grad = others_grad + increase_coef * others_grad.abs().max()

    alpha = target_grad.view(1, -1) + target_grad.view(-1, 1)   # pair-sum of dF_t/dx
    beta = others_grad.view(1, -1) + others_grad.view(-1, 1)    # pair-sum of dF_others/dx

    if increasing:
        mask = (alpha > 0) & (beta < 0)
        scores = alpha * beta.abs()
    else:
        mask = (alpha < 0) & (beta > 0)
        scores = alpha.abs() * beta
    eye = torch.eye(num_features, dtype=torch.bool, device=jac.device)
    mask = mask & ~eye                                  # no self-pairs

    scores = scores * mask.float()
    flat = scores.view(-1).argmax()
    return int(flat // num_features), int(flat % num_features)


def attack_one(model, x, target, pixels, n_classes, device, theta=1.0):
    increasing = theta > 0
    num_features = int(np.prod(x.shape[1:]))
    shape = x.shape
    max_iters = int(np.ceil(pixels / 2.0))             # 2 features committed per iteration

    if increasing:
        search_domain = (x < 0.99).reshape(num_features).clone()
    else:
        search_domain = (x > 0.01).reshape(num_features).clone()

    pred = model(x).argmax(1)
    it = 0
    while it < max_iters and pred.item() != target and search_domain.sum() != 0:
        jac = compute_jacobian(model, x, n_classes, device)     # recompute: F is non-linear
        p, q = saliency_pair(jac, target, increasing, search_domain, num_features)
        flat = x.reshape(-1).clone()
        flat[p] += theta                               # saturate the two selected features
        flat[q] += theta
        x = flat.clamp(0.0, 1.0).reshape(shape)
        search_domain[p] = 0                           # drop saturated features
        search_domain[q] = 0
        pred = model(x).argmax(1)
        it += 1
    return x


def run_attack(model, images, labels, pixels, device, n_classes):
    model.eval()
    images = images.to(device)
    adv = []
    for x, y in zip(images, labels.to(device)):
        x = x.unsqueeze(0)
        target = int((y + 1) % n_classes)              # a chosen target class != true label
        adv.append(attack_one(model, x, target, pixels, n_classes, device).clamp(0, 1))
    return torch.cat(adv, 0)
```

## Why it works / what it buys

- **`L0` sparsity by construction:** greedy saturation touches one (pair of) feature(s) at a
  time and drops them once saturated, so the support stays small — the count of changed features,
  not their magnitude, is what's controlled.
- **Targeted, two-sided saliency:** scoring on the signed forward derivative — target up *and*
  others down — gives directional, class-aware feature selection that a loss gradient or a
  sign-blind magnitude map cannot.
- **Tracks the moving boundary:** recomputing the Jacobian each iteration re-ranks features
  against the network's updated (non-linear) sensitivities.
- **Cost knob:** pairs (`O(M^2)`/iter) balance the strictness of the both-signs gate against the
  combinatorial blow-up of larger feature groups.
