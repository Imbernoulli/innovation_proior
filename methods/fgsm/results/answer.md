# Fast Gradient Sign Method (FGSM), distilled

FGSM generates an adversarial example in one step by moving every input coordinate by a fixed
budget `ε` in the direction of the sign of the loss gradient:

```
x_adv = x + ε · sign(∇_x J(θ, x, y)),   then clip to the valid range.
```

It is the closed-form maximizer of the *first-order* increase of the loss over an `ℓ_∞` ball,
so it costs a single forward + backward pass — orders of magnitude cheaper than an iterative
constrained optimizer — which is what makes both large-scale analysis and adversarial training
practical.

## Problem it solves

Given white-box access to a classifier (parameters and gradients), a clean input `x ∈ [0,1]^m`,
its label `y`, and a per-pixel budget `||η||_∞ ≤ ε`, produce a perturbation `η` that reliably
raises the loss (drives misclassification) at the cost of about one backward pass, with
`x + η` still a valid image.

## Key idea

Adversarial examples come from **linearity in high dimensions**, not from extreme
nonlinearity. For a linear unit, perturbing the input changes the activation by `w^T η`; under
`||η||_∞ ≤ ε` this is maximized by `η = ε·sign(w)`, giving `w^T η = ε ||w||_1 = ε·m·n`. The
budget `ε` does not grow with dimension `n`, but the activation change grows linearly with
`n`, so many individually-imperceptible per-pixel changes sum coherently into a large output
swing ("accidental steganography").

Modern networks are deliberately near-linear (ReLU, maxout, gated/non-saturating units) to be
trainable, so the same move applies to the **loss**. Linearize `J` around `x`:

```
J(θ, x + η, y) ≈ J(θ, x, y) + η^T ∇_x J(θ, x, y),   g := ∇_x J(θ, x, y),
```

and maximize the increase over the `ℓ_∞` ball. By Hölder's inequality with the conjugate pair
`(∞, 1)`,

```
η^T g ≤ ||η||_∞ ||g||_1 ≤ ε ||g||_1,
```

the bound is attained by `η_i = ε·sign(g_i)` for each nonzero `g_i` (zero-gradient coordinates
are arbitrary, and `0` is the canonical choice). Thus a canonical argmax is

```
η* = ε·sign(g),
max_{||η||_∞ ≤ ε} η^T g = ε ||g||_1.
```

The feasible set is an axis-aligned cube and the objective is linear, so the maximum is a
corner. Only the *sign* of each partial derivative matters under `ℓ_∞`, because every coordinate
has its own independent allowance `ε`. The `ℓ_∞` norm is the natural one because feature
precision is per-coordinate: a change to one pixel below the sensor/encoding precision is
semantically irrelevant.

## Why one step

The goal is speed. For a model that is genuinely linear in `x` (logistic regression) the step
is **exact** — the true worst case in the box, not an approximation. For nonlinear nets it is
a first-order approximation, but it reliably fools them, which is itself evidence for the
linearity hypothesis. Cheapness is what makes adversarial training feasible.

## Logistic-regression special case

With `P(y=1) = σ(w^T x + b)` and softplus loss `ζ(z) = log(1 + e^z)`, the input-gradient sign
is

```
sign(∇_x J) = sign(−y w) = −y·sign(w),
```

so `η = ε·sign(∇_x J) = −y·ε·sign(w)`. Since
`w^T sign(w) = Σ_i |w_i| = ||w||_1`, the signed margin changes by
`y w^T η = −ε ||w||_1`. The literal `±1` worst-case loss is
`E_{x,y}[ζ(ε ||w||_1 − y(w^T x + b))]`; in the compact binary-margin notation used for the
weight-decay comparison, training on the worst-case perturbation gives

```
minimize  E_{x,y} [ ζ( y(ε ||w||_1 − w^T x − b) ) ].
```

The `ε ||w||_1` term resembles `ℓ_1` regularization but is **subtracted inside the
activation** rather than added to the cost, so it deactivates once the example has comfortable
margin (`ζ` saturates). Plain `ℓ_1` weight decay is therefore strictly more pessimistic; it
over-estimates achievable damage and needs a coefficient smaller than `ε`.

## Adversarial training (harnessing)

Because FGSM is cheap, regenerate worst-case points each minibatch against the current model:

```
J̃(θ, x, y) = α J(θ, x, y) + (1 − α) J(θ, x + ε·sign(∇_x J(θ, x, y)), y),   α = 0.5.
```

This minimizes an upper bound on the loss over the `ε` max-norm box (worst-case input noise).
Random zero-mean noise is far weaker — its expected inner product with `g` is zero, so on
average it does not raise the loss; FGSM picks the single damaging corner. Since `sign(·)` has
zero/undefined derivative, the perturbation is held fixed when taking the outer gradient step.

## Why adversarial examples generalize

Under the linear view the misclassification region is a broad subspace (any `η` with
`η^T g > 0` and large enough `ε`), not a fine pocket; sweeping `ε` moves the logits
piecewise-linearly and the wrong class is stable over a wide band. Different models trained on
the same task learn approximately the same linear reference weights, so their gradient
directions align — hence examples transfer across architectures and disjoint training sets,
and the models tend to agree on the wrong class.

## Working code

Filling the perturbation slot of the white-box `ℓ_∞` harness (untargeted; `device` and
`n_classes` are unused because the attack needs only the loss gradient and the budget):

```python
import torch
import torch.nn as nn


def run_attack(
    model: nn.Module,
    images: torch.Tensor,   # (N, C, H, W), values in [0, 1]
    labels: torch.Tensor,   # (N,)
    eps: float,             # per-pixel L-inf budget
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    _ = n_classes
    model.eval()
    x = images.detach().clone().to(device).requires_grad_(True)
    y = labels.detach().clone().to(device)
    logits = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)                    # J(theta, x, y)
    grad = torch.autograd.grad(
        loss, x, retain_graph=False, create_graph=False
    )[0]                                         # g = dJ/dx

    with torch.no_grad():
        x_adv = x + eps * grad.sign()            # eta = eps * sign(g): L-inf-ball maximizer
        x_adv = torch.clamp(x_adv, min=0, max=1)

    return x_adv.detach()
```
