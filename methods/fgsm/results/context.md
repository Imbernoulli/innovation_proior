## Research question

A trained image classifier `f` with parameters `θ` and a differentiable training loss
`J(θ, x, y)` assigns the right label to almost every naturally occurring image, yet there
exist inputs `x̃ = x + η` that are visually indistinguishable from a correctly classified `x`
— every pixel changed by less than the precision the sensor or 8-bit encoding can even
represent — that the model confidently misclassifies. These *adversarial examples* have a
striking property: the same perturbed image often fools a *different* network, with a different
architecture, trained on a *disjoint* slice of the data, and the different models tend to
agree on the (wrong) class.

The concrete question is how to produce such a perturbation. The only known generator runs an
iterative constrained optimization per image (below). The setting: given full white-box access
to `θ` and the gradient `∇_x J`, and a strict per-pixel budget `||η||_∞ ≤ ε`, produce a
perturbation that raises the loss while keeping the result a valid image (pixels in `[0, 1]`).

## Background

The discovery that set this up (Szegedy, Zaremba, Sutskever, Bruna, Erhan, Goodfellow &
Fergus, ICLR 2014): a trained network's input→output map is far less locally smooth than the
kernel-method intuition assumes. For a correctly classified `x` one can find a tiny `r` with
`x + r` misclassified, and the empirical findings about these `r` are the load-bearing facts
here:

- **Imperceptibility.** On ImageNet the perturbations were so small the difference was
  invisible to the eye; on MNIST the average per-pixel distortion to reach 0% accuracy was on
  the order of 0.05–0.1 in `[0, 1]` units.
- **Cross-model transfer.** An example crafted for one network was misclassified by networks
  with different depth, width, regularization, or initial weights at substantial rates
  (tens of percent), far above what random distortion of the same magnitude induces.
- **Cross-training-set transfer.** Splitting MNIST into two disjoint halves and training
  separate networks on each, examples crafted on one still fooled the other — degraded but
  well above the random-noise baseline.
- **Random noise is a weak attack.** Gaussian noise of the *same* per-pixel magnitude leaves
  most predictions correct (e.g. half of badly noised MNIST digits still classified right),
  whereas the crafted examples are essentially never classified correctly. The crafted
  direction matters; the magnitude alone does not.
- **Shallow linear models are vulnerable too.** Plain softmax / logistic regression on pixels
  has the same problem.
- **Adversarial training.** Mixing crafted examples into training regularizes (Szegedy
  reported MNIST gains), using a constrained optimizer in the inner loop.

The conceptual tools available. A model "behaves linearly" wherever its loss is well
approximated by its first-order Taylor expansion around the input,
`J(θ, x + η, y) ≈ J(θ, x, y) + η^T ∇_x J(θ, x, y)`; the gradient `∇_x J` is computable by
backpropagation in one backward pass. Many of the architectures in use are easy to optimize
because of their piecewise-linear or non-saturating components — rectified linear units, maxout
units, LSTM gating, and sigmoid networks kept in their non-saturating regime. A separate fact
about feature precision concerns the choice of norm: 8-bit image encodings discard everything
below `1/255` of the dynamic range, so a change to a single feature that is below the sensor's
precision is semantically irrelevant — which makes a *per-coordinate* (max-norm) budget a
natural one, rather than an aggregate `ℓ_2` budget.

## Baselines

**Box-constrained L-BFGS attack (Szegedy et al., ICLR 2014).** The method that first produced
adversarial examples. To push `x` to a chosen target label `l`, solve

```
minimize ||r||_2   subject to   f(x + r) = l ,   x + r ∈ [0, 1]^m ,
```

i.e. find the smallest `ℓ_2` distortion that changes the label while staying a valid image.
The hard label constraint is relaxed by a penalty and a line search over `c > 0`:

```
minimize  c·||r||_2 + loss_f(x + r, l)   subject to   x + r ∈ [0, 1]^m ,
```

solved with box-constrained L-BFGS, increasing `c` until `f(x + r) = l`. For a convex loss
this recovers the true minimum-distortion point; for a network it is an approximation. It
finds tiny, transferable perturbations by running an iterative constrained optimization, with a
line search over the penalty weight, per example.

**Random per-pixel noise (control).** Add `±ε` to each pixel, or draw each pixel's
perturbation from `U(−ε, ε)`. A zero-mean perturbation has expected inner product zero with any
fixed reference direction, so on average it does not move the loss in a particular direction;
the same per-pixel magnitude that a crafted perturbation uses to flip the label leaves most
random-noise inputs correctly classified. It establishes that *direction*, not magnitude, is
what a successful perturbation gets right.

**Generic regularization and noise augmentation.** Dropout, weight decay, model averaging, and
random input noise are available ways to reduce overfitting or improve local stability. They
operate over the training distribution rather than asking, for the current input and current
model, which point inside the allowed pixel box increases the loss.

## Evaluation settings

- **Datasets / ranges.** MNIST and CIFAR-10 (and an ImageNet demonstration), pixel values
  scaled to `[0, 1]`. Perturbation budgets `ε` are stated in those same units (e.g. `≈ 1/255`
  for an 8-bit ImageNet image; larger values such as `0.1`–`0.25` on MNIST/CIFAR).
- **Models.** Shallow softmax / logistic regression on raw pixels; maxout networks; a
  convolutional maxout network; radial-basis-function networks; and, for transfer studies,
  multiple networks differing in depth, width, initialization, or training subset.
- **Metrics.** Error rate (fraction misclassified) on perturbed inputs, and the model's
  average confidence on the examples it gets wrong; for transfer, the error rate of model B on
  perturbations crafted against model A, and the rate at which two models assign the *same*
  wrong class.
- **Protocol.** A pretrained classifier is loaded; up to 1000 inputs that it
  initially classifies correctly are collected; the attack is run under `||x_adv − x||_∞ ≤ ε`
  with `ε = 2/255` and `x_adv ∈ [0, 1]`; outputs that violate the norm, leave `[0, 1]`, or are
  non-finite count as failures; the reported attack success rate is `ASR = 1 − robust_acc`
  over the initially-correct inputs.

## Code framework

The attack plugs into a white-box harness that already owns the model, a batch of clean images
in `[0, 1]`, their labels, the budget `ε`, and automatic differentiation to obtain `∇_x` of the
classification loss. The single empty slot is the rule that turns the input gradient and the
budget into one candidate image, followed by the validity clamp that keeps pixels in `[0, 1]`.

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
    loss = loss_fn(logits, y)
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]

    with torch.no_grad():
        # TODO: from `grad` and `eps`, form one candidate image, then clamp it to [0, 1].
        x_adv = x  # placeholder
    return x_adv.detach()
```

The outer harness supplies the model, labels, budget, loss, gradient, and validity clamp; the
empty branch is where the perturbation rule will live.
