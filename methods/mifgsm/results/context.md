## Research question

A trained image classifier `f(x)` can be fooled by an adversarial example `x*` that lies
close to a legitimate input `x` but is misclassified. The standard threat model bounds the
perturbation in the `L_inf` norm, `||x* - x||_inf <= eps`, so that no single pixel moves by
more than `eps` and the change stays imperceptible. Two settings matter at once. In the
**white-box** setting the attacker has the model's parameters and can compute the gradient of
the loss with respect to the input; the figure of merit is the white-box success rate (how
often the crafted example fools the very model it was crafted against). In the **black-box /
transfer** setting the attacker has no access to the target and instead crafts on a *surrogate*
model, relying on the empirical fact that adversarial examples often transfer; the figure of
merit is the transfer success rate against a held-out model.

The question is how to design a gradient-based attack that respects the same `L_inf` budget
and the same one-backprop-per-step cost, while performing well across both settings.

## Background

The vulnerability of deep networks to small, deliberately-chosen input perturbations is by
now well established (Szegedy et al. 2014, "Intriguing properties of neural networks";
Goodfellow et al. 2015). The load-bearing concepts:

- **Crafting as constrained loss-maximization.** For a correctly classified input `x` with
  label `y` and a (cross-entropy) loss `J(x, y)`, a non-targeted adversarial example solves
  `argmax_{x*} J(x*, y)` subject to `||x* - x||_inf <= eps`. The model is differentiable in
  its input, so the gradient `∇_x J(x, y)` is available by backpropagation at the cost of one
  backward pass — the same primitive used to train the network, now pointed at the input
  instead of the weights.

- **The sign of the gradient is the `L_inf`-optimal direction.** For a *linear* objective
  `η · ∇_x J` maximized over the box `||η||_inf <= eps`, the optimum is `η = eps ·
  sign(∇_x J)`: each coordinate independently takes its extreme `±eps` in the direction that
  increases the loss. This is exact for a linear loss and the first-order picture for a
  non-linear one.

- **Momentum in gradient methods (Polyak 1964; Sutskever et al. 2013).** Plain gradient steps
  on an ill-conditioned or noisy surface oscillate and stall in narrow valleys and at poor
  local optima. Accumulating a velocity vector `g_{t+1} = mu · g_t + ∇` and stepping along it
  damps oscillation, builds speed along consistent directions, and helps "barrel through"
  small humps and poor local minima/maxima. Sutskever et al. (ICML 2013) showed that
  well-tuned momentum stabilizes stochastic gradient descent enough to rival Hessian-free
  second-order optimization.

- **Why transfer happens (Liu et al. 2017, "Delving into transferable adversarial examples").**
  Different models trained on the same task learn decision boundaries that *align* well around
  a data point, which is why an example adversarial for one is often adversarial for another.
  A diagnostic geometric finding from the same study is that the input-gradient directions of
  different models are close to orthogonal even while their boundaries align.

## Baselines

The relevant prior attacks are:

**FGSM — fast gradient sign method (Goodfellow et al. 2015).** Linearize the loss around `x`,
`J(x + η, y) ≈ J(x, y) + η · ∇_x J(x, y)`, and maximize over the `L_inf` box in one shot:

```
x* = x + eps · sign(∇_x J(x, y)).
```

One backprop, no iteration. The resulting perturbation takes a single large step in the
direction every coordinate's gradient sign agrees on and transfers reasonably across models.

**I-FGSM / basic iterative method (Kurakin et al. 2016/2017).** Apply the fast gradient sign
repeatedly with a small step `alpha`, clipping back into the `eps`-ball after each step:

```
x*_0 = x,   x*_{t+1} = Clip_{x, eps}{ x*_t + alpha · sign(∇_x J(x*_t, y)) }.
```

Kurakin et al. used `alpha = 1` on the `[0,255]` pixel scale with `min(eps + 4, 1.25·eps)`
iterations; another common way to keep the total path length tied to the budget is
`alpha = eps/T`. Re-computing the gradient at the current point each step lets it follow the
curved loss surface, reaching high white-box success.

**Optimization-based attacks (Szegedy et al. 2014; Carlini & Wagner 2017).** Directly minimize
a Lagrangian `λ · ||x* - x||_p − J(x*, y)` with a general-purpose optimizer (box-constrained
L-BFGS, or Adam in C&W).

## Evaluation settings

The natural yardsticks for an `L_inf` evasion attack, as they already existed:

- **Dataset / protocol.** A set of inputs the target classifies *correctly* (so that fooling
  is meaningful) — e.g. ImageNet validation images, one per class, all initially correct.
  Pixel values in a fixed range (`[0,1]` or `[0,255]`).
- **Threat model and budget.** White-box gradient access for crafting; the crafted example
  must satisfy `||x* - x||_inf <= eps` and stay in the valid pixel range. A perturbation that
  violates the norm or the range, or is non-finite, counts as a failed attack.
- **Metrics.** Success rate = fraction of (initially correct) inputs whose adversarial version
  is misclassified, equivalently `1 − robust_acc`. Reported separately for the **white-box**
  model (crafted-on = evaluated-on) and for **black-box / transfer** targets (crafted on a
  surrogate, evaluated on a held-out model), since a method can be strong on one and weak on
  the other.
- **Knobs swept as diagnostics.** Number of iterations, step size, and the size of the
  perturbation `eps`, to see how white-box and transfer success move as each is varied.
- **Architectures.** A spread of network families as surrogate and target (Inception, ResNet,
  and similar), so transfer is measured across genuinely different architectures.

## Code framework

The attack plugs into a fixed evaluation harness. The harness owns the model, the data, the
norm/range checking, and the success-rate accounting; the one thing it does not own is the
procedure that turns a clean batch into an adversarial batch. That procedure — `run_attack` —
is the single empty slot. Everything around it already exists: a differentiable model that
maps images to logits, a cross-entropy loss, autograd to get `∇_x J`, and the elementwise
clamp/projection primitives that enforce the `L_inf` ball and the valid pixel range.

```python
import torch
import torch.nn as nn


def run_attack(
    model: nn.Module,
    images: torch.Tensor,    # (N, C, H, W), values in [0, 1]
    labels: torch.Tensor,    # (N,)
    eps: float,              # L_inf budget: ||x_adv - x||_inf <= eps
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """Craft adversarial images under an L_inf budget, white-box.

    Returns adv_images, same shape as images, values in [0, 1], with
    ||adv_images - images||_inf <= eps. The harness checks the norm and range.
    """
    model.eval()
    x = images.detach().to(device)
    labels = labels.detach().to(device)

    # TODO: the crafting procedure we will design.
    #   We may query the model and its input-gradient ∇_x J(x, y) as many
    #   times as we like, keep any per-batch state we choose, and must return
    #   a perturbed batch inside the eps-ball and the [0, 1] range.
    pass


# primitives the procedure can build on (these already exist):
#   loss_fn = nn.CrossEntropyLoss()
#   logits  = model(x)                                      # forward pass
#   loss    = loss_fn(logits, labels)
#   grad    = torch.autograd.grad(loss, x, retain_graph=False,
#                                 create_graph=False)[0]    # input gradient
#   delta  = torch.clamp(x_adv - x, -eps, eps)    # project into the L_inf ball
#   x_adv  = torch.clamp(x + delta, 0.0, 1.0)     # keep inside valid pixel range
```

The harness loads a model, selects up to a fixed number of initially-correct samples, calls
`run_attack`, validates the `L_inf` constraint and the `[0,1]` range, and reports clean
accuracy, robust accuracy, and the success rate `1 − robust_acc`. The crafting rule inside
`run_attack` is what is to be designed.
