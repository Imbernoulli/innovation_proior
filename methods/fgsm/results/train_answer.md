The research question is how to produce adversarial examples for a trained image classifier under a strict per-pixel budget. We are given full white-box access, so the input gradient is available, but the only existing generator is an iterative box-constrained L-BFGS attack that solves a penalized optimization with a line search over the penalty weight for every single image. That finds tiny, transferable perturbations, yet it is far too expensive to run at scale or to embed inside a training loop. Random uniform noise is cheap, but it is not an attack: its expected inner product with any fixed direction is zero, so most noised inputs remain correctly classified. What is missing is a closed-form perturbation that costs about one backward pass and still reliably raises the loss.

The right explanation is not extreme nonlinearity or overcapacity. Adversarial examples arise from linearity in high dimensions. Consider a single linear unit whose activation changes by w^T eta when the input is perturbed by eta. If every coordinate of eta is bounded by epsilon in absolute value, the worst case is obtained coordinate by coordinate: push each pixel by epsilon in the direction of the corresponding weight sign. That gives eta = epsilon sign(w) and an activation change of epsilon times the L1 norm of w. The per-pixel budget does not grow with dimension, but the coherent output swing grows linearly with the number of input features, so many individually imperceptible changes add up to a large, damaging signal. This already explains why even a shallow linear classifier is vulnerable.

Modern networks are deliberately built to be nearly linear, because piecewise-linear activations such as ReLU and maxout, and gated units kept in a non-saturating regime, are what make deep models optimizable. So the same move should work on the classifier loss J. Linearize the loss around the clean input x: J(theta, x + eta, y) is approximately J(theta, x, y) plus eta^T g, where g is the gradient of the loss with respect to the input. This gradient is obtained by backpropagation, just stopping at the input instead of the weights. We now maximize eta^T g subject to the per-pixel constraint that every coordinate of eta has absolute value at most epsilon.

By Holder's inequality for the conjugate pair infinity and one, eta^T g is bounded by the L-infinity norm of eta times the L1 norm of g, which is at most epsilon times the L1 norm of g. Equality is achieved by pushing every coordinate to the boundary in the direction of the gradient sign, so a maximizer is eta = epsilon sign(g). The feasible set is an axis-aligned cube and the objective is linear, so the optimum lies at a corner. Under an L-infinity budget, what matters is the sign of each partial derivative, not its magnitude, because every pixel has its own independent allowance of epsilon. The L-infinity norm is the natural choice because image precision is per-coordinate: an 8-bit sensor discards changes below one part in 255, so imperceptibility should be enforced coordinate-wise rather than through an aggregate L2 budget.

This yields the Fast Gradient Sign Method, or FGSM. It computes one forward pass and one backward pass, takes the elementwise sign of the input gradient, scales by epsilon, adds the result to the clean image, and clamps the pixels back to the valid range. For a model that is genuinely linear in the input, the step is exact: it is the true worst-case perturbation inside the epsilon box. For nonlinear networks it is a first-order approximation, but it remains effective because those networks are locally near-linear, which is also why the resulting perturbations transfer across architectures and training sets.

The method is fast enough to make adversarial training practical. Instead of precomputing expensive L-BFGS examples, one can regenerate the worst-case point inside every minibatch against the current weights by minimizing a blend of the clean loss and the loss on x plus epsilon sign(g). Because the perturbation is recomputed each step, the attack chases the model as it trains. The sign function has zero derivative almost everywhere, so the perturbation is treated as constant when taking the outer gradient step; the model simply learns to be correct on the current worst-case point.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_attack(
    model: nn.Module,
    images: torch.Tensor,   # (N, C, H, W), values in [0, 1]
    labels: torch.Tensor,   # (N,)
    eps: float,             # per-pixel L-inf budget
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    _ = (device, n_classes)
    model.eval()
    x = images.detach().clone().requires_grad_(True)
    y = labels.detach().clone()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]

    with torch.no_grad():
        # eta = eps * sign(g): the L-inf-ball maximizer of the linearized loss increase
        x_adv = x + eps * grad.sign()
        delta = torch.clamp(x_adv - images, min=-eps, max=eps)
        x_adv = torch.clamp(images + delta, min=0.0, max=1.0)

    return x_adv.detach()
```
