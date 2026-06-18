**Problem.** MI-FGSM saturated the easy scenarios (0.98–0.999 on ResNet/MobileNetV2) but VGG stalled
at 0.714 / 0.720. Its momentum was justified by *transfer*, but here attack and grader are the same
model, so a stale velocity can blunt the inner maximizer — overshooting or oscillating near the
high-loss boundary instead of settling on it. I want the strongest faithful solution to the inner
maximization on this one model.

**Key idea.** Projected gradient ascent: many small `L_inf`-steepest steps that re-read the current
gradient, each projected back into the feasible set. Drop the velocity entirely. The Euclidean
projection onto the `L_inf` box *is* coordinate-wise clipping of the perturbation to `[−eps,eps]`, and
the budget∩`[0,1]` intersection is the interval `[max(x_i−eps,0), min(x_i+eps,1)]` — implemented by
clipping the perturbation then the image. Launch from a uniform random point in the `eps`-box to dodge
the corrupted gradient at the clean point and to sample a different basin of the non-concave inner max.

**Why it works here.** With no momentum, each step is faithful local sign-ascent of the *current* loss,
so a larger step `alpha = eps/4` explores the box fast without a stale velocity overshooting; `40` steps
give `10·eps` reach (five box diameters) so late steps refine on the high-loss boundary. Random restarts
concentrate on a common loss ceiling across distinct maxima — the basis for calling this the strongest
*first-order* attack. Danskin's theorem makes this maximizer the exact object a robust-training defense
would call, so "solve the inner max well" is the principled target.

**Hyperparameters.** `steps = 40`, `alpha = eps/4`, random start `uniform(-eps,eps)` clamped to
`[0,1]`, cross-entropy loss (pluggable; margin-loss is the drop-in), per-step projection of the
perturbation to `[−eps,eps]` then the image to `[0,1]`. No momentum.

```python
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torch.nn.functional as F

    _ = (device, n_classes)
    model.eval()
    steps = 40
    alpha = eps / 4.0

    x = images.detach()
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)

        x_adv = x_adv.detach()

    return x_adv
```
