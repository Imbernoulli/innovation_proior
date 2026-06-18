**Problem.** FGSM left clear headroom — 0.617 / 0.621 ASR on VGG11BN-C10/C100 versus 0.82–0.88 on the
ResNet and MobileNetV2 scenarios. One signed step maximizes the loss's tangent plane at the clean
point; over a full-budget jump that plane mis-describes the curved loss, and the VGG survivors are
curvature-underfit, not safe. I need to follow the curved surface, not bet on one plane.

**Key idea.** Iterate the `L_inf`-steepest sign step, re-reading the gradient each step — but stabilize
the direction with momentum so the ascent does not zig-zag into shallow sharp maxima on the bumpy loss
surface. Accumulate a velocity `g_{t+1} = μ·g_t + ∇_x J / ||∇_x J||_1` (per-iteration `L_1`
normalization so each step votes equally, not whichever step had the largest gradient), then take the
`L_inf`-optimal step `x_{t+1} = x_t + alpha·sign(g_{t+1})`, projecting back into the `eps`-box and
`[0,1]`. `μ = 0` collapses to iterative FGSM; one step from the clean point collapses to FGSM — the
family contains both ancestors.

**Why it works here.** Iterating recovers FGSM's underfit on the curved (VGG) survivors; momentum plus
a random start gives a steadier, less-trapped ascent than greedy iterative FGSM. In this single-model
white-box setting transfer is not the operative reason (I attack and am graded by the same
architecture); the gain is a stronger inner maximizer. `T = 40`, `alpha = eps/10` gives `4·eps` total
reach (well past the `2·eps` diameter) so late steps refine on the high-loss boundary; the random
launch dodges the corrupted gradient at the clean point.

**Hyperparameters.** `steps = 40`, `alpha = eps/10`, `decay = 1.0`, random start `uniform(-eps,eps)`
clamped to `[0,1]`, mean-abs gradient normalization (`L_1`-proportional), cross-entropy loss, per-step
projection of the perturbation to `[-eps,eps]` then the image to `[0,1]`.

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
    alpha = eps / 10.0
    decay = 1.0

    x = images.detach()
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    momentum = torch.zeros_like(x)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        momentum = decay * momentum + grad

        with torch.no_grad():
            x_adv = x_adv + alpha * momentum.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)

        x_adv = x_adv.detach()

    return x_adv
```
