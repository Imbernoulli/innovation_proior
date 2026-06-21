## Research question

White-box `L_inf` evasion: I have full access to a trained image classifier — parameters and gradients — a clean image it classifies correctly, and a strict per-pixel budget `eps = 2/255`. The only thing I design is the **attack procedure** that turns the clean batch into an adversarial batch inside that budget. The model, the data, and the scoring are fixed. The objective is to maximize attack success rate (ASR): the fraction of initially-correct samples pushed to a wrong prediction while every pixel stays within `eps` and the image stays in `[0,1]`. On these undefended classifiers, the larger `8/255` budget already saturates ASR, so the tighter `2/255` regime is where attack quality is measurable, especially on VGG where even strong attacks leave ASR below 1.

## Prior art / Background / Baselines

- **Szegedy et al. (2014).** They cast adversarial examples as box-constrained minimization of `c·||r||_2 + loss_f(x + r, l)` and solve it per image with L-BFGS plus a line search over `c`. Gap: each image needs its own iterative constrained optimization, which is too slow to run at scale or to embed in a training loop.

## Fixed substrate / Code framework

The evaluation harness is frozen. Each scenario is a `(model, dataset)` pair from {ResNet20, VGG11-BN, MobileNetV2} × {CIFAR-10, CIFAR-100}, using public pretrained checkpoints from `chenyaofo/pytorch-cifar-models`. For each scenario the harness:

1. Loads the trusted **score model** and collects up to 1000 samples it initially classifies correctly.
2. For every batch, hands my code a **fresh deep copy** of the score model, the clean `images` in `[0,1]`, the `labels`, the budget `eps = 2/255`, the `device`, and `n_classes`.
3. Checks the output is finite, lies in `[0,1]`, and satisfies per-sample `||x_adv − x||_inf ≤ eps + 1e-6`; any violation is an **attack failure** for that sample.
4. Scores `robust_acc` with the score model and reports `asr = 1 − robust_acc` over the initially-correct samples.

The budget is enforced per-sample, so my code must project into the `eps`-box and into `[0,1]` itself or forfeit the sample. I see the model only through the copy passed to `run_attack` — a pure white-box first-order interface (forward + `torch.autograd.grad`); there is no access to the score model that grades me.

## Editable interface

Exactly one region is editable — the body of `run_attack` in `torchattacks/bench/custom_attack.py`. The contract is fixed:

`run_attack(model, images, labels, eps, device, n_classes) -> adv_images`

with `images` of shape `(N, C, H, W)` in `[0,1]`, `labels` of shape `(N,)`, and `adv_images` the same shape, in `[0,1]`, within `eps` of `images` in `L_inf`. The starting scaffold returns the clean images unchanged.

```python
import torch
import torch.nn as nn

# =====================================================================
# EDITABLE: implement run_attack below
# =====================================================================
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """
    White-box L_inf adversarial attack.
    images: (N, C, H, W) in [0, 1] on device. labels: (N,) on device.
    n_classes: 10 for CIFAR-10, 100 for CIFAR-100.
    Returns adv_images satisfying (adv_images - images).abs().max() <= eps.
    """
    _ = (model, labels, eps, device, n_classes)
    return images.clone()

# =====================================================================
# END EDITABLE REGION
# =====================================================================
```

## Evaluation settings

Five scored scenarios — ResNet20-C10, VGG11BN-C10, ResNet20-C100, VGG11BN-C100, and the hidden MobileNetV2-C100 — plus MobileNetV2-C10 where it is reported. One seed, `42`. Budget `eps = 2/255`. Up to 1000 initially-correct samples per scenario, batch size 100. The single metric, higher is better, is `asr = 1 − robust_acc` per scenario. The reference baseline implementations live in the `torchattacks` package; the harness exposes only the white-box first-order interface above.
