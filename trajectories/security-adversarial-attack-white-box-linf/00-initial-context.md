## Research question

White-box `L_inf` evasion: I have full access to a trained image classifier — parameters and
gradients — a clean image the model classifies correctly, and a strict per-pixel budget
`eps = 2/255`. The single thing I design is the **attack procedure** that turns the clean batch into
an adversarial batch inside that budget. Everything else — the model, the data, the way success is
scored — is fixed. The number being maximized is attack success rate (ASR): the fraction of
initially-correct samples I push to a wrong prediction while every pixel stays within `eps` and the
image stays in `[0,1]`. The `8/255` budget RobustBench uses for *defended* models saturates ASR to
~1.0 on these undefended classifiers and leaves no headroom; the tight `2/255` regime is what makes
attack quality measurable, especially on the wider VGG architecture where even strong attacks leave
ASR well below 1.

## Prior art before the first rung (white-box attack lineage)

The first rung reacts to the existence proof that adversarial examples exist at all, and to the cost
of the only method that originally produced them.

- **Existence of adversarial examples (Szegedy et al. 2014).** For a correctly classified `x` there
  is a tiny `r`, below visual perception, with `x + r` misclassified; the same crafted `r` often fools
  a *different* model, and random noise of the same size does not. Produced by minimizing
  `c·||r||_2 + loss_f(x + r, l)` over the box with box-constrained L-BFGS and a line search over `c`.
  Gap: an iterative constrained optimizer with a line search **per single image** — far too slow to
  study at scale or to fold into a training loop, and it gives no cheap closed-form move.

That is the lineage the first rung answers: a cheap, closed-form, one-pass attack that needs no inner
optimizer. The rungs above it then climb from that single step toward the strongest reference attack.

## The fixed substrate

The evaluation harness is frozen and must not be touched. Each scenario is a `(model, dataset)` pair
drawn from {ResNet20, VGG11-BN, MobileNetV2} × {CIFAR-10, CIFAR-100}, using public pretrained
checkpoints loaded from `chenyaofo/pytorch-cifar-models`. For each scenario the harness:

1. loads the trusted **score model** and collects up to 1000 samples it initially classifies
   correctly;
2. for every batch, hands my code a **fresh deep copy** of the score model (so I cannot tamper with
   its weights, BatchNorm statistics, or hooks), the clean `images` in `[0,1]`, the `labels`, the
   budget `eps = 2/255`, the `device`, and `n_classes` (10 or 100);
3. checks the output is finite, lies in `[0,1]`, and satisfies per-sample
   `||x_adv − x||_inf ≤ eps + 1e-6` — any violation (wrong shape, non-finite, out of range, over
   budget) is counted as an **attack failure** for that sample;
4. scores `robust_acc` with the score model and reports `asr = 1 − robust_acc`, denominator = the
   number of initially-correct samples.

Two consequences shape every rung. First, the budget is enforced per-sample by the harness, so my
code must project into the `eps`-box and into `[0,1]` itself or it forfeits the sample. Second, I see
the model only through the copy passed to `run_attack` — a pure white-box first-order interface
(forward + `torch.autograd.grad`); there is no access to the score model that grades me, so an attack
that overfits a private quirk of the copy still has to flip the score model's decision.

## The editable interface

Exactly one region is editable — the body of `run_attack` in `torchattacks/bench/custom_attack.py`
(lines 7–22 of the template). The contract is fixed:

`run_attack(model, images, labels, eps, device, n_classes) -> adv_images`

with `images` of shape `(N, C, H, W)` in `[0,1]`, `labels` of shape `(N,)`, and `adv_images` the same
shape, in `[0,1]`, within `eps` of `images` in `L_inf`. Every method on the ladder is a fill of this
one function. The starting point is the scaffold default: **return the clean images unchanged** (a
0-ASR floor that the first real attack must beat).

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

Five scored scenarios — ResNet20-C10, VGG11BN-C10, ResNet20-C100, VGG11BN-C100, and the hidden
MobileNetV2-C100 — plus MobileNetV2-C10 where it is reported. One seed, `42`. Budget `eps = 2/255`.
Up to 1000 initially-correct samples per scenario, batch size 100. The single metric, higher is
better, is `asr = 1 − robust_acc` per scenario. The reference baseline implementations live in the
`torchattacks` package; the harness exposes only the white-box first-order interface above.
