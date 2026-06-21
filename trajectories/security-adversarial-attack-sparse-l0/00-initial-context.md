## Research question

Fool an image classifier under a strict **`L0`** budget: change at most `pixels` spatial pixels of a correctly classified image (a pixel counts as changed if any of its channels moves), but each changed pixel may take any color in `[0,1]`. This is the opposite trade from dense `L_inf`/`L_2` attacks, which cap the *magnitude* of a change spread across every pixel; here the count of touched pixels is capped and the amplitude is free. The single thing being designed is the attack procedure `run_attack` — which pixels to move, and to what — under a fixed budget `pixels = 24`. Everything else — the target models, the data, the validity checks — is frozen.

The targets are adversarially-robust `L2` models from the RobustBench zoo. On an undefended classifier a strong sparse attack saturates near 100%, so the informative regime is robust models, where success rates spread across the `[0,1]` range.

## Prior art / Background / Baselines

These methods establish the background; each leaves a concrete gap.

- **Box-constrained `L-BFGS` (Szegedy et al. 2014).** Minimizes `||r||_2` subject to misclassification and box constraints. Gap: the `L2` penalty has no incentive to zero coordinates, so the perturbation is dense — the wrong shape for an `L0` budget — and it runs a heavy per-example optimization.
- **Fast gradient sign method (Goodfellow et al. 2015).** Takes one step `eps*sign(grad_x J)`. Gap: the sign step writes a nonzero into every coordinate, producing the densest possible perturbation, and it targets the loss rather than a chosen support.
- **Class saliency maps (Simonyan et al. 2014).** Backprops a class score to the input to highlight which pixels matter. Gap: it is a visualization — single class, magnitude-only, no notion of converting one class to another — not an attack procedure.
- **DeepFool (Moosavi-Dezfooli et al. 2016).** Iteratively walks to the nearest linearized decision boundary with a minimal `L2` step. Gap: dense `L2` again.
- **JSMA.** Greedy forward-derivative saliency map that picks pixels to increase the target-class score. Gap: requires gradients and uses a fixed forward-selection heuristic.
- **One-pixel attack.** Uses differential evolution to change a small number of individual pixels. Gap: the search scales poorly with image size and budget.
- **SparseFool.** Finds a minimal `L0` perturbation via local linearization of the decision boundary. Gap: requires gradients and can fail when the linear approximation is poor.
- **Pixle.** Randomly perturbs pixel locations and keeps changes that improve success. Gap: purely randomized, with low sample efficiency.

## Fixed substrate / Code framework

A single evaluation harness (`bench/run_eval.py`) is frozen and must not be touched. It loads one adversarially-robust `L2` CIFAR-10 model from the RobustBench zoo (`Rebuffi-R18-L2`, `Augustin-L2`, `Engstrom-L2`); collects up to 150 samples the model classifies correctly on clean input; passes a fresh deep copy of the model to the attack; runs `run_attack`; then validates every output and scores it. A returned image is counted as a failure unless it is finite, lies in `[0,1]` (within `1e-6`), and changes at most `pixels` spatial pixels — where a pixel is "changed" iff `((adv - x).abs() > 1e-5).any(dim=channels)`. The reported numbers are `clean_acc`, `robust_acc`, and `asr = 1 - robust_acc`. Higher ASR is stronger. The attack may use full model access including gradients; the budget, not the access, is the constraint.

## Editable interface

Exactly one region is editable — the body of `run_attack` in `torchattacks/bench/custom_attack.py`. Every solution fills this contract:

```
run_attack(model, images, labels, pixels, device, n_classes) -> adv_images
```

`images` is `(N, C, H, W)` in `[0,1]` on `device`; `labels` is `(N,)`; `pixels` is the `L0` budget (24); `n_classes` is 10. The return must be the same shape, in `[0,1]`, and respect the budget. The baselines `onepixel`, `sparsefool`, `jsma`, and `pixle` are thin wrappers around the reference implementations in `torchattacks`; `sparse_rs` is inlined because `torchattacks` has no Sparse-RS implementation. Each solution replaces exactly this function body and nothing else.

The starting point is the scaffold default: a no-op attack that returns the clean image untouched (ASR 0 by construction).

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
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """
    Sparse L0 adversarial attack.
    images: (N, C, H, W) in [0, 1] on device. labels: (N,) on device.
    pixels: max number of modified spatial pixels (H, W) per sample.
    n_classes: 10 for CIFAR-10, 100 for CIFAR-100.
    Returns adv_images satisfying an L0 pixel budget validated by evaluator.
    """
    _ = (model, labels, pixels, device, n_classes)
    return images.clone()

# =====================================================================
# END EDITABLE REGION
# =====================================================================
```

## Evaluation settings

Three CIFAR-10 settings, each a different adversarially-robust `L2` RobustBench model: `Rebuffi-R18-L2`, `Augustin-L2`, and `Engstrom-L2` (hidden). One seed, 42. Budget `pixels = 24`, up to 150 correctly-classified samples per model. The single metric is ASR (`asr = 1 - robust_acc`) per model, higher is better; invalid outputs score as attack failures. Wall-clock per model is reported but not scored.
