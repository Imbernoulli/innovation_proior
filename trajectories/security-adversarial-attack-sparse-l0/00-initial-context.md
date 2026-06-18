## Research question

Fool an image classifier under a strict **`L0`** budget: change at most `pixels` *spatial* pixels of a
correctly classified image (a pixel counts as changed if *any* of its channels moves), but each changed
pixel may take any color in `[0,1]`. This is the opposite trade from the familiar dense `L_inf`/`L_2`
attacks, which cap the *magnitude* of a change spread across every pixel; here the count of touched
pixels is capped and the amplitude is free. The single thing being designed is the attack procedure
`run_attack` — which pixels to move, and to what — under a fixed budget `pixels = 24` (the canonical
Sparse-RS CIFAR-10 `L0` budget). Everything else — the target models, the data, the validity checks — is
frozen.

The setting is deliberately hard: the targets are *adversarially-robust* `L2` models from the
RobustBench zoo, not undefended networks. On an undefended classifier a strong sparse attack saturates
at ~100% trivially, so the informative regime — the one that actually ranks attacks — is robust models,
where the success rates spread out across the whole `[0,1]` range.

## Prior art before the first rung (the sparse-attack lineage)

The first rung reacts to the dense-attack orthodoxy and to the handful of methods that already tried to
make an attack *sparse*. These precede the ladder; each leaves a gap the ladder's rungs answer.

- **Box-constrained `L-BFGS` (Szegedy et al. 2014).** Minimize `||r||_2` subject to `f(x+r)=target`
  and `x+r in [0,1]`, by line-searching a penalty constant. First to find imperceptible adversarial
  examples and transferability. Gap: the `L2` penalty has no incentive to zero coordinates, so the
  perturbation is *dense* — exactly the wrong shape for an `L0` budget — and it runs a heavy
  per-example optimization.
- **Fast gradient sign method (Goodfellow et al. 2015).** One step `eps*sign(grad_x J)`. Fast, with a
  clean linear story for why many tiny coordinated changes sum to a large logit shift. Gap: the sign
  step writes a nonzero into *every* coordinate — the densest possible perturbation under `L_inf`, the
  opposite of sparse — and it targets the loss, not a chosen support.
- **Class saliency maps (Simonyan et al. 2014).** Backprop a class score to the input,
  `M_ij = max_c |(dS_c/dI)_{ijc}|`, to highlight which pixels matter. Gap: a *visualization* — single
  class, magnitude-only (sign-blind), no notion of converting one class to another — not an attack
  procedure, but it plants the idea that per-feature, per-class input sensitivity is the right object.
- **DeepFool (Moosavi-Dezfooli et al. 2016).** Iteratively walk to the nearest *linearized* decision
  boundary with a minimal `L2` step. Gap: dense `L2` again; but its local-linear boundary model is the
  geometric engine a later sparse method reuses.

What these share is the crack the ladder pries open: they all bound a *norm* and let the support run
free, and most read back a single loss gradient. A genuine `L0` attack must instead *select* a tiny set
of coordinates — a discrete, combinatorial choice on top of the continuous "to what value" choice — and
several of them must do it through scores alone, because the cheapest realistic threat model never sees
the weights.

## The fixed substrate

A single evaluation harness (`bench/run_eval.py`) is frozen and must not be touched. It: loads one
adversarially-robust `L2` CIFAR-10 model from the RobustBench zoo (`Rebuffi-R18-L2`, `Augustin-L2`,
`Engstrom-L2`); collects up to **150** samples the model classifies correctly on clean input; passes a
**fresh deep copy** of the model to the attack (so user code cannot tamper with the scoring model's
weights, BatchNorm stats, or hooks); runs `run_attack`; then validates every output and scores it. A
returned image is counted as a *failure* (the sample stays robust) unless it is finite, lies in
`[0,1]` (within `1e-6`), **and** changes at most `pixels` spatial pixels — where a pixel is "changed" iff
`((adv - x).abs() > 1e-5).any(dim=channels)`. The reported numbers are `clean_acc` (always 1.0 by
construction), `robust_acc` (fraction still correct, invalid outputs included), and the headline
`asr = 1 - robust_acc`. Higher ASR is stronger. The attack may use full model access including gradients
(the deep copy is differentiable); the budget, not the access, is the constraint.

## The editable interface

Exactly one region is editable — the body of `run_attack` in `torchattacks/bench/custom_attack.py`
(the file's lines 7–23). Every rung on the ladder is a fill of this same contract:

```
run_attack(model, images, labels, pixels, device, n_classes) -> adv_images
```

`images` is `(N, C, H, W)` in `[0,1]` on `device`; `labels` is `(N,)`; `pixels` is the `L0` budget
(24); `n_classes` is 10. The return must be the same shape, in `[0,1]`, and respect the budget. The
baselines `onepixel`, `sparsefool`, `jsma`, and `pixle` are thin wrappers around the reference
implementations in `torchattacks`; `sparse_rs` is inlined because `torchattacks` has no Sparse-RS. Each
rung replaces exactly this function body and nothing else.

The starting point is the scaffold default: a **no-op** attack that returns the clean image untouched
(ASR 0 by construction). The ladder fills it in.

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

Three CIFAR-10 settings, each a different adversarially-robust `L2` RobustBench model:
`Rebuffi-R18-L2` (the exact model the Sparse-RS L0 evaluation uses), `Augustin-L2`, and `Engstrom-L2`
(hidden). One seed, 42. Budget `pixels = 24`, up to 150 correctly-classified samples per model. The
single metric is **ASR** (`asr = 1 - robust_acc`) per model, higher is better; invalid outputs (shape,
range, non-finite, or budget violation) score as attack failures. Wall-clock per model is reported but
not scored.
