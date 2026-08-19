Measured results — rung-1 recipe and architecture (class-token readout only) held fixed, RegNetY-16GF
teacher (82.9% top-1, frozen) fixed, only the training loss varied. ImageNet-1k, top-1 accuracy.

## Soft distillation (Hinton-style KD, τ=3.0, λ=0.1)

| model | top1_imagenet |
|---|---|
| Tiny | 72.2 |
| Small | 79.8 |
| Base | 81.8 |
| Base, fine-tuned 384² | 83.2 |

## Hard-label distillation (½ CE(true) + ½ CE(teacher argmax), parameter-free)

| model | top1_imagenet |
|---|---|
| Tiny | 74.3 |
| Small | 80.9 |
| Base | 83.0 |
| Base, fine-tuned 384² | 84.0 |

Soft distillation reproduces the rung-1 undistilled numbers almost exactly at 224² (72.2/79.8/81.8,
identical to no distillation) and gains only 0.1 point at 384² (83.2 vs. 83.1) — no measurable benefit
from the teacher signal at this recipe. Hard-label distillation gains over the undistilled baseline at
every size and every resolution: +2.1 (Ti), +1.1 (S), +1.2 (B, 224²), +0.9 (B, 384²). Hard distillation
strictly dominates soft distillation on every reported number, using the same teacher, same recipe, same
architecture, same parameter count, and no additional hyperparameters.
