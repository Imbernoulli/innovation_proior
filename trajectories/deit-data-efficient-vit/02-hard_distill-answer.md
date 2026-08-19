**Problem.** The rung-1 recipe fix (procedure only, unmodified architecture) is close to exhausted as a
lever: the Base size reached 81.8%/83.1% by manufacturing invariance from the data side, but every available
augmentation is already stacked. The next lever is qualitatively different — bring in an external signal
from a teacher classifier that already encodes structure this architecture has no built-in prior for.
Fix the teacher (RegNetY-16GF convnet, 84M params, 82.9% top-1, trained on the same data/augmentation as
the student) and test two ways of turning its output into a training signal, holding architecture (class
token only) and the rung-1 recipe fixed.

**Variant A — soft distillation (Hinton-style KD).**

L = (1−λ)·L_CE(ψ(Z_s), y) + λ·τ²·KL(ψ(Z_s/τ), ψ(Z_t/τ)), τ=3.0, λ=0.1

Transfers the teacher's full softmax distribution (relative confidence across classes), at the cost of
two extra hyperparameters.

**Variant B — hard-label distillation.**

L = ½·L_CE(ψ(Z_s), y) + ½·L_CE(ψ(Z_s), y_t), y_t = argmax_c Z_t(c)

Parameter-free: the teacher's argmax on the same (possibly augmented) crop the student sees is treated
exactly like a second ground-truth label. Re-evaluated per crop, so it can track what a Mixup/CutMix/
erasing-transformed image actually shows even when the original dataset label no longer fully describes
it.

```python
import torch
import torch.nn.functional as F

def soft_distillation_loss(student_logits, teacher_logits, labels, tau=3.0, lam=0.1):
    ce = F.cross_entropy(student_logits, labels)
    kl = F.kl_div(
        F.log_softmax(student_logits / tau, dim=-1),
        F.softmax(teacher_logits / tau, dim=-1),
        reduction="batchmean",
    ) * (tau ** 2)
    return (1 - lam) * ce + lam * kl


def hard_distillation_loss(student_logits, teacher_logits, labels):
    y_t = teacher_logits.argmax(dim=-1)                       # teacher's hard pseudo-label, this crop
    ce_true = F.cross_entropy(student_logits, labels)
    ce_teacher = F.cross_entropy(student_logits, y_t)
    return 0.5 * ce_true + 0.5 * ce_teacher


def training_loss(student_outputs, labels, teacher=None, inputs=None, mode="hard"):
    with torch.no_grad():
        teacher_logits = teacher(inputs)                       # same augmented crop the student trained on
    if mode == "soft":
        return soft_distillation_loss(student_outputs, teacher_logits, labels)
    elif mode == "hard":
        return hard_distillation_loss(student_outputs, teacher_logits, labels)
    raise ValueError(mode)


# architecture, recipe (batch size, optimizer, augmentation stack, epochs, sizes) all unchanged from
# rung 1 -- DataEfficientViT with class-token-only readout, trained/evaluated identically, only the loss
# function and the now-present frozen teacher forward pass differ.
teacher = load_pretrained("RegNetY-16GF")                      # 82.9% top-1, frozen, same data/augment
teacher.eval()
```

**Test.** Train Ti/S/B with each of the two loss variants, teacher and recipe otherwise identical to
rung 1, ImageNet-1k only. Report 224² top-1 for both variants against the rung-1 undistilled baseline,
and the 384²-fine-tuned number for the Base size, to see whether either distillation objective — and if so which
one — moves the needle beyond what the recipe alone already achieved.
