The training-procedure fix closed most of the distance to convnets on its own: the Base size reached 81.8% (224²),
83.1% after a 384² fine-tune, both up sharply from the 77.91–79.35% starting bar, with the same
architecture and dataset throughout. That confirms the earlier diagnosis — this was substantially a
training-procedure gap — but it also tells me the procedure lever is close to exhausted. The recipe
already stacks RandAugment, Mixup, CutMix, random erasing, stochastic depth, label smoothing, and
repeated augmentation; there is limited room left to push further within that category. The next lever
in the toolbox is qualitatively different: instead of only manufacturing invariance from the raw data,
bring in an external signal — a teacher classifier that already encodes visual structure this
architecture had to learn from scratch. If a convnet's convolution supplies locality and translation
equivariance structurally, a convnet classifier's output distribution, even reduced to a soft or hard
label, carries some trace of having been shaped by that structural prior. Distillation lets a Transformer
with no such prior of its own absorb some of that trace through supervision instead of through
architecture. I fix the teacher for this rung and every distillation rung after it: a RegNetY-16GF
convnet, 84M parameters, 82.9% top-1 on its own, trained on the same data with the same augmentation as
the student, so any effect measured is attributable to the distillation signal itself rather than to the
teacher having seen images the student never saw.

The toolbox gives two concrete ways to define that signal, and I want to test both under matched
conditions — same architecture (class token only, nothing about the token layout touched yet), same
recipe as the previous rung, same teacher, same three sizes — because I do not yet know which suits this
setting better. Soft distillation keeps the teacher's full output distribution and minimizes a
temperature-softened KL divergence between student and teacher softmax outputs, blended with the
ground-truth cross-entropy via a mixing coefficient λ and rescaled by τ² to keep the softened gradient
magnitude comparable across temperature choices; it needs two extra hyperparameters (τ, λ) and is built
to transfer the teacher's relative confidence across classes — the fact that a wolf photo also gets some
probability mass on "husky" is itself information, and soft distillation passes that structure through.
Hard-label distillation instead takes only the teacher's argmax on the same input the student sees and
treats it exactly like a second ground-truth label: half the loss is ordinary cross-entropy against the
true label, half against the teacher's hard decision. It is parameter-free, and because the teacher is
evaluated on the same augmented crop the student trains on, its hard decision can track what is actually
visible in that crop even when Mixup, CutMix, or aggressive erasing has changed the image enough that the
original dataset label no longer fully describes the content shown.

That last property is why I suspect, without yet knowing, that hard distillation may have more room to
help here specifically because of what the rung-1 recipe already contains. Label smoothing already
softens the hard cross-entropy target; Mixup and CutMix already inject continuous, non-one-hot targets
into a large fraction of batches. Soft distillation's contribution — a smoothed, multi-class target
distribution — heavily overlaps in kind with signal the recipe already supplies from other sources, so
its marginal information may be smaller here than it would be in a recipe without heavy label mixing.
Hard distillation's contribution is different in kind: a second, independently produced decision about
the image's class, not another softening of the one-hot target already in play. Whether that qualitative
difference shows up as a measurable accuracy difference is exactly what running both under identical
conditions is meant to decide, and I am not committing to a prediction beyond that asymmetry — it is
entirely possible the two end up close on some size where the recipe's own smoothing is weaker relative
to model capacity.

Concretely, I freeze the rung-1 recipe and rung-1 architecture (class token only, plain linear classifier
on its output) and change only the loss, running each of the two objectives — soft KD at the standard
temperature-3, λ=0.1 setting, and hard-label distillation with no extra hyperparameters — against the
fixed RegNetY-16GF teacher, on all three sizes, reporting the same 224² and 384²-fine-tuned top-1 numbers
as before so the comparison to the undistilled baseline is direct. Whichever objective wins becomes the
fixed loss for every later rung that touches the distillation architecture itself, exactly as the rung-1
recipe is now fixed underneath both variants tested here.

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
