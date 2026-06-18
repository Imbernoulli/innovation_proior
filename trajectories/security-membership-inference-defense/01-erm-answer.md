**Problem.** A classifier trained on a private 50/50 member split leaks membership through confidence:
the fixed attack thresholds the max softmax probability, which plain training drives to ~1 on members
and leaves lower on non-members. The baseline must establish the no-defense floor — how much the model
leaks when nothing fights its over-confidence.

**Key idea (the floor).** Standard cross-entropy against the one-hot target. Its optimum sits at an
infinite correct-vs-rest logit gap and the gradient `p − y` only vanishes as `p → y`, so over 300
augmentation-free epochs the model grinds member losses to ~0 and member confidence to the ceiling.
That separation — members at the ceiling, non-members scattered below — *is* the membership signal the
attack reads. No privacy term is applied; this is the absence of a defense.

**Why this is weakest.** Nothing in the objective puts a floor under member confidence, so leakage
scales with overfit: worst on vgg16bn-cifar100 (high-capacity net, 100 classes, no augmentation →
memorization with low test accuracy), milder on resnet20-cifar10, mildest on mobilenetv2-fmnist (an
easy task that generalizes well, leaving little train-test gap to exploit).

**Step-1 edit.** Leave the scaffold at its default `compute_loss`. The only thing the baseline edits is
nothing — plain cross-entropy. Its failure (high `mia_auc`, especially on CIFAR-100) is what forces a
loss that softens the target at step 2.

**Hyperparameters.** None beyond the fixed loop (SGD lr 0.1, momentum 0.9, weight-decay 1e-4,
MultiStepLR ×0.1 at 150/225, 300 epochs, batch 128, no augmentation).

```python
import torch
import torch.nn.functional as F


class MembershipDefense:
    """Standard cross-entropy training (ERM) — the no-defense floor."""

    def __init__(self):
        pass

    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels)
```
