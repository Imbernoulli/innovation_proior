## Research question

A classifier trained on a private set leaks who was in that set. The attack is simple: the adversary feeds the trained model a sample and reads its confidence — here the max softmax probability — to decide whether that sample was in the training data. Training drives member confidences toward ~1 while non-member confidences stay lower and more scattered, so a threshold separates the two populations.

The design target is the **training-time loss**, the scalar objective the fixed loop minimizes per minibatch. Architecture, optimizer, schedule, data split, and the attack itself are frozen.

## Prior art / Background / Baselines

These methods all try to keep the model's output distribution from revealing membership.

- **Plain cross-entropy / ERM.** Minimizes `−log p(y)` against a one-hot target. The logit gradient only vanishes as `p → y`, so the optimizer pushes member losses to ~0 and member confidence to ~1.
- **Label smoothing (Szegedy et al., 2016).** Replaces the one-hot target with `q(k) = (1−ε)δ_{k,y} + ε/K`, so an infinite logit gap becomes infinitely costly and member confidence is bounded below 1.
- **Confidence penalty (Pereyra et al., 2017).** Subtracts the predictive entropy, `L = L_CE − β·H(p)`, penalizing the most confident predictions most strongly.

## Fixed substrate / Code framework

The training loop is fixed. It loads the full training set, shuffles by seed, and splits it 50/50 by index into a **member** subset (used for training) and a **non-member** subset (never trained on); the held-out test set is separate. It builds the benchmark architecture — ResNet-20 for CIFAR-10, VGG-16-BN for CIFAR-100, MobileNetV2 for FashionMNIST — and trains for **300 epochs** with **SGD** (lr 0.1, momentum 0.9, weight-decay 1e-4) under a **MultiStepLR** schedule (decay ×0.1 at epochs 150 and 225), batch size 128. **Train-time augmentation is OFF**, so the model overfits and produces a measurable membership signal to defend against.

After training, the loop runs a confidence-based membership-inference attack: it computes the max softmax probability on every member and every non-member example and reports the AUC of that score as a member/non-member discriminator. It also reports test accuracy, the mean member-minus-non-member confidence gap, and the composite privacy score. The loop calls exactly one method per minibatch — `compute_loss(logits, labels, epoch)` — and backpropagates whatever scalar it returns.

## Editable interface

Only one region is editable: the `MembershipDefense` class in `custom_membership_defense.py`. The method to fill is `compute_loss(self, logits, labels, epoch)`, which returns the scalar loss the fixed loop minimizes. `logits` has shape `(B, K)`; `labels` has shape `(B,)`; `epoch` is the 0-indexed training epoch. The method may read `logits.size(1)` to learn `K` and adapt per dataset.

The default fill is plain cross-entropy. Each defense replaces exactly this method body.

```python
# EDITABLE region of custom_membership_defense.py — default fill (plain cross-entropy / ERM)
import torch
import torch.nn.functional as F


class MembershipDefense:
    """Training-time regularizer for privacy-utility tradeoffs.

    compute_loss replaces nn.CrossEntropyLoss() in the fixed training loop.
    Design a loss that reduces membership-inference leakage (lower MIA AUC)
    while preserving test accuracy.
    """

    def __init__(self):
        pass

    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels)
```

## Evaluation settings

Three benchmarks are evaluated at seed 42 with the fixed 300-epoch recipe:

- **resnet20-cifar10** (10 classes)
- **vgg16bn-cifar100** (100 classes; low accuracy and easy to overfit/leak)
- **mobilenetv2-fmnist** (10 classes; already low leakage under plain training)

Reported metrics:

- `test_acc`: higher is better
- `mia_auc`: the attack's AUC; 0.5 is ideal, higher is worse leakage
- `privacy_gap`: mean member confidence minus mean non-member confidence; smaller is better
- `privacy_score = test_acc − max(mia_auc − 0.5, 0)`: the primary composite; higher is better
