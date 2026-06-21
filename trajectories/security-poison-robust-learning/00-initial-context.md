## Research question

A fraction of the training labels has been adversarially flipped — each poisoned example's target is moved to `(original + 1) % num_classes` — and the poisoned indices are unknown. The design task is the **training objective**: a loss function (or a sample-weighting rule built into the loss) that still learns the true class structure without memorizing the corrupted targets. The model, optimizer, data pipeline, augmentation, learning-rate schedule, and poison injection are all fixed. The only editable surface is `RobustLoss.compute_loss(logits, labels, epoch)`, which returns the scalar minibatch loss.

## Prior art / Background / Baselines

- **ERM with cross entropy.** Train the softmax classifier by minimizing the mean `-log f_y` over the batch. Gap: it has no way to treat a label as potentially corrupt, so high-capacity networks can drive training error to zero even on flipped labels.

- **Symmetric losses / noise-tolerance criterion (Ghosh, Kumar & Sastry, 2017).** A sufficient condition for noise tolerance is a loss whose class scores sum to a constant; mean absolute error satisfies this. Gap: the MAE gradient `−2 f_y(1−f_y)` vanishes as `f_y → 0`, so hard examples receive almost no update and training progresses very slowly.

- **Label smoothing (Szegedy et al., 2016).** Replace one-hot targets with a soft distribution that spreads a fixed mass over every class. Gap: it biases every example, clean or corrupt, and does not reduce the weight placed on flipped labels in particular.

## Fixed substrate / Code framework

A research-scale vision training loop is frozen and must not be touched. Three architectures — ResNet-20, VGG-16-BN, MobileNetV2 — are built from scratch with Kaiming init and trained with SGD (`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`) under cosine annealing for 100 epochs, batch size 128, with standard augmentation and normalization. The training set is wrapped by a fixed `PoisonedDataset` that selects a seeded random subset and flips each selected label to `(label + 1) % num_classes`. The loop calls `robust_loss.compute_loss(logits, labels, epoch)` once per minibatch in place of `nn.CrossEntropyLoss`, backpropagates, and steps the optimizer. Nothing else is exposed: the loss sees only the current logits, the possibly poisoned labels, and the integer epoch.

## Editable interface

Exactly one region is editable — the `RobustLoss` class in `custom_robust_loss.py`. `compute_loss` receives `logits` of shape `(B, K)`, `labels` of shape `(B,)`, and `epoch` (0-indexed), and returns a scalar loss tensor. The module may hold fixed hyperparameters in `__init__`, but no per-sample state survives between calls. `torch` and `torch.nn.functional as F` are already imported.

The starting point is the scaffold default: **standard cross entropy on the poisoned labels** (ERM). Each method replaces exactly this method body and nothing else.

```python
# EDITABLE region of custom_robust_loss.py -- default fill (standard cross entropy / ERM)
import torch
import torch.nn.functional as F


class RobustLoss:
    """Default cross-entropy objective on the (possibly poisoned) labels."""

    def __init__(self):
        pass

    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels)
```

## Evaluation settings

Three fixed benchmarks, all seed 42: `resnet20-cifar10-labelflip` (ResNet-20 on CIFAR-10, 10% flip), `vgg16bn-cifar100-labelflip` (VGG-16-BN on CIFAR-100, 10% flip), and `mobilenetv2-fmnist-labelflip` (MobileNetV2 on FashionMNIST, 15% flip). Reported metrics are `test_acc` (clean test accuracy, higher is better), `poison_fit` (fraction of poisoned training samples predicted as the flipped label, lower is better), and `robust_score = (test_acc + (1 − poison_fit)) / 2` (higher is better). A method improves by raising clean accuracy while keeping poison_fit low.
