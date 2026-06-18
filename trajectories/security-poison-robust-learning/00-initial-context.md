## Research question

A fraction of the training labels has been adversarially flipped — each poisoned example's target is moved to `(original + 1) % num_classes` — and I do not know which examples are poisoned. The one thing being designed is the **training objective**: a loss function (or sample-weighting rule built into the loss) that, on poisoned data, still learns the real class structure while refusing to memorize the corrupted targets. The model, the optimizer, the data pipeline, the augmentation, the learning-rate schedule, and the poison-injection process are all fixed. The only editable surface is a single `RobustLoss.compute_loss(logits, labels, epoch)` method that returns the scalar minibatch loss.

## Prior art before the first rung (label-noise objective lineage)

The first rung reacts to the standard objective and to the formal result that frames the whole field. These are the methods that precede the ladder.

- **Empirical risk minimization with cross entropy (the default).** Train the softmax classifier by descending the mean of `-log f_y` over the batch. The per-sample parameter gradient carries a `1/f_y` factor, so cross entropy is an implicit hard-example weighter — it pours gradient into the points it currently gets wrong. On clean data that is the signal; on poisoned data it is the disease, because a flipped example *looks exactly like a hard example* (the image content disagrees with the given label), so `f_ỹ` is tiny, `1/f_ỹ` is huge, and the objective yanks hard to memorize the lie. Gap: no notion that a target might be corrupt; high capacity then drives training error to zero even on random labels.
- **The noise-tolerance criterion (Ghosh, Kumar & Sastry, AAAI 2017).** Pinned down what "robust" means formally: a loss `L` is noise-tolerant if the minimizer of the noisy risk also minimizes the clean risk, and a *sufficient* condition is that the loss is **symmetric** — `Σ_{k=1}^K L(f(x), k) = C`, a constant independent of `x` and `f`. Under uniform noise the noisy risk then collapses to an affine function of the clean risk, so argmin is preserved for noise rate below `1 − 1/K`. Mean absolute error is symmetric, hence robust, but its gradient `−2 f_y(1−f_y)` vanishes as `f_y → 0`, so it refuses to push the very hard examples that carry the learning signal — it trains painfully slowly. Gap: the only clean robustness lever (symmetry) costs the gradients that make deep nets converge.
- **Label smoothing (Szegedy et al. 2016).** Soften the one-hot target by spreading `ε` mass over all classes. Eases late-stage overconfidence, but it biases the model at *every* point including the clean ones, and it does not selectively distrust the suspicious labels — it is a fixed, label-agnostic regularizer. Gap: attacks overconfidence generically, not memorization of the specific corrupted targets.

The tension the ladder must resolve is exactly this: cross entropy has the gradients but no robustness; symmetric losses have the robustness but not the gradients. Every rung below is an attempt to buy both at once, expressed as a single drop-in loss.

## The fixed substrate

A research-scale vision training loop is frozen and must not be touched. Three architectures — ResNet-20, VGG-16-BN, MobileNetV2 — built from scratch (Kaiming init), trained with SGD (`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`) under a CosineAnnealing schedule for 100 epochs, batch size 128, standard augmentation (random crop with reflect padding, horizontal flip, per-dataset normalization). The training set is wrapped by a fixed `PoisonedDataset` that selects a seeded random subset and flips each selected label to `(label + 1) % num_classes`. The loop calls `robust_loss.compute_loss(logits, labels, epoch)` once per minibatch in place of `nn.CrossEntropyLoss`, backpropagates the returned scalar, and steps the optimizer. Nothing else is exposed: the loss sees the current minibatch logits, the (possibly poisoned) labels, and the integer epoch — no sample indices, no access to the model or optimizer, no per-example history across epochs, no clean validation set.

## The editable interface

Exactly one region is editable — the `RobustLoss` class in `custom_robust_loss.py`. `compute_loss` receives `logits` (minibatch model outputs, shape `(B, K)`), `labels` (the possibly poisoned integer targets, shape `(B,)`), and `epoch` (the 0-indexed current epoch, available for warm-up or annealing schedules), and must return a scalar loss tensor. The module may hold fixed hyperparameters in `__init__` but no per-sample state survives between calls. `torch` and `torch.nn.functional as F` are imported in the module. Every method on the ladder is a fill of this same contract.

The starting point is the scaffold default: **standard cross entropy on the poisoned labels** (ERM). Each later method replaces exactly this method body and nothing else.

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

Three benchmarks, each at a fixed poison rate: `resnet20-cifar10-labelflip` (ResNet-20 on CIFAR-10, 10% flip), `vgg16bn-cifar100-labelflip` (VGG-16-BN on CIFAR-100, 10% flip), `mobilenetv2-fmnist-labelflip` (MobileNetV2 on FashionMNIST, 15% flip), seed 42. Three reported quantities: `test_acc` (accuracy on the clean test set, higher is better), `poison_fit` (the fraction of poisoned training samples on which the model predicts the *poisoned* wrong label — lower is better, it measures how much the model memorized the lies), and the primary metric `robust_score = (test_acc + (1 − poison_fit)) / 2`, higher is better. A method wins by raising clean accuracy while driving poison_fit down.
