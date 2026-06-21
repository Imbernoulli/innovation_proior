## Research question

Train deep convolutional image classifiers to higher test accuracy by designing the **classification loss only**, with the model architectures, optimizer, data pipeline, and the evaluation metric all held fixed. Plain cross-entropy is the incumbent; the single thing being designed is the function that turns raw logits and integer labels into the scalar that is back-propagated during training. Everything else about the run is frozen.

## Prior art / Background / Baselines

- **Plain cross-entropy.** Maps logits to softmax probabilities and penalizes the negative log-probability of the true class. Gap: it assigns equal gradient weight to every example regardless of how confident or how wrong it already is, so confident correct and confident wrong examples pull with comparable strength.
- **Focal loss.** Modulates cross-entropy by `(1 - P_t)^γ` to down-weight easy, high-confidence examples. Gap: it is designed for extreme foreground/background imbalance; on balanced classification it suppresses too many easy examples and discards usable signal.
- **Label smoothing.** Replaces the one-hot target with a mixture of the true label and a uniform distribution over classes. Gap: the smoothing weight is fixed across every example and class, so it cannot adapt to per-example uncertainty.
- **PolyLoss.** Treats cross-entropy as a polynomial in `(1 - P_t)` and adds a tunable coefficient to the leading term. Gap: the coefficient is global and static, so the loss does not adjust to individual examples or to the training stage.

## Fixed substrate / Code framework

The training harness is frozen and must not be touched. Models: CIFAR-adapted **ResNet-56** (`[9,9,9]` BasicBlocks), **VGG-16-BN** (BatchNorm throughout, dropout 0.5), and a CIFAR-adapted **MobileNetV2** (inverted residuals, width 1.0). Weight init is fixed Kaiming-normal. Optimizer is **SGD** with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, under **cosine annealing over 200 epochs**, batch size 128. Augmentation is `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`. The training loop calls `compute_loss(outputs, targets, config)` on each batch; the reported `test_loss` is always plain cross-entropy, so the loss design affects training only.

## Editable interface

Exactly one function is editable — `compute_loss(logits, targets, config)` in `pytorch-vision/custom_loss.py`. It receives raw logits `[B, C]`, integer targets `[B]`, and a `config` dict with `num_classes` (int), `epoch` (int, 0-indexed), and `total_epochs` (int), and must return a differentiable scalar loss. `torch` and `torch.nn.functional as F` are in scope. The starting point is plain cross-entropy:

```python
# EDITABLE region of pytorch-vision/custom_loss.py (lines 246-266) — default fill
def compute_loss(logits, targets, config):
    """Compute classification loss.

    Args:
        logits: [B, C] raw model output (pre-softmax)
        targets: [B] integer class labels (0 to C-1)
        config: dict with keys num_classes (int), epoch (int, 0-indexed),
                total_epochs (int)
    Returns:
        scalar loss tensor (differentiable)
    """
    return F.cross_entropy(logits, targets)
```

## Evaluation settings

Three architecture/dataset pairs, each a single seed (42): **ResNet-56 on CIFAR-100**, **VGG-16-BN on CIFAR-100**, and **MobileNetV2 on FashionMNIST**. The metric is the best test accuracy (%, higher is better) reached during the 200-epoch run, reported per pair. The custom loss must stay differentiable, accept raw logits and integer labels, and must not change the datasets, model definitions, optimizer setup, or test-time evaluation.
