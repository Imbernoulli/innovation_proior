## Research question

Train deep convolutional image classifiers to higher test accuracy by designing the **classification loss only**, with the model architectures, optimizer, data pipeline, and the evaluation metric all held fixed. Plain cross-entropy is the incumbent; the single thing being designed is the function that turns raw logits and integer labels into the scalar that is back-propagated during training. Everything else about the run is frozen.

## Prior art before the first rung (cross-entropy and the loss lineage)

The first rung reacts to plain cross-entropy and to the focal-loss line; these are the methods that precede the ladder.

- **Plain cross-entropy.** With softmax probability `P_t` on the true class, `L_CE = -log(P_t)`. Its optimum sits at an infinite correct-vs-rest logit gap, so training drives the network to ever-greater confidence; the gradient never vanishes at finite logits, and every example — confidently right, confidently wrong, or ambiguous — contributes the same constant leading push. Gap: nothing in the objective adapts to how confident or how wrong an example already is, so it tends toward overconfident, less-generalizing solutions.
- **Focal loss (Lin et al., 2017).** Multiplies cross-entropy by a modulating factor, `L_FL = -(1-P_t)^γ log(P_t)`, with focusing parameter `γ` (default 2.0). For an easy example (`P_t → 1`) the factor `(1-P_t)^γ` collapses toward zero and suppresses its loss; for a hard example the factor is near 1. Born to fight the 1:1000 easy-background flood of dense object detection, where most of the loss budget is otherwise spent on already-correct background. Gap: it down-weights the *easy* examples by confidence, which is exactly the right move when an over-represented easy class is drowning the gradient — but on a *balanced* classification problem there is no such flood, so suppressing easy examples mostly throws away usable signal rather than rebalancing anything.
- **Label smoothing (Szegedy et al., 2016).** Replaces the one-hot target with `(1-ε) one_hot + ε/C`, putting a floor `ε/C` under every class so the correct logit has no incentive to escape to `+∞`. Equivalent to `(1-ε) H(one_hot, p) + ε H(uniform, p)` — cross-entropy plus a pull toward the uniform prior. Bounds overconfidence, regularizes, and improves calibration. Gap: it is a static edit of the *target*, identical for every example and class, and does not adapt to training dynamics.
- **PolyLoss (Leng et al., 2022).** Expands cross-entropy in the Mercator series `-log(P_t) = Σ_j (1/j)(1-P_t)^j`, exposing it as a polynomial in `(1-P_t)` with fixed coefficients `1/j`, and perturbs the leading coefficient: Poly-1 is `L = -log(P_t) + ε₁(1-P_t)` (default `ε₁ = 2.0` for image classification). A positive `ε₁` strengthens the confidence-pressure that plain cross-entropy lets decay too soon. Gap: it is one extra static term on the leading polynomial; the coefficient does not adapt per example or per epoch.

## The fixed substrate

The training harness is frozen and must not be touched. Models: CIFAR-adapted **ResNet-56** (`[9,9,9]` BasicBlocks, 3×3 stem, global average pool), **VGG-16-BN** (BatchNorm throughout, adaptive pool, a 512→512→C classifier head with dropout 0.5), and a CIFAR-adapted **MobileNetV2** (inverted residuals, width 1.0). Weight init is fixed Kaiming-normal. Optimizer is **SGD** with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, under **cosine annealing over 200 epochs**, batch size 128. Augmentation is CIFAR-style: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`. The training loop calls `compute_loss(outputs, targets, config)` on each batch; the reported `test_loss` is always plain cross-entropy regardless of the training loss, so the loss design affects *training only*.

## The editable interface

Exactly one function is editable — `compute_loss(logits, targets, config)` in `pytorch-vision/custom_loss.py`. It receives raw logits `[B, C]`, integer targets `[B]`, and a `config` dict with `num_classes` (int), `epoch` (int, 0-indexed), and `total_epochs` (int), and must return a differentiable scalar loss. `torch` and `torch.nn.functional as F` are in scope. Every method on the ladder is a fill of this same contract; the starting point is the scaffold default — plain cross-entropy.

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

Three architecture/dataset pairs, each a single seed (42): **ResNet-56 on CIFAR-100** (deep residual, 100 classes), **VGG-16-BN on CIFAR-100** (deep non-residual with BatchNorm, 100 classes), and **MobileNetV2 on FashionMNIST** (lightweight inverted-residual, 10 classes). The metric is the best test accuracy (%, higher is better) reached during the 200-epoch run, reported per architecture/dataset pair. The custom loss must stay differentiable, accept raw logits and integer labels, and must not change the datasets, model definitions, optimizer setup, or test-time evaluation.
