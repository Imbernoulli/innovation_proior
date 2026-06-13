## Research question

Cross-entropy is the default training objective for image classifiers, and focal loss is the default once the foreground/background imbalance of dense prediction enters the picture. Both are *fixed* functions: once you write down `-log(P_t)` or `-(1-P_t)^γ log(P_t)`, the entire shape of the loss — and, more importantly, the entire shape of its gradient as a function of the target-class confidence `P_t` — is hard-wired. There is no reason to believe that this single hard-wired shape is the right one for every architecture, dataset, and class distribution. The question is whether the standard loss can be opened up into a *family* with a small number of interpretable knobs, such that cross-entropy and focal loss fall out as particular settings and the knob values can be selected with a tiny task-specific search. The design space of "any differentiable map from predictions and labels to a scalar" is far too large to search directly; the goal is a principled, low-dimensional parameterization of it.

## Background

**Cross-entropy and its gradient.** For a softmax classifier, let `P_t` be the predicted probability of the ground-truth class. Cross-entropy is `L_CE = -log(P_t)`. Its gradient with respect to `P_t` is `-1/P_t`, and after composing with the softmax, the gradient with respect to the logits has the familiar `p - y` form. The behavior that matters for what follows is how strongly the loss pushes as a function of how confident the model already is: cross-entropy keeps pushing `P_t` toward 1 with a gradient that never vanishes at finite logits.

**Focal loss (Lin et al., 2017).** Designed to fight the 1:1000 easy-background flood in dense object detection, focal loss multiplies cross-entropy by a modulating factor: `L_FL = -(1-P_t)^γ log(P_t)`. For an easy example (`P_t → 1`) the factor `(1-P_t)^γ` collapses toward zero, suppressing its loss and gradient; for a hard example (`P_t` small) the factor is near 1. The focusing parameter `γ` (commonly 2.0) controls how aggressively easy examples are down-weighted. It earns its keep on imbalanced detection, but is not shown to consistently help balanced classification.

**A series identity for the logarithm.** The natural logarithm has the Mercator series `-log(x) = Σ_{j≥1} (1/j)(1-x)^j` for `x ∈ (0,1]`, the range a probability lives in. Substituting `x = P_t` rewrites cross-entropy as

  `L_CE = Σ_{j≥1} (1/j)(1-P_t)^j = (1-P_t) + ½(1-P_t)² + ⅓(1-P_t)³ + …`,

an exact infinite-series form of `-log(P_t)` in powers of `(1-P_t)`. This is a standard algebraic identity, not a new loss.

**Diagnostic finding on truncation (Feng et al., 2020).** A prior line of work proposed *dropping* the higher-order terms of this series — `L_Drop = Σ_{j=1}^N (1/j)(1-P_t)^j` — to recover mean-absolute-error-like robustness to label noise. The diagnostic observation about clean, large-class-count classification is that this truncation *hurts*: on ImageNet-1K with ResNet-50 one must keep more than 600 terms before `L_Drop` matches the accuracy of full cross-entropy, and the loss of accuracy is not recoverable by retuning the learning rate. Early in training the model is near chance, so `P_t` is near zero, and at `P_t ≈ 0.001` the order-500 term still carries a factor `0.999^499 ≈ 0.6`, far from negligible. So the higher-order terms cannot simply be removed.

**Diagnostic finding on the leading term.** Measured over the course of ResNet-50/ImageNet-1K training, the single leading term `(1-P_t)` of the series contributes more than half of the cross-entropy gradient for roughly the last two thirds of training.

## Baselines

**Cross-entropy.** `L_CE = -log(P_t)`. Strong, general default. Gap: the `-log` shape is fixed; there is no argument that it is *optimal* for a given architecture, dataset, or class distribution, and it exposes no knob to adapt to a task without rewriting the criterion.

**Focal loss.** `L_FL = -(1-P_t)^γ log(P_t)`. Gap: it has exactly one knob, `γ ≥ 0`, and it earns its keep on imbalanced detection by down-weighting easy examples; it is not shown to consistently help balanced classification, and `γ` can only ever down-weight, never the other way.

**Truncated / Taylor cross-entropy (Feng et al., 2020).** `L_Drop = Σ_{j=1}^N (1/j)(1-P_t)^j`, the series truncated at order `N`. Recovers MAE-like noise robustness as `N → 1`. Gap: on clean high-class-count data it needs hundreds of terms to avoid losing accuracy (see the diagnostic above), so it does not yield a compact, few-parameter loss.

**Learned-loss / meta-learning approaches (e.g. TaylorGLO).** Search a multivariate Taylor parameterization of the loss with CMA-ES. Gap: the search space scales with the polynomial order, so the demonstrated wins are on low-class-count problems with a handful of parameters; for 1000-class problems where many terms matter, the black-box search becomes prohibitive.

## Evaluation settings

The natural yardsticks span the tasks where cross-entropy and focal loss are the incumbents: 2D image classification (ImageNet-1K and the larger, imbalanced ImageNet-21K, with ResNet-50 and EfficientNetV2 backbones; top-1 accuracy), 2D instance segmentation and object detection (COCO with Mask R-CNN; box and mask AP/AR), and 3D object detection (Waymo Open Dataset with PointPillars and Range Sparse Net; 3D AP). For smaller-scale classification studies, CIFAR with wide-residual and residual networks is standard. Training uses each model's established optimizer, schedule, and augmentation without modification, so the loss is the only thing varied. The metric is top-1 accuracy for classification and AP/AR for detection and segmentation, higher is better.

## Code framework

The available primitive is a standard softmax classification setup: raw logits over `C` classes, integer targets, and the usual cross-entropy criterion. What is open is the shape of the per-example classification loss as a function of the target-class probability `P_t`.

```python
import torch
import torch.nn.functional as F

# logits:  [B, C] raw model outputs (pre-softmax)
# targets: [B]    integer class labels in [0, C)
# F.cross_entropy(logits, targets) -> -log(P_t), the standard criterion


def classification_loss(logits, targets, loss_params):
    """Per-batch scalar classification loss.

    The standard choice is cross-entropy, L_CE = -log(P_t). The open question is
    how to expose a small, task-tunable modification without rewriting the
    surrounding training loop.
    """
    # TODO: recover P_t, the softmax probability at the target class
    # TODO: choose a low-dimensional adjustment to the per-example loss curve
    raise NotImplementedError
```
