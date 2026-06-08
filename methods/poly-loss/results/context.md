## Research question

Cross-entropy is the default training objective for image classifiers, and focal loss is the default once the foreground/background imbalance of dense prediction enters the picture. Both are *fixed* functions: once you write down `-log(P_t)` or `-(1-P_t)^γ log(P_t)`, the entire shape of the loss — and, more importantly, the entire shape of its gradient as a function of the target-class confidence `P_t` — is hard-wired. There is no reason to believe that this single hard-wired shape is the right one for every architecture, dataset, and class distribution. The question is whether the standard loss can be opened up into a *family* with a small number of interpretable knobs, such that cross-entropy and focal loss fall out as particular settings and the knob values can be selected with a tiny task-specific search. The design space of "any differentiable map from predictions and labels to a scalar" is far too large to search directly; the goal is a principled, low-dimensional parameterization of it.

## Background

**Cross-entropy and its gradient.** For a softmax classifier, let `P_t` be the predicted probability of the ground-truth class. Cross-entropy is `L_CE = -log(P_t)`. Its gradient with respect to `P_t` is `-1/P_t`, and after composing with the softmax, the gradient with respect to the logits has the familiar `p - y` form. The behavior that matters for what follows is how strongly the loss pushes as a function of how confident the model already is: cross-entropy keeps pushing `P_t` toward 1 with a gradient that never vanishes at finite logits.

**Focal loss (Lin et al., 2017).** Designed to fight the 1:1000 easy-background flood in dense object detection, focal loss multiplies cross-entropy by a modulating factor: `L_FL = -(1-P_t)^γ log(P_t)`. For an easy example (`P_t → 1`) the factor `(1-P_t)^γ` collapses toward zero, suppressing its loss and gradient; for a hard example (`P_t` small) the factor is near 1. The focusing parameter `γ` (commonly 2.0) controls how aggressively easy examples are down-weighted. It earns its keep on imbalanced detection, but is not shown to consistently help balanced classification.

**The polynomial / Taylor view.** The natural logarithm has the Mercator series `-log(x) = Σ_{j≥1} (1/j)(1-x)^j` for `x ∈ (0,1]`. Substituting `x = P_t` expresses cross-entropy exactly as a weighted sum of polynomial bases in `(1-P_t)`:

  `L_CE = Σ_{j≥1} (1/j)(1-P_t)^j = (1-P_t) + ½(1-P_t)² + ⅓(1-P_t)³ + …`

so cross-entropy is the member of the family `Σ_j α_j (1-P_t)^j` with coefficients `α_j = 1/j`. The same substitution applied to focal loss gives `L_FL = Σ_j (1/j)(1-P_t)^{j+γ}`, i.e. focal loss is cross-entropy with every term's power shifted up by `γ`. A striking consequence appears in the gradient: because the coefficient `1/j` exactly cancels the power `j` when differentiating `(1-P_t)^j`, the gradient of cross-entropy is the clean geometric series `-dL_CE/dP_t = Σ_{j≥1} (1-P_t)^{j-1} = 1 + (1-P_t) + (1-P_t)² + …`. The leading gradient term is the constant `1`, independent of `P_t`; the higher-order terms are suppressed as `P_t → 1`.

**Diagnostic finding on truncation (Feng et al., 2020).** A prior line of work proposed *dropping* the higher-order polynomial terms — `L_Drop = Σ_{j=1}^N (1/j)(1-P_t)^j` — to recover mean-absolute-error-like robustness to label noise. The diagnostic observation about clean, large-class-count classification is that this truncation *hurts*: on ImageNet-1K with ResNet-50 one must keep more than 600 terms before `L_Drop` matches the accuracy of full cross-entropy, and the loss of accuracy is not recoverable by retuning the learning rate. The reason is that high-order gradient terms remain large *early* in training, when `P_t` is near zero: at `P_t ≈ 0.001` the 500th gradient term's coefficient is `0.999^499 ≈ 0.6`, far from negligible. So the higher-order bases cannot simply be removed.

**Diagnostic finding on the leading term.** Measured over the course of ResNet-50/ImageNet-1K training, the single leading polynomial `(1-P_t)` contributes more than half of the cross-entropy gradient for roughly the last two thirds of training. This concentration of the gradient in the first term makes the leading coefficient the natural first candidate for a low-dimensional adjustment.

## Baselines

**Cross-entropy.** `L_CE = -log(P_t)`, equivalently `α_j = 1/j` for all `j` in the polynomial family. Strong, general default. Gap: the `1/j` coefficient assignment is mathematically natural but there is no argument that it is *optimal* for a given task; in particular its constant leading gradient term `1` counts every example equally regardless of confidence, which can over-serve a majority class on imbalanced data while being too weak to keep increasing target-class confidence in balanced classification.

**Focal loss.** `L_FL = -(1-P_t)^γ log(P_t)`, equivalently a *horizontal* shift of the coefficient profile by `γ`; its training push has no `P_t`-independent leading constant when `γ > 0`. Gap: it has exactly one direction of freedom (shift everything by `γ ≥ 0`), it can only suppress the constant confidence pressure rather than strengthen it, and it is tied to the imbalanced-detection regime; when the task needs stronger target-class confidence pressure, shifting the profile the focal way has the wrong sign.

**Truncated / Taylor cross-entropy (Feng et al., 2020).** `L_Drop = Σ_{j=1}^N (1/j)(1-P_t)^j`, i.e. set all coefficients above order `N` to zero. Recovers MAE-like noise robustness as `N → 1`. Gap: setting tail coefficients to zero is a *vertical* push of those coefficients to zero, and on clean high-class-count data it needs hundreds of terms to avoid losing accuracy, so it does not yield a compact coefficient design.

**Learned-loss / meta-learning approaches (e.g. TaylorGLO).** Search a multivariate Taylor parameterization of the loss with CMA-ES. Gap: the search space scales with the polynomial order, so the demonstrated wins are on low-class-count problems with a handful of parameters; for 1000-class problems where hundreds of terms matter, the black-box search becomes prohibitive.

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
