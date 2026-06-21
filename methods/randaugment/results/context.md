## Research question

Hand-designed data augmentation (flips, crops, color jitter) reliably improves image models, but the *choice* of transforms and their strengths is domain expertise that does not transfer between tasks. Learning the augmentation policy from data removes that manual work and yields large accuracy gains — the dominant approach trains a controller by reinforcement learning on a small *proxy task* to discover a good policy, which is then transferred to the real, larger task. The research question: how should an automated augmentation procedure be designed for image classification and detection?

## Background

**Data augmentation and its diversity benefit.** Augmentation generates extra training data by transforming images in label-preserving ways — flips and crops on natural images, elastic distortions on digits, and also out-of-distribution operations like erasing patches (Cutout, Random Erasing) or convex image/label blends (Mixup). The consistently reported source of the gain from richer augmentation is the increased *diversity* of examples the network sees, which improves generalization at no inference cost.

**Learned augmentation policies.** AutoAugment (Cubuk et al., 2018) frames augmentation as a search problem. A policy is a set of sub-policies; each is applied stochastically, so a network sees many transformed variants. The policy is found by a controller (an RNN trained with reinforcement learning, specifically Proximal Policy Optimization) that proposes policies, trains a child network with each, and is rewarded by the child's validation accuracy. Follow-ups made the search cheaper — Fast AutoAugment (Lim et al., 2019) via density matching, Population Based Augmentation (Ho et al., 2019) via an evolutionary schedule — but all retain a *separate* search procedure.

**The proxy-task assumption.** Like neural architecture search, learned augmentation searches on a small proxy task and transfers the result, assuming the proxy predicts the target. For augmentation this assumption is relevant to consider: the *optimal augmentation strength* is observed to depend on both model size and dataset size — larger models and larger datasets benefit from *stronger* augmentation.

**A magnitude regularity.** Population Based Augmentation reported that the optimal transform *magnitudes* tend to follow a similar schedule over the course of training — rising together — rather than each evolving independently.

**The magnitude scale.** Following AutoAugment, each transform's strength is placed on a single integer scale from 0 to 10 (with PARAMETER_MAX = 10), and a level is mapped to the transform's native parameter linearly: rotation by ±(level/10)·30°, shear by (level/10)·0.3, translation by (level/10)·(max pixels), solarize threshold int((level/10)·256), posterize to int((level/10)·4) bits, and the photometric enhancements (color, contrast, brightness, sharpness) by (level/10)·1.8 + 0.1. AutoContrast, Equalize, and Identity take no magnitude.

## Baselines

**Hand-designed augmentation.** Fixed flips, crops, color jitter, plus operations like Cutout and Mixup. Core idea: inject prior knowledge of label-preserving transforms. Strengths are tuned by hand.

**AutoAugment (Cubuk et al., 2018).** Search space: a policy of 5 sub-policies, each sub-policy 2 operations applied in sequence; each operation is (type ∈ ~16 transforms, probability ∈ 11 discrete values, magnitude ∈ 10 discrete values). The space has size ≈ (16 × 10 × 11)^10 ≈ 2.9 × 10^32. An RNN controller trained with RL/PPO maximizes child-network validation accuracy on a proxy task; the found policy (30+ parameters) is transferred. Core idea: stochasticity at multiple levels (which sub-policy, per-operation probability, direction) maximizes diversity.

**Fast AutoAugment (Lim et al., 2019).** Same search space and formalism, but the policy is found by density matching rather than RL, far faster.

**Population Based Augmentation (Ho et al., 2019).** Evolves an augmentation *schedule* with a population-based method; cheaper than AutoAugment. Core idea: optimal magnitudes change (increase) during training.

## Evaluation settings

The yardsticks are image-classification benchmarks: CIFAR-10/100 (32×32; Wide-ResNet-28-2/-28-10, Shake-Shake, PyramidNet+ShakeDrop), SVHN (core and full; Wide-ResNets, with Cutout applied after augmentation), and ImageNet (ResNet-50 and EfficientNet-B5/B7) reporting top-1 accuracy; plus COCO object detection (ResNet-101/200 RetinaNet) reporting mAP, with geometric transforms also applied to bounding boxes. Default flips and crops (and scale jitter for detection) are applied *before* the learned augmentation. Training protocols are standard: Wide-ResNets for 200 epochs, lr 0.1, batch 128, weight decay 5e-4, cosine decay; ImageNet ResNet-50 for 180 epochs with momentum; EfficientNets per their own schedule. The relevant comparison is, for matched architectures and schedules, against hand-designed augmentation and against the learned-augmentation methods above — and a probe of how the best augmentation strength varies with model and dataset size.

## Code framework

The available ingredients are a set of image transform operations each parameterized by a magnitude on the 0–10 scale, a default flip/crop pipeline, a data loader, a network, and a standard training loop. The open slot is the *policy*: given an image, which operations to apply and at what strength — and how that policy is determined relative to the rest of training.

```python
import numpy as np

transforms = [
    'Identity', 'AutoContrast', 'Equalize',
    'Rotate', 'Solarize', 'Color', 'Posterize',
    'Contrast', 'Brightness', 'Sharpness',
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
]


def apply_op(image, op, magnitude):
    """Apply transform `op` at integer `magnitude` in [0, 10] to `image`."""
    # provided: maps magnitude -> the op's native parameter and applies it
    ...


def build_policy():
    """Decide which operations (and at what magnitude) to apply per image."""
    # TODO: choose the augmentation policy and how it is parameterized.
    pass


def augment(image, policy):
    # TODO: apply the policy to the image (after default flip/crop).
    pass

# default flip+crop, network, SGD/cosine schedule, standard train loop unchanged.
```
