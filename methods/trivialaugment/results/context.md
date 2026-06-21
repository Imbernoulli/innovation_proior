# Context: automatic image-augmentation policies (circa 2018-2021)

## Research Question

Image classifiers benefit from training-time transformations that preserve the label: crop,
flip, rotate, shear, translate, shift color, mask a patch, and so on. These augmentations add no
inference-time cost, but the design space is large. A policy has to decide which image operation
to apply, how strongly to apply it, whether to apply it at all, and how many operations to chain.
The practical question is whether an augmentation policy must be searched separately for every
model and dataset, or whether most of the benefit can be recovered by a fixed, search-free rule.

## Augmentation Space

The common image-classification setup starts with a finite augmentation set `A` and a finite
strength set. For the RandAugment-style space, `A` has 14 operations:
`Identity`, `AutoContrast`, `Equalize`, `Rotate`, `Solarize`, `Color`, `Posterize`, `Contrast`,
`Brightness`, `Sharpness`, `ShearX`, `ShearY`, `TranslateX`, and `TranslateY`. Strengths are the
integers `{0, ..., 30}`. Some operations ignore strength (`Identity`, `AutoContrast`,
`Equalize`); others map the strength index to a numeric argument.

Several operations have direction or side relative to the original image. Rotation, shear, and
translation can go positive or negative. In the torchvision wide implementation, the four
enhancers are also represented by a signed magnitude around the identity factor: the operation is
called with factor `1 + magnitude`, so a negative magnitude weakens the attribute and a positive
magnitude strengthens it. Posterize is reversed: fewer retained bits means a stronger operation.
Solarize is also reversed in effect: threshold `255` is essentially a no-op, while threshold `0`
inverts almost everything.

A policy is a distribution over this finite operation-strength surface, possibly with extra
probabilities and composition depth. For a dataset empirical distribution `P_hat`, a uniform
single-operation policy over both operations and strengths would generate the mixture

`(1 / (|A| * 31)) * sum_{a in A} sum_{m=0}^{30} (a_m)_# P_hat`,

where `(a_m)_# P_hat` is the data distribution after applying operation `a` at strength `m`.

## Prior Baselines

**AutoAugment (Cubuk et al., 2019).** A controller RNN trained by PPO emits policies. A policy is
five sub-policies; each sub-policy has two sequential operations; each operation has a probability
and a magnitude. With 16 operations, 10 magnitude bins, and 11 probability bins, the five
sub-policy search space is about `(16 * 10 * 11)^10 ~= 2.9e32`. The controller samples about
15,000 policies. CIFAR search is run on a reduced proxy task - 4,000 CIFAR-10 images and a small
Wide-ResNet-40-2 - and the resulting policy is transferred to full training. The standard CIFAR
pipeline around the learned policy uses normalization, 50 percent horizontal flip,
zero-padding/random crop, and final Cutout.

**RandAugment (Cubuk et al., 2020).** The policy is simplified to two scalars. For each image,
sample `N` operations uniformly from the 14-operation set, apply them sequentially, and use one
global magnitude `M` for all operations. The remaining knobs are `N in {1,2,3}` and
`M in {0,...,30}`, tuned by grid search on the target task.

**UniformAugment (LingChen et al., 2020).** The search-free theory treats augmentation as uniform
coverage of an approximate invariant transform set. In the original algorithm, each selected
operation has a sampled application probability `p ~ U(0,1)` and a sampled continuous magnitude
`lambda ~ U(0,1)`. Some practical comparisons simplify that drop mechanism to a fixed leave-out
probability of `0.5` while preserving random strength.

**Cutout (DeVries and Taylor, 2017).** Cutout masks one random square region of the input image,
for example `16 x 16` pixels on CIFAR-style inputs. It is a single regularizer, closer to a
standard pipeline component than to a full automatic policy; CIFAR automatic-augmentation
experiments commonly append Cutout after the chosen policy.

## Theory and Diagnostic Clues

The supervised objective is expected risk

`R(theta) = E_{(x,y)~P}[loss(f_theta(x), y)]`,

estimated from data by empirical risk

`R_hat(theta) = (1/n) * sum_i loss(f_theta(x_i), y_i)`.

If `T` is a distribution over approximately label-preserving transforms, the augmented empirical
objective is

`R_hat_T(theta) = (1/n) * sum_i E_{t~T}[loss(f_theta(t(x_i)), y_i)]`.

From the group-invariance view used by UniformAugment, the transformed points form an
augmentation orbit around each example. Averaging the loss over that orbit reduces the variance
of the loss and gradient vector when the transforms stay approximately in-class. This supports a
coverage argument: the important ingredient is exposing the optimizer to diverse, label-preserving
views, not finely optimizing the sampling weight of each view.

RandAugment's appendix compares constant, random, linearly increasing, and random-with-
increasing-upper-bound magnitude schedules. Random magnitude ties the best result, while the
constant schedule is chosen mainly because it exposes only one hyperparameter. The same appendix
and sensitivity plots indicate that only a few strength values are often enough.

## Fixed Code Surface

The landing surface is a torchvision-style training transform. The model, optimizer, learning-rate
schedule, and training loop are outside the augmentation method. The method fills one callable
slot that receives a PIL image or image tensor and returns the augmented image; the standard
dataset wrappers and tensor conversion remain fixed:

```python
from torchvision import transforms


class TrainAugmentation:
    def __call__(self, img):
        raise NotImplementedError


def build_train_transform(config):
    return transforms.Compose([
        TrainAugmentation(),
        transforms.RandomCrop(config["img_size"], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])
```
