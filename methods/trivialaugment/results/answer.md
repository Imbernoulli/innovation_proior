# TrivialAugment, distilled

TrivialAugment (TA) is a parameter-free, search-free image-augmentation strategy: for each
training image, sample **one** operation uniformly at random from a fixed set `A`, sample **one**
strength uniformly at random from `{0, ..., 30}`, apply it, and return the image. There is no
number-of-operations knob, no per-operation application probability, no global magnitude to tune,
and no search phase — nothing to tune at all. Despite being far simpler than the searched policies
it competes with, it matches or beats them across architectures and datasets.

## Problem it solves

Automatic augmentation methods (AutoAugment, RandAugment, Fast AutoAugment, PBA, UniformAugment)
search the space of augmentation policies over `A × {0,...,30}` to fit a policy to a given
model and dataset. The search ranges from one extra training run to over half a GPU-year, adds a
separate hard-to-reproduce optimization phase, and — because it is usually run on a small proxy
task — is mis-calibrated, since the optimal augmentation strength rises with model and dataset
size. TA delivers comparable accuracy with zero search and zero per-task tuning, in one short file.

## Key idea

The only available theory for why augmentation helps says that averaging each example's loss over
its (approximately label-preserving) augmentation orbit reduces gradient variance and tightens the
empirical-to-expected-risk gap. The active ingredient is therefore **diverse coverage of the
orbit**, not the fine optimization of the sampling distribution — so a uniform, unsearched
sampling should capture the benefit. Stripping every design choice that this coverage argument
does not require:

- **One operation per image** (`N = 1`). With a single op drawn uniformly, the augmented-dataset
  distribution is the plain mixture `(1/|A|) Σ_a (a applied to the dataset)` — the simplest honest
  cover of `A`. Stacking operations adds complexity and a knob, and risks pushing images outside
  the label-preserving orbit the whole argument depends on.
- **No application probability.** "Don't augment" is just sampling `Identity`, which lives in `A`;
  the no-op probability is then exactly `1/|A|`, set by the set rather than a hyperparameter.
- **Random strength per image, uniform over `{0, ..., 30}`.** A random per-step magnitude was
  measured to tie a tuned constant magnitude; it was dropped from prior work only to preserve a
  single tunable knob. Sampling strength per image gives a mixture of weak and strong
  augmentations — coverage of the whole orbit, not one fixed shell.
- **Discrete strengths.** A few distinct magnitudes already suffice; discretizing also limits
  augmentation variance to a useful range (vs. an unbounded continuous draw).

TA is **not** RandAugment with `N = 1`: RandAugment fixes one magnitude `M` for the whole run and
applies it to every image, whereas TA samples a fresh strength per image. That per-image
randomness is the load-bearing difference.

## Final algorithm

```
TA(x):                                   # x: one image, A: augmentation set, strengths {0,...,30}
    a <- uniform sample from A
    m <- uniform sample from {0, ..., 30}
    return a(x, m)
```

The augmentation set `A` (14 operations) is the one prior work converged on: Identity,
AutoContrast, Equalize, Rotate, Solarize, Color, Posterize, Contrast, Brightness, Sharpness,
ShearX, ShearY, TranslateX, TranslateY. Geometric ops (rotate/shear/translate) and the four
enhancers are *signed* — their direction is chosen by a fair coin flip. The strength index
`m ∈ {0,...,30}` maps linearly into each op's range. The **Wide** space uses wide strength ranges
(shear ≤ 0.99, translate ≤ 32 px, rotate ≤ 135°, posterize 8→2 bits, enhancers ~0.01–2.0,
solarize 255→0), which pairs naturally with per-image random strength: a wider range widens the
spread of strengths without committing every image to the extreme.

## Working code

The contribution is the per-image augmentation; it fills the front of the fixed pipeline. The
operation semantics mirror the standard PIL-based implementations (enhancers applied as
`enhance(1.0 + magnitude)` so `m=0` is the identity; posterize as retained-bit count; solarize as
a threshold).

```python
import random
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms


def _apply_op(img, op_name, magnitude):
    if op_name == "Identity":
        return img
    if op_name == "ShearX":
        return img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
    if op_name == "ShearY":
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))
    if op_name == "TranslateX":
        return img.transform(img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0))
    if op_name == "TranslateY":
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude))
    if op_name == "Rotate":
        return img.rotate(magnitude)
    if op_name == "Brightness":
        return ImageEnhance.Brightness(img).enhance(1.0 + magnitude)
    if op_name == "Color":
        return ImageEnhance.Color(img).enhance(1.0 + magnitude)
    if op_name == "Contrast":
        return ImageEnhance.Contrast(img).enhance(1.0 + magnitude)
    if op_name == "Sharpness":
        return ImageEnhance.Sharpness(img).enhance(1.0 + magnitude)
    if op_name == "Posterize":
        return ImageOps.posterize(img, int(magnitude))
    if op_name == "Solarize":
        return ImageOps.solarize(img, magnitude)
    if op_name == "AutoContrast":
        return ImageOps.autocontrast(img)
    if op_name == "Equalize":
        return ImageOps.equalize(img)
    raise ValueError(op_name)


class TrivialAugmentWide:
    """TrivialAugment on the Wide augmentation space: one uniform op + one uniform
    strength per image; signed ops flip sign with probability 1/2. No hyperparameters."""

    def __init__(self, num_magnitude_bins=31):
        self.num_magnitude_bins = num_magnitude_bins

    def _augmentation_space(self, n):
        lin = lambda lo, hi: [lo + (hi - lo) * b / (n - 1) for b in range(n)]
        return {  # op_name: (per-bin magnitudes, signed)
            "Identity":     ([0.0] * n,                                          False),
            "ShearX":       (lin(0.0, 0.99),                                     True),
            "ShearY":       (lin(0.0, 0.99),                                     True),
            "TranslateX":   (lin(0.0, 32.0),                                     True),
            "TranslateY":   (lin(0.0, 32.0),                                     True),
            "Rotate":       (lin(0.0, 135.0),                                    True),
            "Brightness":   (lin(0.0, 0.99),                                     True),
            "Color":        (lin(0.0, 0.99),                                     True),
            "Contrast":     (lin(0.0, 0.99),                                     True),
            "Sharpness":    (lin(0.0, 0.99),                                     True),
            "Posterize":    ([int(round(8 - b / ((n - 1) / 6.0))) for b in range(n)], False),
            "Solarize":     (lin(255.0, 0.0),                                    False),
            "AutoContrast": ([0.0] * n,                                          False),
            "Equalize":     ([0.0] * n,                                          False),
        }

    def __call__(self, img):
        space = self._augmentation_space(self.num_magnitude_bins)
        op_name = random.choice(list(space.keys()))             # one op, uniform over A
        magnitudes, signed = space[op_name]
        m = random.randint(0, self.num_magnitude_bins - 1)      # one strength, uniform
        magnitude = magnitudes[m]
        if signed and random.random() < 0.5:
            magnitude = -magnitude
        return _apply_op(img, op_name, magnitude)


def build_train_transform(config):
    """Fixed pipeline: TA at the front, standard wrappers, mandatory ToTensor + Normalize."""
    return transforms.Compose([
        TrivialAugmentWide(),
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
```

## Relation to prior methods

- **AutoAugment** = a learned policy of 5 sub-policies × 2 ops, each with a probability and a
  magnitude, found by an RL controller over a `~10^32` space at GPU-year cost. TA keeps the same
  operation set and discretization but replaces the entire searched policy with uniform sampling.
- **RandAugment** = uniform op selection but `N` ops chained at a single global magnitude `M`,
  both grid-searched per task. TA removes the grid search, sets `N = 1`, and makes the strength
  per-image-random instead of a fixed `M` (so TA is *not* RandAugment with `N=1`).
- **UniformAugment** = search-free uniform sampling, but with `N = 2`, a per-op application
  probability `p ∼ U(0,1)`, and a continuous strength `λ ∼ U(0,1)`. TA collapses each of those:
  one op, no probability (Identity ∈ A), discrete per-image strength.
- **Cutout** = a single fixed 16×16 masking transform; complementary, conventionally appended
  after the augmentation strategy on CIFAR rather than competing with it.
