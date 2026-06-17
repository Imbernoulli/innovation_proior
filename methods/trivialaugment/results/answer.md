# TrivialAugmentWide, distilled

TrivialAugment samples one operation and one strength for each training image:

```text
TA(x):
    a ~ Uniform(A)
    m ~ Uniform({0, ..., 30})
    return a(x, m)
```

There is no searched policy, no number-of-operations knob, no application probability, and no
global magnitude. `Identity` is part of `A`, so "no augmentation" is just one outcome of the same
uniform operation draw.

## Canonical Wide Space

For the torchvision `TrivialAugmentWide` landing, `A` has 14 operations and `num_magnitude_bins`
is fixed at `31`.

| Operation | Magnitude bins | Signed |
| --- | --- | --- |
| `Identity` | scalar `0` | no |
| `ShearX`, `ShearY` | `linspace(0, 0.99, 31)` | yes |
| `TranslateX`, `TranslateY` | `linspace(0, 32, 31)`, cast to integer pixels | yes |
| `Rotate` | `linspace(0, 135, 31)` degrees | yes |
| `Brightness`, `Color`, `Contrast`, `Sharpness` | `linspace(0, 0.99, 31)`, applied as factor `1 + magnitude` | yes |
| `Posterize` | `8 - round(arange(31) / 5)`, retained bits from `8` down to `2` | no |
| `Solarize` | `linspace(255, 0, 31)` threshold | no |
| `AutoContrast`, `Equalize` | scalar `0` | no |

For shears, torchvision passes `degrees(atan(magnitude))` into `F.affine`; this makes the tangent
of the shear angle equal the sampled shear factor. The sign flip is a fair coin for signed
operations. The enhancer maximum is `0.99` in torchvision, so signed factors span `0.01` to
`1.99`; the paper's appendix rounds the wide enhancer range as `0.01` to `2.0`, and the original
AutoML one-file code uses that direct factor range.

## Faithful Scaffold Code

Use the canonical torchvision implementation rather than a hand-rolled PIL approximation:

```python
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_train_transform(config):
    return transforms.Compose([
        transforms.TrivialAugmentWide(
            num_magnitude_bins=31,
            interpolation=InterpolationMode.NEAREST,
        ),
        transforms.RandomCrop(config["img_size"], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])
```

This is not RandAugment with `N = 1`: RandAugment still fixes one tuned magnitude `M` for the
whole run, while TrivialAugment samples a fresh strength bin for each image.
