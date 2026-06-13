**Problem (from step 1).** RandAugment lifted the CIFAR pairs over the geometric floor, but its strength is a *fixed* operating point — `num_ops=2`, `magnitude=9` applied uniformly to every image — that the harness gives no way to tune, and two composed strong ops per image risk eroding class-discriminative signal on the hard CIFAR-100 pair. The suspect is the un-tunable fixed strength, not diversity itself.

**Key idea.** Remove the knobs entirely. Keep RandAugment's trusted move (uniform op sampling forces diversity) but strip the second move to its floor: apply *one* operation per image (no composition), and sample its magnitude *uniformly at random* per image from the operation's full range instead of fixing it. The magnitude-insensitivity result that justified tying magnitudes also says the specific value barely matters — so do not pick one; sample it. Zero hyperparameters.

**Why it works.** Single-op sampling preserves cross-dataset diversity (every op still appears) while removing the over-composition risk. Random per-image magnitude removes M as a tunable (nothing to mis-set) and adds *strength* diversity — weak perturbations that preserve signal, strong ones that force invariance — without choosing where to sit. On a regime RandAugment over-regularized, the lower expected strength eases off and should help; elsewhere the spread averages to a comparable effect.

**Scaffold edit / hyperparameters.** The task exposes torchvision's `TrivialAugmentWide`, which is exactly this algorithm (one uniform op, magnitude uniform over a wide range) and is parameter-free. The literal edit puts `TrivialAugmentWide()` at the front of the PIL-space pipeline — the same position RandAugment held — with the rest identical, making this a clean controlled swap of the policy block. The FashionMNIST grayscale caveat (policy sees a 1-channel PIL image; color ops act on grayscale) applies as before.

**What to watch.** CIFAR-100 is the decisive pair: easing from two fixed-strong ops to one random-magnitude op should land ResNet-56 *at or above* 74.35 if RandAugment was over-regularizing it. CIFAR-10 roughly even with 93.51 (single op at random magnitude is on average gentler). FashionMNIST a wash near 94.4.

```python
# EDITABLE region of custom_augment.py (lines 246-275) -- step 2: TrivialAugment
def build_train_transform(config):
    """TrivialAugmentWide: single random op with random magnitude.

    Pipeline: TrivialAugmentWide() + RandomCrop + HFlip + ToTensor + Normalize.
    """
    return transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
```
