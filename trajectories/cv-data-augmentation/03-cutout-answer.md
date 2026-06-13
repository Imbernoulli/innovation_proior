**Problem (from step 2).** The photometric-policy family plateaued: RandAugment and TrivialAugment merely *trade* across the CIFAR pairs (one wins CIFAR-10 by 0.15, the other CIFAR-100 by 0.36), moving accuracy around rather than up. The family *diversifies* content but never *removes* it, so a small CIFAR ResNet can still latch onto one most-discriminative region and overfit; diversifying that cue's appearance does not break the latch.

**Key idea.** Switch families to information *deletion*. For each training image, zero a contiguous square region of side L at a uniform-random center — dropout moved to input space with a spatial prior. Removing a *contiguous* block defeats the neighbor-redundancy that makes pointwise removal useless in conv nets; removing it *at the input* makes the content absent from every feature map, forcing the network to recover the class from surrounding context (occlusion robustness, whole-image reasoning).

**Why it works.** Clipping the patch to the image bounds lets near-edge patches land only partly, giving a mix of heavily- and barely-occluded images (a net that sees *only* occluded images overcorrects). The mask runs *after* `ToTensor` and *before* `Normalize` and fills with zeros, so a zeroed pixel becomes `(0-mean)/std` ≈ the dataset mean after normalization — barely disturbing the BN statistics, not a black bias. No test-time rescaling: it is augmentation, not multiplicative noise.

**Scaffold edit / hyperparameters.** Define `Cutout(n_holes=1, length=16)` inside `build_train_transform` and insert it after `ToTensor`, before `Normalize`, keeping the geometric floor (`RandomCrop` + `HFlip`) underneath; the photometric policy is *dropped* (different family — deletion vs diversification). The harness fixes `length=16` for all pairs: optimal for CIFAR-10, slightly over-large for CIFAR-100's 100 fine-grained classes (no per-dataset L tuning exposed). On FashionMNIST the mask runs after the harness's channel-repeat, so the same hole zeros all three channels of the grayscale image.

**What to watch.** Cutout should clear both policy rungs on CIFAR-10 (L=16 is exactly tuned here) and lift FashionMNIST most reliably (a 16×16 occlusion is a strong, content-relevant perturbation the grayscale-muted color policies could not deliver). CIFAR-100 is the bounded pair — at least competitive with 74.71, but a smaller margin since L=16 is too large to shrink. A clear win on the first two with a hold on the third means deletion beats diversification on these tasks.

```python
# EDITABLE region of custom_augment.py (lines 246-275) -- step 3: Cutout
def build_train_transform(config):
    """Cutout augmentation: random square mask after ToTensor.

    Pipeline: RandomCrop + HFlip + ToTensor + Cutout(1, 16) + Normalize.
    """
    class Cutout:
        def __init__(self, n_holes=1, length=16):
            self.n_holes = n_holes
            self.length = length

        def __call__(self, img):
            h, w = img.size(1), img.size(2)
            mask = torch.ones_like(img)
            for _ in range(self.n_holes):
                y = torch.randint(0, h, (1,)).item()
                x = torch.randint(0, w, (1,)).item()
                y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
                x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
                mask[:, y1:y2, x1:x2] = 0
            return img * mask

    return transforms.Compose([
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(config['mean'], config['std']),
    ])
```
