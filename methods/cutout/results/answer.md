# Cutout

## Problem

Large CNNs overfit, and the usual activation-level regularizer — dropout — is much weaker inside convolutional layers than fully-connected ones. The reason is diagnostic: neighboring pixels (and activations) in an image are highly redundant, so zeroing a single unit removes almost no information — the neighbors pass it forward anyway. Per-map dropout variants (SpatialDropout, max-drop) attack this but fall behind plain dropout once batch normalization is present. The goal is a regularizer that genuinely helps convolutional networks, survives stacking on top of batch norm and standard augmentation, costs nearly nothing, and is trivial to implement.

## Key idea

Move the removal to the input layer and make it contiguous: during training, zero out a fixed-size square region at a random location of each image. Because the content is removed once, at the input, it is absent from *every* downstream feature map (unlike per-map dropout, where it survives in other maps); because the region is contiguous, neighbor-redundancy cannot fill it back in. The network is forced to recover the class from surrounding context — exactly occlusion-robustness. Cutout is dropout in input space with a spatial prior, and behaves more like data augmentation (generating plausibly-occluded images) than like multiplicative noise.

## Why it works

- **Input-layer removal** propagates the hole through all feature maps, so the removed content is genuinely gone — not merely noisy and inconsistent as with per-map dropout variants.
- **Contiguous square** defeats the neighbor-redundancy that makes pointwise removal ineffective in conv layers.
- **Occlusion simulation** teaches the model to use the full image context instead of betting on a single most-discriminative feature, improving robustness when that feature is absent.
- **Mix of corrupted and clean images** — achieved by letting the patch hang past the image borders so near-edge patches only partly land — is critical; a model that sees *only* occluded images overcorrects.

## Design choices

- **Drop at the input, not intermediate features**: removed content disappears from every map, forcing context use.
- **Contiguous region, not scattered pixels**: scattered removal is undone by neighbor redundancy.
- **Square patch, one length knob L**: the *amount* removed matters far more than the shape; accuracy is parabolic in L (rises to an optimum, then drops below baseline). More classes → smaller optimal L (fine detail matters more than context): CIFAR-10 L=16, CIFAR-100 L=8, SVHN L=20.
- **Uniform random center, patch allowed past borders**: yields a mix of heavily- and barely-occluded images. (Equivalent alternative: constrain the patch fully inside but apply cutout with 50% probability.)
- **Zero fill on zero-mean-normalized data**: a zeroed patch sits at ≈ the dataset mean, so batch statistics are barely disturbed; no information is injected, unlike random-noise fill.
- **No test-time rescaling**: it is augmentation, not multiplicative noise needing an expectation correction — test images are fed unmodified.
- **Random square, not targeted/saliency-based**: a targeted (max-activation) version works but requires per-epoch saliency maps; random fixed-size squares match it and are far simpler and free.

## Code

A single transform appended to the data-loading pipeline; the network, optimizer, schedule, and loss are unchanged. It runs on the CPU during loading, effectively for free.

```python
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Cutout(object):
    """Randomly mask out n_holes square patches of side `length` from an image."""

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    Cutout(n_holes=1, length=16),
])
trainset = datasets.CIFAR10(root='~/data', train=True, download=False,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                           shuffle=True, num_workers=8)
```

Typical settings: WRN-28-10 or ResNet-18, 200 epochs, batch 128, SGD with Nesterov momentum 0.9 and weight decay 5e-4, learning rate 0.1 decayed by 5× at epochs 60/120/160, standard pad-crop + horizontal-flip augmentation underneath, per-channel mean/std normalization, one hole of side length tuned by validation grid search (16 on CIFAR-10, 8 on CIFAR-100, 20 on SVHN). Cutout stacks on top of existing dropout, batch norm, and augmentation.
