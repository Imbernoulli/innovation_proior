# Cutout

Cutout is a training-time image regularizer: after tensor conversion and per-channel normalization, randomly zero a clipped square region of each training image while leaving the label, model, loss, optimizer, and test-time pipeline unchanged.

The method is motivated by the failure mode of convolutional dropout. Pointwise removal is weak when neighboring pixels or activations carry nearly the same information. Removing a contiguous input region makes the visual cue absent from every downstream feature map, so the classifier must use surrounding context and less dominant features.

## Exact transform

```python
import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
```

For the even lengths used in the reported settings, an unclipped patch is exactly `length x length`. For an odd `length`, the canonical code masks `2 * (length // 2)` pixels per side because the upper slice bound is exclusive. At borders, the actual masked area is clipped to `(y2 - y1) * (x2 - x1)`, which is intentional.

## Training placement

```python
from torchvision import transforms


normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
)

train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
train_transform.transforms.append(Cutout(n_holes=1, length=16))
```

For SVHN, the reference script uses `mean=[109.9, 109.7, 113.8] / 255` and `std=[50.1, 50.6, 50.8] / 255`, no data augmentation, `length=20`, and learning-rate milestones `[80, 120]` with `gamma=0.1`. For CIFAR, it uses optional crop/flip augmentation, `length=16` on CIFAR-10, `length=8` on CIFAR-100, and milestones `[60, 120, 160]` with `gamma=0.2`.

## Design rules

- Apply the mask only during training.
- Apply it after normalization so zero corresponds approximately to the dataset mean.
- Use a single randomly placed square in the usual setting.
- Sample the center inside the image and clip the square at the boundary; this creates both strongly and lightly occluded examples.
- Do not rescale weights or activations at test time.
- Keep the original label and the ordinary cross-entropy objective.

The early targeted version used stored high-activation feature maps to choose what to hide, but random fixed-size regions matched it in the original experiments without the extra feature-map bookkeeping.
