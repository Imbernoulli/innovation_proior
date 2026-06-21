The core problem is how to regularize a modern convolutional network on small image benchmarks without adding parameters, losses, or test-time cost. The usual toolkit already includes random crops, horizontal flips, batch normalization, and sometimes dropout, but these leave a specific gap. Dropout works well in fully connected layers because removing one hidden unit from a dense layer genuinely breaks the co-adaptation that unit had with its neighbors. In convolutional layers, the same logic fails: neighboring pixels and neighboring activations are highly correlated, so zeroing a single spatial position usually leaves the same information available through its neighbors. The result is closer to mild noise injection than to the structured model averaging that makes dropout effective. Feature-map-level drop variants move in the right direction, but they operate after the image has already been encoded, so evidence removed in one channel can survive in another. The network then sees an inconsistent noisy representation rather than an image that is actually missing a cue.

The stronger idea is to delete visual evidence before any convolution sees it. If a contiguous region of the input is set to zero, every downstream feature map inherits the same absence. There is no parallel channel preserving the removed cue, and the classifier must base its decision on surrounding context and less dominant features. This is exactly the robustness one wants when the failure mode is over-reliance on a small set of local cues. Scattered pixel corruption is too easy because local redundancy repairs it statistically; a connected missing region is hard enough to force genuine context reasoning, but simple enough to implement as a single data-augmentation transform.

The method is called Cutout. During training only, after converting the image to a tensor and normalizing it per channel, a square mask of fixed side length is placed at a uniformly random center and the covered pixels are multiplied by zero. Because the mask is applied after normalization, the zeroed pixels correspond roughly to the dataset mean, avoiding a strong artificial black patch in raw pixel space and keeping batch statistics from being distorted more than necessary. The square is clipped to the image boundaries when its center is near an edge, so the effective occlusion ranges from the full nominal size down to a small partial patch. This mixture is important: if every training image were strongly occluded, the network would overcorrect; the blend of heavy, light, and everything in between keeps the regularization stable. The label is left unchanged, the loss stays ordinary cross-entropy, and no rescaling is applied at test time, because Cutout is augmentation rather than multiplicative noise.

The reported settings use one hole per image, with length 16 on CIFAR-10, length 8 on CIFAR-100, and length 20 on SVHN. The operation slots directly into the standard torchvision pipeline after ToTensor and normalization, leaving the model and optimizer untouched. The only hyperparameter to tune is the patch length, selected on a held-out validation split. An early targeted variant tried to hide regions with the highest feature activation, but random fixed-size regions matched its accuracy without any saliency bookkeeping, so the final method keeps the random placement. The key design rules are to apply the mask only during training, to apply it after normalization so zero corresponds to the dataset mean, to use a single randomly placed square in the usual setting, to sample the center inside the image and clip the patch at the boundary, to avoid any test-time rescaling, and to keep the original label and cross-entropy objective.

```python
import torch
import numpy as np
from torchvision import transforms


class Cutout(object):
    """Randomly mask out one or more patches from an image."""
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

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    ),
    Cutout(n_holes=1, length=16),
])
```

That is the entire method: a small, cheap input transform that makes the network learn from images with genuinely missing evidence.
