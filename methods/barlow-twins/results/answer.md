# Barlow Twins

## Problem

Self-supervised joint-embedding learning trains a network so that two augmented views of an image map
to similar representations. The bare "make the views agree" objective is degenerate: a constant output
satisfies it perfectly (collapse). Existing methods avoid collapse with extra machinery — negative
samples (SimCLR/MoCo/InfoNCE), an asymmetric predictor + stop-gradient + EMA target (BYOL/SimSiam),
or a clustering step (SwAV) — none of which is the minimization of a single objective whose optimum is
non-trivial by construction.

## Key idea

Apply Barlow's redundancy-reduction principle: a good representation should be invariant to the
distortion *and* have decorrelated (non-redundant) components. Encode both in the **cross-correlation
matrix** between the two views' embeddings, computed along the batch dimension, and make it the
identity. The diagonal-to-1 condition gives invariance; the off-diagonal-to-0 condition decorrelates
the components. Decorrelation is what makes collapse impossible — no negatives, no predictor, no
stop-gradient, no momentum encoder, no clustering, fully symmetric twin networks.

## Method

Two views A, B of a batch of N images go through the **same** network f_theta, producing embeddings
Z^A, Z^B (each N×D). Mean-center each feature across the batch. The cross-correlation matrix (D×D) is

```
            sum_b  z^A_{b,i} z^B_{b,j}
  C_ij = -------------------------------------------------------,    C_ij in [-1, 1]
         sqrt( sum_b (z^A_{b,i})^2 ) * sqrt( sum_b (z^B_{b,j})^2 )
```

and the objective is

```
  L = sum_i (1 - C_ii)^2  +  lambda * sum_i sum_{j != i} C_ij^2
        \________________/        \_______________________/
         invariance term           redundancy-reduction term
```

- **Invariance term** drives each diagonal entry to 1: feature i of view A and feature i of view B are
  perfectly correlated across the batch, i.e. feature i is invariant to the distortion.
- **Redundancy-reduction term** drives each off-diagonal entry to 0: distinct features are
  decorrelated and carry non-redundant information.

**Why no collapse.** Normalizing each feature by its batch standard deviation makes a constant feature
(zero batch variance) unable to reach C_ii = 1, so the invariance term alone excludes the constant
solution. The off-diagonal term forbids the alternative cheap escape of copying one informative
direction into all features (copies, including sign-flipped copies, have |C_ij| = 1 and are
penalized). The zero-loss correlation target is C = I: D features each invariant to the distortion
and mutually decorrelated.

**Information-bottleneck view.** With IB objective `I(Z,Y) - beta I(Z,X)` and deterministic encoder
(`H(Z|Y)=0`), the objective reduces to `H(Z|X) + ((1-beta)/beta) H(Z)`. For beta > 1, the second
coefficient is negative, so minimizing the objective minimizes distortion-induced variability while
maximizing representation entropy. Under a Gaussian model `H(Z) = (1/2) log|Cov(Z)|`; with
unit-variance rescaling, Hadamard's inequality gives the maximum determinant at the identity
covariance, so zeroing off-diagonal correlations is a tractable entropy proxy. The loss weight
`lambda` is the practical trade-off attached to this entropy proxy, not numerically equal to `beta`.

**Architecture / optimization.** Encoder = ResNet-50 (2048-d, classifier head removed) + projector of
three 8192-wide linear layers (BN+ReLU after the first two, bare linear last). Per-feature batch
normalization defines C. LARS optimizer, learning rate scaled by batch/256 with 10-epoch warmup and
cosine decay to lr/1000, weight decay 1e-6, biases/BN excluded from LARS and weight decay, lambda =
0.0051, batch size 2048. BYOL-style augmentations with view-dependent blur and solarization
probabilities. In distributed training, the batch-normalization layers are converted to synchronized
batch normalization before wrapping the model.

## Code

```python
import torch
import torch.nn as nn
import torchvision


def off_diagonal(x):
    # flattened view of the off-diagonal elements of a square (n x n) matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, projector="8192-8192-8192", lambd=0.0051, batch_size=2048):
        super().__init__()
        self.lambd = lambd
        self.batch_size = batch_size

        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector MLP: wide on purpose (high-dim embeddings help)
        sizes = [2048] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # non-affine BN = per-feature mean-center + divide by batch std (defines C)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        # In distributed training, convert the module with
        # nn.SyncBatchNorm.convert_sync_batchnorm(model) before DDP wrapping.

    def forward(self, y1, y2):
        # same network on both views; no asymmetry of any kind
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix (D x D)
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(self.batch_size)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)  # sum C across GPUs

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()   # invariance: diag -> 1
        off_diag = off_diagonal(c).pow_(2).sum()             # redundancy: off-diag -> 0
        loss = on_diag + self.lambd * off_diag
        return loss


# --- two-view augmentation pipeline (BYOL-style, view-dependent blur/solarize) ---
from PIL import ImageFilter, ImageOps
import random
from torchvision import transforms
from PIL import Image


class GaussianBlur:
    def __init__(self, p): self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(random.random() * 1.9 + 0.1))
        return img


class Solarization:
    def __init__(self, p): self.p = p
    def __call__(self, img):
        return ImageOps.solarize(img) if random.random() < self.p else img


class Transform:
    def __init__(self):
        common = [
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        finish = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform       = transforms.Compose(common + [GaussianBlur(1.0), Solarization(0.0)] + finish)
        self.transform_prime = transforms.Compose(common + [GaussianBlur(0.1), Solarization(0.2)] + finish)

    def __call__(self, x):
        return self.transform(x), self.transform_prime(x)


# --- training step ---
def train_step(model, optimizer, y1, y2):
    loss = model(y1, y2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()   # LARS; lr scaled by batch/256, warmup + cosine decay
    return loss.item()
```

The 2048-d backbone output (the "representations") is what is used for downstream tasks; the wide
projector output (the "embeddings") is used only inside the loss and discarded after pretraining.
