Self-supervised visual representation learning usually starts from the same setup: take an unlabeled image, create two distorted views with random augmentations, and train a network so the two views produce similar embeddings. The intuition is that whatever survives the distortions is the semantic content we want. But this program has a fundamental degeneracy. If the objective is simply "make the two views agree," the network can satisfy it perfectly by ignoring its input and emitting a constant vector for every image. Every pair agrees, the loss is zero, and nothing is learned. This is the collapse problem.

Existing methods avoid collapse by adding auxiliary machinery. Contrastive methods such as SimCLR and MoCo keep a large pool of negative examples and push embeddings away from other images, which requires enormous batches or memory queues and performs poorly as the embedding dimension grows. BYOL and SimSiam remove negatives but introduce asymmetry between branches, an extra predictor network, stop-gradients, and exponential moving-average targets. Clustering approaches alternate between feature assignment and representation updates. None of these is a single symmetric objective whose optimum is non-trivial by construction; they all keep "agree" as the main objective and bolt on collapse prevention as a side effect of architecture or optimization tricks. What we want instead is a loss whose constant solution is a high-loss state automatically.

I propose Barlow Twins. The core move is to apply Barlow's redundancy-reduction principle to the two-view setting. A good representation should satisfy two desiderata simultaneously: it should be invariant to the augmentation, and its components should be non-redundant, i.e., decorrelated. These two requirements are encoded in a single object, the D-by-D cross-correlation matrix between the two views' embeddings computed along the batch dimension. After mean-centering and standardizing each feature across the batch, entry C_ij measures the correlation between feature i of view A and feature j of view B. The objective asks this matrix to equal the identity. The diagonal terms are pushed to 1, meaning each feature is invariant across the two views. The off-diagonal terms are pushed to 0, meaning distinct features carry independent information. The loss is simply the squared Frobenius distance to the identity matrix: the sum of squared diagonal deviations from 1 plus a small weight times the sum of squared off-diagonal correlations.

This construction excludes collapse without any extra machinery. A constant feature has zero variance across the batch, so after standardization it cannot correlate with itself to 1; the diagonal-to-1 target alone rules out the trivial constant map. A slightly subtler failure mode is to find one informative direction and copy it into every feature. That would satisfy all diagonal targets, but identical or sign-flipped copies are perfectly correlated off-diagonal, so the redundancy term penalizes them. The only zero-loss configuration is the identity cross-correlation matrix: a set of features that are each invariant to the distortion and mutually decorrelated. Because the spread is measured by feature decorrelation rather than by pairwise sample distances, the objective remains cheap to estimate with modest batches and actually benefits from high-dimensional embeddings, in contrast to contrastive methods that saturate as the projector width grows. The small weight on the off-diagonal term reflects the fact that there are order D^2 off-diagonal entries versus only D diagonal entries; a value around 5e-3 prevents the redundancy term from overwhelming the invariance term.

The architecture is straightforward. A ResNet-50 encoder with its classification head removed produces 2048-dimensional backbone features. On top sits a projector with three 8192-wide linear layers: batch normalization and ReLU after the first two, and a bare linear at the end. The projector output is what enters the loss; the backbone output is kept for downstream evaluation. The per-feature standardization is implemented with a non-affine BatchNorm1d. Training uses two augmented views per image with the standard pipeline, LARS optimizer with learning rate scaled by batch size divided by 256, warmup and cosine decay, and the redundancy weight lambda set to 0.0051. In distributed training, batch normalization is converted to synchronized batch normalization so the cross-correlation statistics remain consistent across workers.

```python
import random

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms


class GaussianBlur:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


class Solarization:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class Transform:
    def __init__(self):
        self.transform = self._pipeline(blur_p=1.0, solarize_p=0.0)
        self.transform_prime = self._pipeline(blur_p=0.1, solarize_p=0.2)

    @staticmethod
    def _pipeline(blur_p, solarize_p):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=blur_p),
            Solarization(p=solarize_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform_prime(x)


def build_projector(projector):
    sizes = [2048] + list(map(int, projector.split("-")))
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def pairwise_feature_objective(z1, z2, normalizer, batch_size, weight):
    c = normalizer(z1).T @ normalizer(z2)
    c.div_(batch_size)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(c)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    return on_diag + weight * off_diag


class TwinEmbeddingModel(nn.Module):
    def __init__(self, projector="8192-8192-8192", objective_weight=0.0051, batch_size=2048):
        super().__init__()
        self.objective_weight = objective_weight
        self.batch_size = batch_size
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.projector = build_projector(projector)
        output_dim = int(projector.split("-")[-1])
        self.feature_normalizer = nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        return pairwise_feature_objective(
            z1, z2, self.feature_normalizer, self.batch_size, self.objective_weight
        )


def train_step(model, optimizer, y1, y2):
    optimizer.zero_grad()
    loss = model(y1, y2)
    loss.backward()
    optimizer.step()
    return loss.item()
```
