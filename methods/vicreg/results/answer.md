# VICReg: Variance-Invariance-Covariance Regularization

## Method

For two embedding batches \(X,Y\in R^{B\times d}\), use a three-part objective:

\[
s_{\text{math}}(X,Y)=\frac{1}{B}\sum_{b=1}^B \|x_b-y_b\|_2^2
\]

\[
v(X)=\frac{1}{d}\sum_{j=1}^d \max\left(0,\gamma-\sqrt{\operatorname{Var}(X_{:,j})+\epsilon}\right)
\]

\[
C(X)=\frac{1}{B-1}\sum_{b=1}^B (x_b-\bar x)(x_b-\bar x)^T,\qquad
c(X)=\frac{1}{d}\sum_{i\ne j} C(X)_{ij}^2
\]

\[
\ell_{\text{math}}(X,Y)=\lambda s_{\text{math}}(X,Y)+\mu[v(X)+v(Y)]+\nu[c(X)+c(Y)].
\]

Published ImageNet settings use \(\gamma=1\), \(\epsilon=10^{-4}\), \(\lambda=\mu=25\), and \(\nu=1\).
The standard-deviation hinge is the important anti-collapse detail: in the active region its gradient
is proportional to \(-(x_k-\bar x)/\sqrt{\operatorname{Var}(x)+\epsilon}\), so small nonzero
deviations get amplified much more than they would under a variance hinge. With nonzero
\(\epsilon\), an exactly constant column has zero embedding-gradient but positive loss; the term is a
near-collapse gradient amplifier and removes the constant solution as a zero-loss optimum.

Variance and covariance are both required. Variance alone permits copied coordinates; covariance
alone is minimized by a constant batch. Invariance makes the non-collapsed, decorrelated coordinates
stable across augmentations.

## Canonical PyTorch Core

The official `facebookresearch/vicreg` code uses slightly different constant scaling from those
equations: `F.mse_loss` is an elementwise mean over `B*d`, and the two branch variance losses are
averaged with `/ 2`. The tuned command-line coefficients are calibrated to this implementation.

```python
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features)
        cov_loss += off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    f = list(map(int, mlp_spec.split("-")))
    layers = []
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
```

Implementation details that matter for faithfulness:

- Gather embeddings from all distributed workers before estimating variance and covariance.
- Center each branch before covariance; centering before `x.var(dim=0)` is harmless because variance
  is translation-invariant.
- In the original PyTorch 1.8 code, `x.var(dim=0)` uses the unbiased sample variance, consistent with
  the covariance denominator `batch_size - 1`.
- Use the effective global batch size in the covariance denominator after gathering.
- Use `sim_coeff=25`, `std_coeff=25`, `cov_coeff=1`, `mlp="8192-8192-8192"`.
- Hidden projector layers are Linear + BatchNorm + ReLU; the final embedding layer is Linear with
  `bias=False` and no final normalization.
- LARS excludes one-dimensional parameters, such as biases and normalization parameters, from weight
  decay and LARS adaptation.
- The official augmentation code uses two BYOL-style views: both use random resized crop, flip,
  color jitter with probability 0.8, grayscale with probability 0.2, and ImageNet normalization; view
  one uses Gaussian blur probability 1.0 and solarization 0.0, view two uses Gaussian blur 0.1 and
  solarization 0.2.

For the 1000-epoch ImageNet run, the reference command uses batch size 2048 and base learning rate
0.2. The code sets `base_lr_eff = base_lr * batch_size / 256`, warms up for 10 epochs, then cosine
decays to `base_lr_eff * 0.001`.
