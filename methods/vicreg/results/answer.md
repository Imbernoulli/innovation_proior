# VICReg: Variance-Invariance-Covariance Regularization

## Problem

Joint-embedding self-supervised learning trains a network so that two augmented views of an image
produce similar embeddings. The bare "make views agree" objective is solved by a constant vector —
**collapse**. Prior methods avoid collapse with negatives (SimCLR/MoCo), clustering (SwAV), asymmetric
predictor + stop-gradient / momentum encoder (SimSiam/BYOL), or a cross-branch decorrelation matrix
with embedding standardization (Barlow Twins). All of these either need large batches/memory, are
dynamics-dependent and ill-understood, or *couple* the two branches (shared weights, EMA, or a
cross-correlation), which forces the branches to be identical.

## Key idea

Prevent collapse with two **explicit regularizers applied to each branch separately**, plus a simple
invariance term — no negatives, no momentum encoder, no stop-gradient, no predictor, no quantization,
no normalization of the embeddings. Because the regularizers are per-branch, the two branches need
share nothing: they can have different architectures, weights, or input modalities.

Given two batches of embeddings `Z = [z_1..z_n]`, `Z' = [z'_1..z'_n]` (each `n × d`), write `z^j` for
the j-th coordinate across the batch.

1. **Invariance** — pull the two views together, plain MSE, no normalization:
   `s(Z, Z') = (1/n) Σ_i ‖z_i − z'_i‖²`.

2. **Variance** — hinge keeping each dimension's batch standard deviation above a floor `γ`:
   `v(Z) = (1/d) Σ_j max(0, γ − √(Var(z^j) + ε))`, with `γ = 1`, `ε = 1e-4`.
   Using the **standard deviation, not the variance**, is essential: `d√Var/dVar = 1/(2√Var) → ∞` as
   `Var → 0`, so the restoring gradient is strongest exactly at collapse; the variance-in-the-hinge
   variant has vanishing gradient there and fails to escape collapse. This term forbids **trivial
   collapse** (everything shrinking to a point — zero variance).

3. **Covariance** — drive every off-diagonal of each branch's embedding covariance to zero:
   `C(Z) = (1/(n−1)) Σ_i (z_i − z̄)(z_i − z̄)ᵀ`, `c(Z) = (1/d) Σ_{i≠j} [C(Z)]²_{ij}`.
   This decorrelates the dimensions and forbids **informational collapse** (the guaranteed variance
   being duplicated into a low-dimensional subspace). Borrowed from Barlow Twins' decorrelation, but on
   each branch's covariance, not a cross-correlation between branches — so no inter-branch coupling and
   no embedding standardization is needed (the variance term owns the scale).

**Why both are needed.** Variance alone permits all dimensions to copy one informative direction
(informational collapse). Covariance alone collapses outright — the cheapest way to zero all
off-diagonal covariances is to send everything to a constant (all covariances zero). Together,
variance forces per-dimension spread and covariance spreads that variance across `d` decorrelated
dimensions; invariance ties the two views so the budget is spent on augmentation-stable features.

## Objective

```
ℓ(Z, Z') = λ·s(Z, Z') + μ·[v(Z) + v(Z')] + ν·[c(Z) + c(Z')]
```

On ImageNet: `λ = μ = 25`, `ν = 1`, `γ = 1`, `ε = 1e-4`. The recipe is `λ = μ > ν` (balancing the two
terms that fight — invariance vs. variance — and keeping the ~`d²`-term covariance gradient from
destabilizing training); `λ = μ` with `ν > μ` is unstable, `λ = μ` with `ν < μ` is stable, and the
exact value of `λ = μ` matters little.

## Architecture / training

- **Encoder** `f_θ`: ResNet-50, 2048-d representation (kept for downstream; the head is discarded
  after pretraining).
- **Expander** `h_φ`: MLP that *expands* the representation (e.g. `2048-8192-8192-8192`), Linear + BN +
  ReLU on hidden layers, final plain Linear (`bias=False`), no normalization on the output. Expanding
  (not projecting down) lets linear decorrelation in the embedding remove higher-order *dependencies*
  in the representation; performance rises with embedding width and saturates around 8192.
- **Optimizer**: LARS, 1000 epochs, weight decay `1e-6`, `lr = base_lr × batch/256` with
  `base_lr = 0.2`, batch 2048, cosine decay with 10 warmup epochs, final lr `0.002`.
- **Augmentations**: random resized crop (224), horizontal flip, color jitter, grayscale, Gaussian
  blur, solarization, ImageNet normalization.
- **No embedding normalization**: l2-normalizing the embeddings hurts ~3.5%; standardizing them turns
  the covariance into a correlation in a narrow `[−1,1]` range and hurts ~0.2%.

## Code

```python
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])      # embedding dim d
        self.backbone, self.embedding = build_resnet(args.arch, zero_init_residual=True)
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))                  # (n, d)
        y = self.projector(self.backbone(y))

        # invariance: MSE between paired views, no normalization
        repr_loss = F.mse_loss(x, y)

        # gather across GPUs, then center each dimension over the batch
        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        # variance: hinge on the STD (gamma = 1 -> `1 - std`), eps = 1e-4
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # covariance: off-diagonal squared, per branch, scaled by 1/d
        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) \
            + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss      # lambda = 25
            + self.args.std_coeff * std_loss     # mu     = 25
            + self.args.cov_coeff * cov_loss     # nu     = 1
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"          # e.g. 2048-8192-8192-8192
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

Training step: sample a batch, build two views `x, y`, `loss = model(x, y)`, `loss.backward()`,
`optimizer.step()` with LARS (bias/norm params excluded from weight decay and LARS adaptation) under a
warmup + cosine learning-rate schedule. No target network, no memory bank, no stop-gradient.

After pretraining, discard the expander and use `model.backbone` (the frozen ResNet-50 representation)
for downstream tasks.
