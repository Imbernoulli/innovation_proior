# Contrastive loss (DrLIM), distilled

DrLIM — Dimensionality Reduction by Learning an Invariant Mapping — learns a single
parametric, shared-weight (siamese) embedding `G_W: R^D -> R^d` (`d << D`) so that ordinary
Euclidean distance in the output space reflects a *given* neighborhood relation (similar /
dissimilar), with no input-space distance assumed and with invariance to nonlinear input
transformations learned from data. The contribution is the **contrastive loss**: an attract-only
term that pulls similar pairs together plus a margin repulsion that pushes dissimilar pairs apart,
which together make a constant (collapsed) embedding a high-energy, non-optimal state.

The **naive baseline** is this loss with the repulsion deleted — only the similar-pair term,
`1/2 ||z1 - z2||^2` — i.e. minimizing the distance between two views with nothing opposing it.
A global optimum is the constant map (collapse), which is exactly why it serves as the lower-bound
baseline.

## Problem it solves

Map high-dimensional points to a low-dimensional space such that a *simple* output distance
matches a relational similarity signal (neighbors vs non-neighbors) supplied from prior
knowledge, labeling, temporal proximity, or known invariances. Requirements the classical
reducers (PCA/MDS; LLE/Isomap/Laplacian Eigenmaps/Hessian LLE; Kernel PCA) fail to meet
simultaneously: produce a reusable *function* that generalizes to unseen inputs (not just embed
the training set), avoid being tied to an input-space distance/kernel or fixed neighbor list, and
learn invariance to complex nonlinear transformations.

## Key idea

Use a shared-weight siamese network: one differentiable `G_W` applied to both members of a pair,
output-space distance `D_W = ||G_W(X1) - G_W(X2)||_2`, and a pair loss gated by a binary relation
label `Y` (0 = similar, 1 = dissimilar):

- **Attract term** (similar, `Y=0`): `L_S = 1/2 * D_W^2`. Gradient `D_W * dD_W/dW` — an
  attract-only spring (rest length 0): force grows with distance, vanishes at coincidence.
- **Repel term** (dissimilar, `Y=1`): `L_D = 1/2 * max(0, m - D_W)^2`. Gradient
  `-(m - D_W) * dD_W/dW` for `D_W < m`, else `0` — an `m`-rest-length repulse-only spring:
  pushes apart only when closer than the margin `m`, hardest at coincidence, zero at and beyond
  `m`.

Why the repulsion is required: the attract term alone has `G_W ≡ const` as a global minimizer,
since then every similar `D_W = 0` and the loss is 0 — collapse. This is the
**energy-based-model asymmetry**: an unnormalized energy lowered on
similar pairs has no automatic counter-pressure, unlike a normalized probability where raising
one configuration's score lowers others'. The explicit contrastive term supplies that
counter-pressure by *raising* the energy of dissimilar pairs; at the collapsed point every
dissimilar pair is at distance 0 and maximally violates the margin, so collapse becomes a
high-loss state. The attract/repel balance also spreads points to cover the output manifold
evenly rather than clustering. The squared (not bare) hinge gives a linear-in-displacement force
that vanishes smoothly at the margin; the margin `m` sets the embedding scale and stops effort on
already-far pairs.

## Final objective

```
D_W(X1, X2) = || G_W(X1) - G_W(X2) ||_2                       (Euclidean output distance)

L(W, Y, X1, X2) = (1 - Y) * 1/2 * (D_W)^2                      (Y = 0: similar / attract)
                +    Y    * 1/2 * { max(0, m - D_W) }^2        (Y = 1: dissimilar / repel)
```

Total loss sums over labeled pairs (up to O(P^2) of them); trained by stochastic gradient
descent through the two shared instances of `G_W` (total gradient = sum of the two
contributions). Gradients:

```
dL_S/dW = D_W * dD_W/dW                                        (attractive, ~ Hooke F = -K X)
dL_D/dW = -(m - D_W) * dD_W/dW   if D_W < m,   else 0          (repulsive, zero past margin m)
```

`G_W` need only be differentiable in `W`; a convolutional network is the natural choice for pixel
images (shared weights / multiple layers learn shift-invariant local features), a small MLP for
low-resolution object images. Output dimension `d << D`.

## Working code (siamese harness + contrastive loss)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GW(nn.Module):
    """Shared embedding G_W: R^D -> R^d, d << D. Same weights applied to both
    members of a pair (siamese); differentiable in W. Body is generic (MLP here;
    a conv net for pixel images)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveLoss(nn.Module):
    """L = (1 - Y) * 1/2 * D_W^2  +  Y * 1/2 * max(0, m - D_W)^2,
    D_W = ||G_W(x1) - G_W(x2)||_2,  Y = 0 (similar) / 1 (dissimilar)."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, Y):
        D = torch.norm(z1 - z2, dim=-1)                  # D_W
        sim = 0.5 * D.pow(2)                             # attract-only (rest length 0)
        dis = 0.5 * F.relu(self.margin - D).pow(2)       # m-repulse-only (rest length m)
        return ((1.0 - Y) * sim + Y * dis).mean()


def train(gw, criterion, pairs, opt):
    """Siamese loop; each batch is (x1, x2, Y). Shared weights -> the gradient
    into W is the sum of contributions from both instances of G_W."""
    for x1, x2, Y in pairs:
        z1, z2 = gw(x1), gw(x2)
        loss = criterion(z1, z2, Y.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
```

## The naive baseline (lower bound)

Keep only the `Y = 0` (attract) branch and feed two transformed views of the same input as the
similar pair. The PyTorch baseline is the bare mean squared error between view embeddings, which
is the same attract-only objective up to averaging and a constant scale:

```python
import torch.nn.functional as F


def naive_invariance_loss(z1, z2):
    # Y = 0 branch alone: attract-only, no repulsion.
    # Global optimum is G_W = const (collapse) -- the degenerate lower-bound baseline.
    return F.mse_loss(z1, z2)
```

With no dissimilar-pair term in the objective, a constant encoder makes every such pair distance
zero and is a global minimizer. The contrastive term is precisely what this baseline omits.
