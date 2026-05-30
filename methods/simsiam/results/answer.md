# SimSiam — Simple Siamese Representation Learning

## Problem

Learn visual representations from unlabeled images with a Siamese "two views should agree"
objective, *without* collapsing to a constant — and do it without the machinery that prior methods
use to prevent collapse: no negative pairs, no momentum encoder, no large batches.

## Key idea

Take the bare Siamese network — a *single* encoder `f` with weights shared between the two
branches (no momentum) — add a prediction MLP `h` on one branch, use the symmetrized negative-
cosine loss, and apply a **stop-gradient** on the target branch. Empirically the stop-gradient is
the one ingredient that is *necessary and sufficient* to prevent collapse; removing it (with
everything else fixed) sends the loss to its minimum `-1` and the output std to zero. Negatives,
the momentum encoder, large batches, BN, symmetrization, and the cosine form all affect accuracy
or optimization but are *not* what prevents collapse.

## Method

Two augmented views `x1, x2` of image `x`. Encoder `f` (backbone + 3-layer projection MLP),
predictor `h` (2-layer bottleneck MLP). With `p1 = h(f(x1))`, `z2 = f(x2)`, the per-term distance
is the negative cosine similarity

    D(p, z) = - (p/‖p‖₂) · (z/‖z‖₂),

equal to the MSE of `ℓ₂`-normalized vectors up to a factor of 2. The symmetrized loss with
stop-gradient (`sg`) on the targets:

    L = ½ D(p1, sg(z2)) + ½ D(p2, sg(z1)),     min value −1.

The target branch (`z`) is treated as a constant; the prediction branch (`p`) is trained.

**Why it works (alternating-optimization / EM hypothesis).** Introduce a per-image target set `η`
(one vector per image, not a network output) and consider

    L(θ, η) = E_{x, T} ‖ F_θ(T(x)) − η_x ‖²,   solve  min_{θ, η} L(θ, η)

by alternating (analogous to k-means: `θ` ↔ centers, `η_x` ↔ assignments):

    θ^t ← argmin_θ L(θ, η^{t−1})      # SGD; η^{t−1} fixed ⇒ stop-gradient is the consequence
    η^t ← argmin_η L(θ^t, η)          # per image: η^t_x = E_T[ F_{θ^t}(T(x)) ]  (the mean)

One-step alternation = SimSiam: approximate `η^t_x ≈ F_{θ^t}(T'(x))` by sampling one augmentation
`T'` (the "other view"), substitute, and take a single SGD step on `θ` — yielding a shared-weight
Siamese net with stop-gradient. The predictor `h` approximates the dropped expectation `E_T[·]`
(its optimum is `h(z1) = E_T[f(T(x))]`); symmetrization is denser sampling of the expectation.
The constant solution exists but the alternating optimizer, starting from a scattered random init
and updating each `η_x` independently (never a joint gradient over all images), does not fall into
it.

## Design choices

- **Stop-gradient on the target branch** — the load-bearing piece; the natural consequence of
  holding `η` fixed in the alternation.
- **Predictor `h` is required** — for the symmetric loss, removing `h` makes the gradient equal to
  that of `D(z1, z2)` scaled by ½, rendering stop-gradient vacuous → collapse. `h` is a 2-layer
  **bottleneck** (hidden = dim/4) and uses a **constant (non-decayed) lr** so it keeps adapting to
  the moving representation.
- **No momentum encoder / no negatives / SGD at ordinary batch sizes** — none are needed for
  collapse prevention; works from batch 64 to 2048 (4096 hurts via large-batch SGD, not collapse).
- **BN** on projection hidden+output and predictor hidden (none on predictor output); helps
  optimization, not collapse prevention. Projection output BN has no affine.
- **Defaults:** ResNet-50; dim 2048, predictor hidden 512; SGD lr `0.05×bs/256`, cosine decay,
  momentum 0.9, weight decay 1e-4.

## Code

```python
import math
import torch
import torch.nn as nn
import torchvision.models as models


class SimSiam(nn.Module):
    """SimSiam: shared encoder f (backbone + projection MLP) and predictor h."""
    def __init__(self, base_encoder=models.resnet50, dim=2048, pred_dim=512):
        super().__init__()
        # encoder f: backbone with its fc replaced by a 3-layer projection MLP
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim), nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim), nn.ReLU(inplace=True),
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False))          # output BN, no affine
        self.encoder.fc[6].bias.requires_grad = False    # bias absorbed by BN

        # predictor h: 2-layer bottleneck MLP (hidden = dim/4)
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim), nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim))

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()          # stop-gradient on targets


criterion = nn.CosineSimilarity(dim=1)                   # D(p, z) = -(p̂ · ẑ)

def simsiam_loss(p1, p2, z1, z2):
    return -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5


def build_optimizer(model, base_lr=0.05, batch_size=512,
                    momentum=0.9, weight_decay=1e-4):
    init_lr = base_lr * batch_size / 256
    param_groups = [
        {'params': model.encoder.parameters(),   'fix_lr': False},
        {'params': model.predictor.parameters(), 'fix_lr': True},   # constant lr
    ]
    opt = torch.optim.SGD(param_groups, init_lr,
                          momentum=momentum, weight_decay=weight_decay)
    return opt, init_lr


def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs):
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
    for g in optimizer.param_groups:
        g['lr'] = init_lr if g.get('fix_lr', False) else cur_lr   # predictor: no decay


def train_one_epoch(loader, model, optimizer, init_lr, epoch, total_epochs):
    adjust_learning_rate(optimizer, init_lr, epoch, total_epochs)
    model.train()
    for images, _ in loader:                  # images = [view1, view2]
        p1, p2, z1, z2 = model(images[0], images[1])
        loss = simsiam_loss(p1, p2, z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
