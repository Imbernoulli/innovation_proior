# SimSiam — Simple Siamese Representation Learning

## Problem

Learn visual representations from unlabeled images with a Siamese "two augmented views should agree" objective, without relying on explicit negative pairs, online clustering, a momentum encoder, or large batches.

## Method

Draw two augmentations \(x_1,x_2\) of image \(x\). A shared encoder \(f\), consisting of a backbone plus a 3-layer projection MLP, produces \(z_1=f(x_1)\) and \(z_2=f(x_2)\). A 2-layer bottleneck predictor \(h\) produces \(p_1=h(z_1)\) and \(p_2=h(z_2)\).

The per-direction loss is negative cosine similarity:

\[
D(p,z)=-{p\over \|p\|_2}\cdot {z\over \|z\|_2}.
\]

The squared distance between normalized vectors is \(2+2D(p,z)\), so the cosine form is equivalent to normalized squared error up to a positive scale and additive constant. The symmetrized training loss is

\[
L={1\over2}D(p_1,\operatorname{sg}(z_2))+
  {1\over2}D(p_2,\operatorname{sg}(z_1)),
\]

where \(\operatorname{sg}\) is stop-gradient. The minimum possible loss is \(-1\). In a controlled ablation, removing only stop-gradient sends the loss to \(-1\), collapses the l2-normalized output std to zero, and gives chance ImageNet linear evaluation; keeping it yields non-collapsed outputs and about 67.7% top-1 at 100 epochs.

## Why It Works

My hypothesis is an EM-like alternating optimization. Introduce one free target vector per image:

\[
\mathcal L(\theta,\eta)=
\mathbb E_{x,T}\left[\|F_\theta(T(x))-\eta_x\|_2^2\right].
\]

Alternate between

\[
\theta^t\leftarrow\arg\min_\theta \mathcal L(\theta,\eta^{t-1})
\]

and

\[
\eta^t\leftarrow\arg\min_\eta \mathcal L(\theta^t,\eta).
\]

When solving for \(\theta\), \(\eta^{t-1}\) is fixed, so the target branch receives no gradient. For MSE, the \(\eta\) update is

\[
\eta_x^t=\mathbb E_T[F_{\theta^t}(T(x))].
\]

Approximating this expectation with one sampled augmentation \(T'\) gives the other Siamese view:

\[
\eta_x^t\approx F_{\theta^t}(T'(x)).
\]

Taking one SGD step on the resulting \(\theta\) subproblem yields the SimSiam update. The predictor \(h\) is interpreted as a learned approximation to the missing augmentation expectation: as a regression map, its optimum is a conditional mean \(h^*(z_1)=\mathbb E[z_2\mid z_1]\), which relates to \(\mathbb E_T[f(T(x))]\) for the same image. Symmetrization is denser sampling of the augmentation expectation.

This is a hypothesis about the optimization being followed, not a proof that collapse is impossible. The constant solution still exists; non-collapse is an empirical observation explained by the altered alternating trajectory.

## Canonical Code

This is the core of the archived FAIR PyTorch implementation, with distributed training boilerplate omitted.

```python
import math
import torch
import torch.nn as nn
import torchvision.models as models


class SimSiam(nn.Module):
    """Shared encoder f and predictor h."""

    def __init__(self, base_encoder=models.resnet50, dim=2048, pred_dim=512):
        super().__init__()

        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        prev_dim = self.encoder.fc.weight.shape[1]

        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False),
        )
        self.encoder.fc[6].bias.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim),
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


criterion = nn.CosineSimilarity(dim=1)


def simsiam_loss(p1, p2, z1, z2):
    return -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5


def build_optimizer(model, base_lr=0.05, batch_size=512,
                    momentum=0.9, weight_decay=1e-4):
    init_lr = base_lr * batch_size / 256
    param_groups = [
        {"params": model.encoder.parameters(), "fix_lr": False},
        {"params": model.predictor.parameters(), "fix_lr": True},
    ]
    optimizer = torch.optim.SGD(
        param_groups,
        init_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    return optimizer, init_lr


def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs):
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
    for group in optimizer.param_groups:
        group["lr"] = init_lr if group.get("fix_lr", False) else cur_lr


def train_one_epoch(loader, model, optimizer, init_lr, epoch, total_epochs):
    adjust_learning_rate(optimizer, init_lr, epoch, total_epochs)
    model.train()
    for images, _ in loader:
        p1, p2, z1, z2 = model(images[0], images[1])
        loss = simsiam_loss(p1, p2, z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Load-Bearing Details

- **Stop-gradient:** the controlled collapse switch in the reported baseline; target projections are constants within each loss term.
- **Predictor:** required for the default method. Without it, the symmetric stop-gradient loss has the same gradient direction as the no-stop-gradient agreement loss, scaled by \(1/2\).
- **Predictor learning rate:** kept constant in the canonical recipe because the predictor must keep adapting to the moving representation.
- **Projection head:** 3-layer MLP; BN after every fully connected layer including the output; no ReLU on the output; output BN has no affine.
- **Predictor head:** 2-layer bottleneck MLP; BN and ReLU only on the hidden layer; output layer has bias and no BN/ReLU.
- **Optimizer:** SGD, base lr \(0.05\times\mathrm{BatchSize}/256\), momentum 0.9, weight decay \(10^{-4}\), cosine decay on the encoder, non-decayed predictor lr.
