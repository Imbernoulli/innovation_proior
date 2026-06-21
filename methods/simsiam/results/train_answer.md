We want to learn visual representations from unlabeled images with the simplest possible Siamese recipe: draw two random augmentations of one image, encode both views with a weight-sharing network, and make the two outputs agree. The whole difficulty lives in one degenerate fixed point. Pure agreement is maximized by the constant map — if every image collapses to the same vector, the two views match perfectly while the representation carries no information at all. The existing systems each buy their way out of collapse in a different currency, and each currency comes with infrastructure. Contrastive learning pays with negatives: it pulls the two views of an image together and pushes views of different images apart, which makes the constant solution impossible but also makes the number and quality of negatives central, so it drags in very large batches (SimCLR at 4096) or a momentum-maintained queue (MoCo). Clustering-based methods pay with assignments and a balance constraint: DeepCluster and SwAV make views agree through cluster codes but force the codes to spread across prototypes, where the Sinkhorn balanced-assignment machinery is the real anti-collapse device. Momentum-target methods like BYOL pay with a second network: an online branch predicts a target branch whose weights are a slow moving average of the online weights. The question I want to settle is which of these components is actually intrinsic to avoiding collapse, and whether a plain shared-weight Siamese network can learn nontrivial features if only the right minimal asymmetry is present.

The BYOL recipe is the suspicious one, because the momentum target conflates two distinct properties: the target weights are a moving average, *and* the target branch is not trained by the current loss gradient. Those are tied together in the usual implementation, so to find out which one matters I strip away the moving average entirely and keep a single shared encoder, then ask whether the target-side gradient rule alone is enough. This is the method I propose, and I call it SimSiam — a simple Siamese network. I take two augmentations $x_1, x_2$ of an image. A shared encoder $f$, a backbone followed by a 3-layer projection MLP, produces $z_1 = f(x_1)$ and $z_2 = f(x_2)$, and a 2-layer bottleneck predictor $h$ produces $p_1 = h(z_1)$ and $p_2 = h(z_2)$. The per-direction loss compares a prediction from one view to the projection of the other view by negative cosine similarity,
$$D(p,z) = -\frac{p}{\|p\|_2}\cdot\frac{z}{\|z\|_2}.$$
The sign matters: minimizing $D$ maximizes cosine similarity and bottoms out at $-1$. This is not an arbitrary choice of metric — the squared distance between the two normalized vectors is $\|\hat p - \hat z\|_2^2 = 2 - 2\,\hat p\cdot\hat z = 2 + 2D(p,z)$, so negative cosine and normalized mean-squared error differ only by a positive scale and an additive constant and define exactly the same minimizers. The training loss is symmetrized over the two directions, and the one indispensable operation is placed only on the target side:
$$L = \tfrac{1}{2}D\big(p_1,\operatorname{sg}(z_2)\big) + \tfrac{1}{2}D\big(p_2,\operatorname{sg}(z_1)\big),$$
where $\operatorname{sg}$ is stop-gradient. This does not freeze either view globally. In the first term the encoder on $x_2$ receives no gradient through $z_2$, but in the second term it receives gradient through $p_2$, and symmetrically for $x_1$; each view serves as a constant target once and as a predicted branch once.

This makes the experiment clean and the result sharp. With stop-gradient in place, the loss descends without diving to $-1$, the kNN monitor improves, and the per-channel standard deviation of the l2-normalized output stays near the isotropic reference $1/\sqrt{d}$. Remove only stop-gradient, with architecture and hyperparameters otherwise identical, and the loss races to $-1$, the output std collapses to zero, and linear evaluation is at chance — whereas keeping it yields about $67.7\%$ ImageNet top-1 at 100 epochs. So the constant solution genuinely exists for this structure, and the target-side gradient rule is the switch that flips between collapse and useful learning. I check that no hidden architectural component is secretly responsible. The predictor $h$ is required: drop it and, for the symmetric stop-gradient loss, the gradient points in the same direction as the gradient of the plain agreement loss $D(z_1,z_2)$ scaled by $1/2$ — the stop-gradient becomes algebraically vacuous because one term supplies the $z_1$ side and the other the $z_2$ side of the ordinary gradient, and collapse follows. The predictor must also be *trained*: freezing it at random initialization leaves the loss high and non-converging, a failure distinct from the collapse signature, which is why the canonical recipe holds the predictor learning rate constant so it keeps adapting to the moving representation. Batch normalization helps optimization and accuracy but, since both arms of the stop-gradient comparison share the same BN configuration, it is not the causal collapse switch. The similarity function is not the explanation either — a symmetrized cross-entropy similarity also trains without collapse — nor is symmetrization itself, since a one-direction loss already avoids collapse and symmetrization mainly densifies the estimate of the augmentation expectation.

What makes it work, beyond the code trick, is that the target-side constant is exactly what alternating optimization produces, so SimSiam is best read as an EM-like procedure rather than ordinary gradient descent on a single loss. Introduce a second block of variables, one free target vector $\eta_x$ per image, and consider
$$\mathcal L(\theta,\eta) = \mathbb E_{x,T}\big[\|F_\theta(T(x)) - \eta_x\|_2^2\big],$$
where $F_\theta$ is the encoder-like network, $T$ is the augmentation distribution, and $\eta_x$ is not a network output but an optimization variable indexed by the image — the analogue of a k-means assignment or a per-sample latent code. Solve it by alternating $\theta^t \leftarrow \arg\min_\theta \mathcal L(\theta,\eta^{t-1})$ and $\eta^t \leftarrow \arg\min_\eta \mathcal L(\theta^t,\eta)$. In the $\theta$ subproblem $\eta^{t-1}$ is held fixed, so no gradient flows into it — the stop-gradient is not a heuristic but the direct consequence of optimizing one block while the other is frozen. The $\eta$ subproblem separates by image, and for MSE its minimizer is the augmentation mean $\eta_x^t = \mathbb E_T[F_{\theta^t}(T(x))]$. That expectation is impractical to compute, so I approximate it with a single sampled augmentation $T'$, giving $\eta_x^t \approx F_{\theta^t}(T'(x))$ — which is precisely the other Siamese view. Substituting back yields $\theta^{t+1} \leftarrow \arg\min_\theta \mathbb E_{x,T}\|F_\theta(T(x)) - F_{\theta^t}(T'(x))\|_2^2$, where the second term is evaluated at the previous parameters and is constant for the current subproblem; the two augmentations $T$ and $T'$ are the two views, and reducing this subproblem by one SGD step instead of solving it fully gives the plain shared-weight Siamese update with the target projection treated as a constant. This view also explains the predictor: the one-sample approximation discards the augmentation expectation, and a regression predictor trained to map one view's projection toward the other has optimal form $h^*(z_1) = \mathbb E[z_2\mid z_1]$, a conditional mean that corresponds to the per-image augmentation average $\mathbb E_T[f(T(x))]$ the one-sample $\eta$ update ignores — so $h$ is a learned stand-in for the missing expectation, and indeed approximating that expectation explicitly with a moving-average target lets one drop the predictor and still get a nontrivial (though much worse) representation. I am careful about what this buys: the $\eta$ objective still admits the constant solution, so the derivation does not prove collapse is impossible. The honest claim is that the alternating trajectory differs — the per-image targets start from a randomly initialized, non-constant network, they update per image rather than as a joint gradient pulling all targets together, and empirically the optimizer follows the scattered-output path. SimSiam closes as a minimal Siamese recipe — one shared encoder, a projection MLP, a trained predictor MLP, negative cosine between normalized prediction and target projection, and the target held constant per term — which is BYOL without the moving-average encoder, SimCLR without negatives, and SwAV without clustering, and in controlled experiments this small asymmetric update is enough to learn useful representations without any of that extra machinery.

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
