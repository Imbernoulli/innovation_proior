# VICReg: Variance-Invariance-Covariance Regularization

We want a representation for images learned with no labels, by the joint-embedding recipe: take two augmented views of each image, push their embeddings together, and keep the encoder. The part that is uncontroversial — two views of the same image should land near each other — is also the part that, alone, is useless. A constant encoder that maps every input to the same vector makes every pair of views agree perfectly and drives the agreement loss to zero without learning anything. So the agreement term needs a partner whose job is to make constant and low-information embeddings expensive, and the real question is what that partner should be. The existing answers all supply the anti-collapse force from somewhere outside the loss itself. Contrastive methods like SimCLR and MoCo push other images apart, which makes a constant solution bad, but they estimate that repulsion from many negative pairs and so lean on very large batches or a memory queue. Clustering methods like DeepCluster and SwAV avoid pairwise negatives but trade them for an assignment solver and a balancing step (SwAV's optimal transport) that does the anti-collapse work. BYOL and SimSiam remove negatives entirely and just regress one view onto the other, but their scalar similarity loss still admits the constant solution; collapse is held off by architectural asymmetry — a moving-average target network, a predictor, a stop-gradient — rather than by anything the loss forbids. Barlow Twins and W-MSE come closest to what I want because they talk about redundancy and whitening, but Barlow's statistic is a cross-branch correlation matrix and W-MSE applies a whitening operator. I want the loss itself to name what is forbidden, as an ordinary statistic of one embedding batch, with no negatives, no queue, no target network, no stop-gradient, and no cluster balancing.

I propose VICReg, an objective built from three terms — invariance, variance, and covariance — each a plain statistic of the embedding batches, that together rule out both ways the representation can collapse. For two embedding batches $X,Y\in\mathbb{R}^{B\times d}$ (one per view), the invariance term is just the mean squared distance between paired embeddings,

$$s(X,Y)=\frac{1}{B}\sum_{b=1}^{B}\|x_b-y_b\|_2^2,$$

which ties the learned statistics to image content: it demands that whatever the embedding encodes be stable across the two sampled augmentations. The first anti-collapse term targets the most literal forbidden event, zero spread. The trivially collapsed solution is the one where every coordinate column $X_{:,j}$ has zero variance across the batch, so I require each coordinate to keep at least a fixed standard deviation through a hinge,

$$v(X)=\frac{1}{d}\sum_{j=1}^{d}\max\!\left(0,\ \gamma-\sqrt{\operatorname{Var}(X_{:,j})+\epsilon}\right),$$

with $\gamma=1$ and $\epsilon=10^{-4}$. The hinge, not a raw maximization, is deliberate: a coordinate above the floor stops receiving pressure, so the term controls scale without inflating it without bound, and with $\gamma=1$ the loss itself fixes the embedding scale, which is why I do not l2-normalize. The choice of standard deviation rather than variance inside the hinge is the load-bearing detail. Writing $\delta_k=x_k-\bar x$ and using the unbiased sample variance $\operatorname{Var}(x)=(n-1)^{-1}\sum_k\delta_k^2$, the gradient of the active hinge $\gamma-\sqrt{\operatorname{Var}(x)+\epsilon}$ with respect to one sample is

$$\frac{\partial}{\partial x_k}\left(\gamma-\sqrt{\operatorname{Var}(x)+\epsilon}\right)=-\frac{\delta_k}{(n-1)\sqrt{\operatorname{Var}(x)+\epsilon}},$$

so gradient descent moves above-mean samples farther up and below-mean samples farther down, amplifying spread. Had I used a variance hinge $\gamma-\operatorname{Var}(x)$, its derivative $-2\delta_k/(n-1)$ vanishes linearly as the column approaches its mean — exactly when I most need a restoring force. I should be honest about the edge case: at an exactly constant column every $\delta_k=0$, so the embedding-gradient is zero there too; the term does not magically repel a perfectly constant batch. What it does is make the constant batch carry positive loss (so it is no longer a zero-loss optimum) and give near-collapse deviation patterns a far stronger restoring gradient than a variance hinge ever would.

A spread floor alone leaves a second, subtler failure: every coordinate can have standard deviation one while all coordinates carry the same scalar signal, so the embedding is not constant but is informationally collapsed into a low-dimensional subspace. Catching this requires a statistic *between* coordinates, the covariance matrix

$$C(X)=\frac{1}{B-1}\sum_{b=1}^{B}(x_b-\bar x)(x_b-\bar x)^\top,\qquad c(X)=\frac{1}{d}\sum_{i\ne j}C(X)_{ij}^2.$$

Penalizing only the off-diagonal entries drives redundant linear co-variation between distinct coordinates toward zero, forcing the variance the spread term guarantees to occupy many coordinates rather than one repeated direction. These two terms are genuinely complementary and neither substitutes for the other: the covariance term alone is perfectly content with a constant batch, where every off-diagonal covariance is zero, and the variance term alone cannot see two coordinates copying each other. The full objective is

$$\ell(X,Y)=\lambda\,s(X,Y)+\mu\,[v(X)+v(Y)]+\nu\,[c(X)+c(Y)],$$

with the published ImageNet coefficients $\lambda=\mu=25$ and $\nu=1$. One more design choice keeps the loss honest: the final embeddings fed to the loss are left unnormalized. L2-normalizing would pin coordinate standard deviations near the unit-sphere scale $1/\sqrt{d}$ and fight the explicit floor; batch-standardizing the final embeddings would turn the covariance penalty into a correlation penalty and hide the very scale the variance term exists to control. BatchNorm inside the hidden projector layers is fine and helps optimization, but the loss must see the raw final embeddings. One caveat carried into the code: the equations above and the implementation below use slightly different normalizing constants — the code's `F.mse_loss` is an elementwise mean over $B\cdot d$ entries, and the two branch variance losses are averaged with a `/ 2` — and the tuned coefficients `sim_coeff=25`, `std_coeff=25`, `cov_coeff=1` are calibrated to that implementation, so the code is the form to trust. The result is neither a contrastive loss, nor a whitening transform, nor a dynamics-only trick: it is an agreement loss whose two explicit batch-statistic regularizers make zero spread and redundant coordinates costly.

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
