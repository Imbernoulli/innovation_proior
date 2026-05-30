# PointNet

## Problem

Learn directly from raw 3D point clouds — unordered sets of $(x,y,z)$ points —
for a unified set of tasks (object classification, part segmentation, scene
parsing), avoiding the voxelization or multi-view rendering that other approaches
use to make the data CNN-friendly (both of which are voluminous and introduce
quantization artifacts). The network must respect three properties of point sets:
**permutation invariance** (any of $n!$ orderings gives the same answer),
**locality** (nearby points form meaningful structure), and **transformation
invariance** (rigid/affine motion of the whole cloud must not change the output).

## Key idea

**Symmetric aggregation for order invariance.** Approximate a set function as
$$f(\{x_1,\dots,x_n\})\approx \gamma\Big(\underset{x_i}{\mathrm{MAX}}\;h(x_i)\Big),$$
where $h$ is a shared per-point MLP (the same MLP applied independently to every
point) and $\mathrm{MAX}$ is element-wise max-pooling over points — a symmetric
function, so the whole network is permutation-invariant by construction. Sorting
fails (no perturbation-stable high-dimensional ordering exists) and RNNs over the
sequence fail (order cannot be trained away and they do not scale to thousands of
points).

**Universal approximation (Theorem 1).** For any continuous set function $f$ on
$\mathcal X=\{S\subseteq[0,1]^m,|S|=n\}$ (continuous w.r.t. Hausdorff distance) and
any $\epsilon>0$, there exist a continuous $h$ and a symmetric $g=\gamma\circ\mathrm{MAX}$
with $|f(S)-\gamma(\mathrm{MAX}_{x\in S}h(x))|<\epsilon$, given enough max-pool width
$K$. The constructive proof snaps points to a $K$-cell grid (so the worst case is a
voxel-occupancy encoder), shows the occupancy vector is the max-pool of soft interval
indicators, and lets $\gamma$ map occupancy back to $f$.

**Critical points and robustness (Theorem 2).** Let $\mathbf u(S)=\mathrm{MAX}_{x\in S}h(x)\in\mathbb R^K$,
$f=\gamma\circ\mathbf u$. Then there exist a critical set $\mathcal C_S$ and an
upper-bound set $\mathcal N_S$ such that $f(T)=f(S)$ for any $\mathcal C_S\subseteq T\subseteq\mathcal N_S$,
and $|\mathcal C_S|\le K$. Proof: each output coordinate's max is attained by some
argmax point; their union (at most $K$ points) reproduces $\mathbf u$, hence $f$;
deleting non-argmax points or adding points with $h(x)\le\mathbf u(S)$ leaves the
element-wise max unchanged. So the output is determined by $\le K$ key points (the
object's "skeleton") — explaining robustness to missing points and outliers. $K$ is
the **bottleneck dimension** and governs expressiveness.

**Local + global for segmentation.** The max-pool gives one global descriptor (enough
for classification). For per-point labels, concatenate the global descriptor onto each
point's local feature and run another shared MLP — so each point's prediction sees both
its local geometry and the global shape context.

**Joint alignment (T-Net) for transformation invariance.** A mini-network (itself a
shared-MLP + max-pool + FC PointNet) predicts an affine transform applied to the data
by matrix multiplication — no resampling/aliasing, unlike image spatial transformers.
An **input T-Net** outputs a $3\times3$ matrix (init identity) applied to coordinates;
a **feature T-Net** outputs a $64\times64$ matrix (init identity) applied to point
features. The high-dimensional feature transform is regularized toward orthogonality,
$$L_{\text{reg}}=\big\|I-AA^\top\big\|_F^2,$$
because an orthogonal map preserves information and stabilizes optimization (the
$3\times3$ input transform needs no such constraint).

## Architecture (classification)

input $(B,N,3)$ → input T-Net $3\times3$, matmul → shared MLP$(64,64)$ → feature
T-Net $64\times64$, matmul → shared MLP$(64,128,1024)$ → **max-pool over $N$** →
global $1024$ → FC$(512)$, FC$(256$, dropout$)$, FC$(k)$. BatchNorm + ReLU on hidden
layers. Segmentation concatenates the global $1024$ to each point's $64$-dim local
feature → shared MLP → $m$ per-point scores.

Loss: softmax cross-entropy $+\;0.001\cdot\|I-AA^\top\|_F^2$. Adam, lr $0.001$,
momentum $0.9$, batch $32$, lr halved every $20$ epochs; BN decay $0.5\to0.99$;
dropout keep $0.7$ on the last hidden layer; $\sim$1024 input points.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1),    nn.BatchNorm1d(64),   nn.ReLU(),
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k))
    def forward(self, x):                              # (B, k, N)
        B = x.size(0)
        x = self.conv(x).max(dim=2)[0]                 # symmetric max-pool
        x = self.fc(x).view(B, self.k, self.k)
        return x + torch.eye(self.k, device=x.device).unsqueeze(0)   # init identity

class PointNet(nn.Module):
    def __init__(self, num_classes=40, seg=False, m=None):
        super().__init__()
        self.seg = seg
        self.input_tnet = TNet(3); self.feature_tnet = TNet(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),  nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(),
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        if seg:
            self.seg_mlp = nn.Sequential(
                nn.Conv1d(1024 + 64, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Conv1d(512, 256, 1),       nn.BatchNorm1d(256), nn.ReLU(),
                nn.Conv1d(256, 128, 1),       nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, m, 1))
        else:
            self.cls = nn.Sequential(
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(256, num_classes))
    def forward(self, x):                              # (B, 3, N)
        N = x.size(2)
        x = torch.bmm(self.input_tnet(x).transpose(1, 2), x)
        x = self.mlp1(x)
        A = self.feature_tnet(x)
        x = torch.bmm(A.transpose(1, 2), x)
        local = x
        g = self.mlp2(x).max(dim=2)[0]                 # max-pool -> global
        if self.seg:
            g = g.unsqueeze(2).repeat(1, 1, N)
            return self.seg_mlp(torch.cat([local, g], 1)), A
        return self.cls(g), A

def pointnet_loss(pred, label, A, reg_weight=0.001):
    cls = F.cross_entropy(pred, label)
    I = torch.eye(A.size(1), device=A.device).unsqueeze(0)
    reg = ((I - torch.bmm(A, A.transpose(1, 2))) ** 2).sum(dim=[1, 2]).mean()
    return cls + reg_weight * reg
```
