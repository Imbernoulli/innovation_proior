# PointNet

## Core Method

A point cloud is processed as an unordered set. The network applies the same pointwise MLP $h$ to every point and aggregates the resulting per-point features with an elementwise symmetric max:

$$
f(\{x_1,\ldots,x_n\}) \approx \gamma\left(\max_{i=1}^n h(x_i)\right).
$$

The max is taken coordinatewise, so reordering the input points cannot change the global descriptor. The practical classifier is:

$$
\text{input }(B,N,3)
\rightarrow T_1\in\mathbb{R}^{3\times3}
\rightarrow \mathrm{MLP}(64,64)
\rightarrow T_2\in\mathbb{R}^{64\times64}
\rightarrow \mathrm{MLP}(64,128,1024)
\rightarrow \max_N
\rightarrow \mathrm{FC}(512,256,k).
$$

The segmentation network keeps the transformed $64$-dimensional local feature for each point, concatenates the broadcast $1024$-dimensional global feature to it, and applies shared layers to produce per-point scores.

## Theoretical Guarantees

For continuous set functions on compact point sets under Hausdorff distance, the shared-MLP plus max-pool plus post-MLP form can approximate the function arbitrarily well with enough pooled coordinates. The constructive proof is an occupancy-grid argument: snap points to a fine grid, encode occupied cells with pointwise features and a symmetric max, then map the occupancy code to the target function.

The continuous-soft-indicator step is an approximation, not literal binary recovery. With hard indicators the occupancy vector is exact; with continuous features, the post-map must tolerate neighborhoods of the finite grid codes, and the error budget covers that approximation.

For a trained network with

$$
u(S)=\max_{x\in S} h(x)\in\mathbb{R}^K,\qquad f=\gamma\circ u,
$$

there is a critical point set $\mathcal C_S$ of size at most $K$: choose one argmax point for each coordinate of $u(S)$. This set alone reproduces the max-pooled vector. Any added point $x$ with $h(x)\le u(S)$ elementwise also leaves the output unchanged, giving the upper-bound set $\mathcal N_S$ and the sandwich property:

$$
\mathcal C_S\subseteq T\subseteq \mathcal N_S \implies f(T)=f(S).
$$

The feature transform is regularized toward orthogonality:

$$
L_{\mathrm{reg}}=\|I-AA^\top\|_F^2.
$$

The released TensorFlow code implements the penalty as `tf.nn.l2_loss(A A^T - I)`, i.e. one half of the summed squared entries, multiplied by `0.001`.

## Reference-Faithful PyTorch Artifact

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    def __init__(self, sizes, activate_last=True):
        super().__init__()
        blocks = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            layers = [nn.Conv1d(a, b, 1)]
            if activate_last or not last:
                layers += [nn.BatchNorm1d(b), nn.ReLU()]
            blocks.append(nn.Sequential(*layers))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv = SharedMLP([k, 64, 128, 1024])
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)
        nn.init.zeros_(self.fc3.weight)
        with torch.no_grad():
            self.fc3.bias.copy_(torch.eye(k).reshape(-1))

    def forward(self, x):  # x: (B, k, N)
        B = x.size(0)
        x = self.conv(x).max(dim=2).values
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x).view(B, self.k, self.k)


class PointNet(nn.Module):
    def __init__(self, num_classes=40, seg=False, m=None):
        super().__init__()
        self.seg = seg
        self.input_tnet = TNet(3)
        self.feature_tnet = TNet(64)
        self.mlp1 = SharedMLP([3, 64, 64])
        self.mlp2 = SharedMLP([64, 64, 128, 1024])
        if seg:
            self.seg_mlp = SharedMLP([1024 + 64, 512, 256, 128, 128, m],
                                     activate_last=False)
        else:
            self.fc1 = nn.Linear(1024, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.drop = nn.Dropout(p=0.3)  # TensorFlow keep_prob=0.7
            self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):  # x: (B, 3, N)
        N = x.size(2)
        T = self.input_tnet(x)
        x = torch.bmm(T.transpose(1, 2), x)
        x = self.mlp1(x)
        A = self.feature_tnet(x)
        x = torch.bmm(A.transpose(1, 2), x)
        local = x
        g = self.mlp2(x).max(dim=2).values
        if self.seg:
            g = g.unsqueeze(2).repeat(1, 1, N)
            return self.seg_mlp(torch.cat([local, g], dim=1)), A
        g = self.drop(F.relu(self.bn1(self.fc1(g))))
        g = self.drop(F.relu(self.bn2(self.fc2(g))))
        return self.fc3(g), A


def pointnet_loss(pred, label, A, reg_weight=0.001):
    cls = F.cross_entropy(pred, label)
    I = torch.eye(A.size(1), device=A.device, dtype=A.dtype).unsqueeze(0)
    diff = torch.bmm(A, A.transpose(1, 2)) - I
    reg = 0.5 * diff.pow(2).sum()
    return cls + reg_weight * reg
```

Faithfulness notes: the T-Net final layer starts as identity by zero-initialized weights plus identity bias; classification uses dropout after both hidden FC layers; the basic segmentation head has two `128` shared layers before the score layer; and the regularizer keeps the TensorFlow `l2_loss` half-sum constant.
