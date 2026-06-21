I want to do deep learning directly on 3D geometry, and the thing that bothers me is that nobody touches the points: to use a convolutional network you first regularize the cloud into something with a lattice — a $32^3$ occupancy grid, or a dozen rendered views — and only then run a CNN. Both fight the data. A voxel grid of a thin chair is almost all empty, memory and compute scale cubically with resolution, and whatever resolution I pick quantizes away fine geometry; multi-view rendering discards the explicit 3D structure and forces me to hand-pick viewpoints. The reason for all this contortion is that convolution needs a regular grid to share weights over. So the real question is whether I can build a network that eats raw points — just a set of $(x,y,z)$ coordinates — and never regularizes them at all. To do that I have to respect what a point cloud actually is, namely a set, and a set imposes three demands: it is unordered, so feeding the same $n$ points in any of $n!$ orders must give the same answer; its points live in a metric space, so nearby points form local structure I would like to exploit (this matters most for segmentation); and it is a geometric object, so rotating or translating the whole cloud must not change the prediction.

The unordered property is what breaks everything I know, so I pressure-test the three ways to handle it. Sorting the points into a canonical order and then running an ordinary MLP only sounds like it dissolves the problem: a usable order would have to be *stable* under perturbation, i.e. a bijection from a high-dimensional point space onto the real line that keeps spatially close points close in value — and you cannot flatten a high-dimensional neighborhood onto a line without tearing near pairs apart, so no perturbation-stable order exists and the ordering issue never goes away. Treating the set as a sequence and training a recurrent net on randomly permuted orderings fails too: order genuinely affects what a sequence model learns and cannot be fully trained away, and whatever reordering tolerance an RNN has is for a few dozen elements, not the thousands of points in a cloud. That leaves the clean route — do not fight the ordering, aggregate with a function that is *intrinsically* order-independent. A symmetric function returns the same value under any permutation of its arguments; sum, product, and the element-wise maximum are all symmetric. If the aggregation step is symmetric, the whole network is permutation-invariant by construction, with no sorting and no augmentation.

The method I propose is PointNet, and its backbone is the form
$$f(\{x_1,\ldots,x_n\}) \approx \gamma\!\left(\max_{i=1}^{n} h(x_i)\right),$$
where $h:\mathbb{R}^N\to\mathbb{R}^K$ is a shared per-point MLP applied independently to every point, the $\max$ is taken coordinatewise over the $n$ points, and $\gamma$ is a small network on the pooled descriptor. Read it literally: $h$ lifts each raw point into a $K$-dimensional feature, the max asks "what is the strongest response of feature $k$ anywhere in the set?", and with $K$ such features the network is learning $K$ different things to notice about the shape. Because the pool is symmetric, the global descriptor is order-invariant; that is the entire mechanism that makes the architecture legal on sets. I implement $h$ as stacked $1\times1$ convolutions over the point axis (the same small MLP run on every point), batch-normed and ReLU'd, and $\gamma$ as a couple of fully connected layers.

Before building on top of this I want to know it is not crippled by the single max-pool bottleneck, so I prove the form can approximate any set function continuous with respect to Hausdorff distance $d_H$ (small set-perturbations giving small output changes — exactly the robustness a classifier should have). The proof is an occupancy-grid argument: fix $\epsilon$, get the continuity $\delta_\epsilon$, chop the domain into cells of width $\le\delta_\epsilon$ and snap each point to its cell so the snapped set $\tilde S$ has $d_H(S,\tilde S)\le\delta_\epsilon$ and hence $|f(S)-f(\tilde S)|<\epsilon$. Give each cell an occupancy feature; max-pooling those features over the points yields a code that depends only on *which* cells are occupied, not on point order, and a continuous post-map $\gamma$ can match $f$ on those finitely many grid codes. The one subtlety is that hard cell indicators are discontinuous; with continuous soft bumps such as $h_k(x)=e^{-d(x,I_k)}$ an empty cell does not pool to exactly zero, so I do not pretend the pooled vector is binary — I pick continuous features and a $\gamma$ that is stable on neighborhoods of the grid codes and spend part of the error budget on that approximation. The payoff is concrete: in the worst case the net is *allowed* to behave like a voxel-occupancy encoder, but it stores no dense cubic grid, and gradient descent can find a smarter probing than uniform cells.

That same factorization gives the robustness story, which is the other reason this architecture is the right one. Write $u(S)=\max_{x\in S}h(x)\in\mathbb{R}^K$ with $f=\gamma\circ u$. For each output coordinate $j$ the max is attained by at least one argmax point $x_j$; collecting these gives a *critical set* $\mathcal C_S$ of at most $K$ points, and keeping only them reproduces every coordinate's maximum, so $f(\mathcal C_S)=f(S)$. Deleting any non-argmax point leaves the maxes untouched, and *adding* any point $x$ with $h(x)\le u(S)$ in every coordinate also leaves the element-wise max unchanged; collecting the addable points into an upper set $\mathcal N_S$ gives the sandwich
$$\mathcal C_S \subseteq T \subseteq \mathcal N_S \implies f(T)=f(S).$$
So the output is decided by at most $K$ key points and is invariant to dropping non-critical points and to inserting sub-threshold noise — that is the robustness to missing data and outliers — and it tells me $K$, the max-pool width, is not a free knob but the *bottleneck dimension* governing how finely shapes can be discriminated, which is why I push it to $1024$.

The global descriptor is perfect for classification but throws away which point is which, and segmentation needs a label per point that depends on both the point's local neighborhood and the whole object. So I put locality back the simplest way that works: after max-pooling, broadcast the $1024$-dimensional global feature back onto every point's own $64$-dimensional local feature, concatenate, and run another shared per-point MLP to emit per-point scores. Each point now sees its local feature *and* the global shape context. For the third demand, rigid/affine invariance, I borrow the spatial-transformer idea of letting the network canonicalize its own input — but here applying an affine transform to a point cloud is *just multiplying the coordinates by a matrix*, with no resampling and no aliasing. I add a mini-network (the same shared-MLP, max-pool, FC backbone) called a T-net that regresses a $3\times3$ matrix from the cloud and matrix-multiplies the input by it, initialized to the identity so it starts from "no transform." I apply the same idea in feature space with a second T-net predicting a $64\times64$ transform on the per-point features before pooling. But a $64\times64$ matrix is over four thousand parameters and is unstable to regress freely — it can drift into ill-conditioned maps that destroy information — so I want it close to a rotation, i.e. orthogonal, and softly enforce that with
$$L_{\mathrm{reg}}=\|I-AA^\top\|_F^2$$
added to the loss with a small weight ($0.001$). An orthogonal $A$ has $AA^\top=I$, preserves norms and angles, and is invertible without loss; with this penalty the feature alignment helps, without it the high-dimensional transform tends to hurt. The $3\times3$ input transform needs no such constraint — at that size the regression is well-behaved on its own.

Assembled, the classifier takes the $n$ points, aligns coordinates with the $3\times3$ T-net, lifts each point to $64$ features with a shared MLP, aligns features with the orthogonality-regularized $64\times64$ T-net, lifts through $64,128,1024$ with another shared MLP, max-pools to a $1024$-dim global descriptor, and runs fully connected $512\to256\to k$ with batch norm, ReLU, and dropout (keep-probability $0.7$) on the hidden layers. Segmentation branches off by concatenating the broadcast global feature onto each point's transformed $64$-dim local feature and applying shared layers $512\to256\to128\to128\to m$. The training loss is softmax cross-entropy plus the orthogonality regularizer on the feature transform. One implementation detail that matters: to start exactly at "no transform," each T-net's final affine layer is zero-initialized in weight with an identity bias, rather than adding identity on top of a random matrix.

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
