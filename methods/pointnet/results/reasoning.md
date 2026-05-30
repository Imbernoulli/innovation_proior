OK, let me think this through from scratch. I want to do deep learning directly on 3D geometry — point clouds. The thing that nags at me is how everyone currently does it: they don't touch the points, they first convert the cloud into something a convolutional net likes. Voxelize it into a $32^3$ occupancy grid, or render it from a dozen viewpoints into images, then run a CNN. Both work, but both feel like fighting the data. A voxel grid of a thin chair is almost entirely empty, and the memory and compute scale cubically with resolution, so I'm paying enormous cost to represent mostly nothing, and at whatever resolution I pick I've quantized away fine geometry. Multi-view rendering throws away the explicit 3D structure and makes me choose viewpoints. The reason for all this contortion is that convolution needs a regular lattice to share weights over. So the real question is: can I build a net that eats the raw points — just a set of $(x,y,z)$ — and never regularizes them at all?

If I'm going to consume points directly, I have to respect what a point cloud actually is, and that's a *set*. Let me list what "set" forces on me, because each constraint is going to dictate a piece of the architecture. First, it's unordered: there's no canonical first point, so if I feed the same $n$ points in a different order I must get the same answer — invariance to all $n!$ permutations. Second, the points aren't floating in a void; they live in a metric space, so nearby points form meaningful local structure, and ideally I'd exploit that. Third, it's a geometric object: if I rotate or translate the whole cloud, a chair is still a chair and its seat is still its seat, so the representation should be invariant to those rigid (and more broadly affine) motions. The unordered property is the one that breaks everything I know how to do, so let me start there.

How do I make a network invariant to the order of its inputs? I can think of three strategies and I want to actually pressure-test them, not just pick one.

Strategy one: sort the points into some canonical order first, then treat them as an ordered vector and run an ordinary MLP or 1-D CNN. This *sounds* like it dissolves the problem. But let me ask whether a good sort even exists. I'd need an ordering of points in $\mathbb{R}^3$ (or higher, once I've lifted them to features) that is *stable* — a tiny perturbation of a point shouldn't reshuffle the whole order, because then the network's input would jump discontinuously and it could never learn a consistent mapping. Suppose such a stable ordering existed. An ordering is a map that assigns each high-dimensional point a position on the real line — a bijection from a high-dimensional space to $\mathbb{R}^1$. For it to be *stable* under perturbations, it would have to send points that are close in $\mathbb{R}^d$ to values that are close on the line, i.e. preserve spatial proximity while collapsing dimension. That's exactly the thing that can't be done in general — you can't flatten a high-dimensional neighborhood structure onto a line without tearing some near pairs apart. So no perturbation-stable canonical order exists, the ordering issue doesn't actually go away, and I'd expect an MLP on sorted points to do only marginally better than on unsorted points. Strike one.

Strategy two: treat the set as a sequence and train a recurrent net, feeding the points in random orders during training so it learns to ignore order. The trouble is twofold. There's prior evidence — Vinyals et al.'s *Order Matters* (2015), which studied feeding sets to sequence models — that order genuinely affects what an RNN learns and can't be fully trained away. And even setting that aside, an RNN's tolerance to reordering holds for short sequences of a few dozen elements, but a point cloud has *thousands* of points. There's no reason to expect a recurrent net to behave invariantly across thousands of permuted steps, and it'd be slow besides. Strike two.

Strategy three: don't fight the ordering at all — aggregate the points with a function that is *intrinsically* order-independent. A symmetric function takes $n$ vectors and returns a vector that doesn't depend on their order. Plain addition and multiplication are symmetric; so is taking the element-wise maximum. If the aggregation step is symmetric, the whole network is permutation-invariant by construction — no sorting, no augmentation, no hoping. This is the one. Now I need to turn "use a symmetric function" into an actual architecture.

Here's the form I'll commit to. I want to approximate a general function on a point set as
$$f(\{x_1,\dots,x_n\})\approx g\big(h(x_1),\dots,h(x_n)\big),$$
where $h:\mathbb{R}^N\to\mathbb{R}^K$ is applied to *each point independently* and $g:(\mathbb{R}^K)^n\to\mathbb{R}$ is symmetric. Read it: $h$ lifts each raw point into a $K$-dimensional feature, and $g$ symmetrically pools those features into a set-level answer. For $h$ I'll use a multilayer perceptron with shared weights — literally the same small MLP run on every point — and for $g$ I'll use the simplest powerful symmetric pooling I have, the element-wise max over the $n$ points, followed by another network $\gamma$ that maps the pooled vector to the output. So $g=\gamma\circ\mathrm{MAX}$. Each of the $K$ coordinates of $h$ is its own scalar function of a point; max-pooling that coordinate over the set asks "what is the strongest response of feature $k$ across all points?" With $K$ such features I'm learning $K$ different things to notice about the set. That's the entire backbone: shared per-point MLP, then max-pool, then a small net on the pooled descriptor.

Before I build more on top of this, I want to know whether it's expressive enough to be worth it, or whether I've crippled the model by insisting on a single max-pool bottleneck. Let me try to prove that this form — shared $h$, then $\mathrm{MAX}$, then continuous $\gamma$ — can approximate *any* continuous set function arbitrarily well. "Continuous" here means continuous with respect to Hausdorff distance $d_H$ between sets: small set-perturbations give small changes in $f$, which is exactly the robustness I want a classifier to have.

Let me do it in one dimension first; it'll generalize coordinate-wise. Take $\mathcal X=\{S\subseteq[0,1]:|S|=n\}$ and $f$ continuous w.r.t. $d_H$. Fix $\epsilon>0$. By continuity there's a $\delta_\epsilon$ such that $d_H(S,S')<\delta_\epsilon\Rightarrow|f(S)-f(S')|<\epsilon$. Now discretize: let $K=\lceil 1/\delta_\epsilon\rceil$, which chops $[0,1]$ into $K$ equal intervals, and define $\sigma(x)=\lfloor Kx\rfloor/K$, snapping each point to the left end of its interval. Let $\tilde S=\{\sigma(x):x\in S\}$. Every point moves by less than one interval width $1/K\le\delta_\epsilon$, so $d_H(S,\tilde S)<\delta_\epsilon$, hence $|f(S)-f(\tilde S)|<\epsilon$. So if I can reconstruct $f$ on the *snapped* set $\tilde S$, I'm within $\epsilon$ of $f$ on $S$.

Now I need to recover $\tilde S$ from a max-pool of per-point features. Define a soft indicator per interval, $h_k(x)=e^{-d(x,[\,\tfrac{k-1}{K},\tfrac{k}{K}\,])}$, where $d$ is point-to-interval distance — it's near $1$ when $x$ is in interval $k$ and decays outside it. Stack them: $\mathbf h(x)=[h_1(x);\dots;h_K(x)]\in\mathbb{R}^K$. Max-pool each coordinate over the set: $v_j=\max_i h_j(x_i)$, and $\mathbf v=\mathrm{MAX}(\mathbf h(x_1),\dots,\mathbf h(x_n))$. The $j$-th coordinate of $\mathbf v$ lights up exactly when *some* point occupies interval $j$ — so $\mathbf v$ is (essentially) the occupancy vector in $\{0,1\}^K$, and it's symmetric in the points by construction. The occupancy vector is in bijection with the snapped set: define $\tau(\mathbf v)=\{(k-1)/K:v_k\ge 1\}$, the set of left-ends of occupied intervals, and then $\tau(\mathbf v(x_1,\dots,x_n))\equiv\tilde S$. Finally let $\gamma:\mathbb{R}^K\to\mathbb{R}$ be a continuous function that agrees with $f\circ\tau$ on occupancy vectors, $\gamma(\mathbf v)=f(\tau(\mathbf v))$. Chaining it all together,
$$\big|\gamma(\mathbf v(x_1,\dots,x_n))-f(S)\big|=\big|f(\tau(\mathbf v))-f(S)\big|=|f(\tilde S)-f(S)|<\epsilon,$$
and $\gamma(\mathbf v)=\gamma(\mathrm{MAX}(\mathbf h(x_1),\dots,\mathbf h(x_n)))=(\gamma\circ\mathrm{MAX})(\mathbf h(x_1),\dots,\mathbf h(x_n))$, with $\gamma\circ\mathrm{MAX}$ symmetric. So the architecture *can* approximate any continuous set function, given enough max-pool width $K$. The proof is constructive and tells me something concrete: in the worst case the net is allowed to behave like a voxel-occupancy encoder, partitioning space into $K$ cells. In practice gradient descent will find a smarter probing than uniform voxels — but it's reassuring that the floor is "at least as good as voxelizing," with none of the cubic cost, because I only ever store $K$ numbers, not $K^3$.

That same proof immediately raises a sharper question I should chase, because the answer is the model's robustness story. Look at the pooled vector $\mathbf u(S)=\mathrm{MAX}_{x\in S}\,h(x)\in\mathbb{R}^K$, with $f=\gamma\circ\mathbf u$. How much of the input set actually *determines* the output? Claim: $f(S)$ is decided by at most $K$ points. Here's why. Since $f$ factors through $\mathbf u$, I only need to understand what fixes $\mathbf u$. For each of the $K$ output coordinates $j$, the max $\mathbf u_j(S)=\max_i h_j(x_i)$ is achieved by *at least one* point — call it $x_j$, the argmax for that coordinate. Collect these argmaxes, $\mathcal C_S=\{x_1,\dots,x_K\}$ (at most $K$ of them, possibly fewer since one point can win several coordinates). Then $\mathbf u(\mathcal C_S)=\mathbf u(S)$ — keeping just those points reproduces every coordinate's maximum — so $f(\mathcal C_S)=f(S)$. And now the two-sided robustness. On the one hand I can *delete* any non-argmax point and the maxes are untouched. On the other hand I can *add* any point $x$ whose features lie below the current maxima, $h(x)\le\mathbf u(S)$ in every coordinate, and again the element-wise max is unchanged. Let $\mathcal N_S$ be $\mathcal C_S$ together with all such addable points. Then for *any* set $T$ sandwiched between them, $\mathcal C_S\subseteq T\subseteq\mathcal N_S$, we get $\mathbf u(T)=\mathbf u(S)$, hence $f(T)=f(S)$. So the output is invariant to throwing away non-critical points and to inserting noise points (as long as their features don't exceed the maxima), and it's pinned down by a *critical set* $\mathcal C_S$ of at most $K=$ the max-pool dimension. That's why this thing should be remarkably robust to missing data and outliers — it summarizes a shape by a sparse set of key points, and I'd bet those key points trace out the skeleton of the object. And it tells me the max-pool width $K$ isn't a free knob: it's the *bottleneck dimension*, the number of distinct features I can notice, so it should directly govern how finely the model can discriminate shapes — too small and I can't cover the space, so I'd expect accuracy to climb as I widen $K$ and then saturate.

Good — the permutation-invariance backbone is settled and I understand why it works. Now the second property: locality, which matters specifically for segmentation. The max-pool gives me one global descriptor $[f_1,\dots,f_K]$ for the whole cloud. For classification that's perfect — feed it to a couple of fully connected layers and predict the class. But segmentation needs a label *per point*, and a label like "this point is on the chair's leg vs. seat" depends on both where the point sits locally *and* what the whole object is. The pure global descriptor has thrown away which point is which. So I need to put local information back in. The simplest effective move: after computing the global feature, concatenate it onto *each* point's own per-point feature, then run another shared per-point MLP on the combined vector. Now every point's prediction sees its own local feature *and* the global shape context, and I can output per-point scores. I can sanity-check that this really captures local geometry by asking the segmentation network to regress per-point *normals* — a normal is determined purely by a point's neighborhood — and if it can do that, the local-plus-global concatenation genuinely carries neighborhood information.

Now the third property: invariance to rigid/affine motion. If I rotate the whole cloud, the answer shouldn't change. I could try to bake this in with augmentation, but there's a cleaner idea borrowed from spatial transformer networks: let the network *canonicalize* its own input. A small sub-network looks at the cloud and predicts a transformation that aligns it to some canonical pose, and I apply that transform before the main feature extraction. For images, spatial transformers needed a special sampling-and-interpolation layer, and resampling introduces aliasing. But here's the thing about points — applying an affine transform to a point cloud is *just multiplying the coordinates by a matrix*. No resampling, no interpolation, no aliasing, no new layer type. So I add a mini-network — built from the same parts as the main net, a shared per-point MLP, a max-pool, and a couple of fully connected layers — that regresses an affine matrix, and I matrix-multiply the input points by it. I'll initialize it to predict the identity, so the network starts from "no transform" and learns to deviate. Call it the alignment net, or T-net.

The same idea should help in *feature* space, not just on raw coordinates: insert a second alignment net after the first feature layer that predicts a transform on the per-point feature vectors, aligning features from different clouds into a common frame before pooling. But now I have to be careful, and this is where a naive copy of the input version breaks. The input transform is a $3\times3$ matrix — tiny, easy to regress. The feature transform is, say, $64\times64$ — that's over four thousand parameters predicted by a sub-network, and an unconstrained matrix that big makes optimization unstable; the regressor can wander into degenerate or ill-conditioned transforms that destroy information. I want the transform to be close to a rotation — something that realigns without losing or distorting information. The clean way to encode "information-preserving linear map" is *orthogonality*: an orthogonal matrix has $AA^\top=I$, preserves norms and angles, and is invertible without loss. So I'll softly enforce it by adding a regularizer to the loss that penalizes departure from orthogonality,
$$L_{\text{reg}}=\big\|I-AA^\top\big\|_F^2,$$
where $A$ is the predicted feature-transform matrix and $\|\cdot\|_F$ is the Frobenius norm. With a small weight on this term the feature alignment becomes stable and helps; without it the high-dimensional transform tends to hurt. I don't put this constraint on the $3\times3$ input transform — at that size the regression is well-behaved on its own.

Let me assemble the whole classification pipeline and then say how segmentation branches off. Take the $n$ input points, $(x,y,z)$ each. Run the input alignment net, get a $3\times3$ matrix, multiply the coordinates by it. Apply a shared MLP to lift each point to $64$ features. Run the feature alignment net, get a $64\times64$ matrix (orthogonality-regularized), apply it to the per-point features. Apply another shared MLP raising each point to $128$ then $1024$ features. Max-pool over the $n$ points to get a single $1024$-dim global descriptor. For classification, pass that through fully connected layers $512\to256\to k$, with batch norm and ReLU on the hidden layers and dropout on the last hidden layer for regularization. For segmentation, take the $1024$-dim global feature, concatenate it onto each point's $64$-dim local feature, and run a shared MLP down to $m$ per-point scores. The training loss is the softmax cross-entropy for the labels plus the orthogonality regularizer on the feature transform, with a small weight.

Now to real code, grounded in a standard implementation. The shared per-point MLP is most naturally a stack of $1\times1$ convolutions over the point axis (the convolution shares one small MLP across all $n$ points, which is precisely $h$), batch-normed and ReLU'd; the max-pool is a reduction over the point axis; the T-nets are mini versions of the same backbone whose final linear layer is biased to output the identity matrix.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """Alignment net (T-net): regress a kxk transform, initialized to identity.
       Built from the same parts as the main net: shared per-point MLP, max-pool,
       fully connected layers."""
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(                       # shared per-point MLP h
            nn.Conv1d(k, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k))                       # -> flattened transform
    def forward(self, x):                                # x: (B, k, N)
        B = x.size(0)
        x = self.conv(x).max(dim=2)[0]                   # symmetric max-pool over points
        x = self.fc(x).view(B, self.k, self.k)
        # bias toward identity so training starts from "no transform"
        return x + torch.eye(self.k, device=x.device).unsqueeze(0)

class PointNet(nn.Module):
    def __init__(self, num_classes=40, seg=False, m=None):
        super().__init__()
        self.seg = seg
        self.input_tnet   = TNet(3)                      # 3x3 input alignment
        self.feature_tnet = TNet(64)                     # 64x64 feature alignment
        self.mlp1 = nn.Sequential(                       # h up to 64
            nn.Conv1d(3, 64, 1),  nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(                       # h up to 1024
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(),
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        if seg:
            self.seg_mlp = nn.Sequential(                # combined local+global -> per-point
                nn.Conv1d(1024 + 64, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Conv1d(512, 256, 1),       nn.BatchNorm1d(256), nn.ReLU(),
                nn.Conv1d(256, 128, 1),       nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, m, 1))
        else:
            self.cls = nn.Sequential(
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(),
                nn.Dropout(0.3),                         # dropout on last hidden layer
                nn.Linear(256, num_classes))

    def forward(self, x):                                # x: (B, 3, N)
        N = x.size(2)
        x = torch.bmm(self.input_tnet(x).transpose(1, 2), x)   # align input coords
        x = self.mlp1(x)                                       # per-point local feats (B,64,N)
        A = self.feature_tnet(x)                               # 64x64 feature transform
        x = torch.bmm(A.transpose(1, 2), x)                    # align features
        local = x                                              # keep local feats for seg
        g = self.mlp2(x).max(dim=2)[0]                         # MAX-POOL -> global (B,1024)
        if self.seg:
            g_exp = g.unsqueeze(2).repeat(1, 1, N)             # broadcast global to each point
            return self.seg_mlp(torch.cat([local, g_exp], 1)), A   # per-point scores
        return self.cls(g), A                                  # class scores

def pointnet_loss(pred, label, A, reg_weight=0.001):
    cls = F.cross_entropy(pred, label)
    # orthogonality regularizer on the feature transform: keep A near orthogonal
    I = torch.eye(A.size(1), device=A.device).unsqueeze(0)
    reg = ((I - torch.bmm(A, A.transpose(1, 2))) ** 2).sum(dim=[1, 2]).mean()
    return cls + reg_weight * reg
```

So the causal chain, start to finish: I refuse to voxelize or render, so I must consume the raw set — and a set is unordered, which kills sorting (no perturbation-stable order exists) and kills sequence models (order can't be trained away, doesn't scale to thousands of points), leaving symmetric aggregation as the only clean route. That fixes the form $f\approx\gamma\circ\mathrm{MAX}\circ h$: a shared per-point MLP followed by max-pooling, which I prove can approximate any continuous set function and whose output is pinned down by at most $K$ critical points — giving both the bottleneck-dimension knob and the robustness to missing/extra points. Segmentation then needs locality, so I concatenate the global descriptor back onto each per-point feature; and rigid/affine invariance comes from letting small alignment nets canonicalize first the coordinates and then the features, the latter kept stable by an orthogonality penalty because a $64\times64$ transform is otherwise too unconstrained to optimize.
