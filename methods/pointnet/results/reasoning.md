OK, let me think this through from scratch. I want to do deep learning directly on 3D geometry — point clouds. The thing that nags at me is how everyone currently does it: they don't touch the points, they first convert the cloud into something a convolutional net likes. Voxelize it into a $32^3$ occupancy grid, or render it from a dozen viewpoints into images, then run a CNN. Both work, but both feel like fighting the data. A voxel grid of a thin chair is almost entirely empty, and the memory and compute scale cubically with resolution, so I'm paying enormous cost to represent mostly nothing, and at whatever resolution I pick I've quantized away fine geometry. Multi-view rendering throws away the explicit 3D structure and makes me choose viewpoints. The reason for all this contortion is that convolution needs a regular lattice to share weights over. So the real question is: can I build a net that eats the raw points — just a set of $(x,y,z)$ — and never regularizes them at all?

If I'm going to consume points directly, I have to respect what a point cloud actually is, and that's a *set*. Let me list what "set" forces on me, because each constraint is going to dictate a piece of the architecture. First, it's unordered: there's no canonical first point, so if I feed the same $n$ points in a different order I must get the same answer — invariance to all $n!$ permutations. Second, the points aren't floating in a void; they live in a metric space, so nearby points form meaningful local structure, and ideally I'd exploit that. Third, it's a geometric object: if I rotate or translate the whole cloud, a chair is still a chair and its seat is still its seat, so the representation should be invariant to those rigid (and more broadly affine) motions. The unordered property is the one that breaks everything I know how to do, so let me start there.

How do I make a network invariant to the order of its inputs? I can think of three strategies and I want to actually pressure-test them, not just pick one.

Strategy one: sort the points into some canonical order first, then treat them as an ordered vector and run an ordinary MLP or 1-D CNN. This *sounds* like it dissolves the problem. But let me ask whether a good sort even exists. I'd need an ordering of points in $\mathbb{R}^3$ (or higher, once I've lifted them to features) that is *stable* — a tiny perturbation of a point shouldn't reshuffle the whole order, because then the network's input would jump discontinuously and it could never learn a consistent mapping. Suppose such a stable ordering existed. An ordering is a map that assigns each high-dimensional point a position on the real line — a bijection from a high-dimensional space to $\mathbb{R}^1$. For it to be *stable* under perturbations, it would have to send points that are close in $\mathbb{R}^d$ to values that are close on the line, i.e. preserve spatial proximity while collapsing dimension. That's exactly the thing that can't be done in general — you can't flatten a high-dimensional neighborhood structure onto a line without tearing some near pairs apart. So no perturbation-stable canonical order exists, the ordering issue doesn't actually go away, and I'd expect an MLP on sorted points to do only marginally better than on unsorted points. Strike one.

Strategy two: treat the set as a sequence and train a recurrent net, feeding the points in random orders during training so it learns to ignore order. The trouble is twofold. There's prior evidence — Vinyals et al.'s *Order Matters* (2015), which studied feeding sets to sequence models — that order genuinely affects what an RNN learns and can't be fully trained away. And even setting that aside, an RNN's tolerance to reordering holds for short sequences of a few dozen elements, but a point cloud has *thousands* of points. There's no reason to expect a recurrent net to behave invariantly across thousands of permuted steps, and it'd be slow besides. Strike two.

Strategy three: don't fight the ordering at all — aggregate the points with a function that is *intrinsically* order-independent. A symmetric function takes $n$ vectors and returns a vector that doesn't depend on their order. Plain addition and multiplication are symmetric; so is taking the element-wise maximum. If the aggregation step is symmetric, the whole network is permutation-invariant by construction — no sorting, no augmentation, no hoping. Unlike the first two, this strategy doesn't have a known failure mode lurking in it: I'm not asking a discontinuous ordering to exist, and I'm not asking a recurrent net to forget order across thousands of steps. So this is the route left standing. Now I need to turn "use a symmetric function" into an actual architecture.

Here's a form worth committing to. I want to approximate a general function on a point set as
$$f(\{x_1,\dots,x_n\})\approx g\big(h(x_1),\dots,h(x_n)\big),$$
where $h:\mathbb{R}^N\to\mathbb{R}^K$ is applied to *each point independently* and $g:(\mathbb{R}^K)^n\to\mathbb{R}$ is symmetric. Read it: $h$ lifts each raw point into a $K$-dimensional feature, and $g$ symmetrically pools those features into a set-level answer. For $h$ I'll use a multilayer perceptron with shared weights — literally the same small MLP run on every point — and for $g$ I'll use the simplest powerful symmetric pooling I have, the element-wise max over the $n$ points, followed by another network $\gamma$ that maps the pooled vector to the output. So $g=\gamma\circ\mathrm{MAX}$. Each of the $K$ coordinates of $h$ is its own scalar function of a point; max-pooling that coordinate over the set asks "what is the strongest response of feature $k$ across all points?" With $K$ such features I'm learning $K$ different things to notice about the set. That's the entire backbone: shared per-point MLP, then max-pool, then a small net on the pooled descriptor.

Before I build more on top of this, I want to know whether it's expressive enough to be worth it, or whether I've crippled the model by insisting on a single max-pool bottleneck. Let me try to prove that this form — shared $h$, then $\mathrm{MAX}$, then continuous $\gamma$ — can approximate *any* continuous set function arbitrarily well. "Continuous" here means continuous with respect to Hausdorff distance $d_H$ between sets: small set-perturbations give small changes in $f$, which is exactly the robustness I want a classifier to have.

Let me do it in one dimension first; the higher-dimensional version is the same idea with product cells. Take $\mathcal X=\{S\subseteq[0,1]:|S|=n\}$ and $f$ continuous w.r.t. $d_H$. Fix $\epsilon>0$. By continuity there's a $\delta_\epsilon$ such that $d_H(S,S')<\delta_\epsilon\Rightarrow|f(S)-f(S')|<\epsilon$. Now discretize: let $K=\lceil 1/\delta_\epsilon\rceil$, which chops $[0,1]$ into $K$ equal intervals, and define $\sigma(x)=\lfloor Kx\rfloor/K$, with the endpoint $x=1$ handled by the final cell. Let $\tilde S=\{\sigma(x):x\in S\}$. Every point moves by at most one interval width $1/K\le\delta_\epsilon$, so $d_H(S,\tilde S)\le\delta_\epsilon$; if I want the strict inequality in the continuity condition, I just choose a slightly finer grid. Then $|f(S)-f(\tilde S)|<\epsilon$. So if I can make the network recover enough information about the snapped set $\tilde S$, I'm within the continuity budget of $f$ on $S$.

The idealized recovery is simple. For each interval, imagine an occupancy feature that says whether some point of $S$ falls in that interval. Max-pooling those interval features over the points gives a code whose $j$-th coordinate depends only on whether interval $j$ is occupied, not on the order of the points. A map $\tau$ can send that occupancy code back to the set of occupied left endpoints, so $\tau(\mathrm{MAX}_i h(x_i))=\tilde S$ in the hard-indicator version, and a following continuous $\gamma$ can match $f\circ\tau$ on those finitely many grid codes. The only subtlety is that hard indicators are discontinuous. If I instead use continuous soft interval functions such as $h_k(x)=e^{-d(x,I_k)}$, absent intervals do not become literal zeros; they take values below $1$. So I should not pretend the soft max-pooled vector is exactly in $\{0,1\}^K$. What I need is weaker and enough: choose continuous bump features and a continuous $\gamma$ that is stable on neighborhoods of the finite grid codes, and spend part of the error budget on this continuous approximation. Then
$$\left|\gamma\left(\mathrm{MAX}_{x_i\in S}h(x_i)\right)-f(S)\right|<\epsilon$$
for sufficiently many pooled coordinates. I'll trust the snapping half of this — the $|f(S)-f(\tilde S)|<\epsilon$ step is just the definition of continuity — and flag the soft-indicator half as the part I'm waving my hands at; it's a real approximation that eats some of the budget, not an identity. What the argument does establish cleanly is something concrete about *capacity*: in the worst case the net is allowed to behave like a voxel-occupancy encoder, partitioning space into cells. In practice gradient descent can find a smarter probing than uniform voxels, but the useful floor is "at least an occupancy-style summary," with no dense cubic grid stored by the network.

That same factorization raises a sharper question I should chase, because the answer is the model's robustness story. Look at the pooled vector $\mathbf u(S)=\mathrm{MAX}_{x\in S}\,h(x)\in\mathbb{R}^K$, with $f=\gamma\circ\mathbf u$. How much of the input set actually *determines* the output? It feels like it should be only a few points, not all $n$ — but I want to pin the number down rather than assert it. Since $f$ factors through $\mathbf u$, I only need to understand what fixes $\mathbf u$. For each of the $K$ output coordinates $j$, the max $\mathbf u_j(S)=\max_i h_j(x_i)$ is achieved by *at least one* point — call it $x_j$, the argmax for that coordinate. Collect these argmaxes, $\mathcal C_S=\{x_1,\dots,x_K\}$ (at most $K$ of them, possibly fewer since one point can win several coordinates). Then $\mathbf u(\mathcal C_S)=\mathbf u(S)$ — keeping just those points reproduces every coordinate's maximum — so $f(\mathcal C_S)=f(S)$. And now the two-sided robustness. On the one hand I can *delete* any non-argmax point and the maxes are untouched. On the other hand I can *add* any point $x$ whose features lie below the current maxima, $h(x)\le\mathbf u(S)$ in every coordinate, and again the element-wise max is unchanged. Let $\mathcal N_S$ be $\mathcal C_S$ together with all such addable points. Then for *any* set $T$ sandwiched between them, $\mathcal C_S\subseteq T\subseteq\mathcal N_S$, we get $\mathbf u(T)=\mathbf u(S)$, hence $f(T)=f(S)$.

I don't want to take that chain on faith, so let me run a tiny case by hand. Use $K=2$ features, $h(x)=\big(x,\;-(x-2)^2\big)$, and the set $S=\{0,1,2,3\}$ on the line. The four feature rows are $(0,-4),(1,-1),(2,0),(3,-1)$, so the column maxima are $\mathbf u(S)=(3,\,0)$: coordinate $1$ is won by $x=3$ and coordinate $2$ by $x=2$. The critical set is therefore $\mathcal C_S=\{2,3\}$ — two of the four points, exactly $K$ — and pooling just those two reproduces $(3,0)$. Now test the sandwich. Add $x=1.9$: its features are $(1.9,-0.01)$, both strictly below $(3,0)$, and indeed $\mathbf u(S\cup\{1.9\})=(3,0)$ is unchanged. Add instead $x=5$: its features are $(5,-9)$, and the first coordinate $5>3$ exceeds the running max, so $\mathbf u$ jumps to $(5,0)$ — the output *does* move, exactly as the condition $h(x)\le\mathbf u(S)$ predicts it should. The hand-computation matches the claim on both the invariant additions and the disruptive one, which is the check I wanted. So the output is invariant to throwing away non-critical points and to inserting noise points whose features don't exceed the maxima, and it's pinned down by a *critical set* of at most $K=$ the max-pool dimension. That's a strong reason to expect robustness to missing data and outliers — the shape is summarized by a sparse set of key points, and I'd guess (but haven't shown) those key points trace out the skeleton of the object. It also says the max-pool width $K$ isn't a free knob: it's the *bottleneck dimension*, the number of distinct features I can notice, so it should directly govern how finely the model can discriminate shapes — too small and I can't cover the space, so I'd expect accuracy to climb as I widen $K$ and then saturate.

The permutation-invariance backbone is settled. Now the second property: locality, which matters specifically for segmentation. The max-pool gives me one global descriptor $[f_1,\dots,f_K]$ for the whole cloud. For classification that's perfect — feed it to a couple of fully connected layers and predict the class. But segmentation needs a label *per point*, and a label like "this point is on the chair's leg vs. seat" depends on both where the point sits locally *and* what the whole object is. The pure global descriptor has thrown away which point is which. So I need to put local information back in. The simplest effective move: after computing the global feature, concatenate it onto *each* point's own per-point feature, then run another shared per-point MLP on the combined vector. Now every point's prediction sees its own local feature *and* the global shape context, and I can output per-point scores. I can't train anything here to know it carries enough local geometry, so I'll hold this as a hypothesis with a concrete test attached: ask the segmentation network to regress each point's *surface normal*, which is determined purely by a point's immediate neighborhood. If the per-point features after concatenation are rich enough to predict normals, that's evidence the representation really encodes local structure rather than only global shape; if normal regression fails, the concatenation is too coarse and I'd need a finer local feature. That's the experiment I'd run before believing this branch.

Now the third property: invariance to rigid/affine motion. If I rotate the whole cloud, the answer shouldn't change. I could try to bake this in with augmentation, but there's a cleaner idea borrowed from spatial transformer networks: let the network *canonicalize* its own input. A small sub-network looks at the cloud and predicts a transformation that aligns it to some canonical pose, and I apply that transform before the main feature extraction. For images, spatial transformers needed a special sampling-and-interpolation layer, and resampling introduces aliasing. But here's the thing about points — applying an affine transform to a point cloud is *just multiplying the coordinates by a matrix*. No resampling, no interpolation, no aliasing, no new layer type. So I add a mini-network — built from the same parts as the main net, a shared per-point MLP, a max-pool, and a couple of fully connected layers — that regresses an affine matrix, and I matrix-multiply the input points by it. I'll initialize it to predict the identity, so the network starts from "no transform" and learns to deviate. Call it the alignment net, or T-net.

The same idea should help in *feature* space, not just on raw coordinates: insert a second alignment net after the first feature layer that predicts a transform on the per-point feature vectors, aligning features from different clouds into a common frame before pooling. But now I have to be careful, and this is where a naive copy of the input version breaks. The input transform is a $3\times3$ matrix — tiny, easy to regress. The feature transform is, say, $64\times64$ — that's over four thousand parameters predicted by a sub-network, and an unconstrained matrix that big makes optimization unstable; the regressor can wander into degenerate or ill-conditioned transforms that destroy information. I want the transform to be close to a rotation — something that realigns without losing or distorting information. The clean way to encode "information-preserving linear map" is *orthogonality*: an orthogonal matrix has $AA^\top=I$, preserves norms and angles, and is invertible without loss. So I'll softly enforce it by adding a regularizer to the loss that penalizes departure from orthogonality,
$$L_{\text{reg}}=\big\|I-AA^\top\big\|_F^2,$$
where $A$ is the predicted feature-transform matrix and $\|\cdot\|_F$ is the Frobenius norm. Before relying on it, let me check it actually rewards what I think it rewards by plugging in a few matrices. A true rotation should cost nothing: for the $2\times2$ rotation by $0.7$ rad, $AA^\top-I$ comes out to machine zero and $L_{\text{reg}}\approx 0$. Identity: $0$. A uniform scaling $A=2I_3$ is the kind of "wandering" I want to discourage — then $AA^\top-I=3I_3$, with squared Frobenius norm $9\cdot 3=27$, so the half-summed penalty (matching the TensorFlow `l2_loss`) is $13.5$; large, as it should be. The dangerous case is a *rank-deficient* transform that collapses a feature dimension, e.g. $A=\mathrm{diag}(1,1,0)$ in $3$D: here $AA^\top-I=\mathrm{diag}(0,0,-1)$, penalty $0.5$ — nonzero, so the regularizer does push back against exactly the information-destroying maps I'm worried about, though only mildly. So the term is well-aimed: zero on the rotations I want, positive and growing on scalings and collapses I don't. With a small weight on it I'd expect the feature alignment to stay near-orthogonal and help rather than hurt; that expectation is what I'd confirm by ablating the term in training. I don't put this constraint on the $3\times3$ input transform — at that size the regression is well-behaved on its own.

Let me assemble the whole classification pipeline and then say how segmentation branches off. Take the $n$ input points, $(x,y,z)$ each. Run the input alignment net, get a $3\times3$ matrix, multiply the coordinates by it. Apply a shared MLP to lift each point to $64$ features. Run the feature alignment net, get a $64\times64$ matrix (orthogonality-regularized), apply it to the per-point features. Apply another shared MLP raising each point through $64$, $128$, then $1024$ features. Max-pool over the $n$ points to get a single $1024$-dim global descriptor. For classification, pass that through fully connected layers $512\to256\to k$, with batch norm and ReLU on the hidden layers and keep-probability $0.7$ dropout after each hidden fully connected layer. For basic segmentation, take the $1024$-dim global feature, concatenate it onto each point's $64$-dim local feature, and run shared layers $512\to256\to128\to128\to m$ for per-point scores. The training loss is the softmax cross-entropy for the labels plus the orthogonality regularizer on the feature transform, with a small weight.

Now I can write the implementation. The shared per-point MLP is a stack of $1\times1$ convolutions over the point axis, batch-normed and ReLU'd except at score layers; the max-pool is a reduction over the point axis; the T-nets are mini versions of the same backbone. To start exactly at "no transform," I should initialize the final affine layer of each T-net with zero weights and an identity bias, rather than adding identity on top of a randomly initialized transform.

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

    def forward(self, x):                                # x: (B, k, N)
        B = x.size(0)
        x = self.conv(x).max(dim=2).values               # symmetric max-pool over points
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x).view(B, self.k, self.k)

class PointSetNet(nn.Module):
    def __init__(self, num_classes=40, seg=False, m=None):
        super().__init__()
        self.seg = seg
        self.input_tnet   = TNet(3)                      # 3x3 input alignment
        self.feature_tnet = TNet(64)                     # 64x64 feature alignment
        self.mlp1 = SharedMLP([3, 64, 64])               # h up to 64
        self.mlp2 = SharedMLP([64, 64, 128, 1024])       # h up to 1024
        if seg:
            self.seg_mlp = SharedMLP([1024 + 64, 512, 256, 128, 128, m],
                                     activate_last=False)
        else:
            self.fc1 = nn.Linear(1024, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.drop = nn.Dropout(p=0.3)                # TensorFlow keep_prob=0.7
            self.fc3 = nn.Linear(256, num_classes)

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
        g = self.drop(F.relu(self.bn1(self.fc1(g))))
        g = self.drop(F.relu(self.bn2(self.fc2(g))))
        return self.fc3(g), A                                  # class scores

def pointnet_loss(pred, label, A, reg_weight=0.001):
    cls = F.cross_entropy(pred, label)
    # Mirrors TensorFlow tf.nn.l2_loss: one half of the sum of squares.
    I = torch.eye(A.size(1), device=A.device, dtype=A.dtype).unsqueeze(0)
    diff = torch.bmm(A, A.transpose(1, 2)) - I
    reg = 0.5 * diff.pow(2).sum()
    return cls + reg_weight * reg
```
