OK, let me think this through from scratch. I already have a network that consumes an unordered point set and is permutation-invariant: encode each point with a shared MLP $h$, take the element-wise max over all points, then map that pooled vector through another MLP $\gamma$,
$$f(x_1,\dots,x_n)=\gamma\Big(\underset{i}{\mathrm{MAX}}\,\{h(x_i)\}\Big).$$
It's a universal approximator of continuous set functions and it's robust to corruption. So why am I not done? Let me stare at what that single $\mathrm{MAX}$ actually does. Every point is encoded *independently* — $h(x_i)$ never looks at $x_i$'s neighbors — and then *one* pooling collapses the entire cloud into a single global vector. There is no stage anywhere in between that forms a *local* feature: nothing that says "these few points near here form a corner," "this little patch is a flat surface." The model jumps in one step from per-point encodings straight to a global summary. For a single clean object that's survivable, but for fine-grained patterns and for cluttered scenes I suspect this is a real handicap — there's no local context to generalize from. I can't fully test that suspicion here, but I can at least name what's structurally missing: an intermediate representation between "one point" and "the whole cloud."

Contrast that with why convolutional nets work so well on grids. A CNN never tries to summarize the whole image in one shot. It builds a *hierarchy*: low layers have small receptive fields and detect tiny local patterns, then pooling/striding coarsens the grid, and higher layers see larger receptive fields built out of the lower-level local features. That multi-resolution abstraction of *local* patterns is the ingredient my single-pool model lacks. So a natural thing to try: get that same hierarchy for point sets. If I apply my set encoder *recursively* — first to small local neighborhoods to get local features, then group those into larger units and encode again, and so on up to the whole cloud — I'd have the same low-to-high abstraction, only over a metric set instead of a grid. Whether that actually buys accuracy I won't know until it's trained, but it's the most direct way to import the property a CNN has and I don't.

To build a hierarchy I have to answer two questions, and they turn out to be coupled. First, how do I *partition* the cloud into local regions at each level? Second, how do I *abstract* the points (or features) in a region into a single feature? The coupling is the important part: the partition has to produce regions with *common structure* across the cloud, because I want to apply *one* shared local-feature learner to all of them — that's the point-set version of convolutional weight sharing, and without it I'd need a different network per region, which is hopeless. So the partition and the learner have to be designed together.

The learner is the easy choice. I already have a set encoder that's a universal set approximator, handles a variable number of points, is permutation-invariant, and is robust — so I'll use a mini version of it as the per-region feature learner. A region is just a small point set; my encoder is exactly the tool for turning a small point set into a feature vector. So one level of the hierarchy — call it a *set abstraction* level — takes a point set of $N$ points and produces a smaller set of $N'$ points, each carrying a feature that summarizes its local neighborhood. It naturally factors into three steps: pick the region centers (sampling), gather each center's neighborhood (grouping), and encode each neighborhood (the mini set encoder).

Start with sampling — how to choose the $N'$ centroids. I could sample uniformly at random, but for the same number of centroids I'd like the best *coverage* of the cloud so my regions tile it well. Farthest point sampling does this: start from a random point, then repeatedly add the point that is farthest (in the metric) from the set already chosen. The result spreads centroids out to cover the whole set. And notice a property I like, which is the opposite of how a CNN works: a CNN scans the grid at fixed strides, *agnostic* to where the data actually is, wasting capacity on empty regions. Farthest point sampling places centroids *based on the data itself*, so the receptive fields follow the geometry. Data-dependent sampling, for free.

Now grouping — given a centroid, which points are its neighborhood? On a grid this is "pixels within $k$ index-steps," but here the neighborhood has to be defined by the *metric*. Two options. I could take the $k$ nearest neighbors of the centroid — a fixed *count* of points. Or I could take a ball query: all points within a fixed *radius* $r$ (capped at some maximum $K$ for implementation). Let me think about which property I want. With $k$-NN, the physical *size* of the region floats with density — in a dense area the $k$ nearest points are all crammed close, in a sparse area they sprawl far out, so "the local feature" means a different physical scale in different places. With a ball query the region has a fixed *scale* $r$ everywhere, so a local pattern learned at radius $r$ in one place means the same physical thing in another place. For recognizing local geometric patterns — and especially for per-point semantic labeling — I want that scale to be consistent across the cloud, so the learned local feature generalizes spatially. So I'll go with the ball query.

The one thing that nags me about the ball is that the number of points inside it varies from region to region, while tensor code wants a fixed neighbor count $K$. The standard fix is to cap at $K$ and, when a ball is underfull, pad the index list by repeating a found neighbor — say a ball finds only 5 points and $K=32$, so I fill the remaining 27 slots with copies of the first. Before I accept that, I should make sure the padding doesn't quietly corrupt the encoded feature. The region encoder ends in an element-wise max over the $K$ neighbors. Duplicating a point that is already in the region adds a column to the max that is identical to one already present, and $\max(a,\dots,a,\dots)=\max(a,\dots)$ — the max is idempotent under repeats. So padding-by-repeat leaves the pooled feature exactly unchanged; it's a pure no-op for the output. That's reassuring, and it's not a coincidence I lucked into: it works *because* the encoder pools with max. If I had chosen sum- or mean-pooling instead, repeating points would inflate or reweight the aggregate and the padding would be a real bug. So the variable-count ball and the max-pooling encoder are compatible by design, and I can use the ball freely.

Now the encoding step, and there's a subtlety here that matters for generalization. If I feed each region's points to the mini-encoder in their *absolute* coordinates, then the same geometric pattern sitting at a different location in space gives totally different inputs, and the encoder has to relearn it everywhere. What I want is for a region's description to depend only on the *shape* of the points relative to their centroid, not on where the region sits. The obvious fix is to translate every point in a region into a *local frame* centered at the centroid: subtract the centroid coordinate, $x_i^{(j)}\leftarrow x_i^{(j)}-\hat x^{(j)}$ for each coordinate $j$. Let me actually check this does what I claim rather than just assert it. Take a little region of four points $\{(0,0,0),(0.1,0,0),(0,0.1,0),(0,0,0.1)\}$ with the first as centroid; the relative coordinates are $\{(0,0,0),(0.1,0,0),(0,0.1,0),(0,0,0.1)\}$. Now translate the whole region by $t=(5,-3,2)$ and recompute relative-to-centroid: every absolute coordinate shifts by $t$ and so does the centroid, so the subtraction cancels $t$ exactly — I get back the identical four relative vectors (numerically the difference is at the level of floating-point rounding, $\sim\!4\times10^{-16}$). Since the mini-encoder sees only these relative coordinates, its output for the shifted region is bit-for-bit the same as for the original. Good — the local frame really does give translation invariance, and the relative positions are exactly the point-to-point geometry I want the encoder to read. Combine those relative coordinates with any incoming per-point features, run the mini set encoder (shared MLP per point in the region, then max over the region's $K$ points), and out comes one feature vector per centroid. So a set abstraction level maps $N\times(d+C)$ to $N'\times(d+C')$: fewer points, each with a richer local-context feature.

Stack a few of these and I have the hierarchy: each level subsamples and enlarges the effective receptive field, exactly like conv-with-pooling — but with two honest differences I should keep straight. The neighborhood is defined by *metric distance*, not array index; and the kernel is a *set function* (my encoder), not a convolution. With the last level grouping everything into one region, I get a single global feature for classification.

Now the harder problem is density. Real scans are *not* uniformly dense — perspective, radial falloff, and motion make some regions densely sampled and others sparse. I worry this wrecks the fixed-radius scheme I just built, so let me put a number on the worry instead of hand-waving. Fix the small radius $r=0.2$ I'm planning to use at the first level, and treat the points on a surface patch as a Poisson process at some areal density. The expected count inside a ball of radius $r$ is density $\times\,\pi r^2$. Take a dense patch and a sparse one (the kind random dropout would produce) and simulate. At a density that puts $\sim\!60$ points in the small ball — comfortably dense — essentially no ball ever has $\le 2$ points. At a sparse density where the *expected* count drops below one point per ball, I find that about 94% of those small balls contain two or fewer points. So in sparse regions the small ball is empty or near-empty the overwhelming majority of the time, and an encoder fed two points can't recognize any local geometry — the "pattern" is just sampling noise. The failure is real and quantitative, not a worst-case anecdote. This is the inverse of the grid intuition where smaller kernels help — here, if I shrink the neighborhood, sampling deficiency corrupts it. The resolution can't be a single fixed scale. What I really want: in *dense* regions, look closely and grab the finest detail; in *sparse* regions, back off to a larger neighborhood where there are enough points to see something reliable. The network should *adapt* its scale to the local density.

How do I make a single abstraction level density-adaptive? The most direct idea: don't commit to one scale — look at *several* scales at once and let the network combine them. At a given level, run the grouping at multiple radii $r_1<r_2<\dots$, each with its own mini set encoder, and *concatenate* the resulting features into one multi-scale feature per centroid. Now each centroid carries what it saw at a small scale *and* at a large scale, and the following layers can lean on whichever is reliable. Call this multi-scale grouping.

But concatenating multi-scale features only helps if the network actually *learns* to weight them according to density — otherwise it just always trusts the fine scale and I'm back to the sparse-region failure. Concatenation alone gives the network the *option* to down-weight the fine scale; it doesn't force it to. What would make it take that option? It has to have *seen* cases where the fine scale was garbage and the large scale was the only signal. So I need to show it sparsity during training: randomly drop input points. Concretely, for each training cloud I draw a dropout ratio $\theta$ uniformly from $[0,p]$ and then drop each point independently with probability $\theta$; I set $p=0.95$, below $1$, so I avoid empty sets. Sweeping $\theta$ gives the network clouds at many sparsity levels, and the per-point randomness gives many *non-uniform* patterns — exactly the regime where my Poisson check said the small ball collapses. In fixed-size tensor code I can keep the tensor shape by replacing selected entries with a duplicate (which, as I checked above, the max-pool ignores), and for labeled scene points I can zero their sample weight. My expectation is that, trained across this range of sparsities, the gradient will teach the network to discount the small-scale feature when points are sparse and lean on the large-scale one; I can't confirm that weighting without actually training, so I'd want to verify it later by measuring accuracy under test-time dropout. At test time I keep all points. So multi-scale grouping plus random input dropout are the two halves of the density-adaptivity mechanism.

Multi-scale grouping works but it's expensive, and I should see exactly where the cost lands. At every centroid I run a mini-encoder over a *large*-radius neighborhood, which contains many points; and the lowest abstraction level has the *most* centroids. So I'm running large-neighborhood encoders at thousands of centroids — that's the bulk of the compute, and it's redundant because the higher levels already inspect large extents of space. Can I get multi-scale information without re-encoding large neighborhoods at the dense low levels? Here's a cheaper construction. Form the feature of a region at level $L_i$ as the concatenation of *two* vectors. The first is the ordinary one: summarize the features of the *sub-regions* from the lower level $L_{i-1}$ through the abstraction level — this is the "already abstracted lower-resolution" path. The second is computed by running a single encoder directly on the *raw* points of the region at this level. The reason I expect this to carry density information is asymmetric reliability: when the region is *sparse*, the first vector is built from sub-regions that are *even sparser* — by the same Poisson argument, their small balls are mostly empty — so the recursive path is the more corrupted of the two and the raw-points path is the safer signal; when the region is *dense*, the first vector recursively inspected at higher resolution below and carries genuinely finer detail. So the two vectors disagree exactly along the density axis, which is the signal the network needs; whether it actually exploits it is again a training question I'd settle with the same dropout-robustness measurement. The payoff is on cost: I never encode a large neighborhood at the lowest, most populous level, since the large-scale view there is replaced by the recursive sub-region summary. Multi-resolution grouping — same adaptivity signal, much cheaper.

That handles classification, where I only need the final global feature. Segmentation needs a label for *every original point*, but my abstraction levels have been throwing points away by subsampling. One option is to never subsample — keep all points as centroids at every level — but that's prohibitively expensive. The better way is to abstract *down* and then *propagate features back up*. So I add propagation levels mirroring the abstraction levels, going from the coarse subsampled set back to the finer one. The question is how to assign a feature to a fine point given features on the coarser set. The fine point isn't one of the coarse centroids, so I *interpolate*: take its $k$ nearest coarse points and form an inverse-distance-weighted average of their features,
$$f^{(j)}(x)=\frac{\sum_{i=1}^{k} w_i(x)\,f_i^{(j)}}{\sum_{i=1}^{k} w_i(x)},\qquad w_i(x)=\frac{1}{d(x,x_i)^p},$$
with $p=2$ and $k=3$ by default. Before I trust this, let me check it behaves sanely in the case I most care about — a fine point that sits essentially on top of a coarse point should inherit that coarse point's feature, not some blurred average. Take three coarse points carrying scalar features $f=(1,2,3)$ at $(0,0,0),(1,0,0),(0,1,0)$, and slide a fine point in toward the first one. With $p=2$ the weight on coarse point $i$ is $1/d^2$, so as $d\to 0$ for the nearest point its weight blows up and dominates the normalization. Numerically: at distance $0.5$ the interpolated value is $1.636$ (a real mixture of all three), at $0.1$ it's $1.031$, at $0.01$ it's $1.0003$, and at $10^{-3}$ it's $1.000003$ — converging cleanly to $f_0=1$, with the weight on the nearest coarse point going $0.45\to0.98\to0.9998\to1.0$. So the interpolation is continuous and reduces to the exact coarse feature in the coincident limit, which is precisely the consistency I'd demand of an upsampling rule; and the $1/d^2$ falloff makes nearer coarse points dominate, the sensible prior for a smooth feature field. But interpolation alone has lost the high-resolution detail that existed at the finer level on the way *down*. So I add a *skip link*: concatenate the interpolated coarse feature with the feature this fine point had at the corresponding abstraction level (carried across by the skip), and then run a small per-point MLP — a "unit PointNet," the analogue of a $1\times1$ convolution — to fuse them into an updated per-point feature. Repeat propagation level by level until I'm back at the original points, each now carrying a feature that combines its own fine detail with progressively larger-scale context, ready for a per-point label.

Let me write the pieces concretely with the tensor cases I will need: the sampling+grouping, the set abstraction module (single-scale and multi-scale), and the feature-propagation module.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sample(xyz, npoint): ...        # (B,N,3) -> (B,npoint) indices
def ball_query(radius, nsample, xyz, new_xyz): ... # fixed radius, cap nsample, repeat if underfull
def index_points(points, idx): ...                 # gather by index
def three_nn_sqdist(query, ref, k=3): ...           # -> squared distances, indices

def sample_and_group(npoint, radius, nsample, xyz, feats):
    # 1) Sampling: farthest point sampling for good, data-dependent coverage
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)                       # (B,npoint,3) centroids
    # 2) Grouping: ball query -> fixed metric SCALE per region
    idx = ball_query(radius, nsample, xyz, new_xyz)            # (B,npoint,nsample)
    grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)  # LOCAL frame (subtract centroid)
    if feats is not None:
        grouped = torch.cat([grouped_xyz, index_points(feats, idx)], dim=-1)
    else:
        grouped = grouped_xyz
    return new_xyz, grouped                                    # (B,npoint,nsample,d+C)

class SetAbstraction(nn.Module):
    """One level: sample centroids, group neighborhoods, encode each with a
       mini set-encoder (shared MLP per point in region, then max over region)."""
    def __init__(self, npoint, radius, nsample, in_ch, mlp, group_all=False):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.group_all = group_all
        layers, c = [], in_ch + 3                  # +3 for the relative xyz
        for out in mlp:
            layers += [nn.Conv2d(c, out, 1), nn.BatchNorm2d(out), nn.ReLU()]; c = out
        self.mlp = nn.Sequential(*layers)
    def forward(self, xyz, feats):                 # xyz:(B,N,3) feats:(B,N,C)
        if self.group_all:                         # last level: one region = whole set
            new_xyz = xyz.mean(1, keepdim=True) * 0
            grouped = (torch.cat([xyz, feats], -1) if feats is not None else xyz).unsqueeze(1)
        else:
            new_xyz, grouped = sample_and_group(self.npoint, self.radius,
                                                self.nsample, xyz, feats)
        g = grouped.permute(0, 3, 1, 2)            # (B, d+C, npoint, nsample)
        g = self.mlp(g)                            # shared MLP on each point in each region
        new_feats = g.max(dim=3)[0].transpose(1, 2)   # MAX over the region -> (B,npoint,mlp[-1])
        return new_xyz, new_feats

class SetAbstractionMSG(nn.Module):
    """Density-adaptive level: group at MULTIPLE radii, encode each, concatenate."""
    def __init__(self, npoint, radii, nsamples, in_ch, mlps):
        super().__init__()
        self.npoint, self.radii, self.nsamples = npoint, radii, nsamples
        self.branches = nn.ModuleList()
        for mlp in mlps:
            layers, c = [], in_ch + 3
            for out in mlp:
                layers += [nn.Conv2d(c, out, 1), nn.BatchNorm2d(out), nn.ReLU()]; c = out
            self.branches.append(nn.Sequential(*layers))
    def forward(self, xyz, feats):
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
        outs = []
        for radius, nsample, branch in zip(self.radii, self.nsamples, self.branches):
            idx = ball_query(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
            grouped = (torch.cat([grouped_xyz, index_points(feats, idx)], -1)
                       if feats is not None else grouped_xyz)
            g = branch(grouped.permute(0, 3, 1, 2)).max(dim=3)[0]   # per-scale feature
            outs.append(g)
        return new_xyz, torch.cat(outs, dim=1).transpose(1, 2)      # concat scales

class FeaturePropagation(nn.Module):
    """Carry features from coarse set up to fine set: IDW interpolation + skip + unit MLP."""
    def __init__(self, in_ch, mlp):
        super().__init__()
        layers, c = [], in_ch
        for out in mlp:
            layers += [nn.Conv1d(c, out, 1), nn.BatchNorm1d(out), nn.ReLU()]; c = out
        self.mlp = nn.Sequential(*layers)
    def forward(self, xyz_fine, xyz_coarse, feats_fine, feats_coarse):
        if xyz_coarse.size(1) == 1:
            interp = feats_coarse.expand(-1, xyz_fine.size(1), -1)
        else:
            sqdist, idx = three_nn_sqdist(xyz_fine, xyz_coarse, k=min(3, xyz_coarse.size(1)))
            inv = 1.0 / sqdist.clamp_min(1e-10)                # p=2 because sqdist=d^2
            w = inv / inv.sum(-1, keepdim=True)
            interp = (index_points(feats_coarse, idx) * w.unsqueeze(-1)).sum(2)
        if feats_fine is not None:
            interp = torch.cat([interp, feats_fine], dim=-1)   # skip link
        return self.mlp(interp.transpose(1, 2)).transpose(1, 2)  # unit PointNet
```

The classification network stacks three abstraction levels — sample 512 centroids at radius 0.2 with the mini-encoder, then 128 centroids at radius 0.4, then group all into one global feature — followed by fully connected layers with dropout, trained with softmax cross-entropy; the segmentation network mirrors the abstraction levels with propagation levels and skip links back to the original points.

Stepping back, the chain that got me here: a single global pooling has no place to form local structure, and the one property a CNN has that I lacked is abstracting *local* patterns at increasing scales — so I applied my set encoder recursively over a nested partition, each set-abstraction level sampling centroids by farthest point sampling (data-dependent coverage), grouping by ball query (fixed metric scale, and I checked the ball's variable count is harmless because the encoder pools with max), and encoding each centroid-relative neighborhood — relative because I verified the centroid subtraction makes the encoding translation-invariant to rounding error — with a mini set encoder. The thing that forced the design past a plain hierarchy was density: I computed that at a fixed small radius a sparse region's ball holds $\le 2$ points about 94% of the time, so a single small scale genuinely collapses there. That pushed me to group at multiple scales — or across resolutions, more cheaply — paired with random input dropout so training exposes the sparsity the multi-scale features are meant to hedge against. The two density claims I'm *least* sure of are training outcomes (that the network actually learns the density weighting); those I've flagged for a dropout-robustness measurement rather than asserted. And because abstraction subsamples, per-point segmentation recovers full-resolution features by inverse-distance interpolation from the coarse level — which I checked reduces exactly to the coarse feature in the coincident limit — plus skip links and a unit per-point MLP, propagated level by level back to every original point.
