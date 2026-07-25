I will call the method PointNet++. The problem I am solving is learning deep features directly on point clouds, which are unordered sets of points sampled from a metric space. A point cloud has two properties that any good model must respect. First, it is a set, so the network must be permutation invariant: reshuffling the points must not change the prediction. Second, the points come from a metric space, usually three-dimensional Euclidean space, and real scans exhibit highly non-uniform sampling density: some regions are densely sampled while others are sparse because of perspective, radial falloff, and motion. The goal is to build a hierarchical, local-feature representation that generalizes across these variations.

My starting observation is that the simplest permutation-invariant set encoder already works as a universal approximator: encode every point with a shared multilayer perceptron, take an element-wise max over all points, and then apply another MLP to the pooled vector. That construction is robust, but it has a crucial limitation. It collapses the entire cloud into a single global vector in one shot, so every point is encoded in isolation and no intermediate local structure is ever formed. A corner looks like a corner only because the global max happens to fire; there is no explicit notion of "these nearby points form a local patch." Convolutional neural networks succeed on images because they build a multi-resolution hierarchy: early layers have small receptive fields that detect local patterns, and later layers combine those patterns into larger-scale structure. I want the same hierarchical local-feature learning for irregular, unordered, non-uniformly sampled point sets.

The central idea of PointNet++ is to apply the set encoder recursively on a nested partitioning of the cloud. Each level of the hierarchy, which I call a set abstraction level, maps a set of points with their features to a smaller set of points with richer features. A set abstraction level has three steps. First, sampling: I choose a subset of centroids using farthest point sampling. Starting from a random point, I repeatedly add the point that is farthest in metric distance from the points already chosen. This gives good coverage with few centroids and, unlike a fixed grid, places the centroids in a data-dependent way so that receptive fields follow the actual geometry. Second, grouping: for each centroid I gather a local neighborhood using a ball query, that is, all points within a fixed radius, capped at a maximum count. A ball query gives a fixed metric scale everywhere in the cloud, so a local feature learned at one place means the same physical thing at another place. This is preferable to k-nearest neighbors, where the physical size of the neighborhood floats with density and the same feature vector can correspond to very different scales. Third, encoding: I translate every point in the neighborhood into a centroid-relative frame by subtracting the centroid coordinate, which gives translation invariance and makes the encoding depend on point-to-point relations rather than absolute location. I then run a shared MLP over the points in the region and max-pool over the region to produce one feature vector per centroid.

Stacking several set abstraction levels gives the desired hierarchy. Early levels have many centroids and small neighborhoods, so they capture fine local geometry. Later levels have fewer centroids and larger neighborhoods, so they aggregate those local patterns into larger-scale structure. The final level can group all remaining points into a single region, producing a global descriptor for classification.

The non-uniform density of real scans breaks a naive single-scale hierarchy. In a sparse region, a small ball may contain only a handful of points, so the local feature becomes dominated by sampling noise rather than by meaningful geometry. The solution is to make each level density adaptive. One option is multi-scale grouping: at each centroid I group points at several radii, encode each scale with its own mini set encoder, and concatenate the resulting features. The network then learns to weight the small-scale feature when the region is dense and the large-scale feature when the region is sparse. To teach this weighting, I apply random input dropout during training: for each training cloud I draw a dropout ratio uniformly from zero up to a high value such as 0.95, then drop each input point independently with that probability. This exposes the network to many different densities and non-uniform patterns, so it learns which scale is reliable. At test time I keep all points.

Multi-scale grouping is effective but expensive, because the lowest level has the most centroids and the largest-radius neighborhoods contain the most points. A cheaper alternative is multi-resolution grouping. At each level I concatenate two vectors per region. The first vector is the ordinary output of the abstraction level applied to the features from the previous, finer level. The second vector is computed by running a single set encoder directly on the raw points of the region at the current level. When the region is sparse, the first vector is unreliable because it was built from even sparser sub-regions, so the network should rely more on the second. When the region is dense, the first vector carries finer recursive detail and is more valuable. The same random input dropout training teaches the network this weighting, but multi-resolution grouping avoids encoding large neighborhoods at the populous lowest level.

For per-point tasks such as semantic segmentation, the abstraction levels throw away points by subsampling, so I need to propagate features back up to the original resolution. I do this with feature propagation levels that mirror the abstraction levels. To assign a feature to a fine point given features on a coarser set, I use inverse-distance-weighted interpolation over the nearest coarse points. Nearer coarse points dominate, which is the natural prior for a smooth feature field. Interpolation alone would lose the fine detail that existed on the way down, so I add a skip link that concatenates the interpolated coarse feature with the feature the fine point had at the corresponding abstraction level. A small per-point MLP, essentially a unit PointNet, fuses these into an updated per-point feature. Repeating this level by level brings the features back to every original point, now combining local detail with large-scale context.

The canonical classification architecture stacks three set abstraction levels: 512 centroids at radius 0.2 with a mini-encoder producing 128 channels, then 128 centroids at radius 0.4 producing 256 channels, then a global level producing a 1024-dimensional descriptor, followed by fully connected layers with dropout and softmax cross-entropy. The segmentation architecture mirrors the abstraction levels with feature propagation levels and skip links back to the input points. I have kept the method name PointNet++ throughout, because the plus-plus captures exactly this extension from a single global pooling to a hierarchical, density-adaptive, multi-scale point-set network.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sample(xyz, npoint): ...        # (B,N,3) -> (B,npoint) indices
def ball_query(radius, nsample, xyz, new_xyz): ... # fixed radius, cap nsample, repeat if underfull
def index_points(points, idx): ...
def three_nn_sqdist(query, ref, k=3): ...           # -> squared distances, indices

def sample_and_group(npoint, radius, nsample, xyz, feats):
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = ball_query(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)        # local frame
    grouped = (torch.cat([grouped_xyz, index_points(feats, idx)], -1)
               if feats is not None else grouped_xyz)
    return new_xyz, grouped

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_ch, mlp, group_all=False):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        layers, c = [], in_ch + 3
        for out in mlp:
            layers += [nn.Conv2d(c, out, 1), nn.BatchNorm2d(out), nn.ReLU()]; c = out
        self.mlp = nn.Sequential(*layers)
    def forward(self, xyz, feats):
        if self.group_all:
            new_xyz = xyz[:, :1] * 0
            grouped = (torch.cat([xyz, feats], -1) if feats is not None else xyz).unsqueeze(1)
        else:
            new_xyz, grouped = sample_and_group(self.npoint, self.radius, self.nsample, xyz, feats)
        g = self.mlp(grouped.permute(0, 3, 1, 2))
        return new_xyz, g.max(dim=3)[0].transpose(1, 2)               # max over region

class SetAbstractionMSG(nn.Module):
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
            outs.append(branch(grouped.permute(0, 3, 1, 2)).max(dim=3)[0])
        return new_xyz, torch.cat(outs, dim=1).transpose(1, 2)

class FeaturePropagation(nn.Module):
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
            inv = 1.0 / sqdist.clamp_min(1e-10)                 # p=2 because sqdist=d^2
            w = inv / inv.sum(-1, keepdim=True)
            interp = (index_points(feats_coarse, idx) * w.unsqueeze(-1)).sum(2)
        if feats_fine is not None:
            interp = torch.cat([interp, feats_fine], dim=-1)         # skip link
        return self.mlp(interp.transpose(1, 2)).transpose(1, 2)      # unit PointNet

class PointNet2Cls(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.sa1 = SetAbstraction(512, 0.2, 32, 0,   [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = SetAbstraction(None, None, None, 256, [256, 512, 1024], group_all=True)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes))
    def forward(self, xyz):                          # xyz: (B,N,3)
        l1_xyz, l1 = self.sa1(xyz, None)
        l2_xyz, l2 = self.sa2(l1_xyz, l1)
        _,      l3 = self.sa3(l2_xyz, l2)
        return self.fc(l3.squeeze(1))
```
