# PointNet++

## Problem

Learn deep features on point sets sampled from a metric space, with local
structure at multiple scales (which a single global pooling cannot capture) and
robustness to the **non-uniform sampling density** of real 3D scans. The idea is to
apply a permutation-invariant set encoder *recursively* on a nested partitioning of
the cloud, building a CNN-like multi-resolution hierarchy of local features.

## Key idea

**Set abstraction (SA) level** — three steps mapping $N\times(d+C)\to N'\times(d+C')$:

1. **Sampling (farthest point sampling).** Iteratively pick the point most distant
   (in the metric) from those already chosen — good coverage with few centroids, and
   *data-dependent* receptive fields (unlike a CNN's data-agnostic fixed-stride scan).
2. **Grouping (ball query).** For each centroid, gather all points within radius $r$
   (capped at $K$). A ball gives a *fixed metric scale* per region — so local features
   generalize across space — preferable to $k$-NN (fixed count, floating scale) for
   local pattern recognition.
3. **Mini-PointNet.** Translate each region's points into the centroid-relative frame
   $x_i\leftarrow x_i-\hat x$ (translation invariance, captures point-to-point
   relations), then encode with a shared per-point MLP and max-pool over the region.

Stacking SA levels coarsens the cloud and enlarges receptive fields, like
convolution-with-pooling — but neighborhoods are defined by metric distance and the
kernel is a set function.

**Density-adaptive grouping (the "++").** A single small scale fails in sparse regions
(too few points → sampling noise). Two layer types adapt scale to local density:

- **Multi-scale grouping (MSG).** Group at several radii, encode each with its own
  mini-PointNet, concatenate. The network learns to weight scales by density via
  **random input dropout**: per training cloud draw $\theta\sim\mathrm{Uniform}[0,p]$
  ($p=0.95$ in the training rule), drop each point with probability $\theta$; keep all points
  at test time. In fixed-size tensor code, the same augmentation can be represented by
  replacing dropped entries with a duplicate point and, for labeled scene points, zeroing
  their sample weight.
- **Multi-resolution grouping (MRG).** Concatenate two vectors per region: one
  summarizing features of the lower-level sub-regions through the SA level, one from a
  single PointNet on the region's *raw* points. In sparse regions the first is
  unreliable (sub-regions even sparser) so the second is weighted higher; in dense
  regions the first carries finer recursive detail. Cheaper than MSG (no large-scale
  encoding at the populous lowest level).

**Feature propagation (FP) for segmentation.** SA subsamples; per-point tasks need
features at all original points. Propagate coarse→fine by inverse-distance-weighted
interpolation over $k=3$ nearest coarse points,
$$f^{(j)}(x)=\frac{\sum_{i}w_i(x)f_i^{(j)}}{\sum_i w_i(x)},\qquad w_i(x)=\frac{1}{d(x,x_i)^p}\ (p=2),$$
concatenate with the skip-linked feature from the matching SA level, and fuse with a
"unit PointNet" (per-point MLP). Repeat up to the original resolution.

## Architecture (SSG classification)

SA1: $\mathrm{npoint}=512$, $r=0.2$, $\mathrm{nsample}=32$, $\mathrm{mlp}=[64,64,128]$;
SA2: $\mathrm{npoint}=128$, $r=0.4$, $\mathrm{nsample}=64$, $\mathrm{mlp}=[128,128,256]$;
SA3: group-all, $\mathrm{mlp}=[256,512,1024]$ → global $1024$; FC$(512$, dropout $0.5)$,
FC$(256$, dropout $0.5)$, FC$(40)$. BatchNorm + ReLU; softmax cross-entropy loss.
Segmentation stacks FP levels with skip links back to the input points.

## Code

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
