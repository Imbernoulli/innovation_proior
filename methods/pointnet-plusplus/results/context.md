# Context

## Research question

The goal is to learn deep features on point sets sampled from a *metric space* —
formally a discrete metric space $\mathcal X=(M,d)$ with $M\subseteq\mathbb R^n$ and
the Euclidean metric $d$ — and predict semantic information about $\mathcal X$
(a class label for the whole set, or a label per point). The data is a 3D point
cloud, e.g. from a depth scanner, and it has two properties that any solution must
honor: as a *set* it is invariant to permutations of its members, and the metric
$d$ defines local neighborhoods whose properties — notably **sampling density** —
vary across the cloud (perspective, radial falloff, and motion all make real scans
denser in some regions than others).

The question is how to build a deep network that learns features on such an
*irregular, unordered, non-uniformly sampled* point set.

## Background

**A permutation-invariant set encoder.** The available building block is a network
that maps an unordered set to a vector by encoding each point with a shared MLP and
then applying a symmetric pooling (element-wise max),
$$f(x_1,\dots,x_n)=\gamma\Big(\underset{i}{\mathrm{MAX}}\,\{h(x_i)\}\Big),$$
with $\gamma,h$ MLPs. This is permutation-invariant by construction and can
approximate any continuous set function arbitrarily well given enough pooling width;
the per-point encoding $h(x_i)$ can be read as a spatial encoding of the point. It is
also empirically robust to point corruption.

**Generic set networks vs. the metric.** Prior work on learning representations of
generic sets treats each element as independent and relies on a global normalization
plus a max/average pooling to aggregate; the result is either a single-point
embedding or one global set feature.

**The convolutional hierarchy.** A CNN stacks layers each of which looks at a local
neighborhood (a kernel of fixed index-extent) and pools/strides to a coarser grid,
so deeper layers see larger receptive fields and abstract larger-scale patterns over
a multi-resolution hierarchy. The size of the local neighborhood is the kernel size,
and on grids smaller kernels tend to help (VGG, Simonyan & Zisserman 2014).

**Choosing centroids and neighborhoods.** Two primitives are available for imposing
structure on a metric point set. Farthest point sampling iteratively picks the point
most distant from those already chosen, giving good coverage of the set with few
centroids and, unlike a fixed grid, placing centroids in a data-dependent way. For a
neighborhood around a centroid there are two standard range queries: a ball query
(all points within a fixed radius, hence a fixed metric *scale*) and $k$-nearest
neighbors (a fixed *count* of points).

**Interpolation for upsampling.** When features have been computed only on a
subsampled set but are needed at the original resolution (for per-point prediction),
inverse-distance-weighted interpolation over the $k$ nearest sampled points,
$f(x)=\frac{\sum_i w_i(x)f_i}{\sum_i w_i(x)}$ with $w_i(x)=1/d(x,x_i)^p$, is the
standard scattered-data interpolation tool.

## Baselines

**The single-shot global set encoder (PointNet, Qi et al. 2016).** Encode each point
with a shared MLP, then one global max-pool, then classify; for segmentation,
concatenate the single global feature back onto each point. It is permutation
invariant, a universal continuous-set-function approximator, and robust to
corruption.

**Generic set / Deep-Sets-style networks.** Aggregate independent per-element
embeddings by a symmetric pooling after global normalization.

**Volumetric CNNs.** Impose a regular grid and run 3D convolutions, getting the
convolutional hierarchy.

## Evaluation settings

The yardsticks are 3D shape classification on ModelNet40 (CAD models, point clouds
of $(x,y,z)$ — typically around 1024 points — with the usual rotation/jitter
augmentation; metric overall accuracy), 3D shape part segmentation on the ShapeNet
part dataset (per-point part labels; metric mean part IoU), and indoor scene semantic
segmentation on ScanNet indoor scenes (per-point semantic labels over rooms).
A useful stress test is robustness to **non-uniform sampling density**: training
and/or testing under random point dropout at varying ratios, so that a model must
perform across densities rather than at a single fixed density. A normal-channel
input variant (each point carries its surface normal in addition to coordinates)
is part of the setting.

## Code framework

The pieces below already exist as standard primitives: a permutation-invariant
set encoder (shared per-point MLP followed by a symmetric max-pool), batch
normalization, ReLU, fully connected layers with dropout, a farthest-point-sampling
routine, a ball / $k$-NN range query over a metric point set, a gather/group
operation that collects neighbor points by index, and an inverse-distance
interpolation over nearest neighbors.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- existing primitives ---------------------------------------------------
def farthest_point_sample(xyz, npoint): ...      # -> indices of npoint centroids
def ball_query(radius, nsample, xyz, centroids): ...   # -> neighbor indices
def index_points(points, idx): ...               # gather by index
def knn(xyz_query, xyz_ref, k): ...              # k nearest neighbors

class SharedMLP(nn.Module):                      # shared per-point/region MLP (exists)
    def __init__(self, sizes): ...
    def forward(self, x): ...

def set_encoder(region_points):
    # permutation-invariant encode-then-max over a region's points (exists)
    return SharedMLP(...)(region_points).max(dim=...)[0]

# --- to be designed --------------------------------------------------------
class Net(nn.Module):
    def forward(self, xyz):
        # TODO: from the primitives above, build a model that learns local
        #       features, stays robust to non-uniform density, and (for
        #       segmentation) yields a feature at every original point
        raise NotImplementedError
```
