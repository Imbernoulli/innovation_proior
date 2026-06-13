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

The pain point is sharper than "process a set." A network that aggregates all
points into a single global descriptor in one shot — encode each point, then one
symmetric pooling over the whole set — does respect permutation invariance, but it
has no notion of *local* structure at intermediate scales: every point is encoded
independently and then merged into one global vector, so fine-grained local
geometric patterns are never formed and the model generalizes poorly to complex
scenes with many parts. Convolutional networks succeed on grids precisely because
they build a *multi-resolution hierarchy* of local receptive fields — small
neighborhoods low in the network, progressively larger ones higher up — and that
hierarchical abstraction of local patterns is what gives them generalizability. The
question is how to get the same hierarchical local-feature learning for an
*irregular, unordered, non-uniformly sampled* point set, where there is no grid to
slide a kernel over.

Whatever the solution, it has to remain robust to the density variation that real
scans exhibit — a local pattern that is recognizable in a densely sampled region
may be destroyed by sampling deficiency in a sparse one.

## Background

**A permutation-invariant set encoder.** The available building block is a network
that maps an unordered set to a vector by encoding each point with a shared MLP and
then applying a symmetric pooling (element-wise max),
$$f(x_1,\dots,x_n)=\gamma\Big(\underset{i}{\mathrm{MAX}}\,\{h(x_i)\}\Big),$$
with $\gamma,h$ MLPs. This is permutation-invariant by construction and can
approximate any continuous set function arbitrarily well given enough pooling width;
the per-point encoding $h(x_i)$ can be read as a spatial encoding of the point. It is
also empirically robust to point corruption. Its limitation as a *whole-cloud*
encoder is the single global pooling: no local context.

**Generic set networks vs. the metric.** Prior work on learning representations of
generic sets treats each element as independent and relies on a global normalization
plus a max/average pooling to aggregate; the result is either a single-point
embedding or one global set feature, with no explicit modeling of point-to-point
*relations*. For point clouds the metric distance is information that such generic
set networks throw away — neighborhoods, local context, and sampling density are all
defined by $d$ and are exactly what a point-cloud model should exploit.

**The convolutional hierarchy.** A CNN stacks layers each of which looks at a local
neighborhood (a kernel of fixed index-extent) and pools/strides to a coarser grid,
so deeper layers see larger receptive fields and abstract larger-scale patterns over
a multi-resolution hierarchy. The size of the local neighborhood is the kernel size,
and on grids smaller kernels tend to help (VGG, Simonyan & Zisserman 2014). The
ingredients of a CNN that do *not* transfer directly to point sets: it assumes a
grid to define neighborhoods, a fixed-stride scan agnostic of where the data
actually is, and a fixed uniform density (the last violated by real scans).

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
corruption. Its gap: a single global pooling captures no local structure at
intermediate scales, so it misses fine-grained patterns and generalizes poorly to
scenes; the per-point segmentation features see only their own encoding plus one
global vector, with no genuine local neighborhood context.

**Generic set / Deep-Sets-style networks.** Aggregate independent per-element
embeddings by a symmetric pooling after global normalization. Gap: they ignore the
metric — no point-to-point relations, no local context, no notion of sampling
density.

**Volumetric CNNs.** Impose a regular grid and run 3D convolutions, getting the
convolutional hierarchy "for free." Gap: cubic cost on mostly-empty grids,
quantization, and a fixed-stride scan that is agnostic to where the data actually is.

## Evaluation settings

The yardsticks are 3D shape classification on ModelNet40 (CAD models, point clouds
of $(x,y,z)$ — typically around 1024 points — with the usual rotation/jitter
augmentation; metric overall accuracy), 3D shape part segmentation on the ShapeNet
part dataset (per-point part labels; metric mean part IoU), and indoor scene semantic
segmentation on the Stanford 3D indoor dataset (per-point semantic labels over rooms).
A defining protocol for this work is robustness to **non-uniform sampling density**:
training and/or testing under random point dropout at varying ratios, so that a model
must perform across densities rather than at a single fixed density. A normal-channel
input variant (each point carries its surface normal in addition to coordinates) is
part of the setting.

## Code framework

The pieces below already exist as standard primitives: a permutation-invariant
set encoder (shared per-point MLP followed by a symmetric max-pool), batch
normalization, ReLU, fully connected layers with dropout, a farthest-point-sampling
routine, a ball / $k$-NN range query over a metric point set, a gather/group
operation that collects neighbor points by index, and an inverse-distance
interpolation over nearest neighbors. What does *not* yet exist is how these
primitives should be organized into a model that learns local features and is robust
to the density variation real scans exhibit, and — for per-point prediction — how to
produce a label at every original point.

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
