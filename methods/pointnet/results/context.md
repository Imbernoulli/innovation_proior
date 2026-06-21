# Context

## Research question

The goal is a deep network that learns directly from raw 3D point clouds — a
point cloud being an unordered set $\{P_i \mid i=1,\dots,n\}$ where each $P_i$ is
its $(x,y,z)$ coordinate (optionally with extra channels such as color or normal)
— and that serves a *unified* set of recognition tasks: object classification
(output $k$ class scores for the whole cloud) and segmentation (output $n\times m$
scores, one label distribution per point). The dominant deep architectures of the
time, convolutional networks, require *regular* input formats — pixel grids, voxel
grids — so that weight sharing and kernel optimizations apply. The standard practice
is to first convert point clouds into a voxel grid or a collection of rendered images
and then run a CNN.

## Background

**The regular-grid assumption of convolution.** Convolutional networks get their
efficiency from weight sharing over a regular lattice: the same kernel slides over a
grid of pixels or voxels. This is why 3D recognition pipelines first regularize the
data. Volumetric grids and multi-view renderings are the two standard regularizers,
each discussed under Baselines.

**Symmetric functions and order invariance.** A function of several arguments is
*symmetric* if its value is unchanged under any permutation of those arguments;
addition and multiplication are the elementary examples. A symmetric function that
takes $n$ vectors and returns one vector is exactly the tool for collapsing an
unordered set into an order-independent summary. The relevant theoretical backdrop is the universal
approximation property of multilayer perceptrons: a single hidden layer with enough
units can approximate any continuous function on a compact domain — which suggests
that if the *form* of an order-invariant architecture is right, enough capacity can
make it approximate the desired set function.

**Hausdorff distance and continuous set functions.** To even speak of a network
being robust to small input perturbations, one needs a notion of two point sets
being close. The Hausdorff distance $d_H(S,S')$ measures how far the two sets are as
sets (the greatest distance from a point in one set to its nearest point in the
other). A set function $f$ is *continuous* w.r.t. $d_H$ if small Hausdorff
perturbations of the input produce small changes in $f$ — the property a
classification or segmentation score should have.

**Learnable input alignment.** Spatial transformer networks (Jaderberg et al. 2015)
showed that a network can learn to *canonicalize* its input: a small sub-network
predicts a spatial transformation that is applied to the data before the main network
sees it, making the main network's job invariant to that family of transformations.
For images this required a specially built sampling-and-interpolation layer.

**Sets and sequences.** One alternative for handling unordered input is to treat the
set as a sequence and use a recurrent network, training on randomly permuted
orderings in the hope that the model becomes order-agnostic. *Order Matters* (Vinyals
et al. 2015) studied exactly this and found that the order in which a set is presented
to a sequence model genuinely affects what it learns; recurrent models tolerate
re-ordering only for short sequences.

## Baselines

**Volumetric CNNs (3DShapeNets, Wu et al. 2015; VoxNet, Maturana & Scherer 2015).**
Convert the cloud to an occupancy grid (e.g. $32\times32\times32$) and apply 3D
convolutions. They are also sensitive to orientation (VoxNet averages predictions over many rotated
views).

**Multi-view CNNs (MVCNN, Su et al. 2015; Qi et al. 2016).** Render the shape from
several virtual camera viewpoints into 2D images, run a 2D CNN per view, and pool
across views. They leverage mature 2D CNNs and large image-pretrained models, and
require a rendering step with viewpoints chosen by hand.

**Sorting the points into a canonical order, then an MLP/CNN.** The simplest way to
make an order-dependent network order-invariant: impose a deterministic order first.

**RNN over the (permutation-augmented) point sequence.** Feed the points as a sequence
to a recurrent net and train on many random orderings.

**Spatial transformer networks (Jaderberg et al. 2015).** The mechanism for learnable
input canonicalization, built for images with a resampling layer.

## Evaluation settings

The natural yardsticks are 3D shape classification on ModelNet40 (CAD models from 40
categories, point clouds sampled from the surfaces, with up-axis rotation and jitter
augmentation; the metric is overall classification accuracy), 3D object part
segmentation on the ShapeNet part dataset (16 object categories with per-point part
labels; metric mean intersection-over-union over parts), and indoor scene semantic
parsing on the Stanford 3D indoor dataset (per-point semantic labels over rooms). A
typical input is on the order of 1024 points represented by $(x,y,z)$. Two
robustness-relevant protocols belong to the setting: testing under random point
*dropout* (varying the fraction of missing points), and comparing against a volumetric
network that consumes the same clouds voxelized to a $32^3$ grid. A useful sanity-check
setting is MNIST recast as 2D point sets (thresholded pixels as $(x,y)$ points).

## Code framework

The pieces below already exist as standard primitives: a per-point shared multilayer
perceptron (the same small MLP applied independently to every point, implemented as
$1\times1$ convolutions over the point axis), batch normalization and ReLU, fully
connected layers with dropout, generic reductions over a chosen axis, an affine
transform of point coordinates by matrix multiplication, the Adam optimizer, and a
softmax cross-entropy loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedMLP(nn.Module):
    """The same MLP applied independently to every point (1x1 convs). (exists)"""
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv1d(a, b, 1) for a, b in zip(sizes[:-1], sizes[1:])])
        self.bns = nn.ModuleList([nn.BatchNorm1d(b) for b in sizes[1:]])
    def forward(self, x):                       # x: (B, C, N)
        for conv, bn in zip(self.layers, self.bns):
            x = F.relu(bn(conv(x)))
        return x

# --- the slots the method will fill ----------------------------------------
class SetAggregator(nn.Module):
    """Collapse per-point features (B, K, N) into one order-invariant vector."""
    def forward(self, point_feats):
        # TODO: which order-invariant reduction over the N points?
        raise NotImplementedError

class AlignmentNet(nn.Module):
    """Make the representation invariant to rigid/affine motion of the cloud."""
    def forward(self, x):
        # TODO
        raise NotImplementedError

class Net(nn.Module):
    def forward(self, points):                  # points: (B, 3, N)
        # TODO: produce class scores for the whole cloud, and per-point scores
        raise NotImplementedError

def loss_fn(pred, label, feature_transform):
    # softmax cross-entropy + TODO: any regularizer the alignment needs
    raise NotImplementedError
```
