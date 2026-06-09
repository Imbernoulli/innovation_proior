## Research question

Region-based convolutional detectors had just made object detection on PASCAL VOC dramatically
more accurate by classifying bottom-up region proposals with a deep network. But the recipe that
delivered that accuracy was a slow, multi-stage contraption: fine-tune a ConvNet on warped
proposals, then extract and cache features for every proposal of every image to disk, then train
per-class SVMs on those cached features, then train bounding-box regressors as yet another stage.
The precise question is whether the same (or better) detection accuracy can be obtained from a
*single*, end-to-end-trainable network — one that processes an image once, shares all convolutional
computation across that image's many proposals, jointly learns classification and box refinement
in one training stage, updates *all* of its layers (including the convolutional ones), and needs no
on-disk feature cache. The goal is to keep the deep-features-on-proposals accuracy while removing
the pipeline's training/test cost and its multi-stage inelegance.

## Background

**Why detection is expensive.** Detection demands accurate localization, which forces two things:
many candidate locations ("proposals") must be processed, and each candidate gives only a rough
location that must be refined. Both push cost up. The most accurate systems of the time ran a deep
ConvNet *once per proposal* — thousands of forward passes per image — and trained classifier and box
regressor in separate stages.

**Region proposals as a cascade.** Category-independent proposal methods (selective search and the
like) emit a sparse set of a few thousand candidate windows per image with high recall. Classifying
a sparse proposal set is a form of cascade: the proposal stage cheaply rejects the vast majority of
possible windows, leaving the expensive classifier a small set to evaluate. A dense alternative
(scanning over scale, position, aspect ratio at tens of thousands of boxes per image) is also
possible but changes the statistics of the candidate set.

**Bounding-box parameterization.** The standard way to refine a proposal box toward a target box is
to regress a four-tuple of offsets: a scale-invariant translation of the box center (normalized by
the proposal's width and height) and a log-space shift of its width and height. This makes the
regression target invariant to the proposal's absolute size and keeps the predicted size positive.

**Fixed-length features from variable regions.** A deep classification ConvNet ends in fully
connected layers that require a fixed-length input, but a proposal is an arbitrary rectangle. Two
families of solutions existed. One crops/warps each proposal in *image* space to a canonical size
and runs the whole network on it (one forward pass per proposal — no sharing). The other computes a
convolutional feature map over the *whole image once* and then pools, for each proposal, the part of
the feature map inside it into a fixed-size output — sharing all convolutional work across proposals.
The second is built on spatial pyramid pooling: pool a region into several grid resolutions (e.g.
multiple H×W grids) and concatenate, yielding a fixed-length vector regardless of region size.

**Multi-task learning.** When several tasks share a representation, training them jointly can
improve each, because the shared trunk is shaped by all of them at once (Caruana).

**Regression losses.** An L2 regression loss is sensitive to outliers and, when the
regression targets are unbounded, can produce exploding gradients that demand careful
learning-rate tuning.

## Baselines

**R-CNN (Girshick et al.).** A deep ConvNet classifies ~2000 selective-search proposals per image,
each warped in image space to a fixed 227×227 input and forward-propagated *independently*. Training
is three stages: fine-tune the ConvNet with log loss; fit per-class linear SVMs (with hard-negative
mining) to the fixed CNN features, replacing the fine-tuned softmax; fit per-class bounding-box
regressors. Drawbacks: training is multi-stage; features for every proposal of every image are
written to disk (hundreds of GB; ~2.5 GPU-days for a very deep net on 5k images); and test time is
dominated by one full ConvNet forward pass *per proposal* (≈47 s/image with a very deep net), because
no computation is shared across proposals.

**SPPnet (He et al., 2014).** Speeds R-CNN up by sharing convolutional computation: compute the conv
feature map for the whole image once, then extract each proposal's feature by spatial-pyramid-pooling
the corresponding region of that map into a fixed-length vector. This gives 10–100× faster test time
and ~3× faster training. But it keeps the multi-stage pipeline (feature extraction → log-loss
fine-tuning → SVMs → box regressors), still writes features to disk, and — crucially — its
fine-tuning algorithm *cannot update the convolutional layers beneath the pooling layer*; only the
fully connected layers above the pooling are adapted. Freezing the conv trunk limits the
accuracy of very deep networks, whose deep conv features are exactly what one would most want to
adapt to detection.

**OverFeat and proposal-network detectors.** Other systems that train a classifier and a box
localizer, but with stage-wise training and/or a separate localization network rather than one
jointly trained model.

## Evaluation settings

The yardstick is PASCAL VOC detection — VOC2007 for ablating design decisions (best practice), and
VOC2010/2012 for held-out test via the evaluation server — across 20 object classes, with mean
average precision (mAP) as the metric (a detection is correct at IoU > 0.5 with a ground-truth box
of the right class; duplicates count as false positives). Proposals come from selective search
(~2000/image; a "quality" mode can sweep 1k–10k). Image-level-pretrained ImageNet classification
networks of varying depth (a small AlexNet-like net, a medium net, and a deep 16-layer net) are the
available initializations. Training/test speed (seconds per image, GPU-hours to train) and the
fraction of forward-pass time spent in convolutional vs. fully connected layers are reported
alongside accuracy. Average Recall (AR) is a proposal-quality proxy metric in use at the time. MS
COCO (with its IoU-averaged AP metric) is an emerging larger benchmark.

## Code framework

The available primitives are: pre-trained ImageNet classification ConvNets usable as backbones; a
convolutional feature extractor over a whole image; standard SGD with momentum and weight decay;
softmax cross-entropy; a region-proposal routine; box-offset utilities (the scale-invariant,
log-space parameterization) and per-class non-maximum suppression. The scaffold wires these together
and leaves empty the slots the method must fill: how a variable-size region of the shared feature
map becomes a fixed-length feature (forward *and* backward), how the network branches into its
prediction heads, what single loss trains classification and localization together, and how
mini-batches are sampled so that training all layers is efficient.

```python
import torch
from torch import nn

def proposals(image):
    # existing region-proposal routine -> ~2000 candidate boxes
    ...

def nms_per_class(boxes, scores, iou_thresh):
    # existing greedy per-class non-maximum suppression
    ...

def encode_boxes(proposal, target):
    # existing scale-invariant translation + log-space size parameterization
    ...


class RegionPooling(nn.Module):
    """Turn the part of a feature map inside each region into a fixed HxW feature."""
    def __init__(self, output_h, output_w, spatial_scale):
        super().__init__()
        self.output_h, self.output_w = output_h, output_w
        self.spatial_scale = spatial_scale

    def forward(self, feature_map, rois):
        # TODO: for each roi, fixed-size pooling of the corresponding map region
        raise NotImplementedError
    # TODO: a backward that routes gradients to the pooled input locations


class Detector(nn.Module):
    """Whole image + proposals -> per-region class scores and box refinements."""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone          # conv trunk from a pretrained ImageNet net
        self.region_pool = None           # TODO: fixed-size region feature
        self.head = None                  # TODO: fc layers feeding the prediction branches
        self.cls = None                   # TODO: K+1 class scores
        self.box = None                   # TODO: per-class box offsets

    def forward(self, images, rois):
        feat = self.backbone(images)      # whole-image conv feature map, computed once
        # TODO: pool each roi, run the head, branch into class scores + box offsets
        raise NotImplementedError


def detection_loss(cls_scores, box_offsets, labels, box_targets):
    # TODO: single loss jointly training classification and localization
    raise NotImplementedError


def sample_minibatch(dataset, num_images, rois_per_image):
    # TODO: choose how to draw images and regions so all-layer training is efficient
    raise NotImplementedError
```
