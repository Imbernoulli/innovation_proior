## Research question

Given an image, an object detector must output a set of tight bounding boxes, each labelled with an object category. By 2015 the dominant recipe was the *region-based* CNN: first an external algorithm hypothesizes a few thousand candidate regions ("object proposals"), then a CNN classifies each candidate and refines its box. Two successive engineering advances — spatial pyramid pooling and a single shared convolutional pass — had collapsed the cost of the *classification* stage from tens of seconds to a couple hundred milliseconds per image on a GPU. That progress moved the bottleneck somewhere else.

The precise problem: **with the per-region CNN now cheap, the external proposal generator has become the dominant cost of the whole pipeline, and it lives on a different device (CPU) and on a different feature representation (hand-engineered low-level cues) than the detector.** A solution would have to produce high-quality proposals at a cost that is negligible next to the detection network, ideally by *reusing the computation the detector already performs*, while preserving (or improving) detection accuracy. Getting there is what stands between research-grade detection and a system that runs at several frames per second end to end.

## Background

**Region-based detection.** The detection problem is harder than whole-image classification because the number, location, scale, and aspect ratio of objects are unknown. The region-based approach sidesteps an exhaustive search over all boxes by trusting a separate module to nominate a manageable candidate set (typically ~2000 boxes), then spending CNN effort only on those.

**The proposal stage and its cost.** The standard proposer was Selective Search (Uijlings et al., 2013): it over-segments the image into superpixels and greedily merges them using engineered low-level features (color, texture, size, fill) to emit a hierarchy of candidate regions. It is high-recall but slow — about 2 seconds per image in its CPU implementation. EdgeBoxes (Zitnick & Dollár, 2014) offered the best quality/speed tradeoff of the era at ~0.2 s per image by scoring boxes from edge maps. A measured, diagnostic fact of the time: once the detection network had been accelerated, the proposal step alone consumed *as much or more* wall-clock time as the entire detector. Selective Search at 2 s was an order of magnitude slower than the ~0.2 s detector pass; even EdgeBoxes at 0.2 s roughly matched it.

**Shared convolutional features.** A parallel line of work showed that convolutional feature maps need to be computed only *once* per image and then reused for every region. OverFeat (Sermanet et al., 2014) computed conv features over an image pyramid and slid classifiers/regressors across them. SPP (He et al., 2014) added a spatial-pyramid-pooling layer that turns an arbitrary sub-window of a shared feature map into a fixed-length vector, so a region-based detector no longer needs to re-run the convolutions per region. Fully convolutional networks (Long et al., 2015) reframed sliding a small network over a feature map as plain convolution, with weights shared across all spatial positions — naturally translation-invariant up to the network stride.

**Multi-scale handling, the state of the art.** Objects appear at many sizes and shapes, and there were two established ways to cope. (a) *Image / feature pyramids*: resize the image to several scales and recompute features at each scale (used in DPM with HOG, and in CNN detectors). Effective but time-consuming — it multiplies the convolution cost. (b) *Filter pyramids*: run windows / filters of several sizes and aspect ratios over a single feature map (e.g. DPM trains separate 5×7 and 7×5 templates per aspect ratio). Both schemes *enumerate* scales explicitly, paying for each one.

**Bounding-box regression.** R-CNN introduced a now-standard parameterization for refining a candidate box toward a ground-truth box: predict center offsets normalized by the candidate's width/height, and width/height corrections in log space. This makes the regression targets roughly scale- and translation-invariant.

## Baselines

**R-CNN (Girshick et al., 2014).** Take Selective Search proposals; warp each to a fixed size; run a CNN per proposal to get features; classify with per-class SVMs; refine boxes with a regressor. Core gap: it runs the full CNN *separately on every one of ~2000 proposals*, so it is extremely slow, and it is a multi-stage pipeline (CNN features, SVMs, regressors trained separately). Accuracy is bounded by the external proposals; the CNN is essentially a classifier and barely predicts box geometry.

**SPPnet (He et al., 2014).** Compute the conv feature map of the whole image once; for each proposal, use spatial pyramid pooling to extract a fixed-length feature from the corresponding sub-window. This shares conv computation across all proposals, giving a large speedup at test time. Gaps: still a multi-stage training pipeline with separately trained SVMs and regressors; fine-tuning cannot easily propagate through the SPP layer into the conv layers; and it still depends on an external proposer.

**Fast R-CNN (Girshick, 2015).** The clean single-stage *trainer*: one shared conv pass; an RoI-pooling layer (a one-level SPP) maps each proposal to a fixed 7×7 feature; two sibling output heads — a softmax over (K+1) classes and per-class box regressors — trained jointly with a single multi-task loss,

L(p, u, t, v) = L_cls(p, u) + λ·[u ≥ 1]·L_loc(t, v),

where L_cls is log loss on the true class u, L_loc is a smooth-L1 (robust) loss on the 4 regression targets, and the indicator gates the box loss to non-background RoIs. End-to-end fine-tuning back into the conv layers, near-real-time region-wise cost. The remaining gap is the one that defines the present problem: **Fast R-CNN still consumes proposals from an external module, and that module is now the bottleneck — and it shares nothing with the detector's features.**

**OverFeat (Sermanet et al., 2014).** A one-stage, class-specific detector: a fully-connected layer (later recast as convolution) predicts box coordinates and class scores directly from sliding windows over a conv-feature scale pyramid, assuming essentially one object per window location and one aspect ratio per window. Gap: features come from a single-aspect-ratio window over a scale pyramid, and the location+category decision is made in one shot without a downstream stage that re-pools features from the *predicted* region.

**MultiBox (Erhan et al., 2014; Szegedy et al., 2014).** A network whose last fully-connected layer predicts a fixed set of ~800 class-agnostic boxes at once (the centers obtained by k-means over training boxes); these serve as proposals fed to R-CNN. Gaps: the proposal network is applied to a single image crop (or a few large crops), not fully convolutionally; it shares no features with the detector; its 800 learned boxes are *not* translation-invariant (the same object shifted is not guaranteed the same proposal); and its output layer is enormous (millions of parameters), inviting overfitting on small datasets.

## Evaluation settings

The natural yardsticks already existed. **PASCAL VOC 2007** (~5k trainval, ~5k test images, 20 categories) and **VOC 2012** are the primary detection benchmarks; the metric is mean Average Precision (mAP) at IoU 0.5. **MS COCO** (Lin et al., 2014) provides a larger benchmark with an mAP averaged over IoU thresholds from 0.5 to 0.95 as the primary metric, plus mAP@0.5. Proposal generators are sometimes diagnosed with a Recall-vs-IoU curve over the ground-truth boxes as a function of the number of proposals — a loose proxy that is better for diagnosing a proposer than for ranking detectors. Backbone CNNs available as ImageNet-pretrained initializers include the Zeiler–Fergus net (ZF, 5 conv layers) and VGG-16 (Simonyan & Zisserman, 13 conv layers). Standard protocol: rescale each image so its shorter side is 600 px, single scale; report wall-clock timing on a GPU for the conv pass, the proposal step, and the region-wise computation separately.

## Code framework

The pre-existing primitives: a deep-learning library with conv/ReLU/pooling layers and SGD with momentum, an ImageNet-pretrained backbone, an RoI-pooling layer, a smooth-L1 loss, an NMS routine, IoU/overlap computation between box sets, and the R-CNN box parameterization. The detector trainer (Fast R-CNN) already exists and consumes a list of proposals. What does *not* yet exist is anything that produces those proposals from the shared features — that is the empty slot.

```python
import numpy as np

FEAT_STRIDE = 16          # total downsampling of the backbone's last conv map

# ---- existing box utilities (R-CNN parameterization) --------------------
def bbox_transform(ex_rois, gt_rois):
    """Center/log targets t* taking a reference box to a gt box."""
    ...  # standard R-CNN parameterization

def bbox_transform_inv(boxes, deltas):
    """Apply predicted deltas to reference boxes -> predicted boxes."""
    ...

def clip_boxes(boxes, im_shape):
    """Clip boxes to image bounds."""
    ...

def bbox_overlaps(boxes, query_boxes):
    """IoU matrix between two sets of boxes."""
    ...

def nms(dets, thresh):
    """Greedy non-maximum suppression by score."""
    ...

# ---- existing detector trainer ------------------------------------------
class RegionDetector:
    """Shared conv pass -> RoI pool over given proposals -> (cls, box) heads,
    trained with a multi-task (log-loss + smooth-L1) objective. Consumes a
    list of proposals supplied from OUTSIDE."""
    def forward(self, image, proposals): ...
    def multitask_loss(self, cls_pred, box_pred, labels, box_targets): ...

# ---- empty proposal slot -------------------------------------------------
def generate_reference_boxes(*args, **kwargs):
    # TODO: define the fixed reference boxes the proposer regresses from
    pass

class ProposalModule:
    """TODO: a head on top of the shared conv features that emits
    object proposals + scores cheaply, so the detector no longer needs an
    external proposer. Architecture, targets, loss, and how it shares the
    backbone's features are all to be designed."""
    def build_head(self, feature_map):
        # TODO
        pass
    def assign_targets(self, ground_truth):
        # TODO: turn reference boxes + gt into training labels/targets
        pass
    def loss(self, *args):
        # TODO
        pass
    def infer_proposals(self, feature_map):
        # TODO: emit boxes + scores at test time
        pass

def train_with_shared_features(proposer, detector, data):
    # TODO: make ONE set of conv layers serve both the proposer and the
    # detector, even though each task would pull the convs in its own direction
    pass
```
