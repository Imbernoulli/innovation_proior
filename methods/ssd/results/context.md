# Context

## Research question

Given a single RGB image, locate every object of interest with a tight bounding box and a category label. The yardstick the field cares about is the speed/accuracy trade-off: how high a mean Average Precision can a detector reach, and at how many frames per second on commodity GPU hardware. By 2015 the most accurate detectors had converged on a single recipe — hypothesize many candidate boxes, resample pixels or features inside each one, and run a strong classifier per candidate — and that recipe was inherently slow because the per-candidate resampling and classification dominate the budget. The fastest accurate detector of the day ran at roughly 7 FPS; real-time and embedded uses were out of reach.

The concrete goal: build a detector that does **not** resample features per box hypothesis — that emits all boxes and all class scores in a single forward pass of one network — yet matches the accuracy of the resample-and-reclassify detectors. The earlier single-pass attempts were fast but lost a large accuracy gap, with a particular weakness on small objects. So the question is sharper than "make it fast": can a one-shot, fully-convolutional predictor close the accuracy gap, especially across a wide range of object sizes, without ever cropping and re-encoding a region?

## Background

**The dominant detection pipeline.** After R-CNN (Girshick et al. 2014) replaced hand-engineered features with a CNN classifier applied to region proposals, every leading detector was a variant of: propose regions → for each region resample (crop/pool) features → classify and refine the box. The proposals came first from Selective Search (Uijlings et al. 2013), a low-level-feature segmentation/merging heuristic producing ~2000 boxes per image. SPPnet (He et al. 2014) made this affordable by computing the convolutional feature map once and pooling each proposal out of it with a spatial-pyramid pooling layer, instead of re-running the CNN per crop. Fast R-CNN (Girshick 2015) folded the classifier and the box regressor into one network trained end-to-end with a multi-task loss — a softmax classification term plus a smooth-L1 box-regression term — pooling each proposal with RoI pooling. The proposals were still external.

**Anchors and learned proposals.** Two ideas removed the external proposal generator. MultiBox (Erhan et al. 2014; Szegedy et al. 2015) trained a CNN to directly regress a fixed set of class-agnostic box proposals together with a confidence for each, using *prior* boxes obtained by clustering the training boxes, and a matching loss that assigned each ground-truth box to the best-overlapping prior. Faster R-CNN (Ren et al. 2015) introduced the Region Proposal Network: slide a small network over the final convolutional feature map and, at every location, predict objectness and four box offsets for each of *k* **anchor boxes** — reference boxes of several scales and aspect ratios centred on that location. The offsets are predicted relative to the anchor, with a log parameterization for width and height. Both works established that a convolutional net can predict a structured set of boxes by *regressing offsets to a tiled set of reference boxes*, and that you can train this by matching reference boxes to ground truth.

**The single-pass detectors and their limitation.** OverFeat (Sermanet et al. 2013) ran a classifier convolutionally and predicted one box per location of the topmost feature map. YOLO (Redmon et al. 2015) took the whole topmost feature map, passed it through fully-connected layers, and predicted, for each cell of a coarse grid, a couple of boxes and a set of class probabilities. Both predict from a **single, low-resolution feature map**. A single deep feature map has a coarse stride and a large receptive field per cell, so small objects and dense groups of small objects have little or no support — they fall between cells. YOLO's use of fully-connected layers to emit boxes also discards spatial layout and is parameter-heavy, and it regresses box coordinates from whole-image features rather than from a local reference.

**Feature maps carry scale-specific information.** Work on dense prediction had shown that the earlier, higher-resolution convolutional layers retain fine spatial detail useful for precise localization (Long et al. 2015, fully convolutional networks; Hariharan et al. 2015, hypercolumns), while deeper layers carry coarse, semantic, large-receptive-field information; ParseNet (Liu et al. 2016) further showed feature magnitudes differ markedly across layers. Different layers within one network also have different empirical receptive-field sizes (Zhou et al. 2015). This suggests that objects of different sizes are best handled by feature maps at different depths/resolutions, all available inside a single forward pass — an alternative to the classic remedy of running the image at multiple input resolutions (image pyramids, as in SPPnet/OverFeat) and merging results.

**Keeping resolution cheaply.** The à trous (dilated) convolution algorithm (Holschneider et al. 1990), as used in DeepLab (Chen et al. 2015), enlarges a filter's receptive field by inserting holes, letting a network keep a higher-resolution feature map while still seeing a large context, without the cost of more layers or larger filters. This is the tool for turning a classification backbone's fully-connected layers into convolutional ones without collapsing resolution.

**The base network.** VGG-16 (Simonyan & Zisserman 2015) — a stack of 3×3 convolutions — pretrained on ImageNet (Russakovsky et al. 2015), truncated before its classification head, is the standard feature extractor that detectors fine-tune.

## Baselines

**Fast R-CNN (Girshick 2015).** Given external proposals, compute the conv feature map once, RoI-pool each proposal to a fixed size, and feed a head that outputs a softmax over K+1 classes and a class-specific smooth-L1 box regression. Trained end-to-end with the multi-task loss L = L_cls + λ[u≥1]L_loc, where L_loc is smooth-L1 over the four parameterized offsets (t_x,t_y,t_w,t_h) with t_x=(x−x_a)/w_a, t_w=log(w/w_a), etc. Gap left open: still depends on a separate, slow proposal mechanism (Selective Search ~2 s/image); only the post-classification was accelerated.

**Faster R-CNN (Ren et al. 2015).** Replaces Selective Search with a Region Proposal Network. The RPN slides a 3×3 conv over the last shared feature map; at each of the ~W×H locations it places *k*=9 anchors (3 scales × 3 aspect ratios) and predicts, per anchor, 2 objectness scores and 4 offsets, regressed relative to the anchor with the Fast R-CNN log parameterization. Anchors with IoU > 0.7 to a ground-truth box (and the max-overlap anchor per box) are positive; IoU < 0.3 negative. Proposals are then RoI-pooled and reclassified by a Fast R-CNN head; training alternates between the two. Runs ~7 FPS. Gaps left open: (1) all anchors live on a **single** feature map, so one stride and one receptive-field band must cover all object scales; (2) it keeps the second resample-and-reclassify stage, the dominant cost, and the two-network coupling complicates training.

**MultiBox (Erhan et al. 2014; Szegedy et al. 2015).** A CNN regresses a fixed number of bounding-box proposals plus a confidence per box, with the priors obtained by k-means clustering the ground-truth boxes. The training loss matches each ground-truth box to the prior of highest overlap (a bipartite assignment) and sums a confidence (log-loss) term and an L2 localization term over matched priors. Gaps left open: confidences are class-agnostic objectness, so a separate classifier is still needed downstream; priors live on one feature map; the bipartite "best-overlap only" matching forces the net to commit to a single prior per object.

**YOLO (Redmon et al. 2015).** One network maps the whole image to an S×S grid; each cell predicts B boxes (x,y,w,h, confidence) and one shared set of C class probabilities, via fully-connected layers on top of the topmost feature map. No proposals, very fast (~45 FPS) but ~63 mAP. Gaps left open: a single coarse grid and a single feature scale, hence poor recall on small and clustered objects; FC box prediction discards the spatial structure of the feature map and predicts coordinates from global features; only one feature resolution.

**OverFeat (Sermanet et al. 2013).** A fully-convolutional sliding-window detector that predicts a box from each location of the topmost feature map after scoring object categories. Single scale, one box per location.

## Evaluation settings

- **PASCAL VOC 2007 / 2012**: 20 object categories; the standard `trainval`/`test` splits (VOC2007 `test` = 4952 images). Metric: mean Average Precision at IoU ≥ 0.5. Common training-data regimes are "07" (VOC2007 trainval), "07+12" (union of VOC2007 and VOC2012 trainval), and adding COCO.
- **COCO**: 80 categories, smaller objects on average, denser scenes. Metrics: AP averaged over IoU 0.5:0.95, plus AP@0.5 and AP@0.75, and AP/AR broken down by object area (small/medium/large). Standard `trainval35k`/`test-dev` protocol.
- **ILSVRC DET**: 200 categories, large-scale detection.
- **Speed**: frames per second measured on a single GPU (Titan X, cuDNN), reported alongside mAP so the speed/accuracy trade-off is explicit.
- **Diagnostic tooling**: the Hoiem et al. (2012) detector-error analysis (decomposing false positives into localization / similar-category / other / background, and sensitivity to object area and aspect ratio) is the standard way to attribute where a detector's errors come from.

## Code framework

The pieces below already exist before the method: an ImageNet-pretrained VGG-16 backbone, the smooth-L1 and softmax/cross-entropy losses, IoU computation, the offset (cx,cy,w,h, log-encoded) box parameterization from the proposal/regression literature, non-maximum suppression, and an SGD training loop. What does **not** yet exist is the detection head itself: how reference boxes are laid out, which feature map(s) feed predictions, how predictions are produced, how reference boxes are matched to ground truth, and the exact training objective. Those are the empty slots.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# ---- existing primitives ----
def vgg16_truncated(in_channels=3):
    """ImageNet-pretrained VGG-16 conv stack, classifier head removed."""
    ...  # standard 3x3 conv/maxpool stack -> returns nn.ModuleList

def jaccard(boxes_a, boxes_b):
    """IoU between two sets of boxes in (xmin,ymin,xmax,ymax) form."""
    ...

def encode_offsets(matched_gt, ref_boxes):
    """(cx,cy,w,h) offset parameterization with log on (w,h),
    relative to a reference box. Known from the proposal/regression literature."""
    ...

def nms(boxes, scores, overlap, top_k):
    """Greedy per-class non-maximum suppression."""
    ...


# ---- the slots the method will fill ----

class ReferenceBoxes:
    """TODO: where do reference boxes live and what shapes do they take?
    Lay out the tiled set of reference boxes the detector predicts against:
    which feature map(s), how many per location, which scales/aspect ratios.
    Returns reference boxes in (cx,cy,w,h) form."""
    def __call__(self):
        pass  # TODO

class DetectionHead(nn.Module):
    """TODO: the detector built on top of the backbone.
    Decide which feature map(s) to attach predictors to, and how a predictor
    turns a feature map into per-reference-box class scores + box offsets."""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # TODO: predictor(s); reference-box layout
    def forward(self, x):
        pass  # TODO: -> (loc_preds, conf_preds, reference_boxes)

def match_refs_to_gt(threshold, gt_boxes, gt_labels, ref_boxes):
    """TODO: assign reference boxes to ground-truth boxes, decide positives
    vs. negatives, and produce encoded localization + classification targets."""
    pass  # TODO

class DetectionLoss(nn.Module):
    """TODO: the training objective combining a classification term and a
    localization term over matched reference boxes, with whatever sampling
    is needed to cope with the box imbalance."""
    def forward(self, predictions, targets):
        pass  # TODO

# ---- existing training loop ----
def train(model, loader, num_classes):
    criterion = DetectionLoss(num_classes)          # TODO body above
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                          weight_decay=5e-4)
    for images, targets in loader:
        preds = model(images)
        loss_l, loss_c = criterion(preds, targets)
        (loss_l + loss_c).backward()
        opt.step(); opt.zero_grad()
```
