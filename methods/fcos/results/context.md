# Context

## Research question

Object detection asks an algorithm to localize and classify every instance in an image with an axis-aligned box. By 2018 essentially every strong detector — Faster R-CNN, SSD, YOLOv2/v3, RetinaNet — was built on **anchor boxes**: a dense grid of pre-defined reference boxes of several scales and aspect ratios, tiled at every feature location, each classified as foreground/background and refined by a regressed offset. Detection accuracy depends on the anchor hyper-parameters (the chosen sizes, aspect ratios, and count); on COCO, tuning these alone moves RetinaNet by up to ~4% AP. To reach high recall, anchors are tiled densely (>180K for an 800-px-shorter-side image with FPN), and matching anchors to ground truth at train time is done by computing IoU between every anchor and every box.

Every other dense-prediction task — semantic segmentation, depth, keypoints — is handled by a fully convolutional network (FCN) that makes a per-pixel prediction and computes all convolutions once over the image. The question taken up here: **can detection be done as per-pixel prediction, FCN-style, with no anchors and no proposals at all?**

## Background

The field rests on two lines: anchor-based detection and the fully-convolutional dense-prediction paradigm.

**Fully convolutional networks for dense prediction.** Long et al. (2015) showed an FCN predicts a label at every output location in a single forward pass, sharing computation across positions; this became the standard for semantic segmentation, depth estimation, and keypoint detection. The appeal is uniformity: one network, per-pixel targets, no region cropping.

**Feature Pyramid Networks (Lin et al. 2017).** FPN builds a multi-scale feature pyramid with a top-down pathway: the backbone produces feature maps C3, C4, C5 (strides 8, 16, 32); a 1×1 lateral conv plus top-down upsampling-and-add produces P3, P4, P5, and extra strided convs produce coarser levels. Different object scales are handled at different pyramid levels. FPN is the mechanism modern detectors use to detect both small and large objects; it is a load-bearing component here, not the contribution.

**Focal loss / RetinaNet (Lin et al. 2017).** RetinaNet is a one-stage anchor detector on FPN. With dense anchors, the foreground/background class distribution is extremely imbalanced, dominated by easy negatives. Focal loss down-weights easy examples: FL(p_t) = −(1−p_t)^γ log(p_t), with γ=2 and an α balancing term, applied to per-anchor binary classifiers. RetinaNet's head is two parallel four-conv towers (one for classification, one for box regression) shared across pyramid levels. Its training/testing protocol is the natural yardstick.

**IoU loss / UnitBox (Yu et al. 2016).** Instead of regressing box coordinates with smooth-L1 on each side independently, UnitBox regresses the four distances (left, top, right, bottom) from a pixel to the box sides and optimizes the IoU between the predicted and ground-truth box directly as a unit, which respects the fact that the four coordinates jointly define one box.

**Per-pixel detection precedents.** DenseBox (Huang et al. 2015) and its descendants (e.g. UnitBox) predict, at each spatial location, a 4D vector (the offsets to the four box sides) plus a class score — the FCN-style per-pixel detection idea. To handle different box sizes, DenseBox crops and resizes training images to a fixed scale, running detection on an image pyramid. These methods were applied in specialized domains such as scene-text and face detection.

**YOLOv1 (Redmon et al. 2016)** is the best-known anchor-free detector: it predicts boxes only at locations near each object's center, on a coarse grid. YOLOv2 added anchors.

## Baselines

**RetinaNet (anchor-based one-stage).** On each FPN level, tile A anchors per location (typically 9: three scales × three aspect ratios). Label an anchor positive if its IoU with some ground-truth box exceeds a high threshold, negative below a low threshold, ignored in between. A shared head of two four-conv towers predicts, per anchor, K class logits (focal-loss binary classifiers) and 4 box-regression deltas relative to the anchor. Test by thresholding scores and running NMS.

**DenseBox-family per-pixel detectors.** At each location regress (l,t,r,b) to the box sides plus a class score, FCN-style, trained on image pyramids.

**YOLOv1.** Coarse grid, boxes predicted near object centers only.

**CornerNet (Law & Deng 2018).** Anchor-free; detects a pair of box *corners* as keypoints and groups corners belonging to the same instance using a learned associative embedding.

## Evaluation settings

- **MS-COCO** detection benchmark: 80 classes. Train on `trainval35k` (115K images); ablate on `minival` (5K); report on `test-dev` (20K, evaluation server). Primary metric: AP averaged over IoU thresholds 0.5:0.95, plus AP50, AP75, and AP at small/medium/large object scales (AP_S/M/L) and average recall (AR).
- **Backbones:** ResNet-50 / ResNet-101 (ImageNet-pretrained), with stronger variants (ResNeXt-101) for the high-end comparison.
- **Training protocol (matching the anchor-based yardstick):** SGD, 90K iterations, base learning rate 0.01 with a mini-batch of 16 images, divided by 10 at 60K and 80K; weight decay 1e-4, momentum 0.9. Input shorter side 800, longer side ≤1333. The detection head uses Group Normalization in its conv towers for stable training.
- **Post-processing:** per-level score threshold (e.g. 0.05) then non-maximum suppression.
- **Diagnostics of interest:** best-possible-recall (the upper bound on recall a sample-assignment can achieve) and the fraction of locations that are inside more than one ground-truth box.

## Code framework

The primitives already exist: a backbone CNN producing C3/C4/C5, an FPN producing a list of pyramid features {P3…P7}, and PyTorch conv/normalization layers. A one-stage detector attaches a **head** that, per pyramid level, runs shared conv towers and emits per-location predictions, plus a **target builder** that turns ground-truth boxes into per-location training targets, a **loss**, and an **inference decoder**. The open slots are what each location should predict, how locations are assigned to ground truth, and how predictions are decoded into boxes.

```python
import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    """Shared head applied to every FPN level. Two conv towers feed
    per-location prediction layers. What the per-location outputs ARE,
    and any per-level output rescaling, is the open design question."""
    def __init__(self, in_channels, num_classes, num_convs=4, num_levels=5):
        super().__init__()
        cls_tower, box_tower = [], []
        for _ in range(num_convs):
            for tower in (cls_tower, box_tower):
                tower += [nn.Conv2d(in_channels, in_channels, 3, padding=1),
                          nn.GroupNorm(32, in_channels), nn.ReLU()]
        self.cls_tower = nn.Sequential(*cls_tower)
        self.box_tower = nn.Sequential(*box_tower)
        # TODO: per-location prediction layers (what does each location output?)
        # TODO: any per-level rescaling of the regression output

    def forward(self, features):
        # features: list of FPN levels P3..P7
        outputs = []
        for level, x in enumerate(features):
            cls_feat = self.cls_tower(x)
            box_feat = self.box_tower(x)
            # TODO: produce per-location predictions from cls_feat, box_feat
            pass
        return outputs

def build_targets(locations, gt_boxes, gt_labels):
    """Map each feature-map location (mapped back to image coords) to a
    training target. How a location is assigned to a box, what regression
    target it gets, and how multi-box / multi-level ambiguity is resolved,
    are all open."""
    pass  # TODO

def detector_loss(predictions, targets):
    """Classification loss over all locations + regression loss over
    positive locations (+ possibly an extra per-location quality term)."""
    pass  # TODO

def decode(predictions, locations):
    """Turn per-location predictions back into image-space boxes + scores."""
    pass  # TODO
```
