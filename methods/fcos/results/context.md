# Context

## Research question

Object detection asks an algorithm to localize and classify every instance in an image with an axis-aligned box. By 2018 essentially every strong detector — Faster R-CNN, SSD, YOLOv2/v3, RetinaNet — was built on **anchor boxes**: a dense grid of pre-defined reference boxes of several scales and aspect ratios, tiled at every feature location, each classified as foreground/background and refined by a regressed offset. This had become the de facto recipe, and it works, but it drags in a cluster of problems that all trace back to the anchors themselves. The detection accuracy is highly sensitive to the anchor hyper-parameters (the chosen sizes, aspect ratios, and count) — on COCO, tuning these alone moves RetinaNet by up to ~4% AP. Because anchor shapes are fixed, a detector struggles with object shapes outside the designed range, and the anchors must be redesigned for each new task or domain. To hit high recall, anchors must be tiled extremely densely (>180K for an 800-px-shorter-side image with FPN), almost all of them negative, which aggravates the foreground/background imbalance. And matching anchors to ground truth at train time requires computing IoU between every anchor and every box.

Every other dense-prediction task — semantic segmentation, depth, keypoints — is solved cleanly by a fully convolutional network (FCN) that makes a per-pixel prediction and computes all convolutions once over the image. Detection is the lone holdout, kept apart from that neat per-pixel paradigm essentially because of the anchor machinery. The precise question: **can detection be done as per-pixel prediction, FCN-style, with no anchors and no proposals at all — and still match or beat anchor-based detectors?** A solution would have to recover the recall that dense anchors buy, resolve the ambiguity that arises when one location sits inside several overlapping ground-truth boxes, and somehow suppress the flood of poorly-localized boxes that a naive per-pixel regressor produces — all without reintroducing anchor hyper-parameters.

## Background

The field rests on two lines: anchor-based detection and the fully-convolutional dense-prediction paradigm.

**Fully convolutional networks for dense prediction.** Long et al. (2015) showed an FCN predicts a label at every output location in a single forward pass, sharing computation across positions; this became the standard for semantic segmentation, depth estimation, and keypoint detection. The appeal is uniformity: one network, per-pixel targets, no region cropping. Detection is the task that did not fit, because a box is not a per-pixel label.

**Feature Pyramid Networks (Lin et al. 2017).** FPN builds a multi-scale feature pyramid with a top-down pathway: the backbone produces feature maps C3, C4, C5 (strides 8, 16, 32); a 1×1 lateral conv plus top-down upsampling-and-add produces P3, P4, P5, and extra strided convs produce coarser levels. Different object scales are handled at different pyramid levels. FPN is the mechanism every modern detector uses to detect both small and large objects; it is a load-bearing component here, not the contribution.

**Focal loss / RetinaNet (Lin et al. 2017).** RetinaNet is a one-stage anchor detector on FPN whose central insight is that the extreme foreground/background imbalance from dense anchors swamps the cross-entropy loss with easy negatives. Focal loss down-weights easy examples: FL(p_t) = −(1−p_t)^γ log(p_t), with γ=2 and an α balancing term, applied to per-anchor binary classifiers. RetinaNet's head is two parallel four-conv towers (one for classification, one for box regression) shared across pyramid levels. It is the anchor-based detector to beat, and its training/testing protocol is the natural yardstick.

**IoU loss / UnitBox (Yu et al. 2016).** Instead of regressing box coordinates with smooth-L1 on each side independently, UnitBox regresses the four distances (left, top, right, bottom) from a pixel to the box sides and optimizes the IoU between the predicted and ground-truth box directly as a unit, which respects the fact that the four coordinates jointly define one box. This is the natural regression loss for a per-pixel distance parameterization.

**Diagnostic facts about per-pixel detection that were already known.** DenseBox (Huang et al. 2015) and its descendants (e.g. UnitBox) already predicted, at each spatial location, a 4D vector (the offsets to the four box sides) plus a class score — exactly the FCN-style per-pixel detection idea. But these methods were considered unsuitable for *generic* detection for two empirically-observed reasons. (1) To handle different box sizes, DenseBox cropped and resized training images to a fixed scale, forcing detection on an image pyramid — which violates the FCN philosophy of computing all convolutions once. (2) When ground-truth boxes overlap, a location inside several of them has no well-defined regression target — an intractable ambiguity that was believed to break per-pixel detection on crowded generic scenes. As a result these methods were confined to specialized domains (scene-text, face detection), where overlap is rare. A separate concern was recall: with a large output stride, it seemed a per-pixel detector might be unable to recall objects that no feature location lands inside (the best-possible-recall worry).

**YOLOv1 (Redmon et al. 2016)** is the best-known anchor-free detector: it predicts boxes only at locations near each object's center, on a coarse grid. Using only near-center points was meant to yield higher-quality boxes, but it caps recall — which is why YOLOv2 added anchors. This is the key clue that *which* foreground locations a detector trains on, and how many, trades off recall against box quality.

## Baselines

**RetinaNet (anchor-based one-stage).** On each FPN level, tile A anchors per location (typically 9: three scales × three aspect ratios). Label an anchor positive if its IoU with some ground-truth box exceeds a high threshold, negative below a low threshold, ignored in between. A shared head of two four-conv towers predicts, per anchor, K class logits (focal-loss binary classifiers) and 4 box-regression deltas relative to the anchor. Test by thresholding scores and running NMS. Gap: the anchor scales/ratios/count are hand-tuned hyper-parameters that strongly affect AP and must be redesigned per task; dense anchors create severe imbalance and require IoU matching; the output has 9× more variables per location than strictly necessary.

**DenseBox-family per-pixel detectors.** At each location regress (l,t,r,b) to the box sides plus a class score, FCN-style. Gap: train on image pyramids (no shared computation across scales) and break on overlapping generic objects because of the unresolved regression-target ambiguity; considered low-recall and domain-specific.

**YOLOv1.** Coarse grid, boxes predicted near object centers only. Gap: low recall from restricting to near-center locations, which pushed the line back toward anchors.

**CornerNet (Law & Deng 2018).** Anchor-free, but detects a pair of box *corners* as keypoints and then must *group* corners belonging to the same instance using a learned associative embedding. Gap: the corner-grouping post-processing is complicated and slow, and requires special network design — anchor-free but not simple.

## Evaluation settings

- **MS-COCO** detection benchmark: 80 classes. Train on `trainval35k` (115K images); ablate on `minival` (5K); report on `test-dev` (20K, evaluation server). Primary metric: AP averaged over IoU thresholds 0.5:0.95, plus AP50, AP75, and AP at small/medium/large object scales (AP_S/M/L) and average recall (AR).
- **Backbones:** ResNet-50 / ResNet-101 (ImageNet-pretrained), with stronger variants (ResNeXt-101) for the high-end comparison.
- **Training protocol (matching the anchor-based yardstick):** SGD, 90K iterations, base learning rate 0.01 with a mini-batch of 16 images, divided by 10 at 60K and 80K; weight decay 1e-4, momentum 0.9. Input shorter side 800, longer side ≤1333. The detection head uses Group Normalization in its conv towers for stable training.
- **Post-processing:** per-level score threshold (e.g. 0.05) then non-maximum suppression.
- **Diagnostics of interest** (pre-method, about the per-pixel idea itself): best-possible-recall (the upper bound on recall a sample-assignment can achieve) and the fraction of locations that are ambiguous (inside more than one ground-truth box).

## Code framework

The primitives already exist: a backbone CNN producing C3/C4/C5, an FPN producing a list of pyramid features {P3…P7}, and PyTorch conv/normalization layers. A one-stage detector attaches a **head** that, per pyramid level, runs shared conv towers and emits per-location predictions, plus a **target builder** that turns ground-truth boxes into per-location training targets, a **loss**, and an **inference decoder**. What is unknown — the open slots — is exactly what each location should predict, how locations are assigned to ground truth, and how predictions are decoded into boxes.

```python
import torch
import torch.nn as nn

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float))
    def forward(self, x):
        return x * self.scale

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
