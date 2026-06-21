# Context

## Research question

Given one RGB image, produce every object instance as a class label and a tight bounding box. The practical yardstick is not only accuracy: the detector has to sit on a speed/accuracy frontier, measured as mean Average Precision and frames per second on a single GPU.

By late 2015, the most accurate detectors share a costly shape. They first generate many candidate regions, then resample pixels or feature-map cells for each candidate, and finally classify and refine each candidate. The fastest high-accuracy model in this family reports about 7 FPS with VGG-style features. The open question is how to build a single-pass fully convolutional detector that covers small objects, large objects, and crowded scenes without a second classification stage.

## Background

R-CNN made proposal-based CNN detection dominant: use Selective Search to propose roughly thousands of regions, crop or warp each region, run a CNN classifier, then refine boxes. SPPnet and Fast R-CNN reduce the repeated CNN cost by computing a convolutional feature map once and pooling each proposal from it. Fast R-CNN also trains classification and bounding-box regression together with a softmax term and a Smooth L1 localization term. The external proposal step remains.

Faster R-CNN removes the hand-engineered proposal generator by adding a Region Proposal Network. The RPN slides a small convolutional predictor over the last shared feature map. At each location it places several anchors with different scales and aspect ratios, predicts objectness, and regresses center/width/height offsets relative to each anchor. Its offset parameterization divides center shifts by anchor size and uses logs for width and height. The proposals are still passed to an RoI-pooled detector head.

MultiBox shows another route: predict a fixed set of prior boxes and confidences, then train by matching priors to ground-truth boxes. Its priors and matching loss are directly relevant, but its outputs are class-agnostic proposals that still need downstream classification.

Single-pass detectors already exist. OverFeat predicts from a top feature map in a sliding-window style. YOLO predicts from a coarse grid with fully connected layers on top of whole-image features, achieving around 45 FPS for the standard model.

Feature maps inside a CNN are not interchangeable. Earlier maps have finer spatial sampling; deeper maps have larger receptive fields and stronger semantics. Dense prediction work such as FCN and hypercolumns uses this fact for localization. Dilated convolution provides a way to keep resolution while increasing receptive field, and VGG-16 is the common ImageNet-pretrained backbone available to the detector.

## Baselines

Fast R-CNN takes external proposals, RoI-pools each proposal from a shared convolutional map, and applies classification plus class-specific box regression. It establishes the Smooth L1 localization term and multi-task detection training.

Faster R-CNN adds the RPN. It uses tiled anchors, convolutional prediction, and center/log-size offset regression, then uses those proposals in a second RoI-pooled classification/regression stage.

MultiBox directly regresses a fixed set of priors and confidences using a matching loss. It supplies the idea that a fixed output set can be supervised by overlap matching.

YOLO uses one network pass over a whole image and predicts a 7 by 7 grid with two boxes per cell and class probabilities. It demonstrates real-time speed at around 45 FPS for the standard model.

OverFeat runs a convolutional classifier and regressor over a top feature map. It is spatially shared and simple.

## Evaluation settings

PASCAL VOC 2007 and 2012 use 20 object classes and mean Average Precision at IoU >= 0.5. Common training regimes are VOC2007 trainval, VOC2007 plus VOC2012 trainval, and variants with COCO pretraining.

COCO uses 80 object classes and is harsher on small objects and localization quality. Its AP metric averages over IoU thresholds from 0.5 to 0.95 and reports AP/AR by object size.

ILSVRC DET has 200 detection classes and tests whether the architecture scales beyond VOC and COCO.

Speed is reported as FPS on a single Nvidia GPU, often with batch size stated. Since candidate resampling can dominate runtime, the evaluation has to include the full detector, not only a classifier head.

Error analysis tools decompose false positives into localization, similar-category, other-category, and background errors, and also break performance down by object size and aspect ratio.

## Code framework

The starting code already has an ImageNet-pretrained VGG-style backbone, IoU computation, Smooth L1, softmax or cross-entropy, the standard center/log-size box-offset parameterization, non-maximum suppression, and an SGD training loop.

The detector head is left open: where the reference boxes live, how many shapes each location owns, which feature maps provide predictions, how class scores and offsets are emitted, how references are matched to ground truth, and how the loss handles the large background imbalance.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def vgg16_truncated(in_channels=3):
    """ImageNet-pretrained VGG-16 convolutional stack, classifier removed."""
    ...


def jaccard(boxes_a, boxes_b):
    """IoU for boxes in (xmin, ymin, xmax, ymax) form."""
    ...


def encode_offsets(matched_gt, ref_boxes):
    """Known center/log-size offset encoding relative to reference boxes."""
    ...


def nms(boxes, scores, overlap, top_k):
    """Greedy per-class non-maximum suppression."""
    ...


class ReferenceBoxes:
    def __call__(self):
        pass


class DetectionHead(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        pass


def match_refs_to_gt(threshold, gt_boxes, gt_labels, ref_boxes):
    pass


class DetectionLoss(nn.Module):
    def forward(self, predictions, targets):
        pass


def train(model, loader, num_classes):
    criterion = DetectionLoss(num_classes)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                          weight_decay=5e-4)
    for images, targets in loader:
        preds = model(images)
        loss_l, loss_c = criterion(preds, targets)
        (loss_l + loss_c).backward()
        opt.step()
        opt.zero_grad()
```
