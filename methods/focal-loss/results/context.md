# Context: accuracy of dense, one-stage object detectors

## Research question

Modern object detectors come in two families. The accurate ones run in two stages: a first stage proposes a sparse set of candidate object regions and a second stage classifies and refines each proposal. The fast ones run in a single stage: a fully convolutional network sweeps a regular, dense grid of candidate boxes (positions × scales × aspect ratios) and predicts a class and a box offset at every one of them, in one shot, with no proposal step.

The dense, single-stage family is simpler and faster. On the COCO detection benchmark it trails the two-stage family by roughly 10–40% relative AP. The question: can a single-stage detector that classifies a dense grid of boxes be made to match the accuracy of a two-stage proposal-driven detector while keeping its speed?

## Background

**The two families.** Two-stage detection has been the dominant paradigm since R-CNN. The first stage (selective search, EdgeBoxes, or a learned Region Proposal Network) reduces the near-infinite space of possible boxes to one or two thousand candidates that are *likely* to be objects; the second stage runs a classifier on those. Single-stage detectors skip the proposal step and predict directly over a fixed, dense sampling of the image — typically on the order of 10^4–10^5 candidate locations per image once positions, scales, and aspect ratios are enumerated. Prior speed/accuracy studies found the gap persists even when the two-stage detector is handicapped (smaller input, fewer proposals) so that compute is comparable.

**Foreground–background class imbalance.** A dense detector evaluates 10^4–10^5 candidate locations per image, but only a few of them contain an object; the majority are background, on the order of 1:1000 foreground-to-background. The same imbalance is present in classic dense detectors (boosted cascades, deformable part models) and in modern single-stage networks. Most locations are easy negatives — clearly-not-an-object patches the model already classifies correctly. This imbalance is a defining property of the dense-detection setting.

**How two-stage detectors handle imbalance.** The proposal stage acts as an imbalance filter: it discards the majority of easy background before the classifier sees it, leaving ~1–2k candidates that are disproportionately real objects. The second stage then trains on minibatches built with biased sampling — e.g. a fixed 1:3 positive-to-negative ratio — which acts as a class-balancing factor. The two-stage cascade addresses imbalance architecturally and by sampling, before the loss is computed.

**The cross-entropy loss.** For binary classification with label y ∈ {±1} and predicted probability p for the y=1 class, cross entropy is CE = −log p when y=1 and −log(1−p) otherwise. Writing p_t for the probability of the *correct* class (p_t = p if y=1, else 1−p), this is CE(p_t) = −log p_t. A confidently-correct example (p_t ≫ 0.5) still incurs a loss of −log(0.9) ≈ 0.1 nats.

**Robust losses.** A separate line of work designs robust losses (e.g. Huber loss) that *down-weight outliers* — examples with large error, i.e. hard examples — to keep them from dominating regression.

## Baselines

**R-CNN family (Girshick 2014; Fast R-CNN, Girshick 2015; Faster R-CNN, Ren et al. 2015).** Two-stage, proposal-driven. R-CNN warps each region proposal and runs a CNN classifier; Fast R-CNN shares convolution and adds RoI pooling plus a smooth-L1 box-regression loss; Faster R-CNN folds proposal generation into a Region Proposal Network (RPN) sharing features with the detector. The RPN introduces *anchors* — translation-invariant reference boxes of several scales and aspect ratios tiled at every spatial location — each classified object/not-object and regressed toward a ground-truth box. Core to all of them is the cascade: the proposal stage filters out most background, so the classifier trains on a roughly balanced candidate set.

**Online hard example mining — OHEM (Shrivastava et al. 2016).** Improves two-stage training by building each minibatch from the highest-loss examples: score every candidate by its loss, apply non-maximum suppression, then take the top-loss ones (optionally enforcing a 1:3 positive:negative ratio, as in single-stage hard-negative mining). It emphasizes misclassified examples and is tuned by batch size, NMS threshold, and ratio.

**Single-stage dense detectors (SSD, Liu et al. 2016; YOLO, Redmon et al. 2016/2017).** Fully convolutional, predicting class and box at a dense grid in one pass. To cope with imbalance, SSD applies hard-negative mining keeping a fixed neg:pos ratio per minibatch.

**Classic dense detectors (Viola–Jones boosted cascade 2001; DPM, Felzenszwalb et al. 2010).** Sliding-window classifiers over a dense grid. They confront the same imbalance and address it with bootstrapping / hard-negative mining (Sung & Poggio 1994): iteratively grow the negative set with currently-misclassified background.

## Evaluation settings

The natural yardstick is the COCO detection benchmark (Lin et al. 2014). Training uses the standard `trainval35k` split (the 80k `train` images plus a 35k subset of `val`); ablation and sensitivity studies report on the held-out `minival` (the remaining 5k `val` images); final numbers use `test-dev`, whose labels are private and scored on the evaluation server. The metric is COCO average precision (AP) averaged over IoU thresholds from 0.5 to 0.95, with AP_50 and AP_75 at fixed thresholds and AP_S / AP_M / AP_L broken out by object size. Speed is reported as per-image inference time on a single GPU, so methods are placed on a speed–accuracy plane. ImageNet-1k pretrained backbones (ResNet-50, ResNet-101) are the standard starting weights. Box regression uses the standard R-CNN box parameterization with a smooth-L1 loss.

## Code framework

The available primitives are a convolutional backbone with a multi-scale feature pyramid (a top-down/lateral construction over ResNet stages), an anchor generator, a smooth-L1 box-regression loss, and an SGD training loop. What remains open is the dense head attached to each pyramid level, how it is initialized, the classification loss applied to its outputs, and how that loss is normalized.

```python
import math
import torch
from torch import nn
from torch.nn import functional as F

# backbone_fpn(image) -> list of feature maps, one per pyramid level (C channels each)
# anchor_generator(features) -> reference boxes per level
# smooth_l1_loss(pred_deltas, gt_deltas, reduction="sum") -> box regression loss
# match_anchors_to_gt(anchors, gt) -> per-anchor label in {bg, ignore, class_k} + matched box
# encode_boxes(anchors, gt_boxes) -> box-regression targets


class DenseHead(nn.Module):
    """Shared per-level dense head for class scores and box offsets."""
    def __init__(self, in_channels, num_anchors, num_classes, conv_dims=None, initial_prob=None):
        super().__init__()
        # TODO: build the class-score subnet from small convolutional blocks
        # TODO: build the box-offset subnet from small convolutional blocks
        # TODO: build the final class-score and box-offset convolutions
        # TODO: initialize the head
        pass

    def forward(self, features):
        raise NotImplementedError


def dense_binary_loss(inputs, targets, loss_params, reduction="sum"):
    """Per-entry binary classification loss before dense-anchor normalization."""
    # TODO: choose the loss shape for dense class imbalance
    raise NotImplementedError


class PositiveAnchorNormalizer:
    def __init__(self, momentum=100):
        self.momentum = momentum
        self.value = None

    def update(self, num_pos):
        # TODO: choose how this count normalizes the losses
        raise NotImplementedError


def classification_loss(pred_logits, gt_labels, num_classes, loss_normalizer, loss_params):
    """Loss on the dense per-anchor class predictions, summed over ALL anchors in
    the image and normalized. Also returns the shared normalizer and positive mask."""
    # TODO: turn gt_labels into per-class binary targets (drop the background column)
    # TODO: choose the per-anchor classification loss under dense imbalance
    # TODO: choose the normalization factor
    raise NotImplementedError


def detector_losses(anchors, pred_logits, pred_deltas, gt_labels, gt_boxes,
                    num_classes, loss_normalizer, loss_params):
    loss_cls, normalizer_value, pos_mask = classification_loss(
        pred_logits, gt_labels, num_classes, loss_normalizer, loss_params
    )
    deltas = torch.cat(pred_deltas, dim=1) if isinstance(pred_deltas, (list, tuple)) else pred_deltas
    target_deltas = encode_boxes(anchors, gt_boxes)
    # TODO: apply the box regression loss only on foreground anchors
    if pos_mask.any():
        loss_box = smooth_l1_loss(deltas[pos_mask], target_deltas[pos_mask], reduction="sum")
    else:
        loss_box = deltas.sum() * 0
    return {"loss_cls": loss_cls, "loss_box_reg": loss_box / normalizer_value}
```
