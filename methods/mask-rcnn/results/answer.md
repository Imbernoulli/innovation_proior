# Mask R-CNN

## Problem

Instance segmentation: detect every object in an image, classify it, localize it with a box, and produce
a pixel-accurate mask for *each individual instance* (so overlapping same-class objects get separate
masks). The aim is a single framework that is as simple, fast, and general as the two-stage detectors are
for boxes and FCNs are for semantic segmentation.

## Key idea

Take the two-stage region detector (stage 1: a Region Proposal Network; stage 2: per-region class + box
heads) and add a **third sibling branch** to stage 2 that outputs a mask, computed **in parallel** with
classification and box regression. Three design choices make it work:

1. **Decoupled, per-class binary masks.** The mask branch outputs K masks of size m×m (one per class),
   applies a **per-pixel sigmoid**, and is trained with **binary cross-entropy on the ground-truth-class
   channel only**. Classes do not compete (no per-pixel softmax). The existing classification branch alone
   decides the label and hence which mask to read out. This decoupling avoids the overlapping-instance
   artifacts of methods that entangle mask and class prediction.
2. **Fully convolutional mask head.** The mask is a *spatial* output, so it is produced by convolutions
   that preserve the m×m layout (not by an fc layer that collapses it to a vector).
3. **RoIAlign.** The standard region-feature extractor (RoIPool) quantizes coordinates twice — mapping the
   region to the feature grid by rounding (`[x/16]`), and rounding bin boundaries — which misaligns the
   feature with the region by up to half a bin. This is harmless for translation-robust classification but
   destroys pixel-to-pixel masks. **RoIAlign** removes all quantization: it uses continuous coordinates
   (`x/16`), samples a few points per bin by **bilinear interpolation**, and aggregates (avg/max). Proper
   alignment — not bilinear sampling per se — is the active ingredient (a quantize-then-bilinear variant,
   RoIWarp, performs like RoIPool); the gain grows with feature stride and is largest at strict IoU.

## Final objective and algorithm

Multi-task loss on each sampled RoI (equal weights):

  **L = L_cls + L_box + L_mask**

- `L_cls`: softmax cross-entropy over K+1 classes (from the classification sibling).
- `L_box`: smooth-L1 on the ground-truth-class box regression outputs.
- `L_mask`: average binary cross-entropy over the m² pixels of the **ground-truth-class** mask channel k,
  with a per-pixel sigmoid:
  `L_mask = −(1/m²) Σ_{i,j} [ y_{ij} log σ(z^k_{ij}) + (1 − y_{ij}) log(1 − σ(z^k_{ij})) ]`.
  Defined **only on positive RoIs** (IoU ≥ 0.5 with a ground truth); other K−1 channels get no gradient.

RoIAlign sample-point coordinate inside bin (p, q) of an m×m grid over a region with feature-space
top-left (x0, y0) and size (w, h), with G samples per axis:
  `y = y0 + q·(h/m) + (iy + 0.5)·(h/m)/G`,  `x = x0 + p·(w/m) + (ix + 0.5)·(w/m)/G`,
each read by bilinear interpolation from its 4 neighboring grid points; the G² reads in a bin are averaged.

**Training:** stage 1 (RPN) unchanged; stage 2 samples RoIs at 1:3 positive:negative; mask branch runs on
positive proposals; targets are GT masks resampled to m×m with RoIAlign.
**Inference:** run class+box branches, apply NMS, keep the top ~100 detections; run the mask branch on
those boxes; read out only the predicted-class mask; resize m×m to the box and threshold at 0.5.

**Extension to keypoints:** model each of K keypoints as a one-hot m×m mask (a single foreground pixel) and
train with an m²-way **softmax** cross-entropy (the answer is exactly one pixel, so pixels should compete),
at higher output resolution (~56×56). Same per-region spatial branch, loss matched to the target.

## Code

Grounded in the standard torchvision implementation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, MultiScaleRoIAlign


# ---- RoIAlign: aligned region-feature extraction (no quantization, bilinear) ----
# output_size 14 for masks (deconv -> 28), 7 for the box head; sampling_ratio=2 points/axis.
mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"],
                                   output_size=14, sampling_ratio=2)
box_roi_pool  = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"],
                                   output_size=7,  sampling_ratio=2)


# ---- Box head: parallel class + per-class box siblings ----
class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return F.relu(self.fc7(F.relu(self.fc6(x))))


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.cls_score(x), self.bbox_pred(x)


# ---- Mask head: fully convolutional, per-class masks ----
class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers=(256, 256, 256, 256), dilation=1):
        d, blocks, nf = dilation, [], in_channels
        for f in layers:
            blocks += [nn.Conv2d(nf, f, 3, 1, padding=d, dilation=d), nn.ReLU(inplace=True)]
            nf = f
        super().__init__(*blocks)


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0),  # upsample 14 -> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, num_classes, 1, 1, 0),           # K class-specific masks
        )


# ---- Losses ----
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, 0)
    regression_targets = torch.cat(regression_targets, 0)
    cls_loss = F.cross_entropy(class_logits, labels)
    pos = torch.where(labels > 0)[0]
    labels_pos = labels[pos]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    box_loss = F.smooth_l1_loss(box_regression[pos, labels_pos],
                                regression_targets[pos], beta=1 / 9, reduction="sum")
    return cls_loss, box_loss / labels.numel()


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    rois = torch.cat([matched_idxs.to(boxes)[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, matched_idxs):
    M = mask_logits.shape[-1]
    labels = torch.cat([gl[idx] for gl, idx in zip(gt_labels, matched_idxs)], 0)
    targets = torch.cat([project_masks_on_boxes(m, p, i, M)
                         for m, p, i in zip(gt_masks, proposals, matched_idxs)], 0)
    if targets.numel() == 0:
        return mask_logits.sum() * 0
    idx = torch.arange(labels.shape[0], device=labels.device)
    # per-pixel sigmoid (via BCE-with-logits) on the GT-class channel only -> no class competition
    return F.binary_cross_entropy_with_logits(mask_logits[idx, labels], targets)


def maskrcnn_inference(mask_logits, pred_labels):
    probs = mask_logits.sigmoid()
    n = probs.shape[0]
    labels = torch.cat(pred_labels)
    idx = torch.arange(n, device=labels.device)
    probs = probs[idx, labels][:, None]               # keep only the predicted-class mask
    return probs.split([len(l) for l in pred_labels], 0)


# ---- Stage-2 head: three parallel siblings; multi-task loss L = L_cls + L_box + L_mask ----
class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_head, box_predictor,
                 mask_roi_pool, mask_head, mask_predictor):
        super().__init__()
        self.box_roi_pool, self.box_head, self.box_predictor = box_roi_pool, box_head, box_predictor
        self.mask_roi_pool, self.mask_head, self.mask_predictor = mask_roi_pool, mask_head, mask_predictor

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, reg_targets = \
                self.select_training_samples(proposals, targets)   # 1:3 pos:neg, IoU>=0.5 -> positive

        box_feat = self.box_roi_pool(features, proposals, image_shapes)        # aligned 7x7
        cls_logits, box_reg = self.box_predictor(self.box_head(box_feat))

        result, losses = [], {}
        if self.training:
            lc, lb = fastrcnn_loss(cls_logits, box_reg, labels, reg_targets)
            losses = {"loss_classifier": lc, "loss_box_reg": lb}
        else:
            boxes, scores, lbls = self.postprocess_detections(                 # NMS -> top detections
                cls_logits, box_reg, proposals, image_shapes)
            result = [{"boxes": b, "labels": l, "scores": s}
                      for b, l, s in zip(boxes, lbls, scores)]

        if self.training:
            mask_props, pos_idx = [], []
            for i in range(len(proposals)):                                    # masks on positives only
                pos = torch.where(labels[i] > 0)[0]
                mask_props.append(proposals[i][pos]); pos_idx.append(matched_idxs[i][pos])
        else:
            mask_props = [r["boxes"] for r in result]                          # masks on final detections

        mask_feat = self.mask_roi_pool(features, mask_props, image_shapes)      # aligned 14x14
        mask_logits = self.mask_predictor(self.mask_head(mask_feat))

        if self.training:
            losses["loss_mask"] = maskrcnn_loss(
                mask_logits, mask_props,
                [t["masks"] for t in targets], [t["labels"] for t in targets], pos_idx)
        else:
            for m, r in zip(maskrcnn_inference(mask_logits, [r["labels"] for r in result]), result):
                r["masks"] = m                                                 # resize to box + threshold 0.5 downstream
        return result, losses
```

The full system pairs this stage-2 head with a backbone (ResNet / ResNeXt, optionally with a feature
pyramid) and an unchanged RPN stage 1; training uses SGD with the standard detection schedule.
