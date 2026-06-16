# FCOS: Fully Convolutional One-Stage Object Detection

## Problem

Anchor-based detectors tile a dense grid of pre-defined reference boxes, which forces hand-tuned anchor hyper-parameters (scales, ratios, count) that strongly affect accuracy, creates extreme foreground/background imbalance, and requires IoU matching at train time. FCOS removes anchors and proposals entirely and solves detection as per-pixel prediction, FCN-style.

## Key idea

Treat every feature-map location as a training sample (like a pixel in semantic segmentation). For a location inside a ground-truth box, regress the four distances to the box sides and classify it.

- **Regression target** at location (x,y) inside box (x0,y0,x1,y1): `l*=x−x0, t*=y−y0, r*=x1−x, b*=y1−y`. Since these are positive, the regression branch outputs `exp(s_i·x)` with a per-FPN-level trainable scalar `s_i`. Decode boxes by inverting these four equations.
- **Multi-level FPN routing resolves overlap ambiguity.** Use levels P3–P7 (strides 8,16,32,64,128). A location regresses on level i only if `max(l*,t*,r*,b*)` falls in that level's range; ranges m₂..m₇ = 0, 64, 128, 256, 512, ∞. Because overlapping objects usually differ greatly in size, routing by size separates them so each location is, on its level, almost always inside one box. Residual ties go to the **minimal-area** box. This also yields high best-possible-recall, since every interior location is positive.
- **Center-ness branch** suppresses low-quality boxes from off-center locations without any hyper-parameter:

  `centerness* = sqrt( (min(l*,r*)/max(l*,r*)) · (min(t*,b*)/max(t*,b*)) )` ∈ [0,1].

  A single conv predicts it (trained with BCE); the square root slows the decay so mildly off-center locations keep a graded signal. At inference the **final score = class probability × center-ness**, knocking down off-center boxes so NMS removes them.

## Final objective

```
L = (1/N_pos) Σ_{x,y} L_cls(p_{x,y}, c*_{x,y})
  + (λ/N_pos) Σ_{x,y} 1{c*_{x,y}>0} L_reg(t_{x,y}, t*_{x,y})
  + (1/N_pos) Σ_{x,y} 1{c*_{x,y}>0} BCE(centerness)
```
with `L_cls` = focal loss (γ=2, α=0.25), `L_reg` = IoU loss (UnitBox), λ=1, N_pos the number of positive locations. The shared head has two four-conv towers (with GroupNorm) feeding class/box/center-ness prediction layers; outputs are 9× fewer per location than a 9-anchor detector. Post-processing is per-level score threshold (0.05) then NMS.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float))
    def forward(self, x):
        return x * self.scale

class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_classes=80, num_convs=4, num_levels=5):
        super().__init__()
        cls_tower, box_tower = [], []
        for _ in range(num_convs):
            for tower in (cls_tower, box_tower):
                tower += [nn.Conv2d(in_channels, in_channels, 3, padding=1),
                          nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)]
        self.cls_tower = nn.Sequential(*cls_tower)
        self.box_tower = nn.Sequential(*box_tower)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred  = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

    def forward(self, features):           # features: P3..P7
        logits, bbox_reg, ctrness = [], [], []
        for level, x in enumerate(features):
            cls_feat = self.cls_tower(x)
            box_feat = self.box_tower(x)
            logits.append(self.cls_logits(cls_feat))
            ctrness.append(self.centerness(cls_feat))
            bbox_reg.append(torch.exp(self.scales[level](self.bbox_pred(box_feat))))
        return logits, bbox_reg, ctrness

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(0, w * stride, stride, device=device) + stride // 2
    shifts_y = torch.arange(0, h * stride, stride, device=device) + stride // 2
    ys, xs = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    return torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=1)

def centerness_target(reg_targets):        # (...,4) = l*,t*,r*,b*
    l, t, r, b = reg_targets.unbind(-1)
    lr = torch.stack([l, r], -1); tb = torch.stack([t, b], -1)
    return torch.sqrt((lr.min(-1).values / lr.max(-1).values.clamp(min=1e-6)) *
                      (tb.min(-1).values / tb.max(-1).values.clamp(min=1e-6)))

SIZE_RANGES = [(0, 64), (64, 128), (128, 256), (256, 512), (512, float("inf"))]

def assign_targets(locations_per_level, gt_boxes, gt_labels):
    """FCOS assignment: inside a box, within this level's regression range,
    and minimal-area ground truth if more than one valid box remains."""
    cls_targets, reg_targets = [], []
    if gt_boxes.numel() == 0:
        for locs in locations_per_level:
            cls_targets.append(torch.zeros(locs.size(0), dtype=torch.long, device=locs.device))
            reg_targets.append(locs.new_zeros((locs.size(0), 4)))
        return torch.cat(cls_targets), torch.cat(reg_targets)

    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    for locs, (lower, upper) in zip(locations_per_level, SIZE_RANGES):
        xs, ys = locs[:, 0], locs[:, 1]
        reg = torch.stack((
            xs[:, None] - gt_boxes[:, 0],
            ys[:, None] - gt_boxes[:, 1],
            gt_boxes[:, 2] - xs[:, None],
            gt_boxes[:, 3] - ys[:, None],
        ), dim=-1)
        inside_box = reg.min(dim=-1).values > 0
        max_reg = reg.max(dim=-1).values
        in_level = (max_reg >= lower) & (max_reg <= upper)

        candidate_areas = areas.expand(locs.size(0), -1).clone()
        candidate_areas[~(inside_box & in_level)] = float("inf")
        min_area, matched = candidate_areas.min(dim=1)
        background = torch.isinf(min_area)

        labels = gt_labels[matched].clone()
        labels[background] = 0
        targets = reg[torch.arange(locs.size(0), device=locs.device), matched]
        targets[background] = 0

        cls_targets.append(labels)
        reg_targets.append(targets)
    return torch.cat(cls_targets), torch.cat(reg_targets)

def sigmoid_focal_loss(logits, labels, gamma=2.0, alpha=0.25):
    target = torch.zeros_like(logits)
    pos = labels > 0
    target[pos, labels[pos] - 1] = 1.0
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return alpha_t * (1.0 - p_t).pow(gamma) * ce

def iou_loss(pred, target, eps=1e-6):
    pl, pt, pr, pb = pred.unbind(-1)
    tl, tt, tr, tb = target.unbind(-1)
    inter_w = torch.minimum(pl, tl) + torch.minimum(pr, tr)
    inter_h = torch.minimum(pt, tt) + torch.minimum(pb, tb)
    inter = inter_w * inter_h
    pred_area = (pl + pr) * (pt + pb)
    target_area = (tl + tr) * (tt + tb)
    union = pred_area + target_area - inter
    return -torch.log((inter / union.clamp(min=eps)).clamp(min=eps))

def fcos_loss(logits, bbox_reg, ctrness, cls_targets, reg_targets):
    """Inputs are flattened over P3..P7 and image locations."""
    pos = cls_targets > 0
    N_pos = pos.sum().clamp(min=1).float()
    cls_loss = sigmoid_focal_loss(logits, cls_targets, gamma=2.0, alpha=0.25).sum() / N_pos
    if pos.any():
        reg_loss = iou_loss(bbox_reg[pos], reg_targets[pos]).sum() / N_pos
        ctr_targets = centerness_target(reg_targets[pos])
        ctr_loss = F.binary_cross_entropy_with_logits(
            ctrness[pos].squeeze(-1), ctr_targets, reduction="sum") / N_pos
    else:
        reg_loss = bbox_reg.sum() * 0.0
        ctr_loss = ctrness.sum() * 0.0
    return cls_loss + reg_loss + ctr_loss

def decode(logits, bbox_reg, ctrness, locations, score_thresh=0.05, nms_thresh=0.6):
    """Invert l,t,r,b around each location; rank by class probability times centerness."""
    ctr = ctrness.sigmoid()
    if ctr.dim() == 1:
        ctr = ctr[:, None]
    scores = logits.sigmoid() * ctr
    keep_loc, keep_cls = (scores > score_thresh).nonzero(as_tuple=True)
    if keep_loc.numel() == 0:
        return (locations.new_zeros((0, 4)),
                logits.new_zeros((0,)),
                torch.empty((0,), dtype=torch.long, device=logits.device))

    xy = locations[keep_loc]
    l, t, r, b = bbox_reg[keep_loc].unbind(-1)
    boxes = torch.stack((xy[:, 0] - l, xy[:, 1] - t,
                         xy[:, 0] + r, xy[:, 1] + b), dim=1)
    scores = scores[keep_loc, keep_cls]
    labels = keep_cls + 1
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    return boxes[keep], scores[keep], labels[keep]
```
