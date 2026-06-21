Object detection in 2018 was dominated by anchor-based detectors such as Faster R-CNN, SSD, YOLOv2/v3, and RetinaNet. These methods tile a dense grid of pre-defined reference boxes at every feature location, classify each box as foreground or background, and regress a refinement offset. The anchor machinery works, but it carries a cluster of problems: the scales, aspect ratios, and count of anchors are hand-tuned hyper-parameters that can move COCO AP by several points, they must be redesigned for each new task or domain, and the dense tiling creates extreme foreground/background imbalance while requiring expensive IoU matching between every anchor and every ground-truth box. Per-pixel detectors such as DenseBox and UnitBox already showed that a fully convolutional network can regress box-side distances at each location, but they were considered unsuitable for generic detection because they relied on image pyramids rather than shared convolutions and because overlapping ground-truth boxes create an ambiguous regression target for locations inside more than one box. The challenge is therefore to build a detector that is fully convolutional and anchor-free, yet keeps the recall and localization quality of dense anchor detectors.

The proposed method is FCOS, short for Fully Convolutional One-Stage Object Detection. FCOS treats every feature-map location as a training sample, just like a pixel in semantic segmentation. For a location that falls inside a ground-truth box, the network regresses the four distances to the box sides, classifies the location with one of the object categories, and predicts a scalar center-ness score that measures how close the location is to the center of the object. The regression target at image location (x, y) inside box (x0, y0, x1, y1) is l* = x − x0, t* = y − y0, r* = x1 − x, b* = y1 − y. Because these distances are always positive, the raw network output is passed through exp(s_i · x), where s_i is a trainable scalar per FPN level so that each pyramid level learns its own output scale. At inference the box corners are recovered by subtracting and adding the predicted distances from the location.

FCOS uses Feature Pyramid Network levels P3 through P7 with strides 8, 16, 32, 64, and 128 to handle different object sizes. Instead of assigning anchors of different sizes to different levels, FCOS assigns each location based on the maximum of its four regression targets. The allowed ranges are 0 to 64 for P3, 64 to 128 for P4, 128 to 256 for P5, 256 to 512 for P6, and 512 to infinity for P7. This routing resolves the overlap ambiguity because most overlaps involve objects of very different scales, which are then handled on different levels. If a location still falls inside more than one valid box on the same level, the smaller-area box is chosen. This size routing also preserves recall because any interior feature location can serve as a positive sample, so objects only need to contain a single feature center.

The loss combines three terms. Classification uses focal loss over all locations to handle the heavy background imbalance. Regression uses IoU loss on positive locations, optimizing the box as a whole rather than four independent coordinates. Finally, a center-ness branch is trained with binary cross-entropy on positive locations. The center-ness target is sqrt( (min(l*, r*) / max(l*, r*)) * (min(t*, b*) / max(t*, b*)) ), which is 1 at the object center and decays toward the edges. At test time the final detection score is the class probability multiplied by the predicted center-ness, so low-quality boxes from off-center locations are ranked lower and removed by non-maximum suppression. The following code implements the FCOS head, target assignment, loss, and decoding.

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

    def forward(self, features):
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

def centerness_target(reg_targets):
    l, t, r, b = reg_targets.unbind(-1)
    lr = torch.stack([l, r], -1)
    tb = torch.stack([t, b], -1)
    return torch.sqrt(
        (lr.min(-1).values / lr.max(-1).values.clamp(min=1e-6)) *
        (tb.min(-1).values / tb.max(-1).values.clamp(min=1e-6))
    )

SIZE_RANGES = [(0, 64), (64, 128), (128, 256), (256, 512), (512, float("inf"))]

def assign_targets(locations_per_level, gt_boxes, gt_labels):
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
