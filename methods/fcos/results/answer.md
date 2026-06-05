# FCOS: Fully Convolutional One-Stage Object Detection

## Problem

Anchor-based detectors tile a dense grid of pre-defined reference boxes, which forces hand-tuned anchor hyper-parameters (scales, ratios, count) that strongly affect accuracy, creates extreme foreground/background imbalance, and requires IoU matching at train time. FCOS removes anchors and proposals entirely and solves detection as per-pixel prediction, FCN-style, while matching or beating the anchor-based counterpart.

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

def centerness_target(reg_targets):        # (...,4) = l*,t*,r*,b*
    l, t, r, b = reg_targets.unbind(-1)
    lr = torch.stack([l, r], -1); tb = torch.stack([t, b], -1)
    return torch.sqrt((lr.min(-1).values / lr.max(-1).values.clamp(min=1e-6)) *
                      (tb.min(-1).values / tb.max(-1).values.clamp(min=1e-6)))

SIZE_RANGES = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)]  # P3..P7

def assign_targets(locations_per_level, gt_boxes, gt_labels):
    """Positive if location is inside a box AND that box's max side-distance is
    within the level's size range; minimal-area box on ties; else background."""
    ...  # produces cls_targets, reg_targets per location

def fcos_loss(logits, bbox_reg, ctrness, cls_targets, reg_targets, ctr_targets):
    pos = cls_targets > 0
    N_pos = pos.sum().clamp(min=1.0)
    cls_loss = sigmoid_focal_loss(logits, cls_targets, gamma=2.0, alpha=0.25).sum() / N_pos
    reg_loss = iou_loss(bbox_reg[pos], reg_targets[pos]).sum() / N_pos
    ctr_loss = F.binary_cross_entropy_with_logits(ctrness[pos], ctr_targets[pos],
                                                  reduction="sum") / N_pos
    return cls_loss + reg_loss + ctr_loss

def decode(logits, bbox_reg, ctrness, locations, score_thresh=0.05):
    scores = logits.sigmoid() * ctrness.sigmoid()   # down-weight off-center boxes
    # keep scores > thresh; x0=x-l, y0=y-t, x1=x+r, y1=y+b; then NMS
    ...
```
