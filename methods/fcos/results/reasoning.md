OK, let me think this through from scratch. What's actually bothering me is that detection is the odd one out. Semantic segmentation, depth, keypoints — they're all solved by a fully convolutional network that takes the image once, runs convolutions across the whole thing, and spits out a per-pixel prediction. Clean, uniform, computation shared. Detection alone refuses to fit that mold, and when I ask why, the answer is always the same word: anchors. Every strong detector — Faster R-CNN, SSD, YOLOv2/v3, RetinaNet — tiles a dense grid of pre-defined reference boxes at every location, classifies each, and regresses an offset. It works, but look at the baggage it brings. The scales and aspect ratios and count of anchors are hyper-parameters that move COCO AP by ~4% just by being retuned, so they have to be carefully designed and redesigned for each new task. To get high recall you tile them insanely densely — 180K-plus for a single image with FPN — and almost all are negatives, so you get a brutal foreground/background imbalance. And at train time every anchor needs an IoU computed against every ground-truth box to decide if it's positive. All of that exists only because we chose to represent candidates as boxes-at-locations rather than as locations.

So let me ask the question directly: can I do detection as per-pixel prediction, FCN-style, no anchors, no proposals? Treat each location on a feature map as a training sample, exactly like a pixel in segmentation, and have it directly predict the object it belongs to. This isn't a new idea on its face — DenseBox and the UnitBox family already did per-pixel detection, regressing at each location a 4D vector to the four box sides plus a class. But they were written off for generic detection, and I should understand precisely why before I repeat their mistakes, because the failure modes are the whole game.

Two reasons they failed. First, to cope with very different object sizes, DenseBox cropped and resized images to a fixed scale and ran on an image pyramid — which throws away the one thing that makes FCN nice, computing all convolutions once. Second, and this is the one everyone points to: overlapping ground-truth boxes. If a location sits inside two boxes at once, what does it regress to? There's no well-defined target. On crowded generic scenes that ambiguity was believed to be fatal, which is why these methods stayed in scene-text and face detection where overlap is rare. There was also a recall worry — with a big output stride, maybe a per-pixel detector just can't reach objects that no location lands inside.

Let me start building and see if these are really walls or just things nobody pushed through.

Set up the parameterization. Backbone feature map F_i with total stride s. A location (x,y) on the feature map maps back to image coordinates roughly at the center of its receptive field, (⌊s/2⌋ + xs, ⌊s/2⌋ + ys). For a ground-truth box B = (x0, y0, x1, y1) with class c, a location is a positive sample if it falls inside the box, and then its class target is c. For the box itself, I don't regress corner coordinates — I regress the four distances from the location to the box's sides:

  l* = x − x0,  t* = y − y0,  r* = x1 − x,  b* = y1 − y.

This is the DenseBox/UnitBox parameterization and it's the natural one for a point-inside-a-box: every interior location can describe the same box by its own four distances, and the four are always positive for an interior point. That positivity matters for the output layer — since l,t,r,b must be ≥0, I'll put exp(·) on top of the regression branch so the network outputs a real number and exp maps it to (0,∞). At inference I just invert the four equations to recover the box corners from a location and its predicted (l,t,r,b).

Now here's a first observation that's already different from anchor detectors, and I think it's a feature, not a bug. An anchor detector only calls an anchor positive if its IoU with a box is high enough — so each object trains the regressor from a handful of anchors. My scheme makes *every* location inside the box a positive sample. That's a lot more foreground geometry per object feeding the regressor, and it avoids the fragile question of which pre-defined box should represent the object.

The loss, then. Classification over all locations, regression only over positives. For classification I have the same imbalance problem RetinaNet diagnosed — almost all locations are background — so I'll use focal loss, C binary classifiers with the (1−p_t)^γ down-weighting of easy examples, exactly as in RetinaNet; no reason to reinvent that. For regression I want to optimize the box as a whole, not four independent coordinates, so IoU loss as in UnitBox: it takes the predicted (l,t,r,b) and the target, forms the two boxes, and maximizes their overlap directly. So:

  L = (1/N_pos) Σ_{x,y} L_cls(p_{x,y}, c*_{x,y})  +  (λ/N_pos) Σ_{x,y} 1{c*_{x,y}>0} L_reg(t_{x,y}, t*_{x,y}),

with N_pos the number of positives, λ=1, the indicator restricting regression to foreground. Head structure I'll borrow from RetinaNet: two parallel towers of four convs each (one for classification, one for regression), shared across all pyramid levels, with the final class layer emitting 80 logits and the box layer emitting 4. Note this already has 9× fewer output variables per location than a 9-anchor detector — the simplicity is showing up immediately.

Now the two walls. Take recall first. The fear is simple: a large stride means few locations, so maybe I can't recall objects that no feature location lands inside. The right diagnostic is best-possible recall — the fraction of ground-truth boxes that get assigned at least one training sample. But because *any* interior location is positive, the requirement is weaker than anchor matching: the box only needs to contain one feature center, not one anchor whose IoU crosses a threshold. Even a single stride-16 feature level assigns almost every box to some location, around 95.6% BPR, while the standard anchor RetinaNet setting that keeps only IoU≥0.4 low-quality matches is around 90.9%. So the recall worry is much smaller than it first looked; dense interior points buy recall without anchors.

But the overlap ambiguity is real, and I need a mechanism, not a hope. Here's where FPN earns its keep in a way that's different from how anchor detectors use it. Anchor detectors assign different anchor *sizes* to different pyramid levels. I have no anchors, so instead I'll directly limit the *range of regression distances* each level is allowed to handle. Compute (l*,t*,r*,b*) for a location on every level. If max(l*,t*,r*,b*) at that location exceeds the level's upper bound m_i, or falls below the previous level's bound m_{i−1}, that location is set negative on that level — it doesn't regress there. I'll use levels P3…P7 with strides 8,16,32,64,128, and bounds m_2..m_7 = 0, 64, 128, 256, 512, ∞. So P3 handles boxes whose max side-distance is in [0,64], P4 in [64,128], and so on up to P7 handling everything above 512.

Why does this resolve the ambiguity? Because overlap that causes trouble is often between objects of *very different sizes* — a small object sitting inside a large one. If I route them to different pyramid levels by size, then on any single level a location is usually inside only one box of that level's size class. The assignment-count diagnostic matches that reasoning: without the pyramid, ambiguous samples are about 23% of positives; with the multi-level size routing they fall to about 7%, and if same-category overlaps are ignored because either object gives the same class label, the different-category ambiguity is about 3.75%. And for the residual cases where a location still lands in two boxes on the same level, I take the simplest tie-break: regress to the box with the **minimal area**. The cost of that choice is only that the location risks missing some larger overlapping object — but that larger object is recalled by its own non-overlapping locations. So both walls — recall and ambiguity — fall to FPN plus this size-range routing, with no anchor hyper-parameters introduced.

One refinement on the head while I'm at it. I'm sharing the head across pyramid levels — that's parameter-efficient and known to help. But the levels regress different size ranges ([0,64] for P3, [64,128] for P4, …), so forcing the *same* exp(x) on all of them is a bit unreasonable; the natural scale of the output differs by level. So instead of exp(x), use exp(s_i · x) with a trainable scalar s_i per level — let each level learn the base of its own exponential while the rest of the head stays shared.

There is still a geometric leak in this design. Every interior location gets the same class label, but not every interior location is equally good for localization. A point near the edge or corner has a long, lopsided (l,t,r,b), and regressing an accurate box from there is harder than regressing from near the center. The classifier, however, only sees "inside the object" and has no reason to give that edge location a lower category score. That creates exactly the bad kind of detection: a high-confidence, low-IoU box that NMS cannot reliably remove, because NMS ranks by score.

Now, an anchor detector partly avoids this for free: its two IoU thresholds mean only anchors well-aligned with a box become positive, so off-center junk never gets a high foreground label. I threw away those thresholds on purpose. I need to recover that effect without a hyper-parameter. What I want is a signal that says "this location is near the center of its object" vs "this location is off near the edge," and to use it to down-weight the off-center boxes' scores so NMS removes them.

I have exactly that signal sitting in the regression targets already. For a location, look at l* and r*: if the location is horizontally centered, l*≈r*, so min(l*,r*)/max(l*,r*)≈1; if it's off to one side, that ratio is near 0. Same for t*,b* vertically. So define a "center-ness" target that's 1 at the center and decays to 0 at the edges:

  centerness* = sqrt( [min(l*,r*)/max(l*,r*)] × [min(t*,b*)/max(t*,b*)] ).

Each bracket is in [0,1], their product is in [0,1], so centerness* ∈ [0,1] — perfect for a value I can train with binary cross-entropy. Why the square root? Because without it the product of two ratios decays very fast as you move off-center — a location that's only moderately off-center would already get a tiny target. The sqrt slows that decay so the signal is graded more gently across the box interior, which I want, because mildly off-center locations still produce decent boxes. Add a single extra conv layer — one branch, parallel to the classification tower — that predicts this scalar, trained with BCE, and add that to the loss.

How do I use it? At test time, multiply the predicted center-ness by the classification score to get the final ranking score. A box from an off-center location now has its score knocked down by a small center-ness factor, so it sinks in the ranking and NMS filters it out; a box from a near-center location keeps its score. This recovers the quality-ranking role that anchor IoU thresholds used to play, but without reintroducing threshold hyper-parameters.

Let me make sure this center-ness is not redundant with what I already have. Could I instead just *compute* center-ness from the predicted (l,t,r,b) vector at test time, no extra branch? That couples the quality signal to the same noisy box prediction I am trying to distrust. The cleaner design is a separately supervised scalar target: one extra branch, trained from the ground-truth geometry, then multiplied into the class score. There's also an alternative I'm explicitly choosing against: I could mimic the anchor thresholds by only treating the central region of each box as positive — but that's an extra hyper-parameter (how central), and the whole point was to get rid of those. Center-ness gets the same suppression effect parameter-free.

That gives me FCOS: fully convolutional one-stage detection, no anchors or proposals, with each location emitting a class vector, four box-side distances, and one center-ness scalar.

The concrete form is a level-shared head with the two towers, the three prediction layers, the per-level Scale for exp(s_i·x), and target/loss/decode logic that keeps the assignment rule explicit.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

class Scale(nn.Module):
    # the trainable s_i so each FPN level learns the base of its exp(s_i * x)
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
                # GroupNorm (not BN) in the towers: stable at the small
                # detection batch size; ReLU after each conv
                tower += [nn.Conv2d(in_channels, in_channels, 3, padding=1),
                          nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)]
        self.cls_tower = nn.Sequential(*cls_tower)
        self.box_tower = nn.Sequential(*box_tower)
        # per-location predictions: C binary class logits, 4 box distances,
        # and the single-layer center-ness (parallel to the cls tower)
        self.cls_logits  = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred   = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness  = nn.Conv2d(in_channels, 1, 3, padding=1)
        # one learnable scale per pyramid level for exp(s_i * x)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

    def forward(self, features):
        logits, bbox_reg, ctrness = [], [], []
        for level, x in enumerate(features):       # P3..P7
            cls_feat = self.cls_tower(x)
            box_feat = self.box_tower(x)
            logits.append(self.cls_logits(cls_feat))
            ctrness.append(self.centerness(cls_feat))
            # exp(s_i * raw): map to (0, inf) since l,t,r,b are positive,
            # with a per-level learnable base
            bbox_reg.append(torch.exp(self.scales[level](self.bbox_pred(box_feat))))
        return logits, bbox_reg, ctrness

def compute_locations(h, w, stride, device):
    # map each feature location back to the center of its receptive field:
    # floor(s/2) + x*s  (and same for y)
    shifts_x = torch.arange(0, w * stride, stride, device=device) + stride // 2
    shifts_y = torch.arange(0, h * stride, stride, device=device) + stride // 2
    ys, xs = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    return torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=1)

def centerness_target(reg_targets):
    # reg_targets: (..., 4) = (l*, t*, r*, b*)
    l, t, r, b = reg_targets.unbind(-1)
    lr = torch.stack([l, r], -1); tb = torch.stack([t, b], -1)
    # sqrt slows the decay so mildly off-center locations keep a graded signal
    return torch.sqrt((lr.min(-1).values / lr.max(-1).values.clamp(min=1e-6)) *
                      (tb.min(-1).values / tb.max(-1).values.clamp(min=1e-6)))

# size-range routing: object_sizes_of_interest per level (m_{i-1}, m_i)
SIZE_RANGES = [(0, 64), (64, 128), (128, 256), (256, 512), (512, float("inf"))]

def assign_targets(locations_per_level, gt_boxes, gt_labels):
    """For each location on each level: positive if it falls inside a box AND
    that box's max side-distance is within the level's size range; on ties take
    the minimal-area box; everything else is background."""
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
    # focal loss over ALL locations (handles the fg/bg imbalance, no anchors)
    cls_loss = sigmoid_focal_loss(logits, cls_targets, gamma=2.0, alpha=0.25).sum() / N_pos
    if pos.any():
        # IoU loss over positives only (optimizes the box as a whole)
        reg_loss = iou_loss(bbox_reg[pos], reg_targets[pos]).sum() / N_pos
        # center-ness via BCE over positives
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
    scores = logits.sigmoid() * ctr                    # the key test-time use
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

The chain end to end: anchors brought hyper-parameters and imbalance and IoU-matching, so drop them and make every feature location a per-pixel sample regressing (l,t,r,b) with IoU loss and classifying with focal loss; recall survives because interior locations are dense, overlap ambiguity dies under FPN size-range routing with minimal-area tie-breaking, and the leftover low-quality off-center boxes are suppressed by a parameter-free center-ness branch multiplied into the score at test time — a fully convolutional, anchor-free detector.
