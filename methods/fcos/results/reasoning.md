OK, let me think this through from scratch. What's actually bothering me is that detection is the odd one out. Semantic segmentation, depth, keypoints — they're all solved by a fully convolutional network that takes the image once, runs convolutions across the whole thing, and spits out a per-pixel prediction. Clean, uniform, computation shared. Detection alone refuses to fit that mold, and when I ask why, the answer is always the same word: anchors. Every strong detector — Faster R-CNN, SSD, YOLOv2/v3, RetinaNet — tiles a dense grid of pre-defined reference boxes at every location, classifies each, and regresses an offset. It works, but look at the baggage it brings. The scales and aspect ratios and count of anchors are hyper-parameters that move COCO AP by ~4% just by being retuned, so they have to be carefully designed and redesigned for each new task. To get high recall you tile them insanely densely — 180K-plus for a single image with FPN — and almost all are negatives, so you get a brutal foreground/background imbalance. And at train time every anchor needs an IoU computed against every ground-truth box to decide if it's positive. All of that exists only because we chose to represent candidates as boxes-at-locations rather than as locations.

So let me ask the question directly: can I do detection as per-pixel prediction, FCN-style, no anchors, no proposals? Treat each location on a feature map as a training sample, exactly like a pixel in segmentation, and have it directly predict the object it belongs to. This isn't a new idea on its face — DenseBox and the UnitBox family already did per-pixel detection, regressing at each location a 4D vector to the four box sides plus a class. But they were written off for generic detection, and I should understand precisely why before I repeat their mistakes, because the failure modes are the whole game.

Two reasons they failed. First, to cope with very different object sizes, DenseBox cropped and resized images to a fixed scale and ran on an image pyramid — which throws away the one thing that makes FCN nice, computing all convolutions once. Second, and this is the one everyone points to: overlapping ground-truth boxes. If a location sits inside two boxes at once, what does it regress to? There's no well-defined target. On crowded generic scenes that ambiguity was believed to be fatal, which is why these methods stayed in scene-text and face detection where overlap is rare. There was also a recall worry — with a big output stride, maybe a per-pixel detector just can't reach objects that no location lands inside.

Let me start building and see if these are really walls or just things nobody pushed through.

Set up the parameterization. Backbone feature map F_i with total stride s. A location (x,y) on the feature map maps back to image coordinates roughly at the center of its receptive field, (⌊s/2⌋ + xs, ⌊s/2⌋ + ys). For a ground-truth box B = (x0, y0, x1, y1) with class c, a location is a positive sample if it falls inside the box, and then its class target is c. For the box itself, I don't regress corner coordinates — I regress the four distances from the location to the box's sides:

  l* = x − x0,  t* = y − y0,  r* = x1 − x,  b* = y1 − y.

This is the DenseBox/UnitBox parameterization and it's the natural one for a point-inside-a-box: every interior location can describe the same box by its own four distances, and the four are always positive for an interior point. That positivity matters for the output layer — since l,t,r,b must be ≥0, I'll put exp(·) on top of the regression branch so the network outputs a real number and exp maps it to (0,∞). At inference I just invert the four equations to recover the box corners from a location and its predicted (l,t,r,b).

Now here's a first observation that's already different from anchor detectors, and I think it's a feature, not a bug. An anchor detector only calls an anchor positive if its IoU with a box is high enough — so each object trains the regressor from a handful of anchors. My scheme makes *every* location inside the box a positive sample. That's a lot more foreground samples per object feeding the regressor. My intuition is that more positive samples should give a better-trained, more accurate box regressor. Hold that thought; it's testable later as box quality at strict IoU.

The loss, then. Classification over all locations, regression only over positives. For classification I have the same imbalance problem RetinaNet diagnosed — almost all locations are background — so I'll use focal loss, C binary classifiers with the (1−p_t)^γ down-weighting of easy examples, exactly as in RetinaNet; no reason to reinvent that. For regression I want to optimize the box as a whole, not four independent coordinates, so IoU loss as in UnitBox: it takes the predicted (l,t,r,b) and the target, forms the two boxes, and maximizes their overlap directly. So:

  L = (1/N_pos) Σ_{x,y} L_cls(p_{x,y}, c*_{x,y})  +  (λ/N_pos) Σ_{x,y} 1{c*_{x,y}>0} L_reg(t_{x,y}, t*_{x,y}),

with N_pos the number of positives, λ=1, the indicator restricting regression to foreground. Head structure I'll borrow from RetinaNet: two parallel towers of four convs each (one for classification, one for regression), shared across all pyramid levels, with the final class layer emitting 80 logits and the box layer emitting 4. Note this already has 9× fewer output variables per location than a 9-anchor detector — the simplicity is showing up immediately.

Now the two walls. Take recall first. The fear: a large stride means few locations, so maybe I can't recall enough objects. Let me actually reason about best-possible-recall — the fraction of ground-truth boxes that get assigned at least one positive sample. I can measure it. With just one mid-level feature map (stride 16, no pyramid), almost every box still contains *some* feature location, so BPR comes out around 95.6% — already higher than the ~90.9% an anchor RetinaNet achieves in the standard Detectron setting that only keeps IoU≥0.4 matches. So the recall worry was overstated: because every interior location is a positive, boxes rarely go unassigned. The concern is "actually not a problem."

But the overlap ambiguity is real, and I need a mechanism, not a hope. Here's where FPN earns its keep in a way that's different from how anchor detectors use it. Anchor detectors assign different anchor *sizes* to different pyramid levels. I have no anchors, so instead I'll directly limit the *range of regression distances* each level is allowed to handle. Compute (l*,t*,r*,b*) for a location on every level. If max(l*,t*,r*,b*) at that location exceeds the level's upper bound m_i, or falls below the previous level's bound m_{i−1}, that location is set negative on that level — it doesn't regress there. I'll use levels P3…P7 with strides 8,16,32,64,128, and bounds m_2..m_7 = 0, 64, 128, 256, 512, ∞. So P3 handles boxes whose max side-distance is in [0,64], P4 in [64,128], and so on up to P7 handling everything above 512.

Why does this resolve the ambiguity? Because overlap that causes trouble is almost always between objects of *very different sizes* — a small object sitting inside a large one. If I route them to different pyramid levels by size, then on any single level a location is, with high probability, inside only one box of that level's size class. Let me sanity-check by measuring the fraction of ambiguous locations (inside more than one box of the relevant size). Without the pyramid it's ~23% of positives; with the multi-level size routing it drops to ~7%, and if I further ignore same-category overlaps (which don't actually matter — if two same-class objects overlap, predicting either is "correct," and the other can still be recalled by locations that only fall in it), it's down to ~3.75%. That's small enough not to hurt. And for the residual cases where a location still lands in two boxes on the same level, I take the simplest tie-break: regress to the box with the **minimal area**. The cost of that choice is only that the location risks missing some larger overlapping object — but that larger object is recalled by its own non-overlapping locations. So both walls — recall and ambiguity — fall to FPN plus this size-range routing, with no anchor hyper-parameters introduced.

One refinement on the head while I'm at it. I'm sharing the head across pyramid levels — that's parameter-efficient and known to help. But the levels regress different size ranges ([0,64] for P3, [64,128] for P4, …), so forcing the *same* exp(x) on all of them is a bit unreasonable; the natural scale of the output differs by level. So instead of exp(x), use exp(s_i · x) with a trainable scalar s_i per level — let each level learn the base of its own exponential. Cheap, slightly better.

So I run this. And it lands close to anchor RetinaNet but not quite there — there's a residual gap. Let me look at *what's* failing rather than just accepting the number. The detector is producing a lot of boxes, and I notice a population of low-quality ones: boxes predicted from locations far from the object's center. That makes intuitive sense — a location near the edge or corner of a box has a long, lopsided (l,t,r,b), and regressing an accurate box from there is hard; it tends to produce a box that's mislocated but still carries a high classification score. A high-confidence, low-IoU box is exactly a false positive that NMS can't kill, because NMS ranks by score and these have high scores. That's the leak.

Now, an anchor detector partly avoids this for free: its two IoU thresholds mean only anchors well-aligned with a box become positive, so off-center junk never gets a high foreground label. I threw away those thresholds on purpose. I need to recover that effect without a hyper-parameter. What I want is a signal that says "this location is near the center of its object" vs "this location is off near the edge," and to use it to down-weight the off-center boxes' scores so NMS removes them.

I have exactly that signal sitting in the regression targets already. For a location, look at l* and r*: if the location is horizontally centered, l*≈r*, so min(l*,r*)/max(l*,r*)≈1; if it's off to one side, that ratio is near 0. Same for t*,b* vertically. So define a "center-ness" target that's 1 at the center and decays to 0 at the edges:

  centerness* = sqrt( [min(l*,r*)/max(l*,r*)] × [min(t*,b*)/max(t*,b*)] ).

Each bracket is in [0,1], their product is in [0,1], so centerness* ∈ [0,1] — perfect for a value I can train with binary cross-entropy. Why the square root? Because without it the product of two ratios decays very fast as you move off-center — a location that's only moderately off-center would already get a tiny target. The sqrt slows that decay so the signal is graded more gently across the box interior, which I want, because mildly off-center locations still produce decent boxes. Add a single extra conv layer — one branch, parallel to the classification tower — that predicts this scalar, trained with BCE, and add that to the loss.

How do I use it? At test time, multiply the predicted center-ness by the classification score to get the final ranking score. A box from an off-center location now has its score knocked down by a small center-ness factor, so it sinks in the ranking and NMS filters it out; a box from a near-center location keeps its score. This is the piece that closes the gap — it cleanly suppresses the low-quality boxes that the discarded IoU thresholds used to suppress, and it does it with zero new hyper-parameters, just one extra layer.

Let me make sure this center-ness is doing real work and isn't redundant with what I already have. Could I instead just *compute* center-ness from the predicted (l,t,r,b) vector at test time, no extra branch? I could try — but the predicted vector is itself noisy exactly for the bad boxes, so deriving the suppression signal from the same unreliable prediction doesn't help; a separately-trained center-ness branch, supervised by the geometric target, is what actually moves the number. So the separate branch is necessary. (One sibling thought: this branch could equally hang off the regression tower instead of the classification tower; either is defensible since center-ness is a geometric quantity. I'll keep it parallel to the classification branch as the default.) There's also an alternative I'm explicitly choosing against: I could mimic the anchor thresholds by only treating the central region of each box as positive — but that's an extra hyper-parameter (how central), and the whole point was to get rid of those. Center-ness gets the same suppression effect parameter-free.

Step back and trace the causal chain. The pain was that detection couldn't be FCN-style per-pixel prediction because anchors brought hyper-parameters, imbalance, and IoU-matching cost. Drop anchors, treat every location as a sample, regress (l,t,r,b) to the box sides with IoU loss and classify with focal loss — that's a clean FCN detector but it has two believed-fatal flaws. Recall turns out fine, because every interior location is positive. Overlap ambiguity is killed by routing objects to FPN levels by size range, which separates the overlapping big/small pairs, with minimal-area tie-breaking for the rare residue. And the remaining accuracy gap — low-quality off-center boxes that NMS can't suppress because their scores are high — is closed by a center-ness branch that learns sqrt of the product of the min/max distance ratios and multiplies into the score at test time. No anchors, no proposals, 9× fewer outputs per location, and it matches and then beats the anchor-based counterpart under the same training and testing settings. I'll call it FCOS — fully convolutional one-stage detection.

Now the code, level-shared head with the two towers, the three prediction layers, the per-level Scale for exp(s_i·x), and the target/loss/decode logic.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
SIZE_RANGES = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)]  # P3..P7

def assign_targets(locations_per_level, gt_boxes, gt_labels):
    """For each location on each level: positive if it falls inside a box AND
    that box's max side-distance is within the level's size range; on ties take
    the minimal-area box; everything else is background."""
    # ... compute l*,t*,r*,b* for every (location, box); inside = all > 0;
    #     keep boxes with SIZE_RANGES[level][0] <= max(l,t,r,b) <= [1];
    #     among remaining, pick the minimal-area box; else background.
    pass  # TODO (mechanics only; the assignment rule above is the content)

def fcos_loss(logits, bbox_reg, ctrness, cls_targets, reg_targets, ctr_targets):
    pos = cls_targets > 0
    N_pos = pos.sum().clamp(min=1.0)
    # focal loss over ALL locations (handles the fg/bg imbalance, no anchors)
    cls_loss = sigmoid_focal_loss(logits, cls_targets, gamma=2.0, alpha=0.25).sum() / N_pos
    # IoU loss over positives only (optimizes the box as a whole)
    reg_loss = iou_loss(bbox_reg[pos], reg_targets[pos]).sum() / N_pos
    # center-ness via BCE over positives
    ctr_loss = F.binary_cross_entropy_with_logits(
        ctrness[pos], ctr_targets[pos], reduction="sum") / N_pos
    return cls_loss + reg_loss + ctr_loss

def decode(logits, bbox_reg, ctrness, locations, score_thresh=0.05):
    """Score = sqrt or product? -> multiply class prob by center-ness so
    off-center (low center-ness) boxes are down-weighted before NMS."""
    scores = logits.sigmoid() * ctrness.sigmoid()      # the key test-time use
    # keep scores > thresh; box from a location + (l,t,r,b) by inverting:
    # x0 = x - l, y0 = y - t, x1 = x + r, y1 = y + b; then NMS.
    pass  # TODO (decode mechanics)
```

The chain end to end: anchors brought hyper-parameters and imbalance and IoU-matching, so drop them and make every feature location a per-pixel sample regressing (l,t,r,b) with IoU loss and classifying with focal loss; recall survives because interior locations are dense, overlap ambiguity dies under FPN size-range routing with minimal-area tie-breaking, and the leftover low-quality off-center boxes are suppressed by a parameter-free center-ness branch multiplied into the score at test time — a fully convolutional, anchor-free detector that's simpler and stronger than its anchor-based counterpart.
