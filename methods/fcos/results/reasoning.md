OK, let me think this through from scratch. What's actually bothering me is that detection is the odd one out. Semantic segmentation, depth, keypoints — they're all solved by a fully convolutional network that takes the image once, runs convolutions across the whole thing, and spits out a per-pixel prediction. Clean, uniform, computation shared. Detection alone refuses to fit that mold, and when I ask why, the answer is always the same word: anchors. Every strong detector — Faster R-CNN, SSD, YOLOv2/v3, RetinaNet — tiles a dense grid of pre-defined reference boxes at every location, classifies each, and regresses an offset. It works, but look at the baggage it brings. The scales and aspect ratios and count of anchors are hyper-parameters that move COCO AP by ~4% just by being retuned, so they have to be carefully designed and redesigned for each new task. To get high recall you tile them insanely densely — 180K-plus for a single image with FPN — and almost all are negatives, so you get a brutal foreground/background imbalance. And at train time every anchor needs an IoU computed against every ground-truth box to decide if it's positive. All of that exists only because we chose to represent candidates as boxes-at-locations rather than as locations.

So let me ask the question directly: can I do detection as per-pixel prediction, FCN-style, no anchors, no proposals? Treat each location on a feature map as a training sample, exactly like a pixel in segmentation, and have it directly predict the object it belongs to. This isn't a new idea on its face — DenseBox and the UnitBox family already did per-pixel detection, regressing at each location a 4D vector to the four box sides plus a class. But they were written off for generic detection, and I should understand precisely why before I repeat their mistakes, because the failure modes are the whole game.

Two reasons they failed. First, to cope with very different object sizes, DenseBox cropped and resized images to a fixed scale and ran on an image pyramid — which throws away the one thing that makes FCN nice, computing all convolutions once. Second, and this is the one everyone points to: overlapping ground-truth boxes. If a location sits inside two boxes at once, what does it regress to? There's no well-defined target. On crowded generic scenes that ambiguity was believed to be fatal, which is why these methods stayed in scene-text and face detection where overlap is rare. There was also a recall worry — with a big output stride, maybe a per-pixel detector just can't reach objects that no location lands inside.

Let me start building and see if these are really walls or just things nobody pushed through.

Set up the parameterization. Backbone feature map F_i with total stride s. A location (x,y) on the feature map maps back to image coordinates roughly at the center of its receptive field, (⌊s/2⌋ + xs, ⌊s/2⌋ + ys). For a ground-truth box B = (x0, y0, x1, y1) with class c, a location is a positive sample if it falls inside the box, and then its class target is c. For the box itself, I don't regress corner coordinates — I regress the four distances from the location to the box's sides:

  l* = x − x0,  t* = y − y0,  r* = x1 − x,  b* = y1 − y.

This is the DenseBox/UnitBox parameterization and it's the natural one for a point-inside-a-box: every interior location can describe the same box by its own four distances, and the four are always positive for an interior point. That positivity matters for the output layer — since l,t,r,b must be ≥0, I'll put exp(·) on top of the regression branch so the network outputs a real number and exp maps it to (0,∞). At inference I just invert the four equations to recover the box corners: x0 = x − l, y0 = y − t, x1 = x + r, y1 = y + b. For the box (10,20,90,80) and a location at (50,50), (l*,t*,r*,b*) = (40,30,40,30), and inverting gives (50−40, 50−30, 50+40, 50+30) = (10,20,90,80) exactly — addition undoing subtraction, so encode and decode round-trip, and the four targets are positive for any interior point by construction.

Now here's a first observation that's already different from anchor detectors. An anchor detector only calls an anchor positive if its IoU with a box is high enough — so each object trains the regressor from a handful of anchors. My scheme makes *every* location inside the box a positive sample. That's a lot more foreground geometry per object feeding the regressor, and it avoids the fragile question of which pre-defined box should represent the object. I'll come back to whether "more positives" helps or hurts; for now it's just a different design.

The loss, then. Classification over all locations, regression only over positives. For classification I have the same imbalance problem RetinaNet diagnosed — almost all locations are background — so I'll use focal loss, C binary classifiers with the (1−p_t)^γ down-weighting of easy examples, exactly as in RetinaNet; no reason to reinvent that. For regression I want to optimize the box as a whole, not four independent coordinates, so IoU loss as in UnitBox: it takes the predicted (l,t,r,b) and the target, forms the two boxes, and maximizes their overlap directly. The loss never reconstructs the boxes — it works directly in (l,t,r,b) — so it's not obvious that this distance-space formula gives the true geometric IoU of the two reconstructed boxes. The intersection of two boxes that share the same center point has half-widths min(pl,tl) on the left and min(pr,tr) on the right, so its width is min(pl,tl)+min(pr,tr), and likewise its height; each box's area is (l+r)(t+b). Numbers: target (40,30,40,30) around (50,50) is the box (10,20,90,80), area 4800; predicted (20,15,60,45) is the box (30,35,110,95). The formula gives intersection width min(20,40)+min(60,40) = 60, height min(15,30)+min(45,30) = 45, inter = 2700, union = 4800+4800−2700 = 6900, IoU = 0.391 (loss −log(0.391) = 0.938) — and the geometric IoU of the two reconstructed boxes computed directly (intersection x∈[30,90], y∈[35,80], area 60×45=2700 over the same 6900 union) gives the identical 0.391. The distance-space formula is the real IoU, so I can use it directly. So:

  L = (1/N_pos) Σ_{x,y} L_cls(p_{x,y}, c*_{x,y})  +  (λ/N_pos) Σ_{x,y} 1{c*_{x,y}>0} L_reg(t_{x,y}, t*_{x,y}),

with N_pos the number of positives, λ=1, the indicator restricting regression to foreground. Head structure I'll borrow from RetinaNet: two parallel towers of four convs each (one for classification, one for regression), shared across all pyramid levels, with the final class layer emitting 80 logits and the box layer emitting 4. That's 4 box outputs per location against the 9-anchor detector's 9×4 — fewer output variables, which is at least a simplicity gain even if I can't yet claim it's an accuracy one.

Now the two walls. Take recall first. The fear is concrete: a large stride means few locations, so maybe I can't recall objects that no feature location lands inside. The right diagnostic is best-possible recall — the fraction of ground-truth boxes that get assigned at least one training sample. The interesting question is whether the interior-point rule recalls *more* than anchor matching, because the requirement is structurally weaker: the box only needs to contain one feature center, not one anchor whose IoU with the box crosses a threshold. I want to quantify the direction with a quick simulation. I drop a stride-16 grid of centers over an 800×800 image and throw 2000 random boxes at it, sizes from 8px up to ~200px. Counting boxes that contain at least one center: 97.8%. Then I give every location one square anchor of side ≈ 4×stride and keep only matches with IoU ≥ 0.4, the standard low-quality threshold: 21.1% of the same boxes get a match. My toy uses a single anchor scale so it understates the real multi-scale anchor recall, but the gap is enormous and in the predicted direction: a small box easily sits a center inside it but rarely reaches IoU 0.4 with a fixed-side anchor. And when I count it directly on the real training set — just feature level P4 alone, stride 16, no FPN yet — I get a BPR of 95.6%, against RetinaNet's 90.9% in the standard implementation once only IoU≥0.4 matches are kept: the same direction the toy simulation predicted, not a number I was hoping for. The recall worry is much smaller than it first looked; dense interior points buy recall without anchors.

But the overlap ambiguity is real, and I need a mechanism, not a hope. Anchor detectors assign different anchor *sizes* to different pyramid levels. I have no anchors, so instead I'll directly limit the *range of regression distances* each level is allowed to handle. Compute (l*,t*,r*,b*) for a location on every level. If max(l*,t*,r*,b*) at that location exceeds the level's upper bound m_i, or falls below the previous level's bound m_{i−1}, that location is set negative on that level — it doesn't regress there. I'll use levels P3…P7 with strides 8,16,32,64,128, and bounds m_2..m_7 = 0, 64, 128, 256, 512, ∞. So P3 handles boxes whose max side-distance is in [0,64], P4 in [64,128], and so on up to P7 handling everything above 512.

Does this actually break the ambiguity, though? The argument is that overlap which causes trouble is usually between objects of *very different sizes* — a small object sitting inside a large one — and routing them to different levels by size separates them. Let me make that argument concrete instead of waving at it. Take the worst case: a small box (60,60,120,120), area 3600, sitting fully inside a big box (0,0,200,200), area 40000, and a location at (90,90) that is inside both. Without any routing, both boxes contain (90,90), so the target is genuinely ambiguous. Now route by size: for the big box, max(l,t,r,b) at (90,90) is max(90,90,110,110) = 110, which lands it in the [64,128] level (P4), while for the small box max(l,t,r,b) = max(30,30,30,30) = 30, which lands it in [0,64] (P3). So the two boxes are assigned to *different* levels, and on any single level only one of them is a candidate for this location — the tie disappears on each level where prediction actually happens. That's exactly the mechanism, and the nested-box case where I'd most expect ambiguity is precisely the case routing resolves. Counting it directly on the training set confirms this: ambiguous samples drop from about 23% of positives without the pyramid to about 7% with size routing, and to about 3.75% if same-category overlaps are discounted (either object gives the same label there, so it doesn't matter). For the residual cases where a location still lands in two boxes on the *same* level, I take the simplest tie-break: regress to the box with the **minimal area**. I can check the assignment code does that — in my nested example, if I disable routing and run the min-area rule, the location at (90,90) matches box index 1, the small box (area 3600 < 40000), and gets its label. The cost of that choice is only that the location risks missing the larger overlapping object — but that larger object is recalled by its own non-overlapping locations. So both walls — recall and ambiguity — yield to FPN plus this size-range routing, with no anchor hyper-parameters introduced.

One refinement on the head while I'm at it. I'm sharing the head across pyramid levels — that's parameter-efficient and known to help. But the levels regress different size ranges ([0,64] for P3, [64,128] for P4, …), so forcing the *same* exp(x) on all of them is a bit unreasonable; the natural scale of the output differs by level. So instead of exp(x), use exp(s_i · x) with a trainable scalar s_i per level — let each level learn the base of its own exponential while the rest of the head stays shared.

There is still a geometric leak in this design, and it could quietly hurt precision. Every interior location gets the same class label, but not every interior location is equally good for localization. A point near the edge or corner has a long, lopsided (l,t,r,b), and regressing an accurate box from there is harder than regressing from near the center. The classifier, however, only sees "inside the object" and has no reason to give that edge location a lower category score. That creates exactly the bad kind of detection: a high-confidence, low-IoU box that NMS cannot reliably remove, because NMS ranks by score. So the very "every interior location is positive" choice that bought me recall is now spending it back as low-quality high-score boxes — I need to deal with that.

Now, an anchor detector partly avoids this for free: its two IoU thresholds mean only anchors well-aligned with a box become positive, so off-center junk never gets a high foreground label. I threw away those thresholds on purpose. I need to recover that effect without a hyper-parameter. What I want is a signal that says "this location is near the center of its object" vs "this location is off near the edge," and to use it to down-weight the off-center boxes' scores so NMS removes them.

I have exactly that signal sitting in the regression targets already. For a location, look at l* and r*: if the location is horizontally centered, l*≈r*, so min(l*,r*)/max(l*,r*)≈1; if it's off to one side, that ratio is near 0. Same for t*,b* vertically. So I want a "center-ness" target that's 1 at the center and decays toward 0 at the edges, built from the product of those two ratios. The product of two [0,1] ratios is already in [0,1], so I could train it directly. But let me look at how fast it decays before I settle the form, because if it collapses too quickly the signal is useless across most of the box interior. Hold a location vertically centered (t*=b*, vertical ratio 1) and slide it horizontally across a width-100 box. At 45% of the way (l=45,r=55) the horizontal ratio is 0.818; at 40% (l=40,r=60) it's 0.667; at 30% it's 0.429; at 20% it's 0.250; at 10% it's 0.111. So even the bare ratio already drops to two-thirds by the time you're only 10% off-center — and the *product* of two such ratios decays as the square of that, which is harsh: a moderately off-center location would get a tiny target and be treated almost like background. I want mildly off-center locations to keep a graded, not-tiny signal, since they still produce decent boxes. Taking the square root of the product fixes the decay rate: with the sqrt, the same positions give 0.905, 0.816, 0.655, 0.500, 0.333 — at 40% off-center it's 0.816 instead of 0.667, and the decay across the interior is gentle and roughly linear rather than steep. So:

  centerness* = sqrt( [min(l*,r*)/max(l*,r*)] × [min(t*,b*)/max(t*,b*)] ).

Each bracket is in [0,1], their product is in [0,1], the sqrt of a value in [0,1] is in [0,1], so centerness* ∈ [0,1] — a value I can train with binary cross-entropy. The sqrt is doing real work, not decoration: I just watched it change a 0.667 into a 0.816 and flatten the falloff. Add a single extra conv layer — one branch, parallel to the classification tower — that predicts this scalar, trained with BCE, and add that to the loss.

How do I use it? At test time, multiply the predicted center-ness by the classification score to get the final ranking score. A box from an off-center location now has its score knocked down by a small center-ness factor, so it sinks in the ranking and NMS filters it out; a box from a near-center location keeps its score. This recovers the quality-ranking role that anchor IoU thresholds used to play, but without reintroducing threshold hyper-parameters.

Is this center-ness branch redundant with what I already have — could I instead just *compute* center-ness from the predicted (l,t,r,b) vector at test time, no extra branch? That couples the quality signal to the same noisy box prediction I am trying to distrust — if the box is bad, its computed center-ness is bad too, so it can't flag itself. A separately supervised scalar, trained from the ground-truth geometry, stays an independent vote on quality even when the box regression is off. There's also an alternative I'm explicitly choosing against: I could mimic the anchor thresholds by only treating the central region of each box as positive — but that's an extra hyper-parameter (how central), and the whole point was to get rid of those. Center-ness gets the same suppression effect parameter-free, so I'll keep the separately supervised branch.

So the design that comes out of this: fully convolutional, no anchors or proposals, with each location emitting a class vector, four box-side distances, and one center-ness scalar — recall held up by dense interior positives, overlap resolved by FPN size-range routing with minimal-area tie-breaking, off-center boxes suppressed by the center-ness multiply. FCOS.

The head itself is just the two four-conv GroupNorm towers feeding three per-location layers (class logits, box distances, center-ness), shared across levels, with a per-level `Scale` module supplying the trainable s_i on the box branch. The pieces worth writing out are the ones I actually derived above: how a location maps back to image coordinates, the size-range assignment with minimal-area tie-break, the center-ness target, and the IoU loss I checked numerically.

```python
class Scale(nn.Module):
    # the trainable s_i so each FPN level learns the base of its exp(s_i * x)
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float))
    def forward(self, x):
        return x * self.scale

def compute_locations(h, w, stride, device):
    # map each feature location back to the center of its receptive field:
    # floor(s/2) + x*s  (and same for y)
    shifts_x = torch.arange(0, w * stride, stride, device=device) + stride // 2
    shifts_y = torch.arange(0, h * stride, stride, device=device) + stride // 2
    ys, xs = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    return torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=1)

# size-range routing: object_sizes_of_interest per level (m_{i-1}, m_i)
SIZE_RANGES = [(0, 64), (64, 128), (128, 256), (256, 512), (512, float("inf"))]

def assign_targets(locations_per_level, gt_boxes, gt_labels):
    """For each location on each level: positive if it falls inside a box AND
    that box's max side-distance is within the level's size range; on ties take
    the minimal-area box; everything else is background."""
    cls_targets, reg_targets = [], []
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

def iou_loss(pred, target, eps=1e-6):
    # the distance-space formula I verified above reduces to the geometric IoU
    pl, pt, pr, pb = pred.unbind(-1)
    tl, tt, tr, tb = target.unbind(-1)
    inter_w = torch.minimum(pl, tl) + torch.minimum(pr, tr)
    inter_h = torch.minimum(pt, tt) + torch.minimum(pb, tb)
    inter = inter_w * inter_h
    pred_area = (pl + pr) * (pt + pb)
    target_area = (tl + tr) * (tt + tb)
    union = pred_area + target_area - inter
    return -torch.log((inter / union.clamp(min=eps)).clamp(min=eps))

def centerness_target(reg_targets):
    # reg_targets: (..., 4) = (l*, t*, r*, b*); sqrt slows the decay so
    # mildly off-center locations keep a graded signal, as I checked above
    l, t, r, b = reg_targets.unbind(-1)
    lr = torch.stack([l, r], -1); tb = torch.stack([t, b], -1)
    return torch.sqrt((lr.min(-1).values / lr.max(-1).values.clamp(min=1e-6)) *
                      (tb.min(-1).values / tb.max(-1).values.clamp(min=1e-6)))
```

Classification loss is focal loss over every location exactly as in RetinaNet; regression loss is `iou_loss` over positives; center-ness is BCE against `centerness_target`, also over positives — the three terms summed and divided by N_pos, matching the L above. Decode just inverts (l,t,r,b) around each location's coordinates, multiplies sigmoid(class logit) by sigmoid(center-ness) for the ranking score, and runs NMS — mechanical once the targets and losses are fixed. A fully convolutional, anchor-free detector.
