Let me start from what actually hurts. The detectors that win the benchmarks all do the same thing: they hypothesize a few thousand candidate boxes, then for each candidate they crop the image (or crop the feature map) and run a classifier on that crop, refining the box at the same time. R-CNN did it by re-running a CNN on each cropped region; SPPnet and Fast R-CNN made it cheaper by computing the convolutional feature map once and pooling each candidate out of it; Faster R-CNN even learned the candidates with a region proposal network instead of using Selective Search. But the shape of the computation never changed — there is always a *per-candidate resample-and-classify* stage. And that stage is where the time goes. The strongest of these runs around 7 frames per second on a top-end GPU. For anything real-time, or anything embedded, that is hopeless.

So the thing I want to remove is the resampling. Not make it faster — remove it. One forward pass of one network that emits, for every object in the image, a box and a class, and never crops a region to re-encode it. The question is whether you can do that and stay accurate, because people have tried the one-pass route and paid for it.

Look at what the one-pass detectors do. OverFeat slides a classifier convolutionally and reads one box off each location of the topmost feature map. YOLO takes the whole topmost feature map, flattens it through fully-connected layers, and for each cell of a coarse grid predicts a couple of boxes and a shared set of class probabilities. Both are fast. Both are noticeably less accurate — YOLO sits around 63 mAP where Faster R-CNN is at 73 — and both are especially weak on small objects. I want to understand *why*, precisely, because the why is going to tell me what to build.

Two things jump out. First, they predict from a *single* feature map, and it's a deep, low-resolution one. A deep feature map has a coarse stride: each cell summarizes a large patch of the image with a large receptive field. A small object, or several small objects packed together, simply doesn't get its own cell — it falls between the grid points, or it shares a cell with other things. There's nowhere to put a small detection. Second, YOLO predicts boxes with a fully-connected layer on top of the flattened map. That throws away the spatial layout of the feature map — the predictor at one image location can't reuse the predictor at another — and it regresses absolute box coordinates from whole-image features, which is a hard regression target with a lot of parameters.

Now contrast with what the proposal networks figured out. Faster R-CNN's region proposal network doesn't flatten anything. It slides a small convolution over the last feature map, and at each location it places a handful of reference boxes — *anchors* — of a few preset scales and aspect ratios, and predicts, per anchor, an objectness score and four numbers that *offset* the anchor into a tight box. The prediction is convolutional, so the same small predictor runs at every location and shares parameters; and the regression target is an *offset relative to a known reference box*, not an absolute coordinate, which is a much gentler thing to learn. MultiBox before it had the same flavor: regress a fixed set of prior boxes plus a confidence each, and train by matching each ground-truth box to the best-overlapping prior.

So the proposal side already solved the two things YOLO got wrong — convolutional prediction, and offsets-to-references. The catch is that Faster R-CNN throws that good prediction away: it uses those boxes only as *proposals*, then resamples features inside each and runs a second classifier. That's the stage I want gone. And, critically, its anchors all live on *one* feature map. One stride, one receptive-field size, asked to cover everything from a tiny bottle to a bus.

Let me pull those threads together. What if I keep the convolutional, offset-to-reference prediction from the proposal network — but instead of producing class-agnostic proposals to be re-classified, I have it directly emit, per reference box, the class scores *and* the offsets, in one shot? No second stage, no resampling. That kills the speed problem in principle. It's essentially MultiBox's regressor, except (a) the confidences are per-class instead of objectness, and (b) there's no follow-up classifier — the one prediction is the answer.

But if I do only that, I've basically built a convolutional YOLO, and I'll inherit the small-object weakness, because I'm still reading off a single deep feature map. I need to fix the scale problem at the same time, or the accuracy gap won't close.

How do detectors usually handle multiple scales? The classic answer is an image pyramid: resize the input to several sizes, run the net on each, merge the results. SPPnet and OverFeat do versions of this. But that means several forward passes, and I'm trying to get *down* to one. So an image pyramid is exactly the wrong direction.

Here's the thing I keep circling back to. A convolutional network already computes a stack of feature maps at decreasing resolution inside a single forward pass — that's free, I'm computing it anyway. And the dense-prediction people have shown that those maps are not interchangeable: the earlier, higher-resolution layers keep fine spatial detail (that's why segmentation work reaches back into them), while the deeper layers are coarse and semantic with a big receptive field. The layers even have different empirical receptive-field sizes. So I don't need an image pyramid — I have a *feature* pyramid sitting right there in the backbone. A small object will have support in an early, high-resolution map where the grid is fine; a large object will have support in a deep, coarse map whose cells each cover a lot of image.

So the design falls out: attach the convolutional predictor not to one feature map but to *several* feature maps of decreasing resolution, and let each map be responsible for a band of object sizes. The high-res maps catch small objects; the low-res maps catch large ones. One forward pass, parameters shared within each map, and the multi-scale coverage that the single-map detectors lacked. I don't even have to stop at the backbone's native maps — I can bolt extra convolutional layers onto the end that keep shrinking the spatial size, say from a 38×38 map down through 19, 10, 5, 3, and finally a 1×1 map that sees the whole image. Six maps of decreasing resolution.

Now I have to make this concrete: what reference boxes do I tile, on each map, and how do I predict from them?

Take a feature map of size m×n with p channels. At each of the m×n locations I want to predict against k reference boxes. For each reference box I need c class scores (including a background class) and 4 offset numbers. The natural predictor — keeping it convolutional, the lesson from before — is a small 3×3×p kernel that, applied at every location, produces one output. So I need (c+4)k such kernels per feature map: 4k of them produce the offset numbers, ck produce the class scores. Run them over the whole m×n map and I get (c+4)k·m·n outputs. No fully-connected layer anywhere, no flattening; each output is computed from a 3×3 neighborhood of local features, and the offsets are read relative to the reference box sitting at that location. That's the whole detection head — a couple of 3×3 convolutions per feature map.

Next: where exactly do the reference boxes sit and what sizes are they? Tiling the centers is easy — I put one set of boxes at each cell, with the center at the cell center. For a feature map that is |f_k| cells on a side, cell (i,j) gets boxes centered at ((j+0.5)/|f_k|, (i+0.5)/|f_k|) in normalized image coordinates. The 0.5 puts the center in the middle of the cell rather than the corner.

The sizes are the interesting part. I want each of my m feature maps to be responsible for a particular range of object scales, smoothly increasing as the maps get coarser. So let map k (counting k=1 for the highest-resolution map up to k=m for the coarsest) own a scale s_k, and space the scales linearly between a minimum and a maximum:

  s_k = s_min + (s_max − s_min)/(m − 1) · (k − 1),  k ∈ [1, m].

With s_min = 0.2 and s_max = 0.9, the finest map gets boxes that are 20% of the image and the coarsest gets 90%, with the rest evenly spaced in between. The reason for tying scale to map this way is exactly the receptive-field argument: I *want* the fine map to be responsible only for small objects and the coarse map only for large ones, so I deliberately make the reference boxes small on the fine map and large on the coarse map. The boxes don't have to coincide with the actual receptive field of the layer — the network can learn to be responsive to the scale I assign — but matching the assignment to the resolution is what lets each layer specialize and makes the learning problem on each layer easy.

A single square box per location isn't enough, though — objects aren't square. So at each location I add several aspect ratios. Let a_r ∈ {1, 2, 3, 1/2, 1/3}. For a box of scale s_k and aspect ratio a_r I want width·height to stay at the scale's area while the shape stretches, so:

  w_k = s_k·√a_r,  h_k = s_k/√a_r.

Check: w·h = s_k², independent of a_r, and w/h = a_r. Good — every aspect ratio at a given map keeps the same area, just reshaped. For the square case a_r = 1 there's a subtlety: the linear scale steps leave a gap between s_k and s_{k+1}, and an object whose size sits between two maps' scales would be poorly covered. So for a_r = 1 I add one extra box at the geometric-mean scale s'_k = √(s_k · s_{k+1}), bridging consecutive maps. That gives 6 boxes per location in general (the five aspect ratios plus the extra square). On maps where I expect fewer shapes I can drop the {3, 1/3} pair and use 4. Tiling these over all six maps gives a large, diverse set of reference boxes covering a spread of scales and shapes — for a 300×300 input it works out to 38²·4 + 19²·6 + 10²·6 + 5²·6 + 3²·4 + 1²·4 = 8732 boxes. That's an order of magnitude more box hypotheses than YOLO's 98, and they sample location, scale, and shape densely — which is the whole point, because a richer discretization of the output box space makes it more likely that *some* reference box is already close to each true object, leaving the regressor only a small correction.

Now the training problem. The awkward part, compared to a proposal-based detector, is that my outputs are a *fixed* set of detectors — box (k) at location (i,j) on map — and I have to decide which of them is responsible for which ground-truth object before I can compute a loss. This is an assignment problem.

MultiBox's answer was bipartite: match each ground-truth box to the single reference box of highest overlap, one-to-one. That guarantees every object gets at least one detector assigned, which I definitely want — otherwise an object contributes no positive signal. But one-to-one is harsh. With 8732 reference boxes, an object frequently overlaps several of them well; forcing the network to pick exactly one and treat the rest as negative makes it fight itself, suppressing boxes that are actually good. So I'll relax it: first, like MultiBox, match each ground-truth box to its best-overlap reference box (so nothing is orphaned); then *additionally* match a reference box to *any* ground-truth box with Jaccard overlap above a threshold of 0.5. A reference box can thus be positive if it overlaps any object well enough. This means a single object can claim several reference boxes — the indicator x_ij^p that the i-th reference box matches the j-th ground-truth of class p can have Σ_i x_ij^p ≥ 1 — and the network is allowed to predict high scores for several overlapping boxes instead of being forced to anoint one. That makes learning easier and, empirically, more stable.

With matches in hand, the loss. It's two terms — am I getting the class right, and am I getting the box right — and I'll average over the number of matched boxes N so the scale doesn't depend on how many objects are in the image:

  L(x, c, l, g) = (1/N)·(L_conf(x, c) + α·L_loc(x, l, g)),

with N the number of matched reference boxes (if N = 0, set the loss to 0), and α weighting the two terms; cross-validation lands on α = 1, so the two terms simply add.

The localization term. I don't regress raw box coordinates — that's the YOLO mistake. I regress the offset from the reference box to the matched ground-truth box, in the same parameterization the proposal/regression literature uses. With the reference box d = (d_cx, d_cy, d_w, d_h) and the matched ground truth g, the regression targets are

  ĝ_cx = (g_cx − d_cx)/d_w,  ĝ_cy = (g_cy − d_cy)/d_h,
  ĝ_w  = log(g_w / d_w),     ĝ_h  = log(g_h / d_h).

The center offset is divided by the reference box's own width/height, so a shift is measured in units of the box — scale-invariant. Width and height go through a log because they're positive and multiplicative: a ground-truth box twice the reference width and one half should be symmetric residuals (+log2 and −log2), which log gives and a raw ratio does not, and it keeps the target near zero when the box is already about right. The network predicts l = (l_cx, l_cy, l_w, l_h) and I penalize the difference with a smooth-L1 loss — quadratic for small residuals so it's gentle near the optimum, linear for large ones so a badly-placed box or an outlier doesn't blow up the gradient:

  L_loc(x, l, g) = Σ_{i∈Pos} Σ_{m∈{cx,cy,w,h}} x_ij^k · smooth_L1(l_i^m − ĝ_j^m).

Only positive (matched) boxes contribute to localization — there's no meaningful box to regress for a background detector.

The confidence term is a softmax over the c class scores including a background class (index 0). A positive box is pushed toward its matched class, a negative box toward background:

  L_conf(x, c) = − Σ_{i∈Pos} x_ij^p · log(ĉ_i^p) − Σ_{i∈Neg} log(ĉ_i^0),
  where ĉ_i^p = exp(c_i^p) / Σ_p exp(c_i^p).

Now I hit a wall, and it's a bad one. I have 8732 reference boxes per image and usually a handful of objects. After matching, the overwhelming majority of boxes are negatives — background. If I sum the confidence loss over all of them, the gradient is a tidal wave of "you're background" and the few positives are drowned out; the net collapses to predicting background everywhere and the loss still looks like it's going down. I can't just use all the negatives.

The naive fix is to randomly subsample negatives down to some ratio. But random negatives are mostly trivial — empty sky, obvious background — and contribute almost no gradient; I'd be spending my negative budget on examples the net already gets right. What I actually want are the *hard* negatives: the background boxes the net is currently most confident are objects, because those are the false positives that will hurt at test time. So after matching, I sort the negative boxes by their confidence loss — highest loss first, i.e. the most egregiously mis-scored background — and keep only the top ones, enough to make the ratio of negatives to positives at most 3:1. Three negatives per positive keeps the problem balanced enough that the positive signal isn't swamped, while focusing every negative I do keep on a case that's actually informative. This makes optimization faster and training more stable.

A couple of practical things about the backbone before I can build it. I'll use a VGG-16 pretrained on ImageNet, truncated before its classification head, as the base. VGG's fully-connected layers fc6 and fc7 I want to turn into convolutions so the whole thing stays fully convolutional and keeps spatial resolution — but a plain conversion would either be enormous (fc6 sees a 7×7×512 field) or lose resolution if I keep the downsampling pool5. The à trous trick solves this: keep pool5 at stride 1 instead of 2 so I don't halve the resolution, and make conv6 a 3×3 convolution with dilation 6 so it still sees a wide receptive field through the "holes" without the parameter cost or the downsampling. Subsample fc6/fc7's weights into these convolutions. This keeps a usefully high-resolution map and runs about 20% faster than carrying the full VGG, with no accuracy loss. Then I append the extra shrinking conv layers (conv8 through conv11) for the coarser maps.

One more snag specific to using an early layer. I want to predict from conv4_3, the highest-resolution map, because it's my main hope for small objects. But conv4_3 sits much earlier in the network and its feature activations have a substantially larger magnitude than the deeper layers'. If I feed it into the same kind of predictor, its scale dominates and training is unstable. The fix is to L2-normalize the features at each location of conv4_3 and then multiply by a single learned scale (initialized around 20), so the layer's contribution is brought into line with the others but the network can still tune how much it matters.

And at inference, since I'm producing thousands of boxes, I prune aggressively: drop everything below a confidence of 0.01, run per-class non-maximum suppression with a Jaccard threshold of 0.45 to remove duplicate detections of the same object, and keep the top 200 boxes per image. NMS is cheap relative to the single forward pass.

Let me also sanity-check the data side, because a one-shot detector that never resamples is more exposed to object scale and position than a region-based one — it can't lean on a pooling step that's robust to translation. So I lean hard on data augmentation to manufacture scale and position variety: for each training image, with some probability use the whole image, or sample a patch constrained to have a minimum Jaccard overlap (0.1, 0.3, 0.5, 0.7, or 0.9) with the objects, or sample a fully random patch; patch size between 0.1 and 1 of the image and aspect ratio between 1/2 and 2; keep a ground-truth box if its center lands in the patch; then resize, randomly flip, and apply photometric distortions. The cropping acts as a "zoom in" that creates larger object instances. The opposite — a "zoom out" — I get by occasionally placing the image on a larger mean-filled canvas (16× the area) before cropping, which manufactures small objects; this is what most helps the weak small-object case.

Time to write it. The backbone is VGG-16 truncated, with the à trous conv6/conv7, then extra shrinking layers; predictions come off conv4_3 (L2-normalized), conv7, and the four extra maps; each prediction source gets a 3×3 conv for offsets and a 3×3 conv for class scores, sized to the number of reference boxes at that map; the reference boxes are tiled by the scale and aspect-ratio rules above; the loss matches, encodes offsets, and mines hard negatives.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from itertools import product


# ---- Backbone: VGG-16 truncated, with atrous conv6/conv7 ----
def vgg(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2, 2)]
        elif v == 'C':                       # ceil_mode keeps the 38x38 size
            layers += [nn.MaxPool2d(2, 2, ceil_mode=True)]
        else:
            layers += [nn.Conv2d(in_channels, v, 3, padding=1),
                       nn.ReLU(inplace=True)]
            in_channels = v
    # pool5 at stride 1 (keep resolution), fc6 -> dilated conv6, fc7 -> conv7
    layers += [nn.MaxPool2d(3, 1, padding=1),
               nn.Conv2d(512, 1024, 3, padding=6, dilation=6),  # a trous
               nn.ReLU(inplace=True),
               nn.Conv2d(1024, 1024, 1),
               nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, in_channels):
    # Extra shrinking conv layers -> the coarse feature maps (conv8..conv11)
    layers, flag = [], False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':                                     # stride-2: shrink
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     (1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, (1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


# A 1xN learned scale on top of L2-normalized conv4_3 features, so the early,
# high-magnitude layer is brought in line with the deeper prediction sources.
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant_(self.weight, scale)
        self.eps = 1e-10

    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        x = x / norm
        return self.weight.view(1, -1, 1, 1) * x


# ---- Reference (default) boxes: the multi-scale tiling ----
class DefaultBoxes:
    """Tile default boxes over the chosen feature maps, in (cx,cy,w,h) form."""
    def __init__(self, image_size, feature_maps, steps,
                 min_sizes, max_sizes, aspect_ratios):
        self.image_size = image_size
        self.feature_maps = feature_maps          # e.g. [38,19,10,5,3,1]
        self.steps = steps                        # effective stride per map
        self.min_sizes = min_sizes                # s_k * image_size, per map
        self.max_sizes = max_sizes                # used for the s'_k box
        self.aspect_ratios = aspect_ratios        # e.g. [[2],[2,3],...]

    def __call__(self):
        boxes = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k              # cell-center, normalized
                cy = (i + 0.5) / f_k
                s_k = self.min_sizes[k] / self.image_size           # scale s_k
                boxes += [cx, cy, s_k, s_k]                         # a_r = 1
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                boxes += [cx, cy, s_k_prime, s_k_prime]             # extra box
                for ar in self.aspect_ratios[k]:                    # a_r and 1/a_r
                    boxes += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    boxes += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        out = torch.Tensor(boxes).view(-1, 4)
        return out.clamp_(max=1, min=0)


# ---- The detector: one forward pass over multiple feature maps ----
class SSD(nn.Module):
    def __init__(self, base, extras, head, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])     # 3x3 conv per source: 4*k filters
        self.conf = nn.ModuleList(head[1])    # 3x3 conv per source: c*k filters

    def forward(self, x):
        sources, loc, conf = [], [], []
        for k in range(23):                   # up to conv4_3 relu
            x = self.vgg[k](x)
        sources.append(self.L2Norm(x))        # prediction source 1: conv4_3
        for k in range(23, len(self.vgg)):    # through conv7 (fc7)
            x = self.vgg[k](x)
        sources.append(x)                     # source 2: conv7
        for k, v in enumerate(self.extras):   # conv8_2, conv9_2, conv10_2, conv11_2
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)             # sources 3..6: decreasing resolution
        # convolutional predictors: per source, per location, k boxes x (c+4)
        for (s, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(s).permute(0, 2, 3, 1).contiguous())
            conf.append(c(s).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))


def multibox(vgg, extras, num_boxes, num_classes):
    loc, conf = [], []
    for v, nb in zip([vgg[21], vgg[-2]], num_boxes[:2]):       # conv4_3, conv7
        loc += [nn.Conv2d(v.out_channels, nb * 4, 3, padding=1)]
        conf += [nn.Conv2d(v.out_channels, nb * num_classes, 3, padding=1)]
    for v, nb in zip(extras[1::2], num_boxes[2:]):             # the extra maps
        loc += [nn.Conv2d(v.out_channels, nb * 4, 3, padding=1)]
        conf += [nn.Conv2d(v.out_channels, nb * num_classes, 3, padding=1)]
    return vgg, extras, (loc, conf)


# ---- Offset encoding (cx,cy log w,h), relative to a default box ----
def encode(matched, defaults, variances):
    # matched: GT in (xmin,ymin,xmax,ymax); defaults: (cx,cy,w,h)
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - defaults[:, :2]
    g_cxcy /= (variances[0] * defaults[:, 2:])         # center offset / box size
    g_wh = (matched[:, 2:] - matched[:, :2]) / defaults[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]              # log of the size ratio
    return torch.cat([g_cxcy, g_wh], 1)


def jaccard(box_a, box_b):
    A, B = box_a.size(0), box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    return inter / (area_a + area_b - inter)


def point_form(boxes):       # (cx,cy,w,h) -> (xmin,ymin,xmax,ymax)
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


# ---- Matching: best-overlap (never orphan a GT) + every box with IoU>0.5 ----
def match(threshold, truths, defaults, variances, labels, loc_t, conf_t, idx):
    overlaps = jaccard(truths, point_form(defaults))
    best_prior_overlap, best_prior_idx = overlaps.max(1)   # best box per GT
    best_truth_overlap, best_truth_idx = overlaps.max(0)   # best GT per box
    # force each GT's best box to be a positive for that GT
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1                      # +1: reserve 0 = background
    conf[best_truth_overlap < threshold] = 0               # IoU<0.5 -> negative
    loc_t[idx] = encode(matches, defaults, variances)
    conf_t[idx] = conf


def log_sum_exp(x):
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# ---- The MultiBox loss: conf + alpha*loc, averaged over N matches,
#      with hard negative mining at 3:1 ----
class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, defaults, overlap_thresh=0.5,
                 neg_pos_ratio=3, variances=(0.1, 0.2)):
        super().__init__()
        self.num_classes = num_classes
        self.defaults = defaults
        self.threshold = overlap_thresh
        self.negpos = neg_pos_ratio
        self.variances = variances

    def forward(self, predictions, targets):
        loc_data, conf_data = predictions
        num = loc_data.size(0)
        defaults = self.defaults
        num_priors = defaults.size(0)

        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):                          # match per image
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]
            match(self.threshold, truths, defaults, self.variances,
                  labels, loc_t, conf_t, idx)

        pos = conf_t > 0                                # matched (positive) boxes

        # localization loss: smooth-L1 over positives only
        pos_idx = pos.unsqueeze(2).expand_as(loc_data)
        loss_l = F.smooth_l1_loss(loc_data[pos_idx].view(-1, 4),
                                  loc_t[pos_idx].view(-1, 4), reduction='sum')

        # per-box confidence loss, used to rank negatives by hardness
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # hard negative mining: rank negatives by loss, keep top 3:1
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0                                 # exclude positives from ranking
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # confidence loss over positives + mined negatives
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # L = (Lconf + alpha*Lloc) / N,  alpha = 1
        N = num_pos.sum().clamp(min=1)
        return loss_l / N, loss_c / N
```

The causal chain, end to end: the accurate detectors are slow because they resample features inside every box hypothesis and reclassify, so I removed that stage and predict class scores and box offsets directly, in one convolutional pass. Predicting from a single deep feature map (the fast detectors' choice) kills small-object recall, so I attached the predictor to several feature maps of decreasing resolution — fine maps for small objects, coarse maps for large — using the free feature pyramid the backbone already computes, with extra shrinking layers tacked on. On each map I tiled default boxes whose scales increase linearly across maps (s_k = s_min + (s_max−s_min)(k−1)/(m−1)) and whose aspect ratios stretch a fixed area (w = s_k√a_r, h = s_k/√a_r, plus a bridging box at √(s_k s_{k+1})), and a small 3×3 conv predicts, per box, c class scores and 4 offsets. I trained it by matching each ground-truth to its best box and to every box overlapping it past 0.5, encoding the localization target as a smooth-L1 regression on center/log-size offsets, summing a softmax confidence term, averaging over the matched count, and mining hard negatives down to 3:1 so the thousands of background boxes don't drown the signal — with à trous to keep the backbone resolution cheap, an L2-normalized conv4_3, and aggressive scale-jittering augmentation to compensate for never resampling.
