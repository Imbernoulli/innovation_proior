OK, let me think this through from scratch. I want to detect every object in an image and also hand back a pixel-accurate mask for each one — not a single per-pixel labeling of the scene, but a separate mask per instance, so that two people standing shoulder to shoulder come out as two masks, not one merged blob. That last part is the whole difficulty: it's detection (find and classify and localize each thing) and segmentation (label pixels) at once, and the segmentation has to respect instances.

What do I already have on the table? Two mature pipelines, and they're mature for a reason, so let me actually use them rather than invent something exotic.

On the detection side, the strong base is the two-stage region detector. Run a convolutional backbone once over the whole image to get a shared feature map. Then a region proposal network slides over that map: at each spatial location it lays down k anchors at several scales and aspect ratios, and two little 1×1-conv siblings score each anchor for objectness and regress a box delta. That gives me a few hundred to a thousand candidate boxes. For each candidate, a second stage pulls a fixed-size feature out of the shared map and runs a small head that does two things in parallel — a softmax over the K+1 classes, and a per-class box refinement — trained jointly with a multi-task loss L = L_cls + L_box, where L_cls is cross-entropy and L_box is a smooth-L1 on the regression outputs of the ground-truth class only. The thing I want to hold onto here is the *shape* of the second stage: it's two sibling outputs computed in parallel off the same pooled region feature, and the whole net trains end to end in one stage. That parallel-siblings design is what made this pipeline simple and fast in the first place — the original region method was a slow multi-stage cascade (run a CNN per warped region, then an SVM, then a separate regressor), and collapsing classification and box regression into parallel heads on shared features is exactly what fixed it.

On the segmentation side, the strong base is the fully convolutional network. Throw away the fully-connected layers so the whole network stays spatial; the output is a map, and at each output pixel you put a softmax over the C categories with a multinomial cross-entropy loss, then upsample back toward input resolution. The lesson I want from this is narrow but important: a *spatial* output should be produced by *convolutions that keep the spatial layout*. If I collapse a region's feature into a flat vector and ask a fully-connected layer to spit out a mask, I've thrown away the very 2-D structure the mask is. Keep it convolutional and the m-by-m grid of the mask stays in one-to-one correspondence with an m-by-m grid of features.

Lay those two lessons side by side and a candidate falls out almost on its own. The detection base already produces, for each region, a pooled fixed-size feature feeding two parallel sibling heads. The mask is just one more thing I want to read off that same region. So rather than build a new pipeline, the cheapest thing I can try is to hang a *third* sibling on the second-stage head — a small fully-convolutional branch that reads the same pooled region feature and emits a mask, in parallel with the class and box siblings. Stage one would stay exactly as it is; the only addition is one branch and one term, L = L_cls + L_box + L_mask. That's so minimal it's almost suspicious — most instance-segmentation systems are far more elaborate — so before I commit I should check whether the existing systems are elaborate for a *reason* I'm about to walk into, or just by historical accident.

Now, is "parallel" actually the right call, or am I just copying a pattern? Let me look at what the alternatives do, because most existing instance-segmentation systems do *not* do this. The segment-proposal family learns to propose class-agnostic mask candidates first and then classifies each candidate with a region classifier — segmentation precedes recognition. The cascade family proposes boxes, then predicts a mask from each box, then categorizes — three dependent stages. The fully-convolutional-instance family predicts one shared bank of position-sensitive channels that jointly encode class, box, and mask. The common thread in all of these is that mask prediction and class prediction are *entangled*: either masks are computed first and feed classification, or a single shared representation has to serve all three tasks at once. Two consequences. One, entanglement means multi-stage pipelines, which are slow and fiddly. Two — and this is the one that should worry me — when the mask and the class share a representation or compete, overlapping instances of the same class break things: a shared position-sensitive map produces systematic artifacts and spurious edges right where two same-class objects touch. So "parallel, independent siblings" isn't just aesthetic mimicry; decoupling the mask from the class is plausibly what *avoids* the overlap failure. Let me keep that hypothesis and see if it forces a specific loss design.

How exactly do I supervise the mask branch, then? Here's where I have to be careful, because the naive thing is to copy the semantic-segmentation FCN wholesale: have the branch output a per-pixel softmax over the K classes and train it with multinomial cross-entropy. Let me think about what that actually does inside a region. Per-pixel softmax means at every pixel the K classes *compete* — the pixel's foreground probability for class "dog" is pushed down when "cat" goes up, because they're normalized together. But wait — by the time I'm running the mask branch on a region, the class of that region is the job of the *classification sibling*. I don't need the mask branch to also decide "is this pixel dog or cat?". I only need it to decide "is this pixel part of *the* object in this region, or background?" Making the classes compete inside the mask is solving a problem I've already solved elsewhere, and worse, it re-couples segmentation with classification — exactly the entanglement I just argued causes the overlap artifacts.

So flip it. Let the branch output K separate masks, one per class, each an m×m map, so the output is K·m² numbers per region. Apply a per-pixel **sigmoid**, independently, to each — no normalization across classes. For a region whose ground-truth class is k, define the mask loss *only on the k-th mask*, as the average binary cross-entropy over its m² pixels:

  L_mask = −(1/m²) Σ_{i,j} [ y_{ij} log σ(z^k_{ij}) + (1 − y_{ij}) log(1 − σ(z^k_{ij}) ) ],

where z^k is the k-th output map and y∈{0,1} is the ground-truth mask cropped to the region. I want to be sure the words "the other K−1 masks get no gradient" are literally true of this loss and not just something I'm hoping for, because the whole decoupling claim rests on it. Let me trace it on a toy case. Take two positive regions with ground-truth classes k=2 and k=0, K=3 classes, an m=2 output, so the branch emits a [2,3,2,2] tensor of logits. The loss selects, per region, only the channel indexed by that region's class — region 0 reads channel 2, region 1 reads channel 0 — and computes binary cross-entropy of just those selected 2×2 maps against the targets. Backprop through that should leave a gradient on exactly the selected channels and exact zeros everywhere else. Running it: region 0 gets a nonzero gradient on channel 2 (norm ≈ 0.13) and *exactly* 0.0000 on channels 0 and 1; region 1 gets a nonzero gradient on channel 0 (≈ 0.19) and exactly 0.0000 on channels 1 and 2. So the indexing really does deliver gradient to one channel per region and nothing to the rest — no path by which class "dog" pushes down class "cat" inside the mask. The decoupling isn't a story I'm telling about the loss; it's a property the loss provably has. Division of labor, made concrete: the box branch owns "what is it"; the mask branch owns "which pixels are it", per class, in isolation.

Let me sanity-check that this decoupling is doing real work and I'm not fooling myself. If decoupling is the point, then whether the mask is class-specific or class-agnostic should not dominate — because the per-class identity of the mask is not carrying the classification load anyway, the box branch is. So a single class-agnostic m×m mask (predict one mask regardless of class) ought to stay close enough to the K class-specific version to be a useful control. If instead it cratered, that would mean the mask branch was secretly relying on class-specific shape or competition to work, and my decoupling story would be incomplete. And the harder, more direct test: swap my sigmoid+binary loss back to the FCN-style softmax+multinomial loss and see if it gets worse. If decoupling matters, coupling should cost clear accuracy, not just noise.

And keep the branch fully convolutional, not fully connected — that was the FCN lesson. Concretely the mask head can be a short stack of 3×3 convolutions that hold the spatial resolution, then a 2×2 transposed convolution (a deconvolution) with stride 2 that upsamples the m×m feature, then a 1×1 conv to produce the K output maps. Why convolutional and not an fc layer that emits m² numbers? Because the fc collapses the 2-D layout into a vector and then has to re-learn 2-D structure from scratch in its weights, costing parameters and accuracy; the conv head keeps every layer in explicit pixel correspondence. The fc-mask approach is exactly what the segment-proposal methods used, so this is another control I should test directly.

Now I hit the real problem, and it's not in the head at all. It's in *how I pull the region feature out of the shared map in the first place*. Stare at the standard region-feature extractor for a second, because I've been taking it for granted.

A proposal box is in continuous image coordinates, say its left edge is at x = 134.7 pixels, on a feature map with stride 16. The standard operator does this: first map the box onto the feature grid by dividing by the stride and **rounding** — 134.7/16 = 8.41875 → rounds to cell 8. Then it splits that quantized box into, say, a 7×7 grid of bins, and **rounds each bin boundary** to integer cells too. Then it max-pools the features inside each integer bin. Two separate roundings. Let me actually measure the first one in image pixels: cell 8 maps back to image coordinate 8·16 = 128, so the region's left edge has been moved from 134.7 to 128 — a shift of −6.7 pixels — before I've even started binning. And 6.7 is just where 134.7 happened to land; the worst case is a snap of half a feature cell, i.e. stride/2 = 8 pixels at stride 16, and 16 pixels at stride 32. So the feature I extract for "the region at x=134.7" is actually the feature for a region displaced by up to that much, and the *bins inside it* are then further misaligned by a second rounding of the same order.

For classification this never mattered, and that's *why* nobody fixed it: a class label is invariant to a few pixels of translation — a dog shifted 8 pixels is still a dog — so the downstream classifier shrugs the misalignment off. But now I'm asking this same pooled feature to support a *pixel-to-pixel* mask. The mask branch is supposed to keep an explicit m×m correspondence between its output cells and locations in the region. If the feature extraction has already smeared the region by half a bin and snapped its internal grid, then "output pixel (3,4)" no longer corresponds to a definite place in the image — the correspondence I'm relying on is broken before the mask branch even runs. The rounding that was harmless for boxes is *exactly* what poisons masks. That's the crack.

So I need to extract the region feature *without any rounding*. Let me design that. Don't quantize the box: use x/16 = 8.42, full stop, keep the fraction. Don't quantize the bins: divide the (continuous) region evenly into the m×m grid, so each bin has continuous floating boundaries. Now each bin covers a little continuous patch of the feature map, and I want a single value per bin. The trouble is the feature map only has values at integer grid points, and my bin centers/sample points fall *between* grid points. But I already have the tool for reading a feature map at a non-integer location: bilinear sampling. To get the value at a fractional point (x, y), take the four surrounding integer grid points and interpolate with weights given by the fractional distances — that's a smooth, differentiable read, no rounding anywhere.

Let me write the sampling out concretely so I'm sure there's no hidden quantization. For region with continuous top-left (in feature coordinates) at (x0, y0), width w and height h on the feature map, bin (p, q) of an m×m grid has size (w/m, h/m). Inside each bin I place a small regular grid of sample points — say a 2×2 grid — at the bin's quarter positions. The sample point indexed (iy, ix) within bin (q, p) sits at

  y = y0 + q·(h/m) + (iy + 0.5)·(h/m)/G,
  x = x0 + p·(w/m) + (ix + 0.5)·(w/m)/G,

where G is the per-axis number of sample points. Read each (x, y) by bilinear interpolation from its four neighboring grid points, then aggregate the G² samples in the bin — average them (max works too). Before I trust this, let me put numbers through it and confirm two things: that the bins tile the region evenly, and that no sample coordinate ever lands on a snapped integer cell. Take the region from above whose left edge is at the *un-rounded* feature coordinate x0 = 134.7/16 = 8.41875… — but to keep the arithmetic legible let me round the box itself to x0 = 8.42, width w = 7.0 in feature units, m = 7 bins, G = 2 samples per axis. Bin width is w/m = 7.0/7 = 1.0, so bins should start at 8.42, 9.42, 10.42, … and the two sample x's in bin p sit at the bin's quarter and three-quarter marks. Computing the x for the first two bins: bin p=0 gives 8.42 + (0.5)(1.0)/2 = 8.6700 and 8.42 + (1.5)(1.0)/2 = 9.1700; bin p=1 gives 9.6700 and 10.1700. Two checks pass. The four numbers are 8.67, 9.17, 9.67, 10.17 — none is an integer, so every read genuinely falls *between* grid points and goes through bilinear interpolation rather than snapping to a cell. And the spacing between consecutive bins' samples is exactly 1.0 = w/m, i.e. the bins tile the region uniformly with no gap or overlap, and the whole grid is anchored at the true 8.42 rather than at the rounded 8.0. So the output is a fixed m×m feature for the region that stays aligned with the region's true location; nothing in the coordinate path rounds.

Two things to check on this design. First, how sensitive is it to the number of sample points or to max-vs-average? My intuition: not very, *as long as I don't quantize*. The samples are all bilinear reads of a smooth feature map, so a 2×2 grid averaged versus 4×4 versus a single center sample should land in the same place; whether I aggregate by max or average should also barely move. If the result *were* very sensitive to those knobs, it would mean the sampling itself is the active ingredient — but I don't think it is. Which points to the second, sharper check.

I claimed the misalignment is the villain. But the bilinear-sampling primitive is also new relative to the plain rounding-and-max extractor, so how do I know it's the *alignment* that matters and not just "bilinear is nicer than max-pool"? There's a clean control. Consider an extractor that uses bilinear resampling *but still quantizes the region first*, like the rounding extractor does, and only then resamples within the quantized box. That isolates the variable: it has the bilinear machinery but throws away alignment. If bilinear were the magic, this control would do well; if alignment is the magic, this control would perform essentially like the old rounding extractor — and far below my no-quantization version. I'm betting on the latter: the quantize-then-bilinear control should sit right next to plain RoIPool, both well below the aligned extractor. That result would pin the cause squarely on alignment.

And I'd expect the size of the alignment win to *grow with the stride*, because the misalignment magnitude is half a feature cell, which is stride/2 pixels — bigger stride, bigger smear. So at stride 16 the aligned extractor should help, and at stride 32 it should help more, with the most visible gains at the strict high-IoU metric (AP75) where boundary precision is everything. There's a nice downstream consequence if that holds: large-stride features have long been considered too coarse for precise localization, and people stuck to smaller strides for that reason. If alignment is what was actually hurting, then with the aligned extractor a stride-32 feature might become competitive with a stride-16 one — which would mean the "large stride is too coarse" folklore was really "large stride amplifies the quantization error," and I've removed the quantization. Worth verifying, because it changes which backbones are usable.

Let me also nail down the rest of the head so the spatial story is consistent. On a multi-scale pyramid backbone, the region head reads from the pyramid level matched to the region's size, and a mask of resolution like 14 pooled then deconv'd to 28 keeps the boundary detail. On a single-scale backbone where the head reuses a heavy final-stage block, the mask comes off at 14. The exact resolution isn't sacred; the point is enough spatial resolution to place a boundary, produced convolutionally. Everything in the second stage now reads from the *aligned* feature, so class, box, and mask all see a region feature that actually corresponds to the region.

Training details, taken straight from the detection base because the segmentation system turns out to be robust to them. A proposal is positive if its IoU with a ground-truth box is at least 0.5, negative otherwise. The mask loss is defined *only on positive regions* — there's no foreground to segment in a background region, so L_mask there is meaningless — and within a positive region, only on the ground-truth-class channel, as established. The mask target for a positive region is the ground-truth mask intersected with the region and resampled to m×m, using the same aligned sampling so the target and the prediction live on the same grid. Sample regions per image with a 1:3 positive-to-negative ratio. The three losses are just summed with equal weight; I don't see a reason to tune a weighting and the system isn't delicate about it.

One subtlety that's easy to get wrong: training and inference don't run the mask branch on the same regions, and that's deliberate. During training the mask branch runs on the *positive proposals* (so it sees decent but imperfect regions and learns to be robust). At test time, though, I first run only the class+box branches over all proposals, do non-maximum suppression, keep the top detections (say the highest-scoring 100 boxes), and *then* run the mask branch on those final boxes. Two payoffs: I compute masks only for the handful of boxes I actually report, so the mask branch adds only a small overhead on top of the detector; and the boxes it runs on are the refined, high-quality detections rather than raw proposals, which is *better* for mask quality, not just cheaper. For each detection I read out the single mask channel k where k is the class the classification branch predicted — the other K−1 masks are ignored — resize that m×m map to the detection box, and threshold at 0.5 to get a binary instance mask.

Now let me check the multi-task interaction, because adding a mask branch could in principle drag down the box numbers (shared features, competing gradients). My expectation from the decoupling argument is the opposite for box vs mask: since the tasks are complementary and the mask gradient pushes the shared features toward better spatial sensitivity, training the box and mask jointly should *help* the box AP a little compared to training boxes alone with the same aligned extractor. (If I strip the mask branch but keep the aligned extractor, I'd get a box detector that already beats the old rounding-based one purely from alignment; adding the mask branch back should add a bit more box AP on top, attributable solely to multi-task training.)

There's one more thing I get almost for free, and it's a good stress test of the "it's just a per-instance spatial map" framing. A human-pose keypoint is also a per-instance spatial target: a single location in the region. So model each of the K keypoints of an instance as its own one-hot m×m mask, where exactly *one* pixel is foreground. But notice the loss has to change here, and the change is instructive. For a segmentation mask, *many* pixels are foreground and they're independent, so per-pixel sigmoid + binary cross-entropy is right. For a keypoint, exactly *one* pixel is correct out of m², so the natural target is one-hot and the natural loss is an m²-way softmax cross-entropy — the pixels now *should* compete, because the answer is "which single pixel." That's the same machinery (a per-region spatial branch) with the loss matched to the structure of the target: independent foreground → sigmoid; exactly-one → softmax. Keypoints want finer localization than masks, so push the output resolution higher (something like 56×56). The K keypoint types stay independent of each other, one map each, same as the K class masks stayed independent.

Let me now write the second stage end to end, grounded in how this actually gets built. The aligned extractor first.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align          # continuous-coordinate bilinear RoIAlign
from torchvision.ops import MultiScaleRoIAlign # routes RoIs to the right pyramid level

  # RoIAlign = extract a fixed output_size feature per RoI by sampling at
  # CONTINUOUS coordinates via bilinear interpolation -- the key fix.
  # output_size 14 for masks (then deconv -> 28), 7 for the box head.
mask_roi_align = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"],
                                    output_size=14, sampling_ratio=2)
box_roi_align  = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"],
                                    output_size=7,  sampling_ratio=2)
```

The box head is just the existing parallel class+box siblings on the aligned 7×7 feature:

```python
class TwoMLPHead(nn.Module):           # fc6 / fc7 over the pooled region feature
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        return F.relu(self.fc7(F.relu(self.fc6(x))))

class FastRCNNPredictor(nn.Module):    # the two siblings: class + per-class box
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.cls_score(x), self.bbox_pred(x)
```

The third sibling, the mask branch: a fully-convolutional stack (keep spatial layout), a deconv to upsample, then 1×1 conv to K class masks. No fc, no softmax across classes.

```python
class MaskRCNNHeads(nn.Sequential):    # 4x conv 3x3, 256 channels, ReLU -- stays spatial
    def __init__(self, in_channels, layers=(256, 256, 256, 256), dilation=1):
        d = dilation; blocks = []; nf = in_channels
        for f in layers:
            blocks += [nn.Conv2d(nf, f, 3, 1, padding=d, dilation=d), nn.ReLU(inplace=True)]
            nf = f
        super().__init__(*blocks)

class MaskRCNNPredictor(nn.Sequential):  # deconv 2x2 stride2 (14->28) -> 1x1 conv -> K maps
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0), nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, num_classes, 1, 1, 0),          # K class-specific masks
        )
```

The mask loss: per-pixel sigmoid + binary cross-entropy, on the ground-truth-class channel only. The target is the GT mask resampled onto the proposal with the *same* aligned sampling, so prediction and target share a grid.

```python
def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # crop+resize each GT mask to the MxM proposal grid via the SAME aligned sampler
    rois = torch.cat([matched_idxs.to(boxes)[:, None], boxes], dim=1)
    gt = gt_masks[:, None].to(rois)
    return roi_align(gt, rois, (M, M), spatial_scale=1.0)[:, 0]

def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, matched_idxs):
    M = mask_logits.shape[-1]
    labels  = torch.cat([gl[idx] for gl, idx in zip(gt_labels, matched_idxs)])
    targets = torch.cat([project_masks_on_boxes(m, p, i, M)
                         for m, p, i in zip(gt_masks, proposals, matched_idxs)])
    if targets.numel() == 0:
        return mask_logits.sum() * 0
    idx = torch.arange(labels.shape[0], device=labels.device)
    # pick ONLY the GT-class channel -> no competition among classes; sigmoid via BCE-with-logits
    return F.binary_cross_entropy_with_logits(mask_logits[idx, labels], targets)
```

Mask read-out at inference: sigmoid, then take the channel of the *predicted* class (the rest are discarded), one mask per detection.

```python
def maskrcnn_inference(mask_logits, pred_labels):
    probs = mask_logits.sigmoid()
    n = probs.shape[0]
    labels = torch.cat(pred_labels)
    idx = torch.arange(n, device=labels.device)
    probs = probs[idx, labels][:, None]                  # keep only predicted-class mask
    return probs.split([len(l) for l in pred_labels], 0)
```

And the second stage wiring it all together — same stage one (RPN) as the base detector, three parallel siblings in stage two, mask loss only on positive proposals in training, mask branch only on the post-NMS top detections at test:

```python
class RoIHeads(nn.Module):
    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, reg_targets = \
                self.select_training_samples(proposals, targets)   # 1:3 pos:neg, IoU>=0.5 -> positive

        box_feat = self.box_roi_pool(features, proposals, image_shapes)   # ALIGNED 7x7
        cls_logits, box_reg = self.box_predictor(self.box_head(box_feat))

        result, losses = [], {}
        if self.training:
            lc, lb = fastrcnn_loss(cls_logits, box_reg, labels, reg_targets)
            losses = {"loss_classifier": lc, "loss_box_reg": lb}
        else:
            boxes, scores, lbls = self.postprocess_detections(            # NMS -> top-100 boxes
                cls_logits, box_reg, proposals, image_shapes)
            result = [{"boxes": b, "labels": l, "scores": s}
                      for b, l, s in zip(boxes, lbls, scores)]

        # --- third sibling: mask branch ---
        if self.training:
            mask_props, pos_idx = [], []
            for i in range(len(proposals)):                              # masks only on positives
                pos = torch.where(labels[i] > 0)[0]
                mask_props.append(proposals[i][pos]); pos_idx.append(matched_idxs[i][pos])
        else:
            mask_props = [r["boxes"] for r in result]                   # masks on FINAL detections

        mask_feat   = self.mask_roi_pool(features, mask_props, image_shapes)   # ALIGNED 14x14
        mask_logits = self.mask_predictor(self.mask_head(mask_feat))

        if self.training:
            lm = maskrcnn_loss(mask_logits, mask_props,
                               [t["masks"] for t in targets],
                               [t["labels"] for t in targets], pos_idx)
            losses["loss_mask"] = lm                                     # L = L_cls + L_box + L_mask
        else:
            for m, r in zip(maskrcnn_inference(mask_logits, [r["labels"] for r in result]), result):
                r["masks"] = m
        return result, losses
```

So the causal chain, start to finish: I wanted per-instance masks, so I bolted a third sibling onto the two-stage detector's second stage instead of building a new pipeline — keeping the parallel-siblings design that made detection simple. To avoid the overlap failures that plague entangled systems, I made the mask branch decoupled: K independent per-class masks with per-pixel sigmoid and a binary loss applied only on the ground-truth class, so classes never compete and the existing classification branch alone decides the label. I kept the branch fully convolutional so the m×m spatial layout survives. Then the real obstacle surfaced — the region-feature extractor rounds coordinates twice, which is invisible to classification but fatal to a pixel-to-pixel mask — so I replaced rounding with continuous coordinates plus bilinear sampling, an aligned extractor, and reasoned (via the quantize-then-bilinear control and the stride dependence) that alignment, not bilinear per se, is what matters, with the payoff growing at large strides. Equal-weighted three-task loss, masks on positives in training and on the top detections at test, read out only the predicted class's mask — and the same per-region spatial branch, with softmax swapped in for sigmoid because the target is now exactly-one-pixel, extends straight to keypoints.
