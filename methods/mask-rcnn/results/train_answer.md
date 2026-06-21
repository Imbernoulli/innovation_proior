I want to detect every object in an image and hand back a pixel-accurate mask for each one — not a single per-pixel labeling of the scene, but a separate mask per instance, so that two people standing shoulder to shoulder come out as two masks, not one merged blob. That instance requirement is the whole difficulty: the task is detection (find, classify, and localize each thing) and segmentation (label pixels) at once, and the segmentation has to respect instances, separating overlapping same-class objects rather than fusing them. A good solution should match the speed and one-stage-training simplicity that detection frameworks already enjoy, be accurate at strict high-IoU mask overlap where a wrong boundary makes a mask useless, survive the hard case of many touching same-class objects, and stay general enough to swap backbones and extend to other instance-level tasks.

The existing options each miss one of these. The two-stage region detectors solve detection well — run a backbone once, propose regions with a Region Proposal Network, then classify and box-refine each region — but they produce boxes, not masks. Fully convolutional networks do per-pixel labeling well but have no native notion of "which instance," so touching same-class objects merge; and their per-pixel softmax couples segmentation with classification. The instance-segmentation systems that do exist all entangle the mask with the class: the segment-proposal family predicts masks first (via a fully-connected layer that discards spatial structure) and then classifies them; the cascade family runs three dependent stages, propose-then-mask-then-categorize; the fully-convolutional-instance family predicts one shared bank of position-sensitive channels that jointly encode class, box, and mask. The common consequence is that these are slow, multi-stage, and — more worryingly — when mask and class share a representation or compete, overlapping same-class instances produce systematic artifacts and spurious edges exactly where two objects touch. The entanglement that buys them their structure is also what breaks them on the case I most need to get right.

I propose Mask R-CNN. The core move is to resist over-engineering: take the two-stage region detector unchanged — stage 1 the RPN, stage 2 the per-region class and box siblings — and add a *third sibling* to stage 2, a small branch that outputs a mask, computed in parallel with classification and box regression. The whole net still trains end to end in one stage with a multi-task loss
$$L = L_{\text{cls}} + L_{\text{box}} + L_{\text{mask}},$$
where $L_{\text{cls}}$ is softmax cross-entropy over the $K{+}1$ classes and $L_{\text{box}}$ is a smooth-$L_1$ on the box-regression outputs of the ground-truth class only. The three losses are summed with equal weight; the system is not delicate about a weighting, so I do not tune one. Keeping the parallel-siblings shape is deliberate: that design is exactly what once turned the slow multi-stage region method (a CNN per warped region, then an SVM, then a separate regressor) into a single fast network, and the mask branch is just one more head reading the same pooled region feature.

What makes it work is three design choices, and the first is how the mask branch is supervised. The naive thing is to copy the semantic-segmentation FCN wholesale — output a per-pixel softmax over the $K$ classes and train with multinomial cross-entropy. But per-pixel softmax makes the classes *compete* at every pixel: the foreground probability for "dog" is pushed down when "cat" rises, because they are normalized together. By the time the mask branch runs on a region, deciding the class is already the classification sibling's job. The mask branch does not need to also answer "dog or cat?"; it only needs "is this pixel part of *the* object in this region, or background?" Forcing class competition inside the mask re-couples segmentation with classification — precisely the entanglement that causes the overlap artifacts. So I flip it: the branch outputs $K$ separate masks, one per class, each an $m \times m$ map ($K \cdot m^2$ numbers per region), and applies a per-pixel *sigmoid* to each independently, with no normalization across classes. For a region whose ground-truth class is $k$, the mask loss is defined only on the $k$-th map, as the average binary cross-entropy over its $m^2$ pixels,
$$L_{\text{mask}} = -\frac{1}{m^2}\sum_{i,j}\Big[\, y_{ij}\log\sigma(z^k_{ij}) + (1-y_{ij})\log\big(1-\sigma(z^k_{ij})\big)\,\Big],$$
where $z^k$ is the $k$-th output map and $y \in \{0,1\}$ is the ground-truth mask cropped to the region. The other $K-1$ masks receive no gradient from this region. This is a clean division of labor: there is no competition among classes, each class's mask is learned as an independent foreground/background problem, and the classification branch is the sole thing that decides the label and hence which of the $K$ masks to read out. That decoupling is the concrete reason the system does not suffer the overlapping-instance failures of the entangled methods. The mask loss is also defined only on positive RoIs (IoU $\geq 0.5$ with a ground-truth box) — there is no foreground to segment in a background region — and the per-class identity of the mask carries no classification load, so even a single class-agnostic mask stays close, confirming that decoupling, not class-specific shape, is what matters.

The second choice is to keep the mask branch fully convolutional. A mask is a *spatial* output, so it should be produced by convolutions that preserve the $m \times m$ layout, keeping every output cell in explicit pixel correspondence with a feature cell. If I collapse the region feature into a flat vector and ask a fully-connected layer to emit $m^2$ numbers, I throw away the very 2-D structure the mask is, and the fc layer has to re-learn that structure from scratch in its weights — more parameters, worse accuracy. Concretely the head is a short stack of $3\times 3$ convolutions that hold resolution, a $2\times 2$ transposed convolution (deconvolution) with stride 2 that upsamples (e.g. $14 \to 28$), then a $1\times 1$ conv producing the $K$ output maps.

The third choice is the one that turned out to be the real obstacle, and it is not in the head at all — it is in how the region feature is pulled out of the shared map. The standard extractor, RoIPool, quantizes coordinates *twice*. A proposal box in continuous image coordinates, say left edge $x = 134.7$ on a stride-16 map, is first mapped to the feature grid by dividing and *rounding* ($134.7/16 = 8.42 \to$ cell 8), then split into an $m \times m$ grid whose bin boundaries are *rounded* again, then max-pooled inside each integer bin. Each rounding snaps the region by up to half a feature cell — about 8 pixels at stride 16, 16 at stride 32. For classification this never mattered, and that is *why* nobody fixed it: a class label is invariant to a few pixels of translation, so the classifier shrugs it off. But the mask branch relies on an explicit cell-to-location correspondence; if extraction has already smeared the region by half a bin and snapped its internal grid, then "output pixel $(3,4)$" no longer maps to a definite place in the image, and the correspondence is broken before the mask branch even runs. The rounding that is harmless for boxes is exactly what poisons masks.

So I replace it with RoIAlign, which removes all quantization. Do not round the box: use $x/16 = 8.42$ and keep the fraction. Do not round the bins: divide the continuous region evenly into the $m \times m$ grid with floating boundaries. Inside each bin place a small regular grid of sample points (e.g. $2\times 2$) and read each one by bilinear interpolation from its four neighboring integer grid points — a smooth, differentiable read with no rounding anywhere. For a region with feature-space top-left $(x_0, y_0)$ and size $(w,h)$, the sample indexed $(iy, ix)$ in bin $(p,q)$ of an $m\times m$ grid, with $G$ samples per axis, sits at
$$y = y_0 + q\cdot\frac{h}{m} + (iy + 0.5)\cdot\frac{h/m}{G}, \qquad x = x_0 + p\cdot\frac{w}{m} + (ix + 0.5)\cdot\frac{w/m}{G},$$
and the $G^2$ reads in a bin are averaged (max works too). Every coordinate is continuous; the output is a fixed $m\times m$ feature faithfully aligned with the region's true location. The result is barely sensitive to the number of sample points or to average-vs-max — *as long as* nothing is quantized — which is the tell that the active ingredient is alignment, not the bilinear sampler. The sharp control confirms it: an extractor that resamples bilinearly *but still quantizes the region first* (RoIWarp) performs essentially like plain RoIPool, far below the aligned version, pinning the cause on alignment. And because the misalignment is half a feature cell, the gain grows with stride — larger at stride 32 than stride 16, and largest at the strict high-IoU metric where boundary precision is everything — so a stride-32 feature, long dismissed as too coarse, becomes competitive once the quantization it amplified is gone.

Training and inference deliberately run the mask branch on different regions. In training it runs on the positive proposals (RoIs sampled 1:3 positive-to-negative), so it sees decent but imperfect regions and learns to be robust; the mask target is the ground-truth mask resampled to $m\times m$ with the *same* RoIAlign sampling, so prediction and target share a grid. At test time I first run only the class and box branches over all proposals, apply non-maximum suppression, keep the top detections (about 100), and *then* run the mask branch on those final boxes — which both adds only small overhead, since masks are computed for just the reported boxes, and improves mask quality, since those boxes are refined high-quality detections rather than raw proposals. For each detection I read out only the channel $k$ of its predicted class, discard the other $K-1$ masks, resize the $m\times m$ map to the box, and threshold at 0.5. Because the tasks are complementary and the mask gradient pushes the shared features toward better spatial sensitivity, joint training tends to help box AP a little rather than hurt it. And the framing generalizes almost for free: a human-pose keypoint is also a per-instance spatial target, a single location, so model each of $K$ keypoints as its own one-hot $m\times m$ mask — but here exactly one pixel is correct, so the pixels *should* compete, and the loss becomes an $m^2$-way softmax cross-entropy at higher resolution (about $56\times 56$). Same per-region spatial branch, with the loss matched to the structure of the target: independent foreground wants sigmoid, exactly-one-pixel wants softmax.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, MultiScaleRoIAlign


# ---- RoIAlign-style region-feature extraction (continuous coordinates + bilinear sampling) ----
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
    # matched_idxs selects which GT-mask plane each positive proposal is cropped from.
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
