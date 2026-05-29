# Faster R-CNN

## Problem it solves

In a region-based object detector, a CNN classifier became cheap once convolutional features were computed once per image and pooled per region (SPPnet, Fast R-CNN). That exposed the *external region-proposal* step (Selective Search ~2 s/image, EdgeBoxes ~0.2 s/image) as the new test-time bottleneck — and that step runs on the CPU over hand-engineered features, sharing nothing with the detector. Faster R-CNN makes proposals nearly free by computing them with a small network on top of the detector's *own* shared convolutional feature map.

## Key idea

A **Region Proposal Network (RPN)** is a fully convolutional head on the shared conv feature map. At every cell it slides a small network (3×3 conv + ReLU) and reads off two sibling 1×1 conv layers: a *cls* layer (objectness) and a *reg* layer (box deltas). Multiple scales/aspect ratios are handled not by image or filter pyramids but by **anchors** — k fixed reference boxes per location (default 3 scales {128², 256², 512²} × 3 ratios {1:1, 1:2, 2:1} = 9). Each anchor has its own regressor that predicts a correction taking it to a nearby object, so a fixed-size feature can emit boxes of very different sizes. The RPN and the Fast R-CNN detector then **share one backbone**, so at test time a single conv pass serves both.

## Final method

**RPN head.** Shared convs → 3×3 conv (256-d for ZF, 512-d for VGG) → ReLU → two 1×1 convs: *cls* with 2k outputs, *reg* with 4k outputs. With a feature map of W×H there are W·H·k anchors.

**Box parameterization (from R-CNN).** For an anchor (x_a, y_a, w_a, h_a), predicted box (x, y, w, h), and gt (x*, y*, w*, h*):

t_x = (x − x_a)/w_a, t_y = (y − y_a)/h_a, t_w = log(w/w_a), t_h = log(h/h_a)

and invert with x = t_x w_a + x_a, y = t_y h_a + y_a, w = w_a exp(t_w), h = h_a exp(t_h). The k regressors do **not** share weights — each anchor shape owns its regressor, supplying the scale information a fixed-size feature lacks.

**Labels.** Anchor is positive if IoU ≥ 0.7 with any gt, *or* it is tied for the highest IoU to some gt (fallback so every gt gets a positive); negative if IoU < 0.3 with all gt; otherwise ignored.

**Multi-task loss** (per image):

L({p_i}, {t_i}) = (1/N_cls) Σ_i L_cls(p_i, p*_i) + λ (1/N_reg) Σ_i p*_i L_reg(t_i, t*_i)

L_cls = log loss over object/not-object; L_reg = smooth-L1(t_i − t*_i); p*_i gates the box loss to positives. N_cls = 256 (minibatch), N_reg ≈ 2400 (anchor locations), λ = 10 to balance the two normalized terms.

**Training the RPN.** Image-centric sampling: 1 image/minibatch, 256 sampled anchors, ≤1:1 positive:negative (pad with negatives). New layers ~ N(0, 0.01); shared convs from ImageNet pretraining. SGD, momentum 0.9, weight decay 5e-4, lr 1e-3 for 60k then 1e-4 for 20k. Cross-boundary anchors are **ignored in training** (they break convergence); at test time proposals are **clipped** to the image. NMS at IoU 0.7 on the proposals → ~2000, then top-N (e.g. 300) to the detector.

**Sharing features (4-step alternating training).** (1) Train RPN from ImageNet init. (2) Train a separate Fast R-CNN from ImageNet init on step-1 proposals (convs not yet shared). (3) Re-init RPN from the detector, **freeze** the shared convs, fine-tune only RPN-specific layers (now shared). (4) Keep convs frozen, fine-tune only Fast R-CNN-specific layers. Result: one backbone, two heads, one unified network. (An *approximate joint training* variant trains the merged net in one loop, treating proposals as fixed RoIs and ignoring the gradient w.r.t. proposal coordinates.)

**Implementation.** Single scale, shorter side 600 px; total conv stride 16. ~20000 anchors per 1000×600 image, ~6000 after dropping cross-boundary.

## Code

The core NumPy pieces are:

```python
import numpy as np

# ---- anchors: k=9 reference boxes for one feature-map cell ----------------
def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.arange(3, 6)):
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    return np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                      for i in range(ratio_anchors.shape[0])])

def _whctrs(a):
    w = a[2] - a[0] + 1; h = a[3] - a[1] + 1
    return w, h, a[0] + 0.5 * (w - 1), a[1] + 0.5 * (h - 1)

def _mkanchors(ws, hs, xc, yc):
    ws, hs = ws[:, None], hs[:, None]
    return np.hstack((xc - 0.5 * (ws - 1), yc - 0.5 * (hs - 1),
                      xc + 0.5 * (ws - 1), yc + 0.5 * (hs - 1)))

def _ratio_enum(anchor, ratios):       # vary aspect ratio, keep area ~constant
    w, h, xc, yc = _whctrs(anchor)
    size = w * h
    ws = np.round(np.sqrt(size / ratios)); hs = np.round(ws * ratios)
    return _mkanchors(ws, hs, xc, yc)

def _scale_enum(anchor, scales):       # vary scale -> 128/256/512 family
    w, h, xc, yc = _whctrs(anchor)
    return _mkanchors(w * scales, h * scales, xc, yc)

# ---- box parameterization (R-CNN) ----------------------------------------
def bbox_transform(ex, gt):
    ew = ex[:, 2]-ex[:, 0]+1.0; eh = ex[:, 3]-ex[:, 1]+1.0
    ecx = ex[:, 0]+0.5*ew; ecy = ex[:, 1]+0.5*eh
    gw = gt[:, 2]-gt[:, 0]+1.0; gh = gt[:, 3]-gt[:, 1]+1.0
    gcx = gt[:, 0]+0.5*gw; gcy = gt[:, 1]+0.5*gh
    return np.vstack(((gcx-ecx)/ew, (gcy-ecy)/eh,
                      np.log(gw/ew), np.log(gh/eh))).T

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    w = boxes[:, 2]-boxes[:, 0]+1.0; h = boxes[:, 3]-boxes[:, 1]+1.0
    cx = boxes[:, 0]+0.5*w; cy = boxes[:, 1]+0.5*h
    dx, dy, dw, dh = deltas[:, 0::4], deltas[:, 1::4], deltas[:, 2::4], deltas[:, 3::4]
    px = dx*w[:, None]+cx[:, None]; py = dy*h[:, None]+cy[:, None]
    pw = np.exp(dw)*w[:, None]; ph = np.exp(dh)*h[:, None]
    out = np.zeros_like(deltas)
    out[:, 0::4] = px-0.5*pw; out[:, 1::4] = py-0.5*ph
    out[:, 2::4] = px+0.5*pw; out[:, 3::4] = py+0.5*ph
    return out

def clip_boxes(boxes, im_shape):
    boxes[:, 0::4] = np.clip(boxes[:, 0::4], 0, im_shape[1]-1)
    boxes[:, 1::4] = np.clip(boxes[:, 1::4], 0, im_shape[0]-1)
    boxes[:, 2::4] = np.clip(boxes[:, 2::4], 0, im_shape[1]-1)
    boxes[:, 3::4] = np.clip(boxes[:, 3::4], 0, im_shape[0]-1)
    return boxes

# ---- training-time anchor target assignment ------------------------------
# Assumed primitives from the detector stack: bbox_overlaps(boxes, query_boxes), nms(dets, thresh).
def anchor_targets(anchors, gt_boxes, im_info, feat_h, feat_w, feat_stride=16,
                   pos_thr=0.7, neg_thr=0.3, batch=256, fg_frac=0.5):
    sx = np.arange(feat_w)*feat_stride; sy = np.arange(feat_h)*feat_stride
    sx, sy = np.meshgrid(sx, sy)
    shifts = np.vstack((sx.ravel(), sy.ravel(), sx.ravel(), sy.ravel())).T
    A, K = anchors.shape[0], shifts.shape[0]
    all_anchors = (anchors.reshape(1, A, 4) +
                   shifts.reshape(1, K, 4).transpose(1, 0, 2)).reshape(K*A, 4)
    inside = np.where((all_anchors[:, 0] >= 0) & (all_anchors[:, 1] >= 0) &
                      (all_anchors[:, 2] < im_info[1]) &
                      (all_anchors[:, 3] < im_info[0]))[0]   # drop cross-boundary
    anc = all_anchors[inside]
    labels = np.full((len(inside),), -1, np.float32)
    ov = bbox_overlaps(anc, gt_boxes)
    argmax = ov.argmax(1); max_ov = ov[np.arange(len(inside)), argmax]
    gt_argmax = np.where(ov == ov.max(0))[0]
    labels[max_ov < neg_thr] = 0
    labels[gt_argmax] = 1
    labels[max_ov >= pos_thr] = 1
    num_fg = int(fg_frac*batch); fg = np.where(labels == 1)[0]
    if len(fg) > num_fg:
        labels[np.random.choice(fg, len(fg)-num_fg, replace=False)] = -1
    num_bg = batch - np.sum(labels == 1); bg = np.where(labels == 0)[0]
    if len(bg) > num_bg:
        labels[np.random.choice(bg, len(bg)-num_bg, replace=False)] = -1
    targets = bbox_transform(anc, gt_boxes[argmax, :4])
    inside_w = np.zeros((len(inside), 4), np.float32)
    inside_w[labels == 1, :] = 1.0          # gate reg loss to positives
    return labels, targets, inside_w, inside

# ---- test-time proposal generation ---------------------------------------
def generate_proposals(scores, deltas, anchors, im_info, feat_h, feat_w,
                       feat_stride=16, pre_nms=6000, post_nms=300,
                       nms_thr=0.7, min_size=16):
    sx = np.arange(feat_w)*feat_stride; sy = np.arange(feat_h)*feat_stride
    sx, sy = np.meshgrid(sx, sy)
    shifts = np.vstack((sx.ravel(), sy.ravel(), sx.ravel(), sy.ravel())).T
    A, K = anchors.shape[0], shifts.shape[0]
    anc = (anchors.reshape(1, A, 4) +
           shifts.reshape(1, K, 4).transpose(1, 0, 2)).reshape(K*A, 4)
    if scores.shape[1] == 2 * A:          # canonical bg/fg softmax layout
        scores = scores[:, A:, :, :]
    deltas = deltas.transpose(0, 2, 3, 1).reshape(-1, 4)
    scores = scores.transpose(0, 2, 3, 1).reshape(-1, 1)
    proposals = clip_boxes(bbox_transform_inv(anc, deltas), im_info[:2])
    keep = _filter_boxes(proposals, min_size*im_info[2])
    proposals, scores = proposals[keep], scores[keep]
    order = scores.ravel().argsort()[::-1]
    if pre_nms > 0:
        order = order[:pre_nms]
    proposals, scores = proposals[order], scores[order]
    keep = nms(np.hstack((proposals, scores)), nms_thr)
    if post_nms > 0:
        keep = keep[:post_nms]
    return proposals[keep], scores[keep]

def _filter_boxes(boxes, min_size):
    ws = boxes[:, 2]-boxes[:, 0]+1; hs = boxes[:, 3]-boxes[:, 1]+1
    return np.where((ws >= min_size) & (hs >= min_size))[0]
```
