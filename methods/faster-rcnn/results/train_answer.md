By 2015, region-based object detectors had gotten dramatically faster. R-CNN's original sin—running a full CNN separately for each of ~2000 proposals—had been fixed by SPPnet and then Fast R-CNN. Those methods compute one shared convolutional feature map for the whole image and pool a fixed-size feature for each candidate region, so the per-region network becomes a thin, fast head. The trouble is that this optimization simply shifted the bottleneck. The candidate regions still came from an external module such as Selective Search or EdgeBoxes, which runs on the CPU over hand-engineered low-level cues like superpixels, color, and texture. Selective Search costs about two seconds per image, and even the faster EdgeBoxes takes roughly as long as the detector's own convolution pass. Worse, the proposer recomputes its own feature representation from scratch and never uses the rich convolutional features the detector has already paid for. The opportunity, then, is not to build a marginally faster standalone proposer, but to make proposal generation nearly free by turning it into a small head on the detector's own shared feature map.

The method that does this is Faster R-CNN. Its core addition is a Region Proposal Network (RPN) that lives on top of the same convolutional backbone as the Fast R-CNN detector. The RPN is fully convolutional: it slides a small network over the shared feature map and, at every spatial location, emits a set of objectness scores and bounding-box refinements. Concretely, the feature map is passed through a 3×3 convolution with ReLU, and then through two sibling 1×1 convolutions. One outputs 2k scores per location, encoding object-vs-background for k anchors; the other outputs 4k box corrections per location. Because the head uses shared weights everywhere, it is translation invariant by construction, and because it reuses the backbone convolutions that the detector needs anyway, its marginal cost is tiny.

The main design challenge is handling objects at many scales and aspect ratios from a single fixed-size feature map. Faster R-CNN solves this with anchors. At each sliding location it pins down k fixed reference boxes—by default nine anchors formed from three scales (128², 256², 512²) and three aspect ratios (1:1, 1:2, 2:1). Each anchor has its own dedicated box regressor, which predicts a small correction relative to that anchor. The correction is parameterized in a scale-invariant way: center offsets normalized by the anchor's width and height, and width/height corrections in log space. This lets a fixed-size feature cell emit proposals for both small and large objects without paying for image pyramids or filter pyramids. The scale and shape information is carried entirely by the anchor references, not by extra convolutions.

Training the RPN uses a multi-task loss. Every anchor is labeled positive if it overlaps any ground-truth box by at least 0.7 IoU, or if it is the highest-overlap anchor for some ground-truth box; it is negative if its IoU is below 0.3 with all ground-truth boxes; otherwise it is ignored. The loss sums a log loss on objectness over sampled anchors and a smooth-L1 regression loss gated to positive anchors only, normalized so the two terms are balanced. Cross-boundary anchors are ignored during training because they otherwise destabilize optimization. At test time the predicted deltas are applied to the tiled anchors, the resulting boxes are clipped to the image, and non-maximum suppression at IoU 0.7 reduces the roughly 20,000 candidates down to a short, high-quality proposal list for the detector.

Finally, the RPN and detector must share a single backbone. This is done by alternating training: first train the RPN; then train a separate Fast R-CNN detector on those proposals; then reinitialize the RPN on top of the detector's conv layers and fine-tune only the RPN-specific layers while keeping the conv layers frozen; finally fine-tune only the detector head while still freezing the shared convs. The result is one unified network where a single convolutional pass produces both proposals and final detections. A single-scale 1000×600 image yields about 20,000 anchors, of which roughly 6,000 are kept during training after dropping boundary-crossing ones.

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
