By 2015, region-based object detectors had gotten dramatically faster. R-CNN's original sin—running a full CNN separately for each of ~2000 proposals—had been fixed by SPPnet and then Fast R-CNN. Those methods compute one shared convolutional feature map for the whole image and pool a fixed-size feature for each candidate region, so the per-region network becomes a thin, fast head. The trouble is that this optimization simply shifted the bottleneck. The candidate regions still came from an external module such as Selective Search or EdgeBoxes, which runs on the CPU over hand-engineered low-level cues like superpixels, color, and texture. Selective Search costs about two seconds per image, and even the faster EdgeBoxes takes roughly as long as the detector's own convolution pass. Worse, the proposer recomputes its own feature representation from scratch and never uses the rich convolutional features the detector has already paid for. The opportunity, then, is not to build a marginally faster standalone proposer, but to make proposal generation nearly free by turning it into a small head on the detector's own shared feature map.

The method that does this is Faster R-CNN. Its core addition is a Region Proposal Network (RPN) that lives on top of the same convolutional backbone as the Fast R-CNN detector. The RPN is fully convolutional: it slides a small network over the shared feature map and, at every spatial location, emits a set of objectness scores and bounding-box refinements. Concretely, the feature map is passed through a 3×3 convolution with ReLU, and then through two sibling 1×1 convolutions. One outputs 2k scores per location, encoding object-vs-background for k anchors; the other outputs 4k box corrections per location. Because the head uses shared weights everywhere, it is translation invariant by construction, and because it reuses the backbone convolutions that the detector needs anyway, its marginal cost is tiny.

The main design challenge is handling objects at many scales and aspect ratios from a single fixed-size feature map. Faster R-CNN solves this with anchors. At each sliding location it pins down k fixed reference boxes—by default nine anchors formed from three scales (128², 256², 512²) and three aspect ratios (1:1, 1:2, 2:1). Each anchor has its own dedicated box regressor, which predicts a small correction relative to that anchor. The correction is parameterized in a scale-invariant way: center offsets normalized by the anchor's width and height, and width/height corrections in log space. This lets a fixed-size feature cell emit proposals for both small and large objects without paying for image pyramids or filter pyramids. The scale and shape information is carried entirely by the anchor references, not by extra convolutions.

Training the RPN uses a multi-task loss. Every anchor is labeled positive if it overlaps any ground-truth box by at least 0.7 IoU, or if it is the highest-overlap anchor for some ground-truth box; it is negative if its IoU is below 0.3 with all ground-truth boxes; otherwise it is ignored. The loss sums a log loss on objectness over sampled anchors and a smooth-L1 regression loss gated to positive anchors only, normalized so the two terms are balanced. Cross-boundary anchors are ignored during training because they otherwise destabilize optimization. At test time the predicted deltas are applied to the tiled anchors, the resulting boxes are clipped to the image, and non-maximum suppression at IoU 0.7 reduces the roughly 20,000 candidates down to a short, high-quality proposal list for the detector.

Finally, the RPN and detector must share a single backbone. This is done by alternating training: first train the RPN; then train a separate Fast R-CNN detector on those proposals; then reinitialize the RPN on top of the detector's conv layers and fine-tune only the RPN-specific layers while keeping the conv layers frozen; finally fine-tune only the detector head while still freezing the shared convs. The result is one unified network where a single convolutional pass produces both proposals and final detections. A single-scale 1000×600 image yields about 20,000 anchors, of which roughly 6,000 are kept during training after dropping boundary-crossing ones.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT_STRIDE = 16

def generate_anchors(base_size=16, ratios=(0.5, 1.0, 2.0),
                     scales=(128, 256, 512)):
    """Return k=9 reference boxes for one feature-map cell."""
    base = np.array([1, 1, base_size, base_size]) - 1
    w, h = base[2] - base[0] + 1, base[3] - base[1] + 1
    size = w * h
    ws = np.round(np.sqrt(size / np.array(ratios)))
    hs = np.round(ws * np.array(ratios))
    ratio_anchors = np.stack([
        np.array([0, 0, ws[i] - 1, hs[i] - 1]) for i in range(len(ratios))
    ])
    anchors = []
    for a in ratio_anchors:
        aw, ah = a[2] - a[0] + 1, a[3] - a[1] + 1
        for s in scales:
            sw, sh = aw * (s / base_size), ah * (s / base_size)
            anchors.append([-0.5 * (sw - 1), -0.5 * (sh - 1),
                            0.5 * (sw - 1), 0.5 * (sh - 1)])
    return np.array(anchors, dtype=np.float32)

def bbox_transform(anchors, gt):
    """R-CNN parameterization: scale-invariant deltas from anchors to gt."""
    aw = anchors[:, 2] - anchors[:, 0] + 1
    ah = anchors[:, 3] - anchors[:, 1] + 1
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah
    gw = gt[:, 2] - gt[:, 0] + 1
    gh = gt[:, 3] - gt[:, 1] + 1
    gcx = gt[:, 0] + 0.5 * gw
    gcy = gt[:, 1] + 0.5 * gh
    dx = (gcx - acx) / aw
    dy = (gcy - acy) / ah
    dw = np.log(gw / aw)
    dh = np.log(gh / ah)
    return np.stack([dx, dy, dw, dh], axis=1)

def bbox_transform_inv(anchors, deltas):
    """Apply predicted deltas to anchors to recover box coordinates."""
    aw = anchors[:, 2] - anchors[:, 0] + 1
    ah = anchors[:, 3] - anchors[:, 1] + 1
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah
    px = deltas[:, 0] * aw + acx
    py = deltas[:, 1] * ah + acy
    pw = np.exp(deltas[:, 2]) * aw
    ph = np.exp(deltas[:, 3]) * ah
    return np.stack([
        px - 0.5 * pw, py - 0.5 * ph,
        px + 0.5 * pw, py + 0.5 * ph
    ], axis=1)

def clip_boxes(boxes, im_shape):
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, im_shape[1] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, im_shape[0] - 1)
    return boxes

class RegionProposalNetwork(nn.Module):
    """Faster R-CNN RPN head: shared conv -> 3x3 -> cls + reg 1x1 heads."""
    def __init__(self, in_channels=512, feat_stride=16,
                 ratios=(0.5, 1.0, 2.0), scales=(128, 256, 512)):
        super().__init__()
        self.feat_stride = feat_stride
        self.anchors = generate_anchors(feat_stride, ratios, scales)
        self.num_anchors = self.anchors.shape[0]
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, 2 * self.num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * self.num_anchors, 1)
        for layer in (self.conv, self.cls_logits, self.bbox_pred):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features, im_info):
        """
        features: (B, C, H, W) shared conv feature map.
        im_info: (height, width, scale) of the input image.
        Returns proposals, scores during inference; raw logits during training.
        """
        x = F.relu(self.conv(features))
        cls_score = self.cls_logits(x)
        bbox_deltas = self.bbox_pred(x)
        if not self.training:
            return self.generate_proposals(cls_score, bbox_deltas, im_info)
        return cls_score, bbox_deltas

    def generate_proposals(self, cls_score, bbox_deltas, im_info,
                           pre_nms=6000, post_nms=300, nms_thr=0.7,
                           min_size=16):
        B, _, H, W = cls_score.shape
        A = self.num_anchors
        # Tile anchors over the HxW grid.
        shifts_x = np.arange(W) * self.feat_stride
        shifts_y = np.arange(H) * self.feat_stride
        sx, sy = np.meshgrid(shifts_x, shifts_y)
        shifts = np.stack([sx.ravel(), sy.ravel(), sx.ravel(), sy.ravel()], axis=1)
        all_anchors = (self.anchors[None, :, :] +
                       shifts[:, None, :]).reshape(-1, 4)
        # Foreground probability from the two-class softmax.
        scores = F.softmax(cls_score.view(B, A, 2, H, W), dim=2)[:, :, 1, :, :]
        scores = scores.permute(0, 2, 3, 1).reshape(-1).detach().cpu().numpy()
        deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(-1, 4).detach().cpu().numpy()
        proposals = bbox_transform_inv(all_anchors, deltas)
        proposals = clip_boxes(proposals, im_info[:2])
        keep = ((proposals[:, 2] - proposals[:, 0] + 1 >= min_size * im_info[2]) &
                (proposals[:, 3] - proposals[:, 1] + 1 >= min_size * im_info[2]))
        proposals, scores = proposals[keep], scores[keep]
        order = scores.argsort()[::-1]
        if pre_nms > 0:
            order = order[:pre_nms]
        proposals, scores = proposals[order], scores[order]
        # Non-maximum suppression (assumes nms(dets, thresh) is available).
        keep = nms(np.hstack([proposals, scores[:, None]]), nms_thr)
        if post_nms > 0:
            keep = keep[:post_nms]
        return proposals[keep], scores[keep]
```
