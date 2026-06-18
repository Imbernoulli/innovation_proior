# SSD: Single Shot MultiBox Detector

## Method

Build a detector that predicts final class scores and box offsets directly from a fixed set of default boxes in one fully convolutional pass. There is no proposal stage, no RoI pooling, and no second classifier.

For each selected feature map of size `h x w`, attach two `3 x 3` convolutional predictors. If a cell has `k` default boxes and there are `c` classes including background, the localization predictor emits `4k` channels and the confidence predictor emits `ck` channels. Across all cells this gives `(c + 4)khw` outputs for that feature map.

Use several feature maps with decreasing resolution so different layers specialize to different object scales. For SSD300 the prediction resolutions are:

`38 x 38, 19 x 19, 10 x 10, 5 x 5, 3 x 3, 1 x 1`.

The paper names the VGG-based prediction sources as `conv4_3`, `conv7 (fc7)`, `conv8_2`, `conv9_2`, `conv10_2`, and `conv11_2`. The official Caffe SSD300 script names the post-`fc7` extra maps `conv6_2`, `conv7_2`, `conv8_2`, and `conv9_2`. The resolutions and computations are the same target architecture; the naming differs because of implementation convention.

## Default Boxes

The generic paper rule for `m` prediction maps uses scales

```text
s_k = s_min + (s_max - s_min) * (k - 1) / (m - 1),  k = 1,...,m
```

with `s_min = 0.2` and `s_max = 0.9`. For aspect ratio `a`, the default-box dimensions are

```text
w = s_k * sqrt(a),  h = s_k / sqrt(a)
```

and for `a = 1` an extra square box uses the geometric-mean scale

```text
s'_k = sqrt(s_k * s_{k+1}).
```

The actual VOC SSD300 implementation adds a smaller first scale on `conv4_3`: min size `30` and max size `60` for a `300 x 300` input, i.e. scales `0.1` and `0.2`. The full official SSD300 prior configuration is:

```text
feature maps:  [38, 19, 10, 5, 3, 1]
steps:         [8, 16, 32, 64, 100, 300]
min_sizes:     [30, 60, 111, 162, 213, 264]
max_sizes:     [60, 111, 162, 213, 264, 315]
aspect ratios: [[2], [2,3], [2,3], [2,3], [2], [2]]
flip ratios:   true
clip priors:   false
boxes/cell:    [4, 6, 6, 6, 4, 4]
total priors:  8732
```

`clip priors: false` matters: boundary-crossing default boxes are kept rather than clipped away.

## Matching And Loss

First force every ground-truth box to match its highest-Jaccard default box. Then match every remaining default box to its best ground truth if overlap is at least the threshold `0.5`. This gives multiple positives for a single object when several nearby defaults fit it well.

For a positive default box `d = (d_cx, d_cy, d_w, d_h)` and ground-truth box `g`, the paper targets are:

```text
g_hat_cx = (g_cx - d_cx) / d_w
g_hat_cy = (g_cy - d_cy) / d_h
g_hat_w  = log(g_w / d_w)
g_hat_h  = log(g_h / d_h)
```

The official Caffe code uses `CENTER_SIZE` encoding with variances `[0.1, 0.1, 0.2, 0.2]`, so the implemented training targets divide the two center residuals by `0.1` and the two log-size residuals by `0.2`. Decoding multiplies by the same variances.

The objective is

```text
L(x, c, l, g) = (L_conf(x, c) + alpha * L_loc(x, l, g)) / N
```

where `N` is the number of matched default boxes, `alpha = 1`, and the loss is zero when `N = 0`.

`L_loc` is a Smooth L1 sum over positive boxes only. `L_conf` is a softmax cross-entropy over positives and selected negatives, with background class `0`. Hard negative mining ranks negative boxes by confidence loss and keeps at most `3` negatives per positive.

## Inference

Decode predicted offsets back to boxes, drop class scores below `0.01`, run per-class NMS with Jaccard threshold `0.45` and `top_k = 400`, then keep the top `200` detections per image.

## Faithful PyTorch Skeleton

```python
from math import sqrt
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=20):
        super().__init__()
        self.weight = nn.Parameter(torch.full((n_channels,), float(scale)))
        self.eps = 1e-10

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(self.eps)
        return self.weight.view(1, -1, 1, 1) * x / norm


class SSDHead(nn.Module):
    def __init__(self, source_channels=(512, 1024, 512, 256, 256, 256),
                 boxes_per_location=(4, 6, 6, 6, 4, 4), num_classes=21):
        super().__init__()
        self.num_classes = num_classes
        self.loc = nn.ModuleList([
            nn.Conv2d(ch, k * 4, kernel_size=3, padding=1)
            for ch, k in zip(source_channels, boxes_per_location)
        ])
        self.conf = nn.ModuleList([
            nn.Conv2d(ch, k * num_classes, kernel_size=3, padding=1)
            for ch, k in zip(source_channels, boxes_per_location)
        ])

    def forward(self, sources):
        loc, conf = [], []
        for x, loc_conv, conf_conv in zip(sources, self.loc, self.conf):
            loc.append(loc_conv(x).permute(0, 2, 3, 1).contiguous())
            conf.append(conf_conv(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([x.view(x.size(0), -1) for x in loc], dim=1)
        conf = torch.cat([x.view(x.size(0), -1) for x in conf], dim=1)
        return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)


class SSD300DefaultBoxes:
    image_size = 300
    feature_maps = (38, 19, 10, 5, 3, 1)
    steps = (8, 16, 32, 64, 100, 300)
    min_sizes = (30, 60, 111, 162, 213, 264)
    max_sizes = (60, 111, 162, 213, 264, 315)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    variances = (0.1, 0.1, 0.2, 0.2)
    clip = False

    def __call__(self, device=None):
        boxes = []
        for k, f in enumerate(self.feature_maps):
            step = self.steps[k]
            sk = self.min_sizes[k] / self.image_size
            sk_next = self.max_sizes[k] / self.image_size
            for i, j in product(range(f), repeat=2):
                cx = (j + 0.5) * step / self.image_size
                cy = (i + 0.5) * step / self.image_size
                boxes.append((cx, cy, sk, sk))
                s_prime = sqrt(sk * sk_next)
                boxes.append((cx, cy, s_prime, s_prime))
                for ar in self.aspect_ratios[k]:
                    ar = float(ar)
                    boxes.append((cx, cy, sk * sqrt(ar), sk / sqrt(ar)))
                    boxes.append((cx, cy, sk / sqrt(ar), sk * sqrt(ar)))
        priors = torch.tensor(boxes, dtype=torch.float32, device=device)
        if self.clip:
            priors = center_size(point_form(priors).clamp_(0, 1))
        return priors


def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), dim=1)


def center_size(boxes):
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2,
                      boxes[:, 2:] - boxes[:, :2]), dim=1)


def jaccard(box_a, box_b):
    a, b = box_a.size(0), box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(a, b, 2),
                       box_b[:, 2:].unsqueeze(0).expand(a, b, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(a, b, 2),
                       box_b[:, :2].unsqueeze(0).expand(a, b, 2))
    inter_wh = (max_xy - min_xy).clamp_min(0)
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0)
    return inter / (area_a + area_b - inter).clamp_min(1e-12)


def encode(matched, priors, variances=(0.1, 0.1, 0.2, 0.2)):
    variances = priors.new_tensor(variances)
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy = g_cxcy / (variances[:2] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh.clamp_min(1e-12)) / variances[2:]
    return torch.cat([g_cxcy, g_wh], dim=1)


def decode(loc, priors, variances=(0.1, 0.1, 0.2, 0.2)):
    variances = priors.new_tensor(variances)
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[:2] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[2:])
    ), dim=1)
    return point_form(boxes)


def match(threshold, truths, priors, labels, loc_t, conf_t, idx,
          variances=(0.1, 0.1, 0.2, 0.2)):
    if truths.numel() == 0:
        loc_t[idx].zero_()
        conf_t[idx].zero_()
        return
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(dim=1)
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx].long() + 1
    conf[best_truth_overlap < threshold] = 0
    loc_t[idx] = encode(matches, priors, variances)
    conf_t[idx] = conf


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes=21, threshold=0.5, neg_pos_ratio=3,
                 variances=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.variances = variances

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        num, num_priors = loc_data.size(0), priors.size(0)
        loc_t = loc_data.new_zeros(num, num_priors, 4)
        conf_t = torch.zeros(num, num_priors, dtype=torch.long, device=loc_data.device)

        for idx in range(num):
            target = targets[idx].to(loc_data.device)
            truths = target[:, :4]
            labels = target[:, 4]
            match(self.threshold, truths, priors, labels, loc_t, conf_t, idx,
                  self.variances)

        pos = conf_t > 0
        num_pos = pos.long().sum(dim=1, keepdim=True)
        total_pos = num_pos.sum()
        if total_pos.item() == 0:
            zero = loc_data.sum() * 0
            return zero, zero

        pos_idx = pos.unsqueeze(2).expand_as(loc_data)
        loss_l = F.smooth_l1_loss(loc_data[pos_idx].view(-1, 4),
                                  loc_t[pos_idx].view(-1, 4),
                                  reduction="sum")

        loss_c = F.cross_entropy(conf_data.view(-1, self.num_classes),
                                 conf_t.view(-1), reduction="none")
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0

        _, loss_idx = loss_c.sort(dim=1, descending=True)
        _, idx_rank = loss_idx.sort(dim=1)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        keep = pos | neg
        loss_c = F.cross_entropy(conf_data[keep], conf_t[keep], reduction="sum")
        normalizer = total_pos.float()
        return loss_l / normalizer, loss_c / normalizer
```
