I frame object detection as a single dense feed-forward regression over a fixed set of reference boxes. The method I am describing is the Single Shot MultiBox Detector, or SSD. Its central claim is that a convolutional network can emit final class scores and final bounding-box offsets in one pass, without an external proposal generator, without per-region feature resampling, and without a second-stage classifier. The speed gains come from eliminating the proposal loop; the accuracy comes from spreading the reference boxes across many feature-map resolutions and many aspect ratios, so that small objects, large objects, and oddly shaped objects are all covered by predictors with appropriate receptive fields.

The starting observation is that two-stage detectors such as Faster R-CNN are accurate because they use anchor boxes and local convolutional predictors, but they are slow because every anchor that survives as a proposal is fed into a second RoI-pooled head. YOLO and OverFeat are fast because they predict from one forward pass, but they lose accuracy because they rely on a single coarse grid and a small number of boxes per cell. SSD keeps the good parts of both sides: it uses the anchor-style local offset regression from proposal networks, but it applies that regression directly to final classes and boxes, and it repeats the predictor at many scales inside the network.

The backbone is a standard ImageNet-pretrained VGG-16 stack with its fully connected layers converted to convolutions. After the truncated VGG layers, extra convolutional layers shrink the feature map down to a 1 by 1 grid. The prediction sources for SSD300 are six feature maps whose spatial sizes are 38 by 38, 19 by 19, 10 by 10, 5 by 5, 3 by 3, and 1 by 1. Earlier maps have fine spatial sampling and are suitable for small objects; later maps have large receptive fields and are suitable for large objects. The first high-resolution source is L2-normalized per spatial location and scaled by a learned per-channel factor, because its feature magnitudes differ from the deeper sources.

For each prediction cell I define a small set of default boxes, also called priors or anchor boxes. Each default box is tied to a cell center and has a fixed size and aspect ratio. The centers are placed at ((j + 0.5) * step / 300, (i + 0.5) * step / 300) for a 300-pixel input, where step is the subsampling factor of the corresponding feature map. The sizes follow a geometric progression. On SSD300 the minimum sizes in pixels are 30, 60, 111, 162, 213, and 264, and the maximum sizes are 60, 111, 162, 213, 264, and 315. For aspect ratio a, the width and height of a default box are sk * sqrt(a) and sk / sqrt(a), so the area stays constant. I use aspect ratios 2 and 3 with their reciprocals, and I add an extra square default box at the geometric mean between consecutive scales. The coarsest maps drop the ratio-3 boxes. The result is 8732 default boxes in total, far denser than a 7 by 7 grid but still cheap because every output is produced by a small convolution.

The head consists of two 3 by 3 convolutional layers at each prediction source. If a source has k default boxes per cell and the detection task has c classes including background, the localization branch emits 4k channels and the confidence branch emits ck channels. After permutation and flattening, every default box has a 4-dimensional localization vector and a c-dimensional class-score vector.

Training requires matching the fixed output set to the variable ground-truth set. I use a two-stage match. First, every ground-truth box is forced to match its highest-overlap default box, so no object is orphaned. Second, any remaining default box whose best overlap with any ground truth is at least 0.5 is also marked positive for that ground truth. This means one object can supervise several nearby defaults, which gives the network useful gradient and lets non-maximum suppression resolve duplicates later. Default boxes that cross the image boundary are kept during training rather than clipped, because clipping would disturb the intended tiling.

For a positive default box d with center, width, and height (d_cx, d_cy, d_w, d_h) and a matched ground-truth box g, the localization target is (g_cx - d_cx) / d_w for the center x coordinate, (g_cy - d_cy) / d_h for the center y coordinate, log(g_w / d_w) for width, and log(g_h / d_h) for height. In the public implementation these four targets are divided by variances 0.1, 0.1, 0.2, and 0.2 to balance their magnitudes, and decoding reverses the same scaling. The signs are meaningful: a positive center target means the object center is to the right of or below the default center, and a positive size target means the object is larger than the default.

The loss is the sum of a localization term and a confidence term, normalized by the number N of matched positive default boxes. The localization term is Smooth L1 over positives only, because negatives have no ground-truth box to regress. The confidence term is softmax cross-entropy over all classes including background, where background is class 0. Because almost all default boxes are background, I rank negatives by their current confidence loss and keep at most three negatives per positive. This hard negative mining focuses the classifier on the most confusing background regions and prevents the large background majority from drowning the positives.

Data augmentation is important because there is no region crop at inference time. I use random crops with minimum Jaccard overlaps of 0.1, 0.3, 0.5, 0.7, or 0.9, plus random sampling, then keep ground-truth boxes whose centers still lie inside the crop, resize to 300 by 300, randomly flip horizontally, and apply photometric distortions. I also place the image on a larger mean-filled canvas before cropping, which creates smaller objects in the training patch and helps the high-resolution source learn small-object detection.

At inference I decode the predicted offsets back to absolute box coordinates, discard class scores below 0.01, run greedy per-class non-maximum suppression with an overlap threshold of 0.45, and keep the top 200 detections per image. The entire pipeline is therefore one forward pass followed by a light post-processing step, with no region proposals and no per-region feature extraction.

The canonical name of the method is Single Shot MultiBox Detector, abbreviated SSD. The following implementation gives the L2-normalized source and the multi-scale detection head, the default-box tiling, the best-plus-threshold matching rule, the center-log-size encoding and decoding, and the confidence-aware hard-negative mining inside the multibox loss that make the method work.

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
