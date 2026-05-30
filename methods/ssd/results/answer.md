# SSD: Single Shot MultiBox Detector

## Problem

Detect every object in an image (box + class label) in a single forward pass, fast enough for real time, while matching the accuracy of detectors that propose regions and then resample/reclassify features per region. The expensive, accuracy-limiting bottleneck of the propose-then-classify pipeline is the per-proposal feature resampling and second-stage classification; the single-pass alternatives of the time (OverFeat, YOLO) removed it but predicted from a single low-resolution feature map and lost accuracy, especially on small objects.

## Key idea

Predict detections directly from a fixed set of **default (anchor) boxes** tiled over **several feature maps of decreasing resolution**, in one fully-convolutional pass — no proposals, no resampling.

- **Multi-scale feature maps.** Take a VGG-16 backbone truncated before its classifier, convert fc6/fc7 to convolutions with the à trous (dilated) trick to keep resolution cheaply, then append extra conv layers that progressively shrink the spatial size. For a 300×300 input the six prediction sources are 38×38, 19×19, 10×10, 5×5, 3×3, 1×1. High-resolution maps detect small objects; low-resolution maps detect large ones.
- **Default boxes.** At each cell of each map, tile *k* default boxes of several aspect ratios. Map *k* (of *m*) gets scale
  s_k = s_min + (s_max − s_min)·(k − 1)/(m − 1), with s_min = 0.2, s_max = 0.9.
  For aspect ratio a_r ∈ {1, 2, 3, 1/2, 1/3}: width w = s_k·√a_r, height h = s_k/√a_r (constant area s_k², shape a_r). For a_r = 1 add one extra box at s'_k = √(s_k·s_{k+1}) to bridge adjacent maps, giving 6 boxes/location (4 when dropping {3, 1/3}). Center at ((j+0.5)/|f_k|, (i+0.5)/|f_k|). Total for SSD300: 8732 boxes.
- **Convolutional predictors.** Each source map (m×n×p) gets a 3×3×p conv producing, per location, *k*·(c + 4) outputs: *c* class scores and 4 box offsets per default box.

## Training

**Matching.** Match each ground-truth box to its highest-overlap default box (so no object is unmatched), then additionally match any default box with Jaccard overlap > 0.5 to a ground-truth box. A default box can be positive for an object even if it isn't that object's single best — multiple overlapping boxes may fire.

**Localization target (offsets relative to the default box d).**
ĝ_cx = (g_cx − d_cx)/d_w, ĝ_cy = (g_cy − d_cy)/d_h, ĝ_w = log(g_w/d_w), ĝ_h = log(g_h/d_h).

**Objective.**
L(x, c, l, g) = (1/N)·(L_conf(x, c) + α·L_loc(x, l, g)), N = #matched boxes (0 ⇒ loss 0), α = 1.
- L_loc = Σ_{i∈Pos} Σ_{m∈{cx,cy,w,h}} x_ij^k · smooth_L1(l_i^m − ĝ_j^m), positives only.
- L_conf = −Σ_{i∈Pos} x_ij^p log(ĉ_i^p) − Σ_{i∈Neg} log(ĉ_i^0), softmax over c classes (0 = background).

**Hard negative mining.** Most default boxes are background. Sort negatives by their confidence loss (hardest first) and keep the top so that neg:pos ≤ 3:1. Faster, more stable training.

**Other details.** L2-normalize conv4_3 features and learn a per-channel scale (init 20), since this early layer has larger activation magnitudes. Heavy data augmentation (constrained random crops with min-IoU sampling = "zoom in"; placing the image on a 16× mean-filled canvas = "zoom out" for small objects; flips; photometric distortion). SGD, lr 1e-3, momentum 0.9, weight decay 5e-4.

**Inference.** Threshold confidences at 0.01, per-class NMS at Jaccard 0.45, keep top 200 detections.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from itertools import product


# ---------- Backbone: VGG-16 truncated + atrous conv6/conv7 ----------
def vgg(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2, 2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(2, 2, ceil_mode=True)]
        else:
            layers += [nn.Conv2d(in_channels, v, 3, padding=1),
                       nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.MaxPool2d(3, 1, padding=1),
               nn.Conv2d(512, 1024, 3, padding=6, dilation=6),  # fc6 -> atrous
               nn.ReLU(inplace=True),
               nn.Conv2d(1024, 1024, 1),                        # fc7
               nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, in_channels):
    layers, flag = [], False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     (1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, (1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant_(self.weight, scale)
        self.eps = 1e-10

    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return self.weight.view(1, -1, 1, 1) * (x / norm)


# ---------- Default boxes ----------
class PriorBox:
    def __init__(self, image_size, feature_maps, steps,
                 min_sizes, max_sizes, aspect_ratios):
        self.image_size = image_size
        self.feature_maps = feature_maps
        self.steps = steps
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios

    def __call__(self):
        boxes = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx, cy = (j + 0.5) / f_k, (i + 0.5) / f_k
                s_k = self.min_sizes[k] / self.image_size
                boxes += [cx, cy, s_k, s_k]
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                boxes += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    boxes += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    boxes += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        return torch.Tensor(boxes).view(-1, 4).clamp_(max=1, min=0)


# ---------- Detector ----------
class SSD(nn.Module):
    def __init__(self, base, extras, head, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, x):
        sources, loc, conf = [], [], []
        for k in range(23):                 # through conv4_3 relu
            x = self.vgg[k](x)
        sources.append(self.L2Norm(x))
        for k in range(23, len(self.vgg)):  # through conv7
            x = self.vgg[k](x)
        sources.append(x)
        for k, v in enumerate(self.extras): # conv8_2..conv11_2
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        for (s, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(s).permute(0, 2, 3, 1).contiguous())
            conf.append(c(s).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))


def multibox(vgg_layers, extra_layers, num_boxes, num_classes):
    loc, conf = [], []
    for v, nb in zip([vgg_layers[21], vgg_layers[-2]], num_boxes[:2]):
        loc += [nn.Conv2d(v.out_channels, nb * 4, 3, padding=1)]
        conf += [nn.Conv2d(v.out_channels, nb * num_classes, 3, padding=1)]
    for v, nb in zip(extra_layers[1::2], num_boxes[2:]):
        loc += [nn.Conv2d(v.out_channels, nb * 4, 3, padding=1)]
        conf += [nn.Conv2d(v.out_channels, nb * num_classes, 3, padding=1)]
    return vgg_layers, extra_layers, (loc, conf)


# SSD300 configuration
base_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C',
            512, 512, 512, 'M', 512, 512, 512]
extras_cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
mbox = [4, 6, 6, 6, 4, 4]


def build_ssd(num_classes=21):
    base, extras, head = multibox(vgg(base_cfg, 3),
                                  add_extras(extras_cfg, 1024),
                                  mbox, num_classes)
    return SSD(base, extras, head, num_classes)


# ---------- Box utilities ----------
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


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


def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)   # force best box positive
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1                      # 0 reserved for background
    conf[best_truth_overlap < threshold] = 0
    loc_t[idx] = encode(matches, priors, variances)
    conf_t[idx] = conf


def log_sum_exp(x):
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# ---------- MultiBox loss ----------
class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, priors, overlap_thresh=0.5,
                 neg_pos_ratio=3, variances=(0.1, 0.2)):
        super().__init__()
        self.num_classes = num_classes
        self.priors = priors
        self.threshold = overlap_thresh
        self.negpos = neg_pos_ratio
        self.variances = variances

    def forward(self, predictions, targets):
        loc_data, conf_data = predictions
        num, num_priors = loc_data.size(0), self.priors.size(0)

        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            match(self.threshold, targets[idx][:, :-1], self.priors,
                  self.variances, targets[idx][:, -1], loc_t, conf_t, idx)

        pos = conf_t > 0

        # localization loss (positives only)
        pos_idx = pos.unsqueeze(2).expand_as(loc_data)
        loss_l = F.smooth_l1_loss(loc_data[pos_idx].view(-1, 4),
                                  loc_t[pos_idx].view(-1, 4), reduction='sum')

        # rank negatives by confidence loss for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0
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

        N = num_pos.sum().clamp(min=1)                     # alpha = 1
        return loss_l / N, loss_c / N
```

The variances (0.1, 0.2) divide the encoded center and size targets respectively — a standard normalization of the regression targets inherited from the anchor-offset literature; at inference the decoder multiplies them back when converting predicted offsets to boxes.
