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

The canonical name of the method is Single Shot MultiBox Detector, abbreviated SSD. The following script illustrates the default-box tiling, the best-plus-threshold matching rule, the center-log-size encoding, and the confidence-aware hard-negative mining that make the method work.

```python
import torch
import torch.nn.functional as F
from itertools import product
from math import sqrt


def make_ssd300_default_boxes():
    image_size = 300
    feature_maps = (38, 19, 10, 5, 3, 1)
    steps = (8, 16, 32, 64, 100, 300)
    min_sizes = (30, 60, 111, 162, 213, 264)
    max_sizes = (60, 111, 162, 213, 264, 315)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    boxes = []
    for k, f in enumerate(feature_maps):
        step = steps[k]
        sk = min_sizes[k] / image_size
        sk_next = max_sizes[k] / image_size
        for i, j in product(range(f), repeat=2):
            cx = (j + 0.5) * step / image_size
            cy = (i + 0.5) * step / image_size
            boxes.append((cx, cy, sk, sk))
            s_prime = sqrt(sk * sk_next)
            boxes.append((cx, cy, s_prime, s_prime))
            for ar in aspect_ratios[k]:
                ar = float(ar)
                boxes.append((cx, cy, sk * sqrt(ar), sk / sqrt(ar)))
                boxes.append((cx, cy, sk / sqrt(ar), sk * sqrt(ar)))
    return torch.tensor(boxes, dtype=torch.float32)


def point_form(b):
    return torch.cat((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), dim=1)


def jaccard(a, b):
    a, b = point_form(a), point_form(b)
    inter_min = torch.max(a[:, None, :2], b[None, :, :2])
    inter_max = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter_wh = (inter_max - inter_min).clamp_min(0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp_min(1e-12)


def encode(matched, priors, variances=(0.1, 0.1, 0.2, 0.2)):
    v = priors.new_tensor(variances)
    centers = ((matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]) / (v[:2] * priors[:, 2:])
    sizes = torch.log(((matched[:, 2:] - matched[:, :2]) / priors[:, 2:]).clamp_min(1e-12)) / v[2:]
    return torch.cat((centers, sizes), dim=1)


def match(threshold, truths, labels, priors):
    if truths.numel() == 0:
        return torch.zeros(len(priors), 4), torch.zeros(len(priors), dtype=torch.long)
    overlaps = jaccard(truths, priors)
    best_prior_overlap, best_prior_idx = overlaps.max(dim=1)
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2.0)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx].long() + 1
    conf[best_truth_overlap < threshold] = 0
    return encode(matches, priors), conf


def hard_negative_mining(conf_data, conf_targets, neg_pos_ratio=3):
    loss = F.cross_entropy(conf_data.view(-1, conf_data.size(-1)), conf_targets.view(-1), reduction='none')
    loss = loss.view(conf_data.size(0), -1)
    pos_mask = conf_targets > 0
    loss[pos_mask] = 0
    _, loss_idx = loss.sort(dim=1, descending=True)
    _, idx_rank = loss_idx.sort(dim=1)
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = torch.clamp(neg_pos_ratio * num_pos, max=pos_mask.size(1) - 1)
    return idx_rank < num_neg


if __name__ == "__main__":
    priors = make_ssd300_default_boxes()
    print("Number of default boxes:", len(priors))

    truths = torch.tensor([[0.25, 0.25, 0.55, 0.55], [0.70, 0.70, 0.95, 0.95]])
    labels = torch.tensor([14, 7])
    loc_t, conf_t = match(0.5, truths, labels, priors)
    positives = (conf_t > 0).sum().item()
    print("Positive matches:", positives)

    conf_data = torch.randn(1, len(priors), 21)
    neg_mask = hard_negative_mining(conf_data, conf_t.unsqueeze(0))
    print("Hard negatives kept:", neg_mask.sum().item())

    selected = (conf_t.unsqueeze(0) > 0) | neg_mask
    print("Total training samples:", selected.sum().item())
```
