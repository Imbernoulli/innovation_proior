# Fast R-CNN

## Problem
Region-based ConvNet detectors were accurate but trained in a slow, multi-stage pipeline
(fine-tune ConvNet → cache features to disk → train per-class SVMs → train box regressors) and ran a
full network forward pass per proposal at test time (tens of seconds per image). Collapse this into a
single network that shares convolutional computation, trains end-to-end in one stage, updates all
layers, and needs no on-disk feature cache — without losing accuracy.

## Key idea
Compute the convolutional feature map of the whole image *once*; for each object proposal, pool the
corresponding region of that shared map into a fixed-size feature via a **RoI pooling layer**; run
fully connected layers and branch into two sibling heads — a (K+1)-way softmax classifier and
per-class bounding-box regressors — trained jointly with a single **multi-task loss**.
**Hierarchical mini-batch sampling** (few images, many RoIs each) makes back-propagation through RoI
pooling into the conv layers efficient, so the entire network can be fine-tuned.

## RoI pooling
A RoI is a rectangle (r, c, h, w) in the conv feature map. RoI max pooling divides the h×w window
into an H×W grid of sub-windows of size ≈ (h/H)×(w/W) and max-pools each into the corresponding
output cell, per channel — a fixed H×W×C feature for any region (H = W = 7 for VGG16). It is a
single-level spatial pyramid pooling. Backward routes gradients through the argmax switches:

  ∂L/∂x_i = Σ_r Σ_j [i = i*(r,j)] · ∂L/∂y_rj,   y_rj = x_{i*(r,j)},

summed over all RoIs r and pooled cells j that selected input x_i.

## Multi-task loss
Per RoI: softmax distribution p = (p_0, …, p_K) over K classes + background, and per-class box
offsets t^k. Label each RoI with a ground-truth class u and (if foreground) box target v:

  L(p, u, t^u, v) = L_cls(p, u) + λ · [u ≥ 1] · L_loc(t^u, v),
  L_cls(p, u) = −log p_u,
  L_loc(t^u, v) = Σ_{i∈{x,y,w,h}} smooth_L1(t^u_i − v_i),
  smooth_L1(x) = 0.5 x²  if |x| < 1, else |x| − 0.5.

The indicator [u ≥ 1] drops localization for background (u = 0, no box); only the *true* class's box
offsets are penalized. smooth_L1 is robust to outliers and avoids the exploding gradients L2 can
produce on unbounded targets. Ground-truth targets are normalized to zero mean / unit variance;
λ = 1.

## Training
- **Initialize** from a pre-trained ImageNet net: replace the last max-pool with RoI pooling
  (H = W = 7); replace the 1000-way head with the two sibling layers; take (image, RoIs) as input.
  New fc layers: classification N(0, 0.01²), box regression N(0, 0.001²), biases 0.
- **Hierarchical sampling:** N = 2 images per mini-batch, R = 128 RoIs (64 per image) — RoIs from the
  same image share the conv forward/backward pass (~64× cheaper than one-RoI-per-image), which lets
  gradients reach the conv layers. 25% foreground (IoU ≥ 0.5 with a GT box); 75% background (max IoU
  in [0.1, 0.5) — the 0.1 floor acts as implicit hard-negative mining). Horizontal flip p = 0.5.
- **SGD:** global lr 0.001 (per-layer 1× weights, 2× biases), 30k iters then 0.0001 for 10k; momentum
  0.9, weight decay 0.0005. Fine-tune conv layers from a middle block upward (conv1 is generic and
  frozen; for VGG16, conv3_1 and up).
- Softmax replaces post-hoc SVMs (one-stage fine-tuning suffices; softmax adds inter-class
  competition), removing a training stage and the disk cache.

## Scale
Single-scale "brute force": shortest side 600 px, longest capped at 1000. Deep nets learn scale
invariance directly, so an image pyramid adds little mAP at large cost.

## Truncated SVD (faster detection)
With thousands of RoIs, fc layers dominate the forward pass. Factor a fc weight W (u×v) as
W ≈ U Σ_t V^T and replace it by two layers (no nonlinearity between): Σ_t V^T (no bias) then U (with
W's bias). Parameters drop u·v → t·(u+v). For VGG16, keep top ~1024 of fc6 and ~256 of fc7.

## Detection
One forward pass: image + ~2000 proposals → per-RoI class posteriors p_k and per-class box offsets.
Confidence for class k is p_k; apply per-class non-maximum suppression.

## Code
```python
import torch, random
from torch import nn
import torch.nn.functional as F


class RegionPooling(nn.Module):
    """Single-level RoI max-pool: any region -> fixed HxW x C; grad flows into the conv trunk."""
    def __init__(self, output_h=7, output_w=7, spatial_scale=1/16.):
        super().__init__()
        self.oh, self.ow, self.scale = output_h, output_w, spatial_scale

    def forward(self, feat, rois):                    # rois: (R,5) = (img_idx, x1,y1,x2,y2) px
        out = feat.new_zeros(rois.size(0), feat.size(1), self.oh, self.ow)
        for r, (n, x1, y1, x2, y2) in enumerate(rois):
            x1, y1, x2, y2 = [int(round(c.item() * self.scale)) for c in (x1, y1, x2, y2)]
            h, w = max(y2 - y1, 1), max(x2 - x1, 1)
            for i in range(self.oh):
                for j in range(self.ow):
                    sy1, sy2 = y1 + (i*h)//self.oh, y1 + ((i+1)*h)//self.oh
                    sx1, sx2 = x1 + (j*w)//self.ow, x1 + ((j+1)*w)//self.ow
                    reg = feat[int(n), :, sy1:max(sy2, sy1+1), sx1:max(sx2, sx1+1)]
                    out[r, :, i, j] = reg.amax(dim=(1, 2))
        return out


class FastDetector(nn.Module):
    def __init__(self, backbone, head, num_classes, feat_dim=4096):
        super().__init__()
        self.backbone = backbone                      # conv trunk, last max-pool removed
        self.region_pool = RegionPooling(7, 7, 1/16.)
        self.head = head                              # fc6, fc7
        self.cls = nn.Linear(feat_dim, num_classes + 1)
        self.box = nn.Linear(feat_dim, 4 * num_classes)
        nn.init.normal_(self.cls.weight, std=0.01);  nn.init.zeros_(self.cls.bias)
        nn.init.normal_(self.box.weight, std=0.001); nn.init.zeros_(self.box.bias)

    def forward(self, images, rois):
        feat = self.backbone(images)                  # whole-image conv map, computed ONCE
        x = self.head(self.region_pool(feat, rois).flatten(1))
        return self.cls(x), self.box(x)


def smooth_l1(x):
    ax = x.abs()
    return torch.where(ax < 1.0, 0.5 * x * x, ax - 0.5)


def multitask_loss(cls_scores, box_offsets, labels, box_targets, lam=1.0):
    L_cls = F.cross_entropy(cls_scores, labels)
    fg = labels >= 1
    R = box_offsets.size(0)
    pred_u = box_offsets.view(R, -1, 4)[torch.arange(R)[fg], labels[fg]]   # true-class offsets
    L_loc = smooth_l1(pred_u - box_targets[fg]).sum() / max(R, 1)
    return L_cls + lam * L_loc


def sample_minibatch(db, num_images=2, rois_per_image=64, fg_frac=0.25,
                     fg_iou=0.5, bg_iou=(0.1, 0.5)):
    rois, labels, targets, imgs = [], [], [], []
    for n in random.sample(range(len(db)), num_images):
        rec = db[n]
        max_iou, gt_idx = roi_gt_overlaps(rec.proposals, rec.gt_boxes)
        n_fg = int(round(fg_frac * rois_per_image))
        fg = where(max_iou >= fg_iou)
        bg = where((max_iou >= bg_iou[0]) & (max_iou < bg_iou[1]))         # hard-ish background
        for k in cat(sample(fg, n_fg), sample(bg, rois_per_image - n_fg)):
            lab = rec.gt_classes[gt_idx[k]] + 1 if max_iou[k] >= fg_iou else 0
            tgt = encode_boxes(rec.proposals[k], rec.gt_boxes[gt_idx[k]]) if lab else 0
            rois.append((len(imgs), *rec.proposals[k])); labels.append(lab); targets.append(tgt)
        imgs.append(maybe_hflip(rec.image, p=0.5))
    return stack(imgs), tensor(rois), tensor(labels), stack(targets)


def truncated_svd_fc(fc, t):                          # detection-time fc compression
    W = fc.weight.data
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    first = nn.Linear(W.size(1), t, bias=False);  first.weight.data = torch.diag(S[:t]) @ Vt[:t]
    second = nn.Linear(t, W.size(0), bias=True);   second.weight.data = U[:, :t]
    second.bias.data = fc.bias.data
    return nn.Sequential(first, second)
```
