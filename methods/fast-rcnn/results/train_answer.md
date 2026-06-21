Region-based convolutional detectors had just made object detection far more accurate on PASCAL VOC, but the price was a slow, multi-stage pipeline. R-CNN and its immediate successors fine-tuned a ConvNet on warped proposals, wrote every proposal's features to disk, trained per-class linear SVMs on those cached features, and then trained bounding-box regressors as yet another stage. At test time they ran a full ConvNet forward pass for each of roughly two thousand proposals per image, so a deep network could spend the better part of a minute on each image. The question is whether the same accuracy can be obtained from a single network that processes an image once, shares all convolutional computation across proposals, jointly learns classification and box refinement, fine-tunes every layer end to end, and never writes features to disk.

The existing approaches fall short for two related reasons. First, they treat each proposal as an independent image, recomputing the entire convolutional feature map thousands of times even though proposals in the same image overlap heavily and should share computation. Second, the training pipeline is split into separate stages with separate objectives, which forces the expensive feature-caching step and prevents the convolutional layers from being jointly adapted to detection. SPPnet already showed that sharing convolutional work through whole-image feature maps and spatial pyramid pooling gives a large speedup, but it still freezes the convolutional layers during fine-tuning and keeps the multi-stage SVM-plus-regressor pipeline. What is missing is a way to train the whole architecture, including the conv layers, in one stage.

The method is Fast R-CNN. Its first ingredient is a region-of-interest pooling layer that replaces the backbone's final max-pool. The backbone runs once over the whole image to produce a single convolutional feature map. For each proposal, the RoI pooling layer extracts the corresponding sub-window of that shared map and max-pools it into a fixed-size feature of shape H by W by C, regardless of the proposal's original size. In practice H and W are set to 7 so the output feeds directly into the first fully connected layer of a VGG-style network. Because the layer is just max-pooling with per-cell argmax switches, gradients route back to the input feature map in the usual way, summed over all proposals that selected a given activation. That is what lets the loss update the convolutional trunk.

The second ingredient is a jointly trained two-headed detector. After the RoI-pooled features pass through fully connected layers, the network splits into two sibling outputs. One head outputs a softmax distribution over K object classes plus a background class. The other head outputs four bounding-box regression offsets for each of the K classes, using the standard scale-invariant translation and log-space size parameterization relative to the proposal box. Both heads are trained together with a single multi-task loss. For a training proposal with true class u and, if it is foreground, a ground-truth box target v, the loss is the classification log loss plus a localization term on the offsets for the true class only. The localization term is skipped for background regions, since there is no box to refine.

For the localization loss I use smooth L1 rather than plain L2. Smooth L1 behaves quadratically for small residuals, so it is well-behaved near the optimum, but only linearly for large residuals, which keeps gradients bounded and avoids the exploding-gradient problems that unbounded L2 targets can create. The ground-truth box targets are normalized to zero mean and unit variance, and the two loss terms are balanced with a single scalar lambda set to one. This single loss removes any need for post-hoc SVMs or cached features; the softmax classifier is trained end to end along with the box regressor, and the inter-class competition in softmax actually helps slightly.

Training the conv layers efficiently requires one more design choice. If each mini-batch sample came from a different image, every proposal's receptive field would cover nearly the whole image, forcing the forward and backward pass to process essentially one full image per sample. That is why earlier work froze the conv layers. Fast R-CNN instead samples hierarchically: pick a small number of images, say N equals 2, and draw many proposals from each, say 64 per image for a total of 128. Proposals from the same image share the whole-image conv pass, so the conv computation drops by roughly a factor of 64 compared with one-proposal-per-image sampling, while gradients still flow all the way back through the RoI pooling layer. About 25 percent of the sampled proposals are foreground with IoU at least 0.5 to a ground-truth box, and the remaining 75 percent are background with maximum IoU in the range 0.1 to 0.5. The lower bound of 0.1 acts as implicit hard-negative mining by keeping background windows that graze objects rather than trivial empty regions. Horizontal flips provide the only data augmentation.

Initialization and optimization follow the pretrain-and-finetune recipe. The backbone is initialized from an ImageNet classification network with its last max-pool removed. New fully connected sibling layers are initialized from zero-mean Gaussians, with standard deviation 0.01 for classification and 0.001 for box regression so offsets start near zero. Training uses SGD with momentum 0.9 and weight decay 0.0005, a global learning rate of 0.001 with biases doubled, 30 thousand iterations, then a lower rate of 0.0001 for 10 thousand more. The bottom convolutional layers are frozen because their generic edge and color filters do not need to change, while the upper conv layers are fine-tuned through the RoI pooling gradient. At test time the shortest image side is scaled to 600 pixels with the longest side capped at 1000; deep networks learn enough scale invariance directly that a full image pyramid is not worth its cost. Because the fully connected layers are now run once per proposal, they dominate forward time, so they can be compressed at inference with truncated SVD at a small accuracy cost.

```python
import torch, random
from torch import nn
import torch.nn.functional as F


class RegionPooling(nn.Module):
    """Single-level RoI max-pool: any region -> fixed HxW x C; grad flows into conv trunk."""
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
