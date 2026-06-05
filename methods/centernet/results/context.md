# Context

## Research question

Object detection represents each object as an axis-aligned bounding box. By 2019 the dominant way to produce those boxes was to enumerate a near-exhaustive list of candidate boxes and classify each. One-stage detectors slide a dense arrangement of pre-defined boxes (anchors) over the image and classify them directly; two-stage detectors first propose regions, then recompute features per region and classify. Both then run non-maximum suppression (NMS) to delete duplicate boxes for the same instance by comparing box IoU. This works empirically, but it is wasteful — most enumerated boxes are background — and it is awkward to train end to end, because the NMS step that resolves duplicates is hard to differentiate, so the duplicate-removal logic lives outside the learned model.

The precise question: **can detection be done without enumerating boxes and without IoU-based NMS at all?** If an object could be represented by a single point — its center — then finding objects becomes finding peaks in a heatmap, a problem the keypoint-estimation literature already solves with a plain fully-convolutional network, and box size becomes just another quantity regressed at that point. A solution would have to (i) localize centers reliably despite the output stride's coarseness, (ii) recover the box from the center, (iii) avoid duplicate detections without a separate NMS stage, and (iv) ideally generalize to other per-object properties (3D box, pose) by simply adding regression outputs at the center.

## Background

The field rests on the sliding-window/anchor detection line and the keypoint-heatmap line, which this question proposes to merge.

**Anchor / region-classification detectors.** R-CNN crops region candidates and classifies each; Fast R-CNN crops features instead. Faster R-CNN generates proposals inside the network by classifying fixed-shape anchors tiled over a low-resolution grid: an anchor is foreground if its IoU with any ground truth exceeds 0.7, background below 0.3, ignored otherwise. One-stage detectors (SSD, YOLOv2/v3, RetinaNet) skip the proposal step and classify the anchors directly, adding shape priors, multiple feature resolutions, and loss re-weighting (focal loss). All of them: enumerate many boxes, classify, then NMS. They use a coarse output stride (typically 16) and rely on multiple anchors per location to cover scales/aspect ratios.

**Keypoint estimation with heatmaps.** Human-pose and landmark estimators predict, per keypoint type, a 2D heatmap whose peaks are the keypoint locations. The ground-truth is built by "splatting" a 2D Gaussian centered at each keypoint, and the network is a fully-convolutional encoder–decoder. Standard backbones for this: the stacked Hourglass network (symmetric down/up sampling with skip connections), up-convolutional ResNets (Xiao et al. 2018, output stride 4), and Deep Layer Aggregation (DLA, Yu et al. 2018, iterative skip aggregation). These networks already produce high-resolution outputs (stride 4) — far finer than detection's stride-16 grids — and find local maxima cleanly.

**CornerNet (Law & Deng 2018)** is the load-bearing ancestor: it casts detection as keypoint estimation, detecting the top-left and bottom-right **corners** of each box as heatmap peaks. Crucially it introduces the training machinery this question will reuse: splatting Gaussians at the target keypoints and a **penalty-reduced pixel-wise focal loss** for the heatmap, so that predictions near a true keypoint are penalized less than far ones. Its limitation: a box is two corners, so after detecting corners the method must *group* corner pairs that belong to the same object, using a learned associative embedding — a combinatorial post-processing step that is intricate and slow. ExtremeNet (Zhou et al. 2019) detects four extreme points plus a center and also needs a grouping/enumeration step.

**A diagnostic fact about anchors and assignment.** An anchor is assigned by box *overlap* with manual IoU thresholds, and to cover an object well, many anchors are needed; insufficient anchor placement is a real source of missed objects (e.g. Faster R-CNN with 15 anchors at IoU 0.5 fails to cover a substantial fraction of objects). A point-based assignment that uses only *location* would drop the thresholds and the multiple-anchors-per-object requirement.

## Baselines

**Anchor-based one-stage detectors (RetinaNet, YOLOv3, SSD).** Dense anchors over a stride-16 grid, per-anchor classification (focal loss for RetinaNet), per-anchor box-offset regression, then NMS. Gap: enumerate and classify many background boxes; multiple anchors per object; require NMS, which blocks end-to-end training and adds latency.

**Two-stage detectors (Faster R-CNN, Mask R-CNN).** Proposal network + per-region feature recomputation + classification + NMS. Gap: accurate but slow; the most complex pipeline; still NMS-bound.

**CornerNet.** Detect box corners as heatmap keypoints (Gaussian targets, penalty-reduced focal loss), then group corner pairs with associative embeddings. Gap: corner grouping is a combinatorial, learned post-processing stage that is slow and requires special design; corners lie on the object boundary, which is empirically harder to localize than an interior point.

## Evaluation settings

- **MS-COCO** detection: 118K train (train2017), 5K val (val2017), 20K test-dev. Metrics: AP averaged over IoU 0.5:0.95, AP50, AP75, AP at small/medium/large scales; speed in FPS.
- **Backbones to instantiate:** ResNet-18, ResNet-101 (up-convolutional, output stride 4), DLA-34, Hourglass-104, spanning a speed/accuracy spectrum.
- **Output stride R = 4** (heatmap resolution = input/4), following the keypoint literature, much finer than detection's usual stride 16.
- **Training:** input 512×512 (output 128×128); Adam; data augmentation by random flip, random scaling, cropping, color jittering. (Loss weights, learning-rate schedule, and epoch counts are the per-task knobs to be set.)
- **Generality targets:** KITTI 3D detection (depth/orientation/dimensions metrics) and COCO human-pose keypoint AP — to test whether adding regression outputs at the center extends the method to new tasks.

## Code framework

The primitives already exist: fully-convolutional encoder–decoder backbones that emit a high-resolution (stride-4) feature map, the Gaussian-splatting target builder and penalty-reduced focal loss from the keypoint literature, and standard L1 regression. A detector is a backbone plus a set of **output heads** (each a 3×3 conv → ReLU → 1×1 conv producing a small number of channels), a **target builder**, a **loss**, and a **decoder**. What is unknown — the open slots — is exactly which heads to attach, what each predicts, how targets are built, and how detections are decoded without NMS.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_head(in_ch, out_ch, head_ch=256):
    # the standard keypoint output head: 3x3 conv -> ReLU -> 1x1 conv
    return nn.Sequential(
        nn.Conv2d(in_ch, head_ch, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(head_ch, out_ch, 1))

class PointDetector(nn.Module):
    """Fully-convolutional backbone (stride 4) + output heads. WHICH heads,
    and what each predicts, is the open design question."""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone        # emits a stride-4 feature map
        self.heads = nn.ModuleDict({
            # TODO: define the output heads (what does each location predict?)
        })

    def forward(self, x):
        feat = self.backbone(x)
        return {name: head(feat) for name, head in self.heads.items()}

def gaussian_splat_target(centers, classes, out_h, out_w, num_classes):
    """Render a target heatmap by splatting a size-adaptive 2D Gaussian at each
    object's (downscaled) center; element-wise max where Gaussians overlap."""
    pass  # TODO

def penalty_reduced_focal_loss(pred, gt):
    """Pixel-wise focal loss where near-peak negatives are penalized less."""
    pass  # TODO

def detector_loss(outputs, targets):
    """Heatmap loss + weighted regression losses for the other heads."""
    pass  # TODO

def decode(outputs, K=100):
    """Turn head outputs into boxes WITHOUT IoU-NMS: extract heatmap peaks,
    read off the per-point regressed quantities, assemble boxes."""
    pass  # TODO
```
