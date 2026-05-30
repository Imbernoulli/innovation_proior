# Context

## Research question

Recognizing objects across a wide range of scales is a core difficulty in detection. The same object class can appear as a 20-pixel speck or fill the frame, and a detector must localize and classify it in both cases. The question is how to build a feature representation that is **strong at every scale** — so that a small object and a large object are each described by features rich enough to recognize and localize them — while paying only a **marginal** compute and memory cost over a plain single-pass forward of the backbone, and while remaining **trainable end-to-end** (the same computation at train and test time).

The two standard ways of getting multi-scale features each fail one of these requirements. Featurizing a pyramid of resized images gives strong features at every scale but is several times slower and is too memory-heavy to train end-to-end, so it is relegated to test time only. Running a single deep feature map is fast and end-to-end trainable but is coarse, so small objects are poorly represented. A solution would have to deliver the accuracy of the former at close to the cost of the latter.

## Background

**Scale and the image pyramid.** The classic device for scale invariance is the image pyramid (Adelson et al. 1984): resize the image to a geometric series of scales (e.g. several scales per octave) and process each. An object's change in size is then offset by a shift in pyramid level, so a fixed-size model scanned over positions *and* levels covers all sizes. In the hand-engineered-feature era this was essential — HOG (Dalal & Triggs 2005) and SIFT (Lowe 2004) were computed densely over entire image pyramids, and part-based detectors like DPM (Felzenszwalb et al. 2010) needed dense scale sampling (~10 scales per octave) to work well. Fast pyramid computation (Dollár et al. 2014) reduced the cost by computing a sparse set of scales and interpolating the rest, but the per-scale featurization remained the bottleneck. Early ConvNet face detectors (Vaillant et al. 1994; Rowley et al. 1995) likewise ran a shallow net over an image pyramid.

**ConvNets and the in-network feature hierarchy.** Deep ConvNets (LeCun et al. 1989; Krizhevsky et al. 2012) replaced engineered features. Beyond representing higher-level semantics, ConvNet features are partially robust to scale, which made detection from a *single* input scale viable. A ConvNet also computes, by construction, a **feature hierarchy**: with subsampling layers the activation maps form a multi-scale, pyramidal sequence of decreasing spatial resolution and increasing depth. Layers that output the same spatial size form a *stage*; for a ResNet the last-block outputs of conv2–conv5 are maps with strides {4, 8, 16, 32} relative to the input. This hierarchy is produced for free in the forward pass.

The crucial diagnostic fact about this hierarchy is a **semantic gap across levels**: the high-resolution early maps have only low-level features (edges, textures) and are weak for object recognition, while the low-resolution deep maps are semantically strong but too coarse to localize small objects. So although the shape is pyramidal, the levels are *not* equally usable — unlike a featurized image pyramid, whose every level (including the high-resolution ones) is semantically strong because the full network was run at that scale.

**Observed limitations that motivate the problem.** Featurizing an image pyramid multiplies inference time (reported roughly fourfold), and training a deep network end-to-end on an image pyramid is infeasible in memory; consequently image pyramids are used only at test time, creating a train/test inconsistency. For these reasons single-scale region-based detectors deliberately forgo image pyramids by default. Yet multi-scale testing on featurized image pyramids remained necessary for the top results on ImageNet and COCO detection — direct evidence that the single-scale map leaves accuracy on the table, especially for small objects.

## Baselines

**Featurized image pyramid.** Resize the input to several scales, run the backbone on each, detect on each level's features, combine. *Core property:* all levels are semantically strong, so one shared head applies at every scale. *Gap:* multiplicative inference cost; memory-infeasible to train end-to-end; used only at test time.

**Single-scale region-based detection (Fast / Faster R-CNN; Girshick 2015; Ren et al. 2015).** SPPnet (He et al. 2014) showed region-based detection can run on features from a single image scale by pooling per-region features from a shared feature map, instead of re-running the net per region. Fast R-CNN pools RoI features (RoIPool to a fixed 7×7) from one map and classifies/regresses with a small head. Faster R-CNN adds a Region Proposal Network: a 3×3 conv plus two sibling 1×1 convs (objectness and box regression) slid over a single feature map, with multi-scale, multi-aspect-ratio *anchors* as reference boxes; positives/negatives are assigned by IoU with ground truth (≥0.7 positive, <0.3 negative). A ResNet realization uses C4 as the single map and the conv5 stack as the head. *Gap:* one coarse map (stride 16 or 32) represents all object sizes; small objects are under-resolved, so accuracy on small objects is weak. Choosing a deeper single map trades resolution for semantics and does not resolve the conflict.

**In-network pyramidal-feature detection (SSD; Liu et al. 2016).** SSD predicts from several layers of the ConvNet feature hierarchy directly, treating them as if they were a featurized image pyramid — and so reuses maps already computed in the forward pass, which is cheap. *Gap:* to avoid the semantically weak shallow maps, SSD starts its pyramid high in the network (around conv4_3 of VGG) and *adds new layers* below, rather than reusing the existing high-resolution early maps. It therefore skips exactly the high-resolution features that matter for small objects, and its levels still differ in semantic strength.

**Top-down/skip-connection architectures (U-Net, Ronneberger et al. 2015; SharpMask, Pinheiro et al. 2016; Stacked Hourglass, Newell et al. 2016; Recombinator, Honari et al. 2016; Ghiasi & Fowlkes 2016).** These combine coarse-strong and fine-weak features via top-down and lateral/skip connections. *Gap:* they fuse everything down to a *single* fine-resolution map and predict only there; they do not make independent predictions at each level as an image pyramid does, and recognizing objects across scales on such a single map can still require an image pyramid. Methods that merely *concatenate* features of multiple layers before predicting (HyperNet, Kong et al. 2016; ParseNet, Liu et al. 2015; ION, Bell et al. 2016; Hypercolumns, Hariharan et al. 2015; FCN, Long et al. 2015) are a related family — equivalent to summing transformed features into one prediction site.

**Dense mask-proposal systems (DeepMask, Pinheiro et al. 2015; SharpMask, Pinheiro et al. 2016; InstanceFCN, Dai et al. 2016).** DeepMask and SharpMask train a crop-based network to decide objectness and predict an instance mask, then run the network convolutionally over an image pyramid at inference so masks of many sizes are covered. Their scale handling is inherited from the dense image pyramid, often with two scales per octave. *Gap:* the same multi-scale cost appears again: many resized images must be processed, training and inference are not the same fully convolutional computation, and small-object mask proposals remain costly.

## Evaluation settings

The natural yardstick is COCO detection (Lin et al. 2014), 80 categories, trained on the union of train and a 35k val subset (`trainval35k`), ablated on a 5k val subset (`minival`), with final evaluation on the held-out `test-dev`/`test-std`. Metrics: COCO-style Average Precision averaged over IoU thresholds (and AP@0.5, PASCAL-style), broken out by object size into AP_s / AP_m / AP_l for small / medium / large objects. For proposal quality, Average Recall (AR) at fixed proposal budgets (AR^100, AR^1k) and by size (AR_s/m/l). Backbones are ResNet-50/101 pre-trained on ImageNet-1k (Russakovsky et al. 2015) then fine-tuned. For instance-segment proposals, segment AR at 1000 proposals against DeepMask/SharpMask/InstanceFCN on COCO validation images. Inference speed (FPS, seconds/image on one GPU) is also a yardstick, since the whole motivation is cost.

## Code framework

The starting code has a backbone ConvNet that exposes its per-stage maps, RoI pooling, an anchor-based RPN head, a Fast R-CNN head, dense mask-prediction heads, and a standard SGD detection training loop. The empty slots are the multi-scale feature builder and the size-based routing rule for region pooling.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- already exists: backbone exposing per-stage feature maps ---
class Backbone(nn.Module):
    """ConvNet that returns its stage outputs as a dict, e.g.
    {"c2": ..., "c3": ..., "c4": ..., "c5": ...} with strides {4,8,16,32}."""
    def forward(self, x):
        # standard feedforward conv stack; returns dict of stage maps
        ...

# --- already exists: RoI pooling to a fixed spatial size ---
def roi_pool(feature_map, boxes, output_size=7):
    # SPP/Fast-R-CNN style pooling of each region to output_size x output_size
    ...

# --- already exists: RPN head (Faster R-CNN), one feature map in ---
class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox = nn.Conv2d(in_channels, num_anchors * 4, 1)
    def forward(self, x):
        t = F.relu(self.conv(x))
        return self.cls(t), self.bbox(t)

# --- already exists: Fast R-CNN detection head ---
class FastRCNNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        ...  # pooled features -> hidden layers -> (cls_score, bbox_pred)
    def forward(self, pooled):
        ...

class MultiScaleFeatureBuilder(nn.Module):
    """TODO: take the backbone's bottom-up stage maps {c2..c5} (high-res/weak
    to low-res/strong) and produce a set of output maps, one per stage, that are
    all equally usable by a single shared head. Design the construction."""
    def __init__(self, in_channels_per_stage, out_channels, add_extra_coarse_level=True):
        super().__init__()
        pass  # TODO

    def forward(self, bottom_up_maps):
        pass  # TODO: return a finest-to-coarsest list of maps


def assign_roi_to_level(
    boxes,
    min_level=2,
    max_level=5,
    canonical_level=4,
    canonical_box_size=224.0,
):
    """TODO: choose which output map each RoI should be pooled from."""
    pass  # TODO

# --- already exists: training loop over the detection dataset ---
def train(model, data_loader, optimizer):
    for images, targets in data_loader:
        feats = model.backbone(images)          # bottom-up maps
        # feats = model.feature_builder(feats)  # <- filled in by the contribution
        loss = model.detector_loss(feats, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```
