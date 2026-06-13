# Context

## Research question

The goal is a single, simple, fast, and general framework for **instance segmentation**: given an image,
detect every object and produce a *pixel-accurate mask for each individual instance*. The task fuses two
otherwise-separate problems. From **object detection** it inherits the need to find and classify every
object and localize it; from **semantic segmentation** it inherits per-pixel labeling. But unlike
semantic segmentation, it must separate *instances* — two overlapping objects of the same class must get
two distinct masks, not one merged blob.

What would a good solution have to achieve? It should (1) match the speed and training simplicity that
detection frameworks already enjoy, so that experiments take hours not weeks; (2) be accurate at *strict*
localization (high-IoU mask overlap), since a mask is only useful if its boundary is correct; (3) handle
the hard case of **within-category overlap** (many people, many cars, touching each other), which is where
naive per-pixel labeling collapses; and (4) be a *framework*, flexible enough to swap backbones and to
extend to other instance-level tasks. The prior art either solved detection well but said nothing about
masks, or solved per-pixel labeling well but could not separate instances, or solved instance
segmentation through slow, multi-stage, segmentation-first cascades.

## Background

The field state rests on a few load-bearing pieces.

**Regions + CNN features.** The dominant detection paradigm attends to a manageable number of candidate
object regions and runs a convolutional network on each. The decisive efficiency move was to run the
backbone *once* over the whole image and extract per-region features from the shared feature map, rather
than re-running the CNN on every cropped region.

**The region-feature extraction operator (RoIPool).** To turn a variable-size region into a fixed-size
feature for the per-region head, the standard operator maps a floating-point region box into the feature
map by dividing its coordinates by the feature stride and **rounding** to the nearest integer cell, e.g.
a coordinate `x` becomes `[x/16]` for a stride-16 map. That quantized region is then partitioned into a
fixed grid of bins (e.g. 7×7); each bin boundary is again rounded to integers; the features inside each
bin are aggregated by max pooling. There are therefore **two roundings**: region-to-grid and grid-to-bins.
Each rounding shifts the region by up to half a cell, so the extracted feature map is misaligned with the
true region by a few pixels at stride 16 and worse at stride 32. This is a known, accepted property: the
downstream classifier is robust to small translations, so the misalignment was never a problem *for
classification*. It is an open question what it costs a task that needs per-pixel spatial fidelity.

**Differentiable bilinear sampling.** A separate line of work introduced a differentiable layer that
samples a feature map at arbitrary, *non-integer* spatial locations by bilinear interpolation from the
four nearest grid points (Jaderberg et al. 2015, spatial transformer networks). Because the interpolation
weights are smooth functions of the sampling coordinates, gradients flow through both the sampled values
and (optionally) the coordinates. This is a general primitive for reading a feature map at non-integer
spatial positions.

**Fully convolutional prediction of spatial outputs.** For dense per-pixel tasks, the established recipe
is to drop the fully-connected layers and make the entire network convolutional, so the output is itself a
spatial map (Long et al. 2015). The output layer applies a **per-pixel softmax** over the C categories
with a multinomial cross-entropy loss, and the coarse output is upsampled (by deconvolution / bilinear
upsampling) back toward input resolution. The lesson carried forward: a *spatial* target should be
predicted by convolutions that preserve spatial layout, not by collapsing the feature into a flat vector.
The property worth noting: the per-pixel softmax makes the C categories **compete** at every pixel — each
pixel is forced to choose one class.

**Multi-scale features in-network.** A feature-pyramid construction builds a top-down pathway with lateral
connections from a single-scale input, producing a set of feature maps at multiple scales all with the
same channel count (Lin et al. 2017). Regions are routed to the pyramid level appropriate to their size.
This gives strong features for objects across scales at little extra cost, and is a natural backbone for a
per-region detector.

**Motivating empirical observations about existing systems.** Two facts about the prior art set up the
problem. First, region-feature extraction with quantization works fine for box classification but has
never been stress-tested on a task requiring pixel alignment; the misalignment magnitude grows with the
feature stride, so large-stride (stride-32) features have historically been considered too coarse for
precise localization. Second, semantic-segmentation FCNs, when adapted to separate instances, struggle on
overlapping instances of the same class: per-pixel multi-class labeling has no native notion of "which
instance", so touching same-class objects merge. Systems that predict a shared set of position-sensitive
channels for class, box, and mask jointly show systematic artifacts and spurious edges precisely on
overlapping objects — evidence that *coupling* the mask prediction with the classification is part of the
difficulty.

## Baselines

**R-CNN (Girshick et al. 2014).** Warp each region proposal to a fixed size, run the full CNN per region,
classify with SVMs, regress the box. Establishes regions+CNN but is slow (one CNN pass per region) and
multi-stage (separately trained CNN, classifier, regressor).

**Fast R-CNN (Girshick 2015).** Run the backbone once; extract per-region features with RoIPool from the
shared map; two sibling fully-connected heads predict, **in parallel**, a softmax over K+1 classes and a
per-class bounding-box regression. Trained end-to-end with a multi-task loss `L = L_cls + L_box`, where
`L_cls` is cross-entropy and `L_box` is a smooth-L1 loss applied to the regression outputs of the
ground-truth class only. Core contribution: parallel class+box siblings, single-stage training, shared
computation. Gap it leaves: it produces boxes, not masks, and its RoIPool quantizes coordinates.

**Faster R-CNN (Ren et al. 2015).** Replaces external proposals with a **Region Proposal Network**: a
small conv head slides over the shared feature map; at each location it scores k anchors (multiple scales
and aspect ratios) for objectness and regresses their box deltas, via two 1×1-conv siblings, trained with
the same cls+box multi-task loss. The full detector is two stages: stage 1 = RPN proposes regions; stage 2
= the Fast R-CNN head classifies and refines each region from RoIPool features; both stages share the
backbone. This is the leading detection framework and the natural base to build on. Gap: still box-only,
still RoIPool.

**FCN for semantic segmentation (Long et al. 2015).** Fully convolutional per-pixel classification with
per-pixel softmax + multinomial loss. Strong at semantic labeling; cannot separate instances (no instance
notion), and its softmax couples segmentation with classification. Gap: no instances.

**DeepMask / SharpMask (Pinheiro et al. 2015, 2016).** Learn to *propose* class-agnostic segment
candidates — a network outputs a mask, but via a **fully-connected** layer — and then classify each
segment with a Fast R-CNN. Segmentation precedes recognition. Gaps: the fc mask discards spatial structure
and needs many parameters; segment-then-classify is slow and less accurate; masks are not produced in
alignment with the detector.

**MNC — multi-task network cascade (Dai et al. 2016).** A three-stage cascade: propose boxes, predict
mask instances from boxes, then categorize. It introduced **RoIWarp**, which also uses bilinear resampling
— but it first *quantizes* the region exactly as RoIPool does, so it does not address alignment. Gaps: the
cascade makes later stages depend on earlier mask predictions (coupling), it is complex and multi-stage,
and RoIWarp performs essentially on par with RoIPool.

**FCIS — fully convolutional instance segmentation (Li et al. 2017).** Predicts a set of position-
sensitive output channels fully convolutionally that *simultaneously* encode class, box, and mask, which
makes it fast. Gap: the shared position-sensitive channels couple the tasks; it shows systematic errors on
overlapping instances and creates spurious edges.

## Evaluation settings

The natural yardstick is the COCO dataset (Lin et al. 2014): ~80 object categories with per-instance mask
and box annotations. The standard training split is the union of the train images and a subset of val
(`trainval35k`); ablations are reported on the held-out `minival` (5k val images); final numbers on
`test-dev`. The primary metric is **mask AP**: average precision computed with *mask* IoU between
predicted and ground-truth instance masks, averaged over IoU thresholds from 0.5 to 0.95; the suite also
reports AP at fixed IoU (AP50, AP75 — AP75 being the strict-localization measure) and AP at small/medium/
large object scales (APS/APM/APL). Box AP uses box IoU under the same protocol. For human pose, the
keypoint metric AP^kp uses object keypoint similarity. A second dataset, Cityscapes (Cordts et al. 2016),
provides fine per-instance annotations over 8 categories at 2048×1024, with large numbers of overlapping
same-category instances per image, and serves as a low-data, high-overlap stress test (mask AP and AP50).
These datasets, splits, and metrics all predate the method and are the standard instance-level benchmarks.

## Code framework

The pieces that already exist: a convolutional backbone producing a (possibly multi-scale) shared feature
map, a region-proposal stage, a fixed-size region-feature extractor, a per-region recognition head, the
optimizer and training loop, and the standard losses. The region-feature extractor below is the *existing*
quantizing operator; the per-region head below has only the box-recognition siblings. The open slots —
how region features are extracted, and what additional per-region output to add and how to supervise it —
are left as stubs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Existing: backbone over the whole image (single- or multi-scale map) ---
class Backbone(nn.Module):
    def forward(self, images):
        # returns a feature map (or a dict of multi-scale maps)
        ...

# --- Existing: region proposal stage (anchors -> objectness + box deltas) ---
class RegionProposalNetwork(nn.Module):
    def forward(self, features, image_shapes, targets=None):
        # returns proposals (list of [N,4] boxes per image) and rpn losses
        ...

# --- Existing region-feature extractor: maps a float RoI to a fixed HxW grid
#     by quantizing coordinates to the feature grid, then max-pooling bins. ---
def roi_pool(features, rois, output_size, spatial_scale):
    # quantize: rois_q = round(rois * spatial_scale); subdivide into output_size
    # bins (each bin boundary rounded); max-pool features within each bin.
    ...

# --- Open slot: an alternative region-feature extractor ---
def roi_extract(features, rois, output_size, spatial_scale):
    # TODO: extract a fixed output_size feature for each RoI from `features`.
    pass

# --- Existing per-region recognition head: class + box siblings (Fast R-CNN) ---
class BoxHead(nn.Module):
    def __init__(self, in_channels, representation_size, num_classes):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc6(x)); x = F.relu(self.fc7(x))
        return self.cls_score(x), self.bbox_pred(x)

# --- Open slot: an additional per-region output for spatial structure ---
class ExtraBranch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # TODO: the per-region branch we will design

    def forward(self, x):
        # TODO
        pass

def extra_loss(*args, **kwargs):
    # TODO: supervision for the extra per-region output
    pass

# --- Existing: assemble the two-stage detector and its multi-task loss ---
class TwoStageDetector(nn.Module):
    def __init__(self, backbone, rpn, box_head):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.box_head = box_head
        # TODO: region extractor(s) and the extra branch get wired in here

    def forward(self, images, targets=None):
        features = self.backbone(images)
        proposals, rpn_losses = self.rpn(features, images.shapes, targets)
        # region features -> box head -> class + box outputs
        # L = L_cls + L_box (+ the extra-branch loss, once defined)
        ...

# --- Existing: optimizer + training loop ---
def train_loop(model, data, opt):
    for images, targets in data:
        losses = model(images, targets)
        loss = sum(losses.values())
        opt.zero_grad(); loss.backward(); opt.step()
```
