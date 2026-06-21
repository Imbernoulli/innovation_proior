# Context: Real-Time Object Detection

## Research question

Given a natural image, find every object in it: a tight bounding box around each one and a class label for each box. This is "object detection" — simultaneously *what* and *where*, for an unknown number of objects per image.

By the mid-2010s the accuracy of detectors had jumped sharply. The leading approaches typically ran multiple separate stages — feature extraction, region proposal or enumeration, classification, and box refinement — trained independently against their own proxy objectives. Inference cost at the rates demanded by interactive and video applications (roughly 30 frames per second) was a standing concern. How to design a detection system that trains against the actual detection objective and runs fast is the open question.

## Background

**Detection as a repurposed classifier.** The prevailing paradigm framed detection as classification applied at many image locations and scales. You take a classifier that says "object / not object" (and which class) and you evaluate it densely — over a sliding window across an image pyramid, or over a set of candidate regions — then clean up the resulting boxes with non-maximal suppression (NMS).

**Hand-engineered features and the deformable-parts era.** Classical pipelines began by extracting robust local features — Haar wavelets (Papageorgiou, Oren & Poggio 1998), SIFT (Lowe 1999), HOG (Dalal & Triggs 2005) — and then ran a classifier or a structured localizer over the feature map. The high-water mark of this line was the **deformable parts model** (Felzenszwalb, Girshick, McAllester & Ramanan 2010): a coarse "root" HOG filter plus several higher-resolution part filters arranged by a star-shaped deformable spatial model, scored by a latent SVM, evaluated as a sliding window over a HOG pyramid. DPM has a genuinely strong model of object shape and layout (root + parts + allowed deformations), which is why it degrades gracefully when the pixel statistics change.

**The CNN-feature revolution and region proposals.** Two things changed the field. (1) Convolutional features learned on ImageNet vastly outperformed hand-engineered descriptors for recognition. (2) **Region proposals** replaced exhaustive sliding windows: Selective Search (Uijlings, van de Sande, Gevers & Smeulders 2013) merges superpixels bottom-up to produce ~2000 class-agnostic candidate boxes per image with high recall, so a detector only has to classify those rather than millions of windows. Edge Boxes (Zitnick & Dollár 2014) is a faster alternative in the same spirit. Proposals made it feasible to put an expensive CNN behind the classifier.

**A minority lineage: predict boxes by regression, with a net.** A separate thread tried to make the network *output geometry directly* rather than classify pre-cut windows. OverFeat (Sermanet et al. 2013) trained one ConvNet for classification and localization and applied its fully-connected layers convolutionally, so a grid of bounding-box regressions could be produced in a single efficient pass. MultiBox / scalable detection (Erhan, Szegedy, Toshev & Anguelov 2014) trained a CNN to regress a fixed set of class-agnostic boxes with confidences, directly replacing Selective Search. Grasp detection (Redmon & Angelova 2014) regressed a single rectangle from an image over a coarse spatial grid. These showed a CNN can emit boxes directly.

**Backbone building blocks.** Network-in-Network (Lin, Chen & Yan 2013) introduced 1×1 convolutions as a cheap way to add nonlinearity and to reduce channel dimensionality between expensive layers. GoogLeNet / Inception (Szegedy et al. 2014) built a 22-layer classification network around that idea, placing 1×1 reductions before costly 3×3 and 5×5 convolutions to control compute. Adding fresh convolutional and fully-connected layers on top of an ImageNet-pretrained backbone was shown to help when adapting a classification network to detection (Ren, He, Girshick, Zhang & Sun 2015). Pretraining typically happened at 224×224 input resolution.

**Diagnostic findings about existing detectors.** Hoiem, Chodpathumwan & Dai (2012) supplied a useful error-analysis vocabulary: sort each category's top detections into correct / localization / similar-class / other-class / background, and ask which failure mode dominates.

## Baselines

**Deformable Parts Model (DPM).** Sliding-window detector over a HOG feature pyramid; a root filter plus part filters with a star deformation model, scored by a latent SVM, then box prediction and NMS. Core score for a placement: filter responses at the root and each part, minus a deformation penalty for displacing parts from their anchors. Explicit, interpretable spatial model; robust across domains. Specially engineered GPU variants (30Hz DPM, Sadeghi & Forsyth 2014) reach real time.

**R-CNN.** Selective Search proposes ~2000 regions; each is warped to a fixed size and pushed through a CNN; per-class linear SVMs score the CNN features; a class-specific linear regressor refines the box; NMS removes duplicates. CNN features gave a large accuracy jump over HOG.

**Fast R-CNN.** Runs the CNN once over the whole image, then for each proposal does RoI pooling on the shared feature map and applies a softmax classifier and a box regressor trained jointly with a multi-task loss. Much faster than R-CNN, more accurate, single training stage for the head.

**Faster R-CNN.** Replaces Selective Search with a Region Proposal Network that shares convolutional features with the detector and learns to emit proposals. Proposals become learnable and cheap; accuracy is high.

**OverFeat.** A single ConvNet does classification and localization; FC layers applied convolutionally yield efficient multi-scale sliding-window box regression.

**MultiBox.** A CNN regresses a fixed set of class-agnostic boxes with confidence scores to replace Selective Search, feeding a downstream patch-classification stage.

## Evaluation settings

The standard benchmark is **PASCAL VOC** (Everingham et al.), with the VOC 2007 and VOC 2012 detection sets, 20 object classes. Models are trained on the VOC 2007+2012 train/val splits (test sets sometimes added across years per the challenge rules) and evaluated on the held-out test sets. The metric is **mean average precision (mAP)**: a detection counts as correct if its class is right and its IOU with a ground-truth box exceeds 0.5; precision-recall is computed per class, average precision is the area under that curve, and mAP averages AP over the 20 classes. Speed is reported as frames per second / inference time per image on a single GPU.

For fine-grained error analysis the Hoiem et al. (2012) protocol categorizes each top-N detection as correct / localization error (right class, 0.1 < IOU < 0.5) / similar-class / other-class / background (IOU < 0.1 with any object). For cross-domain generalization, person detection is evaluated on artwork sets — the Picasso dataset (Ginosar et al. 2014) and the People-Art dataset (Cai et al. 2015) — with models trained on natural-image VOC data and tested on artwork, reported as AP and best F1. Backbones are pretrained on the ImageNet 1000-class classification dataset (Russakovsky et al.) and top-5 accuracy on the ImageNet 2012 validation set gauges backbone quality.

## Code framework

The pieces below already exist before the method: a CNN backbone abstraction, ImageNet pretraining, an optimizer with momentum/weight decay, and a training loop. What does *not* exist is the detection head — how the network's spatial feature map is turned into boxes and labels, and the objective it is trained against. Those are the empty slots.

```python
import torch
import torch.nn as nn

# A convolutional feature extractor (NIN/GoogLeNet-style: 1x1 channel reductions
# interleaved with 3x3 convs), pretrained on ImageNet.
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.act(self.conv(x))

def build_backbone(in_channels=3):
    # stack of CNNBlocks + maxpools producing a spatial feature map
    raise NotImplementedError  # TODO: the conv stack

# The open slot: how to turn the image into detections, and what to train it against.
class Detector(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = build_backbone()
        # TODO: the head that emits the detection prediction for an image
    def forward(self, x):
        x = self.backbone(x)
        # TODO: produce the output prediction
        raise NotImplementedError

class DetectionLoss(nn.Module):
    # TODO: the objective that scores predicted boxes+labels against ground truth
    def forward(self, predictions, targets):
        raise NotImplementedError

# Optimizer + training loop.
def train(model, loader, epochs):
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    criterion = DetectionLoss()
    for _ in range(epochs):
        for images, targets in loader:
            preds = model(images)
            loss = criterion(preds, targets)
            opt.zero_grad(); loss.backward(); opt.step()
```
