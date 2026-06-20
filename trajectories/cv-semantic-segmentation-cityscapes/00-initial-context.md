## Research question

Semantic segmentation asks for a *dense* prediction: a class label for every pixel of an image, not
a single label for the whole image. The task fixed here is **Cityscapes validation mIoU** — urban
street scenes, 19 classes (road, sidewalk, person, car, traffic sign, …), evaluated by the mean over
classes of the per-class intersection-over-union between the predicted and ground-truth masks. Higher
mIoU is better. Cityscapes is hard in a specific way: the classes span an enormous range of spatial
scale (a building or road fills most of the frame; a distant pole or traffic light is a handful of
pixels), the boundaries between classes are long and intricate, and getting a pixel's label right
often requires *context* far outside the patch immediately around it — a gray blob is "road" or
"sidewalk" depending on the whole scene layout.

Everything that defines the *problem* is frozen so the comparison is honest: the Cityscapes
train/val split, the 19-class mIoU metric, a single fixed input crop (**512×1024**), and a fixed
ImageNet-pretrained backbone setting for the convolutional rungs (**ResNet-50 dilated to output
stride 8**, "R-50-D8"). The single free variable is the **segmentation architecture** — how features
are computed, how context is aggregated, and how a coarse feature map is turned back into a
full-resolution dense prediction. Because the backbone, crop, and dataset are held fixed, mIoU is not
bought with more pixels or a bigger pretrained net; the only lever is the architecture that sits on
top of (and around) those features. The ladder is ranked on **Cityscapes val mIoU at the fixed crop**,
each rung a real measured configuration.

## Background — how a classifier becomes a dense predictor

The starting material is image *classification*. A decade of work produced convolutional networks
(AlexNet, VGG, ResNet) that map a whole image to one of N class scores, trained on ImageNet. Such a
network is a stack of convolutions and poolings that progressively *shrinks* the spatial resolution
(224×224 → 7×7) while *growing* the channel depth, then a global pooling and a fully-connected layer
collapse the 7×7 map to a vector of class scores. The features it learns are excellent — but the
architecture throws spatial resolution away on purpose, because for a single image-level label the
*where* does not matter, only the *what*.

Dense prediction breaks that assumption. To label every pixel you need both the *what* (the rich,
high-level, context-aware features that only appear deep in the network, after many strides of
pooling) and the *where* (full input resolution, so the prediction can be placed at the right pixel
and trace a boundary). These two pull against each other inside a classification backbone: depth buys
semantics at the cost of resolution. Two structural facts about the backbone matter throughout. (1) A
plain backbone's deepest feature map is downsampled by a factor of 32; its spatial grid for a
512×1024 crop is on the order of 16×32, far too coarse to localize a thin pole or a clean boundary.
(2) The *receptive field* — how much of the input each output unit can "see" — grows with depth and
pooling, so the context available to a pixel's prediction is itself a function of the architecture.

Two backbone modifications are standard tools at this point and are part of the fixed setting for the
convolutional rungs:

- **Dilated (atrous) convolution.** A convolution with dilation rate *r* spreads its kernel taps *r*
  pixels apart, enlarging the receptive field *without* downsampling and *without* adding parameters.
  Replacing the stride-2 downsampling in the last backbone stages with dilation keeps the feature map
  at a higher resolution (output stride 8 instead of 32) while preserving the pretrained weights'
  receptive field. The R-50-D8 backbone used here is exactly this: ResNet-50 with the last two stages
  dilated, `strides=(1, 2, 1, 1)`, `dilations=(1, 1, 2, 4)`, so its output is 1/8 resolution.
- **Bilinear upsampling.** Whatever resolution the network's prediction comes out at, it is resized
  to the full input resolution by bilinear interpolation for the final loss and metric.

## Baselines — the prior art a dense predictor reacts to

The methods a new segmentation architecture is measured against, and the gap each leaves open:

- **Per-patch / sliding-window classification.** The naive route: run a classification CNN on a patch
  centered at each pixel and take its class. It produces a dense map but is grotesquely expensive
  (one forward pass per pixel, with enormous redundant computation between overlapping patches) and
  its output resolution is tied to how densely you can afford to slide the window. It also fixes the
  context to one patch size, so it cannot adapt the amount of context to the object.
- **Backbone + global head, upsampled.** Take a classification backbone, drop the global pooling and
  the fully-connected classifier, and read off the 1/32-resolution feature map, then bilinearly
  upsample the class scores to full resolution. This is cheap and uses the pretrained features, but
  the prediction is born at 1/32 resolution: every fine spatial detail — boundaries, thin structures,
  small objects — has already been pooled away, and bilinear upsampling cannot invent it back. The
  output is semantically right but spatially blurry, and small or thin classes are systematically
  lost.

The shared limitation: a classification backbone gives strong *what* at coarse *where*, and the
prior art either pays an unaffordable cost to recover resolution (per-patch) or accepts a coarse,
blurry prediction (single upsample). The open problem each rung below attacks is how to recover full
spatial resolution and the right amount of context *cheaply*, on top of the same fixed backbone.

## Evaluation settings

- **Dataset.** Cityscapes — 2975 finely-annotated training images, 500 validation images, 19
  evaluation classes, 1024×2048 native resolution.
- **Metric.** Mean Intersection-over-Union (mIoU) over the 19 classes on the validation set; higher
  is better. The single number each rung is ranked on.
- **Input.** A fixed **512×1024** training crop for the convolutional rungs, with random scaling,
  cropping, and horizontal flip as standard augmentation; evaluation resizes predictions to full
  resolution.
- **Backbone / schedule (convolutional rungs).** ResNet-50 dilated to output stride 8 (R-50-D8),
  ImageNet-pretrained, with SyncBN; a 40k-iteration schedule. Held fixed across the convolutional
  rungs so that only the architecture differs.
- **Loss.** Per-pixel cross-entropy against the dense label map (an auxiliary cross-entropy head on
  an intermediate stage is a standard training aid).

## Code framework

The pieces that already exist before any segmentation head is designed: a dilated backbone that emits
a list of multi-scale feature maps, a generic "decode head" base class that owns the final 1×1
classification conv and the per-pixel loss, and a bilinear `resize`. The architecture to be designed
fills the body of the head.

```python
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

def resize(x, size, mode='bilinear', align_corners=False):
    return nn.functional.interpolate(x, size=size, mode=mode,
                                     align_corners=align_corners)

class BaseDecodeHead(nn.Module):
    """Owns the final 1x1 classifier and the per-pixel loss; subclasses
    implement forward() to turn backbone features into a feature map."""
    def __init__(self, in_channels, channels, num_classes,
                 dropout_ratio=0.1, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def cls_seg(self, feat):
        feat = self.dropout(feat)
        return self.conv_seg(feat)

    def _transform_inputs(self, inputs):
        # select / concat the backbone feature map(s) this head consumes
        ...

    def forward(self, inputs):
        raise NotImplementedError  # TODO: the architecture we will design

# backbone: dilated ResNet-50, output stride 8, emits a list of feature maps
#   inputs[i] = feature map at stage i (1/4, 1/8, 1/16, 1/32 before dilation)
# loss: per-pixel cross-entropy on cls_seg(forward(inputs)) upsampled to full res
```
