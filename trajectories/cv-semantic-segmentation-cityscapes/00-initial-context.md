## Research question

Semantic segmentation asks for a *dense* prediction: a class label for every pixel. The fixed task is **Cityscapes validation mIoU** — 19 urban classes, evaluated by the mean per-class intersection-over-union between predicted and ground-truth masks. Cityscapes is hard because classes span a huge range of spatial scales and because a pixel's correct label often depends on context far outside its immediate neighborhood.

Everything about the problem is frozen for a fair comparison: the Cityscapes train/val split, the 19-class mIoU metric, a single fixed **512×1024** input crop, and a fixed ImageNet-pretrained **ResNet-50 dilated to output stride 8** (R-50-D8). The only free variable is the **segmentation architecture** that sits on top of the backbone. The ladder is ranked on **Cityscapes val mIoU at the fixed crop**, each rung a real measured configuration.

## Prior art / Background / Baselines

Image classification networks shrink spatial resolution and grow channel depth to produce a single label. Semantic segmentation needs both the high-level features those networks learn and the full spatial resolution they discard. Two standard tools are already part of the fixed substrate:

- **Dilated (atrous) convolution** spreads kernel taps apart, enlarging receptive field without extra downsampling or extra parameters. The fixed R-50-D8 backbone replaces the last two stride-2 stages with dilation, giving `strides=(1, 2, 1, 1)` and `dilations=(1, 1, 2, 4)`, so its deepest feature map is 1/8 resolution.
- **Bilinear upsampling** resizes any low-resolution class score map to full input resolution for the final loss and metric.

The methods a new segmentation architecture is measured against:

- **Per-patch / sliding-window classification.** Run a classification CNN on a fixed-size patch centered at each pixel and take its class.
- **Backbone + global head, upsampled.** Drop the global pooling and fully-connected classifier, attach a 1×1 conv to the deepest 1/32-resolution feature map, and bilinearly upsample the scores to full resolution.

The open question is what architecture to place on top of the fixed backbone to raise mIoU.

## Fixed substrate / Code framework

The frozen pieces: a dilated ResNet-50 output-stride-8 backbone that emits multi-scale feature maps, a generic decode-head base class that owns the final 1×1 classifier and per-pixel loss, and a bilinear `resize`.

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

## Editable interface

The only part to design is the decode head's `forward()`. It receives the backbone's list of feature maps and must return a feature map that `cls_seg()` turns into class logits. The head decides how to read the backbone features, combine them, and produce a dense map for the final 512×1024 prediction.

## Evaluation settings

- **Dataset.** Cityscapes — 2975 finely-annotated training images, 500 validation images, 19 evaluation classes, 1024×2048 native resolution.
- **Metric.** Mean Intersection-over-Union (mIoU) over the 19 classes on the validation set; higher is better. The single number each rung is ranked on.
- **Input.** A fixed **512×1024** training crop for the convolutional rungs, with random scaling, cropping, and horizontal flip as standard augmentation; evaluation resizes predictions to full resolution.
- **Backbone / schedule (convolutional rungs).** ResNet-50 dilated to output stride 8 (R-50-D8), ImageNet-pretrained, with SyncBN; a 40k-iteration schedule. Held fixed across the convolutional rungs so that only the architecture differs.
- **Loss.** Per-pixel cross-entropy against the dense label map (an auxiliary cross-entropy head on an intermediate stage is a standard training aid).
