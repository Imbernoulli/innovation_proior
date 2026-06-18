# Context

## Research Question

Semantic segmentation asks for a class label at every pixel. A strong starting point is an ImageNet-pretrained classification backbone, because its late features recognize objects well. The difficulty is that classification backbones are built to become invariant to position, while dense prediction needs both object identity and spatially precise responses.

Two constraints have to be satisfied at the same time. First, the late feature map cannot be too coarse: a stride-32 map is cheap, but it makes boundaries and small objects hard to recover. Second, the classifier cannot use only one fixed field of view: the same class can occupy a few pixels or most of the image. The open design problem is to keep the pretrained backbone useful while computing dense features and adding multi-scale context in one forward pass.

## Background

Fully convolutional networks showed how to convert classification networks into dense predictors: replace fully connected layers with convolutions, predict a coarse score map, and upsample it to the image grid. This keeps whole-image computation efficient, but the coarse score map remains a bottleneck.

Atrous, or dilated, convolution is the main resolution tool already available. For a two-dimensional signal, output location **i**, filter **w**, input **x**, and atrous rate `r`, it computes

```text
y[**i**] = sum_**k** x[**i** + r * **k**] * w[**k**].
```

The rate is the spacing between sampled input positions. `r = 1` is ordinary convolution; `r > 1` is equivalent to inserting `r - 1` holes between adjacent filter taps. This enlarges the effective field of view without adding parameters and without reducing the output grid.

The useful bookkeeping quantity is `output_stride`: input spatial resolution divided by final feature-map resolution. A standard classification backbone usually has `output_stride = 32`. If the last stride-2 operation is removed, subsequent convolutions can use atrous rate 2 so that features are computed twice as densely while retaining roughly the same field of view in input pixels.

Spatial pyramid pooling and related global-context methods show that a classifier can benefit from summaries at multiple spatial scales. In dense prediction, those summaries have to be merged back into a feature map so that each pixel still receives a class score.

## Baselines

**FCN with upsampling.** A fully convolutionalized classifier predicts on a coarse grid and then uses bilinear or learned transposed-convolution upsampling. Skip connections can fuse shallower detail with deeper semantics. The gap is that much fine spatial information has already been discarded by the time the deepest score map is formed.

**Encoder-decoder networks.** SegNet and U-Net use a downsampling encoder for context and a learned decoder, often with pooling indices or skip connections, to recover spatial detail. The gap is extra decoder complexity and the need to reconstruct detail after aggressive downsampling.

**Earlier atrous segmentation systems.** Earlier systems in this line use atrous convolution to compute denser features, and the second version adds atrous spatial pyramid pooling with several parallel rates. They still rely on a separate DenseCRF post-process for boundary refinement, and their pyramid module does not train batch-normalization layers inside the added context head.

**Image pyramids and cascaded context modules.** Running the same network on resized inputs handles object scale but is expensive, especially for deep backbones. Cascading extra convolutional modules can enlarge context, but if those modules keep striding, they quickly decimate the feature map.

## Evaluation Setting

The standard benchmark is semantic segmentation on PASCAL VOC 2012, with 20 foreground object classes plus background, evaluated by mean intersection-over-union across classes. Cityscapes is another relevant dense urban-scene benchmark.

The intended backbone family is ResNet-50 or ResNet-101 pretrained on ImageNet, adapted to a chosen output stride for segmentation. Training uses crop-based dense supervision, SGD with a polynomial learning-rate schedule, and bilinear resizing of logits or predictions back to the image grid for pixel-aligned evaluation.

## Code Framework

The primitives are already present in a modern deep-learning library: a pretrained convolutional backbone whose later stride can be replaced by dilation, ordinary `1x1` and `3x3` convolutions, batch normalization, ReLU, average pooling, and bilinear interpolation. The open design slot is the context head placed between backbone features and final per-pixel logits.

```python
import torch
from torch import nn
from torch.nn import functional as F

class SegHead(nn.Module):
    """Map a backbone feature map to per-pixel class logits."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # The context aggregation mechanism is the open design slot.
        pass

    def forward(self, x):
        raise NotImplementedError

class SegModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)["out"]
        logits = self.classifier(features)
        return F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=True)
```
