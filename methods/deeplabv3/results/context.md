# Context

## Research question

Semantic segmentation asks for a class label at every pixel. The obvious move is to reuse a deep convolutional network pretrained for image classification as a feature extractor and attach a per-pixel classifier. But two structural problems get in the way, and a good solution must address both.

First, reduced resolution. Classification networks deliberately throw away spatial resolution: repeated max-pooling and strided convolution shrink the feature map by a factor of about 32 in each direction. That progressive downsampling builds the translation-invariant, abstract features that make classification work — but invariance to local position is exactly the wrong property for dense prediction, where the answer is the precise location of each label. A segmentation model needs feature responses computed at a much higher spatial density than a classifier produces.

Second, multiple scales. The same object class appears at wildly different sizes in different images (and different instances within one image). A network with a single, fixed receptive field cannot simultaneously see enough context to recognize a large object and enough resolution to delineate a small one. The model needs to probe its input at multiple effective fields-of-view.

The precise goal: take an ImageNet-pretrained backbone and (a) recover dense, high-resolution feature responses without retraining from scratch or exploding the parameter count, and (b) aggregate context across many scales, so that a single forward pass labels pixels accurately across object sizes.

## Background

The field state is a set of fully-convolutional segmentation models and a small toolkit of resolution and context mechanisms.

Fully Convolutional Networks (Long et al. 2014; Sermanet et al. 2013) established that you can run a classification net convolutionally over a whole image to produce a coarse label map, then upsample. The upsampling is usually done with learned deconvolution (transposed convolution). The coarse map is the bottleneck.

Atrous (a.k.a. dilated) convolution is the load-bearing primitive. Originally the "algorithme à trous" for the undecimated wavelet transform (Holschneider et al. 1989), it was brought into DCNNs by Giusti et al. (2013), Sermanet et al. (2013, OverFeat), and Papandreou et al. (2014). For a 2-D signal, output location **i**, filter **w**, input **x**, it computes y[**i**] = Σ_**k** x[**i** + r·**k**] · w[**k**]. The atrous rate r is the stride at which the input is sampled — equivalently, r−1 zeros ("holes", trous in French) are inserted between consecutive filter taps. r=1 is ordinary convolution; larger r enlarges the filter's field-of-view with no extra parameters and no loss of resolution. Crucially it lets you control the *output stride* of a network — the ratio of input resolution to output resolution: if you delete the stride from a late downsampling layer (cutting output stride from 32 to 16) and give the subsequent convolutions rate r=2, the receptive field is preserved while the feature map stays twice as dense.

Spatial pyramid pooling (Grauman & Darrell 2005; Lazebnik et al. 2006; He et al. 2014, SPP) showed that resampling features at multiple scales lets a model classify regions of arbitrary size. ParseNet (Liu et al. 2015) added global image-level context by average-pooling the whole feature map. PSPNet (Zhao et al. 2016) pooled at several grid scales.

A useful taxonomy of how prior segmentation models capture multi-scale context: (1) image pyramid — run the (shared-weight) network on several rescaled copies of the input and fuse; (2) encoder-decoder — a downsampling encoder for context plus an upsampling decoder (with skip connections) to recover detail; (3) context module — extra layers cascaded on top, e.g. a DenseCRF (Krähenbühl & Koltun 2011) or stacked convolutions, to encode long-range relations; (4) spatial pyramid pooling — probe one feature map with parallel filters/pooling at multiple rates.

A diagnostic fact that shapes the design: when a 3×3 atrous convolution's rate grows toward the size of the feature map, fewer and fewer of its taps land on the valid (non-padding) region. In the limit the 3×3 filter degenerates — only its center tap is effective, so it behaves like a 1×1 convolution and captures no spatial context at all. (Measured on a 3×3 filter applied to a 65×65 feature map: the normalized count of valid weights falls as the rate rises.) So you cannot obtain global context simply by cranking the atrous rate.

## Baselines

**FCN + deconvolution / encoder-decoder (SegNet, U-Net).** A classification backbone produces a 32×-downsampled label map; deconvolution or a decoder with skip connections recovers resolution. SegNet (Badrinarayanan et al. 2015) reuses encoder pooling indices; U-Net (Ronneberger et al. 2015) adds encoder→decoder skip connections. Gap: detail lost in the heavy downsampling cannot be fully recovered by upsampling; the decoder adds parameters and complexity.

**DeepLab v1/v2 with atrous convolution + ASPP + DenseCRF.** DeepLab v1/v2 (Chen et al. 2014, 2016) introduced atrous convolution for dense features and Atrous Spatial Pyramid Pooling — parallel atrous convolutions at different rates probing one feature map at several fields-of-view — and refined the output with a DenseCRF post-processing step that sharpens boundaries via pixel-pairwise potentials. Gaps: the ASPP of that era did not use batch normalization in the module; the DenseCRF is a separate, non-end-to-end post-process; and ASPP alone, with large rates, suffers the tap-degeneration problem above and so misses true global context.

**Cascaded context modules / pure deeper networks.** Stacking extra convolutional layers (or extra residual blocks) on top to enlarge the receptive field. If each added block downsamples, output stride grows quickly (duplicating ResNet's last block out to block7 would give output stride 256). Gap: consecutive striding decimates spatial detail, which is known to hurt segmentation.

## Evaluation settings

- **Datasets/metric**: PASCAL VOC 2012 semantic segmentation (20 object classes + background); also Cityscapes. Metric: mean intersection-over-union (mIoU) over classes.
- **Backbones**: ResNet-50 / ResNet-101 pretrained on ImageNet (optionally COCO), adapted to a chosen output stride (8 or 16) via atrous convolution.
- **Training protocol** that existed at the time: SGD with a "poly" learning-rate schedule, lr scaled by (1 − iter/max_iter)^0.9; large crop size (~513) so that large atrous rates remain effective; batch normalization parameters fine-tuned, which needs a sufficiently large crop and batch.
- **Output handling**: the head's coarse logits are bilinearly upsampled to the input resolution to produce the final per-pixel prediction.

## Code framework

The primitives exist: a pretrained ResNet backbone whose later stages can be made atrous by passing a `dilation` to `nn.Conv2d` (and dropping a stride), plus `nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`, adaptive average pooling, and `F.interpolate` for bilinear upsampling. A segmentation model is: backbone → a context "head" that maps backbone features to per-pixel class logits → upsample to the input size. The head is the open slot.

```python
import torch
from torch import nn
from torch.nn import functional as F

class SegHead(nn.Module):
    """Maps backbone feature map -> per-pixel class logits.
    The internal context-aggregation mechanism is the open design question."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # TODO: the context module that captures multi-scale / global context
        pass

    def forward(self, x):
        # TODO
        pass

class SegModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone      # ImageNet-pretrained, adapted to a chosen output stride
        self.classifier = classifier  # SegHead

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)["out"]
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x
```
