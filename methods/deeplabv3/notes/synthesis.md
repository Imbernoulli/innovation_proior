# DeepLabv3 synthesis

## Verified arXiv id
1706.05587 "Rethinking Atrous Convolution for Semantic Image Segmentation" (Chen, Papandreou, Schroff, Adam; Google, 2017).
Canonical code: torchvision.models.segmentation.deeplabv3 (DeepLabHead, ASPP, ASPPConv, ASPPPooling) + _SimpleSegmentationModel (fetched to code/).

## Two pain points (the problem)
Semantic segmentation = per-pixel labeling. Applying ImageNet-pretrained DCNNs (built for classification) gives two problems:
1. Reduced feature resolution: repeated pooling/striding shrinks the feature map by 32× (output_stride=32). That invariance helps classification but destroys the spatial detail dense prediction needs.
2. Objects at multiple scales: a fixed receptive field can't handle small and large instances at once.

## Atrous (dilated) convolution — the core tool
For output location i, filter w, input x:
  y[i] = Σ_k x[i + r·k] · w[k]
r = atrous rate = stride at which input is sampled = inserting r−1 zeros between consecutive filter taps. r=1 is standard conv. Enlarges field-of-view WITHOUT extra parameters and WITHOUT lowering resolution.
output_stride = input resolution / output resolution. To go from OS=32 to OS=16: set the stride of the last downsampling layer to 1, and replace subsequent convs with atrous rate r=2 (compensates for the removed stride so the receptive field is preserved). Generally, removing a stride of s and using rate s keeps the field of view.

## Two layouts explored
(a) Cascade: duplicate ResNet's last block (block4) several times → block5,6,7. Naively each adds stride-2 (would give OS=256). But consecutive striding decimates detail → harmful for segmentation. So instead keep resolution fixed (e.g. OS=16) and use atrous convolution with rates set by desired OS.
- Multi-grid: within block4..block7, the three 3×3 convs get unit rates Multi_Grid=(r1,r2,r3); final rate = unit_rate × block_rate. E.g. OS=16, Multi_Grid=(1,2,4) → block4 rates = 2·(1,2,4) = (2,4,8).

(b) Parallel = ASPP (Atrous Spatial Pyramid Pooling, revisited from DeepLabv2 / Chen 2016). Four parallel branches at different atrous rates probe the feature map at multiple effective fields-of-view (inspired by spatial pyramid pooling: Grauman 2005, Lazebnik 2006, He 2014 SPP).

## The ASPP degeneration problem + image-level features (key insight)
As atrous rate grows toward the feature-map size, the number of VALID filter weights (taps landing on real feature region, not zero-padding) shrinks. In the extreme, a 3×3 atrous conv with very large rate degenerates to a 1×1 conv — only the center tap is effective. (Illustrated: 3×3 filter on 65×65 map, normalized valid-weight count drops as rate rises.) So you can't capture global context just by cranking the rate.
Fix: add image-level (global) features. Apply global average pooling on the last feature map → 1×1 conv with 256 filters + BN → bilinearly upsample back to spatial size. (Like ParseNet Liu 2015, PSPNet Zhao 2016.)

## Final improved ASPP (output_stride=16)
Five parallel branches, each 256 filters + BN (+ ReLU):
(a) one 1×1 conv,
(b) three 3×3 atrous convs with rates = (6, 12, 18),
(c) image-level features (GAP → 1×1 conv 256 + BN → bilinear upsample).
Concatenate all five → 1×1 conv (256 + BN) → final 1×1 conv → logits (num_classes).
Rates doubled to (12,24,36) when output_stride=8. BN inside ASPP is new vs DeepLabv2 (found important to train).
DeepLabv3 drops the DenseCRF post-processing of v1/v2.

## torchvision code grounding
- ASPPConv: Conv2d(3×3, padding=dilation, dilation=dilation, bias=False) + BN + ReLU.
- ASPPPooling: AdaptiveAvgPool2d(1) → Conv2d 1×1 + BN + ReLU → F.interpolate(bilinear) back to size.
- ASPP: [1×1 branch] + [ASPPConv per rate] + [ASPPPooling]; concat; project = Conv2d 1×1 (len·256 → 256) + BN + ReLU + Dropout(0.5).
- DeepLabHead = Sequential(ASPP, Conv2d 3×3 256→256 + BN + ReLU, Conv2d 1×1 256→num_classes).
- torchvision default atrous_rates=(12,24,36) (OS=8); paper's primary OS=16 uses (6,12,18).
- Backbone: ResNet with layer4 dilated (output_stride 8 or 16); IntermediateLayerGetter returns "out"; optional FCNHead aux on layer3.
- Final: classifier output bilinearly upsampled to input size (_SimpleSegmentationModel.forward).

## Ancestors / context (load-bearing)
- FCN (Long 2014): fully-convolutional segmentation, then upsample.
- DeepLab v1/v2 (Chen 2014/2016): atrous conv for segmentation, ASPP, DenseCRF post-proc.
- Atrous origin: algorithme à trous (Holschneider 1989); used in DCNNs by Giusti 2013, Sermanet OverFeat 2013, Papandreou 2014.
- ResNet (He 2015): backbone, block structure.
- BatchNorm (Ioffe 2015).
- Spatial pyramid pooling (Grauman 2005, Lazebnik 2006, He 2014 SPP); PSPNet (Zhao 2016).
- ParseNet (Liu 2015): global pooling for context.
- Multi-grid (Wang 2017 understanding conv; Dai 2017 deformable).
- Four multi-scale paradigms: image pyramid, encoder-decoder (U-Net/SegNet), context module (DenseCRF / cascaded convs), spatial pyramid pooling.
- DenseCRF (Krähenbühl 2011).

## Evaluation settings (no outcomes)
PASCAL VOC 2012 segmentation (21 classes incl background); also Cityscapes; metric mean IoU. Training crop ~513, SGD with "poly" lr ((1−iter/max_iter)^0.9), batch norm fine-tuned with large crop/batch. ResNet-50/101 backbones pretrained on ImageNet/COCO.

## Scaffold ↔ final code correspondence
Pre-method scaffold: a segmentation model = pretrained classification backbone → a context "head" stub → upsample to input size → per-pixel logits. The head is the empty slot. Final code fills: dilated backbone (atrous conv), the ASPP head (parallel 1×1 + atrous 3×3 branches + image-level pooling branch, concat, project, classify). The atrous-conv primitive itself is a dilation arg on Conv2d.
